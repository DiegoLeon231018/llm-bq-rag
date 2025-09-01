# rag_qa.py
import sys
import re
from typing import Optional, Tuple
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from settings import (
    OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL, INDEX_DIR,
    BQ_TABLE, BQ_PROJECT_ID
)
from bq_utils import get_bq_client, get_table_schema, run_query_df

# ----------------------------
# RAG prompts
# ----------------------------
SYSTEM_PROMPT_RAG = """Eres un analista de datos conciso y confiable.
Responde en español usando SOLAMENTE la información del contexto recuperado.
Si la pregunta no se puede responder con el contexto, responde:
"No cuento con suficiente contexto de la tabla para responder con precisión."
Incluye una sección "Fuentes" listando JSON de las filas relevantes.
"""

USER_PROMPT_RAG = """Pregunta del usuario: {question}

Contexto recuperado (filas relevantes de la tabla):
{context}
"""

# ----------------------------
# SQL prompts (month es DATE)
# ----------------------------
SYSTEM_PROMPT_SQL = """Eres un asistente que escribe SOLO una consulta SQL para BigQuery (ANSI SQL).
Contexto del esquema temporal:
- La columna de fecha principal se llama `month` y es de tipo DATE (por ejemplo '2025-08-01').
- Para filtrar por año y mes debes usar: EXTRACT(YEAR FROM month) y EXTRACT(MONTH FROM month).

Requisitos:
- Usa el nombre completo de la tabla: `{project}.{table}`.
- Usa EXCLUSIVAMENTE ASCII básico: sin comillas curvas, sin guiones largos, sin NBSP.
- Usa SOLO comillas simples '...' para literales de texto.
- Si el usuario menciona año y/o mes (por nombre o número), aplícalo con EXTRACT sobre `month`.
- Si pide "mayor", "top", "acumulado", realiza la agregación (SUM/COUNT...), agrupa, ordena y limita.
- Devuelve SOLO la sentencia SQL SIN usar ``` ni etiquetas ni prefijos tipo 'SQL:'.
- Evita DML/DDL. Si es un "top 1" claro, usa LIMIT 1; si no, LIMIT 100.
"""

USER_PROMPT_SQL = """Tabla: `{project}.{table}`
Esquema:
{schema}

Pregunta:
{question}

Escribe el SQL más directo y eficiente para responderla. Si es "mayor acumulado", devuelve el TOP 1 con la métrica y la clave (por ejemplo 'marca'). Usa ORDER BY DESC y LIMIT 1.
"""

# ----------------------------
# Normalizador de meses ES
# ----------------------------
SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12
}
YEAR_RE = re.compile(r"\b(20\d{2})\b")
MONTH_NAME_RE = re.compile(r"\b(" + "|".join(SPANISH_MONTHS.keys()) + r")\b", re.IGNORECASE)

def normalize_temporal_hints(question: str) -> dict:
    """
    Devuelve hints: {"year": 2025, "month": 8} si se mencionan.
    Reconoce meses en español y patrones 'mes 8'.
    """
    q = question.lower()
    y = None
    ym = YEAR_RE.search(q)
    if ym:
        y = int(ym.group(1))

    m = None
    mm = MONTH_NAME_RE.search(q)
    if mm:
        m = SPANISH_MONTHS[mm.group(1).lower()]

    if m is None:
        mnum = re.search(r"\bmes\s*[:=]?\s*(\d{1,2})\b", q)
        if mnum:
            m = int(mnum.group(1))

    return {"year": y, "month": m}

# ----------------------------
# Heurística de intención métrica
# ----------------------------
_METRIC_PATTERNS = re.compile(
    r"(mayor|top|acumulad|suma|sumar|conteo|promedio|media|mediana|tendenc|por mes|por año|agrupad|group by|max|min|difiere|diferencia)",
    re.IGNORECASE
)

def looks_like_metric(question: str) -> bool:
    return bool(_METRIC_PATTERNS.search(question))

# ----------------------------
# Sanitizador de SQL (quita fences, etiquetas y deja solo el SELECT)
# ----------------------------
def sanitize_sql(s: str) -> str:
    """Quita fences ```sql, etiquetas 'SQL:', Unicode raro y punto y coma final.
    Devuelve la primera sentencia que comience con SELECT."""
    if not s:
        return ""
    txt = s.strip()

    # 1) Extraer contenido entre fences si existen
    m = re.search(r"```(?:sql)?\s*(.*?)\s*```", txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        txt = m.group(1).strip()

    # 2) Quitar prefijos tipo "SQL:" o similares
    txt = re.sub(r"^\s*sql\s*:\s*", "", txt, flags=re.IGNORECASE)

    # 3) Normalizar Unicode problemático a ASCII simple
    replacements = {
        "\u00A0": " ",  # NBSP
        "\u2018": "'", "\u2019": "'",  # ‘ ’
        "\u201C": "\"", "\u201D": "\"",  # “ ”
        "\u2013": "-", "\u2014": "-",   # – —
    }
    for k, v in replacements.items():
        txt = txt.replace(k, v)

    # 4) Quedarnos desde el primer SELECT
    m2 = re.search(r"(select\b.*)$", txt, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        txt = m2.group(1).strip()

    # 5) Quitar punto y coma final
    if txt.endswith(";"):
        txt = txt[:-1].strip()

    return txt

# ----------------------------
# Helpers de formato de respuesta
# ----------------------------
POSSIBLE_LABELS = ["marca", "cliente", "producto", "red", "canal", "sku", "version"]
def pick_label_metric(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    # busca label por nombres típicos
    for c in cols:
        if c.lower() in POSSIBLE_LABELS:
            # métrica: primera numérica
            for m in cols:
                if df[m].dtype.kind in "iufc" and m != c:
                    return c, m
    # si no, elige: primera no numérica como label, primera numérica como métrica
    label = next((c for c in cols if df[c].dtype.kind not in "iufc"), cols[0])
    metric = next((c for c in cols if df[c].dtype.kind in "iufc" and c != label),
                  cols[-1] if cols[-1] != label else cols[0])
    return label, metric

def fmt_number(x) -> str:
    if pd.isna(x):
        return "0"
    try:
        if float(x).is_integer():
            return f"{int(x):,}"
        return f"{float(x):,}"
    except Exception:
        return str(x)

def format_top1_sentence(df: pd.DataFrame) -> Optional[str]:
    if df.shape[0] != 1:
        return None
    label_col, metric_col = pick_label_metric(df)
    label_val = str(df.iloc[0][label_col])
    metric_val = fmt_number(df.iloc[0][metric_col])
    return f"La {label_col} con mayor {metric_col.replace('_', ' ')} es {label_val} con {metric_val}."

def format_list(df: pd.DataFrame, top_n: int = 5) -> str:
    label_col, metric_col = pick_label_metric(df)
    rows = []
    for _, r in df.head(top_n).iterrows():
        rows.append(f"- {r[label_col]}: {fmt_number(r[metric_col])}")
    return f"Top {min(top_n, len(df))} por {metric_col.replace('_',' ')}:\n" + "\n".join(rows)

def format_diff_sentence(df: pd.DataFrame) -> Optional[str]:
    """Si el SQL devuelve columnas total_2025, total_2024, diferencia, arma frase."""
    cols = {c.lower(): c for c in df.columns}
    need = {"total_2025", "total_2024", "diferencia"}
    if not need.issubset(set(k.lower() for k in df.columns)):
        return None
    r = df.iloc[0]
    t25 = fmt_number(r[cols["total_2025"]])
    t24 = fmt_number(r[cols["total_2024"]])
    dif = fmt_number(r[cols["diferencia"]])
    return f"Agosto 2025 acumula {t25}; agosto 2024 acumula {t24}; la diferencia es {dif}."

# ----------------------------
# RAG helpers
# ----------------------------
def load_vectordb():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def make_context(docs) -> str:
    blocks = []
    for d in docs:
        preview = d.page_content[:300].replace("\n", " ")
        blocks.append(f"- {preview} ...\n  JSON: {d.metadata.get('row_json','')}")
    return "\n".join(blocks)

# ----------------------------
# SQL path
# ----------------------------
def sql_answer(question: str) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY no está configurado.")
    client = get_bq_client()
    schema = get_table_schema(client, BQ_TABLE)

    # Normaliza año/mes y añade hint explícito
    hints = normalize_temporal_hints(question)
    hint_text = ""
    if hints.get("year") or hints.get("month"):
        parts = []
        if hints.get("year"):
            parts.append(f"año={hints['year']}")
        if hints.get("month"):
            parts.append(f"mes={hints['month']}")
        hint_text = f" [HINT: filtrar con EXTRACT sobre month -> {'; '.join(parts)}]"

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_SQL),
        ("user", USER_PROMPT_SQL),
    ])
    raw_sql = (prompt | llm | StrOutputParser()).invoke({
        "project": BQ_PROJECT_ID,
        "table": BQ_TABLE,
        "schema": schema,
        "question": question + hint_text
    }).strip()

    sql = sanitize_sql(raw_sql)
    if not sql.lower().lstrip().startswith("select"):
        raise ValueError(
            f"El modelo no generó un SELECT válido tras sanitizar.\n--- Crudo ---\n{raw_sql}\n--- Sanitizado ---\n{sql}"
        )

    df = run_query_df(client, sql)

    if df.empty:
        return f"No se encontraron resultados.\n\nSQL usado:\n{sql}"

    # Formato inteligente de la respuesta
    diff = format_diff_sentence(df)
    if diff:
        return f"Respuesta: {diff}\n\nSQL usado:\n{sql}"

    sent = format_top1_sentence(df)
    if sent:
        return f"Respuesta: {sent}\n\nSQL usado:\n{sql}"

    listado = format_list(df, top_n=5)
    return f"{listado}\n\nSQL usado:\n{sql}"

# ----------------------------
# RAG path
# ----------------------------
def rag_answer(question: str, k: int = 5) -> str:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY no está configurado.")
    vectordb = load_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "No cuento con suficiente contexto de la tabla para responder con precisión."
    context = make_context(docs)
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_RAG),
        ("user", USER_PROMPT_RAG)
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})

# ----------------------------
# Router
# ----------------------------
def answer(question: str) -> str:
    return sql_answer(question) if looks_like_metric(question) else rag_answer(question)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python rag_qa.py \"tu pregunta\"")
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    print(answer(q))
