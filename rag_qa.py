# rag_qa.py (FIX llaves en prompt + respuestas naturales + periodos relativos)
# - Sin placeholders {SPEND_KEYS}/{...} en el SYSTEM_PROMPT_SQL (evita KeyError)
# - Sesiones, multi-tabla, hints de "campaña"
# - Meses en español + periodos relativos (este mes, semana pasada) con TIMEZONE
# - Sanitización ASCII en alias
# - Respuestas humanas: totales, comparaciones, top, listas
# - Logging JSONL de SQL
#
# Requiere: settings.py (con TIMEZONE), bq_utils.py, audit_log.py, state_utils.py, índices FAISS (build_index.py)

import sys, re, os, time, argparse, unicodedata
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from settings import (
    OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL, INDEX_DIR,
    BQ_PROJECT_ID, BQ_DATASET, TIMEZONE
)
from bq_utils import get_bq_client, build_schema_registry, run_query_df
from audit_log import log_sql_event
from state_utils import load_session, append_history, set_last_sql

# ----------------------------
# Catalogación de tablas y vectores de esquema
# ----------------------------
_bq_client = get_bq_client()
_SCHEMA_REGISTRY: Dict[str, str] = build_schema_registry(_bq_client, BQ_DATASET)
_TABLE_FQNS: List[str] = list(_SCHEMA_REGISTRY.keys())

def _make_table_desc(fqn: str, schema_str: str) -> str:
    table_id = fqn.split(".")[-1]
    return f"tabla {table_id} en {BQ_DATASET}: columnas {schema_str}"

_TABLE_DESCS: List[str] = [_make_table_desc(fqn, _SCHEMA_REGISTRY[fqn]) for fqn in _TABLE_FQNS]

def _parse_cols(schema_str: str) -> List[str]:
    if not schema_str:
        return []
    cols = []
    for part in schema_str.split(","):
        name = part.strip().split("(")[0].strip()
        if name:
            cols.append(name)
    return cols

_TABLE_COLS: Dict[str, List[str]] = {fqn: _parse_cols(_SCHEMA_REGISTRY[fqn]) for fqn in _TABLE_FQNS}

# ----------------------------
# Hints de columnas por intención
# ----------------------------
CAMP_COL_KEYWORDS = [
    "campaign_name","campaign","camp","campaña","campana","version","pieza",
    "creative","creativ","anuncio","id_camp","cod_camp","spot","name","detalle","detail"
]
SPEND_KEYS = ["gasto","presupuesto","budget","spend","costo","cost","investment","inversion","tarifa","cost_total","amount_spent"]
CLICK_KEYS = ["click","clicks","click_count"]
IMPR_KEYS  = ["impr","impression","impresion","impresiones"]
CONV_KEYS  = ["conv","conversion","conversions","purchases","leads","installs","registros","result"]
ENG_KEYS   = ["engage","engagement","reactions","likes","comments","shares","view","views"]
DATE_KEYS  = ["date","fecha","day","dt","fch","day_date","event_date","month"]
SOCIAL_HINTS = ["facebook","instagram","meta","tiktok","twitter","x_", "youtube","yt","social","red"]

def _campaign_hints_for(fqn: str) -> List[str]:
    cols = _TABLE_COLS.get(fqn, [])
    hits, seen = [], set()
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in CAMP_COL_KEYWORDS):
            if c not in seen:
                hits.append(c); seen.add(c)
    return hits[:8]

def _has_any(s: str, keys: List[str]) -> bool:
    sl = s.lower()
    return any(k in sl for k in keys)

# ----------------------------
# Selección de tablas (embeddings) con boost para social
# ----------------------------
_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
_TABLE_VECS = np.array(_embeddings.embed_documents(_TABLE_DESCS), dtype="float32")
_TABLE_VECS = _TABLE_VECS / (np.linalg.norm(_TABLE_VECS, axis=1, keepdims=True) + 1e-12)

def select_candidate_tables(question: str, topn: int = 3) -> List[str]:
    qv = np.array(_embeddings.embed_query(question), dtype="float32")
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    sims = _TABLE_VECS @ qv
    boost = np.zeros_like(sims)
    if "redes sociales" in question.lower():
        for i, fqn in enumerate(_TABLE_FQNS):
            tid = fqn.split(".")[-1].lower()
            if _has_any(tid, SOCIAL_HINTS):
                boost[i] += 0.12
    idx = (sims + boost).argsort()[-topn:][::-1]
    return [_TABLE_FQNS[i] for i in idx]

def load_vectordb_for_table(table_id: str) -> FAISS:
    local_dir = os.path.join(INDEX_DIR, table_id)
    return FAISS.load_local(local_dir, _embeddings, allow_dangerous_deserialization=True)

# ----------------------------
# Intención y "último SQL"
# ----------------------------
RE_METRIC = re.compile(
    r"(mayor|top|acumulad|suma|sumar|conteo|contar|count|promedio|media|mediana|tendenc|por mes|por año|por anio|agrupad|group by|max|min|difiere|diferenc|cuant|n[uú]mero|cantidad|total|compar|rendimiento|engage|costo por resultado|cpr|cpc|ctr|cpa)",
    re.IGNORECASE
)
RE_LAST_SQL = re.compile(r"(mu[eé]strame|dime|cu[aá]l|ver|muestra).*sql|sql generado|consulta sql|qué sql", re.IGNORECASE)

def intent_of(q: str) -> str:
    if RE_LAST_SQL.search(q):
        return "last_sql"
    return "metric" if RE_METRIC.search(q) else "rag"

def build_history_text(session_id: str, max_turns: int = 6) -> str:
    s = load_session(session_id)
    hist = s.get("history", [])[-max_turns:]
    lines = []
    for h in hist:
        role = h.get("role", "?")
        content = h.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)

# ----------------------------
# Prompts (sin placeholders fantasma)
# ----------------------------
SPANISH_MONTHS_TEXT = "enero=1, febrero=2, marzo=3, abril=4, mayo=5, junio=6, julio=7, agosto=8, septiembre=9, setiembre=9, octubre=10, noviembre=11, diciembre=12"
RELATIVE_TIME_TEXT = f"""
Usa CURRENT_DATE('{TIMEZONE}') para fechas relativas. Reglas:
- "este mes": BETWEEN DATE_TRUNC(CURRENT_DATE('{TIMEZONE}'), MONTH) AND CURRENT_DATE('{TIMEZONE}')
- "la semana pasada": BETWEEN DATE_SUB(DATE_TRUNC(CURRENT_DATE('{TIMEZONE}'), WEEK(MONDAY)), INTERVAL 7 DAY)
                      AND DATE_SUB(DATE_TRUNC(CURRENT_DATE('{TIMEZONE}'), WEEK(MONDAY)), INTERVAL 1 DAY)
- "este año": BETWEEN DATE_TRUNC(CURRENT_DATE('{TIMEZONE}'), YEAR) AND CURRENT_DATE('{TIMEZONE}')
"""

SPEND_TXT = ", ".join(SPEND_KEYS)
CLICK_TXT = ", ".join(CLICK_KEYS)
IMPR_TXT  = ", ".join(IMPR_KEYS)
CONV_TXT  = ", ".join(CONV_KEYS)
ENG_TXT   = ", ".join(ENG_KEYS)
DATE_TXT  = ", ".join(DATE_KEYS)
CAMP_TXT  = ", ".join(CAMP_COL_KEYWORDS)

SYSTEM_PROMPT_RAG = """Eres un analista de datos conciso y confiable.
Responde en español usando SOLAMENTE la información del contexto recuperado y el historial.
Si no hay evidencia suficiente, di:
"No cuento con suficiente contexto para responder con precisión."
Incluye una sección "Fuentes" con JSON de hasta 3 filas relevantes."""

USER_PROMPT_RAG = """Historial reciente:
{history}

Pregunta del usuario: {question}

Contexto recuperado:
{context}
"""

SYSTEM_PROMPT_SQL = f"""Eres un asistente que escribe SOLO una consulta SQL para BigQuery (ANSI SQL).
Reglas:
- Elige UNA de las tablas candidatas listadas y usa su FQN exacto (con backticks).
- ASCII plano. Literales con comillas simples '...'.
- Usa SOLO ASCII en alias y nombres generados (sin tildes ni ñ). Emplea snake_case (ej. campanias_agosto_2025).
- Identifica columnas de fecha por el esquema (DATE/TIMESTAMP). Meses en español: {SPANISH_MONTHS_TEXT}.
- Periodos relativos: {RELATIVE_TIME_TEXT}
- Mapea métricas:
    * gasto/presupuesto/costo/spend -> columnas que contengan alguno de [{SPEND_TXT}]
    * clicks -> [{CLICK_TXT}]
    * impresiones -> [{IMPR_TXT}]
    * conversiones/resultados -> [{CONV_TXT}]
    * engagement -> [{ENG_TXT}]
    * fecha -> [{DATE_TXT}]
  Si no encuentras nombre exacto, elige la mejor candidata por similitud o contexto del esquema.
- "campañas": usa COUNT(DISTINCT <col_campaña>) sobre una columna candidata entre [{CAMP_TXT}].
- "engagement con menos costo": prioriza engagement alto con costo bajo. Puedes usar ratio engagement/costo como métrica.
- "costo por resultado": usa SUM(costo)/NULLIF(SUM(resultados),0) y devuelve serie temporal si piden "desde que arrancó".
- "rendimiento la semana pasada": incluye métricas clave (impresiones, clicks, CTR = clicks/impresiones, conversiones, costo, CPA = costo/conversiones).
Devuelve SOLO la sentencia SQL sin ``` ni prefijos.
"""

USER_PROMPT_SQL = """Historial reciente:
{history}

Tablas candidatas:
{candidates}

Esquemas:
{schemas}

Pistas por tabla para columna de campañas (candidatas):
{campaign_hints}

Pregunta actual:
{question}

Escribe el SQL más directo y eficiente. Si comparas periodos, devuelve ambas columnas y diferencia. Si piden "desde que arrancó", calcula desde MIN(fecha) de esa campaña hasta CURRENT_DATE.
"""

# ----------------------------
# Sanitización SQL y helpers de respuesta
# ----------------------------
def sanitize_sql(s: str) -> str:
    if not s: return ""
    txt = s.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)\s*```", txt, flags=re.IGNORECASE|re.DOTALL)
    if m: txt = m.group(1).strip()
    txt = re.sub(r"^\s*sql\s*:\s*", "", txt, flags=re.IGNORECASE)
    replacements = {
        "\u00A0":" ", "\u2018":"'", "\u2019":"'", "\u201C":"\"", "\u201D":"\"",
        "\u2013":"-", "\u2014":"-"
    }
    for k,v in replacements.items(): txt = txt.replace(k,v)
    m2 = re.search(r"(select\b.*)$", txt, flags=re.IGNORECASE|re.DOTALL)
    if m2: txt = m2.group(1).strip()
    if txt.endswith(";"): txt = txt[:-1].strip()
    return txt

def ascii_only_identifiers(sql: str) -> str:
    parts = re.split(r"(`[^`]*`)", sql)
    out = []
    for p in parts:
        if len(p) >= 2 and p[0] == "`" and p[-1] == "`":
            out.append(p)
        else:
            norm = unicodedata.normalize("NFKD", p).encode("ascii", "ignore").decode("ascii")
            out.append(norm)
    return "".join(out)

# Etiquetas y métricas comunes
POSSIBLE_LABELS = [
    "campaign_name","campaign","campana","campaña","campaignid","campaign_id",
    "version","pieza","creative","anuncio","name","detalle","detail",
    "marca","cliente","producto","red","canal","sku","categoria"
]

def pick_label_metric(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    for c in cols:
        if c.lower() in POSSIBLE_LABELS:
            for m in cols:
                if df[m].dtype.kind in "iufc" and m != c:
                    return c, m
    label = next((c for c in cols if df[c].dtype.kind not in "iufc"), cols[0])
    metric = next((c for c in cols if df[c].dtype.kind in "iufc" and c != label),
                  cols[-1] if cols[-1] != label else cols[0])
    return label, metric

def fmt_number(x) -> str:
    if pd.isna(x): return "0"
    try:
        return f"{float(x):,.0f}" if float(x).is_integer() else f"{float(x):,.2f}"
    except Exception:
        return str(x)

# Nombres humanos por hints
METRIC_HINTS = ["click", "impres", "revenue", "tarifa", "trp", "spots", "views", "instal", "conv", "costo", "cost", "spend", "engage"]
NOUN_BY_HINT = {
    "gasto":"de gasto","presupuesto":"de gasto","spend":"de gasto","costo":"de costo","cost":"de costo",
    "click":"clicks","impres":"impresiones","revenue":"revenue","tarifa":"tarifa","conv":"conversiones",
    "engage":"engagement","views":"vistas","instal":"instalaciones"
}
def infer_noun(question: str, sql: str) -> str:
    base = (question + " " + sql).lower()
    for k, v in NOUN_BY_HINT.items():
        if k in base:
            return v
    return "registros"

def format_top1_sentence(df: pd.DataFrame, question: str, sql: str) -> Optional[str]:
    if df.shape[0] != 1: return None
    label_col, metric_col = pick_label_metric(df)
    metric_h = metric_col.replace("_"," ")
    period = _period_human(question, sql)
    prefix = f"En {period} " if period else ""
    return f"{prefix}la {label_col} con mayor {metric_h} fue {df.iloc[0][label_col]} con {fmt_number(df.iloc[0][metric_col])}."

def format_list(df: pd.DataFrame, question: str, sql: str, top_n: int = 5) -> str:
    label_col, metric_col = pick_label_metric(df)
    metric_h = metric_col.replace("_"," ")
    period = _period_human(question, sql)
    head = f"Top {min(top_n,len(df))} por {metric_h}" + (f" en {period}" if period else "") + ":\n"
    items = [f"- {r[label_col]}: {fmt_number(r[metric_col])}" for _, r in df.head(top_n).iterrows()]
    return head + "\n".join(items)

# -------- Periodos y frases humanas --------
MONTHS_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio",
    7: "julio", 8: "agosto", 9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"
}
RE_MONTH_ES = r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"

def _period_from_sql(sql: str) -> tuple[Optional[int], Optional[int]]:
    y = None; m = None
    my = re.search(r"EXTRACT\s*\(\s*YEAR\s+FROM\s+[^)]+\)\s*=\s*(\d{4})", sql, flags=re.IGNORECASE)
    if my: y = int(my.group(1))
    mm = re.search(r"EXTRACT\s*\(\s*MONTH\s+FROM\s+[^)]+\)\s*=\s*(\d{1,2})", sql, flags=re.IGNORECASE)
    if mm: m = int(mm.group(1))
    return y, m

def _years_from_sql(sql: str) -> List[int]:
    years = set()
    for m in re.findall(r"EXTRACT\s*\(\s*YEAR\s+FROM\s+[^)]+\)\s*=\s*(\d{4})", sql, flags=re.IGNORECASE):
        years.add(int(m))
    m_in = re.search(r"EXTRACT\s*\(\s*YEAR\s+FROM\s+[^)]+\)\s*IN\s*\(([^)]*)\)", sql, flags=re.IGNORECASE)
    if m_in:
        for n in re.findall(r"\d{4}", m_in.group(1)):
            years.add(int(n))
    return sorted(years)

def _month_from_sql(sql: str) -> Optional[int]:
    mm = re.search(r"EXTRACT\s*\(\s*MONTH\s+FROM\s+[^)]+\)\s*=\s*(\d{1,2})", sql, flags=re.IGNORECASE)
    return int(mm.group(1)) if mm else None

def _period_from_question(q: str) -> tuple[Optional[int], Optional[int]]:
    ql = q.lower()
    y = None
    my = re.search(r"(20\d{2})", ql)
    if my: y = int(my.group(1))
    m = None
    mm = re.search(RE_MONTH_ES, ql, re.IGNORECASE)
    if mm:
        name = mm.group(1)
        name = "septiembre" if name == "setiembre" else name
        for k,v in MONTHS_ES.items():
            if v == name:
                m = k; break
    return y, m

def _period_human(q: str, sql: str) -> Optional[str]:
    y, m = _period_from_sql(sql)
    if not y or not m:
        y2, m2 = _period_from_question(q)
        y = y or y2; m = m or m2
    if y and m:
        return f"{MONTHS_ES.get(m, str(m))} de {y}"
    if y and not m:
        return f"{y}"
    return None

def _is_single_numeric_cell(df: pd.DataFrame) -> bool:
    if df.shape == (1, 1):
        val = df.iloc[0, 0]
        try:
            float(val)
            return True
        except Exception:
            return False
    return False

def try_format_total_sentence(df: pd.DataFrame, question: str, sql: str) -> Optional[str]:
    if not _is_single_numeric_cell(df):
        return None
    n = df.iloc[0, 0]
    try:
        n_fmt = f"{float(n):,.0f}" if float(n).is_integer() else f"{float(n):,.2f}"
    except Exception:
        n_fmt = str(n)
    noun = infer_noun(question, sql)
    per = _period_human(question, sql)
    if per:
        return f"En {per} se obtuvo un total {noun} de {n_fmt}."
    return f"El total {noun} es {n_fmt}."

def _nice_metric_name(names: List[str]) -> str:
    base = " ".join(names).lower()
    for h in METRIC_HINTS:
        if h in base:
            if h.startswith("impres"): return "impresiones"
            if h.startswith("click"):  return "clicks"
            if h == "trp":            return "TRPs"
            if h == "spots":          return "spots"
            if h.startswith("revenue"): return "revenue"
            if h.startswith("tarifa"):  return "tarifa"
            if h.startswith("cost") or h == "costo" or h == "spend": return "costo"
            if h.startswith("views"):   return "vistas"
            if h.startswith("instal"):  return "instalaciones"
            if h.startswith("conv"):    return "conversiones"
            if h.startswith("engage"):  return "engagement"
    cleaned = re.sub(r"(_|\d{4}|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)", " ", base)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else "valores"

def try_format_comparison_sentence(df: pd.DataFrame, question: str, sql: str) -> Optional[str]:
    if len(df) != 1:
        return None
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return None
    years = _years_from_sql(sql)
    month = _month_from_sql(sql)
    month_name = MONTHS_ES.get(month) if month else None

    def find_col_for_year(y: int) -> Optional[str]:
        for c in numeric_cols:
            if str(y) in c:
                return c
        return None

    left = right = None
    if len(years) >= 2:
        y2 = years[-1]; y1 = years[0]
        c2 = find_col_for_year(y2)
        c1 = find_col_for_year(y1)
        if c1 and c2:
            left = (y2, c2); right = (y1, c1)
    if not (left and right) and len(numeric_cols) >= 2:
        c2, c1 = numeric_cols[0], numeric_cols[1]
        y_in_c2 = re.search(r"(20\d{2})", c2); y_in_c1 = re.search(r"(20\d{2})", c1)
        y2 = int(y_in_c2.group(1)) if y_in_c2 else (years[-1] if years else None)
        y1 = int(y_in_c1.group(1)) if y_in_c1 else (years[0] if years else None)
        left = (y2, c2); right = (y1, c1)

    if not (left and right):
        return None

    y2, c2 = left; y1, c1 = right
    v2 = float(df.iloc[0][c2]); v1 = float(df.iloc[0][c1])
    diff_col = next((c for c in numeric_cols if "dif" in c.lower()), None)
    diff = float(df.iloc[0][diff_col]) if diff_col is not None else (v2 - v1)
    pct = (diff / v1) * 100.0 if v1 != 0 else None
    metric_name = _nice_metric_name([c1, c2])

    if month_name and y1 and y2:
        per2 = f"{month_name} de {y2}"
        per1 = f"{month_name} de {y1}"
    else:
        per2 = f"{y2}" if y2 else "periodo 2"
        per1 = f"{y1}" if y1 else "periodo 1"

    v2s = f"{v2:,.0f}" if v2.is_integer() else f"{v2:,.2f}"
    v1s = f"{v1:,.0f}" if v1.is_integer() else f"{v1:,.2f}"
    ds  = f"{diff:,.0f}" if diff.is_integer() else f"{diff:,.2f}"
    if pct is not None:
        pcts = f"{pct:+.2f}%"
        return f"En {per2} se obtuvieron {v2s} {metric_name}; en {per1}, {v1s}. La diferencia es {ds} ({pcts})."
    return f"En {per2} se obtuvieron {v2s} {metric_name}; en {per1}, {v1s}. La diferencia es {ds}."

# ----------------------------
# RAG multi-tabla
# ----------------------------
def make_context(docs) -> str:
    blocks = []
    for d in docs[:3]:
        preview = d.page_content[:300].replace("\n"," ")
        blocks.append(f"- [tabla: {d.metadata.get('table','?')}] {preview} ...")
    return "\n".join(blocks)

def rag_answer(question: str, session_id: str, k_per_table: int = 3, tables_top: int = 3) -> str:
    history = build_history_text(session_id)
    cands = select_candidate_tables(question, topn=tables_top)
    all_docs = []
    for fqn in cands:
        table_id = fqn.split(".")[-1]
        try:
            vectordb = load_vectordb_for_table(table_id)
            docs = vectordb.as_retriever(search_kwargs={"k": k_per_table}).get_relevant_documents(question)
            all_docs.extend(docs)
        except Exception:
            pass
    if not all_docs:
        return "No cuento con suficiente contexto para responder con precisión."
    context = make_context(all_docs)
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT_RAG), ("user", USER_PROMPT_RAG)])
    return (prompt | llm | StrOutputParser()).invoke({"history": history, "question": question, "context": context})

# ----------------------------
# SQL path con logging + sesión
# ----------------------------
def _extract_chosen_table_from_sql(sql: str) -> Optional[str]:
    m = re.search(r"`([a-z0-9\-\_]+\.[a-z0-9\-\_]+\.[a-z0-9\-\_]+)`", sql, flags=re.IGNORECASE)
    return m.group(1) if m else None

def sql_answer(question: str, session_id: str) -> str:
    intent = "metric"
    history = build_history_text(session_id)
    cands = select_candidate_tables(question, topn=3)

    hints_lines = []
    for fqn in cands:
        hints = _campaign_hints_for(fqn)
        hints_txt = ", ".join(hints) if hints else "(sin pistas claras)"
        hints_lines.append(f"- {fqn} -> {hints_txt}")
    campaign_hints = "\n".join(hints_lines)

    schemas_str = "\n".join([f"- {fqn}: { _SCHEMA_REGISTRY[fqn] }" for fqn in cands])
    cands_str = "\n".join([f"* {fqn}" for fqn in cands])

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT_SQL), ("user", USER_PROMPT_SQL)])
    raw_sql = (prompt | llm | StrOutputParser()).invoke({
        "history": history,
        "candidates": cands_str,
        "question": question,
        "schemas": schemas_str,
        "campaign_hints": campaign_hints
    }).strip()

    sql = sanitize_sql(raw_sql)
    sql = ascii_only_identifiers(sql)
    if not sql.lower().lstrip().startswith("select"):
        log_sql_event(intent=intent, question=question, sql=raw_sql,
                      tables=cands, chosen_table=None, rows_returned=None, duration_ms=None,
                      error="not_a_select", meta={"sanitized": sql})
        set_last_sql(session_id, sql="", meta={"error": "not_a_select", "candidates": cands})
        raise ValueError(f"El modelo no generó un SELECT válido.\n--- Crudo ---\n{raw_sql}\n--- Sanitizado ---\n{sql}")

    chosen = _extract_chosen_table_from_sql(sql)
    t0 = time.time()
    error = None
    try:
        df = run_query_df(_bq_client, sql)
        rows = int(len(df))
    except Exception as e:
        error = str(e)
        df = pd.DataFrame(); rows = 0
    dur = int((time.time() - t0) * 1000)

    log_sql_event(intent=intent, question=question, sql=sql,
                  tables=cands, chosen_table=chosen, rows_returned=rows, duration_ms=dur,
                  error=error, meta={})

    set_last_sql(session_id, sql=sql, meta={
        "chosen_table": chosen, "tables": cands, "rows_returned": rows,
        "duration_ms": dur, "error": error
    })

    if error:
        raise RuntimeError(f"Error ejecutando SQL: {error}\n\nSQL usado:\n{sql}")
    if df.empty:
        return f"No se encontraron resultados.\n\nSQL usado:\n{sql}"

    # 0) Comparaciones (ej. este mes vs año pasado)
    sent_cmp = try_format_comparison_sentence(df, question, sql)
    if sent_cmp:
        return f"{sent_cmp}\n\nSQL usado:\n{sql}"

    # 1) Totales únicos
    sent_total = try_format_total_sentence(df, question, sql)
    if sent_total:
        return f"{sent_total}\n\nSQL usado:\n{sql}"

    # 2) Top 1
    sent_top1 = format_top1_sentence(df, question, sql)
    if sent_top1:
        return f"{sent_top1}\n\nSQL usado:\n{sql}"

    # 3) Lista Top N
    return f"{format_list(df, question, sql, top_n=5)}\n\nSQL usado:\n{sql}"

# ----------------------------
# Router con sesiones
# ----------------------------
def answer(question: str, session_id: str = "default") -> str:
    append_history(session_id, role="user", content=question)

    mode = intent_of(question)
    if mode == "last_sql":
        s = load_session(session_id)
        last_sql = s.get("last_sql")
        meta = s.get("last_meta") or {}
        if not last_sql:
            resp = "Aún no tengo un SQL en esta sesión. Pídeme algo que requiera métrica/aggregación y lo genero."
        else:
            parts = ["Último SQL ejecutado:", "```sql", last_sql, "```"]
            if meta:
                parts.append(f"Tabla elegida: {meta.get('chosen_table') or 'n/d'} | Filas: {meta.get('rows_returned')} | Latencia: {meta.get('duration_ms')} ms")
                if meta.get("error"):
                    parts.append(f"Error: {meta['error']}")
            resp = "\n".join(parts)
    elif mode == "metric":
        resp = sql_answer(question, session_id)
    else:
        resp = rag_answer(question, session_id)

    append_history(session_id, role="assistant", content=resp)
    return resp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG-BQ con respuestas naturales y periodos relativos (sin placeholders fantasmas)")
    parser.add_argument("--session", type=str, default="default", help="ID de sesión persistente")
    parser.add_argument("question", nargs="+", help="Pregunta en español")
    args = parser.parse_args()
    q = " ".join(args.question)
    print(answer(q, session_id=args.session))
