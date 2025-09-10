# build_index.py
# Indexador FAISS multi-tabla para BigQuery (local)
# - Respeta filtros por regex (TABLE_INCLUDE_REGEX / TABLE_EXCLUDE_REGEX)
# - Reindex forzado (FORCE_REINDEX) y limpieza previa (CLEAN_INDEX_DIR)
# - Trae TODA la tabla con ROWS_PER_TABLE=0 (sin LIMIT)
# - Trunca cada fila por tokens (MAX_EMBED_TOKENS) para evitar petar el límite
# - Batching “seguro” según TOKEN_BUDGET_PER_REQUEST
# - Reintentos con backoff ante rate limit
#
# Ajustar en .env:
#   TABLE_INCLUDE_REGEX=.*           (o ^g_entel, etc.)
#   TABLE_EXCLUDE_REGEX=^$
#   MAX_TABLES=0                     (0 = sin límite de tablas)
#   ROWS_PER_TABLE=0                 (0 = TODAS las filas)
#   INCLUDE_COLUMNS=                 (si quieres reducir ancho)
#   WHERE_SQL=                       (si quieres filtrar filas globalmente)
#   MAX_EMBED_TOKENS=1200
#   EMBED_BATCH_SIZE=32
#   TOKEN_BUDGET_PER_REQUEST=250000
#   FORCE_REINDEX=true|false
#   CLEAN_INDEX_DIR=true|false
#   SKIP_IF_INDEX_EXISTS=true|false
#
# Requiere: settings.py, bq_utils.py

import os, re, time, shutil
from typing import List
from tqdm import tqdm
import numpy as np
import tiktoken

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from settings import (
    OPENAI_API_KEY, EMBEDDING_MODEL, INDEX_DIR,
    WHERE_SQL, BQ_DATASET, BQ_PROJECT_ID, INCLUDE_COLUMNS
)
from bq_utils import get_bq_client, list_dataset_tables, table_fqn, fetch_table_df

# ------------------ Config vía ENV ------------------
TABLE_INCLUDE_REGEX = os.getenv("TABLE_INCLUDE_REGEX", ".*")
TABLE_EXCLUDE_REGEX = os.getenv("TABLE_EXCLUDE_REGEX", "^$")
MAX_TABLES = int(os.getenv("MAX_TABLES", "0"))                 # 0 = sin límite
ROWS_PER_TABLE = int(os.getenv("ROWS_PER_TABLE", "0"))         # 0 = todas las filas
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "1200"))
TOKEN_BUDGET_PER_REQUEST = int(os.getenv("TOKEN_BUDGET_PER_REQUEST", "250000"))

# Reindex options
SKIP_IF_INDEX_EXISTS = os.getenv("SKIP_IF_INDEX_EXISTS", "false").lower() == "true"
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() == "true"
CLEAN_INDEX_DIR = os.getenv("CLEAN_INDEX_DIR", "false").lower() == "true"

# ------------------ Helpers ------------------
_enc = tiktoken.get_encoding("cl100k_base")

def truncate_by_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    toks = _enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return _enc.decode(toks[:max_tokens])

def row_to_text(row: dict, order: list) -> str:
    # Compacto "col: val | col: val" y truncamos por tokens
    base = " | ".join([f"{col}: {row.get(col, '')}" for col in order])
    return truncate_by_tokens(base, MAX_EMBED_TOKENS)

def backoff_sleep(attempt: int):
    # Exponencial con jitter
    time.sleep(min(30, 2 ** attempt) + np.random.rand())

def safe_batch_size() -> int:
    # Aproxima tamaño de lote máximo para no rebasar el budget de tokens por request:
    approx = max(1, TOKEN_BUDGET_PER_REQUEST // max(1, MAX_EMBED_TOKENS))
    return max(1, min(EMBED_BATCH_SIZE, approx))

def save_faiss_incremental(texts, metadatas, embeddings, out_dir: str, faiss_batch: int):
    """
    Construye FAISS incrementalmente. Cada add_texts llama a embeddings.embed_documents
    con el tamaño de lote que pasamos, evitando requests gigantes.
    """
    os.makedirs(out_dir, exist_ok=True)
    vectordb = None
    for i in tqdm(range(0, len(texts), faiss_batch), desc="FAISS add_texts", leave=False):
        tx = texts[i:i + faiss_batch]
        md = metadatas[i:i + faiss_batch]
        # Reintentos en la capa de LangChain/OpenAI si hay rate limit:
        attempt = 0
        while True:
            try:
                if vectordb is None:
                    vectordb = FAISS.from_texts(tx, embeddings, metadatas=md)
                else:
                    vectordb.add_texts(tx, metadatas=md)
                break
            except Exception as e:
                attempt += 1
                if attempt > 5:
                    raise RuntimeError(f"Fallo agregando a FAISS tras reintentos: {e}")
                backoff_sleep(attempt)
    assert vectordb is not None, "No se construyó el índice"
    vectordb.save_local(out_dir)

# ------------------ Pipeline por tabla ------------------
def index_one_table(client, table_id: str):
    fqn = table_fqn(BQ_PROJECT_ID, BQ_DATASET, table_id)
    local_dir = os.path.join(INDEX_DIR, table_id)

    if FORCE_REINDEX and os.path.isdir(local_dir) and CLEAN_INDEX_DIR:
        print(f"[{table_id}] CLEAN_INDEX_DIR=true -> borrando {local_dir} ...")
        shutil.rmtree(local_dir, ignore_errors=True)

    if not FORCE_REINDEX and SKIP_IF_INDEX_EXISTS and os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"[{table_id}] Índice ya existe. Saltando (SKIP_IF_INDEX_EXISTS=true, FORCE_REINDEX=false).")
        return

    limit_rows = ROWS_PER_TABLE if ROWS_PER_TABLE > 0 else None
    print(f"[{table_id}] Cargando {'todas' if limit_rows is None else f'hasta {limit_rows}'} filas de {fqn}...")
    df = fetch_table_df(
        client, fqn,
        include_columns=INCLUDE_COLUMNS if INCLUDE_COLUMNS else None,
        where_sql=WHERE_SQL,
        limit_rows=limit_rows
    )
    if df.empty:
        print(f"[{table_id}] Tabla vacía o filtro agresivo. Saltando.")
        return

    cols = list(df.columns)
    texts, metas = [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Prep {table_id}", leave=False):
        rec = row.to_dict()
        texts.append(row_to_text(rec, cols))
        metas.append({"table": table_id, "row_index": int(i)})

    faiss_batch = safe_batch_size()
    print(f"[{table_id}] MAX_EMBED_TOKENS={MAX_EMBED_TOKENS} | batch seguro={faiss_batch} (budget {TOKEN_BUDGET_PER_REQUEST})")

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    save_faiss_incremental(texts, metas, embeddings, local_dir, faiss_batch)
    print(f"[{table_id}] OK → {local_dir}")

# ------------------ Descubrimiento con filtros ------------------
def filter_tables(table_ids: List[str]) -> List[str]:
    inc = re.compile(TABLE_INCLUDE_REGEX)
    exc = re.compile(TABLE_EXCLUDE_REGEX) if TABLE_EXCLUDE_REGEX else None
    selected = [t for t in table_ids if inc.search(t) and (not exc or not exc.search(t))]
    if MAX_TABLES and MAX_TABLES > 0:
        selected = selected[:MAX_TABLES]
    return selected

def main():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY no está configurado.")
    client = get_bq_client()
    all_tables = list_dataset_tables(client, BQ_DATASET)
    tables = filter_tables(all_tables)
    print(f"Tablas totales en {BQ_PROJECT_ID}.{BQ_DATASET}: {len(all_tables)}")
    print(f"Tablas seleccionadas (máx {MAX_TABLES or '∞'}): {tables}")
    print(f"FORCE_REINDEX={FORCE_REINDEX} | CLEAN_INDEX_DIR={CLEAN_INDEX_DIR} | SKIP_IF_INDEX_EXISTS={SKIP_IF_INDEX_EXISTS}")
    print(f"ROWS_PER_TABLE={'ALL' if ROWS_PER_TABLE==0 else ROWS_PER_TABLE} | INCLUDE_COLUMNS={INCLUDE_COLUMNS or 'ALL'}")
    for t in tables:
        try:
            index_one_table(client, t)
        except Exception as e:
            print(f"[{t}] Error indexando: {e}")
    print("Finalizado.")

if __name__ == "__main__":
    main()
