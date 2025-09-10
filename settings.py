import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# BigQuery
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET")

# Índices locales (cada tabla en: {INDEX_DIR}\{table_id})
INDEX_DIR = os.getenv("INDEX_DIR", "./storage/index_faiss")
TIMEZONE = os.getenv("TIMEZONE", "America/Lima")
# Modelos
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Filtros globales de indexado
INCLUDE_COLUMNS = [c.strip() for c in os.getenv("INCLUDE_COLUMNS", "").split(",") if c.strip()]
WHERE_SQL = os.getenv("WHERE_SQL", "").strip() or None
LIMIT_ROWS = int(os.getenv("LIMIT_ROWS", "0") or 0)

# Credenciales locales
SERVICE_ACCOUNT_JSON = os.getenv("SERVICE_ACCOUNT_JSON")
if SERVICE_ACCOUNT_JSON and not os.path.isabs(SERVICE_ACCOUNT_JSON):
    BASE_DIR = Path(__file__).resolve().parent
    SERVICE_ACCOUNT_JSON = str((BASE_DIR / SERVICE_ACCOUNT_JSON).resolve())

# Logging local de SQL
LOG_SQL_TO_FILE = os.getenv("LOG_SQL_TO_FILE", "true").lower() == "true"
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", os.path.join("storage", "logs", "rag_sql.jsonl"))

# Asegurar carpetas base
Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(LOG_FILE_PATH)).mkdir(parents=True, exist_ok=True)
