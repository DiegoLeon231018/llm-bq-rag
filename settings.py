import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID")
BQ_TABLE = os.getenv("BQ_TABLE")

INDEX_DIR = os.getenv("INDEX_DIR", "./storage/index_faiss")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

INCLUDE_COLUMNS = [c.strip() for c in os.getenv("INCLUDE_COLUMNS", "").split(",") if c.strip()]
WHERE_SQL = os.getenv("WHERE_SQL", "").strip() or None
LIMIT_ROWS = int(os.getenv("LIMIT_ROWS", "0") or 0)

# Resolver ruta al JSON de credenciales si es relativa
SERVICE_ACCOUNT_JSON = os.getenv("SERVICE_ACCOUNT_JSON")
if SERVICE_ACCOUNT_JSON and not os.path.isabs(SERVICE_ACCOUNT_JSON):
    BASE_DIR = Path(__file__).resolve().parent
    SERVICE_ACCOUNT_JSON = str((BASE_DIR / SERVICE_ACCOUNT_JSON).resolve())
