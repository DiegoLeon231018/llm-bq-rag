import os, json, uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from settings import LOG_SQL_TO_FILE, LOG_FILE_PATH

def _file_append_jsonl(payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def log_sql_event(
    *,
    intent: str,
    question: str,
    sql: str,
    tables: List[str],
    chosen_table: Optional[str],
    rows_returned: Optional[int],
    duration_ms: Optional[int],
    error: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None
):
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": uuid.uuid4().hex,
        "intent": intent,
        "question": question,
        "sql": sql,
        "tables": tables or [],
        "chosen_table": chosen_table,
        "rows_returned": int(rows_returned) if rows_returned is not None else None,
        "duration_ms": int(duration_ms) if duration_ms is not None else None,
        "error": error,
        "meta": meta or {}
    }
    if LOG_SQL_TO_FILE:
        try:
            _file_append_jsonl(payload)
        except Exception as e:
            print(f"[audit_log] file log failed: {e}")
