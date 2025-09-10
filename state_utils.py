import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from settings import INDEX_DIR

STORAGE_ROOT = Path(INDEX_DIR).resolve().parent
SESSIONS_DIR = STORAGE_ROOT / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"

def load_session(session_id: str) -> Dict[str, Any]:
    p = _session_path(session_id)
    if not p.exists():
        return {"session_id": session_id, "history": [], "last_sql": None, "last_meta": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"session_id": session_id, "history": [], "last_sql": None, "last_meta": None}

def save_session(session: Dict[str, Any]) -> None:
    p = _session_path(session["session_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")

def append_history(session_id: str, role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
    s = load_session(session_id)
    s["history"].append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "content": content,
        "meta": meta or {}
    })
    save_session(s)

def set_last_sql(session_id: str, sql: str, meta: Dict[str, Any]) -> None:
    s = load_session(session_id)
    s["last_sql"] = sql
    s["last_meta"] = meta
    save_session(s)
