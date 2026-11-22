"""Crash-safe per-user outputs and structured API error logging."""

import json
import os
import pickle
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path


_log_lock = threading.Lock()


def _atomic_write(path, writer, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, mode) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_pickle(path, value):
    _atomic_write(path, lambda handle: pickle.dump(value, handle), "wb")


def atomic_json(path, value):
    def write(handle):
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
    _atomic_write(path, write, "w")


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def completed_user_outputs(run_dir, avatar_id):
    """Return validated outputs, accepting legacy runs without completion markers."""
    run_dir = Path(run_dir)
    behavior_path = run_dir / "behavior" / f"{avatar_id}.pkl"
    interview_path = run_dir / "interview" / f"{avatar_id}.pkl"
    if not behavior_path.is_file() or not interview_path.is_file():
        return None
    try:
        behavior = load_pickle(behavior_path)
        interview = load_pickle(interview_path)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError):
        return None
    if not isinstance(behavior, dict) or not isinstance(interview, dict):
        return None
    return behavior, interview


def token_cost_from_marker_or_log(run_dir, avatar_id):
    run_dir = Path(run_dir)
    marker = run_dir / "completed" / f"{avatar_id}.json"
    if marker.is_file():
        try:
            return float(json.loads(marker.read_text())["user_k_tokens"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    system_log = run_dir / "system_log.txt"
    if system_log.is_file():
        pattern = re.compile(
            rf"User {avatar_id} finished.*?Total token cost: ([0-9.]+)k"
        )
        matches = pattern.findall(system_log.read_text(errors="replace"))
        if matches:
            return float(matches[-1])
    return 0.0


def _exception_status(exc):
    current = exc
    while current is not None:
        for name in ("status_code", "code", "http_status"):
            value = getattr(current, name, None)
            if isinstance(value, int):
                return value
        match = re.search(r"HTTP\s+(\d{3})", str(current))
        if match:
            return int(match.group(1))
        current = current.__cause__ or current.__context__
    return None


def log_api_error(path, exc, *, avatar_id, component, attempt=None, model=None):
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "avatar_id": avatar_id,
        "component": component,
        "attempt": attempt,
        "model": model,
        "exception_type": type(exc).__name__,
        "http_status": _exception_status(exc),
        "message": str(exc)[:2000],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return record
