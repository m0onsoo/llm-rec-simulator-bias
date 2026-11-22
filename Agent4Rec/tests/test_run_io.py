import json
import pickle
from pathlib import Path

from simulation.run_io import (
    atomic_json,
    atomic_pickle,
    completed_user_outputs,
    log_api_error,
    token_cost_from_marker_or_log,
)


def test_completed_outputs_require_two_valid_pickles(tmp_path):
    atomic_pickle(tmp_path / "behavior" / "3.pkl", {1: {"page": 1}})
    assert completed_user_outputs(tmp_path, 3) is None

    atomic_pickle(tmp_path / "interview" / "3.pkl", {"interview": ["ok"]})
    behavior, interview = completed_user_outputs(tmp_path, 3)
    assert behavior[1]["page"] == 1
    assert interview["interview"] == ["ok"]

    (tmp_path / "interview" / "3.pkl").write_bytes(b"truncated")
    assert completed_user_outputs(tmp_path, 3) is None


def test_token_cost_prefers_marker_and_supports_legacy_log(tmp_path):
    (tmp_path / "system_log.txt").write_text(
        "User 4 finished after 5 pages. Total token cost: 1.25k.\n"
    )
    assert token_cost_from_marker_or_log(tmp_path, 4) == 1.25

    atomic_json(tmp_path / "completed" / "4.json", {"user_k_tokens": 2.5})
    assert token_cost_from_marker_or_log(tmp_path, 4) == 2.5


def test_error_log_records_http_status(tmp_path):
    error = RuntimeError("OpenAI-compatible API returned HTTP 429: throttled")
    path = tmp_path / "api_errors.jsonl"
    log_api_error(
        path,
        error,
        avatar_id=7,
        component="avatar_chat",
        attempt=2,
        model="gpt-6-luna",
    )
    record = json.loads(path.read_text())
    assert record["avatar_id"] == 7
    assert record["http_status"] == 429
    assert record["attempt"] == 2
