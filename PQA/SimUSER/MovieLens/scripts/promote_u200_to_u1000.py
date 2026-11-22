#!/usr/bin/env python3
"""Safely seed a 1,000-user run with completed 0..199 user artifacts."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from run_vanilla_common import atomic_json, sha256


STAGES = ("summaries", "candidates", "selection")


def copy_verified(source, target):
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = sha256(source)
    if target.exists():
        if not target.is_file() or sha256(target) != source_hash:
            raise ValueError("existing target differs from source: %s" % target)
        return source_hash
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".%s." % target.name, dir=str(target.parent))
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        if sha256(Path(temporary)) != source_hash:
            raise RuntimeError("copy hash mismatch: %s" % source)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return source_hash


def collection_hash(rows):
    digest = hashlib.sha256()
    for label, value in sorted(rows):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-profile-dir", type=Path, required=True)
    parser.add_argument("--source-memory-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--target-profile-dir", type=Path, required=True)
    parser.add_argument("--target-memory-dir", type=Path, required=True)
    parser.add_argument("--target-run-dir", type=Path, required=True)
    parser.add_argument("--source-ranking-json", type=Path, required=True)
    parser.add_argument("--target-ranking-json", type=Path, required=True)
    parser.add_argument("--num-source-users", type=int, default=200)
    args = parser.parse_args()

    audit = json.loads((args.source_run_dir / "audit_report.json").read_text(encoding="utf-8"))
    manifest_path = args.source_run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_users = list(range(args.num_source_users))
    if not audit.get("ok") or audit.get("complete_users") != args.num_source_users:
        raise ValueError("source audit is not a complete successful run")
    if manifest.get("user_ids") != expected_users:
        raise ValueError("source run user IDs are not the fixed prefix 0..%d" % (args.num_source_users - 1))
    required_invariants = {
        "setting": "agent4rec_common_ranking_vanilla_simuser_no_perception",
        "pqa_enabled": False, "perception_enabled": False,
        "model": "gpt-6-luna", "items_per_page": 4, "max_pages": 5,
    }
    for key, value in required_invariants.items():
        if manifest.get(key) != value:
            raise ValueError("source invariant differs for %s" % key)

    source_rankings = json.loads(args.source_ranking_json.read_text(encoding="utf-8"))
    target_rankings = json.loads(args.target_ranking_json.read_text(encoding="utf-8"))
    for uid in expected_users:
        if source_rankings[str(uid)] != target_rankings[str(uid)]:
            raise ValueError("ranking prefix differs for user %d" % uid)

    copied = []
    for stage in STAGES:
        for uid in expected_users:
            relative = Path(stage) / ("%d.json" % uid)
            value = copy_verified(
                args.source_profile_dir / relative,
                args.target_profile_dir / relative,
            )
            copied.append(("profile/%s" % relative, value))
    for uid in expected_users:
        relative = Path("episodic") / ("%d.json" % uid)
        value = copy_verified(
            args.source_memory_dir / relative,
            args.target_memory_dir / relative,
        )
        copied.append(("memory/%s" % relative, value))
    kg_hash = copy_verified(
        args.source_memory_dir / "kg_memory.json",
        args.target_memory_dir / "kg_memory.json",
    )
    copied.append(("memory/kg_memory.json", kg_hash))
    for uid in expected_users:
        relative = Path("users") / ("%d.json" % uid)
        value = copy_verified(
            args.source_run_dir / relative,
            args.target_run_dir / relative,
        )
        copied.append(("run/%s" % relative, value))

    source_error_log = args.source_run_dir / "api_errors.jsonl"
    if source_error_log.is_file():
        error_hash = copy_verified(source_error_log, args.target_run_dir / "api_errors.jsonl")
        copied.append(("run/api_errors.jsonl", error_hash))

    promotion = {
        "schema_version": 1,
        "source_users": [0, args.num_source_users - 1],
        "target_users": [0, 999],
        "source_run_manifest_sha256": sha256(manifest_path),
        "source_audit_sha256": sha256(args.source_run_dir / "audit_report.json"),
        "source_ranking_sha256": sha256(args.source_ranking_json),
        "target_ranking_sha256": sha256(args.target_ranking_json),
        "ranking_prefix_equal": True,
        "copied_file_count": len(copied),
        "copied_collection_sha256": collection_hash(copied),
        "invariants": required_invariants,
    }
    atomic_json(args.target_run_dir / "promotion_manifest.json", promotion)
    print(json.dumps(promotion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
