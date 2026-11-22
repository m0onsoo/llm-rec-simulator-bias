#!/usr/bin/env python3
"""Audit completeness, ranking exposure, selected IDs, and API errors."""

import argparse
import json
from pathlib import Path

from run_vanilla_common import atomic_json, load_split, reorder_pages, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ranking-json", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--num-users", type=int, default=200)
    args = parser.parse_args()
    rankings = json.loads(args.ranking_json.read_text(encoding="utf-8"))
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    train = load_split(args.train_split)
    test = load_split(args.test_split)
    missing, corrupt, ranking_mismatch, invalid_selected = [], [], [], []
    api_errors = []
    error_path = args.run_dir / "api_errors.jsonl"
    if error_path.is_file():
        for line in error_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                api_errors.append(json.loads(line))
            except ValueError:
                api_errors.append({"corrupt_line": line[:200]})
    for uid in range(args.num_users):
        path = args.run_dir / "users" / ("%d.json" % uid)
        if not path.is_file():
            missing.append(uid)
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status") != "complete" or row.get("user_id") != uid:
                raise ValueError("invalid completion record")
            pages = row["simulation"]["pages"]
        except (OSError, ValueError, KeyError, TypeError):
            corrupt.append(uid)
            continue
        expected = [int(value) for value in rankings[str(uid)]]
        if manifest.get("position_order", "original") == "random":
            original_pages = [expected[index:index + 4] for index in range(0, len(expected), 4)]
            expected_pages, _ = reorder_pages(original_pages, manifest["permutation_seed"], uid)
            expected = [mid for page in expected_pages for mid in page]
        actual = [mid for page in pages for mid in page["exposed_item_ids"]]
        if actual != expected[:len(actual)]:
            ranking_mismatch.append(uid)
        masked = set(train.get(uid, [])) | set(test.get(uid, []))
        if masked & set(actual):
            ranking_mismatch.append(uid)
        for page in pages:
            exposed = set(page["exposed_item_ids"])
            for selected in page["selected_item_ids"]:
                if selected not in exposed:
                    invalid_selected.append({"user_id": uid, "page": page["page"], "item_id": selected})
    report = {
        "requested_users": args.num_users,
        "complete_users": args.num_users - len(missing) - len(corrupt),
        "missing_users": missing,
        "corrupt_users": corrupt,
        "ranking_mismatch_users": sorted(set(ranking_mismatch)),
        "invalid_selected_items": invalid_selected,
        "api_error_records": len(api_errors),
        "api_error_status_counts": {
            str(status): sum(row.get("http_status") == status for row in api_errors)
            for status in sorted({row.get("http_status") for row in api_errors}, key=lambda value: str(value))
        },
        "ranking_json_sha256": sha256(args.ranking_json),
    }
    report["ok"] = not any((missing, corrupt, ranking_mismatch, invalid_selected))
    atomic_json(args.run_dir / "audit_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
