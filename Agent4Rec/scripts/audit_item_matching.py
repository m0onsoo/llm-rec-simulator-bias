"""Audit candidate-constrained item matching in an Agent4Rec run."""

import argparse
from collections import Counter
import json
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    counts = {event: Counter() for event in ("align", "watch", "like")}
    violations = []
    pages = 0
    for path in sorted((args.run_dir / "behavior").glob("*.pkl")):
        with path.open("rb") as handle:
            behavior = pickle.load(handle)
        for page_number, page in behavior.items():
            pages += 1
            candidates = {int(item_id) for item_id in page.get("recommended_id", [])}
            for event in counts:
                records = page.get(f"{event}_matches", [])
                for record in records:
                    method = record.get("match_method", "missing")
                    counts[event][method] += 1
                    item_id = record.get("item_id")
                    if item_id is not None and int(item_id) not in candidates:
                        violations.append({
                            "user": path.stem,
                            "page": page_number,
                            "event": event,
                            "item_id": item_id,
                        })
            if len(page.get("rating_id", [])) != len(page.get("rating", [])):
                violations.append({
                    "user": path.stem,
                    "page": page_number,
                    "event": "rating_length_mismatch",
                })

    report = {
        "run_dir": str(args.run_dir),
        "users": len(list((args.run_dir / "behavior").glob("*.pkl"))),
        "pages": pages,
        "match_methods": {event: dict(counter) for event, counter in counts.items()},
        "candidate_or_rating_violations": violations,
    }
    output = args.output or args.run_dir / "item_matching_audit.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
