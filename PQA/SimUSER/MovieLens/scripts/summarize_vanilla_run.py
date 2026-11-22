#!/usr/bin/env python3
"""Aggregate per-user vanilla SimUSER selections without losing event detail."""

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from run_vanilla_common import atomic_json


def atomic_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=str(path.parent),
        prefix=path.name + ".", suffix=".tmp", delete=False,
    )
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-users", type=int, default=200)
    args = parser.parse_args()

    event_rows = []
    user_rows = []
    item_counts = Counter()
    item_users = defaultdict(set)
    item_degrees = {}
    group_user_means = defaultdict(list)
    group_event_degrees = defaultdict(list)

    for uid in range(args.num_users):
        path = args.run_dir / "users" / ("%d.json" % uid)
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "complete" or row.get("user_id") != uid:
            raise ValueError("incomplete or mismatched user result: %s" % path)
        profile = row["profile"]
        activity = int(profile["activity_group"])
        diversity = int(profile["diversity_group"])
        conformity = int(profile["conformity_group"])
        selected_ids = []
        selected_degrees = []
        exposed_count = 0
        for page in row["simulation"]["pages"]:
            degree_by_item = {
                int(item["item_id"]): int(item["train_degree"])
                for item in page["exposed_items"]
            }
            exposed_count += len(page["exposed_item_ids"])
            for position, item_id in enumerate(page["selected_item_ids"], start=1):
                item_id = int(item_id)
                degree = degree_by_item[item_id]
                event_rows.append({
                    "user_id": uid, "activity_group": activity,
                    "diversity_group": diversity, "conformity_group": conformity,
                    "page": int(page["page"]), "selection_position_on_page": position,
                    "item_id": item_id, "train_degree": degree,
                })
                selected_ids.append(item_id)
                selected_degrees.append(degree)
                item_counts[item_id] += 1
                item_users[item_id].add(uid)
                item_degrees[item_id] = degree
                group_event_degrees[activity].append(degree)
        user_mean = mean(selected_degrees)
        if user_mean is not None:
            group_user_means[activity].append(user_mean)
        user_rows.append({
            "user_id": uid, "activity_group": activity,
            "diversity_group": diversity, "conformity_group": conformity,
            "pages_browsed": len(row["simulation"]["pages"]),
            "exposed_count": exposed_count, "selected_count": len(selected_ids),
            "mean_selected_train_degree": "" if user_mean is None else "%.10f" % user_mean,
            "selected_item_ids": json.dumps(selected_ids, separators=(",", ":")),
        })

    item_rows = [
        {
            "item_id": item_id, "train_degree": item_degrees[item_id],
            "selection_count": item_counts[item_id],
            "unique_user_count": len(item_users[item_id]),
        }
        for item_id in sorted(item_counts, key=lambda value: (-item_counts[value], value))
    ]
    summary = {
        "requested_users": args.num_users,
        "selection_events": len(event_rows),
        "unique_selected_items": len(item_rows),
        "activity_groups": {
            str(group): {
                "users": sum(int(row["activity_group"]) == group for row in user_rows),
                "users_with_selection": len(group_user_means[group]),
                "selection_events": len(group_event_degrees[group]),
                "user_balanced_mean_selected_train_degree": mean(group_user_means[group]),
                "interaction_weighted_mean_selected_train_degree": mean(group_event_degrees[group]),
            }
            for group in (1, 2, 3)
        },
    }

    aggregate = args.run_dir / "aggregate"
    atomic_csv(
        aggregate / "selected_item_events.csv",
        ["user_id", "activity_group", "diversity_group", "conformity_group",
         "page", "selection_position_on_page", "item_id", "train_degree"],
        event_rows,
    )
    atomic_csv(
        aggregate / "selected_item_counts.csv",
        ["item_id", "train_degree", "selection_count", "unique_user_count"],
        item_rows,
    )
    atomic_csv(
        aggregate / "user_selection_metrics.csv",
        ["user_id", "activity_group", "diversity_group", "conformity_group",
         "pages_browsed", "exposed_count", "selected_count",
         "mean_selected_train_degree", "selected_item_ids"],
        user_rows,
    )
    atomic_json(aggregate / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
