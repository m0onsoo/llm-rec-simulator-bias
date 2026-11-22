#!/usr/bin/env python3
"""Generate an Agent4Rec-compatible activity-group degree report for SimUSER."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_NAMES = {1: "Infrequent", 2: "Occasional", 3: "Frequent"}
DEGREE_BINS = [0, 9, 24, 49, 99, 199, 399, np.inf]
DEGREE_LABELS = ["1–9", "10–24", "25–49", "50–99", "100–199", "200–399", "400+"]


def load_user_items(path):
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            values = [int(value) for value in raw.split()]
            if not values:
                continue
            user_id, items = values[0], values[1:]
            if len(items) != len(set(items)):
                raise ValueError("duplicate interaction at %s:%d" % (path, line_number))
            result[user_id] = items
    return result


def describe(values):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(values) == 0:
        raise ValueError("cannot describe an empty value set")
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
    quantiles = np.quantile(values, [0, .1, .25, .5, .75, .9, 1])
    return {
        "n": len(values), "mean": mean, "std": std,
        "ci_low": mean - ci, "ci_high": mean + ci,
        "min": quantiles[0], "p10": quantiles[1], "p25": quantiles[2],
        "median": quantiles[3], "p75": quantiles[4], "p90": quantiles[5],
        "max": quantiles[6],
    }


def markdown_table(headers, rows):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "%.4f" % value
        return str(value)

    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    result.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(result)


def distribution_rows(frame, column):
    rows = []
    for group in (1, 2, 3):
        stats = describe(frame.loc[frame.activity_group == group, column])
        rows.append([
            "%d (%s)" % (group, GROUP_NAMES[group]), stats["n"], stats["mean"],
            stats["std"], "%.4f–%.4f" % (stats["ci_low"], stats["ci_high"]),
            stats["min"], stats["p10"], stats["p25"], stats["median"],
            stats["p75"], stats["p90"], stats["max"],
        ])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-users", type=int, default=200)
    parser.add_argument(
        "--activity-prompt", choices=["included", "removed"], default="included"
    )
    parser.add_argument("--artifact-tag", default="simuser")
    args = parser.parse_args()

    aggregate = args.run_dir / "aggregate"
    events = pd.read_csv(aggregate / "selected_item_events.csv")
    users = pd.read_csv(aggregate / "user_selection_metrics.csv")
    train = load_user_items(args.dataset_dir / "cf_data" / "train.txt")
    degree = Counter()
    for items in train.values():
        degree.update(set(items))

    if len(users) != args.num_users or set(users.user_id) != set(range(args.num_users)):
        raise ValueError("expected the fixed user panel 0..%d" % (args.num_users - 1))
    expected_degrees = events.item_id.map(degree)
    if not np.array_equal(events.train_degree.to_numpy(), expected_degrees.to_numpy()):
        raise ValueError("selected-event degrees do not match cf_data/train.txt")

    selected_user = events.groupby(["user_id", "activity_group"], as_index=False).agg(
        selected_count=("item_id", "size"),
        mean_selected_train_degree=("train_degree", "mean"),
        median_selected_train_degree=("train_degree", "median"),
    )
    selected_user = users[["user_id", "activity_group", "pages_browsed", "exposed_count"]].merge(
        selected_user, on=["user_id", "activity_group"], how="left", validate="one_to_one"
    )
    selected_user["selected_count"] = selected_user.selected_count.fillna(0).astype(int)
    users_without_selection = selected_user.loc[
        selected_user.mean_selected_train_degree.isna(), "user_id"
    ].astype(int).tolist()

    gt_rows = []
    for user_id, activity_group in users[["user_id", "activity_group"]].itertuples(index=False):
        item_degrees = [degree[item_id] for item_id in train[int(user_id)]]
        gt_rows.append({
            "user_id": int(user_id), "activity_group": int(activity_group),
            "train_interactions": len(item_degrees),
            "mean_gt_train_item_degree": float(np.mean(item_degrees)),
        })
    gt_user = pd.DataFrame(gt_rows)
    comparison = selected_user.merge(gt_user, on=["user_id", "activity_group"], validate="one_to_one")

    group_rows = []
    comparison_rows = []
    for group in (1, 2, 3):
        group_users = comparison[comparison.activity_group == group]
        group_events = events[events.activity_group == group]
        gt_mean = group_users.mean_gt_train_item_degree.mean()
        selected_mean = group_users.mean_selected_train_degree.mean()
        group_rows.append([
            "%d (%s)" % (group, GROUP_NAMES[group]), len(group_users), len(group_events),
            group_users.pages_browsed.mean(), group_users.selected_count.mean(),
        ])
        comparison_rows.append([
            "%d (%s)" % (group, GROUP_NAMES[group]), gt_mean, selected_mean,
            selected_mean - gt_mean, selected_mean / gt_mean,
        ])

    top_count = max(1, int(np.ceil(len(degree) * .2)))
    head_items = {item_id for item_id, _ in degree.most_common(top_count)}
    events["is_head_20pct"] = events.item_id.isin(head_items)
    events["degree_bin"] = pd.cut(
        events.train_degree, bins=DEGREE_BINS, labels=DEGREE_LABELS, include_lowest=True
    )
    histogram = events.groupby(
        ["degree_bin", "activity_group"], observed=False
    ).size().unstack(fill_value=0)
    histogram = histogram.div(histogram.sum(axis=0), axis=1) * 100
    histogram_rows = [
        [label] + [histogram.loc[label, group] for group in (1, 2, 3)]
        for label in DEGREE_LABELS
    ]
    head_rows = []
    for group in (1, 2, 3):
        values = events[events.activity_group == group]
        head_rows.append([
            "%d (%s)" % (group, GROUP_NAMES[group]),
            100 * values.is_head_20pct.mean(), values.train_degree.mean(),
        ])

    headers = [
        "Activity group", "Users/events", "Mean", "SD", "Mean 95% CI", "Min",
        "P10", "P25", "Median", "P75", "P90", "Max",
    ]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    activity_description = (
        "included in interaction prompts" if args.activity_prompt == "included"
        else "removed from interaction prompts; retained only as an analysis label"
    )
    text = f"""# SimUSER Popularity Bias by Activity Group

Generated by `SimUSER/MovieLens/scripts/analyze_activity_degree.py` at `{generated}`.

## Scope and definitions

- Setting: vanilla SimUSER core framework, PQA disabled, perception disabled; explicit activity trait {activity_description}.
- Model: `gpt-6-luna`; dataset: MovieLens 1M.
- Fixed test panel: remapped user IDs 0–{args.num_users - 1} ({args.num_users} users).
- Input ranking: the first 20 LightGCN items used by the common Agent4Rec ranking; 4 items per page and at most 5 pages.
- Selection event: every item marked `WATCH` by SimUSER. Repeated selection events are retained if present.
- Item degree: number of distinct users interacting with the item in `cf_data/train.txt`; validation and test are excluded.
- Primary result is **user-balanced**: first calculate each user's mean degree across selected items, then summarize those user means within each activity group.
- The interaction-weighted result treats every selection event equally, so users selecting more items contribute more observations.
- A user with no `WATCH` event has no defined selected-item mean degree and is excluded only from the user-balanced selected-degree distribution; the user remains included in completion and GT statistics.

## Completion and activity-group composition

All {args.num_users} users completed successfully. Ranking order and masking audits passed; there were no missing or corrupt user results. Users with no selected item: {len(users_without_selection)} ({users_without_selection}).

{markdown_table(["Activity group", "Users", "Selection events", "Mean pages", "Mean selections/user"], group_rows)}

## Primary result: per-user mean selected-item train degree

{markdown_table(headers, distribution_rows(comparison, "mean_selected_train_degree"))}

## Same-panel training-history GT

This is the Agent4Rec GT definition restricted to the identical {args.num_users}-user panel. It is not affected by the simulator.

{markdown_table(headers, distribution_rows(comparison, "mean_gt_train_item_degree"))}

## Selected items versus same-panel GT

Positive delta and a ratio above 1 indicate that SimUSER selected items with higher train degree than the activity group's historical training-item baseline.

{markdown_table(["Activity group", "GT mean", "Selected mean", "Absolute delta", "Selected/GT ratio"], comparison_rows)}

## Interaction-weighted selected-item degree distribution

{markdown_table(headers, distribution_rows(events, "train_degree"))}

### Degree-bin shares among selected events (%)

{markdown_table(["Train degree", "Infrequent", "Occasional", "Frequent"], histogram_rows)}

### Selection share assigned to top-20% train-degree items

The head set is fixed globally using the top 20% of items ranked by train degree, matching the Agent4Rec GT report.

{markdown_table(["Activity group", "Head selection share (%)", "Mean interaction-weighted degree"], head_rows)}

## Interpretation boundary

These results describe which items vanilla SimUSER selected under the common LightGCN ranking. Differences across activity groups may reflect different ranking exposure, historical preferences, persona content, or simulator behavior. A causal claim about the injected activity trait requires an otherwise identical activity-removed or activity-shuffled SimUSER run.
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    artifact_dir = args.output.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        artifact_dir / ("%s_u%d_activity_degree_users.csv" % (args.artifact_tag, args.num_users)), index=False
    )
    events.to_csv(
        artifact_dir / ("%s_u%d_selected_events.csv" % (args.artifact_tag, args.num_users)), index=False
    )
    print(args.output)


if __name__ == "__main__":
    main()
