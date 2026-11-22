#!/usr/bin/env python3
"""Paired comparison of SimUSER with and without explicit activity prompts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_NAMES = {1: "Infrequent", 2: "Occasional", 3: "Frequent"}


def bootstrap_ci(values, seed, repetitions=10000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [.025, .975]))


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


def load_metrics(run_dir, label, num_users):
    frame = pd.read_csv(run_dir / "aggregate" / "user_selection_metrics.csv")
    if len(frame) != num_users or set(frame.user_id) != set(range(num_users)):
        raise ValueError("%s does not contain the fixed panel 0..%d" % (label, num_users - 1))
    frame["selected_item_set"] = frame.selected_item_ids.map(lambda value: set(json.loads(value)))
    return frame.rename(columns={
        "pages_browsed": "%s_pages" % label,
        "selected_count": "%s_count" % label,
        "mean_selected_train_degree": "%s_degree" % label,
        "selected_item_set": "%s_set" % label,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-activity-run", type=Path, required=True)
    parser.add_argument("--without-activity-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    with_activity = load_metrics(args.with_activity_run, "with", args.num_users)
    without_activity = load_metrics(args.without_activity_run, "without", args.num_users)
    keep = ["user_id", "activity_group"]
    columns = keep + ["with_pages", "with_count", "with_degree", "with_set"]
    paired = with_activity[columns].merge(
        without_activity[keep + ["without_pages", "without_count", "without_degree", "without_set"]],
        on=keep, validate="one_to_one",
    )
    paired["count_delta"] = paired.without_count - paired.with_count
    paired["page_delta"] = paired.without_pages - paired.with_pages
    paired["degree_delta"] = paired.without_degree - paired.with_degree
    paired["jaccard"] = paired.apply(
        lambda row: (
            len(row.with_set & row.without_set) / len(row.with_set | row.without_set)
            if row.with_set | row.without_set else 1.0
        ), axis=1,
    )
    paired["removed_from_watch"] = paired.apply(
        lambda row: len(row.with_set - row.without_set), axis=1
    )
    paired["added_to_watch"] = paired.apply(
        lambda row: len(row.without_set - row.with_set), axis=1
    )

    rows = []
    for group in (1, 2, 3):
        values = paired[paired.activity_group == group]
        degree_values = values.degree_delta.dropna()
        low, high = bootstrap_ci(degree_values, args.seed + group)
        rows.append([
            "%d (%s)" % (group, GROUP_NAMES[group]), len(values), len(degree_values),
            values.with_degree.mean(), values.without_degree.mean(), degree_values.mean(),
            "%.4f–%.4f" % (low, high), values.with_count.mean(),
            values.without_count.mean(), values.count_delta.mean(), values.jaccard.mean(),
        ])

    switch_rows = []
    for group in (1, 2, 3):
        values = paired[paired.activity_group == group]
        switch_rows.append([
            "%d (%s)" % (group, GROUP_NAMES[group]),
            values.page_delta.mean(), values.removed_from_watch.mean(),
            values.added_to_watch.mean(),
        ])

    text = f"""# SimUSER Activity-Prompt Ablation

## Design

- Paired users: remapped IDs 0–{args.num_users - 1}.
- Both settings use the same LightGCN ranking, item order, split, persona, pickiness, diversity, conformity, episodic memory, KG memory, model, and five-page limit.
- The only intended prompt difference is whether the explicit activity description is included.
- Activity group is retained only as an analysis label in the without-activity setting.
- Degree deltas are `without activity − with activity`; negative values mean activity removal lowered selected-item popularity.
- Degree comparisons include only paired users with at least one selection in both settings. Selection-count and page comparisons include every user.

## Paired user-balanced comparison

{markdown_table(["Activity group", "Users", "Degree pairs", "With activity degree", "Without activity degree", "Degree delta", "Bootstrap 95% CI", "With activity selections", "Without activity selections", "Selection-count delta", "Selection Jaccard"], rows)}

## Behavioral changes

{markdown_table(["Activity group", "Page-count delta", "WATCH→not-WATCH per user", "not-WATCH→WATCH per user"], switch_rows)}

## Interpretation boundary

A non-zero paired change is evidence that the explicit activity prompt changes simulator behavior under otherwise fixed inputs. It does not by itself identify whether the mechanism is preference leakage, fatigue/action reasoning, or residual model stochasticity. Page-count changes and selection-set changes should therefore be reported separately.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    artifact = args.output.parent / "artifacts" / "simuser_activity_ablation_paired_users.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    paired.drop(columns=["with_set", "without_set"]).to_csv(artifact, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
