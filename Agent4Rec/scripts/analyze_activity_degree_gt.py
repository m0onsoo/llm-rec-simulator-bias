"""Build the activity-group ground-truth item-degree report.

Item degree is the number of distinct users interacting with the item in the
training split. The report keeps a manual appendix so later simulation settings
can be appended without losing the immutable GT snapshot.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MANUAL_MARKER = "<!-- MANUAL_APPENDIX_START -->"
GROUP_NAMES = {1: "Infrequent", 2: "Occasional", 3: "Frequent"}


def load_user_items(path):
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            values = [int(value) for value in raw.split()]
            if not values:
                continue
            user, items = values[0], values[1:]
            if len(items) != len(set(items)):
                raise ValueError(f"Duplicate user-item interaction at {path}:{line_number}")
            result[user] = items
    return result


def markdown_table(headers, rows):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def describe(values):
    values = np.asarray(values, dtype=float)
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


def summary_rows(frame, value_column):
    rows = []
    for group in (1, 2, 3):
        stats = describe(frame.loc[frame.activity == group, value_column])
        rows.append([
            f"{group} ({GROUP_NAMES[group]})", stats["n"], stats["mean"], stats["std"],
            f'{stats["ci_low"]:.4f}–{stats["ci_high"]:.4f}', stats["min"], stats["p10"],
            stats["p25"], stats["median"], stats["p75"], stats["p90"], stats["max"],
        ])
    return rows


def plot_user_balanced_gt(user_df, output):
    groups = (1, 2, 3)
    stats = [
        describe(user_df.loc[user_df.activity == group, "mean_train_item_degree"])
        for group in groups
    ]
    means = [values["mean"] for values in stats]
    errors = [values["mean"] - values["ci_low"] for values in stats]
    labels = [GROUP_NAMES[group] for group in groups]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(labels, means, yerr=errors, capsize=5, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("Mean train-item degree")
    ax.set_xlabel("Activity group")
    ax.set_title("User-balanced training-history popularity")
    ax.set_ylim(0, max(means) * 1.22)
    ax.grid(axis="y", alpha=0.25)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0,
                f"{mean:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "analysis" / "POPULARITY_BIAS_BY_ACTIVITY.md",
    )
    args = parser.parse_args()

    dataset = ROOT / "datasets" / args.dataset
    train_path = dataset / "cf_data" / "train.txt"
    activity_path = dataset / "simulation" / "user_statistic.csv"
    activity_num_path = dataset / "simulation" / "user_statistic_num.csv"
    train = load_user_items(train_path)
    activity = pd.read_csv(activity_path)[["user_id", "activity"]]
    activity_num = pd.read_csv(activity_num_path)[["user_id", "activity_num"]]
    users = activity.merge(activity_num, on="user_id", validate="one_to_one")

    degree = Counter()
    for items in train.values():
        degree.update(set(items))

    user_rows = []
    activity_by_user = dict(zip(users.user_id, users.activity))
    for user_id, items in train.items():
        item_degrees = [degree[item_id] for item_id in items]
        group = int(activity_by_user[user_id])
        user_rows.append({
            "user_id": user_id,
            "activity": group,
            "train_interactions": len(items),
            "mean_train_item_degree": float(np.mean(item_degrees)),
            "median_train_item_degree": float(np.median(item_degrees)),
        })
    user_df = pd.DataFrame(user_rows).merge(activity_num, on="user_id", validate="one_to_one")
    if len(user_df) != 1000:
        raise ValueError(f"Expected 1000 users, found {len(user_df)}")

    user_headers = [
        "Activity group", "Users", "Mean", "SD", "Mean 95% CI", "Min", "P10",
        "P25", "Median", "P75", "P90", "Max",
    ]
    group_rows = []
    for group in (1, 2, 3):
        group_users = user_df[user_df.activity == group]
        group_rows.append([
            f"{group} ({GROUP_NAMES[group]})", len(group_users),
            int(group_users.train_interactions.sum()),
            group_users.activity_num.min(), group_users.activity_num.max(),
            group_users.activity_num.mean(),
        ])

    user_means = {
        group: user_df.loc[user_df.activity == group, "mean_train_item_degree"].mean()
        for group in (1, 2, 3)
    }
    lift = user_means[3] / user_means[1]
    delta = user_means[3] - user_means[1]
    spearman = user_df[["activity_num", "mean_train_item_degree"]].corr(
        method="spearman"
    ).iloc[0, 1]

    existing_manual = ""
    if args.output.exists():
        existing = args.output.read_text(encoding="utf-8")
        if MANUAL_MARKER in existing:
            existing_manual = existing.split(MANUAL_MARKER, 1)[1].lstrip("\n")
    if not existing_manual:
        existing_manual = """## Experiment comparison registry

Append future simulation settings here without modifying the GT section above.

| Setting | Activity prompt | Selection event | Users | Degree definition | Result artifact | Status |
| --- | --- | --- | ---: | --- | --- | --- |
| GT: training history | N/A | Train interaction | 1,000 | Train user degree | This report | Complete |

## Planned simulation comparisons

For each setting, report `recommended_id`, `align_id`, `watch_id`, and `like_id` separately. Compare the user-balanced selected-item mean degree with this GT using both an absolute delta and a ratio. A causal claim about persona injection requires an otherwise identical run with the activity description removed or shuffled.
"""

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = f"""# Popularity Bias by Activity Group

Generated by `scripts/analyze_activity_degree_gt.py` at `{generated}`.

## Scope and definitions

- Dataset: `{args.dataset}`; users: 1,000.
- Activity groups are the repository-provided groups derived from each user's full MovieLens rating count: group 1 below the global 60th percentile, group 2 from the 60th to below the 90th percentile, and group 3 at or above the 90th percentile.
- Item degree is the number of distinct users interacting with that item in `cf_data/train.txt`. Validation and test interactions are excluded from the degree calculation.
- Primary GT is **user-balanced**: calculate each user's mean degree across their training items, then summarize those user means within each activity group.
- This report is descriptive. Group differences already present in training history are not evidence that the injected persona caused bias.

## Activity group composition

{markdown_table(["Activity group", "Users", "Train interactions", "Min full-history count", "Max full-history count", "Mean full-history count"], group_rows)}

## Primary GT: distribution of per-user mean train-item degree

{markdown_table(user_headers, summary_rows(user_df, "mean_train_item_degree"))}

![Mean train-item degree by activity group](figures/activity_group_gt_mean_degree.png)

Bars show group means; error bars show 95% confidence intervals across users.

Frequent minus infrequent mean degree: `{delta:.4f}`. Frequent/infrequent ratio: `{lift:.4f}`. Spearman correlation between the original activity count and each user's mean train-item degree: `{spearman:.4f}`.

## Interpretation boundary

This GT answers whether activity groups already consumed differently popular items in the training data. To test whether **injecting** the activity persona creates or amplifies popularity bias, compare otherwise identical simulations with the activity prompt enabled versus removed or shuffled, and measure the change relative to this group-specific GT.

{MANUAL_MARKER}
{existing_manual.rstrip()}
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    user_df.to_csv(artifact_dir / "activity_degree_gt_users.csv", index=False)
    plot_user_balanced_gt(user_df, args.output.parent / "figures" / "activity_group_gt_mean_degree.png")
    args.output.write_text(text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
