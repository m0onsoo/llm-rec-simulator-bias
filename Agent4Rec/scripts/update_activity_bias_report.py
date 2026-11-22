"""Append with/without-activity simulation comparisons to the GT report."""

import argparse
from collections import Counter
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
START = "<!-- SIMULATION_COMPARISON_START -->"
END = "<!-- SIMULATION_COMPARISON_END -->"
EVENT_FIELDS = {
    "recommended": "recommended_id",
    "align": "align_id",
    "watch": "watch_id",
    "like": "like_id",
}
GROUP_NAMES = {1: "Infrequent", 2: "Occasional", 3: "Frequent"}


def load_train_degree(path):
    degree = Counter()
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            values = [int(value) for value in raw.split()]
            degree.update(set(values[1:]))
    return degree


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(headers, rows):
    def value(x):
        if isinstance(x, (float, np.floating)):
            return "NA" if np.isnan(x) else f"{x:.4f}"
        return str(x)
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(value(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def collect_run(run_dir, mode, degree, activity):
    rows = []
    behavior_dir = run_dir / "behavior"
    for user_id, group in activity.items():
        path = behavior_dir / f"{user_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            behavior = pickle.load(handle)
        for event, field in EVENT_FIELDS.items():
            item_ids = []
            for page in behavior.values():
                item_ids.extend(int(item_id) for item_id in page.get(field, []))
            item_degrees = [degree[item_id] for item_id in item_ids]
            rows.append({
                "mode": mode, "user_id": user_id, "activity": group, "event": event,
                "event_count": len(item_ids),
                "mean_selected_degree": np.mean(item_degrees) if item_degrees else np.nan,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-activity", required=True, type=Path)
    parser.add_argument("--without-activity", type=Path)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "analysis" / "POPULARITY_BIAS_BY_ACTIVITY.md",
    )
    args = parser.parse_args()

    gt = pd.read_csv(ROOT / "analysis" / "artifacts" / "activity_degree_gt_users.csv")
    activity = dict(zip(gt.user_id.astype(int), gt.activity.astype(int)))
    degree = load_train_degree(ROOT / "datasets" / "ml-1m" / "cf_data" / "train.txt")
    rows = collect_run(args.with_activity.resolve(), "included", degree, activity)
    modes = ["included"]
    if args.without_activity:
        rows += collect_run(args.without_activity.resolve(), "excluded", degree, activity)
        modes.append("excluded")
    data = pd.DataFrame(rows).merge(
        gt[["user_id", "mean_train_item_degree"]], on="user_id", validate="many_to_one"
    )
    data["individual_gt_delta"] = data.mean_selected_degree - data.mean_train_item_degree
    impressions = data[data.event == "recommended"][
        ["mode", "user_id", "mean_selected_degree"]
    ].rename(columns={"mean_selected_degree": "mean_impression_degree"})
    data = data.merge(impressions, on=["mode", "user_id"], validate="many_to_one")
    data["individual_impression_delta"] = (
        data.mean_selected_degree - data.mean_impression_degree
    )

    summary_rows = []
    for mode in modes:
        for event in EVENT_FIELDS:
            for group in (1, 2, 3):
                values = data[(data["mode"] == mode) & (data.event == event) &
                              (data.activity == group)]
                observed = values.dropna(subset=["mean_selected_degree"])
                group_gt = gt.loc[gt.activity == group, "mean_train_item_degree"].mean()
                summary_rows.append([
                    mode, event, f"{group} ({GROUP_NAMES[group]})", len(observed),
                    100 * len(observed) / len(values), observed.mean_selected_degree.mean(),
                    group_gt, observed.individual_gt_delta.mean(),
                    observed.individual_impression_delta.mean(),
                ])

    effect_rows = []
    if args.without_activity:
        for event in EVENT_FIELDS:
            for group in (1, 2, 3):
                inc = data[(data["mode"] == "included") & (data.event == event) &
                           (data.activity == group)][["user_id", "mean_selected_degree"]]
                exc = data[(data["mode"] == "excluded") & (data.event == event) &
                           (data.activity == group)][["user_id", "mean_selected_degree"]]
                paired = inc.merge(exc, on="user_id", suffixes=("_included", "_excluded")).dropna()
                inc_mean = inc.mean_selected_degree.mean()
                exc_mean = exc.mean_selected_degree.mean()
                effect_rows.append([
                    event, f"{group} ({GROUP_NAMES[group]})", inc_mean, exc_mean,
                    inc_mean - exc_mean, len(paired),
                    (paired.mean_selected_degree_included -
                     paired.mean_selected_degree_excluded).mean(),
                ])

    with_rank = args.with_activity / "rankings" / "full_rankings_1000.npy"
    comparison_text = ""
    without_run_text = "Not run yet"
    ranks_equal_text = "N/A until the excluded condition is complete"
    if args.without_activity:
        without_rank = args.without_activity / "rankings" / "full_rankings_1000.npy"
        ranks_equal_text = str(file_sha256(with_rank) == file_sha256(without_rank))
        without_run_text = f"`{args.without_activity.resolve()}`"
        comparison_text = f"""
### Activity-prompt effect

`Included − excluded` isolates the prompt condition. The paired column uses only users with the event in both runs.

{markdown_table(["Event", "Activity group", "Included mean", "Excluded mean", "Unpaired difference", "Paired users", "Paired mean difference"], effect_rows)}
"""
    section = f"""{START}
## Simulation comparison: activity persona included vs excluded

- Included run: `{args.with_activity.resolve()}`
- Excluded run: {without_run_text}
- Cached ranking arrays byte-identical: `{ranks_equal_text}`
- `Selected − GT` is calculated per user before taking the group mean. Positive values indicate selection of higher-degree items than that user's own training-history baseline.
- `Selected − impression` is also calculated per user. It separates selection from the popularity level already present in the displayed recommendation pages.
- Unresolved title matches are excluded from the corresponding selected-event ID list; raw responses remain available for post-hoc repair.

### Group-specific results

{markdown_table(["Mode", "Event", "Activity group", "Users with event", "Coverage (%)", "Selected mean degree", "Group GT mean", "Mean individual (selected−GT)", "Mean individual (selected−impression)"], summary_rows)}
{comparison_text}
{END}"""

    report = args.report.read_text(encoding="utf-8")
    if START in report and END in report:
        prefix = report.split(START, 1)[0]
        suffix = report.split(END, 1)[1]
        report = prefix + section + suffix
    else:
        report = report.rstrip() + "\n\n" + section + "\n"
    args.report.write_text(report, encoding="utf-8")
    data.to_csv(ROOT / "analysis" / "artifacts" / "activity_prompt_comparison_users.csv",
                index=False)
    print(args.report)


if __name__ == "__main__":
    main()
