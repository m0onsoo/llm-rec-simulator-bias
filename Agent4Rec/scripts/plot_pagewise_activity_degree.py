"""Plot page-wise train-item degree for activity included/excluded runs."""

import argparse
from collections import Counter
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP_NAMES = {1: "Infrequent", 2: "Occasional", 3: "Frequent"}
EVENT_FIELDS = {"impression": "recommended_id", "watch": "watch_id"}
MODE_LABELS = {"included": "Activity included", "excluded": "Activity excluded"}
COLORS = {"included": "#2864DC", "excluded": "#DC503C"}


def load_train_degree(path):
    degree = Counter()
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            values = [int(value) for value in raw.split()]
            degree.update(set(values[1:]))
    return degree


def collect_run(run_dir, mode, activity, degree, max_pages):
    rows = []
    for user_id, group in activity.items():
        path = run_dir / "behavior" / f"{user_id}.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            behavior = pickle.load(handle)
        pages = {int(record.get("page", page)): record for page, record in behavior.items()}
        for page in range(1, max_pages + 1):
            record = pages.get(page)
            reached = record is not None
            for event, field in EVENT_FIELDS.items():
                item_ids = [] if record is None else [int(x) for x in record.get(field, [])]
                values = [degree[item_id] for item_id in item_ids]
                rows.append({
                    "mode": mode,
                    "user_id": user_id,
                    "activity": group,
                    "page": page,
                    "event": event,
                    "reached_page": reached,
                    "event_count": len(item_ids),
                    "mean_degree": float(np.mean(values)) if values else np.nan,
                })
    return rows


def summarize(frame, group_sizes):
    rows = []
    keys = ["mode", "activity", "page", "event"]
    for key, values in frame.groupby(keys, sort=True):
        mode, activity, page, event = key
        observed = values.mean_degree.dropna()
        n = len(observed)
        std = observed.std(ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n else np.nan
        rows.append({
            "mode": mode,
            "activity": activity,
            "activity_name": GROUP_NAMES[activity],
            "page": page,
            "event": event,
            "group_users": group_sizes[activity],
            "users_reached": int(values.reached_page.sum()),
            "reach_rate": float(values.reached_page.mean()),
            "users_with_event": n,
            "event_coverage_among_reached": (
                n / values.reached_page.sum() if values.reached_page.sum() else np.nan
            ),
            "mean_degree": observed.mean() if n else np.nan,
            "std_degree": std,
            "sem_degree": sem,
            "ci95_low": observed.mean() - 1.96 * sem if n else np.nan,
            "ci95_high": observed.mean() + 1.96 * sem if n else np.nan,
        })
    return pd.DataFrame(rows)


def plot(summary, gt, output):
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.8), sharex=True, sharey=True)
    for column, group in enumerate((1, 2, 3)):
        for row, event in enumerate(("impression", "watch")):
            ax = axes[row, column]
            for mode in ("included", "excluded"):
                values = summary[
                    (summary.activity == group)
                    & (summary.event == event)
                    & (summary["mode"] == mode)
                ].sort_values("page")
                x = values.page.to_numpy(dtype=float)
                y = values.mean_degree.to_numpy(dtype=float)
                low = values.ci95_low.to_numpy(dtype=float)
                high = values.ci95_high.to_numpy(dtype=float)
                ax.plot(
                    x, y, marker="o", linewidth=2.1, markersize=5,
                    color=COLORS[mode], label=MODE_LABELS[mode],
                )
                ax.fill_between(x, low, high, color=COLORS[mode], alpha=0.13)
            ax.axhline(
                gt[group], color="#333333", linestyle=":", linewidth=1.8,
                label="Training-history GT",
            )
            ax.grid(axis="y", alpha=0.22)
            ax.set_xticks(range(1, 6))
            if row == 0:
                ax.set_title(GROUP_NAMES[group], fontsize=12, fontweight="bold")
            if column == 0:
                ax.set_ylabel(
                    "Impression mean train degree" if event == "impression"
                    else "Watch-selected mean train degree"
                )
            if row == 1:
                ax.set_xlabel("Page")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=3, frameon=False,
    )
    fig.suptitle(
        "Page-wise item popularity by activity-prompt condition",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.012,
        "Points are user-balanced means among users with the event; bands are 95% normal CIs. "
        "GT is each activity group's mean training-history item degree.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--included", required=True, type=Path)
    parser.add_argument("--excluded", required=True, type=Path)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "analysis" / "figures" / "pagewise_activity_degree.png",
    )
    args = parser.parse_args()

    gt = pd.read_csv(ROOT / "analysis" / "artifacts" / "activity_degree_gt_users.csv")
    activity = dict(zip(gt.user_id.astype(int), gt.activity.astype(int)))
    degree = load_train_degree(ROOT / "datasets" / "ml-1m" / "cf_data" / "train.txt")
    rows = collect_run(args.included.resolve(), "included", activity, degree, args.max_pages)
    rows += collect_run(args.excluded.resolve(), "excluded", activity, degree, args.max_pages)
    frame = pd.DataFrame(rows)
    group_sizes = gt.groupby("activity").size().to_dict()
    summary = summarize(frame, group_sizes)
    gt_means = gt.groupby("activity").mean_train_item_degree.mean().to_dict()

    artifact_dir = ROOT / "analysis" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(artifact_dir / "pagewise_activity_degree_users.csv", index=False)
    summary.to_csv(artifact_dir / "pagewise_activity_degree_summary.csv", index=False)
    plot(summary, gt_means, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
