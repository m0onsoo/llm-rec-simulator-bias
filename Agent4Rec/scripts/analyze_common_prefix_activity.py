"""Compare activity-included/excluded behavior on identical common-prefix exposures."""

import argparse
from collections import Counter
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
GROUP_NAMES = {1: "Infrequent", 2: "Occasional", 3: "Frequent"}
EVENT_FIELDS = {"align": "align_id", "watch": "watch_id", "like": "like_id"}
COLORS = {"align": "#6A5ACD", "watch": "#168AAD", "like": "#D95F59"}


def benjamini_hochberg(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def load_behavior(run_dir, user_id):
    path = run_dir / "behavior" / f"{user_id}.pkl"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    return {int(value.get("page", key)): value for key, value in raw.items()}


def load_train_degree(path):
    degree = Counter()
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            values = [int(value) for value in raw.split()]
            degree.update(set(values[1:]))
    return degree


def paired_test(included, excluded):
    pairs = pd.concat(
        [included.rename("included"), excluded.rename("excluded")], axis=1
    ).dropna()
    difference = pairs.included - pairs.excluded
    n = len(difference)
    if n < 2:
        return n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    sem = stats.sem(difference)
    low, high = stats.t.interval(
        0.95, n - 1, loc=difference.mean(), scale=sem
    ) if sem > 0 else (difference.mean(), difference.mean())
    p_value = stats.ttest_rel(pairs.included, pairs.excluded).pvalue
    return (
        n,
        pairs.included.mean(),
        pairs.excluded.mean(),
        difference.mean(),
        low,
        high,
        p_value,
    )


def cluster_ols_degree_effect(values):
    """Regress paired selection difference on log degree with user-clustered SE."""
    frame = values.copy()
    log_degree = np.log1p(frame.degree.to_numpy(dtype=float))
    degree_std = log_degree.std(ddof=0)
    if degree_std == 0:
        raise ValueError("Item degree has zero variance")
    frame["degree_z"] = (log_degree - log_degree.mean()) / degree_std

    columns = [np.ones(len(frame)), frame.degree_z.to_numpy()]
    names = ["intercept", "degree_z"]
    for position in sorted(frame.position.unique())[1:]:
        columns.append((frame.position == position).to_numpy(dtype=float))
        names.append(f"position_{position}")
    for page in sorted(frame.page.unique())[1:]:
        columns.append((frame.page == page).to_numpy(dtype=float))
        names.append(f"page_{page}")
    design = np.column_stack(columns)
    outcome = (frame.included_selected - frame.excluded_selected).to_numpy(dtype=float)
    bread = np.linalg.pinv(design.T @ design)
    beta = bread @ design.T @ outcome
    residual = outcome - design @ beta

    meat = np.zeros((design.shape[1], design.shape[1]))
    for user_id in frame.user_id.unique():
        mask = frame.user_id.to_numpy() == user_id
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)
    clusters = frame.user_id.nunique()
    observations, parameters = design.shape
    correction = (clusters / (clusters - 1)) * (
        (observations - 1) / (observations - parameters)
    )
    covariance = correction * bread @ meat @ bread
    index = names.index("degree_z")
    estimate = beta[index]
    standard_error = np.sqrt(max(covariance[index, index], 0))
    z_value = estimate / standard_error if standard_error else np.nan
    p_value = 2 * stats.norm.sf(abs(z_value)) if standard_error else np.nan
    return {
        "degree_scale": "1 SD log1p(train degree)",
        "degree_log_mean": log_degree.mean(),
        "degree_log_std": degree_std,
        "slope_difference": estimate,
        "cluster_se": standard_error,
        "ci95_low": estimate - 1.96 * standard_error,
        "ci95_high": estimate + 1.96 * standard_error,
        "p_value": p_value,
    }


def cluster_logit_degree_interaction(values):
    """Fit selected ~ condition * log-degree with user-clustered covariance."""
    base = values.copy()
    log_degree = np.log1p(base.degree.to_numpy(dtype=float))
    degree_std = log_degree.std(ddof=0)
    degree_z = (log_degree - log_degree.mean()) / degree_std
    observations = len(base)
    frame = pd.concat(
        [
            base.assign(
                included=1.0,
                selected=base.included_selected.to_numpy(dtype=float),
                degree_z=degree_z,
            ),
            base.assign(
                included=0.0,
                selected=base.excluded_selected.to_numpy(dtype=float),
                degree_z=degree_z,
            ),
        ],
        ignore_index=True,
    )
    columns = [
        np.ones(2 * observations),
        frame.included.to_numpy(),
        frame.degree_z.to_numpy(),
        (frame.included * frame.degree_z).to_numpy(),
    ]
    names = ["intercept", "included", "degree_z", "included_degree_z"]
    for position in sorted(frame.position.unique())[1:]:
        columns.append((frame.position == position).to_numpy(dtype=float))
        names.append(f"position_{position}")
    for page in sorted(frame.page.unique())[1:]:
        columns.append((frame.page == page).to_numpy(dtype=float))
        names.append(f"page_{page}")
    design = np.column_stack(columns)
    outcome = frame.selected.to_numpy(dtype=float)
    beta = np.zeros(design.shape[1])
    for _ in range(100):
        probability = stats.logistic.cdf(np.clip(design @ beta, -30, 30))
        weight = np.maximum(probability * (1 - probability), 1e-9)
        information = design.T @ (weight[:, None] * design)
        step = np.linalg.pinv(information) @ design.T @ (outcome - probability)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    probability = stats.logistic.cdf(np.clip(design @ beta, -30, 30))
    weight = np.maximum(probability * (1 - probability), 1e-9)
    bread = np.linalg.pinv(design.T @ (weight[:, None] * design))
    meat = np.zeros_like(bread)
    user_values = frame.user_id.to_numpy()
    residual = outcome - probability
    for user_id in frame.user_id.unique():
        mask = user_values == user_id
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)
    clusters = frame.user_id.nunique()
    correction = (clusters / (clusters - 1)) * (
        (len(frame) - 1) / (len(frame) - design.shape[1])
    )
    covariance = correction * bread @ meat @ bread
    index = names.index("included_degree_z")
    estimate = beta[index]
    standard_error = np.sqrt(max(covariance[index, index], 0))
    z_value = estimate / standard_error if standard_error else np.nan
    p_value = 2 * stats.norm.sf(abs(z_value)) if standard_error else np.nan
    return {
        "logit_interaction": estimate,
        "logit_interaction_odds_ratio": np.exp(estimate),
        "logit_cluster_se": standard_error,
        "logit_ci95_low": estimate - 1.96 * standard_error,
        "logit_ci95_high": estimate + 1.96 * standard_error,
        "logit_p_value": p_value,
    }


def collect(included_dir, excluded_dir, activity, degree):
    item_rows = []
    user_rows = []
    reach_rows = []
    candidate_mismatches = []

    for user_id, group in activity.items():
        included_pages = load_behavior(included_dir, user_id)
        excluded_pages = load_behavior(excluded_dir, user_id)
        common_pages = min(max(included_pages), max(excluded_pages))
        expected = set(range(1, common_pages + 1))
        if not expected.issubset(included_pages) or not expected.issubset(excluded_pages):
            raise ValueError(f"Non-contiguous behavior pages for user {user_id}")
        reach_rows.append({
            "user_id": user_id,
            "activity": group,
            "activity_name": GROUP_NAMES[group],
            "included_pages": max(included_pages),
            "excluded_pages": max(excluded_pages),
            "common_pages": common_pages,
        })

        per_user = {
            mode: {
                event: {"selected": [], "exposed": [], "centered_mass": 0.0}
                for event in EVENT_FIELDS
            }
            for mode in ("included", "excluded")
        }
        for page in range(1, common_pages + 1):
            inc = included_pages[page]
            exc = excluded_pages[page]
            inc_ids = [int(item_id) for item_id in inc.get("recommended_id", [])]
            exc_ids = [int(item_id) for item_id in exc.get("recommended_id", [])]
            if inc_ids != exc_ids:
                candidate_mismatches.append((user_id, page, inc_ids, exc_ids))
                continue
            page_degrees = np.array([degree[item_id] for item_id in inc_ids], dtype=float)
            page_center = page_degrees.mean()
            selected = {
                "included": {
                    event: set(map(int, inc.get(field, [])))
                    for event, field in EVENT_FIELDS.items()
                },
                "excluded": {
                    event: set(map(int, exc.get(field, [])))
                    for event, field in EVENT_FIELDS.items()
                },
            }
            for position, (item_id, item_degree) in enumerate(
                zip(inc_ids, page_degrees), 1
            ):
                base = {
                    "user_id": user_id,
                    "activity": group,
                    "activity_name": GROUP_NAMES[group],
                    "common_pages": common_pages,
                    "page": page,
                    "position": position,
                    "item_id": item_id,
                    "degree": item_degree,
                    "page_mean_degree": page_center,
                    "centered_degree": item_degree - page_center,
                }
                for event in EVENT_FIELDS:
                    inc_selected = int(item_id in selected["included"][event])
                    exc_selected = int(item_id in selected["excluded"][event])
                    item_rows.append({
                        **base,
                        "event": event,
                        "included_selected": inc_selected,
                        "excluded_selected": exc_selected,
                    })
                    for mode, is_selected in (
                        ("included", inc_selected), ("excluded", exc_selected)
                    ):
                        values = per_user[mode][event]
                        values["exposed"].append(item_degree)
                        if is_selected:
                            values["selected"].append(item_degree)
                            values["centered_mass"] += item_degree - page_center

        for mode in ("included", "excluded"):
            for event in EVENT_FIELDS:
                values = per_user[mode][event]
                exposed = np.asarray(values["exposed"], dtype=float)
                selected = np.asarray(values["selected"], dtype=float)
                impression_mean = exposed.mean()
                selected_mean = selected.mean() if len(selected) else np.nan
                user_rows.append({
                    "mode": mode,
                    "user_id": user_id,
                    "activity": group,
                    "activity_name": GROUP_NAMES[group],
                    "event": event,
                    "common_pages": common_pages,
                    "impressions": len(exposed),
                    "selected_count": len(selected),
                    "selection_rate": len(selected) / len(exposed),
                    "impression_mean_degree": impression_mean,
                    "selected_mean_degree": selected_mean,
                    "selection_degree_uplift": selected_mean - impression_mean,
                    "centered_degree_mass_per_impression": (
                        values["centered_mass"] / len(exposed)
                    ),
                })

    if candidate_mismatches:
        first = candidate_mismatches[0]
        raise ValueError(
            f"Found {len(candidate_mismatches)} common-page candidate mismatches; "
            f"first={first}"
        )
    return pd.DataFrame(item_rows), pd.DataFrame(user_rows), pd.DataFrame(reach_rows)


def summarize_pairs(user_frame):
    metrics = [
        "selected_count",
        "selection_rate",
        "selected_mean_degree",
        "selection_degree_uplift",
        "centered_degree_mass_per_impression",
    ]
    rows = []
    for (activity, event), values in user_frame.groupby(["activity", "event"]):
        indexed = {
            mode: part.set_index("user_id")
            for mode, part in values.groupby("mode")
        }
        for metric in metrics:
            result = paired_test(
                indexed["included"][metric], indexed["excluded"][metric]
            )
            rows.append({
                "activity": activity,
                "activity_name": GROUP_NAMES[activity],
                "event": event,
                "metric": metric,
                "paired_users": result[0],
                "included_mean": result[1],
                "excluded_mean": result[2],
                "paired_mean_difference": result[3],
                "ci95_low": result[4],
                "ci95_high": result[5],
                "paired_t_p_value": result[6],
            })
    summary = pd.DataFrame(rows)
    summary["bh_q_within_metric"] = summary.groupby("metric")[
        "paired_t_p_value"
    ].transform(benjamini_hochberg)
    return summary


def summarize_item_effects(item_frame):
    rows = []
    for (activity, event), values in item_frame.groupby(["activity", "event"]):
        result = cluster_ols_degree_effect(values)
        logit_result = cluster_logit_degree_interaction(values)
        rows.append({
            "activity": activity,
            "activity_name": GROUP_NAMES[activity],
            "event": event,
            "users": values.user_id.nunique(),
            "paired_impressions": len(values),
            "included_selection_rate": values.included_selected.mean(),
            "excluded_selection_rate": values.excluded_selected.mean(),
            **result,
            **logit_result,
        })
    summary = pd.DataFrame(rows)
    summary["logit_bh_q_value"] = benjamini_hochberg(summary.logit_p_value)
    summary["linear_bh_q_value"] = benjamini_hochberg(summary.p_value)
    return summary


def summarize_reach(reach_frame):
    rows = []
    for activity, values in reach_frame.groupby("activity"):
        counts = values.common_pages.value_counts().to_dict()
        rows.append({
            "activity": activity,
            "activity_name": GROUP_NAMES[activity],
            "users": len(values),
            "mean_included_pages": values.included_pages.mean(),
            "mean_excluded_pages": values.excluded_pages.mean(),
            "mean_common_pages": values.common_pages.mean(),
            "median_common_pages": values.common_pages.median(),
            **{f"common_prefix_{page}_users": counts.get(page, 0) for page in range(1, 6)},
        })
    return pd.DataFrame(rows)


def plot_effects(summary, output):
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.5), sharex=True)
    configurations = [
        ("selection_rate", "Selection-rate difference", "Included − excluded"),
        (
            "selected_mean_degree",
            "Selected mean-degree difference",
            "Included − excluded train degree",
        ),
    ]
    centers = np.arange(3)
    offsets = {"align": -0.22, "watch": 0.0, "like": 0.22}
    for ax, (metric, title, ylabel) in zip(axes, configurations):
        values = summary[summary.metric == metric]
        for event in EVENT_FIELDS:
            part = values[values.event == event].set_index("activity").loc[[1, 2, 3]]
            x = centers + offsets[event]
            y = part.paired_mean_difference.to_numpy(dtype=float)
            low = part.ci95_low.to_numpy(dtype=float)
            high = part.ci95_high.to_numpy(dtype=float)
            ax.errorbar(
                x, y, yerr=np.vstack((y - low, high - y)), marker="o",
                linestyle="none", capsize=4, color=COLORS[event], label=event,
            )
        ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(centers, [GROUP_NAMES[group] for group in (1, 2, 3)])
    axes[-1].set_xlabel("Original activity group")
    axes[0].legend(ncol=3, frameon=False, loc="best")
    fig.suptitle(
        "Paired activity-prompt effects on identical common-prefix impressions",
        fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Bars are user-clustered paired 95% t intervals. Degree comparisons require "
        "at least one selected item in both conditions.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--included", required=True, type=Path)
    parser.add_argument("--excluded", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "analysis" / "artifacts",
    )
    parser.add_argument(
        "--figure", type=Path,
        default=ROOT / "analysis" / "figures" / "common_prefix_activity_effects.png",
    )
    args = parser.parse_args()

    gt = pd.read_csv(ROOT / "analysis" / "artifacts" / "activity_degree_gt_users.csv")
    activity = dict(zip(gt.user_id.astype(int), gt.activity.astype(int)))
    degree = load_train_degree(ROOT / "datasets" / "ml-1m" / "cf_data" / "train.txt")
    item_frame, user_frame, reach_frame = collect(
        args.included.resolve(), args.excluded.resolve(), activity, degree
    )
    paired_summary = summarize_pairs(user_frame)
    item_effects = summarize_item_effects(item_frame)
    reach_summary = summarize_reach(reach_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    item_frame.to_csv(args.output_dir / "common_prefix_item_pairs.csv", index=False)
    user_frame.to_csv(args.output_dir / "common_prefix_user_metrics.csv", index=False)
    paired_summary.to_csv(args.output_dir / "common_prefix_paired_summary.csv", index=False)
    item_effects.to_csv(args.output_dir / "common_prefix_item_effects.csv", index=False)
    reach_frame.to_csv(args.output_dir / "common_prefix_reach_users.csv", index=False)
    reach_summary.to_csv(args.output_dir / "common_prefix_reach_summary.csv", index=False)
    plot_effects(paired_summary, args.figure)

    print("Candidate mismatches: 0")
    print(reach_summary.to_string(index=False))
    print("\nItem-level paired degree-sensitivity effects:")
    print(item_effects.to_string(index=False))
    print(f"\nFigure: {args.figure}")


if __name__ == "__main__":
    main()
