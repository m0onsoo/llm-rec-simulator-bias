"""Measure whether Agent4Rec misses exposed relevant tail items."""

import argparse
from collections import Counter
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
GROUP_NAMES = {0: "All", 1: "Infrequent", 2: "Occasional", 3: "Frequent"}
EVENT_FIELDS = {"align": "align_id", "watch": "watch_id", "like": "like_id"}
MODE_LABELS = {"included": "Activity included", "excluded": "Activity excluded"}
COLORS = {"included": "#2864DC", "excluded": "#DC503C"}


def load_behavior(run_dir, user_id):
    path = run_dir / "behavior" / f"{user_id}.pkl"
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


def load_user_ratings(raw_dir):
    with (raw_dir / "user_id_map.pkl").open("rb") as handle:
        user_map = pickle.load(handle)
    with (raw_dir / "movie_id_map.pkl").open("rb") as handle:
        item_map = pickle.load(handle)
    ratings = pd.read_csv(
        raw_dir / "ratings.dat", sep="::", engine="python", header=None,
        names=["raw_user_id", "raw_item_id", "user_rating", "timestamp"],
    )
    ratings = ratings[
        ratings.raw_user_id.isin(user_map) & ratings.raw_item_id.isin(item_map)
    ].copy()
    ratings["user_id"] = ratings.raw_user_id.map(user_map)
    ratings["item_id"] = ratings.raw_item_id.map(item_map)
    return ratings.set_index(["user_id", "item_id"]).user_rating.to_dict()


def head_items(degree, catalog_size, fraction):
    count = int(np.ceil(catalog_size * fraction))
    ordered = sorted(range(catalog_size), key=lambda item: (-degree[item], item))
    return set(ordered[:count]), count


def collect(included_dir, excluded_dir, activity, degree, movie_ratings, user_ratings, head):
    rows = []
    common_impressions = 0
    for user_id, group in activity.items():
        inc_pages = load_behavior(included_dir, user_id)
        exc_pages = load_behavior(excluded_dir, user_id)
        last_page = min(max(inc_pages), max(exc_pages))
        for page in range(1, last_page + 1):
            inc = inc_pages[page]
            exc = exc_pages[page]
            inc_ids = [int(value) for value in inc.get("recommended_id", [])]
            exc_ids = [int(value) for value in exc.get("recommended_id", [])]
            if inc_ids != exc_ids:
                raise ValueError(f"Candidate mismatch for user={user_id}, page={page}")
            inc_gt = set(map(int, inc.get("ground_truth", [])))
            exc_gt = set(map(int, exc.get("ground_truth", [])))
            if inc_gt != exc_gt:
                raise ValueError(f"Ground-truth mismatch for user={user_id}, page={page}")
            common_impressions += len(inc_ids)
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
            for position, item_id in enumerate(inc_ids, 1):
                if item_id not in inc_gt:
                    continue
                key = (user_id, item_id)
                if key not in user_ratings:
                    raise KeyError(f"Missing held-out user rating for {key}")
                base = {
                    "user_id": user_id,
                    "activity": group,
                    "activity_name": GROUP_NAMES[group],
                    "page": page,
                    "position": position,
                    "item_id": item_id,
                    "degree": degree[item_id],
                    "popularity_group": "head" if item_id in head else "tail",
                    "is_head": int(item_id in head),
                    "history_rating": movie_ratings[item_id],
                    "user_rating": user_ratings[key],
                }
                for event in EVENT_FIELDS:
                    rows.append({
                        **base,
                        "event": event,
                        "included_selected": int(item_id in selected["included"][event]),
                        "excluded_selected": int(item_id in selected["excluded"][event]),
                    })
    return pd.DataFrame(rows), common_impressions


def cluster_bootstrap(values, selected_field, replicates, seed):
    aggregates = values.groupby(["user_id", "is_head"])[selected_field].agg(
        selected="sum", exposed="size"
    ).reset_index()
    users = np.sort(values.user_id.unique())
    rng = np.random.default_rng(seed)
    user_index = {user: index for index, user in enumerate(users)}
    counts = np.zeros((len(users), 4), dtype=float)
    for record in aggregates.itertuples(index=False):
        row = user_index[record.user_id]
        offset = 0 if record.is_head == 1 else 2
        counts[row, offset] = record.selected
        counts[row, offset + 1] = record.exposed

    totals = counts.sum(axis=0)
    point_head = totals[0] / totals[1]
    point_tail = totals[2] / totals[3]
    point = (point_head, point_tail, point_head - point_tail)
    sampled_indices = rng.integers(0, len(users), size=(replicates, len(users)))
    sampled_totals = counts[sampled_indices].sum(axis=1)
    valid = (sampled_totals[:, 1] > 0) & (sampled_totals[:, 3] > 0)
    sampled_totals = sampled_totals[valid]
    head_tpr = sampled_totals[:, 0] / sampled_totals[:, 1]
    tail_tpr = sampled_totals[:, 2] / sampled_totals[:, 3]
    draws = np.column_stack((head_tpr, tail_tpr, head_tpr - tail_tpr))
    lower = np.quantile(draws, 0.025, axis=0)
    upper = np.quantile(draws, 0.975, axis=0)
    two_sided_p = 2 * min(np.mean(draws[:, 2] <= 0), np.mean(draws[:, 2] >= 0))
    return point, lower, upper, min(two_sided_p, 1.0)


def fit_cluster_logit(frame, outcome_field, interaction=False):
    values = frame.copy()
    cell_columns = ["included", "is_head"] if interaction else ["is_head"]
    if (values.groupby(cell_columns)[outcome_field].nunique() < 2).any():
        return {
            "term": "included_head" if interaction else "head",
            "log_odds": np.nan,
            "odds_ratio": np.nan,
            "cluster_se": np.nan,
            "ci95_log_odds_low": np.nan,
            "ci95_log_odds_high": np.nan,
            "p_value": np.nan,
            "model_status": "complete_separation",
        }
    history = values.history_rating.to_numpy(dtype=float)
    user_rating = values.user_rating.to_numpy(dtype=float)
    values["history_z"] = (history - history.mean()) / history.std(ddof=0)
    values["user_rating_z"] = (
        (user_rating - user_rating.mean()) / user_rating.std(ddof=0)
        if user_rating.std(ddof=0) else 0.0
    )
    columns = [np.ones(len(values)), values.is_head.to_numpy(dtype=float)]
    names = ["intercept", "head"]
    if interaction:
        columns.insert(1, values.included.to_numpy(dtype=float))
        names.insert(1, "included")
        columns.insert(3, (values.included * values.is_head).to_numpy(dtype=float))
        names.insert(3, "included_head")
    columns.extend([
        values.history_z.to_numpy(dtype=float),
        values.user_rating_z.to_numpy(dtype=float),
    ])
    names.extend(["history_rating_z", "user_rating_z"])
    for position in sorted(values.position.unique())[1:]:
        columns.append((values.position == position).to_numpy(dtype=float))
        names.append(f"position_{position}")
    for page in sorted(values.page.unique())[1:]:
        columns.append((values.page == page).to_numpy(dtype=float))
        names.append(f"page_{page}")
    design = np.column_stack(columns)
    outcome = values[outcome_field].to_numpy(dtype=float)
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
    residual = outcome - probability
    meat = np.zeros_like(bread)
    user_values = values.user_id.to_numpy()
    for user_id in values.user_id.unique():
        mask = user_values == user_id
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)
    clusters = values.user_id.nunique()
    correction = (clusters / (clusters - 1)) * (
        (len(values) - 1) / (len(values) - design.shape[1])
    )
    covariance = correction * bread @ meat @ bread
    term = "included_head" if interaction else "head"
    index = names.index(term)
    estimate = beta[index]
    standard_error = np.sqrt(max(covariance[index, index], 0))
    z_value = estimate / standard_error if standard_error else np.nan
    p_value = 2 * stats.norm.sf(abs(z_value)) if standard_error else np.nan
    return {
        "term": term,
        "log_odds": estimate,
        "odds_ratio": np.exp(estimate),
        "cluster_se": standard_error,
        "ci95_log_odds_low": estimate - 1.96 * standard_error,
        "ci95_log_odds_high": estimate + 1.96 * standard_error,
        "p_value": p_value,
        "model_status": "ok",
    }


def threshold_sensitivity(frame, degree, catalog_size, fractions=(0.10, 0.20, 0.30)):
    rows = []
    for fraction in fractions:
        head, head_count = head_items(degree, catalog_size, fraction)
        values = frame.copy()
        values["threshold_head"] = values.item_id.isin(head)
        for relevance, relevant in (
            ("interaction", values),
            ("liked_rating_ge_4", values[values.user_rating >= 4]),
        ):
            grouped = list(relevant.groupby(["activity", "event"]))
            grouped += [((0, event), part) for event, part in relevant.groupby("event")]
            for (activity, event), group in grouped:
                for mode, field in (
                    ("included", "included_selected"),
                    ("excluded", "excluded_selected"),
                ):
                    head_values = group[group.threshold_head]
                    tail_values = group[~group.threshold_head]
                    rows.append({
                        "head_fraction": fraction,
                        "head_item_count": head_count,
                        "minimum_head_degree": min(degree[item] for item in head),
                        "relevance": relevance,
                        "activity": activity,
                        "activity_name": GROUP_NAMES[activity],
                        "event": event,
                        "mode": mode,
                        "head_exposed_positives": len(head_values),
                        "tail_exposed_positives": len(tail_values),
                        "head_tpr": head_values[field].mean(),
                        "tail_tpr": tail_values[field].mean(),
                        "head_minus_tail_tpr": (
                            head_values[field].mean() - tail_values[field].mean()
                        ),
                    })
    return pd.DataFrame(rows)


def summarize(frame, replicates, seed):
    rows = []
    models = []
    for relevance, relevant in (
        ("interaction", frame),
        ("liked_rating_ge_4", frame[frame.user_rating >= 4]),
    ):
        grouped = list(relevant.groupby(["activity", "event"]))
        grouped += [((0, event), part) for event, part in relevant.groupby("event")]
        for (activity, event), group in grouped:
            for mode, selected_field in (
                ("included", "included_selected"),
                ("excluded", "excluded_selected"),
            ):
                point, low, high, p_value = cluster_bootstrap(
                    group, selected_field, replicates, seed + activity
                )
                head_values = group[group.is_head == 1]
                tail_values = group[group.is_head == 0]
                rows.append({
                    "relevance": relevance,
                    "activity": activity,
                    "activity_name": GROUP_NAMES[activity],
                    "event": event,
                    "mode": mode,
                    "users": group.user_id.nunique(),
                    "head_exposed_positives": len(head_values),
                    "tail_exposed_positives": len(tail_values),
                    "head_selected_positives": int(head_values[selected_field].sum()),
                    "tail_selected_positives": int(tail_values[selected_field].sum()),
                    "head_tpr": point[0],
                    "head_tpr_ci95_low": low[0],
                    "head_tpr_ci95_high": high[0],
                    "tail_tpr": point[1],
                    "tail_tpr_ci95_low": low[1],
                    "tail_tpr_ci95_high": high[1],
                    "head_minus_tail_tpr": point[2],
                    "gap_ci95_low": low[2],
                    "gap_ci95_high": high[2],
                    "bootstrap_p_value": p_value,
                })
                model = fit_cluster_logit(group, selected_field)
                models.append({
                    "relevance": relevance,
                    "activity": activity,
                    "activity_name": GROUP_NAMES[activity],
                    "event": event,
                    "mode": mode,
                    "users": group.user_id.nunique(),
                    "positive_impressions": len(group),
                    **model,
                })

            stacked = pd.concat([
                group.assign(
                    included=1.0,
                    selected=group.included_selected.to_numpy(dtype=float),
                ),
                group.assign(
                    included=0.0,
                    selected=group.excluded_selected.to_numpy(dtype=float),
                ),
            ], ignore_index=True)
            interaction_model = fit_cluster_logit(stacked, "selected", interaction=True)
            models.append({
                "relevance": relevance,
                "activity": activity,
                "activity_name": GROUP_NAMES[activity],
                "event": event,
                "mode": "included_x_head",
                "users": group.user_id.nunique(),
                "positive_impressions": len(group),
                **interaction_model,
            })
    return pd.DataFrame(rows), pd.DataFrame(models)


def plot(summary, output):
    values = summary[summary.relevance == "interaction"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharey="row")
    for row, event in enumerate(("align", "watch")):
        for column, activity in enumerate((1, 2, 3)):
            ax = axes[row, column]
            part = values[
                (values.event == event) & (values.activity == activity)
            ].set_index("mode")
            for offset, mode in ((-0.05, "included"), (0.05, "excluded")):
                record = part.loc[mode]
                x = np.array([0, 1], dtype=float) + offset
                y = np.array([record.tail_tpr, record.head_tpr])
                low = np.array([record.tail_tpr_ci95_low, record.head_tpr_ci95_low])
                high = np.array([record.tail_tpr_ci95_high, record.head_tpr_ci95_high])
                ax.errorbar(
                    x, y, yerr=np.vstack((y - low, high - y)), marker="o",
                    linewidth=2, capsize=3, color=COLORS[mode], label=MODE_LABELS[mode],
                )
            ax.set_xticks([0, 1], ["Tail", "Head"])
            ax.set_ylim(-0.03, 1.03)
            ax.grid(axis="y", alpha=0.2)
            if row == 0:
                ax.set_title(GROUP_NAMES[activity], fontweight="bold")
            if column == 0:
                ax.set_ylabel(f"{event.upper()} recall on exposed positives")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Popularity-conditioned simulator recall on identical common-prefix exposures",
        fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.01,
        "Positive = exposed validation interaction; head = top 20% of catalog items by train degree. "
        "Intervals are user-cluster bootstrap 95% intervals.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--included", required=True, type=Path)
    parser.add_argument("--excluded", required=True, type=Path)
    parser.add_argument("--head-fraction", type=float, default=0.20)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "analysis" / "artifacts"
    )
    parser.add_argument(
        "--figure", type=Path,
        default=ROOT / "analysis" / "figures" / "head_tail_fidelity.png",
    )
    args = parser.parse_args()
    if not 0 < args.head_fraction < 1:
        raise ValueError("head-fraction must be between zero and one")

    gt = pd.read_csv(ROOT / "analysis" / "artifacts" / "activity_degree_gt_users.csv")
    activity = dict(zip(gt.user_id.astype(int), gt.activity.astype(int)))
    degree = load_train_degree(ROOT / "datasets" / "ml-1m" / "cf_data" / "train.txt")
    movies = pd.read_csv(ROOT / "datasets" / "ml-1m" / "simulation" / "movie_detail.csv")
    movie_ratings = movies.set_index("movie_id").rating.to_dict()
    user_ratings = load_user_ratings(ROOT / "datasets" / "ml-1m" / "raw_data")
    head, head_count = head_items(degree, len(movies), args.head_fraction)

    frame, common_impressions = collect(
        args.included.resolve(), args.excluded.resolve(), activity, degree,
        movie_ratings, user_ratings, head,
    )
    summary, models = summarize(frame, args.bootstrap_replicates, args.seed)
    sensitivity = threshold_sensitivity(frame, degree, len(movies))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "head_tail_fidelity_positive_pairs.csv", index=False)
    summary.to_csv(args.output_dir / "head_tail_fidelity_summary.csv", index=False)
    models.to_csv(args.output_dir / "head_tail_fidelity_models.csv", index=False)
    sensitivity.to_csv(
        args.output_dir / "head_tail_fidelity_threshold_sensitivity.csv", index=False
    )
    plot(summary, args.figure)

    threshold = min(degree[item] for item in head)
    print(
        f"Catalog items={len(movies)}, head items={head_count}, "
        f"minimum head degree={threshold}, common impressions={common_impressions}"
    )
    print("\nPrimary interaction-positive ALIGN/WATCH results:")
    primary = summary[
        (summary.relevance == "interaction") & summary.event.isin(["align", "watch"])
    ]
    print(primary.to_string(index=False))
    print(f"\nFigure: {args.figure}")


if __name__ == "__main__":
    main()
