#!/usr/bin/env python3
"""Audit selection-by-position in completed Agent4Rec and SimUSER runs.

The script deliberately reconstructs the displayed rank from each raw page.  In
particular, SimUSER's existing selected_item_events.csv numbers selected items
in selection order and does not contain the displayed rank.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, chi2_contingency, norm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT = ROOT / "Agent4Rec/storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_batch4_from200"
DEFAULT_SIMUSER = ROOT / "PQA/SimUSER/MovieLens/result/vanilla_common_lightgcn_u1000"
DEFAULT_TRAIN = ROOT / "Agent4Rec/datasets/ml-1m/cf_data/train.txt"
DEFAULT_VALID = ROOT / "Agent4Rec/datasets/ml-1m/cf_data/valid.txt"
DEFAULT_OUTPUT = ROOT / "analysis/position_bias"


def train_degrees(path: Path) -> dict[int, int]:
    degrees: dict[int, int] = {}
    for line in path.read_text().splitlines():
        fields = [int(value) for value in line.split()]
        for item_id in fields[1:]:
            degrees[item_id] = degrees.get(item_id, 0) + 1
    return degrees


def agent_rows(run_dir: Path, degrees: dict[int, int]) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "behavior").glob("*.pkl"), key=lambda p: int(p.stem)):
        user_id = int(path.stem)
        with path.open("rb") as stream:
            behavior = pickle.load(stream)
        for page_key, page in behavior.items():
            if not isinstance(page_key, int):
                continue
            exposed = [int(value) for value in page["recommended_id"]]
            original = exposed
            if page.get("position_order") == "random":
                original = [int(value) for value in page["original_recommended_id"]]
                permutation = [int(value) for value in page["position_permutation"]]
                if exposed != [original[index] for index in permutation]:
                    raise ValueError(f"invalid Agent4Rec permutation: user={user_id}, page={page_key}")
            original_position = {item_id: index for index, item_id in enumerate(original, 1)}
            selected = {int(value) for value in page.get("watch_id", [])}
            relevant = {int(value) for value in page.get("ground_truth", [])}
            if len(exposed) != 4 or not selected.issubset(exposed):
                raise ValueError(f"invalid Agent4Rec page: user={user_id}, page={page_key}")
            for position, item_id in enumerate(exposed, 1):
                rows.append({
                    "simulator": "Agent4Rec", "user_id": user_id, "page": int(page_key),
                    "position": position, "item_id": item_id,
                    "original_position": original_position[item_id],
                    "selected": int(item_id in selected), "ground_truth": int(item_id in relevant),
                    "train_degree": degrees.get(item_id, 0),
                })
    return rows


def simuser_rows(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "users").glob("*.json"), key=lambda p: int(p.stem)):
        record = json.loads(path.read_text())
        user_id = int(record["user_id"])
        if record.get("status") != "complete":
            raise ValueError(f"incomplete SimUSER result: {path}")
        for page in record["simulation"]["pages"]:
            exposed = [int(value) for value in page["exposed_item_ids"]]
            original = exposed
            if page.get("position_order") == "random":
                original = [int(value) for value in page["original_exposed_item_ids"]]
                permutation = [int(value) for value in page["position_permutation"]]
                if exposed != [original[index] for index in permutation]:
                    raise ValueError(f"invalid SimUSER permutation: user={user_id}, page={page['page']}")
            original_position = {item_id: index for index, item_id in enumerate(original, 1)}
            selected = {int(value) for value in page["selected_item_ids"]}
            degree_by_item = {int(row["item_id"]): int(row["train_degree"]) for row in page["exposed_items"]}
            if len(exposed) != 4 or not selected.issubset(exposed):
                raise ValueError(f"invalid SimUSER page: user={user_id}, page={page['page']}")
            for position, item_id in enumerate(exposed, 1):
                rows.append({
                    "simulator": "SimUSER", "user_id": user_id, "page": int(page["page"]),
                    "position": position, "item_id": item_id,
                    "original_position": original_position[item_id],
                    "selected": int(item_id in selected), "ground_truth": np.nan,
                    "train_degree": degree_by_item[item_id],
                })
    return rows


def relevance_sets(path: Path) -> dict[int, set[int]]:
    result = {}
    for line in path.read_text().splitlines():
        fields = [int(value) for value in line.split()]
        result[fields[0]] = set(fields[1:])
    return result


def add_ground_truth(frame: pd.DataFrame, valid_path: Path) -> pd.DataFrame:
    valid = relevance_sets(valid_path)
    computed = np.fromiter(
        (int(int(row.item_id) in valid[int(row.user_id)]) for row in frame.itertuples()),
        dtype=int, count=len(frame),
    )
    agent_mask = frame.simulator == "Agent4Rec"
    if not np.array_equal(frame.loc[agent_mask, "ground_truth"].astype(int).to_numpy(), computed[agent_mask]):
        raise ValueError("Agent4Rec ground_truth fields disagree with valid.txt")
    frame["ground_truth"] = computed
    return frame.sort_values(["simulator", "user_id", "page", "position"]).reset_index(drop=True)


def validate_common_exposures(frame: pd.DataFrame) -> None:
    keys = ["user_id", "page", "position"]
    agent = frame[frame.simulator == "Agent4Rec"][keys + ["item_id"]]
    simuser = frame[frame.simulator == "SimUSER"][keys + ["item_id"]]
    overlap = agent.merge(simuser, on=keys, suffixes=("_agent", "_simuser"), validate="one_to_one")
    if overlap.empty or not (overlap.item_id_agent == overlap.item_id_simuser).all():
        raise ValueError("Agent4Rec and SimUSER do not share identical exposures on overlapping pages")


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def cluster_bootstrap_metrics(data: pd.DataFrame, iterations: int, seed: int) -> dict:
    users = np.array(sorted(data.user_id.unique()))
    per_user_pos = data.groupby(["user_id", "position"]).selected.agg(["sum", "count"]).reset_index()
    clicks = per_user_pos.pivot(index="user_id", columns="position", values="sum").reindex(users).to_numpy()
    shows = per_user_pos.pivot(index="user_id", columns="position", values="count").reindex(users).to_numpy()
    rng = np.random.default_rng(seed)
    differences, ratios, mean_ranks = [], [], []
    for _ in range(iterations):
        sample = rng.integers(0, len(users), len(users))
        selected = clicks[sample].sum(axis=0)
        exposed = shows[sample].sum(axis=0)
        rates = selected / exposed
        differences.append(rates[0] - rates[3])
        ratios.append(rates[0] / rates[3])
        mean_ranks.append(float(np.dot(selected, np.arange(1, 5)) / selected.sum()))
    def interval(values):
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]
    return {"position_1_minus_4_ci": interval(differences), "position_1_over_4_ci": interval(ratios),
            "mean_selected_rank_ci": interval(mean_ranks)}


def cochran_armitage(data: pd.DataFrame) -> tuple[float, float]:
    grouped = data.groupby("position").selected.agg(["sum", "count"])
    x = grouped["sum"].to_numpy(float)
    n = grouped["count"].to_numpy(float)
    weights = grouped.index.to_numpy(float)
    total_success = x.sum()
    total = n.sum()
    mean_weight = np.dot(n, weights) / total
    numerator = np.dot(weights, x - n * total_success / total)
    variance = total_success * (total - total_success) / (total * (total - 1)) * np.dot(n, (weights - mean_weight) ** 2)
    z = numerator / math.sqrt(variance)
    return float(z), float(2 * norm.sf(abs(z)))


def design_matrix(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    degree = np.log1p(data.train_degree.to_numpy(float))
    degree = (degree - degree.mean()) / degree.std()
    columns = [np.ones(len(data)), data.position.to_numpy(float) - 1, degree,
               data.ground_truth.to_numpy(float)]
    names = ["intercept", "position_step", "log_train_degree_z", "ground_truth"]
    for page in range(2, 6):
        columns.append((data.page.to_numpy() == page).astype(float))
        names.append(f"page_{page}")
    return np.column_stack(columns), names


def cluster_robust_logit(data: pd.DataFrame) -> dict:
    x, names = design_matrix(data)
    y = data.selected.to_numpy(float)
    beta = np.zeros(x.shape[1])
    converged = False
    for _ in range(100):
        probability = expit(x @ beta)
        information = x.T @ (x * (probability * (1 - probability))[:, None])
        step = np.linalg.pinv(information) @ (x.T @ (y - probability))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    probability = expit(x @ beta)
    bread = np.linalg.pinv(x.T @ (x * (probability * (1 - probability))[:, None]))
    scores = x * (y - probability)[:, None]
    groups = data.user_id.to_numpy()
    unique_groups = np.unique(groups)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for group in unique_groups:
        score = scores[groups == group].sum(axis=0)
        meat += np.outer(score, score)
    correction = len(unique_groups) / (len(unique_groups) - 1) * (len(y) - 1) / (len(y) - x.shape[1])
    covariance = correction * bread @ meat @ bread
    standard_error = np.sqrt(np.diag(covariance))
    index = names.index("position_step")
    estimate, se = beta[index], standard_error[index]
    return {
        "converged": converged,
        "coefficient": float(estimate), "cluster_robust_se": float(se),
        "odds_ratio_per_rank_step": float(math.exp(estimate)),
        "odds_ratio_95_ci": [float(math.exp(estimate - 1.96 * se)), float(math.exp(estimate + 1.96 * se))],
        "p_two_sided": float(2 * norm.sf(abs(estimate / se))),
        "controls": ["log1p(item train degree)", "held-out ground-truth indicator", "page fixed effects"],
    }


def summarize(data: pd.DataFrame, iterations: int, seed: int) -> tuple[dict, pd.DataFrame]:
    result = {}
    rate_rows = []
    for offset, (simulator, group) in enumerate(data.groupby("simulator", sort=True)):
        position = group.groupby("position").selected.agg(["sum", "count"]).reset_index()
        position["rate"] = position["sum"] / position["count"]
        position[["ci_low", "ci_high"]] = position.apply(
            lambda row: pd.Series(wilson(int(row["sum"]), int(row["count"]))), axis=1
        )
        position.insert(0, "simulator", simulator)
        rate_rows.append(position)
        contingency = np.vstack([position["sum"], position["count"] - position["sum"]])
        chi_square, chi_p, _, _ = chi2_contingency(contingency)
        trend_z, trend_p = cochran_armitage(group)
        selected_by_position = position["sum"].to_numpy()
        total_selected = int(selected_by_position.sum())
        mean_rank = float(np.dot(selected_by_position, np.arange(1, 5)) / total_selected)
        strata = {}
        for label, value in [("non_ground_truth", 0), ("ground_truth", 1)]:
            subset = group[group.ground_truth == value].groupby("position").selected.agg(["sum", "count"])
            strata[label] = {str(int(rank)): {"selected": int(row["sum"]), "exposed": int(row["count"]),
                                               "rate": float(row["sum"] / row["count"])}
                              for rank, row in subset.iterrows()}
        rates = position.rate.to_numpy()
        result[simulator] = {
            "users": int(group.user_id.nunique()), "pages": int(len(group) / 4),
            "exposures": int(len(group)), "selections": total_selected,
            "selection_rate_by_position": {str(int(row.position)): float(row.rate) for row in position.itertuples()},
            "position_1_minus_4": float(rates[0] - rates[3]),
            "position_1_over_4": float(rates[0] / rates[3]),
            "mean_selected_rank": mean_rank, "uniform_null_mean_rank": 2.5,
            "pearson_chi_square": float(chi_square), "pearson_chi_square_p": float(chi_p),
            "cochran_armitage_z": trend_z, "cochran_armitage_p_two_sided": trend_p,
            "user_cluster_bootstrap": cluster_bootstrap_metrics(group, iterations, seed + offset),
            "adjusted_logistic": cluster_robust_logit(group), "selection_rates_by_relevance_stratum": strata,
        }
    return result, pd.concat(rate_rows, ignore_index=True)


def experiment_design(agent_run: Path, simuser_run: Path) -> dict:
    agent = json.loads((agent_run / "run_manifest.json").read_text())
    simuser = json.loads((simuser_run / "run_manifest.json").read_text())
    agent_order = agent.get("position_order", "original")
    simuser_order = simuser.get("position_order", "original")
    if agent_order != simuser_order:
        raise ValueError(f"position-order mismatch: Agent4Rec={agent_order}, SimUSER={simuser_order}")
    result = {"position_order": agent_order}
    if agent_order == "random":
        if agent.get("permutation_seed") != simuser.get("permutation_seed"):
            raise ValueError("Agent4Rec and SimUSER permutation seeds differ")
        result.update({
            "permutation_seed": agent["permutation_seed"],
            "position_permutation_version": agent.get("position_permutation_version", 1),
        })
    return result


def markdown(summary: dict, design: dict) -> str:
    lines = ["# Position-bias audit: Agent4Rec and SimUSER", "", "## Main results", "",
             "| Simulator | Users | Pages | Selections | Rate pos. 1 | Rate pos. 2 | Rate pos. 3 | Rate pos. 4 | P1/P4 | Mean selected rank | Adjusted OR / rank step |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["Agent4Rec", "SimUSER"]:
        row = summary[name]
        rates = row["selection_rate_by_position"]
        adjusted = row["adjusted_logistic"]
        lines.append("| %s | %d | %d | %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.3f | %.3f | %.3f [%.3f, %.3f] |" % (
            name, row["users"], row["pages"], row["selections"], *(100 * rates[str(i)] for i in range(1, 5)),
            row["position_1_over_4"], row["mean_selected_rank"], adjusted["odds_ratio_per_rank_step"],
            *adjusted["odds_ratio_95_ci"]))
    lines += ["", "The adjusted odds ratio controls for log item train degree, held-out relevance, and page number; standard errors are clustered by user.", "",
              "## Statistical evidence", ""]
    for name in ["Agent4Rec", "SimUSER"]:
        row = summary[name]
        boot = row["user_cluster_bootstrap"]
        adjusted = row["adjusted_logistic"]
        lines += [f"### {name}", "",
                  f"- Position 1 minus position 4: {row['position_1_minus_4']:.4f}; user-bootstrap 95% CI [{boot['position_1_minus_4_ci'][0]:.4f}, {boot['position_1_minus_4_ci'][1]:.4f}].",
                  f"- Position 1 / position 4: {row['position_1_over_4']:.3f}; user-bootstrap 95% CI [{boot['position_1_over_4_ci'][0]:.3f}, {boot['position_1_over_4_ci'][1]:.3f}].",
                  f"- Mean selected rank: {row['mean_selected_rank']:.3f}; user-bootstrap 95% CI [{boot['mean_selected_rank_ci'][0]:.3f}, {boot['mean_selected_rank_ci'][1]:.3f}] (uniform-position null: 2.5).",
                  f"- Pearson rank × selection chi-square p={row['pearson_chi_square_p']:.3g}; Cochran–Armitage ordered-trend z={row['cochran_armitage_z']:.3f}, p={row['cochran_armitage_p_two_sided']:.3g}.",
                  f"- Adjusted per-rank-step OR={adjusted['odds_ratio_per_rank_step']:.3f}, 95% CI [{adjusted['odds_ratio_95_ci'][0]:.3f}, {adjusted['odds_ratio_95_ci'][1]:.3f}], p={adjusted['p_two_sided']:.3g}.", ""]
    lines += ["## Interpretation boundary", ""]
    if design["position_order"] == "random":
        lines += [f"This run independently randomized item order within every user/page using deterministic seed {design['permutation_seed']}. Position is therefore randomized with respect to the original recommender rank, so rank-selection differences estimate a causal presentation-order effect for this randomized experiment.", ""]
    else:
        lines += ["These completed vanilla runs used a fixed relevance-ranked order, not randomized permutations. The estimates therefore establish selection-position association, not the causal effect of position. A causal position-bias claim requires rerunning identical user/page candidate sets under randomized or balanced permutations.", ""]
    lines += ["## Method references", "",
              "- Collins et al. (2018), *A Study of Position Bias in Digital Library Recommender Systems*: rank-level CTR under shuffled recommendations and a chi-square rank/selection test.",
              "- Joachims, Swaminathan, and Schnabel (2017), *Unbiased Learning-to-Rank with Biased Feedback*: randomized pilot interventions to estimate rank examination propensities, then inverse-propensity weighting.",
              "- Bito, Ren, and He (2025), *Evaluating Position Bias in Large Language Model Recommendations*: shuffled/reversed candidate orders and Kendall-tau positional consistency.",
              "- Peláez-González, Herrera, and Herrera-Poyatos (2026), *Statistical Detection of Positional Bias in LLM-Based Recommender Systems*: repeated random permutations, position-selection distributions, total-variation positional bias, and query-level bootstrap intervals.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-run", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--simuser-run", type=Path, default=DEFAULT_SIMUSER)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--summary-only", action="store_true",
                        help="write JSON/CSV artifacts without a per-run REPORT.md")
    args = parser.parse_args()

    degree = train_degrees(args.train)
    frame = pd.DataFrame(agent_rows(args.agent_run, degree) + simuser_rows(args.simuser_run))
    frame = add_ground_truth(frame, args.valid)
    validate_common_exposures(frame)
    design = experiment_design(args.agent_run, args.simuser_run)
    if design["position_order"] == "random":
        design["assignment_counts"] = {
            simulator: {
                str(int(original)): {str(int(position)): int(count) for position, count in row.items()}
                for original, row in pd.crosstab(group.original_position, group.position).iterrows()
            }
            for simulator, group in frame.groupby("simulator")
        }
    summary, rates = summarize(frame, args.bootstrap, args.seed)
    summary["_experiment_design"] = design
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "exposure_events.csv", index=False)
    rates.to_csv(args.output_dir / "position_rates.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.summary_only:
        (args.output_dir / "REPORT.md").write_text(markdown(summary, design).rstrip() + "\n")
    print(markdown(summary, design))


if __name__ == "__main__":
    main()
