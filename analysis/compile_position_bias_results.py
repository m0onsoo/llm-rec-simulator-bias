#!/usr/bin/env python3
"""Compile vanilla and randomized Agent4Rec/SimUSER summaries into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "analysis/position_bias/summary.json"
DEFAULT_OUTPUT = ROOT / "analysis/POSITION_BIAS_EXPERIMENT_RESULTS.md"
SIMULATORS = ("Agent4Rec", "SimUSER")


def load_summary(path: Path, expected_order: str) -> dict:
    summary = json.loads(path.read_text())
    design = summary.get("_experiment_design", {})
    order = design.get("position_order")
    if order != expected_order:
        raise ValueError(f"{path}: expected position_order={expected_order}, found {order!r}")
    for simulator in SIMULATORS:
        if simulator not in summary:
            raise ValueError(f"{path}: missing {simulator}")
        row = summary[simulator]
        if row.get("users") != 1000:
            raise ValueError(f"{path}: {simulator} has {row.get('users')} users; expected 1000")
        if set(row.get("selection_rate_by_position", {})) != {"1", "2", "3", "4"}:
            raise ValueError(f"{path}: {simulator} does not contain all four positions")
    return summary


def pvalue(value: float) -> str:
    return f"{value:.3g}"


def ci(value: float, interval: list[float], digits: int = 3) -> str:
    return f"{value:.{digits}f} [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def run_label(summary: dict) -> str:
    design = summary["_experiment_design"]
    if design["position_order"] == "original":
        return "Vanilla, original order"
    return f"Random order, seed {design['permutation_seed']}"


def causal_interpretation(row: dict) -> str:
    bootstrap = row["user_cluster_bootstrap"]
    difference_ci = bootstrap["position_1_minus_4_ci"]
    ratio_ci = bootstrap["position_1_over_4_ci"]
    odds_ci = row["adjusted_logistic"]["odds_ratio_95_ci"]
    rates = [row["selection_rate_by_position"][str(position)] for position in range(1, 5)]
    monotonic = all(left >= right for left, right in zip(rates, rates[1:]))
    if difference_ci[0] > 0 and ratio_ci[0] > 1 and odds_ci[1] < 1:
        shape = "monotonic" if monotonic else "non-monotonic"
        return f"Causal front-position effect detected ({shape} position curve)."
    if difference_ci[1] < 0 and ratio_ci[1] < 1 and odds_ci[0] > 1:
        return "Causal reverse-position effect detected."
    return "No statistically clear causal front-position effect under the prespecified criteria."


def association_interpretation(row: dict) -> str:
    rates = [row["selection_rate_by_position"][str(position)] for position in range(1, 5)]
    monotonic = all(left >= right for left, right in zip(rates, rates[1:]))
    return (
        "Strong monotonic association; fixed relevance order prevents causal attribution."
        if monotonic
        else "Front-position association is present but non-monotonic; fixed order prevents causal attribution."
    )


def render(baseline: dict, randomized: list[dict], baseline_path: Path,
           random_paths: list[Path]) -> str:
    runs = [baseline, *randomized]
    lines = [
        "# Agent4Rec and SimUSER randomized position-bias results",
        "",
        "This is the single final results document for the position-permutation experiment. "
        "It is generated from analyzer outputs; do not transcribe values manually.",
        "",
        "## Inputs and completion status",
        "",
        f"- Vanilla summary: `{baseline_path}`",
    ]
    for path, summary in zip(random_paths, randomized):
        seed = summary["_experiment_design"]["permutation_seed"]
        lines.append(f"- Randomized summary, seed {seed}: `{path}`")
    lines += [
        "- Each included summary contains exactly 1,000 Agent4Rec users and 1,000 SimUSER users.",
        "- Simulator-specific audits must be passed before this report is treated as final.",
        "",
        "## Main conclusion",
        "",
    ]
    for summary in randomized:
        seed = summary["_experiment_design"]["permutation_seed"]
        for simulator in SIMULATORS:
            lines.append(f"- **{simulator}, seed {seed}:** {causal_interpretation(summary[simulator])}")
    lines += [
        "",
        "The vanilla rows are descriptive associations. Only randomized rows support causal "
        "claims about displayed position.",
        "",
        "## Selection rate by displayed position",
        "",
        "| Simulator | Run | Users | Pages | Position 1 | Position 2 | Position 3 | Position 4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in runs:
        for simulator in SIMULATORS:
            row = summary[simulator]
            rates = row["selection_rate_by_position"]
            lines.append(
                f"| {simulator} | {run_label(summary)} | {row['users']:,} | {row['pages']:,} | "
                + " | ".join(f"{100 * rates[str(position)]:.2f}%" for position in range(1, 5))
                + " |"
            )
    lines += [
        "",
        "## Effect summaries",
        "",
        "| Simulator | Run | P1−P4 (95% CI) | P1/P4 (95% CI) | Mean selected position (95% CI) | Trend p | Adjusted OR per step (95% CI; p) | Interpretation |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    for summary in runs:
        randomized_run = summary["_experiment_design"]["position_order"] == "random"
        for simulator in SIMULATORS:
            row = summary[simulator]
            bootstrap = row["user_cluster_bootstrap"]
            adjusted = row["adjusted_logistic"]
            interpretation = causal_interpretation(row) if randomized_run else association_interpretation(row)
            lines.append(
                f"| {simulator} | {run_label(summary)} | "
                f"{ci(row['position_1_minus_4'], bootstrap['position_1_minus_4_ci'], 4)} | "
                f"{ci(row['position_1_over_4'], bootstrap['position_1_over_4_ci'])} | "
                f"{ci(row['mean_selected_rank'], bootstrap['mean_selected_rank_ci'])} | "
                f"`{pvalue(row['cochran_armitage_p_two_sided'])}` | "
                f"{ci(adjusted['odds_ratio_per_rank_step'], adjusted['odds_ratio_95_ci'])}; "
                f"`p={pvalue(adjusted['p_two_sided'])}` | {interpretation} |"
            )
    lines += [
        "",
        "## Prespecified decision rule",
        "",
        "A randomized run is labeled as a causal front-position effect only when all three conditions hold:",
        "",
        "1. the user-cluster bootstrap 95% CI for P1−P4 is entirely above 0;",
        "2. the user-cluster bootstrap 95% CI for P1/P4 is entirely above 1; and",
        "3. the user-cluster-robust adjusted odds-ratio 95% CI is entirely below 1.",
        "",
        "A significant but non-monotonic position curve is reported as non-monotonic rather than "
        "collapsed into a generic bias label. The adjusted model controls for log item train degree, "
        "held-out ground-truth membership, and page fixed effects.",
        "",
        "## Metric provenance",
        "",
        "- Collins et al. (2018), *A Study of Position Bias in Digital Library Recommender Systems*: randomized order, rank-level CTR, and chi-square testing. <https://arxiv.org/abs/1802.06565>",
        "- Joachims, Swaminathan, and Schnabel (2017), *Unbiased Learning-to-Rank with Biased Feedback*: randomized interventions for position propensity identification. <https://www.cs.cornell.edu/~schnabts/downloads/joachims2017ltr.pdf>",
        "- Bito, Ren, and He (2025), *Evaluating Position Bias in Large Language Model Recommendations*: repeated shuffled/reversed candidate orders. <https://arxiv.org/abs/2508.02020>",
        "- Peláez-González, Herrera, and Herrera-Poyatos (2026), *Statistical Detection of Positional Bias in LLM-Based Recommender Systems*: permutation-based position distributions and query-level bootstrap uncertainty. <https://arxiv.org/abs/2610.37962>",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--random-summary", type=Path, action="append", required=True,
                        help="randomized summary.json; repeat for additional permutation seeds")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    baseline = load_summary(args.baseline_summary, "original")
    randomized = [load_summary(path, "random") for path in args.random_summary]
    seeds = [summary["_experiment_design"]["permutation_seed"] for summary in randomized]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate permutation seeds: {seeds}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = render(baseline, randomized, args.baseline_summary, args.random_summary)
    args.output.write_text(report.rstrip() + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
