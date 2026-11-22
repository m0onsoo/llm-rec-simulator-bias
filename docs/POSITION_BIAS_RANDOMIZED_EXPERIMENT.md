# Randomized position-bias experiment handoff

## Purpose

The completed vanilla Agent4Rec and SimUSER runs show that items displayed near the beginning of each four-item page are selected more often. Because the vanilla page order is the LightGCN relevance order, those results establish a position–selection association but do not by themselves identify a causal presentation-order effect.

This follow-up experiment randomizes the order of the same four candidates **within each user/page**. It does not change page membership, the recommender ranking used to construct the pages, user profiles, memories, prompts, model settings, or any other simulation behavior.

In this document, position 1 means the first item displayed within every page, not page 1. `P1/P4` is the position-1 selection rate divided by the position-4 selection rate.

## Code added for the experiment

Both simulators now accept an explicit page-order treatment:

- Agent4Rec: `--position-order {original,random}` and `--permutation-seed INT`
- SimUSER: `--position-order {original,random}` and `--permutation-seed INT`

`original` is the default. In default mode, the existing recommendation order, legacy manifest contents, and output schema are preserved. This is intentional: rerunning vanilla without the new flags follows the prior execution path and does not activate permutation logic.

With `--position-order random`, both runners use the same deterministic local permutation for a given `(permutation_seed, user_id, page)`. The implementation does not touch the global random-number state, so execution concurrency cannot change the assignment. Randomized outputs additionally record:

- `original_recommended_id` / `original_exposed_item_ids`
- `position_permutation` (zero-based indices into the original page)
- `position_order: random`
- `permutation_seed`

The run manifest also records the treatment and permutation version. Never resume a randomized run into a vanilla result directory or vice versa; the manifest check is designed to reject that.

Relevant files:

- `Agent4Rec/parse.py`
- `Agent4Rec/simulation/position_order.py`
- `Agent4Rec/simulation/arena.py`
- `PQA/SimUSER/MovieLens/scripts/run_vanilla_common.py`
- `PQA/SimUSER/MovieLens/scripts/audit_vanilla_run.py`
- `analysis/analyze_position_bias.py`
- `analysis/compile_position_bias_results.py`

## Required primary experiment

Run one complete randomized-order experiment for each simulator with exactly the vanilla settings below.

| Setting | Required value |
|---|---|
| Users | 1,000, IDs 0–999 |
| Candidates | Existing common LightGCN ranking |
| Items per page | 4 |
| Maximum pages | 5 |
| LLM | `gpt-6-luna` |
| Temperature | `0.0` |
| Reasoning effort | `none` where applicable |
| Simulation seed | `101` |
| Activity persona | Included |
| PQA | Disabled |
| SimUSER perception | Disabled |
| Position order | `random` |
| Permutation seed | `20261121` |
| Advertisements | Disabled |

The same permutation seed must be used in Agent4Rec and SimUSER. Use new output directories. `--resume` may be used only to continue the same randomized configuration after interruption.

If API budget permits, repeat the entire treatment with permutation seeds `20261122` and `20261123`. Multiple seeds are preferable because the LLM position-bias literature evaluates repeated candidate permutations. The seed-`20261121` runs are the minimum required experiment; additional seeds are robustness replications.

## Example Agent4Rec command

Run from `Agent4Rec/`. The cached-ranking path below points to the immutable artifact tracked by this repository:

```bash
python main.py \
  --simulation_name agent4rec_lightgcn_luna_1000_5pages_position_random_s20261121 \
  --dataset ml-1m \
  --modeltype LightGCN \
  --model_path Saved \
  --rankings_path ../artifacts/rankings/full_rankings_1000.npy \
  --resume \
  --n_avatars 1000 \
  --items_per_page 4 \
  --max_pages 5 \
  --execution_mode parallel \
  --max_workers 4 \
  --avatar_batch_size 4 \
  --batch_pause_seconds 5 \
  --seed 101 \
  --cuda 0 \
  --llm_model gpt-6-luna \
  --llm_base_url https://api.openai.com/v1 \
  --reasoning_effort none \
  --llm_temperature 0.0 \
  --max_api_retries 8 \
  --activity_persona_mode included \
  --position-order random \
  --permutation-seed 20261121
```

After completion:

```bash
python scripts/audit_item_matching.py \
  storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_position_random_s20261121
```

Do not enable `--add_advert`; randomized position experiments explicitly reject advertisement mode.

## Example SimUSER command

The existing vanilla profile, episodic-memory, KG-memory, ranking JSON, and split artifacts should be reused. No profile or memory regeneration is needed when those artifacts are present. Run from `PQA/` and adapt paths if necessary:

```bash
AGENT_ROOT=../Agent4Rec
MOVIE_ROOT=SimUSER/MovieLens
PROFILE_ROOT=${MOVIE_ROOT}/result/vanilla_profiles_u1000
MEMORY_ROOT=${MOVIE_ROOT}/result/vanilla_memory_u1000
RANDOM_RUN=${MOVIE_ROOT}/result/position_random_lightgcn_u1000_s20261121
RANKING_JSON=${MOVIE_ROOT}/result/agent4rec_common_lightgcn_u1000_top20.json
RANKING_MANIFEST=${MOVIE_ROOT}/result/agent4rec_common_lightgcn_u1000_top20.manifest.json
CF=${AGENT_ROOT}/datasets/ml-1m/cf_data
SIM=${AGENT_ROOT}/datasets/ml-1m/simulation

python -B SimUSER/MovieLens/scripts/run_vanilla_common.py \
  --ranking-json "${RANKING_JSON}" \
  --ranking-manifest "${RANKING_MANIFEST}" \
  --profiles "${PROFILE_ROOT}/user_profiles.json" \
  --episodic-dir "${MEMORY_ROOT}/episodic" \
  --kg "${MEMORY_ROOT}/kg_memory.json" \
  --movie-detail "${SIM}/movie_detail.csv" \
  --train-split "${CF}/train.txt" \
  --valid-split "${CF}/valid.txt" \
  --test-split "${CF}/test.txt" \
  --result-dir "${RANDOM_RUN}" \
  --env-file "${AGENT_ROOT}/.env" \
  --num-users 1000 \
  --model gpt-6-luna \
  --items-per-page 4 \
  --max-pages 5 \
  --workers 4 \
  --user-batch-size 4 \
  --batch-cooldown-seconds 5 \
  --seed 101 \
  --activity-mode included \
  --position-order random \
  --permutation-seed 20261121 \
  --resume
```

Audit it with:

```bash
python -B SimUSER/MovieLens/scripts/audit_vanilla_run.py \
  --run-dir "${RANDOM_RUN}" \
  --ranking-json "${RANKING_JSON}" \
  --train-split "${CF}/train.txt" \
  --test-split "${CF}/test.txt" \
  --num-users 1000
```

If the reusable profiles or memories are absent from the shared checkout, recreate them with the existing `prepare_vanilla_profiles.py` and `prepare_vanilla_memory.py` workflows before starting this command. Do not modify their prompts or settings.

## Analysis command

From the repository root:

```bash
python analysis/analyze_position_bias.py \
  --agent-run Agent4Rec/storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_position_random_s20261121 \
  --simuser-run PQA/SimUSER/MovieLens/result/position_random_lightgcn_u1000_s20261121 \
  --output-dir analysis/position_bias_random_s20261121
```

The analyzer checks that the two run manifests use the same treatment and seed, verifies identical displayed items on pages observed by both simulators, reconstructs every displayed position from raw page output, and writes:

- `REPORT.md`
- `summary.json`
- `position_rates.csv`
- `exposure_events.csv`

Run the analyzer separately for every additional permutation seed. The final compiler includes every supplied seed in the same Markdown. If a pooled cross-seed estimate is later added, it must cluster uncertainty by both user and permutation seed; the current compiler reports each seed separately.

## Mandatory final aggregation into one Markdown file

After both simulator runs pass their audits and the analyzer above creates the randomized `summary.json`, run:

```bash
python analysis/compile_position_bias_results.py \
  --baseline-summary analysis/position_bias/summary.json \
  --random-summary analysis/position_bias_random_s20261121/summary.json \
  --output analysis/POSITION_BIAS_EXPERIMENT_RESULTS.md
```

This command creates the **single final results document** containing both Agent4Rec and SimUSER, with vanilla and randomized rows, confidence intervals, statistical tests, automated causal interpretations, and metric sources:

```text
analysis/POSITION_BIAS_EXPERIMENT_RESULTS.md
```

Do not manually transcribe analyzer values into a separate report. This generated file is the source of truth to commit and share. For additional seeds, repeat `--random-summary`:

```bash
python analysis/compile_position_bias_results.py \
  --random-summary analysis/position_bias_random_s20261121/summary.json \
  --random-summary analysis/position_bias_random_s20261122/summary.json \
  --random-summary analysis/position_bias_random_s20261123/summary.json
```

The compiler refuses an original-order file in a randomized slot, duplicate seeds, missing simulators or positions, and any summary that does not contain exactly 1,000 users per simulator.

## Metrics and sources

### Primary reported metrics

1. **Selection rate by displayed position**: selected exposures divided by all exposures at positions 1–4. This is the click-through-rate analogue used by Collins et al. in a randomized recommender interface.
2. **P1−P4**: absolute percentage-point difference between positions 1 and 4, with a 95% percentile bootstrap interval obtained by resampling users.
3. **P1/P4**: relative risk, i.e. position-1 selection rate divided by position-4 selection rate, with a user-bootstrap 95% interval.
4. **Mean selected position**: the selection-weighted mean of positions 1–4. The equal-position reference is 2.5.
5. **Ordered position trend**: Cochran–Armitage trend statistic. Pearson's position × selection chi-square is retained as a literature-comparable secondary test.
6. **Per-position-step odds ratio**: logistic selection model with displayed position as the treatment. The implementation controls for log item train degree, held-out ground-truth membership, and page fixed effects, with standard errors clustered by user. Under randomized order, the unadjusted estimate is causally identified; the adjusted estimate is a precision/sensitivity analysis.

The user-cluster bootstrap and cluster-robust regression should drive inference because one user produces multiple dependent page/item observations. The naïve chi-square p-value should not be the sole evidence.

### Methodological sources

- Andrew Collins, Dominika Tkaczyk, Akiko Aizawa, and Joeran Beel. **A Study of Position Bias in Digital Library Recommender Systems** (2018). They randomly shuffled recommendations, compared CTR by displayed rank, and used a chi-square test of rank and click dependence. <https://arxiv.org/abs/1802.06565>
- Thorsten Joachims, Adith Swaminathan, and Tobias Schnabel. **Unbiased Learning-to-Rank with Biased Feedback** (WSDM 2017). They explain why fixed relevance-ranked logs confound relevance and examination and use randomized interventions to estimate position propensities. <https://www.cs.cornell.edu/~schnabts/downloads/joachims2017ltr.pdf>
- Ethan Bito, Yongli Ren, and Estrid He. **Evaluating Position Bias in Large Language Model Recommendations** (2025). They repeatedly shuffle/reverse candidate order and measure output stability with Kendall's tau. <https://arxiv.org/abs/2508.02020>
- Carlos Peláez-González, Francisco Herrera, and Andrés Herrera-Poyatos. **Statistical Detection of Positional Bias in LLM-Based Recommender Systems: Methodologies and Empirical Evidence** (2026). They use repeated random permutations, position-selection distributions, independence/uniformity tests, total-variation positional bias, and query-level bootstrap intervals. <https://arxiv.org/abs/2610.37962>
- PyClick is a public implementation of position-based and other click models; its PBM stores examination parameters by displayed rank. <https://github.com/markovi/PyClick>

Kendall's tau is not the primary metric here because Agent4Rec and SimUSER output a subset of watched items rather than a complete ranking. Repeated random seeds nevertheless follow the intervention principle used by the LLM recommendation papers.

## Results table to complete after the randomized run

The tables below are the requested-result template. Jinyi's agent should not edit them by hand; after the full run passes its audit, `analysis/compile_position_bias_results.py` generates the completed single results document from machine-readable summaries.

### Selection rate by displayed position

| Simulator | Run | Users | Pages | Position 1 | Position 2 | Position 3 | Position 4 |
|---|---|---:|---:|---:|---:|---:|---:|
| Agent4Rec | Vanilla, original order | 1,000 | 3,117 | 44.08% | 39.75% | 35.61% | 33.01% |
| Agent4Rec | Random order, seed 20261121 | TBD | TBD | TBD | TBD | TBD | TBD |
| SimUSER | Vanilla, original order | 1,000 | 5,000 | 70.28% | 63.62% | 60.94% | 63.80% |
| SimUSER | Random order, seed 20261121 | TBD | TBD | TBD | TBD | TBD | TBD |

### Effect summaries

| Simulator | Run | P1−P4 (95% CI) | P1/P4 (95% CI) | Mean selected position (95% CI) | Trend p-value | Adjusted OR per position step (95% CI; p) | Interpretation |
|---|---|---|---|---|---|---|---|
| Agent4Rec | Vanilla, original order | 0.1107 [0.0902, 0.1317] | 1.335 [1.264, 1.414] | 2.378 [2.355, 2.399] | `8.09e-22` | 0.863 [0.838, 0.889]; `p=2.81e-22` | Strong monotonic association; not causal because order was relevance-ranked |
| Agent4Rec | Random order, seed 20261121 | TBD | TBD | TBD | TBD | TBD | **TBD—fill from randomized report** |
| SimUSER | Vanilla, original order | 0.0648 [0.0474, 0.0826] | 1.102 [1.073, 1.132] | 2.457 [2.446, 2.468] | `2.55e-13` | 0.918 [0.895, 0.942]; `p=4.84e-11` | Weaker first-position association with a position-4 rebound; not causal |
| SimUSER | Random order, seed 20261121 | TBD | TBD | TBD | TBD | TBD | **TBD—fill from randomized report** |

For randomized runs, conclude causal front-position bias only if the user-cluster interval excludes the no-effect value (`0` for P1−P4, `1` for P1/P4 and the odds ratio) and the direction is substantively consistent across completed permutation seeds. Report non-monotonic patterns rather than reducing them to a single “bias/no bias” label.

## Completion checklist

- [ ] Agent4Rec randomized run has 1,000 valid completed users.
- [ ] SimUSER randomized run has 1,000 valid completed users.
- [ ] Both manifests say `position_order: random` and `permutation_seed: 20261121`.
- [ ] No vanilla output directory was overwritten or resumed with randomized settings.
- [ ] Agent4Rec item-matching audit passes.
- [ ] SimUSER randomized-ranking audit passes.
- [ ] The position-bias analyzer completes and detects randomized design.
- [ ] Per-run `REPORT.md`, `summary.json`, and CSV artifacts are retained.
- [ ] `analysis/compile_position_bias_results.py` creates `analysis/POSITION_BIAS_EXPERIMENT_RESULTS.md`.
- [ ] The generated single final Markdown is committed and shared; no values are manually transcribed.
- [ ] Any partial run, parsing failure, API error, or missing user is disclosed rather than silently excluded.
