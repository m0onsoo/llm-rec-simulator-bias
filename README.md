# Agent4Rec and SimUSER bias experiments

This repository is the shared, reproducible workspace for the MovieLens-1M Agent4Rec and SimUSER experiments. It keeps both simulators, the exact fixed LightGCN ranking supplied to them, and the activity/rating/position-bias analyses in one checkout.

The repository does **not** enable PQA. SimUSER is retained under `PQA/SimUSER` only to preserve its upstream directory layout and existing relative imports.

## Repository layout

```text
Agent4Rec/                         Agent4Rec source and MovieLens-1M inputs
PQA/SimUSER/                      SimUSER source, runners, audits, and Slurm files
artifacts/rankings/               Immutable common LightGCN ranking and manifest
analysis/                         Cross-simulator position-bias analysis
docs/JINYI_RATING_ABLATION_HANDOFF.md
docs/POSITION_BIAS_RANDOMIZED_EXPERIMENT.md
scripts/verify_artifacts.py       Preflight SHA-256 verifier
```

The source snapshots were assembled from:

- Agent4Rec revision `c29a9963ace6315d4e8b5bc57298547a47b96104`, derived from <https://github.com/LehengTHU/Agent4Rec>.
- PQA/SimUSER revision `36e68a5cdc6976dfc3acbcd4331179a3691c1ca6`, derived from <https://github.com/chaehyun1/PQA>.
- The locally prepared SimUSER runners, memory preparation, activity analyses, and Slurm templates present in the source workspace at assembly time.

Generated sessions, memories, profiles, scheduler logs, checkpoints, and API secrets are deliberately excluded.

## 1. Clone and configure

Create a Python environment suitable for the original projects, then install their dependencies. Preserve the cluster environment used for the completed runs when exact reproduction matters.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Agent4Rec/requirements.txt
pip install -r PQA/requirements.txt
cp .env.example Agent4Rec/.env
```

Put `OPENAI_API_KEY` in `Agent4Rec/.env`. Both handoff workflows point to that file, and it is Git-ignored. Never put the key in a Slurm script or commit it.

## 2. Verify the fixed inputs

Run this from the repository root before any API call:

```bash
python scripts/verify_artifacts.py
```

The ranking must have SHA-256:

```text
906e830fc56652cf21a9f2818cd3fa571f9b38d7ff9909f4e11ae72cd15e89d1
```

It has shape `(1000, 3292)` and maps row `u` directly to user/avatar `u`. It contains ordered item IDs, not scores. All experimental conditions must read this same file; changing the Python or CUDA environment cannot change its order.

## 3. Prepare the SimUSER view of the common ranking

Agent4Rec reads the NPY directly. SimUSER uses a lossless JSON projection of the first 20 items per user. Generate it from the same NPY rather than maintaining a second independent recommendation artifact:

```bash
mkdir -p PQA/SimUSER/MovieLens/result
python PQA/SimUSER/MovieLens/scripts/prepare_common_ranking.py \
  --rankings-npy artifacts/rankings/full_rankings_1000.npy \
  --cf-data-dir Agent4Rec/datasets/ml-1m/cf_data \
  --output PQA/SimUSER/MovieLens/result/agent4rec_common_lightgcn_u1000_top20.json \
  --manifest PQA/SimUSER/MovieLens/result/agent4rec_common_lightgcn_u1000_top20.manifest.json \
  --num-users 1000 \
  --items-per-user 20
```

The conversion checks row order, duplicate IDs, split masking, round-trip equality, and records input/output hashes. The generated JSON is intentionally ignored because it is deterministically derived from the tracked NPY.

## 4. Confirm vanilla-safe feature flags

Every added treatment is opt-in. Omitting these flags follows the original path:

| Treatment | Agent4Rec/SimUSER flag | Vanilla default |
|---|---|---|
| Activity text | `--activity_persona_mode` / `--activity-mode` | included |
| Activity crossover | `--activity_persona_transform swap_extremes` | identity |
| Item rating cue | `--item_rating_mode excluded` | included |
| Page position | `--position-order random --permutation-seed SEED` | original |
| PQA | not used | disabled |

Run the local treatment tests before submitting jobs:

```bash
cd Agent4Rec
python -m pytest -q \
  tests/test_item_rating_mode.py \
  tests/test_activity_persona_transform.py \
  tests/test_position_order.py
cd ..
```

## 5. Run the remaining experiments

The detailed, copy-ready commands, acceptance checks, metrics, and result tables are maintained in:

1. [`docs/JINYI_RATING_ABLATION_HANDOFF.md`](docs/JINYI_RATING_ABLATION_HANDOFF.md): rating-hidden conditions and activity-label crossover.
2. [`docs/POSITION_BIAS_RANDOMIZED_EXPERIMENT.md`](docs/POSITION_BIAS_RANDOMIZED_EXPERIMENT.md): within-page randomized Agent4Rec and SimUSER runs.

For position bias, the minimum causal experiment uses the same 1,000 users, ranking, four items per page, five-page maximum, GPT-6 Luna settings, and permutation seed `20261121` in both simulators. If budget permits, repeat with seeds `20261122` and `20261123`. Use a new result directory for every treatment and seed; `--resume` may continue only an identical manifest.

GPU resources are not required when the cached ranking is used. The simulator workloads are CPU plus OpenAI API calls.

## 6. Audit before analysis

Do not interpret a run until all 1,000 users are complete, its status records exit code 0, and its simulator-specific audit passes. In particular:

- candidate IDs and order must equal the fixed ranking page slices;
- rating-hidden logs must not contain `History ratings:`;
- crossover records must show `1→3`, `2→2`, and `3→1` while retaining original group labels;
- randomized manifests must record `position_order: random` and the requested permutation seed;
- a randomized run must never resume into a vanilla directory.

For the randomized position experiment, run from the repository root:

```bash
python analysis/analyze_position_bias.py \
  --agent-run Agent4Rec/storage/ml-1m/LightGCN/AGENT_RANDOM_RUN \
  --simuser-run PQA/SimUSER/MovieLens/result/SIMUSER_RANDOM_RUN \
  --output-dir analysis/position_bias_random_s20261121
```

Do not fill the handoff template manually. After the corresponding audits and analyzer complete successfully, generate the final report from the machine-readable summaries.

Then generate the single combined Agent4Rec/SimUSER results document:

```bash
python analysis/compile_position_bias_results.py \
  --baseline-summary analysis/position_bias/summary.json \
  --random-summary analysis/position_bias_random_s20261121/summary.json \
  --output analysis/POSITION_BIAS_EXPERIMENT_RESULTS.md
```

`analysis/POSITION_BIAS_EXPERIMENT_RESULTS.md` is the final source of truth to commit and share. It contains both simulators and both original/randomized conditions; do not manually copy values between reports.

## Git and artifact policy

`artifacts/rankings/full_rankings_1000.npy` is intentionally tracked as an immutable 26 MB experiment input. Do not modify it in place. If another recommender is added, give its ranking a new filename and manifest entry. Runtime outputs and large generated memories remain outside Git; share only curated aggregate CSV/JSON reports needed to reproduce a paper table or figure.

