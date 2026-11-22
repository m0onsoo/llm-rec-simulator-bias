# A Behavioral Trait Leaks into Preferences: Diagnosing Trait Interference in LLM User Simulators

SimUSER-only working checkout derived from the official code for **"A Behavioral Trait Leaks into Preferences: Diagnosing Trait Interference in LLM User Simulators"** (CIKM 2026 short paper).

LLM user simulators inject a behavioral *activity* trait that is meant to control only how long a user browses. In practice, amplifying activity makes the simulator accept preference-mismatched items to keep browsing (**trait interference**), and satisfaction scores inflate with page count (**evaluation invalidity**).
PQA fixes this by giving the simulator a personalized page-level quality anchor `μ_u`, computed from the user's own history. Each page is labeled ABOVE / NORMAL / BELOW against this anchor before the continue-or-exit decision, so activity modulates browsing depth only within preference-conforming pages.

This checkout contains the [SimUSER](https://aclanthology.org/2025.acl-industry.5/) implementation only. The repository-local PQA/Agent4Rec implementation was removed to avoid confusion with the separate official Agent4Rec clone used by the experiments.

The MovieLens reproduction scripts read datasets, splits, metadata, the API environment, and cached LightGCN rankings from the sibling official clone at:

```text
/home1/moonsoo/projects/ara/click_conversion_reranker/base/Agent4Rec
```

That external repository is an input dependency and must not be confused with this `PQA` checkout.

## Repository structure

```
SimUSER/
├── MovieLens/
│   ├── Phase_1_Persona_Matching/     # 1 summary → 2 persona candidates → 3 self-consistent selection
│   ├── Phase_2_Simulate_Interactions/
│       ├── 4_Persona_Module.py
│       ├── 5_1_download_posters.py   # optional: poster captions (perception)
│       ├── 5_2_Perception_Module.py  # optional: poster captions (perception)
│       ├── 6_Episodic_Memory.py
│       ├── 7_KG_Memory.py
│       └── 8_Brain_Module1.py        # legacy SimUSER + PQA simulation
│   ├── scripts/                       # vanilla/PQA-free reproduction and analysis
│   ├── slurm/                         # CPU Slurm jobs
│   └── analysis/                      # activity/popularity reports
└── CDs_and_Vinyl/                    # same layout (no perception step)
```

## Setup

```bash
source /home1/moonsoo/projects/venv/bin/activate
# Active Slurm workflows read OPENAI_API_KEY from the sibling Agent4Rec/.env.
# Do not print or copy the key into result files.
```

The active MovieLens workflows use `gpt-6-luna` through the existing OpenAI-compatible endpoint configuration.

## Data

Datasets and recommendation lists are not included in this checkout. The current MovieLens reproduction uses the sibling official Agent4Rec clone for:

- `datasets/ml-1m/raw_data/`
- `datasets/ml-1m/cf_data/{train,valid,test}.txt`
- `datasets/ml-1m/simulation/`
- `.env` containing `OPENAI_API_KEY`
- the cached LightGCN ranking under `storage/ml-1m/LightGCN/`

The ranking is converted to SimUSER JSON without changing user IDs, item IDs, or order. Conversion manifests record the source NPY hash and split hashes.

## Running

### Vanilla SimUSER (PQA disabled)

The active MovieLens runner is `SimUSER/MovieLens/scripts/run_vanilla_common.py`. It does not import the legacy PQA brain and records `pqa_enabled: false` in every run manifest.

Prepared Slurm workflows include:

```text
SimUSER/MovieLens/slurm/run_1000_vanilla_common_luna.sbatch
SimUSER/MovieLens/slurm/run_1000_no_activity_luna.sbatch
```

Both use the same cached LightGCN ranking, four items per page, at most five pages, `gpt-6-luna`, CPU execution, atomic per-user outputs, and resumable user batches. The no-activity setting removes the explicit activity description from interaction prompts while retaining the activity group only as an analysis label.

Results, audit reports, aggregate selection tables, and manifests are written under `SimUSER/MovieLens/result/`. Activity-group reports are written under `SimUSER/MovieLens/analysis/`.

`SimUSER/*/Phase_2_Simulate_Interactions/8_Brain_Module1.py` remains as the legacy PQA implementation for reference, but it is not on the vanilla execution path.

## Citation

```bibtex
@article{kim2026behavioral,
  title={A Behavioral Trait Leaks into Preferences: Diagnosing Trait Interference in LLM User Simulators},
  author={Kim, Chaehyun and Kim, Sein and Kang, Hongseok and Park, Chanyoung},
  journal={arXiv preprint arXiv:2609.25572},
  year={2026}
}

```
