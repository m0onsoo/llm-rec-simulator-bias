# Agent4Rec rating-cue ablation and activity crossover: experiment handoff for Jinyi

## Goal and current status

The next experiment tests whether the explicit per-movie `History ratings` value shown to Agent4Rec causes the simulator to select systematically higher-rated and more popular movies. We repeat that direct shown-versus-hidden comparison with and without the activity-frequency persona, producing four controlled conditions:

| Activity persona in prompt | Item historical rating in prompt | Status | Run directory |
|---|---|---|---|
| Included | Included | Complete | `storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_batch4_from200` |
| Excluded | Included | Complete | `storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_no_activity` |
| Included | Excluded | **Primary run needed** | `storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_no_item_rating` |
| Excluded | Excluded | **Robustness run needed** | `storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_no_activity_no_item_rating` |

Run all 1,000 users for up to five pages (four items per page). Do not substitute a smaller panel or a one-page run. The completed conditions used GPT-6 Luna, seed 101, temperature 0, no reasoning effort, and the same cached LightGCN ranking. Every new condition must preserve all of these settings.

This is a rating-**presentation** ablation. The movie's historical mean rating remains available in the dataset for evaluation but must not appear in the item text sent to the LLM. Conformity and diversity personas remain unchanged. The activity-included comparison is the primary test because it changes only the rating cue from the reproduced vanilla condition. The activity-excluded comparison is a robustness check, not a test that activity moderates rating sensitivity.

The separate **activity crossover** tests whether the injected activity description itself causes the group differences. It swaps only the extreme prompt labels: original Infrequent (`1`) receives the Frequent (`3`) description, original Frequent receives the Infrequent description, and Occasional (`2`) remains unchanged as a negative-control group. Original group labels are retained for analysis.

| Original activity mapping | Prompt activity mapping | Item rating shown? | Status | Run directory |
|---|---|---|---|---|
| Identity | Identity | Yes | Complete vanilla reference | `storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_batch4_from200` |
| 1↔3, 2 unchanged | Swapped extremes | Yes | **Crossover run needed** | `storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_activity_swap` |
| Identity | Identity | No | **Primary rating-hidden run needed** | `storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_no_item_rating` |
| 1↔3, 2 unchanged | Swapped extremes | No | **Crossover rating-hidden run needed** | `storage/ml-1m/LightGCN/agent4rec_lightgcn_luna_1000_5pages_activity_swap_no_item_rating` |

## Code and vanilla-safety contract

The new CLI argument is:

```text
--item_rating_mode {included,excluded}
```

`included` is the default. `--exclude_item_rating` is an alias for `--item_rating_mode excluded`.

The activity crossover uses:

```text
--activity_persona_transform swap_extremes
```

Its default is `identity`. The transform is valid only with `--activity_persona_mode included`.

The implementation is deliberately opt-in:

- Default `included` produces the legacy recommendation string byte-for-byte and does not add a new manifest or behavior field.
- Only `excluded` removes the `History ratings: <value>` fragment from recommended-item, forced-validation, and advertisement presentation text.
- Only `excluded` records `item_rating_mode: excluded` in `run_manifest.json` and behavior records.
- Existing vanilla runs therefore retain resume compatibility and output structure. Never change the default when running the original reproduction.

Regression coverage is in `tests/test_item_rating_mode.py` and `tests/test_activity_persona_transform.py`. Before submitting jobs, run:

```bash
python -m pytest -q tests/test_item_rating_mode.py tests/test_activity_persona_transform.py
```

If `pytest` is unavailable, install it in the experiment environment; do not change application dependencies merely to bypass the test.

## Required ranking artifact

All new runs must consume the exact cached ranking below. It is about 26 MB and `storage/` is Git-ignored, so it must be transferred separately or regenerated from the same checkpoint before running.

```text
filename: full_rankings_1000.npy
SHA-256: 906e830fc56652cf21a9f2818cd3fa571f9b38d7ff9909f4e11ae72cd15e89d1
```

Do not regenerate recommendations independently for each condition. Sharing this array makes the underlying ranked candidate list identical. With a cached ranking, these simulations are CPU/API workloads; no GPU request is necessary.

The `.npy` file stores the **complete ordered item-ID list for each user**, not model scores. The simulation assigns row `u` to avatar `u` for users `0..999` and slices that row into consecutive groups of four for pages 1–5. There is no checkpoint inference, sampling, or re-ranking after the file is loaded. Therefore a different Python/CUDA environment cannot change the candidate order.

The artifact alone is not sufficient if the item-ID mapping or execution settings have been changed. Use the repository revision containing rating-ablation commit `a9fd38b` and activity-crossover commit `9d678d8` (or a descendant with no changes to ranking loading, dataset mapping, or page generation), together with the tracked `datasets/ml-1m` files from that revision. The Slurm templates verify SHA-256 hashes for the ranking, train/valid/test splits, movie metadata, personas, and activity statistics before making an API call. The loader also rejects an array whose user/item dimensions do not match the dataset. The templates fix and record the following settings in `run_manifest.json`:

```text
dataset=ml-1m, modeltype=LightGCN, n_avatars=1000
items_per_page=4, max_pages=5, seed=101
llm_model=gpt-6-luna, reasoning_effort=none, temperature=0.0
no advertisements, original ranking order
```

Before interpreting results, compare every `recommended_id` in the four-condition common prefix. The required mismatch count is zero. This final output-level check catches any mapping or configuration mistake even if the input hash passed.

## Submission commands

The parameterized Slurm template is `slurm/run_1000_luna_no_item_rating.sbatch`. Adjust only the site-specific `#SBATCH --account`, partition, module, and virtual-environment lines if Jinyi's cluster differs. Put `OPENAI_API_KEY=...` in the repository-local, Git-ignored `.env` file.

Submit the activity-included/rating-hidden condition first:

```bash
sbatch --export=ALL,ACTIVITY_PERSONA_MODE=included,RANKINGS_PATH=/absolute/path/full_rankings_1000.npy \
  slurm/run_1000_luna_no_item_rating.sbatch
```

After it finishes cleanly, submit the no-activity/rating-hidden condition:

```bash
sbatch --export=ALL,ACTIVITY_PERSONA_MODE=excluded,RANKINGS_PATH=/absolute/path/full_rankings_1000.npy \
  slurm/run_1000_luna_no_item_rating.sbatch
```

Submit the activity crossover with ratings shown:

```bash
sbatch --export=ALL,ITEM_RATING_MODE=included,RANKINGS_PATH=/absolute/path/full_rankings_1000.npy \
  slurm/run_1000_luna_activity_swap.sbatch
```

Submit the same crossover with ratings hidden:

```bash
sbatch --export=ALL,ITEM_RATING_MODE=excluded,RANKINGS_PATH=/absolute/path/full_rankings_1000.npy \
  slurm/run_1000_luna_activity_swap.sbatch
```

Both Slurm templates use `--resume`, two concurrent workers, batches of two, a ten-second inter-batch pause, retry/error status recording, ranking/data hash validation, and an item-ID matching audit. Rating-hidden runs also check that `History ratings:` did not leak into recommendation logs. Re-submit the same command after interruption; completed users are preserved, and the manifest prevents resuming with a different configuration.

Acceptance criteria for every run:

1. Exactly 1,000 complete behavior files exist and `scripts/audit_item_matching.py` passes.
2. The `.status` file reports `exit_code=0`.
3. The ranking and tracked-data hashes pass.
4. Every observed `recommended_id` equals the corresponding cached-ranking page slice.
5. Rating-hidden manifests contain `"item_rating_mode": "excluded"`, and their logs contain no `History ratings:` occurrence.
6. Crossover manifests contain `"activity_persona_transform": "swap_extremes"`; behavior records must show mappings `1→3`, `2→2`, and `3→1`.

## Evaluation design

### Primary estimands

Use `watch_id` as selection because Agent4Rec treats viewing as the adoption action. Report `like_id` as a secondary, stricter outcome. Compute movie popularity as **training-set item degree** only; do not use test interactions. Compute the displayed historical rating from `datasets/ml-1m/simulation/movie_detail.csv`, matching the value that was hidden or shown.

For every user and condition, calculate:

- **Selection rate:** selected impressions / displayed impressions.
- **Popularity uplift:** mean train-degree percentile of selected items minus mean train-degree percentile of displayed impressions. Percentile is preferred to raw degree because degree is heavy-tailed and its absolute scale is difficult to compare.
- **Historical-rating uplift:** mean historical rating of selected items minus mean historical rating of displayed impressions.
- **Session outcomes:** average exit page, view count/ratio, like count/ratio, and satisfaction where available.

Average Recommendation Popularity (ARP) defines popularity as the number of training ratings received by an item and averages it over recommendation lists; this is the source for our train-degree measure ([Abdollahpouri, Burke, and Mobasher, 2019, §Evaluation](https://arxiv.org/abs/1901.07555)). Our selected-minus-impression uplift is a study-specific exposure-normalized adaptation: the displayed list is the within-user baseline, and the selected list is the simulator output. It is not an original Agent4Rec metric.

Agent4Rec's native session metrics are average viewing ratio, number and ratio of likes, exit page, and satisfaction score ([Zhang et al., 2024, §3.3](https://arxiv.org/abs/2310.10108)). The paper also uses Recall@20 and NDCG@20 only when retraining and evaluating recommenders from simulated feedback ([§3.4](https://arxiv.org/abs/2310.10108)); those offline metrics are not needed for this prompt-cue ablation because no recommender is retrained.

### Identical-exposure analysis

The primary comparison must use each user's **four-condition common page prefix**:

```text
pages 1 .. min(last page in all four conditions)
```

Verify that every item ID and order in that prefix matches across all four conditions. This makes the compared impressions identical. Full-session results are still useful as total effects, but later pages mix prompt effects with condition-dependent exit/attrition and must be labeled accordingly.

Within each activity group (Infrequent, Occasional, Frequent), make two direct paired comparisons: rating shown versus hidden when the activity prompt is included, and rating shown versus hidden when the activity prompt is excluded. Report paired user-level means, the paired mean difference, 95% confidence interval, paired Cohen's dz, and raw two-sided paired-test p-value. These comparisons answer the immediate question without requiring a difference-in-differences interaction or a study-specific regression model. Conformity remains unchanged in all four runs and should be discussed as a possible remaining source of rating sensitivity, not removed or modeled unless a later analysis specifically requires it.

For the activity crossover, compare identity versus swapped prompts separately with ratings shown and hidden. Stratify by the user's **original** activity group. A prompt-driven effect should reverse the extreme groups: original Infrequent users should become more active when shown the Frequent description, while original Frequent users should become less active when shown the Infrequent description. Occasional users receive identical activity text and serve as a negative control. Apply the same full-session and identical-exposure common-prefix distinction used for the rating experiment.

## Results already available

### Ground-truth profile popularity

The groups differ before simulation, so selected degree alone must not be interpreted as an LLM effect.

| Activity group | Users | Mean train degree of user's training interactions |
|---|---:|---:|
| Infrequent | 539 | 77.0660 |
| Occasional | 352 | 65.1706 |
| Frequent | 109 | 50.8405 |

### Full-session results with ratings shown

These are total effects and include differential page reach.

| Activity group | Activity prompt | Users | Users in degree comparison | Watch rate | Impression mean degree | Watch-selected mean degree | Selected − impression degree |
|---|---|---:|---:|---:|---:|---:|---:|
| Infrequent | Included | 539 | 480 | 0.2059 | 120.6273 | 128.1351 | +7.5078 |
| Infrequent | Excluded | 539 | 531 | 0.5384 | 112.3479 | 115.8071 | +3.4592 |
| Occasional | Included | 352 | 333 | 0.3676 | 106.5355 | 114.4036 | +7.8680 |
| Occasional | Excluded | 352 | 346 | 0.6180 | 105.5687 | 109.9195 | +4.3508 |
| Frequent | Included | 109 | 108 | 0.7436 | 92.7104 | 93.7147 | +1.0042 |
| Frequent | Excluded | 109 | 108 | 0.7016 | 94.8566 | 97.7510 | +2.8944 |

`Users in degree comparison` counts users with at least one watch in that condition. Impression and selected degree means, and their difference, use that same subset so the three degree columns are arithmetically comparable. Watch rate uses all users in the group and treats zero-watch users as zero.

### Identical-exposure common-prefix results with ratings shown

All 11,896 paired impressions matched exactly between the two completed conditions. Activity text had a very large effect on how many movies were selected, while the selected-set popularity shift was much smaller.

| Activity group | Activity prompt | Users | Users in degree comparison | Watch rate | Impression mean degree | Watch-selected mean degree | Selected − impression degree |
|---|---|---:|---:|---:|---:|---:|---:|
| Infrequent | Included | 539 | 465 | 0.2063 | 120.3203 | 127.5527 | +7.2324 |
| Infrequent | Excluded | 539 | 465 | 0.5580 | 120.3203 | 124.2349 | +3.9146 |
| Occasional | Included | 352 | 329 | 0.3683 | 107.5350 | 115.6333 | +8.0983 |
| Occasional | Excluded | 352 | 329 | 0.6276 | 107.5350 | 111.9400 | +4.4050 |
| Frequent | Included | 109 | 106 | 0.7410 | 94.5792 | 95.5434 | +0.9642 |
| Frequent | Excluded | 109 | 106 | 0.7003 | 94.5792 | 97.4936 | +2.9144 |

Here `Users in degree comparison` counts users who watched at least one item in both conditions. Impression and selected means use the same paired-user subset. The identical impression means within each group confirm that the compared candidates are the same.

The historical rating cue is a plausible confound because train degree and displayed historical rating are correlated in these impressions (Pearson `r=0.5348`, Spearman `ρ=0.5428`). Conditional on selecting in both conditions, the current rating results are:

| Activity group | Impression mean rating | Selected rating: activity included | Rating uplift | Selected rating: activity excluded | Rating uplift | Included − excluded selected rating |
|---|---:|---:|---:|---:|---:|---:|
| Infrequent | 3.9442 | 4.0629 | +0.1252 | 4.0029 | +0.0652 | +0.0600 |
| Occasional | 3.8545 | 3.9531 | +0.0945 | 3.9042 | +0.0456 | +0.0489 |
| Frequent | 3.7929 | 3.7979 | +0.0078 | 3.8167 | +0.0265 | −0.0187 |

The activity-included versus excluded differences in selected rating are only `+0.0600`, `+0.0489`, and `−0.0187` on a five-point scale for the three groups. Activity text strongly changes **how many** movies are watched, but the current evidence does not show a large practical effect on the mean historical rating of the selected movies. Accordingly, activity is not the main rating-bias hypothesis. The rating-hidden runs instead test whether the explicit numerical rating cue itself explains the positive selected-versus-impression rating uplift, while the no-activity run serves as a robustness check.

## Tables to fill after the additional runs

Replace `TBD` cells only after the corresponding runs pass the acceptance checks.

### Four-way common-prefix coverage

| Activity group | Users | Mean common pages | Matched impressions | Candidate mismatches |
|---|---:|---:|---:|---:|
| Infrequent | TBD | TBD | TBD | **must be 0** |
| Occasional | TBD | TBD | TBD | **must be 0** |
| Frequent | TBD | TBD | TBD | **must be 0** |

### Full-session primary watch metrics

| Activity group | Activity prompt | Rating shown? | Users | Users in selected-item comparison | Watch rate | Impression popularity percentile | Selected popularity percentile | Popularity uplift (pp) | Impression rating | Selected rating | Rating uplift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Infrequent | Included | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Infrequent | Included | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Infrequent | Excluded | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Infrequent | Excluded | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Occasional | Included | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Occasional | Included | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Occasional | Excluded | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Occasional | Excluded | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Included | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Included | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Excluded | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Excluded | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Four-condition common-prefix primary watch metrics

| Activity group | Activity prompt | Rating shown? | Users | Users in selected-item comparison | Watch rate | Impression popularity percentile | Selected popularity percentile | Popularity uplift (pp) | Impression rating | Selected rating | Rating uplift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Infrequent | Included | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Infrequent | Included | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Infrequent | Excluded | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Infrequent | Excluded | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Occasional | Included | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Occasional | Included | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Occasional | Excluded | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Occasional | Excluded | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Included | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Included | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Excluded | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Excluded | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

For both tables, compute the selected-item columns only on the explicitly reported comparison subset; do not silently mix all-user impression means with selected-user means. For the common-prefix table, use users with a defined selected mean in both members of the relevant shown-versus-hidden pair.

### Secondary like metrics

| Activity group | Activity prompt | Rating shown? | Like rate | Popularity-percentile uplift (pp) | Historical-rating uplift |
|---|---|---|---:|---:|---:|
| Infrequent | Included | Yes | TBD | TBD | TBD |
| Infrequent | Included | No | TBD | TBD | TBD |
| Infrequent | Excluded | Yes | TBD | TBD | TBD |
| Infrequent | Excluded | No | TBD | TBD | TBD |
| Occasional | Included | Yes | TBD | TBD | TBD |
| Occasional | Included | No | TBD | TBD | TBD |
| Occasional | Excluded | Yes | TBD | TBD | TBD |
| Occasional | Excluded | No | TBD | TBD | TBD |
| Frequent | Included | Yes | TBD | TBD | TBD |
| Frequent | Included | No | TBD | TBD | TBD |
| Frequent | Excluded | Yes | TBD | TBD | TBD |
| Frequent | Excluded | No | TBD | TBD | TBD |

### Activity-crossover results to fill

Create separate full-session and common-prefix tables using these same columns. Include both rating-shown and rating-hidden crossover runs.

| Original activity group | Prompt mapping | Rating shown? | Users | Users in selected-item comparison | Watch rate | Impression popularity percentile | Selected popularity percentile | Popularity uplift (pp) | Impression rating | Selected rating | Rating uplift |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Infrequent | Identity | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Infrequent | Swapped to Frequent | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Occasional | Identity / unchanged | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Identity | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Swapped to Infrequent | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Infrequent | Identity | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Infrequent | Swapped to Frequent | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Occasional | Identity / unchanged | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Identity | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Frequent | Swapped to Infrequent | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Analysis implementation notes for the agent doing the work

Start from `scripts/analyze_common_prefix_activity.py`, which already loads behavior records, validates identical candidates, forms user-level common prefixes, and computes paired effects. Extend it to accept four named run directories rather than silently changing the existing two-condition script. Write new outputs under `analysis/artifacts/rating_ablation_*.csv` and update this document in the same commit.

The relevant behavior fields are `recommended_id`, `watch_id`, `like_id`, and `page`. Use item IDs, not normalized-title matches, whenever resolved IDs exist. Preserve one row for users who selected zero items when computing selection rates; selected-item mean popularity/rating is undefined for those users and its coverage must be reported. Never replace an undefined selected mean with zero.

Recommended deliverables:

- `analysis/artifacts/rating_ablation_common_prefix_users.csv`
- `analysis/artifacts/rating_ablation_summary.csv`
- `analysis/artifacts/rating_ablation_paired_comparisons.csv`
- `analysis/artifacts/activity_crossover_summary.csv`
- `analysis/artifacts/activity_crossover_paired_comparisons.csv`
- a page-wise figure showing impression and selected popularity-percentile means for all four conditions
- this Markdown file with `TBD` cells replaced and a short conclusion that distinguishes full-session total effects from identical-exposure common-prefix effects

## References

- An Zhang et al. “On Generative Agents in Recommendation.” SIGIR 2024. [arXiv:2310.10108](https://arxiv.org/abs/2310.10108), [DOI](https://doi.org/10.1145/3626772.3657844).
- Himan Abdollahpouri, Robin Burke, and Bamshad Mobasher. “Managing Popularity Bias in Recommender Systems with Personalized Re-ranking.” FLAIRS 2019. [arXiv:1901.07555](https://arxiv.org/abs/1901.07555).
- Harald Steck. “Calibrated Recommendations.” RecSys 2018. [DOI:10.1145/3240323.3240372](https://doi.org/10.1145/3240323.3240372). This is conceptual support for comparing an output distribution with a user's/profile baseline; the present selected-minus-impression uplift is not Steck's original calibration metric.
