# Position-bias audit: Agent4Rec and SimUSER

## Main results

| Simulator | Users | Pages | Selections | Rate pos. 1 | Rate pos. 2 | Rate pos. 3 | Rate pos. 4 | P1/P4 | Mean selected rank | Adjusted OR / rank step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Agent4Rec | 1000 | 3117 | 4752 | 44.08% | 39.75% | 35.61% | 33.01% | 1.335 | 2.378 | 0.863 [0.838, 0.889] |
| SimUSER | 1000 | 5000 | 12932 | 70.28% | 63.62% | 60.94% | 63.80% | 1.102 | 2.457 | 0.918 [0.895, 0.942] |

The adjusted odds ratio controls for log item train degree, held-out relevance, and page number; standard errors are clustered by user.

## Statistical evidence

### Agent4Rec

- Position 1 minus position 4: 0.1107; user-bootstrap 95% CI [0.0902, 0.1317].
- Position 1 / position 4: 1.335; user-bootstrap 95% CI [1.264, 1.414].
- Mean selected rank: 2.378; user-bootstrap 95% CI [2.355, 2.399] (uniform-position null: 2.5).
- Pearson rank × selection chi-square p=4.38e-20; Cochran–Armitage ordered-trend z=-9.599, p=8.09e-22.
- Adjusted per-rank-step OR=0.863, 95% CI [0.838, 0.889], p=2.81e-22.

### SimUSER

- Position 1 minus position 4: 0.0648; user-bootstrap 95% CI [0.0474, 0.0826].
- Position 1 / position 4: 1.102; user-bootstrap 95% CI [1.073, 1.132].
- Mean selected rank: 2.457; user-bootstrap 95% CI [2.446, 2.468] (uniform-position null: 2.5).
- Pearson rank × selection chi-square p=2.92e-22; Cochran–Armitage ordered-trend z=-7.316, p=2.55e-13.
- Adjusted per-rank-step OR=0.918, 95% CI [0.895, 0.942], p=4.84e-11.

## Interpretation boundary

These completed vanilla runs used a fixed relevance-ranked order, not randomized permutations. The estimates therefore establish selection-position association, not the causal effect of position. A causal position-bias claim requires rerunning identical user/page candidate sets under randomized or balanced permutations.

## Method references

- Collins et al. (2018), *A Study of Position Bias in Digital Library Recommender Systems*: rank-level CTR under shuffled recommendations and a chi-square rank/selection test.
- Joachims, Swaminathan, and Schnabel (2017), *Unbiased Learning-to-Rank with Biased Feedback*: randomized pilot interventions to estimate rank examination propensities, then inverse-propensity weighting.
- Bito, Ren, and He (2025), *Evaluating Position Bias in Large Language Model Recommendations*: shuffled/reversed candidate orders and Kendall-tau positional consistency.
- Peláez-González, Herrera, and Herrera-Poyatos (2026), *Statistical Detection of Positional Bias in LLM-Based Recommender Systems*: repeated random permutations, position-selection distributions, total-variation positional bias, and query-level bootstrap intervals.
