# SteerEats: data preparation

Data preparation, baselines and first temporal analysis for the DS-GA 3001 project update (run of October 31, 2026).

## Stage one: logistic ranking

The completed model is `outputs/logistic_ranker/logistic_pipeline.joblib`; aggregate results,
feature coefficients, run metadata and training log are in the same folder. Test Recall@10 /
NDCG@10: logistic regression **0.0484 / 0.0357**, compared with Item-CF **0.0277 / 0.0193**
on the same 8,185 eligible users. Validation selected C=1. The input SHA-256 is recorded
in `run_metadata.json`; the completed run used the existing 422,788-row interaction sample.

To reproduce with an existing `data/processed/sample_interactions.parquet`:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-model-lock.txt
bash scripts/run_first_ranker.sh
```

`requirements-model-lock.txt` records versions for model training and loading. Data preparation
also needs the packages in `requirements.txt`. Raw/processed datasets and individual-user
evaluation CSVs stay local; running the model regenerates the individual-user metrics.
The saved joblib artifact contains the fitted scaler and logistic classifier; ranking also
requires historical features computed by `src/07_logistic_ranker.py`. Its scores are ranking
scores, not calibrated preference probabilities. Load the artifact only from this trusted repository.
Reports remain managed separately in Overleaf. The runner updates a local report only when
`report/main.tex` is present; otherwise it completes after saving model and evaluation outputs.

## Setup and run

Python 3.11+ with pandas, pyarrow, geopandas, shapely, scikit-learn and scipy. From this folder:

```bash
python3 src/01_build_nyc_restaurants.py   # NYC restaurants from the Google metadata
python3 src/02_extract_nyc_reviews.py     # their reviews from the 33.5M-line review file (~1 min)
python3 src/03_match_inspections.py       # match to DOHMH inspection records, write audit samples
python3 src/04_sample_and_split.py        # 20,000-user sample and time split
python3 src/05_baselines.py               # popularity, recent popularity, item-CF (~1 min)
python3 src/06_rating_over_time.py        # all-time vs last-12-months ratings
```

## Audit and cleaning (November 19, 2026)

Run after steps 01–04. Originals in `data/processed` are read only; cleaned files go to `data/clean`.

```bash
python src/10_data_audit.py --raw         # findings -> outputs/data_audit/AUDIT.md (+ CSVs for figures)
python src/11_clean.py                    # clean restaurants, reviews, interactions -> outputs/cleaning/CLEANING_LOG.md
python src/12_restaurant_attributes.py    # cuisine groups, chain, price, attribute flags
```

| File | One row per | Notes |
|---|---|---|
| `restaurants_clean.parquet` | restaurant (24,724) | 49 non-food businesses and 59 non-food venues (primary category shopping mall, gas station, convenience store or tourist attraction) removed; 25 duplicate listings merged (same name and address, or reviewed in `config/duplicate_listings.csv`); `merged_listings` count |
| `reviews_clean.parquet` | review (6,767,323) | empty, duplicate and placeholder-date rows removed; New York `local_date`, `local_hour`, `local_dow`; `user_day_reviews`; `cross_posted` |
| `interactions_clean.parquet` | first review of a restaurant by a sampled user (418,914) | same 20,000 users as the original sample, reproduced exactly; splits `train` / `val` / `test` (2020-08-01 to 2021-04-30) / `tail` (from 2021-05-01, collection already half-missing, never evaluated); `label` 1 = 4–5 stars, 0 = 1–2, missing = 3; `bulk_day` |
| `restaurant_attributes.parquet` | restaurant (24,724) | `cuisine_group` (25 groups incl. "Venue" for hotels, event spaces, food halls and markets without a cuisine, `src/taxonomy.py`), `is_chain`, `price_level` + `price_missing`, 12 `attr_*` flags |

Snapshot-only metadata (Google rating, review count, open state) and the 2026 inspection file are never used
as predictive features. `outputs/cleaning/linked_listings.csv` lists listing pairs that share cross-posted
reviews that remain after merging; these are different restaurants (one user's batch upload, or chain
branches), so they are kept apart. The baselines and logistic ranker above still use the
original sample; they are rerun on `interactions_clean.parquet` in the next step.

## Time-safe features and rankers (November 19, 2026)

```bash
python src/13_train_rankers.py          # LR + LightGBM, same features/labels/negatives/candidates (~16 min)
python src/14_evaluate.py               # bootstrap intervals, paired tests, robustness (~12 min)
python tests/check_time_safety.py       # truncation test: features must not change when future data is removed
```

`src/ranker_core.py` builds each snapshot strictly before its cutoff. Item popularity, ratings and
item-item similarity use **all** clean NYC reviews before the cutoff (2–6 million), not only the sample.
30 features in six groups: collaborative (CF on all / liked history), popularity (all-time, 12-month,
3-month, recency, age), rating (smoothed all-time and 12-month), location (distance from the user's median
visit point, same neighborhood, same borough), taste (cuisine affinity, price gap, chain affinity) and
static attributes. Candidates are restaurants reviewed in the 12 months before the cutoff, minus the
user's history; training negatives are half uniform, half popularity-matched (4 per positive).

Test results (8,025 users, 2020-08-01 to 2021-04-30) with 95% bootstrap intervals are in
`outputs/rankers/EVALUATION.md`: logistic regression NDCG@10 0.0480 [0.0453, 0.0509], Item-CF 0.0453
[0.0428, 0.0480], LightGBM 0.0413 [0.0387, 0.0439]. Logistic regression beats Item-CF by 6.0% (paired
interval excludes zero, Wilcoxon p = 4.6e-06); LightGBM trails logistic regression by 14.1%. The ordering
holds without bulk-day targets, in each test quarter, and before and during COVID. These numbers replace
the October results above, which used a different catalog, sample-only popularity and uncleaned data, so
the two sets are not directly comparable. Per-user metrics stay local (`*_per_user.csv`).

## Two-stage protocol (November 19, 2026)

```bash
python src/15_train_two_stage.py                    # stage 1: ~200 candidates; stage 2: LR / LightGBM re-rank (~7 min)
python src/15_train_two_stage.py --seed 1           # metrics-only reruns with other sampling seeds
python src/14_evaluate.py --protocol two_stage      # intervals, paired tests, robustness
python src/16_compare_protocols.py                  # outputs/MODEL_COMPARISON.md
```

Stage 1 (`ranker_core.candidates`) unions four retrievers: item-CF (100), liked-history CF (40),
nearby-and-popular (40) and city-wide trending (20). It reaches 31.1% of relevant test restaurants, which
bounds every re-ranker. Stage 2 trains on positives that stage 1 retrieved plus 10 negatives per positive
drawn from the same candidates. Training on all positives instead made the models learn that high CF
similarity means "not visited", and unshuffled rows let LightGBM's tie-breaking inflate validation NDCG.
Both issues were found in smoke runs and fixed before the reported results.

Two-stage test NDCG@10 (`outputs/rankers_two_stage/EVALUATION.md`): LightGBM 0.0482 [0.0456, 0.0510] and
logistic regression 0.0481 [0.0453, 0.0509] are tied (paired p = 0.97), both about 6% above Item-CF. A
supplementary LightGBM LambdaRank model reaches 0.0508 [0.0480, 0.0537], 5.6% above logistic regression
(p = 0.0034). Across three sampling seeds, logistic regression stays at 0.0479–0.0482, pointwise LightGBM
moves within 0.0482–0.0498, and LambdaRank within 0.0497–0.0508. LightGBM's 14% deficit under the
full-catalog protocol therefore came from its training negatives (random and popular restaurants), not
from the model class. The browser demo uses the two-stage design.

## SHAP explanations (November 19, 2026)

```bash
python src/17_shap.py     # ~7 min; outputs/shap/SHAP.md and aggregates; row-level values in data/clean/shap/ (local)
```

Explains the raw scores (log-odds) of the two-stage logistic regression and LightGBM for the 200
stage-1 candidates of 300 random test users (60,000 rows). Both use interventional SHAP with the same
200-row background, so they answer the same question. The logistic values use the closed form
w_j (x_j − E[x_j]), which matches `shap.LinearExplainer` to 1.8e-15. LightGBM uses TreeExplainer
(additivity residual median 6.8e-08; 2 rows above 1e-3, excluded from agreement statistics) and, for
comparison, path-dependent TreeSHAP.

Findings (`outputs/shap/SHAP.md`): popularity features carry about 43% of attribution in both models.
Logistic regression puts 31% on months since the last review, while LightGBM puts 21% on 3-month
popularity and 18% on CF. These are correlated ways of saying "recently active", and the two models split
the credit differently. Sign disagreements concentrate in correlated features (12-month popularity, item
age, rating change, good-for-kids), where the linear model's coefficients compensate for collinearity.
The models share 2 or 3 of their top-3 features for 79% of candidates. The choice between interventional
and path-dependent value functions barely matters for LightGBM (global rank correlation 0.983).

`python src/18_shap_explore.py` (~1.5 min) goes further (`outputs/shap/explore/EXPLORE.md`). Summing SHAP
within feature groups raises per-candidate agreement from 0.685 to 0.771, so much of the feature-level
disagreement is credit-splitting among correlated features. LightGBM's effects are mostly monotone; it
departs from linear for cuisine affinity and rating (linear R² 0.46 and 0.42), flattens distance between 4
and 12 km, and interacts popularity with CF, history size and distance. When a model misses a restaurant
the user went on to like, the score gap to its 10th recommendation is mainly collaborative for logistic
regression (44%) and popularity for LightGBM (44%); location explains 25% of logistic regression's
misses. Global importance is stable across background samples (rank correlation ≥ 0.993).

## Steering and change explanations (November 20, 2026)

```bash
python src/19_export_cases.py       # 24 steerable cases -> web/public/data/cases/ (+ full local copies)
node tests/replay.mjs               # browser code must reproduce the Python rankings exactly
python tests/check_steering.py      # each control moves the top 10 the intended way
```

`src/steering.py` (reference) and `web/src/steer.js` (browser) implement the same four controls over each
user's 200 candidates. History weight `w_h` scales the Collaborative group's SHAP contribution. The
distance (`alpha`) and popularity (`beta`) penalties apply to the min-max normalized score. MMR diversity
(`lam`) works over cuisine group and neighborhood. Rank changes are logged in the fixed stage order
model → distance → popularity → diversity, which telescopes to zero residual, and they are also
attributed with exact Shapley values over the four controls (16 rankings), which removes the
order-dependence. The replay test runs 336 settings (24 cases × 7 control settings × 2 models): 0
mismatches, fixed-order residual 0, Shapley residual 4e-14.

Each case holds restaurant facts at the 2020-08-01 cutoff, both models' raw scores, and SHAP per feature
and per group. Three cases are scripted by rule (a home-borough-bound Manhattan user, the most
popularity-dominated top 10, the largest disagreement between models); 21 are random. The shareable
`demo_*.json` files carry no user id and no visit history, and round the home point to about 1 km.
For the presentation they flag `later_liked` restaurants (reviewed with 4–5 stars during the test
period), as the team decided on 2026-11-20. Full copies stay in `data/clean/cases/`. In the sweep, diversity saturates at
`lam = 0.5` (10 cuisines and 10 neighborhoods in the top 10), and at `lam = 1` MMR ignores relevance, so the
interface maps its slider to `lam` in [0, 0.5].

## RQ1 steering trade-offs (November 20, 2026)

```bash
python src/20_rq1_sweep.py       # ~5 min: one-at-a-time sweep (8,025 users), joint grid + steering cost (2,000 users)
python src/21_rq1_frontier.py    # matched-outcome comparison and Pareto frontiers
```

Results are in `outputs/rq1/RQ1.md`. Every control costs relevance, at different rates. A moderate history
weight is almost free (w_h = 0.75 keeps 96–98% of NDCG@10), while removing the collaborative signal
(w_h = 0) loses about 40%. Halving mean distance (alpha = 0.25 for logistic regression, 0.5 for LightGBM)
keeps 86–87% of NDCG@10. Six cuisine groups in the top 10 cost 9–13%. Pushing popularity to the bottom
quartile costs 42–55%. Compared at matched outcomes, the models mostly trade off alike, with two
exceptions. Logistic regression keeps more relevance for very local (≤ 1 km: 82% vs 75%) and very niche
lists (popularity ≤ 0.25: 58% vs 45%). LightGBM, whose default list already reaches farther (3.9 km vs
2.6 km; 18% vs 8% outside the home borough), keeps far more when a quarter of the list must come from
other boroughs (64% vs 21%). Steering cost: about 70% of candidate neighborhoods in the user's home
borough can be brought into the top 10 on the grid, but only 33% (logistic regression) and 41% (LightGBM)
of neighborhoods in other boroughs. The LightGBM history-weight results use path-dependent SHAP; on the
300 step-17 users, interventional values give the same pattern (NDCG@10 at w_h = 0.5: 0.0414 vs 0.0420).

## Model lens: logistic regression vs LightGBM (November 21, 2026)

```bash
python src/22_model_lens.py      # ~1 min: both models on the same 200 candidates for all 8,025 test users
node tests/check_lens.mjs        # browser overlap, Kendall tau and SHAP group shares equal Python's
```

Results are in `outputs/model_lens/MODEL_LENS.md`. The two final models' top-10 lists share a median of 6
places (mean 6.12); 7.7% of users get lists sharing 3 or fewer. Over all 200 candidates, Kendall tau has a
median of 0.702 (IQR 0.654 to 0.738). Disagreement goes with lower accuracy for both models: mean NDCG@10 is
0.022–0.024 when the lists share 0–3 places and 0.057–0.059 when they share 7–10. A later-liked place is in
both top 10s for 14.4% of users, only LightGBM's for 4.2%, only logistic regression's for 3.8%. The interface
view (V6) places the selected user in these distributions and shows both models' SHAP side by side.

## Raw data (`data/raw`, not in git)

| File | Source | Size |
|---|---|---|
| `meta-New_York.json.gz` | Google Local 2021, McAuley Lab (UCSD): https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/ | 61 MB, 272,189 businesses |
| `review-New_York.json.gz` | same | 2.65 GB, 33,459,761 reviews |
| `nyc_inspections_by_restaurant.csv` | NYC Open Data, DOHMH Restaurant Inspection Results (`43nn-pn8j`), one row per CAMIS, fetched Oct 31, 2026 | 31,422 restaurants |
| `nyc_nta_2020.geojson` | NYC Open Data, 2020 Neighborhood Tabulation Areas (`9nt8-h7nd`) | 262 areas |

The Google data page asks users to cite Li, Shang & McAuley (ACL 2022) and Yan et al. (SIGIR 2023).
The inspection file was fetched with
`$select=camis,dba,boro,building,street,zipcode,phone,cuisine_description,latitude,longitude,max(inspection_date) as last_inspection`
grouped by all other selected fields. It lists only restaurants active in recent years.

## Processed data (`data/processed`, not in git)

| File | One row per | Columns |
|---|---|---|
| `restaurants_nyc.parquet` | NYC restaurant (24,857) | `gmap_id`, `name`, `address`, `latitude`, `longitude`, `categories`, `google_avg_rating`, `google_num_reviews`, `price_level` (1-4, missing for 31%), `google_state`, `nta2020`, `ntaname`, `boroname`, `permanently_closed_2021` |
| `reviews_nyc.parquet` | review (6,907,631) | `user_id`, `gmap_id`, `time_ms` (Unix ms), `rating` (1-5), `text`, `has_owner_response` |
| `inspection_matches.parquet` | matched restaurant (10,542) | `gmap_id`, `camis`, `name_sim`, `dist_m`, `same_number`, `same_zip`, `dba`, `building`, `street`, `zipcode`, `cuisine_description`, `last_inspection` |
| `sample_interactions.parquet` | first review of a restaurant by a sampled user (422,788) | `user_id`, `gmap_id`, `rating`, `time`, `split` (train / val / test) |

Definitions: a restaurant has a Google category containing "restaurant"; it is in NYC if its point lies in a 2020
NTA. The sample is 20,000 random users (seed 42) with at least 10 NYC restaurant reviews since 2017-01-01.
Validation starts 2019-12-01 and test 2020-08-01 (the last 10% of sample interactions).

## Outputs (`outputs`)

- `03_match_audit_sample.csv`: 100 random matches. `claude_check` is a first pass by an AI assistant (97 yes,
  1 no, 2 unsure); fill `correct` by hand before quoting the number.
- `03_unmatched_sample.csv`: 50 unmatched open restaurants with the nearest inspection record.
- `05_baselines.csv`: Recall@10 and NDCG@10 on validation and test from the first baseline run (8,449 test users);
  superseded by `logistic_ranker/metrics.csv`, where all models use the same 8,185 test users.
- `06_rating_over_time.csv`: per restaurant, all-time and last-12-months mean rating and the shift.

## Results (October 31, 2026)

- Matching: 42.4% of NYC restaurants match an inspection record (51.5% of those open at the 2021 crawl); matched
  restaurants hold 62% of reviews. Half of the unmatched open restaurants' addresses now hold another inspected
  business, so most misses are closures after 2021.
- Comparable evaluation (test, 8,185 eligible users), Recall@10 / NDCG@10: Item-CF 0.0277 / 0.0193;
  popularity 0.0123 / 0.0084; recent popularity 0.0149 / 0.0089; logistic regression 0.0484 / 0.0357.
  Validation selected C=1. These completed-run results supersede the earlier 8,449-user baseline cohort.
- Ratings over time: Google's displayed rating is within 0.2 stars of the all-time mean for 17,606 of 17,614
  restaurants with 50+ reviews; this does not show how Google computes its rating. For 5.5% of the 8,311
  restaurants with 50+ reviews in all and 20+ in the last 12 months of the data, the recent mean differs from the
  all-time mean by at least 0.5 stars (225 down, 229 up).

