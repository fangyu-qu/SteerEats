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

Python 3.13 (tested) with `pip install -r requirements.txt` (pandas, pyarrow, geopandas, scikit-learn, scipy, LightGBM, SHAP, umap-learn). From this folder:

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
LightGBM LambdaRank model (listwise; added after the proposal's pointwise comparison, and since November 26
the diner app's ranker and a third model in the Lab) reaches 0.0508 [0.0480, 0.0537], 5.6% above logistic regression
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

## Calibration and the evaluation panel (November 21, 2026)

```bash
python src/23_calibration.py         # ~3 min: Platt scaling, Brier, ECE, AUC, reliability bins
python src/24_export_evaluation.py   # gathers accuracy, paired tests, periods, seeds, calibration for the web panel
```

Results are in `outputs/calibration/CALIBRATION.md`. Rows are all 1,605,000 (test user, candidate) pairs; 8,708
are visits (base rate 0.54%). Trained on one visit per ten sampled candidates, the models' raw probabilities
average 5.5% (logistic regression) and 4.9% (LightGBM), 9–10 times the real rate, with negative Brier skill.
The final models were refit on train + validation, so Platt scaling is cross-fitted over 5 user folds on test.
Afterwards the mean prediction is 0.54% for both, ECE (15 equal-mass bins) is 1.1e-3 and 3.9e-4, and AUC is
unchanged (0.737 and 0.752); popularity reaches AUC 0.518. A constant base-rate prediction has ECE 2.6e-4,
lower than either model, which is why ECE is reported with AUC and Brier skill. The plan's protocol
(train-only models, Platt fitted on validation) under-predicts on test (mean 0.37% and 0.29%), a sign of the
temporal shift between the windows. Platt scaling is monotone per user, so Recall@10 and NDCG@10 do not change.

## Trade-off view (November 21, 2026)

```bash
python src/25_export_tradeoff.py   # joint-grid rows with paired 95% intervals for the web view; demo-case expectations
node tests/check_tradeoff.mjs      # browser grid outcomes, predicted visits and frontiers equal Python's
```

The view (V5) puts every setting of the RQ1 joint grid (192 per model) on two linked scatters: this user, where
relevance is the model's own predicted visits in the top 10 (Platt-calibrated, one fit per model on all test rows,
now saved by step 23 as `platt_all`), and the 2,000 grid users, where relevance is measured NDCG@10. Both show
the share kept relative to the model's own list, a Pareto staircase, and the current setting; clicking a dot loads
that setting. Per-user relevance is the model's estimate, not a measurement, because most users have 0 or 1
later-reviewed places among their candidates. Over the 2,000 users, no logistic-regression setting has an NDCG@10
change whose paired 95% interval includes zero; for LightGBM two do: beta = 0.25 (−0.0007 [−0.0043, +0.0027])
and alpha = beta = 0.25 (−0.0036 [−0.0073, +0.0002]) (`outputs/rq1/TRADEOFF.md`). Writing the browser check found
that step 21 kept settings tied on the steered quantity but lower in NDCG@10 (several variety settings reach the
cap of 10 cuisine groups); with the tie fixed, `frontiers.csv` lost 5 dominated rows and the matched-outcome table
is unchanged.

## Console previews (November 21, 2026)

```bash
python src/25_export_tradeoff.py --cases-only   # also writes tests/preview_expected.json
node tests/check_preview.mjs                    # browser previews equal src/steering.py's solver
```

Under each console control a small chart sweeps that control over its range, re-solves the outcome targets at
every value with the other controls held, and plots the share of the model's predicted visits for this user that
the top 10 keeps (the same Platt-calibrated estimate as the trade-off view; grey where the target cannot be met).
Hovering reads out the value and the list's distance, popularity and cuisine groups; clicking sets the control.
The summary line now states the share kept by the current list. The check covers 5,184 preview points (24 demo
cases, both models, two console states, four controls). Writing it exposed a parity bug: `steering.outcomes` used
Python's `sum()`, which since Python 3.12 compensates rounding, so a mean could land exactly on a target in Python
(0.3) and just above it in the browser (0.30000000000000004). It now sums left to right like the browser; the 288
saved solver results in `tests/replay_expected.json` are unchanged.

## Similarity map, case studies and user study (November 21, 2026)

```bash
python src/26_similarity_map.py   # ~2 min: SVD item factors, UMAP / PCA, trustworthiness, continuity, k-NN, sweep figure
python src/27_case_studies.py     # the three scripted case studies (seconds)
python src/28_user_study.py --tasks   # study tasks with computed answers (~6 min); without --tasks: analyse sessions
```

**Similarity map (S1).** Restaurants placed by who reviewed them: 4,871,411 reviews before 2020-08-01 by 813,681
people of 19,812 restaurants (at least 20 reviewers), 64-dimensional SVD, then UMAP with PCA beside it
(`outputs/similarity_map/SIMILARITY_MAP.md`). A restaurant's 15 nearest neighbours share its borough 92.0% of
the time (chance 41.9%) but its cuisine group only 26.3% (chance 16.6%): co-reviewing follows place more than
cuisine. UMAP keeps local structure (trustworthiness 0.919, continuity 0.945, the same with a second seed; PCA
0.820 and 0.917; UMAP on same-size noise 0.530 and 0.561). The most distinct cuisine communities are
jewish/kosher, African, Korean and Caribbean (neighbour purity 31, 14, 14 and 12 times their share). Brushing
the projection shows those candidates on the map.

**Case studies** (`outputs/case_studies/CASE_STUDIES.md`, each with a deep link). Manhattan regular (demo_01):
the four controls reach 4 of the 23 unreached other-borough neighbourhoods on the grid, the boost reaches all 23
(Park Slope with +0.02 for LightGBM, +0.08 for logistic regression). Popular-places fan (demo_02): both models
meet "popularity at most 0.40" keeping 74% of predicted visits, but LightGBM's list moves from 3.5 to 12.4 km
away (logistic regression 0.6 to 2.7 km). Models disagree (demo_03): the own lists share 2 of 10 places;
"within 2 km" needs α 0.12 for logistic regression (91% kept) and 0.44 for LightGBM (69% kept).

**User study** (`docs/USER_STUDY.md`). Ready to run, not yet run: `#study=P1` opens study mode, which
counterbalances the full interface against a list-only baseline, times and scores three tasks per condition
against computed answers, and exports a results file. No results exist until sessions are held.

## Restaurant detail (November 22, 2026)

```bash
python src/29_export_restaurant_detail.py   # seconds; needs data/clean (steps 11 and 12); also writes tests/detail_expected.json
node tests/check_detail.mjs                 # browser rating history equals the export and the rankers' snapshot
```

The detail view (V8) shows the selected restaurant's star ratings by month: the mean over the past 3, 6 or 12
months with a 95% band, the reviews per month, the star mix and the price level (shown as "not listed" when
Google has none: 15.4% of the 3,005 restaurants in the demo cases). A line marks the 2020-08-01 snapshot. Left
of it is what the models could see; right of it are the reviews that arrived later (to April 2021), which no
model used. The view also states the two smoothed ratings the rankers used and how far the rating features move
the selected model's score, because stars are an input to the models, not their score.

The export holds counts per month and star level only (24 case files, 4.03 MB; no user ids, no review text).
For all 4,800 candidates the counts reproduce the case files: review counts exactly, the rankers' smoothed
ratings within 1e-6. The check also compares 2,160 rolling-window values between the browser and Python. The
band is the mean ± 1.96 standard errors, clipped to 1–5 stars. The window's variance is pooled with the
city-wide variance (SD 1.1646) at the weight of two reviews, so a few identical ratings do not give a band of
zero width.

What the data says (`outputs/restaurant_detail/DETAIL.md`): a restaurant has a median of 408 reviews before the
snapshot and 101 in the last 12 months; 14.3% have fewer than 20 in those 12 months, where the band is wide.
Ratings are stable before the snapshot: 22 of 2,574 restaurants (0.9%) have a 12-month mean at least 0.5 stars
from their all-time mean. They move more after it: 113 of 1,975 (5.7%; 75 down, 38 up). The largest drops are
three Chipotle branches (1.29 to 1.63 stars).

The lower half of the view, the theme summaries of review text, is the next section.

## Theme summaries of review text (November 22, 2026)

```bash
python src/30_theme_summaries.py --check      # free: do the API key and the connection work?
python src/30_theme_summaries.py --pilot 20   # 20 restaurants, one call each; prints tokens and cost
python src/30_theme_summaries.py --submit     # the rest as one message batch (half price); then --collect --wait
python src/30_theme_summaries.py --export     # check the answers, write web/public/data/themes/ (no model call)
python src/30_theme_summaries.py --compare    # the model pilot (outputs/theme_summaries/MODELS.md)
node tests/check_themes.mjs                   # published summaries have the shape the view reads
```

What do reviewers write about a place? For each candidate restaurant of the demo cases a Claude model
(`claude-sonnet-5-5`) read up to 60 text reviews from before the 2020-08-01 snapshot, drawn with a fixed seed to
keep the restaurant's mix of star ratings, and returned two to five themes, each with the reviews that mention
it and one or two quotes. Under the rating history, the detail view (V8) shows for every theme the share of
sampled reviews the model tagged, the mean rating of those reviews against the whole sample, and the quotes
with their star rating and month. The calls need an Anthropic API key (`ANTHROPIC_API_KEY`). Model answers are
kept in `data/llm_themes` (not in git), so `--export` and the site never call the model. No user ids or
reviewer names are sent to the model.

Since November 26 the Lab's Explorer shows these themes in its last card, "What reviewers say about this place", for
the restaurant selected in the list, the map or the rank chart (`web/src/views/reviews.js`); the rankers do not read them.

Results (`outputs/theme_summaries/THEMES.md`): 2,804 of the 3,005 restaurants are summarised; 201 have fewer
than 10 text reviews of 30 or more characters and get no summary. The model read 154,837 of these restaurants'
843,909 text reviews and returned 13,834 themes with 27,197 quotes. The script looks every quote up in the
review it cites: 31 were not in any sampled review, 204 were too short or too long, 1 held a contact detail and
157 matched the script's patterns for a person's name (which also catch some dish names). These were dropped;
26,772 quotes are published. 16 themes were dropped because the label or summary named a person and 13 because
no quote passed; 13,805 are published (4.9 per restaurant). Themes the model called negative come from reviews
that average 2.59 stars, mixed 3.81 and positive 4.55.

Not checked by a script, and stated in the view: whether a summary sentence is a fair reading of the reviews,
and whether every tagged review really mentions the theme.

Model choice (`outputs/theme_summaries/MODELS.md`): the same 20 restaurants went to three models. All 167
quotes of `claude-haiku-5-5`, 196 of 197 of `claude-sonnet-5-5` and all 197 of `claude-opus-5-5` were in the
review they cite; at list price 1,000 restaurants cost $2, $19 and $56. The published run used 13,122,361
input and 2,468,279 output tokens.

## Groups, cold start, retrievers, text features and the dashboard (November 25, 2026)

```bash
python src/33_rq2_groups.py            # RQ2: 1,000 three-member groups per kind, three strategies, three models (~10 min)
python src/32_cold_start.py            # NDCG@10 from only the k most recent liked places (~10 min)
python src/37_retriever_ablation.py    # what each stage-1 retriever adds to candidate recall (~2 min)
python src/34_text_sample.py           # review text before 2018-07-01: fixed sample + bag-of-words features
python src/36_llm_ablation.py --seed 42   # base vs text features, same rows and users (run seeds 42, 1, 2; ~10 min each)
python src/36_llm_ablation.py --report    # outputs/llm_features/ABLATION.md + web/public/data/text_ablation.json
python src/35_llm_aspects.py --plan       # LLM aspect features: requests and estimated cost, no API call
python src/39_lambdarank_shap.py       # SHAP for LambdaRank on the rows and background of step 17
python src/38_export_overview.py       # dataset facts for the dashboard's overview
python tests/check_group.py && node tests/check_group.mjs   # web/src/group.js = src/group.py
```

Reports: `outputs/rq2/RQ2.md`, `outputs/cold_start/COLD_START.md`, `outputs/retrievers/RETRIEVERS.md`,
`outputs/llm_features/TEXT_SAMPLE.md`, `ABLATION.md` and `LLM_PLAN.md`, `outputs/shap/LAMBDARANK.md`.

- **Group engine** (`src/group.py`, mirrored by `web/src/group.js`): member satisfaction is the member's score,
  min-max over the pool; strategies are average, least misery, and average without misery (threshold 0.40).
- **Text features (RQ3)** use only review text from before 2018-07-01, the earliest training cutoff, so the
  same text is safe for every window (`tests/check_time_safety.py` checks it). Seed 42 with the base features
  reproduces step 15 exactly. The LLM aspect features (`src/35`, `claude-haiku-5-5`, one message batch; needs
  `pip install anthropic` and an API key in the environment) are scored for 18,085 restaurants; answers stay in
  `data/llm_aspects` (not in git). `ABLATION.md` compares base, bag-of-words and LLM features over three seeds.
- **Dashboard** (`web/`): three sections. *Overview* shows one chart and one sentence per question, with the
  numbers one click away; *Explorer* is the steering interface for one test user; *Analysis* holds the
  trade-off space, similarity map, model lens and evaluation. Colours come from the team palette
  (`web/src/styles.css`); chart marks carry an ink outline and a second encoding because the palette alone
  does not pass the colour checks. The interface is named Eat Buddy; the repository and method stay SteerEats.
- The food stickers in `web/public/illustrations/` are AI-generated images (Magic Patterns), not drawings.

## One-sentence steering (November 25, 2026)

> November 26, 2026: the **Say what you want** box was removed from the Lab's Explorer at Yila's request. The diner
> app keeps its sentence box, and the reader (`server/nl.py`, `web/src/nl.js`) and its evaluation are unchanged. What
> follows describes the Explorer as it was.

Type a sentence in the Explorer's **Say what you want** box ("cheap sushi within walking distance"). A language
model reads it into the app's own settings; the ranking model still chooses the places.

- **What a sentence can set.** Targets the solver already had: distance (`max_km`), less mainstream, cuisine
  variety, weight of past visits. Filters, which are new: cuisine groups, price from/up to, needs (vegetarian
  options, halal, open late, alcohol, reservations, groups, kids), borough, neighbourhood. Wishes with no setting
  ("quiet", "outdoor seating") are listed as such and change nothing.
- **Confirm first.** What was understood is shown as removable chips; nothing changes until **Apply**. A setting
  the sentence did not mention stays as it was.
- **Filters narrow the pool.** A filter keeps the candidates that pass it; steering, the ledger and every view
  then run on that pool unchanged (`filterCase` in `web/src/nl.js`). The pool can be small: a user's 200
  candidates hold a median of 4.5 Japanese places, and for half of the demo users no Korean one. The app says how
  many are left and never adds a place. Places with no known price fail a price filter.
- **The model only reads.** `server/nl.py` sends the sentence alone (no user id, location or candidates) to
  `claude-haiku-5-5` with a JSON schema; the answer can only hold known keys and values (`normalize`, the same
  in Python and in the browser). A sentence that is an instruction to the model is marked as not about eating.
- **Without the API** (the GitHub Pages site) keyword rules in `web/src/nl.js` read the sentence, English only,
  and the box says so.

Run the API (the key comes from the environment, never from a file here):

```
export ANTHROPIC_API_KEY=...            # once, in your shell profile
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8787
cd web && npm run dev                   # the dev site proxies /api to port 8787
```

Checks and evaluation:

```
.venv/bin/python tests/check_nl.py        # browser = Python: vocabularies, normalize, filters, filtered rankings; no model call
.venv/bin/python src/40_nl_eval.py --plan # cost of scoring the 64 sentences of tests/nl_cases.json (about $0.02)
.venv/bin/python src/40_nl_eval.py --run  # run on 2026-11-25 with Yila's go-ahead; --report writes outputs/nl_steering/NL_EVAL.md
```

Measured while building it (five calls): about 2,340 input and 90 output tokens per sentence, $0.0003, 2 to 3
seconds; the first sentence after the API starts can take much longer. `src/steering.py` and `web/src/steer.js`
changed together in one line (`top_k` now stops at the pool size); the replay checks stay at 0 failures.

## The diner app (November 26, 2026)

`web/index.html` is now the app; the dashboard and explorer moved to `web/lab.html` (links made earlier with
`#case=` or `#study=` are forwarded there).

- **What it does.** One or two people each borrow a sample taste (one of the 24 demo users, without ids), choose
  where they start from, and get six places. Each card says in plain words why the place suits each person,
  shows both "suits you" numbers on one line, and each person's distance. "See why" opens the reasons, the
  numbers behind them, the place's stars over time, what reviewers wrote and a comparison with another place.
- **The model.** Scores come from LightGBM LambdaRank, the ranker with the highest NDCG@10 on the test period
  (`web/public/data/evaluation.json`). `src/41_export_app.py` scores, for every sample taste, the union of all
  24 candidate sets and explains each score with interventional SHAP on the step-17 background.
- **The ranking** (`web/src/app/engine.js`, mirrored by `src/app_engine.py`): pool = union of the members'
  candidates minus filtered places; per member the model score with the history weight, min-max over the pool,
  minus the distance and popularity penalties; "suits you" = that score min-max over the pool; combined by
  "Fair to everyone" (average without misery, bar 0.40) or "Nobody unhappy" (least misery) with
  `web/src/group.js`; variety by MMR on the group score. "Closer", "Less mainstream" and "More variety" are
  outcomes of the list, and the solver finds the smallest weights that reach them.
- **Say it.** The sentence box uses the same `/api/parse` and `web/src/nl.js` as the Explorer: chips to confirm,
  then targets and filters. Without the API, keyword rules read it.
- **What is not there yet.** Rooms across phones, a third to fifth person (their avatars are not made),
  street-level map tiles. A taste from your own liked places was added later the same day (next section).

```
.venv/bin/python src/41_export_app.py     # app data (about 15 minutes; needs the local data)
.venv/bin/python tests/check_app.py       # browser engine = Python reference, exported files hold no ids
cd web && npm run dev                     # http://localhost:5173/ is the app, /lab.html the Lab
```

One-sentence evaluation (`outputs/nl_steering/NL_EVAL.md`): on 64 sentences the language model reads 56
completely right (88%, Wilson interval 77% to 94%), keyword rules 45 (70%, 58% to 80%). The lead is in Chinese
sentences (6 against 0 of 8) and sentences that are not about eating (6 against 0 of 6). The sentences and the
expected answers were written by the same authors as the prompt, with AI assistance; cost $0.018.

## Live profiles and LambdaRank in the Lab (November 26, 2026)

**A taste of your own.** In the app a person can now choose "Places I like" instead of a sample taste: search a
restaurant by name, pick at least 3, and the server builds the taste.

- `src/31_export_serving_bundle.py` saves what one snapshot (2020-08-01) holds: restaurant arrays, item
  statistics, the item-item similarity, the LambdaRank model and the SHAP background (`server/bundle/`, 10 MB,
  git-ignored). `server/live.py` hands that bundle to the unchanged functions of `src/ranker_core.py`
  (`profile_of`, `features`, `candidates`), so a new profile gets the features a reviewer with exactly those
  liked places would get. This is the profile the cold-start study measured (`outputs/cold_start/`): with
  LambdaRank, 3 liked places keep 62% of the full-history NDCG@10, 5 keep 72%, 10 keep 87%. The app shows these
  numbers next to the picker.
- `POST /api/search {"q"}` finds restaurants by name. `POST /api/party {"members": [...]}` takes one or two
  members, each a sample taste or a list of picked places, and returns the union of their candidates with each
  member's score, collaborative part and six SHAP groups, in the shape of the exported app files. The browser's
  engine then runs as it does on the exported files.
- `tests/check_serving.py`: for each of the 24 sample tastes, and for 8 pairs, the server's candidates, scores
  and SHAP parts equal the exported files (largest difference 0), and the app engine lists the same six places
  from either source. Over three runs on a laptop a new taste answered in 0.6 to 1.1 seconds and a party of two in 2.5 to 3.8. SHAP
  parts add up to the score except on rare rows (1 of 600 in the check, 5 of 60,000 in step 39), where
  interventional TreeSHAP leaves a remainder of up to 0.03; the app then shows the remainder as its own line.
- Privacy: the picked places travel in the request body, are scored and forgotten (no log, no cache); the list
  stays in the person's browser. `server/bundle/profiles.json` holds the sample reviewers' visit lists without
  ids; the server uses it to score a sample taste in a live party and never returns it.
- A place one member picked can still be listed when it is among the other member's candidates, as in the
  offline group study; the card says so.

**One server.** `server/app.py` also serves the built site, so one process is the whole app:

```
cd web && npm run build && cd ..
.venv/bin/python src/31_export_serving_bundle.py     # once, after step 41 (10 seconds with the data on disk)
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8787    # then open http://127.0.0.1:8787/
.venv/bin/python tests/check_serving.py
```

With `--host 0.0.0.0` phones on the same Wi-Fi can open `http://<this computer's address>:8787/`. That also
opens the API, including the language-model proxy, to everyone on that network, and "My location" needs
HTTPS, so on a phone choose a neighbourhood or move the pin. A site without the server (GitHub Pages) asks
`api/health`, gets no answer, and offers sample tastes and keyword rules only.

**LambdaRank in the Lab.** The Explorer's model switch has three models (`lr`, `lgbm`, `lmr`), and the console
note, the trade-off frontier and the model lens compare the model in view with another one you choose.

- Case files (`src/19`): LambdaRank's raw score, interventional SHAP per feature and group, and the
  collaborative split, for the same 24 users (LambdaRank plays no part in choosing them). Its values equal the
  app's export for the same users and places (largest difference 0).
- Calibration (`src/23`): LambdaRank's score is a ranking score, not log-odds, so it has no "raw probability"
  row. After Platt scaling its Brier skill is 0.0074 (LightGBM 0.0074, logistic regression 0.0033), ECE 0.00023
  and AUC 0.749, so "predicted visits" is used for it as for the other two.
- Sweep (`src/20`, 8,025 users): with no steering LambdaRank's NDCG@10 is 0.0508, as in the evaluation. Matched
  outcomes on the 2,000-user grid (`src/21`): a list within 2 km on average keeps 87% of its NDCG@10 (LightGBM
  87%), popularity percentile at most 0.40 keeps 70% (72%), at least 6 cuisine groups keeps 86% (91%).
  Its bootstrap draws come from their own generator, so every earlier interval and the 2,000-user subset are
  unchanged: the result files only gained rows.
- Model lens (`src/22`, 8,025 users): LightGBM and LambdaRank share 7.56 of their 10 top places on average
  (median 8; Kendall tau over all candidates 0.84). Logistic regression and LambdaRank share 6.38 (tau 0.69);
  logistic regression and LightGBM 6.12 (tau 0.70), as reported before.
- Checks with three models: replay 648 lists and 576 solver runs, 43,200 ledgers, the lens for three pairs per
  case, trade-off and preview expectations; all at 0 failures.
- The Overview's steering panel still shows LightGBM, and the user-study tasks and case studies stay on the
  models they were written for.

## The Lab, redesigned (November 26, 2026)

The Lab (`web/lab.html`) was cut from about 26 charts to 12 and restyled: frosted cards on a coral canvas, six
mark colours (terracotta, mist blue, teal, grape, forest, mustard), side navigation, type in Urbanist. The
palette and its measurements are in `docs/PLAN.md` §2.4. The ranker chart keeps its rows (name on the left, value and
interval on the right) and draws each interval as a shaded bar, as a newspaper poll chart does, with a solid
circle on the estimate. Its axis starts at 0.040, which is stated on the chart; the two popularity baselines score about a
twentieth as much and are marked at the left edge with an arrow; their values stay in the right column.

| Section | What it shows |
|---|---|
| Overview | Five fact tiles; which ranker is most accurate (poll-style chart with 95% intervals); what steering costs (192 settings, shaded by cuisine groups); whether protecting the least satisfied member of a group lowers recommendation quality (one row per rule in each kind of group; the least satisfied member's satisfaction and member NDCG@10 side by side, each with its 95% bootstrap interval over 1,000 groups; satisfaction is rescaled within each group's pool, so it is not compared across models). Text features, cold start, retrievers and one-sentence steering are not shown in the Lab (their "More results" rows were removed on November 26); their results are in the sections above and in `outputs/` |
| App | Since November 26 the app's own map is the same kind of street map: it zooms from the whole city down to street and building names, with numbered pins, each person's avatar at their start and a dashed line to the picked place. All buttons in the app and the Lab are rectangles |
| Explorer | Starts on LambdaRank, the model the app uses. Its map is a street map since November 26 (Leaflet on OpenStreetMap tiles, numbered bubbles, zoom buttons and "Reset view"; the tiles are fetched from tile.openstreetmap.org, so the map needs a connection and that server sees the visitor's IP address and the area in view). For one test user: how the list changed step by step, where it went on the map with the top 10, and why a place has its score (SHAP groups and the user's steering). The small previews under the controls are drawn in study mode only (their checkbox was removed on November 26) |
| Analysis | Three cards since the evening of November 26: what each of the three rankers relies on (share of mean absolute SHAP by feature group over 300 random test users, with the chosen user marked); reliability diagrams for the three rankers, the table behind "Numbers"; and the UMAP of the restaurants drawn twice, one borough and one cuisine group picked out, with its trustworthiness. The pairwise lists, the SHAP beeswarm and the per-user steering frontier were removed (the frontier over 2,000 users is on the Overview) |

Rules followed, from the course assignments: one colour and one shape per ranker in every chart; quantities
in one hue from light to dark; no category by colour alone; palettes measured for colour-blind separation and
contrast on the surface they sit on; a rate or a share always with its number of users; a projection always
with what it keeps and what it distorts.

Left out of the page, still in the reports: paired differences, accuracy by test period and across sampling
seeds (`outputs/rankers_two_stage/EVALUATION.md`), the population histograms of list overlap and the per-group
attribution scatter (`outputs/model_lens/MODEL_LENS.md`), the PCA twin of the restaurant map
(`outputs/similarity_map/`), and a place's ratings over time (in the app's place drawer).

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

