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

