# SteerEats: data preparation

Data preparation, baselines and first temporal analysis for the DS-GA 3001 project update (run of October 31, 2026).

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
- `05_baselines.csv`: Recall@10 and NDCG@10 on validation and test.
- `06_rating_over_time.csv`: per restaurant, all-time and last-12-months mean rating and the shift.

## Results (October 31, 2026)

- Matching: 42.4% of NYC restaurants match an inspection record (51.5% of those open at the 2021 crawl); matched
  restaurants hold 62% of reviews. Half of the unmatched open restaurants' addresses now hold another inspected
  business, so most misses are closures after 2021.
- Baselines (test, 8,449 users): item-CF Recall@10 0.026, NDCG@10 0.018; popularity 0.012 / 0.008; popularity
  over the last 12 months 0.014 / 0.008.
- Ratings over time: Google's displayed rating equals the all-time mean (within 0.2 stars for all 17,614
  restaurants with 50+ reviews). For 5.5% of the 8,311 restaurants with 20+ reviews in the last 12 months, the
  recent mean differs from the all-time mean by at least 0.5 stars (225 down, 229 up).
