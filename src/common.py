"""Paths and small helpers shared by the SteerEats data scripts."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

META = RAW / "meta-New_York.json.gz"
REVIEWS = RAW / "review-New_York.json.gz"
INSPECTIONS = RAW / "nyc_inspections_by_restaurant.csv"
NTA = RAW / "nyc_nta_2020.geojson"

RESTAURANTS_NYC = PROCESSED / "restaurants_nyc.parquet"
REVIEWS_NYC = PROCESSED / "reviews_nyc.parquet"
MATCHES = PROCESSED / "inspection_matches.parquet"

SEED = 42

# The three two-stage rankers, by the key used in the case files and in the browser.
RANKERS = {"lr": "Logistic regression", "lgbm": "LightGBM", "lmr": "LambdaRank"}


def normalize_name(name):
    """Lowercase, drop punctuation and filler words so 'Joe's Pizza, Inc.' and 'JOES PIZZA' compare equal."""
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"['’‘`]", "", s)   # Google often uses curly apostrophes: "Fabio’s" = "FABIO'S"
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    filler = {"the", "inc", "llc", "corp", "co", "ltd", "restaurant", "nyc", "ny", "and"}
    return " ".join(w for w in s.split() if w not in filler)

# Cleaned data (src/11_clean.py, src/12_restaurant_attributes.py); originals in data/processed stay untouched.
CLEAN = ROOT / "data" / "clean"
RESTAURANTS_CLEAN = CLEAN / "restaurants_clean.parquet"
REVIEWS_CLEAN = CLEAN / "reviews_clean.parquet"
INTERACTIONS_CLEAN = CLEAN / "interactions_clean.parquet"
RESTAURANT_ATTRS = CLEAN / "restaurant_attributes.parquet"
SAMPLE = PROCESSED / "sample_interactions.parquet"

# Time boundaries (UTC, as in src/04). Collection is already half-missing in May 2021, so rows from
# 2021-05-01 on form a "tail" split that is never evaluated.
VAL_START = "2019-12-01"
TEST_START = "2020-08-01"
TAIL_START = "2021-05-01"
LOCAL_TZ = "America/New_York"
BULK_DAY_MIN = 5   # a user posting >= this many NYC restaurant reviews on one local day

# Review text used as model input (bag-of-words and LLM aspect features) comes from before the earliest
# training cutoff, so the same text is safe for every training, validation and test window.
TEXT_CUTOFF = "2018-07-01"
TEXT_SAMPLE = CLEAN / "text_sample.parquet"
