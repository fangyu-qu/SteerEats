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


def normalize_name(name):
    """Lowercase, drop punctuation and filler words so 'Joe's Pizza, Inc.' and 'JOES PIZZA' compare equal."""
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"['’‘`]", "", s)   # Google often uses curly apostrophes: "Fabio’s" = "FABIO'S"
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    filler = {"the", "inc", "llc", "corp", "co", "ltd", "restaurant", "nyc", "ny", "and"}
    return " ".join(w for w in s.split() if w not in filler)
