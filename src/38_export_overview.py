#!/usr/bin/env python3
"""Step 38: the handful of dataset facts the dashboard's overview shows, counted from the data files.

Outputs: web/public/data/overview.json
Run: .venv/bin/python src/38_export_overview.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from common import INTERACTIONS_CLEAN, RESTAURANT_ATTRS, REVIEWS_CLEAN, ROOT, TEST_START, TEXT_CUTOFF, TEXT_SAMPLE, VAL_START

s = pd.read_parquet(INTERACTIONS_CLEAN, columns=["user_id", "split", "label"])
attrs = pd.read_parquet(RESTAURANT_ATTRS, columns=["gmap_id", "cuisine_group"])
out = dict(
    restaurants=int(len(attrs)), cuisine_groups=int(attrs.cuisine_group.nunique()),
    reviews=int(pq.ParquetFile(REVIEWS_CLEAN).metadata.num_rows),
    sample_users=int(s.user_id.nunique()), interactions=int(len(s)),
    interactions_by_split={k: int(v) for k, v in s.split.value_counts().items()},
    val_start=VAL_START, test_start=TEST_START, text_cutoff=TEXT_CUTOFF,
)
if TEXT_SAMPLE.exists():
    t = pd.read_parquet(TEXT_SAMPLE, columns=["gmap_id"])
    out |= dict(text_reviews=int(len(t)), text_restaurants=int(t.gmap_id.nunique()))
(ROOT / "web" / "public" / "data" / "overview.json").write_text(json.dumps(out, separators=(",", ":")))
print(json.dumps(out, indent=1))
