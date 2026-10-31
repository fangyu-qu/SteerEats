"""Step 4: a small, reproducible recommender sample with a global time split.

- Reviews from 2017-01-01 on (earlier years are sparse and include placeholder dates such as 1990-12-31).
- One interaction per (user, restaurant): the first review, since the task is recommending new restaurants.
- Users with at least 10 NYC restaurant reviews in that window; a random 20,000 of them (seed 42).
- Global time split on the sample: the last 10% of interactions (by time, rounded to the month) are test,
  the 10% before that validation, so no future review is used to predict an earlier one.
Output: data/processed/sample_interactions.parquet (user_id, gmap_id, time, rating, split)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import PROCESSED, REVIEWS_NYC, SEED

START = pd.Timestamp("2017-01-01")
MIN_REVIEWS = 10
N_USERS = 20_000


def main():
    r = pd.read_parquet(REVIEWS_NYC, columns=["user_id", "gmap_id", "time_ms", "rating"])
    r["time"] = pd.to_datetime(r.time_ms, unit="ms")
    r = r[r.time >= START].sort_values("time").drop_duplicates(["user_id", "gmap_id"], keep="first")
    counts = r.groupby("user_id").size()
    core = counts[counts >= MIN_REVIEWS].index
    print(f"users with >= {MIN_REVIEWS} NYC restaurant reviews since {START.date()}: {len(core):,}")

    users = np.random.default_rng(SEED).choice(np.sort(core.to_numpy()), size=N_USERS, replace=False)
    s = r[r.user_id.isin(users)].drop(columns="time_ms").reset_index(drop=True)
    val_start = s.time.quantile(0.8).to_period("M").to_timestamp()
    test_start = s.time.quantile(0.9).to_period("M").to_timestamp()
    s["split"] = np.select([s.time >= test_start, s.time >= val_start], ["test", "val"], "train")
    s.to_parquet(PROCESSED / "sample_interactions.parquet", index=False)

    print(f"sample: {s.user_id.nunique():,} users, {s.gmap_id.nunique():,} restaurants, {len(s):,} interactions, "
          f"{s.time.min().date()} to {s.time.max().date()}")
    print(f"validation from {val_start.date()}, test from {test_start.date()}")
    print(s.split.value_counts().to_string())
    print(f"positive (4-5 stars) share: {(s.rating >= 4).mean():.1%}")


if __name__ == "__main__":
    main()
