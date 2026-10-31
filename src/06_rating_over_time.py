"""Step 6: how much a restaurant's rating changes over time (all NYC reviews, not only the sample).

For each restaurant: all-time mean rating, mean over the last 12 months of the data, and Google's displayed
average from the metadata. "Shifted" restaurants have at least 50 reviews in all, at least 20 in the last
12 months, and a recent mean at least 0.5 stars away from the all-time mean.
Output: outputs/06_rating_over_time.csv (one row per restaurant)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS, RESTAURANTS_NYC, REVIEWS_NYC

MIN_TOTAL, MIN_RECENT, SHIFT = 50, 20, 0.5


def main():
    r = pd.read_parquet(REVIEWS_NYC, columns=["gmap_id", "time_ms", "rating"])
    r["time"] = pd.to_datetime(r.time_ms, unit="ms")
    end = r.time.max()
    recent_start = end - pd.DateOffset(months=12)
    allt = r.groupby("gmap_id").rating.agg(n_all="size", mean_all="mean")
    rec = r[r.time >= recent_start].groupby("gmap_id").rating.agg(n_recent="size", mean_recent="mean")
    meta = pd.read_parquet(RESTAURANTS_NYC, columns=["gmap_id", "name", "boroname", "price_level", "google_avg_rating"])
    t = meta.merge(allt, on="gmap_id").merge(rec, on="gmap_id", how="left")
    t["shift"] = t.mean_recent - t.mean_all
    t.to_csv(OUTPUTS / "06_rating_over_time.csv", index=False)

    eligible = t[(t.n_all >= MIN_TOTAL) & (t.n_recent >= MIN_RECENT)]
    shifted = eligible[eligible["shift"].abs() >= SHIFT]
    print(f"recent window: {recent_start.date()} to {end.date()}")
    print(f"restaurants with >= {MIN_TOTAL} reviews and >= {MIN_RECENT} in the last 12 months: {len(eligible):,}")
    print(f"recent mean >= {SHIFT} stars from all-time mean: {len(shifted):,} ({len(shifted) / len(eligible):.1%}); "
          f"down: {(shifted['shift'] < 0).sum():,}, up: {(shifted['shift'] > 0).sum():,}")
    with_google = t[t.n_all >= MIN_TOTAL].dropna(subset=["google_avg_rating"])
    gap = (with_google.google_avg_rating - with_google.mean_all).abs()
    print(f"Google displayed rating within 0.2 stars of our all-time mean: {(gap <= 0.2).mean():.1%} "
          f"of {len(with_google):,} restaurants with >= {MIN_TOTAL} reviews")
    print("examples:")
    print(shifted.sort_values("shift").head(3)[["name", "n_all", "mean_all", "n_recent", "mean_recent"]].round(2)
          .to_string(index=False))


if __name__ == "__main__":
    main()
