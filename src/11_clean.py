#!/usr/bin/env python3
"""Step 11: clean restaurants, reviews and the interaction sample. Writes to data/clean/;
the files in data/processed are read only and stay unchanged.

Restaurants
  - Drop businesses whose only "restaurant" categories are non-food ones
    ("Restaurant supply store", "Bar restaurant furniture store").
  - Flag listings that share a name and address with another listing (kept; too few to merge safely).
Reviews
  - Drop rows with no user id and no rating (2,033 in the October 2026 extract).
  - Drop exact duplicates of (user, restaurant, time, rating), keeping the first.
  - Drop placeholder dates before 2000 (1990-12-31 and a few others).
  - Drop reviews of removed restaurants.
  - Add New York local date, hour and weekday (timestamps are UTC), and how many NYC restaurant
    reviews the user posted that local day (bulk uploads make posting time != visit time).
  - Flag cross-posted reviews: the same user at the same millisecond on two or more restaurants.
    Most are one review copied onto two listings of the same business. Listing pairs that share
    such reviews from >= 3 users are written to outputs/cleaning/linked_listings.csv for review.
Interactions
  - Rebuild the src/04 sample from clean reviews for the same 20,000 users: reviews since
    2017-01-01, first review per (user, restaurant). Checked against the original sample.
  - Same split boundaries as src/04 (validation 2019-12-01, test 2020-08-01), plus a "tail"
    split from 2021-06-01, when collection fades out (Jun 8,333 / Jul 4,292 / Aug 1,838 / Sep 87).
  - Label: 1 for 4-5 stars, 0 for 1-2 stars, missing for 3 stars.
Outputs: data/clean/{restaurants,reviews,interactions}_clean.parquet, outputs/cleaning/CLEANING_LOG.md
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from common import (BULK_DAY_MIN, CLEAN, INTERACTIONS_CLEAN, LOCAL_TZ, OUTPUTS, RESTAURANTS_CLEAN,
                    RESTAURANTS_NYC, REVIEWS_CLEAN, REVIEWS_NYC, SAMPLE, TAIL_START, TEST_START,
                    VAL_START)

NON_FOOD = re.compile(r"^(restaurant supply store|bar restaurant furniture store)$", re.I)
START = pd.Timestamp("2017-01-01")
PLACEHOLDER_BEFORE = pd.Timestamp("2000-01-01")
LOG = OUTPUTS / "cleaning"


def clean_restaurants(log):
    R = pd.read_parquet(RESTAURANTS_NYC)
    cats = R.categories.fillna("").str.split("; ")
    food = cats.apply(lambda cs: any("restaurant" in c.lower() and not NON_FOOD.match(c) for c in cs))
    removed = R[~food]
    R = R[food].copy()
    key = R.name.str.lower().str.strip() + "|" + R.address.fillna("").str.lower().str.strip()
    R["duplicate_listing"] = key.duplicated(keep=False) & R.address.notna()
    log["restaurants"] = dict(input=int(len(food)), removed_non_food=int(len(removed)),
                              kept=int(len(R)), duplicate_listings_flagged=int(R.duplicate_listing.sum()),
                              removed_examples=removed.categories.head(5).tolist())
    R.to_parquet(RESTAURANTS_CLEAN, index=False)
    return R


def clean_reviews(keep_ids, log):
    t = pq.read_table(REVIEWS_NYC)
    k = t.select(["user_id", "gmap_id", "time_ms", "rating"]).to_pandas()
    n0 = len(k)
    time = pd.to_datetime(k.time_ms, unit="ms")
    empty = (k.user_id.isna() | k.rating.isna()).to_numpy()
    dup = k.duplicated(["user_id", "gmap_id", "time_ms", "rating"]).to_numpy() & ~empty
    placeholder = (time < PLACEHOLDER_BEFORE).to_numpy() & ~empty & ~dup
    gone = (~k.gmap_id.isin(keep_ids)).to_numpy() & ~empty & ~dup & ~placeholder
    keep = ~(empty | dup | placeholder | gone)
    log["reviews"] = dict(input=n0, missing_user_or_rating=int(empty.sum()), exact_duplicates=int(dup.sum()),
                          placeholder_dates=int(placeholder.sum()), non_food_restaurants=int(gone.sum()),
                          kept=int(keep.sum()))

    k, time = k[keep].reset_index(drop=True), time[keep].reset_index(drop=True)
    local = time.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)
    local_date = local.dt.tz_localize(None).dt.normalize()
    day_count = k.groupby([k.user_id, local_date]).user_id.transform("size")
    cross = k.duplicated(["user_id", "time_ms"], keep=False).to_numpy()
    link_pairs(k[cross], log)
    t = t.filter(pa.array(keep))
    t = (t.append_column("time", pa.array(time, pa.timestamp("ms")))
          .append_column("local_date", pa.array(local_date.dt.date, pa.date32()))
          .append_column("local_hour", pa.array(local.dt.hour.astype("int8")))
          .append_column("local_dow", pa.array(local.dt.day_of_week.astype("int8")))
          .append_column("user_day_reviews", pa.array(day_count.astype("int32")))
          .append_column("cross_posted", pa.array(cross)))
    pq.write_table(t, REVIEWS_CLEAN)
    log["reviews"]["bulk_day_reviews"] = int((day_count >= BULK_DAY_MIN).sum())
    log["reviews"]["cross_posted_reviews"] = int(cross.sum())
    cols = ["user_id", "gmap_id", "time", "rating", "local_date", "user_day_reviews", "cross_posted"]
    return t.select(cols).to_pandas()


def link_pairs(x, log, min_users=3):
    """Restaurant pairs that receive the same user's review at the same millisecond."""
    m = x.merge(x, on=["user_id", "time_ms"], suffixes=("_a", "_b"))
    m = m[m.gmap_id_a < m.gmap_id_b]
    pairs = (m.groupby(["gmap_id_a", "gmap_id_b"]).user_id.nunique().rename("shared_users")
              .reset_index().sort_values("shared_users", ascending=False))
    R = pd.read_parquet(RESTAURANTS_NYC, columns=["gmap_id", "name", "address"])
    for side in "ab":
        pairs = pairs.merge(R.add_suffix(f"_{side}"), on=f"gmap_id_{side}", how="left")
    strong = pairs[pairs.shared_users >= min_users]
    LOG.mkdir(parents=True, exist_ok=True)
    strong.to_csv(LOG / "linked_listings.csv", index=False)
    log["linked_listings"] = dict(pairs_any=int(len(pairs)), pairs_min_users=int(len(strong)), min_users=min_users)


def rebuild_sample(r, removed_ids, log):
    orig = pd.read_parquet(SAMPLE)
    users = set(orig.user_id)
    s = r[(r.time >= START) & r.user_id.isin(users)].sort_values("time", kind="stable")
    s = s.drop_duplicates(["user_id", "gmap_id"], keep="first").reset_index(drop=True)

    # Check: the original sample minus removed restaurants should be reproduced exactly.
    o = orig[~orig.gmap_id.isin(removed_ids)]
    key_o = set(zip(o.user_id, o.gmap_id, o.time.to_numpy().astype("datetime64[ms]").astype("int64")))
    key_s = set(zip(s.user_id, s.gmap_id, s.time.to_numpy().astype("datetime64[ms]").astype("int64")))
    log["interactions_check"] = dict(original=int(len(orig)), original_on_removed_restaurants=int(len(orig) - len(o)),
                                     rebuilt=int(len(s)), only_in_original=len(key_o - key_s),
                                     only_in_rebuilt=len(key_s - key_o))

    s["split"] = np.select([s.time >= TAIL_START, s.time >= TEST_START, s.time >= VAL_START],
                           ["tail", "test", "val"], "train")
    s["label"] = np.where(s.rating >= 4, 1.0, np.where(s.rating <= 2, 0.0, np.nan))
    s["bulk_day"] = s.user_day_reviews >= BULK_DAY_MIN
    s = s[["user_id", "gmap_id", "time", "local_date", "rating", "label", "split", "bulk_day",
           "user_day_reviews", "cross_posted"]]
    s.to_parquet(INTERACTIONS_CLEAN, index=False)

    per_user = s.groupby("user_id").size()
    log["interactions"] = dict(
        rows=int(len(s)), users=int(s.user_id.nunique()), restaurants=int(s.gmap_id.nunique()),
        users_below_10=int((per_user < 10).sum()),
        splits={k: dict(rows=int(v), start=str(s[s.split == k].time.min()), end=str(s[s.split == k].time.max()))
                for k, v in s.split.value_counts().items()},
        label_share={"positive": float((s.label == 1).mean()), "negative": float((s.label == 0).mean()),
                     "three_star": float(s.label.isna().mean())},
        bulk_day_share=float(s.bulk_day.mean()), cross_posted=int(s.cross_posted.sum()))


def write_log(log):
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "cleaning_log.json").write_text(json.dumps(log, indent=2))
    R, V, I, C = log["restaurants"], log["reviews"], log["interactions"], log["interactions_check"]
    lines = ["# Cleaning log", "", "Generated by `src/11_clean.py`. Originals in `data/processed` are unchanged.", "",
             "## Restaurants", "",
             f"- Input {R['input']:,}; removed {R['removed_non_food']} non-food businesses; kept {R['kept']:,}.",
             f"- Listings sharing a name and address, flagged `duplicate_listing`: {R['duplicate_listings_flagged']}.", "",
             "## Reviews", "",
             f"- Input {V['input']:,}.",
             f"- Removed: {V['missing_user_or_rating']:,} rows with no user or rating, {V['exact_duplicates']:,} exact duplicates, {V['placeholder_dates']:,} placeholder dates, "
             f"{V['non_food_restaurants']:,} reviews of removed businesses.",
             f"- Kept {V['kept']:,}; {V['bulk_day_reviews']:,} posted on a day the user posted >= {BULK_DAY_MIN} reviews.",
             f"- Cross-posted (same user, same millisecond, several restaurants), flagged not removed: {V['cross_posted_reviews']:,}.",
             f"- Listing pairs sharing cross-posted reviews from >= {log['linked_listings']['min_users']} users: "
             f"{log['linked_listings']['pairs_min_users']} (of {log['linked_listings']['pairs_any']:,} pairs), "
             "in `linked_listings.csv`; likely the same business listed twice.", "",
             "## Interactions", "",
             f"- Rebuilt for the same users: {I['rows']:,} rows, {I['users']:,} users, {I['restaurants']:,} restaurants.",
             f"- Check against the original sample ({C['original']:,} rows, {C['original_on_removed_restaurants']} on removed "
             f"businesses): {C['only_in_original']} rows only in the original, {C['only_in_rebuilt']} only in the rebuild.",
             f"- Users now below 10 interactions: {I['users_below_10']}.",
             "- Splits: " + "; ".join(f"{k} {v['rows']:,} ({v['start'][:10]} to {v['end'][:10]})" for k, v in I["splits"].items()) + ".",
             f"- Labels: positive {I['label_share']['positive']:.1%}, negative {I['label_share']['negative']:.1%}, "
             f"three-star (unlabeled) {I['label_share']['three_star']:.1%}.",
             f"- Interactions on bulk days: {I['bulk_day_share']:.1%}; cross-posted: {I['cross_posted']:,}."]
    (LOG / "CLEANING_LOG.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    log = {}
    before = set(pd.read_parquet(RESTAURANTS_NYC, columns=["gmap_id"]).gmap_id)
    R = clean_restaurants(log)
    print("restaurants done", flush=True)
    r = clean_reviews(set(R.gmap_id), log)
    print("reviews done", flush=True)
    rebuild_sample(r, before - set(R.gmap_id), log)
    write_log(log)


if __name__ == "__main__":
    main()
