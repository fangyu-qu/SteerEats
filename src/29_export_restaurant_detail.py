#!/usr/bin/env python3
"""Step 29: export rating histories for the restaurant detail view (V8).

For every candidate of the demo cases, the view shows how its star ratings arrived over time. This script
writes one compact file per case with each candidate's monthly star counts:
  web/public/data/detail/demo_XX.json   {"id", "restaurants": [{"b": [n1..n5], "m": [[month, n1..n5], ...]}, ...]}
  web/public/data/detail/index.json     shared settings (months, snapshot, prior, city-wide spread)
`restaurants` follows the order of `candidates` in the case file. `b` counts the reviews before the first
exported month; `m` lists only the months that have a review.

Rules
  - Same reviews as the rankers (src/ranker_core.py): clean NYC reviews, first review per user and restaurant.
    Months are UTC months, like the snapshot cutoff, so "before the snapshot" means the same here and there.
  - Months before the snapshot (2020-08-01) are what the models could see. Later months are in the same list
    and the view draws them as later reviews.
  - Exported months end in April 2021: review collection is incomplete from May 2021 (src/11_clean.py).
  - Only monthly counts per star level leave this script: no user ids, names, review text or per-review rows.

Checks (the script stops if one fails): computed from the exported counts alone, the review counts and the
smoothed ratings at the snapshot must equal `reviews`, `recent_reviews`, `rating` and `recent_rating` in the
case files, which come from the rankers' own snapshot (src/19_export_cases.py).

tests/detail_expected.json holds rolling-window values for a few restaurants per case, computed here, so that
tests/check_detail.mjs can confirm the browser (web/src/detail.js) computes the same numbers.
Outputs: web/public/data/detail/*.json, tests/detail_expected.json, outputs/restaurant_detail/DETAIL.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS, REVIEWS_CLEAN, ROOT, TAIL_START, TEST_START
from ranker_core import PRIOR_STRENGTH

CASES = ROOT / "web" / "public" / "data" / "cases"
DETAIL = ROOT / "web" / "public" / "data" / "detail"
EXPECTED = ROOT / "tests" / "detail_expected.json"
REPORT = OUTPUTS / "restaurant_detail"

FIRST_MONTH = "2016-01"   # a year before the first charted month, so every rolling window starts complete
CHART_START = "2017-01"   # the interaction sample starts here; earlier years are sparse
WINDOWS = [3, 6, 12]      # rolling windows offered in the view, in months
RECENT = 12               # the rankers' "recent" window, in months
POOL_WEIGHT = 2           # reviews' worth of city-wide variance pooled into a window's variance
Z = 1.96                  # 95% band
STARS = np.arange(1, 6)
SHIFT = 0.5               # stars; a rating "moved" if two means differ by at least this much
MIN_N = 20                # reviews needed in a window before a move is counted


def month_index(t, first):
    """Months since `first` (a Timestamp at the start of a month), from UTC timestamps."""
    return (t.dt.year - first.year) * 12 + (t.dt.month - first.month)


def window_stats(counts, sd_all):
    """Mean and 95% band of the star ratings in one window, from its counts per star level.

    The band is mean +/- 1.96 standard errors, clipped to 1-5 stars. The window's variance is pooled with the
    city-wide variance at the weight of POOL_WEIGHT reviews, so that a few identical ratings do not produce a
    band of zero width.
    """
    n = int(counts.sum())
    if n == 0:
        return None
    mean = float((counts * STARS).sum() / n)
    ss = float((counts * (STARS - mean) ** 2).sum())
    var = (ss + POOL_WEIGHT * sd_all ** 2) / (n + POOL_WEIGHT - 1)
    se = (var / n) ** 0.5
    return dict(n=n, mean=mean, lo=max(1.0, mean - Z * se), hi=min(5.0, mean + Z * se))


def main():
    first, chart_start = pd.Timestamp(FIRST_MONTH), pd.Timestamp(CHART_START)
    cutoff, tail = pd.Timestamp(TEST_START), pd.Timestamp(TAIL_START)
    n_months = (tail.year - first.year) * 12 + (tail.month - first.month)
    cut = (cutoff.year - first.year) * 12 + (cutoff.month - first.month)
    chart0 = (chart_start.year - first.year) * 12 + (chart_start.month - first.month)

    cases = [json.loads(p.read_text()) for p in sorted(CASES.glob("demo_*.json"))]
    if not cases:
        raise SystemExit(f"No demo cases in {CASES}; run src/19_export_cases.py first")
    if any(c["cutoff"] != TEST_START for c in cases):
        raise SystemExit("A case uses another snapshot than TEST_START")
    wanted = pd.Index(sorted({g for c in cases for g in c["candidates"]["gmap_id"]}))

    # The rankers' review set: first review per (user, restaurant), in time order (ranker_core.Data).
    r = pd.read_parquet(REVIEWS_CLEAN, columns=["user_id", "gmap_id", "time", "rating"])
    r = r.sort_values("time", kind="stable").drop_duplicates(["user_id", "gmap_id"])
    if not r.rating.isin(STARS).all():
        raise SystemExit("A rating is not a whole number of stars from 1 to 5")
    seen = r[r.time < cutoff]
    prior = float(seen.rating.mean())               # the rankers' prior: city-wide mean before the snapshot
    sd_all = float(seen.rating.std(ddof=0))         # city-wide spread before the snapshot
    city_rows = len(seen)

    r = r[r.gmap_id.isin(wanted) & (r.time < tail)].drop(columns="user_id")   # user ids stop here
    item = wanted.get_indexer(r.gmap_id)
    month = month_index(r.time, first).to_numpy()
    star = r.rating.to_numpy(int) - 1
    before = np.zeros((len(wanted), 5), int)
    np.add.at(before, (item[month < 0], star[month < 0]), 1)
    hist = np.zeros((len(wanted), n_months, 5), int)
    ok = month >= 0
    np.add.at(hist, (item[ok], month[ok], star[ok]), 1)

    DETAIL.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    expected, checked, rows = {}, 0, []
    sample_months = sorted(set(range(chart0, n_months, 6)) | {cut - 1, n_months - 1})
    for c in cases:
        cand = c["candidates"]
        ix = wanted.get_indexer(cand["gmap_id"])
        b, h = before[ix], hist[ix]

        # Checks against the rankers' snapshot, from the exported counts alone.
        seen_counts = b + h[:, :cut].sum(axis=1)
        recent_counts = h[:, cut - RECENT:cut].sum(axis=1)
        n_all, n_recent = seen_counts.sum(axis=1), recent_counts.sum(axis=1)
        mean = ((seen_counts * STARS).sum(axis=1) + PRIOR_STRENGTH * prior) / (n_all + PRIOR_STRENGTH)
        recent_mean = ((recent_counts * STARS).sum(axis=1) + PRIOR_STRENGTH * mean) / (n_recent + PRIOR_STRENGTH)
        problems = []
        if not np.array_equal(n_all, np.array(cand["reviews"])):
            problems.append("reviews")
        if not np.array_equal(n_recent, np.array(cand["recent_reviews"])):
            problems.append("recent_reviews")
        if np.abs(mean - np.array(cand["rating"])).max() > 1e-6:
            problems.append("rating")
        if np.abs(recent_mean - np.array(cand["recent_rating"])).max() > 1e-6:
            problems.append("recent_rating")
        if problems:
            raise SystemExit(f"{c['id']}: exported counts do not reproduce the case file: {', '.join(problems)}")
        checked += len(ix)

        (DETAIL / f"{c['id']}.json").write_text(json.dumps(
            {"id": c["id"], "restaurants": [
                {"b": b[k].tolist(), "m": [[int(m), *h[k, m].tolist()] for m in np.flatnonzero(h[k].sum(axis=1))]}
                for k in range(len(ix))]}, separators=(",", ":")))

        # Reference values for the browser check: the candidates with the fewest, median and most reviews.
        order = np.argsort(n_all, kind="stable")
        picks = sorted({int(order[0]), int(order[len(order) // 2]), int(order[-1])})
        exp = []   # rows: candidate, window, month, n, mean, lo, hi, n to date, mean to date
        for k in picks:
            for w in WINDOWS:
                for m in sample_months:
                    s = window_stats(h[k, max(0, m - w + 1):m + 1].sum(axis=0), sd_all) or dict(n=0, mean=None, lo=None, hi=None)
                    cum = b[k] + h[k, :m + 1].sum(axis=0)
                    exp.append([k, w, m, s["n"], s["mean"], s["lo"], s["hi"], int(cum.sum()),
                                float((cum * STARS).sum() / cum.sum()) if cum.sum() else None])
        expected[c["id"]] = exp

        later_counts = h[:, cut:].sum(axis=1)
        for k in range(len(ix)):
            rows.append(dict(gmap_id=cand["gmap_id"][k], name=cand["name"][k], n_all=int(n_all[k]),
                             n_recent=int(n_recent[k]), n_later=int(later_counts[k].sum()),
                             raw_all=(seen_counts[k] * STARS).sum() / n_all[k] if n_all[k] else np.nan,
                             raw_recent=(recent_counts[k] * STARS).sum() / n_recent[k] if n_recent[k] else np.nan,
                             raw_later=(later_counts[k] * STARS).sum() / later_counts[k].sum() if later_counts[k].sum() else np.nan,
                             price_missing=cand["price"][k] is None))

    months = [str(p) for p in pd.period_range(FIRST_MONTH, periods=n_months, freq="M")]
    index = dict(cutoff=TEST_START, first_month=FIRST_MONTH, months=n_months, chart_start=chart0, cutoff_index=cut,
                 last_month=months[-1], windows=WINDOWS, recent_months=RECENT, prior_mean=prior,
                 prior_strength=PRIOR_STRENGTH, sd_all=sd_all, pool_weight=POOL_WEIGHT, z=Z,
                 cases=[c["id"] for c in cases])
    (DETAIL / "index.json").write_text(json.dumps(index, indent=1))
    EXPECTED.write_text(json.dumps(dict(index=index, cases=expected)))

    u = pd.DataFrame(rows).drop_duplicates("gmap_id")
    size = sum(p.stat().st_size for p in DETAIL.glob("demo_*.json"))
    cells = int((hist.sum(axis=2) > 0).sum())
    rec = u[u.n_recent >= MIN_N]
    rec_shift = rec[(rec.raw_recent - rec.raw_all).abs() >= SHIFT]
    lat = u[(u.n_recent >= MIN_N) & (u.n_later >= MIN_N)]
    lat_shift = lat[(lat.raw_later - lat.raw_recent).abs() >= SHIFT]
    down, up = int((lat_shift.raw_later < lat_shift.raw_recent).sum()), int((lat_shift.raw_later > lat_shift.raw_recent).sum())
    top = lat_shift.assign(move=lat_shift.raw_later - lat_shift.raw_recent)
    top = top.reindex(top.move.abs().sort_values(ascending=False).index).head(8)
    last_label = pd.Period(months[-1]).strftime("%B %Y")
    lines = [
        "# Restaurant detail (V8): rating histories", "",
        "Generated by `src/29_export_restaurant_detail.py`. Reviews are the rankers' review set (clean NYC reviews, "
        "first review per user and restaurant); months are UTC months.", "",
        "## What is exported", "",
        f"- {len(cases)} demo cases, {checked:,} candidates, {len(u):,} different restaurants.",
        f"- Monthly star counts from {pd.Period(FIRST_MONTH).strftime('%B %Y')} to {last_label} ({n_months} months; charted from "
        f"{pd.Period(CHART_START).strftime('%B %Y')}), plus one bucket for earlier reviews. Later months are left out because "
        "collection is incomplete.",
        f"- {int(u.n_all.sum()):,} reviews before the {TEST_START} snapshot and {int(u.n_later.sum()):,} later reviews of these restaurants; "
        f"{cells:,} restaurant-months with at least one review.",
        f"- Files: {len(cases)} case files, {size / 1e6:.2f} MB in total. They hold counts per month and star level only.", "",
        "## Check against the rankers' snapshot", "",
        f"For all {checked:,} candidates, the exported counts reproduce the case files: `reviews` and `recent_reviews` exactly, "
        "`rating` and `recent_rating` (smoothed toward the city-wide mean) within 1e-6.",
        f"City-wide mean before the snapshot: {prior:.4f} stars over {city_rows:,} reviews (the rankers' prior, strength "
        f"{PRIOR_STRENGTH}); standard deviation {sd_all:.4f}.", "",
        "## How much review evidence a restaurant has", "",
        f"- Reviews before the snapshot per restaurant: median {int(u.n_all.median())}, quartiles "
        f"{int(u.n_all.quantile(0.25))} to {int(u.n_all.quantile(0.75))}.",
        f"- Reviews in the 12 months before the snapshot: median {int(u.n_recent.median())}; "
        f"{(u.n_recent < MIN_N).mean():.1%} of restaurants have fewer than {MIN_N}, where the band is wide.",
        f"- Price level not listed: {u.price_missing.mean():.1%} of restaurants.", "",
        "## Do ratings move?", "",
        f"Plain means, no smoothing. A move is a difference of at least {SHIFT} stars; both windows need at least {MIN_N} reviews.", "",
        f"- Before the snapshot: of {len(rec):,} restaurants with {MIN_N}+ reviews in the last 12 months, {len(rec_shift):,} "
        f"({len(rec_shift) / len(rec):.1%}) have a 12-month mean that differs from their all-time mean.",
        f"- After the snapshot (August 2020 to {last_label}, which the models could not see): of {len(lat):,} restaurants with {MIN_N}+ reviews "
        f"in both periods, {len(lat_shift):,} ({len(lat_shift) / len(lat):.1%}) moved from their 12-month mean at the snapshot "
        f"({down} down, {up} up).", "",
        "Largest moves after the snapshot:", "",
        "| Restaurant | 12 months before | Reviews | After | Reviews | Move |", "|---|---|---|---|---|---|",
        *[f"| {x['name']} | {x.raw_recent:.2f} | {x.n_recent} | {x.raw_later:.2f} | {x.n_later} | {x.move:+.2f} |" for _, x in top.iterrows()],
    ]
    (REPORT / "DETAIL.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
