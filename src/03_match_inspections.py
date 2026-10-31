"""Step 3: match Google restaurants to NYC inspection records (name + address + location).

Candidates are inspection records within 200 m. A pair matches if
- the names are very similar (>= 0.85) and the places are within 100 m or share the house number, or
- the names are fairly similar (>= 0.6) and share the house number, or
- the names are similar (>= 0.75) and the places are within 50 m.
House numbers are compared without hyphens ("63-28" = "6328"). Each inspection record (CAMIS) is used once, by its
best-scoring Google restaurant. A first version without the 100 m limit for different house numbers matched
different branches of chains (two Starbucks 173 m apart); the audit sample is drawn with a new seed after that fix.
Note: the inspection file only lists restaurants active in the last few years, so places that closed after the
2021 Google crawl cannot match; the match rate is a lower bound on matching quality.
Outputs: data/processed/inspection_matches.parquet, outputs/03_match_audit_sample.csv (100 random matches to
check by hand), outputs/03_unmatched_sample.csv (50 unmatched open restaurants with their nearest candidate)
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).parent))
from common import INSPECTIONS, MATCHES, OUTPUTS, RESTAURANTS_NYC, SEED, normalize_name

EARTH_M = 6_371_000
RADIUS_M = 200
AUDIT_SEED = 2026   # differs from SEED: the first audit sample (seed 42) was used to fix the rules


def name_similarity(a, b):
    if not a or not b:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    short, long_ = sorted([set(a.split()), set(b.split())], key=len)
    contained = short <= long_ and any(len(w) >= 3 for w in short)
    return max(ratio, 0.9 if contained else 0.0)


def house_number(street):
    m = re.match(r"\s*(\d+(?:-\d+)?)", str(street))
    return m.group(1).replace("-", "") if m else None


def parse_google_address(address):
    parts = [p.strip() for p in str(address).split(",")]
    zip_ = re.search(r"\b(\d{5})\b", parts[-1]) if parts else None
    street = parts[-3] if len(parts) >= 3 else ""
    return house_number(street), (zip_.group(1) if zip_ else None)


def main():
    g = pd.read_parquet(RESTAURANTS_NYC)
    ins = pd.read_csv(INSPECTIONS, dtype=str)
    ins["latitude"] = pd.to_numeric(ins.latitude, errors="coerce")
    ins["longitude"] = pd.to_numeric(ins.longitude, errors="coerce")
    located = ins[(ins.latitude.abs() > 1) & (ins.longitude.abs() > 1)].reset_index(drop=True)
    print(f"inspection records: {len(ins):,}; with a location: {len(located):,}")

    located["norm"] = located.dba.map(normalize_name)
    located["number"] = located.building.map(house_number)
    g["norm"] = g.name.map(normalize_name)
    g[["number", "zip"]] = g.address.map(parse_google_address).apply(pd.Series)

    tree = BallTree(np.radians(located[["latitude", "longitude"]].to_numpy()), metric="haversine")
    query = np.radians(g[["latitude", "longitude"]].astype(float).to_numpy())
    idx, dist = tree.query_radius(query, r=RADIUS_M / EARTH_M, return_distance=True)

    pairs = []
    for gi, (cands, ds) in enumerate(zip(idx, dist)):
        row = g.iloc[gi]
        for ci, d in zip(cands, ds):
            c = located.iloc[ci]
            sim = name_similarity(row.norm, c.norm)
            if sim < 0.6:
                continue
            dist_m = d * EARTH_M
            same_number = row.number is not None and row.number == c.number
            if (sim >= 0.85 and (dist_m <= 100 or same_number)) or same_number or (sim >= 0.75 and dist_m <= 50):
                pairs.append((row.gmap_id, c.camis, sim, dist_m, same_number, row.zip == c.zipcode))
    pairs = pd.DataFrame(pairs, columns=["gmap_id", "camis", "name_sim", "dist_m", "same_number", "same_zip"])
    pairs = pairs.sort_values(["name_sim", "same_number", "dist_m"], ascending=[False, False, True])
    best = pairs.drop_duplicates("gmap_id").drop_duplicates("camis")
    best = best.merge(located[["camis", "dba", "building", "street", "zipcode", "cuisine_description",
                               "last_inspection"]], on="camis")
    best.to_parquet(MATCHES, index=False)

    g["matched"] = g.gmap_id.isin(best.gmap_id)
    open_ = g[~g.permanently_closed_2021]
    print(f"matched: {g.matched.sum():,} of {len(g):,} NYC restaurants ({g.matched.mean():.1%}); "
          f"of those not closed at the 2021 crawl: {open_.matched.mean():.1%}")

    # Unmatched open restaurants whose address now has an inspection record under another name: likely replaced
    occupied = set()
    for gi, (cands, ds) in enumerate(zip(idx, dist)):
        row = g.iloc[gi]
        if row.matched or row.permanently_closed_2021 or row.number is None:
            continue
        if any(located.number.iat[ci] == row.number and d * EARTH_M <= 50 for ci, d in zip(cands, ds)):
            occupied.add(row.gmap_id)
    n_unmatched_open = (~open_.matched).sum()
    print(f"unmatched open restaurants: {n_unmatched_open:,}; their address now has another inspected business: "
          f"{len(occupied):,} ({len(occupied) / n_unmatched_open:.0%})")
    print(f"matches with the same house number: {best.same_number.mean():.1%}; same zip: {best.same_zip.mean():.1%}; "
          f"median distance: {best.dist_m.median():.0f} m")
    print(g.groupby("boroname").matched.mean().round(3).to_string())

    OUTPUTS.mkdir(exist_ok=True)
    audit = best.sample(100, random_state=AUDIT_SEED).merge(g[["gmap_id", "name", "address"]], on="gmap_id")
    audit = audit[["name", "address", "dba", "building", "street", "zipcode", "name_sim", "dist_m", "same_number"]]
    audit.assign(correct="").to_csv(OUTPUTS / "03_match_audit_sample.csv", index=False)

    miss = open_[~open_.matched].sample(50, random_state=SEED)
    nearest_d, nearest_idx = tree.query(np.radians(miss[["latitude", "longitude"]].astype(float).to_numpy()), k=1)
    near = located.iloc[nearest_idx[:, 0]].reset_index(drop=True)
    miss = miss[["name", "address"]].reset_index(drop=True)
    miss["nearest_inspection_name"] = near.dba
    miss["nearest_inspection_address"] = near.building + " " + near.street
    miss["nearest_m"] = (nearest_d[:, 0] * EARTH_M).round()
    miss.to_csv(OUTPUTS / "03_unmatched_sample.csv", index=False)


if __name__ == "__main__":
    main()
