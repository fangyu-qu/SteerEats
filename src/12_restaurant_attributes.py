#!/usr/bin/env python3
"""Step 12: one row per clean restaurant with attributes that do not depend on a time cutoff.

Columns
  cuisine_group, cuisine_label, cuisine_kind, cuisine_source, cuisine_groups (all matched groups)
  dohmh_cuisine          DOHMH label for inspection-matched restaurants (context only)
  price_level, price_missing
  chain_key, chain_size, is_chain (normalized name shared by >= 5 NYC restaurants)
  attr_* flags from the Google "MISC" attributes, attrs_missing

Assumption, stated in the report: cuisine, price, chain membership and the selected attributes are
stable over 2017-2021, so a 2021 snapshot can describe earlier years. COVID-era attributes (mask rules,
curbside pickup, no-contact delivery, outdoor seating) are excluded for that reason. Ratings, review
counts and open/closed state are excluded because they include future information; time-dependent
features are computed per snapshot in the ranker instead.
Output: data/clean/restaurant_attributes.parquet, outputs/cleaning/cuisine_groups.csv,
        outputs/cleaning/cuisine_category_map.csv
"""
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import META, MATCHES, OUTPUTS, RESTAURANT_ATTRS, RESTAURANTS_CLEAN, normalize_name
from taxonomy import GROUPS, assign, group_of

ATTRS = {
    "attr_vegetarian_options": ("Offerings", "Vegetarian options"),
    "attr_halal": ("Offerings", "Halal food"),
    "attr_organic": ("Offerings", "Organic dishes"),
    "attr_late_night": ("Offerings", "Late-night food"),
    "attr_alcohol": ("Offerings", {"Alcohol", "Beer", "Wine", "Cocktails", "Hard liquor"}),
    "attr_dine_in": ("Service options", "Dine-in"),
    "attr_delivery": ("Service options", "Delivery"),
    "attr_takeout": ("Service options", "Takeout"),
    "attr_reservations": ("Planning", "Accepts reservations"),
    "attr_good_for_groups": ("Crowd", "Groups"),
    "attr_good_for_kids": ("Amenities", "Good for kids"),
    "attr_wheelchair": ("Accessibility", "Wheelchair accessible entrance"),
}
CHAIN_MIN = 5
OUT = OUTPUTS / "cleaning"


def load_misc(ids):
    rows = {}
    with gzip.open(META, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["gmap_id"] in ids and d["gmap_id"] not in rows:
                rows[d["gmap_id"]] = d.get("MISC") or {}
    return rows


def flags(misc):
    out = {"attrs_missing": not misc}
    for col, (group, value) in ATTRS.items():
        have = set(misc.get(group, []))
        out[col] = bool(have & value) if isinstance(value, set) else value in have
    return out


def main():
    R = pd.read_parquet(RESTAURANTS_CLEAN)
    M = pd.read_parquet(MATCHES, columns=["gmap_id", "cuisine_description"])
    R = R.merge(M.rename(columns={"cuisine_description": "dohmh_cuisine"}), on="gmap_id", how="left")

    assigned = [assign(c, d if isinstance(d, str) else None) for c, d in zip(R.categories, R.dohmh_cuisine)]
    R["cuisine_group"] = [a[0] for a in assigned]
    R["cuisine_groups"] = ["; ".join(a[1]) for a in assigned]
    R["cuisine_source"] = [a[2] for a in assigned]
    R["cuisine_label"] = R.cuisine_group.map(lambda g: GROUPS[g][0])
    R["cuisine_kind"] = R.cuisine_group.map(lambda g: GROUPS[g][1])

    R["price_missing"] = R.price_level.isna()
    R["chain_key"] = R.name.map(normalize_name)
    R["chain_size"] = R.chain_key.map(R.chain_key.value_counts())
    R["is_chain"] = R.chain_size >= CHAIN_MIN

    misc = load_misc(set(R.gmap_id))
    F = pd.DataFrame([flags(misc.get(g, {})) for g in R.gmap_id])
    R = pd.concat([R.reset_index(drop=True), F], axis=1)

    cols = ["gmap_id", "name", "boroname", "nta2020", "ntaname", "latitude", "longitude",
            "cuisine_group", "cuisine_label", "cuisine_kind", "cuisine_source", "cuisine_groups",
            "dohmh_cuisine", "price_level", "price_missing", "chain_key", "chain_size", "is_chain",
            "merged_listings", "attrs_missing", *ATTRS]
    R[cols].to_parquet(RESTAURANT_ATTRS, index=False)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = (R.groupby(["cuisine_group", "cuisine_label", "cuisine_kind"]).size()
                .rename("restaurants").reset_index().sort_values("restaurants", ascending=False))
    summary.to_csv(OUT / "cuisine_groups.csv", index=False)
    cats = R.categories.fillna("").str.split("; ").explode().value_counts().rename("restaurants")
    cmap = cats.rename_axis("google_category").reset_index()
    cmap["group"] = cmap.google_category.map(group_of)
    cmap.to_csv(OUT / "cuisine_category_map.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nprimary group from: {R.cuisine_source.value_counts().to_dict()}")
    print(f"general (no cuisine or format): {(R.cuisine_group == 'general').sum():,}")
    print(f"chains (name shared by >= {CHAIN_MIN}): {R.is_chain.sum():,} restaurants, "
          f"{R.loc[R.is_chain, 'chain_key'].nunique()} chains")
    print(f"price missing: {R.price_missing.mean():.1%}; attributes missing: {R.attrs_missing.mean():.1%}")
    print(R[list(ATTRS)].mean().round(3).to_string())
    unmapped = cmap[cmap.group.isna()].head(25)
    print("\nmost common unmapped categories:\n" + unmapped.to_string(index=False))


if __name__ == "__main__":
    main()
