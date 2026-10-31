"""Step 1: NYC restaurants from the Google Local New York metadata.

A business is a restaurant if one of its Google categories contains "restaurant". It is in NYC if its point
falls inside a 2020 Neighborhood Tabulation Area, which also gives its neighborhood and borough.
Output: data/processed/restaurants_nyc.parquet
"""

import gzip
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import META, NTA, PROCESSED, RESTAURANTS_NYC

PRICE_LEVEL = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}


def load_meta():
    rows = []
    with gzip.open(META, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cats = d.get("category") or []
            rows.append({
                "gmap_id": d["gmap_id"],
                "name": d.get("name"),
                "address": d.get("address"),
                "latitude": d.get("latitude"),
                "longitude": d.get("longitude"),
                "categories": "; ".join(cats),
                "is_restaurant": any("restaurant" in c.lower() for c in cats),
                "google_avg_rating": d.get("avg_rating"),
                "google_num_reviews": d.get("num_of_reviews"),
                "price_level": PRICE_LEVEL.get(d.get("price")),
                "google_state": d.get("state"),
            })
    return pd.DataFrame(rows)


def main():
    meta = load_meta()
    print(f"businesses in NY metadata: {len(meta):,}; restaurants: {meta.is_restaurant.sum():,}")
    rest = meta[meta.is_restaurant & meta.latitude.notna() & meta.longitude.notna()].drop(columns="is_restaurant")
    rest = rest.drop_duplicates("gmap_id")

    points = gpd.GeoDataFrame(rest, geometry=gpd.points_from_xy(rest.longitude, rest.latitude), crs="EPSG:4326")
    nta = gpd.read_file(NTA)[["nta2020", "ntaname", "boroname", "geometry"]].to_crs("EPSG:4326")
    nyc = gpd.sjoin(points, nta, how="inner", predicate="within").drop(columns=["geometry", "index_right"])
    nyc = pd.DataFrame(nyc).drop_duplicates("gmap_id")
    nyc["permanently_closed_2021"] = nyc.google_state.fillna("").str.contains("Permanently closed")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    nyc.to_parquet(RESTAURANTS_NYC, index=False)
    print(f"NYC restaurants: {len(nyc):,} (permanently closed at crawl time: {nyc.permanently_closed_2021.sum():,})")
    print(nyc.boroname.value_counts().to_string())
    print(f"with a price level: {nyc.price_level.notna().mean():.0%}")


if __name__ == "__main__":
    main()
