"""Step 2: keep the reviews of NYC restaurants from the full New York review file (33.5M lines).

Streams the gzip file through `gzip -dc` and parses only lines whose gmap_id is an NYC restaurant.
Output: data/processed/reviews_nyc.parquet (user_id, gmap_id, time_ms, rating, text, has_owner_response)
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from common import RESTAURANTS_NYC, REVIEWS, REVIEWS_NYC

KEY = b'"gmap_id": "'
SCHEMA = pa.schema([("user_id", pa.string()), ("gmap_id", pa.string()), ("time_ms", pa.int64()),
                    ("rating", pa.int8()), ("text", pa.string()), ("has_owner_response", pa.bool_())])
BATCH = 500_000


def main():
    nyc = set(pd.read_parquet(RESTAURANTS_NYC, columns=["gmap_id"]).gmap_id)
    proc = subprocess.Popen(["gzip", "-dc", str(REVIEWS)], stdout=subprocess.PIPE, bufsize=1 << 20)
    writer = pq.ParquetWriter(REVIEWS_NYC, SCHEMA)
    cols = {name: [] for name in SCHEMA.names}
    seen = kept = 0
    for line in proc.stdout:
        seen += 1
        k = line.find(KEY)
        if k < 0:
            continue
        start = k + len(KEY)
        if line[start:line.find(b'"', start)].decode() not in nyc:
            continue
        d = json.loads(line)
        cols["user_id"].append(d["user_id"])
        cols["gmap_id"].append(d["gmap_id"])
        cols["time_ms"].append(d["time"])
        cols["rating"].append(d["rating"])
        cols["text"].append(d["text"])
        cols["has_owner_response"].append(d["resp"] is not None)
        kept += 1
        if len(cols["user_id"]) >= BATCH:
            writer.write_table(pa.table(cols, schema=SCHEMA))
            cols = {name: [] for name in SCHEMA.names}
        if seen % 5_000_000 == 0:
            print(f"{seen:,} lines read, {kept:,} kept", flush=True)
    if cols["user_id"]:
        writer.write_table(pa.table(cols, schema=SCHEMA))
    writer.close()
    proc.wait()
    print(f"done: {seen:,} lines read, {kept:,} NYC restaurant reviews kept")


if __name__ == "__main__":
    main()
