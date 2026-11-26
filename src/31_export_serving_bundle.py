#!/usr/bin/env python3
"""Step 31: the bundle the live server reads (W3, live profiles).

The server (server/live.py) builds a taste profile from places a person picks, and scores it with the app's
ranker (LambdaRank) exactly as the offline pipeline would. Building a Snapshot needs every review and takes
minutes, so this step saves what one snapshot (2020-08-01, the test cutoff) holds:

  server/bundle/arrays.npz    restaurant arrays of ranker_core.Data and item statistics of ranker_core.Snapshot
  server/bundle/sim.npz       item-item similarity (top 50 neighbours), scipy sparse
  server/bundle/meta.json     restaurant names, neighbourhoods and labels; cuisine names; the app's place table
                              (which restaurant is which row of web/public/data/app/places.json, and its `ref`)
  server/bundle/profiles.json the 24 sample tastes as profiles (the restaurants each one reviewed)
  server/bundle/lambdarank_model.txt, background.npz   the ranker and the SHAP background of step 17

The folder is git-ignored and must stay private: profiles.json holds the visit lists of real reviewers
(without their ids). The server never returns them; it only scores them when a sample taste joins a live party.
Everything else is restaurant aggregates.

Run after step 41: .venv/bin/python src/31_export_serving_bundle.py
Check: .venv/bin/python tests/check_serving.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, ROOT, TEST_START
from ranker_core import FEATURES, Data, Snapshot

OUT = ROOT / "server" / "bundle"
APP = ROOT / "web" / "public" / "data" / "app"
NEEDS = ["vegetarian_options", "halal", "late_night", "alcohol", "reservations", "good_for_groups", "good_for_kids"]


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    local = [json.loads(p.read_text()) for p in sorted((CLEAN / "cases").glob("case_*.json"))]
    users = [c["user"]["user_id"] for c in local]
    data = Data()
    A = data.attrs
    snap = Snapshot(data, TEST_START, set(users))
    print(f"data and snapshot ready ({time.time() - t0:.0f}s)", flush=True)

    np.savez_compressed(
        OUT / "arrays.npz",
        lat=data.lat, lon=data.lon, nta=data.nta, boro=data.boro, cuisine=data.cuisine, price=data.price,
        price_missing=data.price_missing, is_chain=data.is_chain, attr=data.attr,
        pop=snap.pop, recent_pop=snap.recent_pop, pop_3m=snap.pop_3m, mean=snap.mean, recent_mean=snap.recent_mean,
        available=snap.available, static=snap.static, global_cuisine=snap.global_cuisine)
    sp.save_npz(OUT / "sim.npz", snap.sim.tocsr())

    # the app's place table: restaurant index of every row of places.json, checked against the names
    index = json.loads((APP / "index.json").read_text())
    places = json.loads((APP / "places.json").read_text())
    pool = np.unique(np.concatenate([np.array([data.ii[g] for g in json.loads((ROOT / "web" / "public" / "data" / "cases" / f"{p['demo']}.json").read_text())["candidates"]["gmap_id"]])
                                     for p in index["profiles"]]))
    assert len(pool) == index["places"] and A.name.iloc[pool].tolist() == places["name"], "app place table differs: run step 41 first"
    boro_names = pd.factorize(A.boroname)[1]
    price = [None if np.isnan(v) else int(v) for v in A.price_level]
    meta = {
        "cutoff": TEST_START, "features": FEATURES, "cuisine_names": list(data.cuisine_names), "needs": NEEDS,
        "boroughs": list(boro_names),
        "name": A.name.tolist(), "borough": A.boroname.tolist(), "nta_name": A.ntaname.tolist(),
        "cuisine_label": A.cuisine_label.tolist(), "price": price, "chain": A.is_chain.astype(bool).tolist(),
        "app_places": [int(i) for i in pool], "app_ref": places["ref"],
        "app_profiles": [p["id"] for p in index["profiles"]],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    # the sample tastes, in the order of the app's profiles (profile_01 is demo case 01)
    profiles = {}
    for p, u in zip(index["profiles"], users):
        q = snap.profile[u]
        profiles[p["id"]] = {"seen": [int(i) for i in q["seen"]], "liked": [int(i) for i in q["liked"]]}
    (OUT / "profiles.json").write_text(json.dumps(profiles, separators=(",", ":")))

    shutil.copyfile(OUTPUTS / "rankers_two_stage" / "lambdarank_model.txt", OUT / "lambdarank_model.txt")
    shutil.copyfile(CLEAN / "shap" / "background.npz", OUT / "background.npz")
    size = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"wrote {OUT.relative_to(ROOT)}: {len(data.items):,} restaurants, {int(snap.available.sum()):,} open at the snapshot, "
          f"{snap.sim.nnz:,} similarities, {len(profiles)} sample tastes; {size:.1f} MB; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
