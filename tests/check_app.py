#!/usr/bin/env python3
"""Diner app: the browser engine (web/src/app/engine.js) and the Python reference (src/app_engine.py) agree.

Random parties of one or two sample profiles, with start points at the profiles' usual areas or moved, random
filters, targets, history weights and strategies. For each: the pool, the solved settings, the list of six, the
outcome numbers, every member's satisfaction and distance for the listed places. Also checks the exported
files: no user id or visit list, SHAP additivity, each profile's candidates inside the places table.
Run: .venv/bin/python tests/check_app.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

import app_engine as eng  # noqa: E402

APP = ROOT / "web" / "public" / "data" / "app"
N_JOBS = 400


def main():
    index = json.loads((APP / "index.json").read_text())
    places = json.loads((APP / "places.json").read_text())
    profiles = {p["id"]: json.loads((APP / p["file"]).read_text()) for p in index["profiles"]}
    fails = 0

    # exported files
    n = index["places"]
    for key, col in places.items():
        cols = col.values() if isinstance(col, dict) else [col]
        if any(len(c) != n for c in cols):
            fails += 1; print(f"FAIL places.{key}: wrong length")
    text = json.dumps(index) + "".join(json.dumps({k: v for k, v in p.items() if k != "groups"})[:2000] for p in profiles.values())
    for word in ("user_id", "history\"", "later_liked", "reviewer"):
        if word in text:
            fails += 1; print(f"FAIL export holds '{word}'")
    worst = 0.0
    for pid, p in profiles.items():
        if len(p["candidates"]) != 200 or len(set(p["candidates"])) != 200 or max(p["candidates"]) >= n:
            fails += 1; print(f"FAIL {pid}: candidates")
        total = index["base"] + np.sum([p["groups"][g] for g in index["groups"]], axis=0)
        worst = max(worst, float(np.abs(total - np.array(p["raw"])).max()))
    if worst > 1e-3:
        fails += 1; print(f"FAIL SHAP additivity on the exported (rounded) values: {worst:.2e}")
    print(f"1. files: {len(profiles)} profiles, {n:,} places; largest |base + six groups - raw| {worst:.1e}")

    rng = np.random.default_rng(11)
    ids = [p["id"] for p in index["profiles"]]
    home = {p["id"]: (p["home_lat"], p["home_lon"]) for p in index["profiles"]}
    cuisines = index["cuisine_names"]
    jobs = []
    for _ in range(N_JOBS):
        party = [str(x) for x in rng.choice(ids, size=int(rng.integers(1, 3)), replace=False)]
        starts = []
        for pid in party:
            lat, lon = home[pid]
            if rng.random() < 0.5:
                lat, lon = round(lat + float(rng.uniform(-0.03, 0.03)), 3), round(lon + float(rng.uniform(-0.03, 0.03)), 3)
            starts.append({"lat": lat, "lon": lon})
        f = dict(cuisines=[], price_max=None, price_min=None, needs=[], borough=None, area=None, min_rating=None)
        roll = rng.random()
        if roll < 0.2: f["cuisines"] = [str(x) for x in rng.choice(cuisines, size=int(rng.integers(1, 4)), replace=False)]
        elif roll < 0.35: f["price_max"] = int(rng.integers(1, 4))
        elif roll < 0.45: f["needs"] = [str(rng.choice(index["needs"]))]
        elif roll < 0.55: f["borough"] = str(rng.choice(["Manhattan", "Brooklyn", "Queens"]))
        elif roll < 0.62: f["area"] = str(rng.choice(["village", "harlem", "park slope"]))
        if rng.random() < 0.25: f["min_rating"] = 4
        targets = {"km": [None, None, 4, 2, 1, 3.25][int(rng.integers(0, 6))], "popularity": [None, None, 0.4, 0.2][int(rng.integers(0, 4))],
                   "cuisines": [None, None, 4, 6][int(rng.integers(0, 4))]}
        strategy = "average" if len(party) == 1 else str(rng.choice(["average_without_misery", "least_misery"]))
        jobs.append(dict(profiles=party, starts=starts, filters=f, targets=targets, w_h=float(rng.choice([1, 1, 0.5, 0])), strategy=strategy))
    r = subprocess.run(["node", str(ROOT / "tests" / "app_node.mjs")], input=json.dumps({"jobs": jobs}), capture_output=True, text=True, check=True)
    got = json.loads(r.stdout)
    empty = short = 0
    err = 0.0
    for k, (job, g) in enumerate(zip(jobs, got)):
        members = [{"profile": profiles[pid], "start": s} for pid, s in zip(job["profiles"], job["starts"])]
        pool = eng.build_pool(places, cuisines, members, job["filters"])
        if pool != g["ids"]:
            fails += 1; print(f"FAIL job {k}: pool of {len(pool)} in Python, {len(g['ids'])} in the browser"); continue
        if not pool:
            empty += 1; continue
        pre = eng.prepare(places, index["dist_cap_km"], members, pool)
        res = eng.solve(members, pre, job["targets"], job["w_h"], job["strategy"])
        top = [pool[j] for j in res["top"]]
        short += len(top) < eng.K
        ok = top == g["top"] and res["count"] == g["count"] and res["met"] == g["met"] \
            and all(abs(res["theta"][x] - g["theta"][x]) < 1e-12 for x in res["theta"])
        if ok and top:
            sat = np.array([[row[j] for j in res["top"]] for row in res["sat"]]); dist = np.array([[row[j] for j in res["top"]] for row in pre["dist"]])
            e = max(float(np.abs(sat - np.array(g["sat"])).max()), float(np.abs(dist - np.array(g["dist"])).max()),
                    max(abs(res["outcomes"][x] - g["outcomes"][x]) for x in ("km", "popularity", "cuisines")))
            err = max(err, e)
            ok = e < 1e-9
        if not ok:
            fails += 1; print(f"FAIL job {k}: {job['profiles']} {job['strategy']} targets {job['targets']}\n  python {top} {res['theta']}\n  browser {g['top']} {g['theta']}")
    print(f"2. engine: {len(jobs)} parties and settings ({empty} with an empty pool, {short} with fewer than {eng.K} places listed); "
          f"largest difference in a number {err:.1e}")
    print(f"{fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
