#!/usr/bin/env python3
"""Step 41: data for the diner app (web/index.html), scored with the best ranker (LambdaRank).

The app lets one or two people borrow a sample taste profile and pick a place together. A profile is one of
the 24 demo users of step 19 (test users at the 2020-08-01 snapshot, no user id published). As in the offline
group study (step 33), the pool of a party is the union of its members' 200 stage-1 candidates, and every
member has a model score for every place in the pool. So this step scores, for every profile, the union of
all 24 candidate sets (the "places"), and explains each score with interventional SHAP on the same
200-row background as steps 17 and 39.

Outputs (restaurant aggregates and model outputs only; no user ids, no visit lists)
  web/public/data/app/index.json       model, feature groups, cuisine names, profiles (home rounded to ~1 km)
  web/public/data/app/places.json      one column per fact over all places (sorted by restaurant index)
  web/public/data/app/profile_XX.json  for one profile: its candidates, and over all places the raw score,
                                       the collaborative part of it and the SHAP value of each feature group
The ranking model's test results (index.json "rankers") are copied from web/public/data/evaluation.json.

Run: .venv/bin/python src/41_export_app.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, ROOT, TEST_START
from ranker_core import FEATURE_GROUPS, FEATURES, Data, Snapshot, candidates
import steering

MODELS = OUTPUTS / "rankers_two_stage"
LOCAL = CLEAN / "cases"
WEB = ROOT / "web" / "public" / "data"
OUT = WEB / "app"
DECIMALS = 6
NEEDS = ["vegetarian_options", "halal", "late_night", "alcohol", "reservations", "good_for_groups", "good_for_kids"]
GROUPS = list(FEATURE_GROUPS)
COLLAB = [FEATURES.index(f) for f in FEATURE_GROUPS["Collaborative"]]


def r(v, d=DECIMALS):
    return np.round(np.asarray(v, dtype=float), d).tolist()


def main():
    import lightgbm as lgb
    import shap
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    local = [json.loads(p.read_text()) for p in sorted(LOCAL.glob("case_*.json"))]
    demo = [json.loads((WEB / "cases" / f"demo_{c['id'][-2:]}.json").read_text()) for c in local]
    users = [c["user"]["user_id"] for c in local]
    data = Data()
    A = data.attrs
    snap = Snapshot(data, TEST_START, set(users))
    print(f"data and snapshot ready ({time.time() - t0:.0f}s)", flush=True)
    rk = lgb.Booster(model_file=str(MODELS / "lambdarank_model.txt"))
    bg = np.load(CLEAN / "shap" / "background.npz")["x"]
    tree = shap.TreeExplainer(rk, data=shap.maskers.Independent(bg, max_samples=len(bg)),
                              feature_perturbation="interventional", model_output="raw")
    base = float(np.atleast_1d(tree.expected_value)[0])

    feats, cands = [], []
    for u, d in zip(users, demo):
        f, seen, cf = snap.user(u)
        ids = candidates(snap, f, seen, cf)
        assert A.gmap_id.iloc[ids].tolist() == d["candidates"]["gmap_id"], f"candidates of {d['id']} differ from the demo case"
        feats.append(f)
        cands.append(ids)
    pool = np.unique(np.concatenate(cands))          # restaurant indices, ascending
    where = {int(i): k for k, i in enumerate(pool)}
    print(f"{len(users)} profiles, {len(pool):,} places ({time.time() - t0:.0f}s)", flush=True)

    # facts of the places (the attribute flags are restaurant features, the same for every user)
    x0 = feats[0][pool]
    sub = A.iloc[pool]
    nta_codes, _ = pd.factorize(A.nta2020)
    ref = {}
    for k, (d, ids) in enumerate(zip(demo, cands)):
        for j, i in enumerate(ids):
            ref.setdefault(int(i), [k, j])           # first demo case that holds the place: its detail and themes
    places = {
        "name": sub.name.tolist(), "lat": r(sub.latitude), "lon": r(sub.longitude), "borough": sub.boroname.tolist(),
        "nta": [int(v) for v in nta_codes[pool]], "nta_name": sub.ntaname.tolist(),
        "cuisine": [int(v) for v in data.cuisine[pool]],
        "price": [None if np.isnan(v) else int(v) for v in sub.price_level], "is_chain": sub.is_chain.tolist(),
        "rating": r(snap.mean[pool], 3), "reviews": snap.pop[pool].astype(int).tolist(),
        "recent_reviews": snap.recent_pop[pool].astype(int).tolist(),
        "needs": {a: [int(v) for v in x0[:, FEATURES.index(f"attr_{a}")]] for a in NEEDS},
        "ref": [ref[int(i)] for i in pool],
    }
    (OUT / "places.json").write_text(json.dumps(places, separators=(",", ":")))

    profiles, worst = [], 0.0
    for k, (u, d, f, ids) in enumerate(zip(users, demo, feats, cands), start=1):
        x = f[pool].astype(np.float64)
        raw = rk.predict(x, raw_score=True)
        phi = tree.shap_values(x, check_additivity=False)
        resid = float(np.abs(base + phi.sum(axis=1) - raw).max())
        worst = max(worst, resid)
        collab = phi[:, COLLAB].sum(axis=1)
        own = raw[[where[int(i)] for i in ids]]
        body = {"id": f"profile_{k:02d}", "candidates": [where[int(i)] for i in ids],
                "raw": r(raw), "collab": r(collab),
                "groups": {g: r(phi[:, [FEATURES.index(n) for n in FEATURE_GROUPS[g]]].sum(axis=1)) for g in GROUPS},
                "shap_residual_max": resid}
        (OUT / f"profile_{k:02d}.json").write_text(json.dumps(body, separators=(",", ":")))
        top = ids[steering.order(own)[:3]]
        profiles.append({"id": body["id"], "file": f"profile_{k:02d}.json", "demo": d["id"],
                         "home_lat": d["user"]["home_lat"], "home_lon": d["user"]["home_lon"],
                         "home_borough": d["user"]["home_borough"], "history_size": d["user"]["history_size"],
                         "cuisine_profile": d["user"]["cuisine_profile"],
                         "top": A.name.iloc[top].tolist()})
        print(f"  {body['id']}: additivity error {resid:.1e} ({time.time() - t0:.0f}s)", flush=True)

    ev = json.loads((WEB / "evaluation.json").read_text())
    rankers = {name: {"ndcg": v["ndcg"], "recall": v["recall"]} for name, v in ev["rankers"].items()
               if name in ("Logistic regression", "LightGBM", "LightGBM LambdaRank (supplementary)", "Item-CF")}
    index = {"cutoff": TEST_START, "model": "LightGBM LambdaRank", "base": round(base, DECIMALS), "groups": GROUPS,
             "cuisine_names": list(data.cuisine_names), "needs": NEEDS, "dist_cap_km": steering.DIST_CAP_KM,
             "places": int(len(pool)), "shap_residual_max": worst, "test_users": ev["users"], "rankers": rankers,
             "profiles": profiles}
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    size = sum(p.stat().st_size for p in OUT.glob("*.json")) / 1e6
    print(f"wrote {len(profiles)} profiles and {len(pool):,} places to {OUT.relative_to(ROOT)} ({size:.1f} MB); "
          f"largest SHAP additivity error {worst:.1e}; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
