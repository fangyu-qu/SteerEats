#!/usr/bin/env python3
"""Step 19: export steerable case files for the browser demo, plus replay expectations.

For a set of test users at the 2020-08-01 snapshot, each case holds the 200 stage-1 candidates with
everything the browser needs to recompute rankings and explanations without a server:
  - restaurant facts (name, cuisine group, neighbourhood, price, rating and review counts at the cutoff,
    coordinates, distance from the user's usual area);
  - for logistic regression and LightGBM: raw score, interventional SHAP per feature and per feature
    group (same 200-row background as step 17), and the score split into collaborative and
    non-collaborative parts used by the history-weight control;
  - normalized distance, popularity percentile, cuisine and neighbourhood codes for steering and MMR.
Cases: three scripted ones chosen by rule (most home-borough-bound Manhattan user, most
popularity-dominated top 10, largest disagreement between the models) and random others; every case has
at least one restaurant the user went on to like among the candidates.

Two versions:
  data/clean/cases/case_XX.json      full (history list, future visits flagged), local only;
  web/public/data/cases/demo_XX.json  shareable: no user id, home rounded to ~1 km, no visit history
                                       list, no future-visit flags.
Replay expectations (tests/replay_expected.json): final top-20 orders and residual checks for several
control settings, computed by src/steering.py from the exported (rounded) demo files.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, ROOT, TEST_START
from ranker_core import FEATURE_GROUPS, FEATURES, Data, Snapshot, candidates, eligible, haversine_km
import steering

SEED = 42
N_POOL = 400
N_CASES = 24
DECIMALS = 6
MODELS = OUTPUTS / "rankers_two_stage"
LOCAL = CLEAN / "cases"
DEMO = ROOT / "web" / "public" / "data" / "cases"
GROUPS = list(FEATURE_GROUPS)
COLLAB = FEATURE_GROUPS["Collaborative"]
REPLAY_THETAS = [
    dict(steering.THETA0),
    {"w_h": 0.0, "alpha": 0.0, "beta": 0.0, "lam": 0.0},
    {"w_h": 1.0, "alpha": 0.5, "beta": 0.0, "lam": 0.0},
    {"w_h": 1.0, "alpha": 0.0, "beta": 0.5, "lam": 0.0},
    {"w_h": 1.0, "alpha": 0.0, "beta": 0.0, "lam": 0.5},
    {"w_h": 0.5, "alpha": 0.3, "beta": 0.2, "lam": 0.4},
    {"w_h": 0.2, "alpha": 1.0, "beta": 1.0, "lam": 1.0},
]


def r(v):
    return np.round(np.asarray(v, dtype=float), DECIMALS).tolist()


def main():
    import joblib
    import lightgbm as lgb
    import shap
    rng = np.random.default_rng(SEED)
    data = Data()
    A = data.attrs
    test = data.interactions[data.interactions.split == "test"]
    pool = rng.choice(np.sort(test[test.label == 1].user_id.unique()), size=N_POOL, replace=False)
    snap = Snapshot(data, TEST_START, set(pool))
    truth = dict(eligible(snap, test[test.user_id.isin(set(pool))])[0])

    lr = joblib.load(MODELS / "logistic_pipeline.joblib")
    w = lr[-1].coef_[0] / lr[0].scale_
    b_lr = lr[-1].intercept_[0] - lr[0].mean_ @ w          # raw score = x @ w + b_lr
    gb = lgb.Booster(model_file=str(MODELS / "lightgbm_model.txt"))
    bg = np.load(CLEAN / "shap" / "background.npz")["x"]
    mu = bg.mean(axis=0)
    tree = shap.TreeExplainer(gb, data=shap.maskers.Independent(bg, max_samples=len(bg)),
                              feature_perturbation="interventional", model_output="raw")

    # Score the pool to pick cases
    info = {}
    for u in pool:
        if u not in truth:
            continue
        f, seen, cf = snap.user(u)
        ids = candidates(snap, f, seen, cf)
        rel = truth[u] & set(ids.tolist())
        if not rel:
            continue
        x = f[ids].astype(np.float64)
        s_lr, s_gb = x @ w + b_lr, gb.predict(x, raw_score=True)
        top_lr, top_gb = ids[steering.order(s_lr)[:10]], ids[steering.order(s_gb)[:10]]
        p = snap.profile[u]
        pop_pct = steering.ranks_from_order(np.lexsort((np.arange(len(ids)), snap.recent_pop[ids]))) - 1
        info[u] = dict(ids=ids, x=x, rel=rel, s_lr=s_lr, s_gb=s_gb, overlap=len(set(top_lr) & set(top_gb)),
                       pop_top=float((pop_pct / (len(ids) - 1))[steering.order(s_lr)[:10]].mean()),
                       home_share=float(np.mean(data.boro[top_lr] == p["boro"])), boro=int(p["boro"]))
    boro_labels = pd.factorize(A.boroname)[1]           # same coding as Data.boro
    manhattan = int(boro_labels.get_loc("Manhattan"))
    users = sorted(info)
    scripted = {}
    mh = [u for u in users if info[u]["boro"] == manhattan]
    scripted["local_manhattan"] = max(mh, key=lambda u: (info[u]["home_share"], -info[u]["pop_top"]))
    rest = [u for u in users if u not in scripted.values()]
    scripted["popularity_dominated"] = max(rest, key=lambda u: info[u]["pop_top"])
    rest = [u for u in rest if u not in scripted.values()]
    scripted["models_disagree"] = min(rest, key=lambda u: (info[u]["overlap"], u))
    rest = [u for u in rest if u not in scripted.values()]
    chosen = list(scripted.values()) + list(rng.choice(np.array(rest), size=N_CASES - 3, replace=False))
    label = {u: k for k, u in scripted.items()}

    LOCAL.mkdir(parents=True, exist_ok=True); DEMO.mkdir(parents=True, exist_ok=True)
    manifest, expected = [], {}
    cuisine_names = list(data.cuisine_names)
    nta_codes, nta_names = pd.factorize(A.nta2020)
    for k, u in enumerate(chosen, start=1):
        c = info[u]; ids, x = c["ids"], c["x"]
        p = snap.profile[u]
        phi_lr = (x - mu) * w
        phi_gb = tree.shap_values(x, check_additivity=False)
        base_gb = float(np.atleast_1d(tree.expected_value)[0])
        model_part = {}
        for m, phi, raw, base in (("lr", phi_lr, c["s_lr"], float(mu @ w + b_lr)), ("lgbm", phi_gb, c["s_gb"], base_gb)):
            collab = phi[:, [FEATURES.index(f) for f in COLLAB]].sum(axis=1)
            model_part[m] = {
                "base": round(base, DECIMALS), "raw": r(raw), "collab": r(collab), "noncollab": r(raw - collab),
                "shap_groups": {g: r(phi[:, [FEATURES.index(f) for f in FEATURE_GROUPS[g]]].sum(axis=1)) for g in GROUPS},
                "shap": {f: r(phi[:, j]) for j, f in enumerate(FEATURES)},
                "shap_residual_max": float(np.abs(base + phi.sum(axis=1) - raw).max())}
        km = haversine_km(p["lat"], p["lon"], data.lat[ids], data.lon[ids])
        pop_rank = steering.ranks_from_order(np.lexsort((np.arange(len(ids)), snap.recent_pop[ids]))) - 1
        sub = A.iloc[ids]
        cands = {
            "gmap_id": sub.gmap_id.tolist(), "name": sub.name.tolist(),
            "lat": r(sub.latitude), "lon": r(sub.longitude), "borough": sub.boroname.tolist(),
            "nta_name": sub.ntaname.tolist(), "cuisine_label": sub.cuisine_label.tolist(),
            "price": [None if np.isnan(v) else int(v) for v in sub.price_level], "is_chain": sub.is_chain.tolist(),
            "rating": r(snap.mean[ids]), "recent_rating": r(snap.recent_mean[ids]),
            "reviews": snap.pop[ids].astype(int).tolist(), "recent_reviews": snap.recent_pop[ids].astype(int).tolist(),
            "distance_km": r(km)}
        case = {
            "cutoff": TEST_START, "models": {"lr": "Logistic regression", "lgbm": "LightGBM"},
            "feature_groups": FEATURE_GROUPS, "features": FEATURES, "cuisine_names": cuisine_names,
            "candidates": cands,
            "cuisine": [int(v) for v in data.cuisine[ids]], "nta": [int(v) for v in nta_codes[ids]],
            "distance_n": r(np.minimum(km, steering.DIST_CAP_KM) / steering.DIST_CAP_KM),
            "popularity_n": r(pop_rank / (len(ids) - 1)),
            "feature_values": {f: r(x[:, j]) for j, f in enumerate(FEATURES)},
            "lr": model_part["lr"], "lgbm": model_part["lgbm"],
            "user": {"home_lat": round(p["lat"], 5), "home_lon": round(p["lon"], 5),
                     "home_borough": boro_labels[p["boro"]],
                     "history_size": int(p["n"]),
                     "cuisine_profile": {cuisine_names[i]: int(n) for i, n in enumerate(p["cuisine"]) if n}},
            "story": label.get(u, "random")}
        case_id = f"{k:02d}"
        full = dict(case, id=f"case_{case_id}",
                    user=dict(case["user"], user_id=u,
                              history=[{"gmap_id": data.items[i], "name": A.name.iloc[i], "lat": round(float(data.lat[i]), 5),
                                        "lon": round(float(data.lon[i]), 5), "cuisine_label": A.cuisine_label.iloc[i]}
                                       for i in p["seen"]]),
                    relevant=[bool(i in c["rel"]) for i in ids])
        (LOCAL / f"case_{case_id}.json").write_text(json.dumps(full))
        demo = dict(case, id=f"demo_{case_id}",
                    user=dict(case["user"], home_lat=round(p["lat"], 2), home_lon=round(p["lon"], 2)))
        km_demo = haversine_km(round(p["lat"], 2), round(p["lon"], 2), data.lat[ids], data.lon[ids])
        demo["candidates"] = dict(cands, distance_km=r(km_demo))
        demo["distance_n"] = r(np.minimum(km_demo, steering.DIST_CAP_KM) / steering.DIST_CAP_KM)
        path = DEMO / f"demo_{case_id}.json"
        path.write_text(json.dumps(demo, separators=(",", ":")))
        manifest.append(dict(id=demo["id"], file=path.name, story=demo["story"], home_borough=demo["user"]["home_borough"],
                             history_size=demo["user"]["history_size"]))

        # Replay expectations from the file exactly as the browser will read it
        reread = json.loads(path.read_text())
        sim = steering.similarity(reread)
        exp = []
        for th in REPLAY_THETAS:
            for m in ("lr", "lgbm"):
                fin = steering.final_ranks(reread, th, m, sim)
                log = steering.stage_log(reread, th, m, sim)
                sh = steering.shapley_controls(reread, th, m, sim)
                exp.append(dict(theta=th, model=m, top20=[int(i) for i in np.argsort(fin)[:20]],
                                stage_residual_max=int(np.abs(log["residual"]).max()),
                                shapley_residual_max=float(np.abs(sh["residual"]).max())))
        expected[demo["id"]] = exp
        print(f"case {case_id} ({demo['story']}): {len(ids)} candidates, {int(np.sum(full['relevant']))} later liked; "
              f"SHAP residual lr {model_part['lr']['shap_residual_max']:.1e}, lgbm {model_part['lgbm']['shap_residual_max']:.1e}",
              flush=True)

    (DEMO / "index.json").write_text(json.dumps(dict(cutoff=TEST_START, cases=manifest), indent=1))
    (ROOT / "tests" / "replay_expected.json").write_text(json.dumps(expected))
    worst_stage = max(e["stage_residual_max"] for v in expected.values() for e in v)
    worst_shap = max(e["shapley_residual_max"] for v in expected.values() for e in v)
    print(f"exported {len(chosen)} cases; max fixed-order residual {worst_stage}; max Shapley residual {worst_shap:.1e}")


if __name__ == "__main__":
    main()
