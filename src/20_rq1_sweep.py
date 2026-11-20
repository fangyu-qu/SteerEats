#!/usr/bin/env python3
"""Step 20: RQ1 sweep. How do the history, distance, popularity and diversity controls change relevance,
diversity, geographic reach and popularity of the top 10, and do the two models trade off differently?

Setting: two-stage candidates (200 per user) of every eligible test user at the 2020-08-01 snapshot,
the controls of src/steering.py, the two-stage logistic regression and LightGBM models.

  One at a time  each control over 5 levels with the others at theta0; all eligible test users.
  Joint grid     w_h {0, .5, 1} x alpha {0, .25, .5, 1} x beta {0, .25, .5, 1} x lam {0, .125, .25, .5}
                 (192 settings) on a random 2,000-user subset; used for trade-off frontiers and
                 steering cost.
  Steering cost  (proposal 4.2) for each candidate neighbourhood n: the smallest L1 change of the
                 controls, on the grid, that puts a restaurant from n into the top 10 (lam measured
                 on its slider scale, lam / 0.5, so each control spans [0, 1]); unreachable if none.

Metrics on the top 10: NDCG@10 and Recall@10 against restaurants the user went on to like, mean
distance (km), share within 2 km, distinct cuisine groups and neighbourhoods, intra-list diversity
(1 - mean pairwise similarity used by MMR), mean popularity percentile, distinct boroughs, share
outside the home borough, overlap with the theta0 top 10, and catalog coverage across users.
Changes in NDCG@10 versus theta0 get 95% paired user-bootstrap intervals.

The history weight needs each score's collaborative SHAP part. Logistic regression uses the exact
closed form (step-17 background). For LightGBM, interventional SHAP on 1.6M rows is too slow, so the
sweep uses LightGBM's path-dependent contributions; the w_h results are re-run with interventional
values on the 300 step-17 users to show the difference.
Self-check: the fast top-10 routine must equal src/steering.py on all exported demo cases.
Outputs: outputs/rq1/ (CSVs, RQ1.md); per-user rows stay in data/clean/rq1/.
"""
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, ROOT, TEST_START
from ranker_core import FEATURE_GROUPS, FEATURES, Data, Snapshot, candidates, eligible, haversine_km
import steering

SEED = 42
B = 1000
N_GRID_USERS = 2000
LAM_MAX = 0.5
MODELS_DIR = OUTPUTS / "rankers_two_stage"
OUT = OUTPUTS / "rq1"
LOCAL = CLEAN / "rq1"
COLLAB = [FEATURES.index(f) for f in FEATURE_GROUPS["Collaborative"]]
LEVELS = {"w_h": [0.0, 0.25, 0.5, 0.75, 1.0], "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
          "beta": [0.0, 0.25, 0.5, 0.75, 1.0], "lam": [0.0, 0.125, 0.25, 0.375, 0.5]}
GRID = {"w_h": [0.0, 0.5, 1.0], "alpha": [0.0, 0.25, 0.5, 1.0], "beta": [0.0, 0.25, 0.5, 1.0],
        "lam": [0.0, 0.125, 0.25, 0.5]}
METRICS = ["ndcg", "recall", "distance_km", "within_2km", "cuisines", "neighbourhoods", "ild",
           "popularity", "boroughs", "outside_home_borough", "overlap_theta0"]


def minmax(v):
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi - lo <= 0 else (v - lo) / (hi - lo)


def top10(case, model, theta, k=10):
    """First k picks of steering.final ranking, computed without the full ordering."""
    z = case[model + "_noncollab"] + theta["w_h"] * case[model + "_collab"]
    s0 = minmax(z)
    s1 = s0 - theta["alpha"] * case["distance_n"]
    s2 = s1 - theta["beta"] * case["popularity_n"]
    rel = minmax(s2)
    lam = theta["lam"]
    if lam == 0:
        return np.lexsort((np.arange(len(rel)), -rel))[:k]
    cu, nt = case["cuisine"], case["nta"]
    max_sim = np.zeros(len(rel))
    taken = np.zeros(len(rel), bool)
    out = []
    for _ in range(k):
        v = (1 - lam) * rel - lam * max_sim
        v[taken] = -np.inf
        i = int(np.argmax(v))                       # first maximum = lowest candidate index, as in steering.py
        out.append(i); taken[i] = True
        max_sim = np.maximum(max_sim, steering.SIM_CUISINE * (cu == cu[i]) + steering.SIM_NTA * (nt == nt[i]))
    return np.array(out)


def self_check():
    files = sorted((ROOT / "web" / "public" / "data" / "cases").glob("demo_*.json"))
    rng = np.random.default_rng(0)
    checked = 0
    for f in files:
        c = json.loads(f.read_text())
        sim = steering.similarity(c)
        arr = {"distance_n": np.asarray(c["distance_n"]), "popularity_n": np.asarray(c["popularity_n"]),
               "cuisine": np.asarray(c["cuisine"]), "nta": np.asarray(c["nta"])}
        for m in ("lr", "lgbm"):
            arr[m + "_noncollab"] = np.asarray(c[m]["noncollab"]); arr[m + "_collab"] = np.asarray(c[m]["collab"])
        for _ in range(6):
            th = {"w_h": rng.choice(LEVELS["w_h"]), "alpha": rng.choice(LEVELS["alpha"]),
                  "beta": rng.choice(LEVELS["beta"]), "lam": rng.choice(LEVELS["lam"])}
            for m in ("lr", "lgbm"):
                ref = np.argsort(steering.final_ranks(c, th, m, sim))[:10]
                assert np.array_equal(ref, top10(arr, m, th)), f"fast top-10 differs on {f.name} {m} {th}"
                checked += 1
    return checked


def build_cases(data, users):
    import joblib
    import lightgbm as lgb
    snap = Snapshot(data, TEST_START, users)
    test = data.interactions[(data.interactions.split == "test") & data.interactions.user_id.isin(users)]
    truth = dict(eligible(snap, test)[0])
    lr = joblib.load(MODELS_DIR / "logistic_pipeline.joblib")
    w = lr[-1].coef_[0] / lr[0].scale_
    b = lr[-1].intercept_[0] - lr[0].mean_ @ w
    mu = np.load(CLEAN / "shap" / "background.npz")["x"].mean(axis=0)
    gb = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_model.txt"))
    cases = {}
    for u, rel in truth.items():
        f, seen, cf = snap.user(u)
        ids = candidates(snap, f, seen, cf)
        x = f[ids].astype(np.float64)
        p = snap.profile[u]
        lr_raw = x @ w + b
        lr_collab = ((x - mu) * w)[:, COLLAB].sum(axis=1)
        contrib = gb.predict(x, pred_contrib=True)
        gb_raw = contrib.sum(axis=1)
        gb_collab = contrib[:, COLLAB].sum(axis=1)
        pop_rank = steering.ranks_from_order(np.lexsort((np.arange(len(ids)), snap.recent_pop[ids]))) - 1
        km = haversine_km(p["lat"], p["lon"], data.lat[ids], data.lon[ids])
        cases[u] = dict(ids=ids, x=x, rel_idx=np.flatnonzero(np.isin(ids, list(rel))), n_rel=len(rel),
                        lr_noncollab=lr_raw - lr_collab, lr_collab=lr_collab,
                        lgbm_noncollab=gb_raw - gb_collab, lgbm_collab=gb_collab,
                        distance_n=np.minimum(km, steering.DIST_CAP_KM) / steering.DIST_CAP_KM, km=km,
                        popularity_n=pop_rank / (len(ids) - 1), cuisine=data.cuisine[ids], nta=data.nta[ids],
                        boro=data.boro[ids], home_boro=p["boro"])
    return cases


def stats(case, top, base_top):
    rel = set(case["rel_idx"].tolist())
    hits = [int(i in rel) for i in top]
    dcg = sum(h / np.log2(j + 2) for j, h in enumerate(hits))
    ideal = sum(1 / np.log2(j + 2) for j in range(min(10, case["n_rel"])))
    cu, nt = case["cuisine"][top], case["nta"][top]
    sim = steering.SIM_CUISINE * (cu[:, None] == cu[None, :]) + steering.SIM_NTA * (nt[:, None] == nt[None, :])
    off = sim[~np.eye(len(top), dtype=bool)]
    return dict(ndcg=dcg / ideal, recall=sum(hits) / case["n_rel"], distance_km=case["km"][top].mean(),
                within_2km=(case["km"][top] <= 2).mean(), cuisines=len(set(cu)), neighbourhoods=len(set(nt)),
                ild=1 - off.mean(), popularity=case["popularity_n"][top].mean(),
                boroughs=len(set(case["boro"][top])), outside_home_borough=(case["boro"][top] != case["home_boro"]).mean(),
                overlap_theta0=len(set(top) & set(base_top)) / 10)


def run(cases, thetas, label):
    rows, tops = [], {}
    for n, (u, c) in enumerate(cases.items()):
        for m in ("lr", "lgbm"):
            base = top10(c, m, steering.THETA0)
            for key, th in thetas.items():
                t = top10(c, m, th)
                tops[(u, m, key)] = c["ids"][t]
                rows.append(dict(user_id=u, model=m, setting=key, **th, **stats(c, t, base)))
        if (n + 1) % 1000 == 0:
            print(f"  {label}: {n + 1}/{len(cases)} users", flush=True)
    return pd.DataFrame(rows), tops


def paired_ci(df, rng, key0):
    out = []
    for (m, key), g in df.groupby(["model", "setting"]):
        base = df[(df.model == m) & (df.setting == key0)].set_index("user_id").ndcg
        d = (g.set_index("user_id").ndcg - base.loc[g.user_id].to_numpy()).to_numpy()
        idx = rng.integers(0, len(d), size=(B, len(d)))
        boot = d[idx].mean(axis=1)
        out.append(dict(model=m, setting=key, ndcg_change=d.mean(), lo=np.quantile(boot, .025), hi=np.quantile(boot, .975)))
    return pd.DataFrame(out)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True); LOCAL.mkdir(parents=True, exist_ok=True)
    print(f"self-check: fast top-10 equals src/steering.py on {self_check()} settings", flush=True)
    data = Data()
    test = data.interactions[data.interactions.split == "test"]
    cases = build_cases(data, set(test.user_id))
    print(f"{len(cases):,} eligible test users ({time.time() - t0:.0f}s)", flush=True)

    # One at a time
    one = {"theta0": dict(steering.THETA0)}
    for ctl, levels in LEVELS.items():
        for v in levels:
            th = dict(steering.THETA0, **{ctl: v})
            if th != steering.THETA0:
                one[f"{ctl}={v}"] = th
    df1, _ = run(cases, one, "one at a time")
    df1.to_parquet(LOCAL / "one_at_a_time_per_user.parquet", index=False)
    s1 = df1.groupby(["model", "setting"])[METRICS].mean().reset_index()
    ci = paired_ci(df1, rng, "theta0")
    s1 = s1.merge(ci, on=["model", "setting"])
    cov = df1.groupby(["model", "setting"]).size().rename("users").reset_index()
    s1 = s1.merge(cov, on=["model", "setting"])
    s1["control"] = s1.setting.str.split("=").str[0].replace({"theta0": "none"})
    s1["value"] = s1.setting.str.split("=").str[1].astype(float).fillna(-1)
    s1.to_csv(OUT / "one_at_a_time.csv", index=False)
    print(f"one-at-a-time done ({time.time() - t0:.0f}s)", flush=True)

    # Joint grid on a subset (+ catalog coverage, steering cost)
    sub_users = rng.choice(np.array(sorted(cases)), size=min(N_GRID_USERS, len(cases)), replace=False)
    sub = {u: cases[u] for u in sub_users}
    grid = {}
    for wh, a, bb, lm in product(GRID["w_h"], GRID["alpha"], GRID["beta"], GRID["lam"]):
        grid[f"w{wh}_a{a}_b{bb}_l{lm}"] = {"w_h": wh, "alpha": a, "beta": bb, "lam": lm}
    df2, tops = run(sub, grid, "joint grid")
    df2.to_parquet(LOCAL / "joint_grid_per_user.parquet", index=False)
    s2 = df2.groupby(["model", "setting", "w_h", "alpha", "beta", "lam"])[METRICS].mean().reset_index()
    coverage = pd.Series({(m, k): len(set(np.concatenate([tops[(u, m, k)] for u in sub_users]))) for m in ("lr", "lgbm") for k in grid})
    s2["catalog_coverage"] = [coverage[(m, k)] for m, k in zip(s2.model, s2.setting)]
    s2["l1_change"] = (1 - s2.w_h) + s2.alpha + s2.beta + s2.lam / LAM_MAX
    s2.to_csv(OUT / "joint_grid.csv", index=False)

    cost_rows = []
    nta_boro = np.zeros(data.nta.max() + 1, dtype=int)
    nta_boro[data.nta] = data.boro                      # each neighbourhood lies in one borough
    l1 = {k: (1 - th["w_h"]) + th["alpha"] + th["beta"] + th["lam"] / LAM_MAX for k, th in grid.items()}
    for u in sub_users:
        c = cases[u]
        ntas = np.unique(c["nta"])
        for m in ("lr", "lgbm"):
            best = {}
            for k in grid:
                for n in set(data.nta[tops[(u, m, k)]].tolist()):
                    if l1[k] < best.get(n, np.inf):
                        best[n] = l1[k]
            for n in ntas:
                cost_rows.append(dict(user_id=u, model=m, nta=int(n), cost=best.get(int(n), np.inf),
                                      home_borough=bool(nta_boro[n] == c["home_boro"])))
    cost = pd.DataFrame(cost_rows)
    cost.to_parquet(LOCAL / "steering_cost_per_user.parquet", index=False)
    finite = cost[np.isfinite(cost.cost)]
    sc = cost.groupby(["model", "home_borough"]).agg(neighbourhoods=("cost", "size"),
                                                     reachable=("cost", lambda s: np.isfinite(s).mean())).reset_index()
    sc = sc.merge(finite.groupby(["model", "home_borough"]).cost.agg(["mean", "median"]).reset_index(),
                  on=["model", "home_borough"])
    sc.to_csv(OUT / "steering_cost.csv", index=False)
    print(f"joint grid and steering cost done ({time.time() - t0:.0f}s)", flush=True)

    # Sensitivity: LightGBM collaborative split, interventional vs path-dependent (step-17 users)
    rows17 = pd.read_parquet(CLEAN / "shap" / "candidate_shap.parquet")
    sens = []
    for kind, prefix in (("interventional", "lgbm__"), ("path-dependent", "lgbm_path__")):
        sc_cases = {}
        for u, g in rows17.groupby("user_id", sort=False):
            if u not in cases:
                continue
            c = dict(cases[u])
            if not np.array_equal(c["ids"], np.array([data.ii[x] for x in g.gmap_id])):
                continue
            collab = g[[prefix + FEATURES[j] for j in COLLAB]].sum(axis=1).to_numpy()
            c["lgbm_collab"], c["lgbm_noncollab"] = collab, g.raw_lgbm.to_numpy() - collab
            sc_cases[u] = c
        th = {f"w_h={v}": dict(steering.THETA0, w_h=v) for v in LEVELS["w_h"]}
        d, _ = run(sc_cases, th, f"sensitivity {kind}")
        d = d[d.model == "lgbm"]
        sens.append(d.groupby("setting")[["ndcg", "overlap_theta0", "distance_km", "cuisines"]].mean().assign(
            collab_split=kind, users=d.user_id.nunique()))
    sens = pd.concat(sens).reset_index()
    sens.to_csv(OUT / "sensitivity_collab_split.csv", index=False)

    # Summary page
    name = {"lr": "Logistic regression", "lgbm": "LightGBM"}
    md = ["# RQ1: steering trade-offs", "", f"Generated by `src/20_rq1_sweep.py`. {len(cases):,} eligible test users, "
          "200 two-stage candidates each, 2020-08-01 snapshot. Values are means over users of top-10 metrics.", "",
          "## One control at a time", ""]
    for m in ("lr", "lgbm"):
        md += [f"### {name[m]}", "", "| Setting | NDCG@10 | ΔNDCG [95% CI] | km | cuisines | neighbourhoods | popularity | outside home borough | overlap with default |",
               "|---|---|---|---|---|---|---|---|---|"]
        g = s1[s1.model == m].sort_values(["control", "value"])
        for x in g.itertuples():
            md.append(f"| {x.setting} | {x.ndcg:.4f} | {x.ndcg_change:+.4f} [{x.lo:+.4f}, {x.hi:+.4f}] | {x.distance_km:.2f} | "
                      f"{x.cuisines:.2f} | {x.neighbourhoods:.2f} | {x.popularity:.3f} | {x.outside_home_borough:.1%} | {x.overlap_theta0:.1%} |")
        md.append("")
    md += ["## Steering cost (joint grid, 2,000 users)", "",
           "Smallest L1 change of the four controls (each on [0, 1]) that brings a restaurant from a candidate "
           "neighbourhood into the top 10.", "", "| Model | Neighbourhoods | In home borough | Reachable | Mean cost | Median cost |",
           "|---|---|---|---|---|---|"]
    md += [f"| {name[x.model]} | {x.neighbourhoods:,} | {'yes' if x.home_borough else 'no'} | {x.reachable:.1%} | {x['mean']:.2f} | {x['median']:.2f} |"
           for _, x in sc.iterrows()]
    md += ["", "## Sensitivity of the history-weight results to LightGBM's SHAP value function", "",
           "| w_h | Split | Users | NDCG@10 | overlap with default |", "|---|---|---|---|---|"]
    md += [f"| {x.setting} | {x.collab_split} | {x.users} | {x.ndcg:.4f} | {x.overlap_theta0:.1%} |"
           for x in sens.sort_values(["setting", "collab_split"]).itertuples()]
    (OUT / "RQ1.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
