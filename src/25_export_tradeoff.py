#!/usr/bin/env python3
"""Step 25: data for the trade-off view (V5). What does a user give up when steering?

Population layer, from the RQ1 joint grid (step 20; 192 settings x 2 models, 2,000 test users):
  mean top-10 outcomes per setting (NDCG@10, distance, popularity, cuisine groups, ...) and a paired
  95% user-bootstrap interval for the change in NDCG@10 against the model's own list (theta0).
  Pareto frontiers are recomputed in the browser from these rows; tests/check_tradeoff.mjs checks
  them against outputs/rq1/frontiers.csv.
Per-user layer: computed in the browser from the case file. Relevance for one user cannot be measured
(most users have 0 or 1 later visits among the candidates), so the view uses the model's own estimate:
predicted visits in the top 10 = sum of Platt-calibrated probabilities, with one Platt fit per model on
all test rows (step 23, platt_all). --cases-only writes tests/tradeoff_expected.json: for every demo
case, model and grid setting, the top-10 outcomes and predicted visits from src/steering.py, and
tests/preview_expected.json: the console previews (each control swept over its range, targets re-solved,
predicted visits kept) for two console states.

Writes web/public/data/tradeoff.json (aggregates only) and outputs/rq1/TRADEOFF.md.
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, ROOT
import steering

SEED = 42
B = 1000
OUT = OUTPUTS / "rq1"
WEB = ROOT / "web" / "public" / "data" / "tradeoff.json"
EXPECTED = ROOT / "tests" / "tradeoff_expected.json"
PREVIEW_EXPECTED = ROOT / "tests" / "preview_expected.json"
# console ranges, as web/src/preview.js
PREVIEW_VALUES = {"w_h": [round(k * 0.05, 2) for k in range(21)], "km": [round(0.5 + k * 0.25, 2) for k in range(23)],
                  "popularity": [None, 0.5, 0.4, 0.3, 0.2], "cuisines": [None, 5, 6, 7, 8]}
PREVIEW_STATES = {"default": (1.0, dict(steering.NO_TARGET)), "steered": (0.5, {"km": 1.5, "popularity": 0.3, "cuisines": 7})}
GRID = {"w_h": [0.0, 0.5, 1.0], "alpha": [0.0, 0.25, 0.5, 1.0], "beta": [0.0, 0.25, 0.5, 1.0],
        "lam": [0.0, 0.125, 0.25, 0.5]}           # step 20's joint grid, in the same order
KEEP = ["ndcg", "recall", "distance_km", "popularity", "cuisines", "neighbourhoods", "outside_home_borough", "overlap_theta0"]
NAME = {"Logistic regression": "lr", "LightGBM": "lgbm"}


def platt():
    cal = json.loads((OUTPUTS / "calibration" / "calibration.json").read_text())["deployed"]
    return {NAME[k]: cal[k]["platt_all"] for k in NAME}


def settings():
    return [dict(w_h=w, alpha=a, beta=b, lam=l) for w, a, b, l in product(*GRID.values())]


def case_expectations():
    ab = platt()
    out = {}
    for path in sorted((ROOT / "web" / "public" / "data" / "cases").glob("demo_*.json")):
        c = json.loads(path.read_text())
        out[c["id"]] = {}
        for m in ("lr", "lgbm"):
            a, b = ab[m]
            p = 1 / (1 + np.exp(-(a * np.asarray(c[m]["raw"]) + b)))
            rows = []
            for th in settings():
                top = steering.top_k(c, dict(steering.THETA0, **th), m)
                o = steering.outcomes(c, top)
                rows.append([o["km"], o["popularity"], o["cuisines"], float(sum(float(p[i]) for i in top))])
            out[c["id"]][m] = rows
    EXPECTED.write_text(json.dumps(out))
    print(f"wrote {EXPECTED.relative_to(ROOT)}: {len(out)} cases x 2 models x {len(settings())} settings")
    preview_expectations(ab)


def preview_expectations(ab):
    out = {}
    for path in sorted((ROOT / "web" / "public" / "data" / "cases").glob("demo_*.json")):
        c = json.loads(path.read_text())
        out[c["id"]] = {}
        for m in ("lr", "lgbm"):
            a, b = ab[m]
            p = 1 / (1 + np.exp(-(a * np.asarray(c[m]["raw"]) + b)))
            visits = lambda top: float(sum(float(p[i]) for i in top))
            base = visits(steering.top_k(c, steering.THETA0, m))
            for state, (w_h, targets) in PREVIEW_STATES.items():
                for key, values in PREVIEW_VALUES.items():
                    rows = []
                    for v in values:
                        t = targets if key == "w_h" else dict(targets, **{key: v})
                        s = steering.solve_targets(c, m, t, v if key == "w_h" else w_h)
                        o = s["outcomes"]
                        rows.append([visits(s["top"]) / base, True if key == "w_h" else bool(s["met"][key]),
                                     o["km"], o["popularity"], o["cuisines"]])
                    out[c["id"]][f"{m}|{state}|{key}"] = rows
    PREVIEW_EXPECTED.write_text(json.dumps(dict(states={k: dict(w_h=w, targets=t) for k, (w, t) in PREVIEW_STATES.items()}, cases=out)))
    print(f"wrote {PREVIEW_EXPECTED.relative_to(ROOT)}: {len(out)} cases x 2 models x {len(PREVIEW_STATES)} states x 4 controls")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-only", action="store_true")
    if ap.parse_args().cases_only:
        return case_expectations()
    rng = np.random.default_rng(SEED)
    grid = pd.read_csv(OUT / "joint_grid.csv")
    per = pd.read_parquet(CLEAN / "rq1" / "joint_grid_per_user.parquet")
    users = sorted(per.user_id.unique())
    idx = rng.integers(0, len(users), size=(B, len(users)))
    keys = [f"w{s['w_h']}_a{s['alpha']}_b{s['beta']}_l{s['lam']}" for s in settings()]
    model_out = {}
    for m in ("lr", "lgbm"):
        wide = per[per.model == m].pivot(index="user_id", columns="setting", values="ndcg").loc[users]
        base = wide["w1.0_a0.0_b0.0_l0.0"].to_numpy()
        g = grid[grid.model == m].set_index("setting")
        rows = []
        for k, s in zip(keys, settings()):
            d = wide[k].to_numpy() - base
            boot = d[idx].mean(axis=1)
            r = g.loc[k]
            assert abs(wide[k].mean() - r.ndcg) < 1e-12, k
            rows.append(dict(**s, **{q: float(r[q]) for q in KEEP}, l1=float(r.l1_change),
                             d_ndcg=float(d.mean()), lo=float(np.quantile(boot, .025)), hi=float(np.quantile(boot, .975))))
        model_out[m] = rows
    out = dict(users=len(users), bootstrap=B, grid=GRID, lam_max=steering.LAM_MAX, platt=platt(), settings=model_out)
    WEB.write_text(json.dumps(out, indent=1))

    md = ["# Trade-off view data (V5)", "",
          f"Generated by `src/25_export_tradeoff.py` from the RQ1 joint grid: {len(keys)} settings per model, "
          f"{len(users):,} test users; paired {B}-sample user bootstrap for the change in NDCG@10 against the model's "
          "own list.", ""]
    for m, label in (("lr", "Logistic regression"), ("lgbm", "LightGBM")):
        rows = pd.DataFrame(model_out[m])
        free = rows[(rows.hi >= 0) & (rows.l1 > 0)]
        md += [f"## {label}", "",
               f"Settings whose NDCG@10 change is not distinguishable from zero (interval includes 0): {len(free)} of {len(rows) - 1}.", ""]
        if len(free):
            md += ["| w_h | alpha | beta | lam | NDCG@10 | change [95% CI] | km | popularity | cuisines |", "|---|---|---|---|---|---|---|---|---|"]
            md += [f"| {r.w_h:g} | {r.alpha:g} | {r.beta:g} | {r.lam:g} | {r.ndcg:.4f} | {r.d_ndcg:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] | "
                   f"{r.distance_km:.2f} | {r.popularity:.3f} | {r.cuisines:.2f} |" for r in free.itertuples()]
            md.append("")
    (OUT / "TRADEOFF.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"wrote {WEB.relative_to(ROOT)}")
    case_expectations()


if __name__ == "__main__":
    main()
