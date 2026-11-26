#!/usr/bin/env python3
"""Behavioural check of the steering controls on the exported demo cases.

Each control is swept alone from 0 to 1 (others at theta0). Averaged over cases and both models, the
top 10 should move the intended way:
  alpha (distance)   mean distance falls;
  beta (popularity)  mean popularity percentile falls;
  lam (diversity)    distinct cuisine groups and neighbourhoods rise;
  w_h (history)      lowering it reduces the collaborative share of the top 10's scores.
Prints the sweep table and fails if a direction is wrong at the extremes.
Run: .venv/bin/python tests/check_steering.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import steering

GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def top10_stats(case, theta, model, sim):
    ranks = steering.final_ranks(case, theta, model, sim)
    top = np.argsort(ranks)[:10]
    m = case[model]
    collab = np.abs(np.asarray(m["collab"])[top]).sum() / (np.abs(np.asarray(m["raw"])[top]).sum() + 1e-12)
    return dict(distance_km=float(np.mean(np.asarray(case["candidates"]["distance_km"])[top])),
                popularity=float(np.mean(np.asarray(case["popularity_n"])[top])),
                cuisines=len(set(np.asarray(case["cuisine"])[top])), neighbourhoods=len(set(np.asarray(case["nta"])[top])),
                collab_share=float(collab))


def main():
    files = sorted((ROOT / "web" / "public" / "data" / "cases").glob("demo_*.json"))
    rows = []
    for f in files:
        case = json.loads(f.read_text())
        sim = steering.similarity(case)
        for model in case["models"] if "models" in case else ("lr", "lgbm"):
            for control in steering.CONTROLS:
                for v in GRID:
                    th = dict(steering.THETA0, **{control: v})
                    rows.append(dict(case=f.stem, model=model, control=control, value=v, **top10_stats(case, th, model, sim)))
    df = pd.DataFrame(rows)
    table = df.groupby(["control", "value"])[["distance_km", "popularity", "cuisines", "neighbourhoods", "collab_share"]].mean()
    print(table.round(3).to_string())
    at = lambda c, v, k: table.loc[(c, v), k]
    checks = {
        "alpha lowers distance": at("alpha", 1.0, "distance_km") < at("alpha", 0.0, "distance_km"),
        "beta lowers popularity": at("beta", 1.0, "popularity") < at("beta", 0.0, "popularity"),
        "lam raises cuisine variety": at("lam", 1.0, "cuisines") > at("lam", 0.0, "cuisines"),
        "lam raises neighbourhood variety": at("lam", 1.0, "neighbourhoods") > at("lam", 0.0, "neighbourhoods"),
        "lower w_h lowers collaborative share": at("w_h", 0.0, "collab_share") < at("w_h", 1.0, "collab_share"),
    }
    for k, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + k)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
