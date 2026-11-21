#!/usr/bin/env python3
"""Step 27: the three scripted case studies, one per research question, computed on the committed demo cases.

Every number comes from src/steering.py (the browser mirrors it exactly; tests/replay.mjs) on
web/public/data/cases/demo_0{1,2,3}.json, so each case's deep link reproduces it in the interface.
Relevance for one user is the model's own estimate: predicted visits in the top 10, Platt-calibrated with
step 23's single fit (platt_all), as a share of the model's own list. "Later liked" places are the user's
4-5 star reviews in the test period (shown in the demo, never used by the model).

  1. Manhattan regular (demo_01): steering outward. Which other-borough neighbourhoods can the four controls
     reach on the RQ1 grid, and what does the neighbourhood boost need?
  2. Popular-places fan (demo_02): "less mainstream" (mean popularity percentile at most 0.40), per model.
  3. Models disagree (demo_03): the two models' lists and reasons, then one shared target (within 2 km).
Writes outputs/case_studies/CASE_STUDIES.md.
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS, ROOT
import steering

CASES = ROOT / "web" / "public" / "data" / "cases"
OUT = OUTPUTS / "case_studies"
NAME = {"lr": "Logistic regression", "lgbm": "LightGBM"}
GRID = {"w_h": [0.0, 0.5, 1.0], "alpha": [0.0, 0.25, 0.5, 1.0], "beta": [0.0, 0.25, 0.5, 1.0], "lam": [0.0, 0.125, 0.25, 0.5]}
PLATT = {"lr": "Logistic regression", "lgbm": "LightGBM"}


def load(cid):
    return json.loads((CASES / f"{cid}.json").read_text())


def probs(c, m):
    a, b = json.loads((OUTPUTS / "calibration" / "calibration.json").read_text())["deployed"][PLATT[m]]["platt_all"]
    return 1 / (1 + np.exp(-(a * np.asarray(c[m]["raw"]) + b)))


def top10(c, theta, m):
    return [int(i) for i in steering.top_k(c, dict(steering.THETA0, **theta), m)]


def describe(c, m, theta, base=None):
    """Outcomes of the top 10 under theta, relative to the model's own list."""
    base = base or top10(c, {}, m)
    top = top10(c, theta, m)
    p = probs(c, m)
    o = steering.outcomes(c, top)
    liked = c["later_liked"]
    return dict(top=top, entered=[i for i in top if i not in base], left=[i for i in base if i not in top],
                km=o["km"], popularity=o["popularity"], cuisines=o["cuisines"],
                kept=float(sum(p[i] for i in top) / sum(p[i] for i in base)),
                liked=sum(bool(liked[i]) for i in top), boroughs=sorted({c["candidates"]["borough"][i] for i in top}))


def names(c, ids, k=4):
    out = [f'{c["candidates"]["name"][i]} ({c["candidates"]["nta_name"][i]})' for i in ids[:k]]
    return ", ".join(out) + (f" and {len(ids) - k} more" if len(ids) > k else "")


def pct(v):
    # never round a value below 100% up to "100%"
    return f"{v * 100:.1f}%" if v < 1 and round(v * 100) >= 100 else f"{v * 100:.0f}%"


def case_outward(md):
    c, m = load("demo_01"), "lgbm"
    home = c["user"]["home_borough"]
    cand = c["candidates"]
    d0 = describe(c, m, {})
    inlist = {int(c["nta"][i]) for i in d0["top"]}
    # neighbourhoods outside the home borough that the model's own list does not already reach
    other = sorted({int(c["nta"][i]) for i in range(len(c["nta"])) if cand["borough"][i] != home} - inlist)
    reach = {}
    for wh, a, b, l in product(*GRID.values()):
        th = dict(w_h=wh, alpha=a, beta=b, lam=l)
        cost = (1 - wh) + a + b + l / steering.LAM_MAX
        for i in top10(c, th, m):
            n = int(c["nta"][i])
            if n in other and cost < reach.get(n, (np.inf,))[0]:
                reach[n] = (cost, th)
    boosts = {}
    for n in other:
        t, ok = steering.solve_boost(c, m, dict(steering.THETA0), n)
        if ok:
            boosts[n] = t["boost"]
    best = min(boosts, key=lambda n: (boosts[n], n))
    tb, _ = steering.solve_boost(c, m, dict(steering.THETA0), best)
    db = describe(c, m, tb)
    lr_b, lr_ok = steering.solve_boost(c, "lr", dict(steering.THETA0), best)
    nta_name = cand["nta_name"][c["nta"].index(best)]
    md += ["## 1. Manhattan regular: steering outward (demo_01, LightGBM)", "",
           f"The model's own list: mean distance {d0['km']:.1f} km, boroughs {', '.join(d0['boroughs'])}, "
           f"{d0['cuisines']} cuisine groups, {d0['liked']} later-liked places.", "",
           f"- The candidates include {len(other)} neighbourhoods outside {home} that the list does not reach. On the 192-setting grid the four indirect "
           f"controls bring a restaurant from {len(reach)} of them into the top 10"
           + (f" (cheapest L1 change {min(v[0] for v in reach.values()):.2f})." if reach else "."),
           f"- The neighbourhood boost reaches {len(boosts)} of {len(other)}. The cheapest is **{nta_name}**: boost "
           f"+{tb['boost']:.2f} for LightGBM, " + (f"+{lr_b['boost']:.2f} for logistic regression." if lr_ok else "not reachable for logistic regression."),
           f"- With that boost: {len(db['entered'])} {'place enters' if len(db['entered']) == 1 else 'places enter'} ({names(c, db['entered'])}), {len(db['left'])} {'leaves' if len(db['left']) == 1 else 'leave'} "
           f"({names(c, db['left'])}); mean distance {d0['km']:.1f} → {db['km']:.1f} km; the list keeps {pct(db['kept'])} of the "
           f"model's predicted visits; later-liked places {d0['liked']} → {db['liked']}.",
           f"- Deep link: `#case=demo_01&model=lgbm&boost={best}`", ""]


def case_mainstream(md):
    c = load("demo_02")
    md += ["## 2. Popular-places fan: less mainstream (demo_02, both models)", "",
           "Target: mean popularity percentile of the top 10 at most 0.40.", "",
           "| Model | Own list: popularity | Solved β | Popularity | Mean km | Cuisine groups | Entered | Predicted visits kept | Later liked |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in ("lr", "lgbm"):
        d0 = describe(c, m, {})
        s = steering.solve_targets(c, m, {"km": None, "popularity": 0.4, "cuisines": None})
        d = describe(c, m, {k: s["theta"][k] for k in ("w_h", "alpha", "beta", "lam")})
        md.append(f"| {NAME[m]} | {d0['popularity']:.2f} | {s['theta']['beta']:.2f} | {d['popularity']:.2f} | {d0['km']:.1f} → {d['km']:.1f} | "
                  f"{d0['cuisines']} → {d['cuisines']} | {len(d['entered'])} | {pct(d['kept'])} | {d0['liked']} → {d['liked']} |")
    md += ["", "Deep link: `#case=demo_02&model=lr&pop=0.4` (switch the model in the top bar).", ""]


def case_disagree(md):
    c = load("demo_03")
    lens = __import__("importlib").import_module("22_model_lens")
    a, b = top10(c, {}, "lr"), top10(c, {}, "lgbm")
    tau = kendalltau(c["lr"]["raw"], c["lgbm"]["raw"]).statistic
    sh = {m: lens.group_shares(c, m) for m in ("lr", "lgbm")}
    gap = sorted(sh["lr"], key=lambda g: abs(sh["lr"][g] - sh["lgbm"][g]), reverse=True)[:2]
    md += ["## 3. Models disagree (demo_03)", "",
           f"- Own lists share {len(set(a) & set(b))} of 10 places; Kendall τ over all {len(c['lr']['raw'])} candidates {tau:.2f}.",
           "- Largest differences in SHAP reliance (share of mean |SHAP|): "
           + "; ".join(f"{g}: logistic regression {pct(sh['lr'][g])}, LightGBM {pct(sh['lgbm'][g])}" for g in gap) + ".",
           f"- Only logistic regression: {names(c, [i for i in a if i not in b])}. Only LightGBM: {names(c, [i for i in b if i not in a])}.", ""]
    t = {"km": 2.0, "popularity": None, "cuisines": None}
    s = {m: steering.solve_targets(c, m, t) for m in ("lr", "lgbm")}
    d = {m: describe(c, m, {k: s[m]["theta"][k] for k in ("w_h", "alpha", "beta", "lam")}) for m in ("lr", "lgbm")}
    md += ["Shared target: within 2 km.", "", "| Model | Solved α | Mean km | Met | Predicted visits kept | Later liked |", "|---|---|---|---|---|---|"]
    md += [f"| {NAME[m]} | {s[m]['theta']['alpha']:.2f} | {d[m]['km']:.1f} | {'yes' if s[m]['met']['km'] else 'no'} | {pct(d[m]['kept'])} | {d[m]['liked']} |"
           for m in ("lr", "lgbm")]
    md += ["", f"Under the shared target the two lists share {len(set(d['lr']['top']) & set(d['lgbm']['top']))} of 10 places.",
           "Deep link: `#case=demo_03&model=lgbm&km=2`", ""]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Case studies", "", "Generated by `src/27_case_studies.py` from the committed demo cases with `src/steering.py`. "
          "Predicted visits kept: the model's Platt-calibrated estimate for this user, as a share of its own list.", ""]
    case_outward(md)
    case_mainstream(md)
    case_disagree(md)
    (OUT / "CASE_STUDIES.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
