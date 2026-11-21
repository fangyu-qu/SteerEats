#!/usr/bin/env python3
"""Step 28: user study (plan section 4, "Usefulness"): task definitions with computed answers, and analysis.

  python src/28_user_study.py --tasks     write web/public/data/study.json (tasks and their correct answers)
  python src/28_user_study.py             analyse the result files in data/user_study/*.json (git-ignored)

Design: 3-5 participants, within subjects, two conditions: the full interface and a list-only baseline (the
console and the resulting top 10 only, like TasteWeights' input and output columns, no explanation views).
Two case sets (A, B) so no participant sees the same user twice; condition order and set are counterbalanced
by participant number. Three tasks per set, the plan's three:
  why    a place left the top 10 under preset targets; which control pushed it out? Correct = the control with
         at least 70% of its Shapley share of the rank change (src/steering.py), with two or more controls active.
  reach  get a restaurant from a named borough into the top 10 while keeping at least 8 of the model's own 10.
         Feasible in both conditions by construction: the script finds a console state (history weight, distance,
         popularity and cuisine targets, no map boost, which the list-only condition lacks) that does it.
  model  which model leans more on a feature group (the one where the models differ most for this user; the
         plan's example is popularity)? Correct = the larger share of mean |SHAP| (step 22's group shares),
         chosen only where the gap is at least 5 points.
Every answer is scored in the browser at "Done"; times and interaction counts are logged. Analysis is
descriptive (success rate, median time per task and condition, post-condition ratings); with at most five
people no test is run. If no result files exist the script says so and writes nothing.
"""
import argparse
import importlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS, ROOT
import steering

CASES = ROOT / "web" / "public" / "data" / "cases"
WEB = ROOT / "web" / "public" / "data" / "study.json"
RESULTS = ROOT / "data" / "user_study"
OUT = OUTPUTS / "user_study"
CONTROL_NAME = {"w_h": "History weight", "alpha": "Distance", "beta": "Popularity", "lam": "Variety"}
TARGET_STATES = [  # (w_h, targets) tried for the "why" task, in order
    (1.0, {"km": 1.5, "popularity": 0.4, "cuisines": None}), (1.0, {"km": 1.0, "popularity": None, "cuisines": 6}),
    (1.0, {"km": None, "popularity": 0.3, "cuisines": 6}), (0.5, {"km": 2.0, "popularity": None, "cuisines": None}),
    (0.5, {"km": None, "popularity": 0.4, "cuisines": None})]
GRID = {"w_h": [0.0, 0.5, 1.0], "alpha": [0.0, 0.25, 0.5, 1.0], "beta": [0.0, 0.25, 0.5, 1.0], "lam": [0.0, 0.125, 0.25, 0.5]}
MODEL = "lgbm"   # the why and reach tasks start on LightGBM


def load(cid):
    return json.loads((CASES / f"{cid}.json").read_text())


def why_task(c):
    sim = steering.similarity(c)
    for w_h, t in TARGET_STATES:
        s = steering.solve_targets(c, MODEL, t, w_h)
        if not all(s["met"].values()):
            continue
        th = s["theta"]
        active = [k for k in ("w_h", "alpha", "beta", "lam") if (th[k] != 1.0 if k == "w_h" else th[k] > 0)]
        if len(active) < 2:
            continue
        sh = steering.shapley_controls(c, th, MODEL, sim)
        before = steering.final_ranks(c, steering.THETA0, MODEL, sim)
        after = steering.final_ranks(c, th, MODEL, sim)
        for i in np.argsort(before)[:10]:
            i = int(i)
            if after[i] <= 10:
                continue
            total = sh["total"][i]
            share = {k: sh["phi"][k][i] / total for k in active}
            k, v = max(share.items(), key=lambda kv: kv[1])
            if v >= 0.7:
                return dict(kind="why", case=c["id"], state=dict(model=MODEL, wh=w_h, km=t["km"], pop=t["popularity"], cui=t["cuisines"]),
                            place=c["candidates"]["name"][i], nta=c["candidates"]["nta_name"][i], index=i,
                            options=[CONTROL_NAME[a] for a in ("w_h", "alpha", "beta", "lam")], answer=CONTROL_NAME[k],
                            evidence=dict(share=round(float(v), 3), rank_before=int(before[i]), rank_after=int(after[i])))
    return None


CONSOLE = dict(w_h=[round(k * 0.1, 1) for k in range(10, -1, -1)], km=[None] + [round(0.5 + k * 0.25, 2) for k in range(23)],
               popularity=[None, 0.5, 0.4, 0.3, 0.2], cuisines=[None, 5, 6, 7, 8])


def reach_task(c):
    home = c["user"]["home_borough"]
    base = set(int(i) for i in steering.top_k(c, steering.THETA0, MODEL))
    bor = c["candidates"]["borough"]
    inlist = {bor[i] for i in base}
    for target in sorted({b for b in bor if b != home and b not in inlist}):
        for w_h, km, pop, cui in product(*CONSOLE.values()):
            t = {"km": km, "popularity": pop, "cuisines": cui}
            s = steering.solve_targets(c, MODEL, t, w_h)
            if any(bor[i] == target for i in s["top"]) and len(set(s["top"]) & base) >= 8:
                return dict(kind="reach", case=c["id"], state=dict(model=MODEL), borough=target, keep=8,
                            evidence=dict(console=dict(wh=w_h, **t)))
    return None


GROUP_TEXT = {"Collaborative": "your history (places like ones you reviewed)", "Popularity": "popularity and recency",
              "Rating": "ratings", "Location": "location", "Taste": "taste match", "Attributes": "amenities"}


def model_task(c, lens):
    sh = {m: lens.group_shares(c, m) for m in ("lr", "lgbm")}
    g = max(sh["lr"], key=lambda k: abs(sh["lr"][k] - sh["lgbm"][k]))
    if abs(sh["lr"][g] - sh["lgbm"][g]) < 0.05:
        return None
    return dict(kind="model", case=c["id"], state=dict(model="lr"), group=g, group_text=GROUP_TEXT.get(g, g),
                options=["Logistic regression", "LightGBM"],
                answer="Logistic regression" if sh["lr"][g] > sh["lgbm"][g] else "LightGBM",
                evidence={m: round(sh[m][g], 3) for m in sh})


def make_tasks():
    lens = importlib.import_module("22_model_lens")
    index = json.loads((CASES / "index.json").read_text())
    pool = [x["id"] for x in index["cases"] if x["story"] == "random"]   # the three story cases stay for the talk
    chosen = {"why": [], "reach": [], "model": []}
    used = set()
    for kind, fn in (("why", why_task), ("reach", reach_task), ("model", lambda c: model_task(c, lens))):
        for cid in pool:
            if cid in used or len(chosen[kind]) == 2:
                continue
            t = fn(load(cid))
            if t:
                chosen[kind].append(t)
                used.add(cid)
    missing = [k for k, v in chosen.items() if len(v) < 2]
    if missing:
        sys.exit(f"Not enough demo cases for tasks: {missing}. Nothing written.")
    sets = {s: [chosen[k][j] for k in ("why", "reach", "model")] for j, s in enumerate("AB")}
    WEB.write_text(json.dumps(dict(version=1, sets=sets, likert=[
        "I understood why the list changed.", "I could steer the list toward what I wanted.", "I trust the recommendations."]), indent=1))
    print(f"wrote {WEB.relative_to(ROOT)}")
    for s, ts in sets.items():
        for t in ts:
            print(f"  set {s} {t['kind']}: {t['case']} answer={t.get('answer') or t.get('borough')} evidence={t['evidence']}")


def analyse():
    files = sorted(RESULTS.glob("*.json"))
    if not files:
        print(f"No result files in {RESULTS.relative_to(ROOT)}; run sessions first (docs/USER_STUDY.md). Nothing written.")
        return
    rows, ratings = [], []
    for f in files:
        r = json.loads(f.read_text())
        for t in r["tasks"]:
            rows.append(dict(p=r["participant"], condition=t["condition"], kind=t["kind"], ok=bool(t["correct"]), sec=t["seconds"],
                             actions=t["actions"]))
        for cond, vals in r.get("ratings", {}).items():
            for q, v in vals.items():
                ratings.append(dict(p=r["participant"], condition=cond, q=q, v=v))
    import pandas as pd
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    g = d.groupby(["kind", "condition"]).agg(n=("ok", "size"), success=("ok", "mean"), median_s=("sec", "median"),
                                              median_actions=("actions", "median")).reset_index()
    g.to_csv(OUT / "summary.csv", index=False)
    md = ["# User study results", "", f"Generated by `src/28_user_study.py` from {len(files)} session file(s). Descriptive only.", "",
          "| Task | Condition | n | Success | Median time (s) | Median actions |", "|---|---|---|---|---|---|"]
    md += [f"| {r.kind} | {r.condition} | {r.n} | {r.success:.0%} | {r.median_s:.0f} | {r.median_actions:.0f} |" for r in g.itertuples()]
    if ratings:
        rt = pd.DataFrame(ratings).groupby(["q", "condition"]).v.median().unstack()
        md += ["", "Median rating (1 = strongly disagree, 7 = strongly agree)", "", "| Statement | " + " | ".join(rt.columns) + " |",
               "|---" * (len(rt.columns) + 1) + "|"]
        md += [f"| {q} | " + " | ".join(f"{v:g}" for v in rt.loc[q]) + " |" for q in rt.index]
    (OUT / "USER_STUDY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", action="store_true")
    make_tasks() if ap.parse_args().tasks else analyse()
