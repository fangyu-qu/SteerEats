#!/usr/bin/env python3
"""Step 32: how many liked places must a new user pick before recommendations are useful?

For test users with at least 10 liked restaurants (4-5 stars) before the snapshot, the taste profile is
rebuilt from only their k most recent liked places, k = 1, 3, 5, 10, as if they had just picked k places
in the onboarding. Everything the features take from the user (collaborative scores, usual area,
cuisine mix, price habit, chain share, history length) then comes from those k places alone. The three
final two-stage models rank the stage-1 candidates built from that reduced profile.

To compare like with like, every k is scored on the same users against the same held-out visits, and
restaurants the user had already reviewed are left out of the candidates at every k (a new user would
not have told us about them, but recommending a place they know is not a miss of the method).

Outputs: outputs/cold_start/COLD_START.md, cold_start.csv; web/public/data/cold_start.json
Run: .venv/bin/python src/32_cold_start.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS, ROOT, SEED, TEST_START
from models import logistic_scorer
from ranker_core import Data, Snapshot, candidates, eligible, metrics, profile_of

OUT = OUTPUTS / "cold_start"
MODELS_DIR = OUTPUTS / "rankers_two_stage"
KS = [1, 3, 5, 10]
MIN_LIKED = 10


def reduced_profile(d, items):
    """The profile Snapshot._profiles would build from these items alone, all of them liked."""
    return profile_of(d, items, items)


def boot(v, rng, n=2000):
    v = np.asarray(v, float)
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data = Data()
    s = data.interactions
    snap = Snapshot(data, TEST_START, set(s.user_id.unique()))
    cases, _ = eligible(snap, s[s.split == "test"])
    cases = [(u, rel) for u, rel in cases if len(snap.profile[u]["liked"]) >= MIN_LIKED]
    print(f"{len(cases):,} test users with at least {MIN_LIKED} liked places ({time.time() - t0:.0f}s)", flush=True)

    lr = joblib.load(MODELS_DIR / "logistic_pipeline.joblib")
    gb = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_model.txt"))
    rk = lgb.Booster(model_file=str(MODELS_DIR / "lambdarank_model.txt"))
    scorers = {"Logistic regression": logistic_scorer(lr), "LightGBM": lambda f: gb.predict(f, raw_score=True),
               "LambdaRank": lambda f: rk.predict(f, raw_score=True)}
    rows = []
    for n, (u, rel) in enumerate(cases):
        full = snap.profile[u]
        for k in KS + ["full"]:
            if k != "full":
                snap.profile[u] = reduced_profile(data, full["liked"][-k:])   # history is in time order
            try:
                f, _, cf = snap.user(u)
            finally:
                snap.profile[u] = full
            cand = candidates(snap, f, full["seen"], cf)
            recall = len(rel & set(cand.tolist())) / len(rel)
            for model, fn in scorers.items():
                sc = fn(f[cand])
                r, nd = metrics(cand[np.lexsort((cand, -sc))][:10].tolist(), rel)
                rows.append(dict(user=n, k=str(k), model=model, ndcg=nd, recall=r, stage1_recall=recall))
        if n % 1000 == 0:
            print(f"  {n}/{len(cases)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    out = []
    for (model, k), g in df.groupby(["model", "k"], sort=False):
        nd, lo, hi = boot(g.ndcg, rng)
        base = df[(df.model == model) & (df.k == "full")].set_index("user").ndcg
        d, dlo, dhi = boot((g.set_index("user").ndcg - base).to_numpy(), rng)
        out.append(dict(model=model, k=k, users=len(g), ndcg=nd, ndcg_lo=lo, ndcg_hi=hi, recall=float(g.recall.mean()),
                        stage1_recall=float(g.stage1_recall.mean()), diff_vs_full=d, diff_lo=dlo, diff_hi=dhi,
                        share_of_full=nd / base.mean()))
    out = pd.DataFrame(out)
    out.to_csv(OUT / "cold_start.csv", index=False)
    L = ["# Cold start: how many liked places does a new user need?", "",
         f"{len(cases):,} test users with at least {MIN_LIKED} liked restaurants before {TEST_START}. Their profile is rebuilt "
         "from only the k most recent liked places; \"full\" is their whole history. Same users and held-out visits at "
         "every k. Brackets are 95% bootstrap intervals over users; the difference is paired per user.", ""]
    for model in scorers:
        L += [f"## {model}", "", "| Liked places used | NDCG@10 | Share of full-history NDCG | Difference from full | Recall@10 | Stage-1 recall |",
              "|---|---|---|---|---|---|"]
        for r in out[out.model == model].itertuples():
            L.append(f"| {r.k} | {r.ndcg:.4f} [{r.ndcg_lo:.4f}, {r.ndcg_hi:.4f}] | {r.share_of_full:.0%} | "
                     f"{r.diff_vs_full:+.4f} [{r.diff_lo:+.4f}, {r.diff_hi:+.4f}] | {r.recall:.4f} | {r.stage1_recall:.3f} |")
        L.append("")
    L += ["Written by `src/32_cold_start.py`."]
    (OUT / "COLD_START.md").write_text("\n".join(L) + "\n")
    (ROOT / "web" / "public" / "data" / "cold_start.json").write_text(json.dumps(
        dict(users=len(cases), min_liked=MIN_LIKED, ks=[str(k) for k in KS] + ["full"], models=list(scorers),
             rows=json.loads(out.round(5).to_json(orient="records"))), separators=(",", ":")))
    print("\n".join(L))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
