#!/usr/bin/env python3
"""Step 14: uncertainty and robustness for the rankers trained in step 13.

1. Test metrics with 95% user-level bootstrap intervals (2,000 resamples) and paired
   comparisons (bootstrap interval of the mean difference, Wilcoxon signed-rank test).
2. Robustness, re-ranking with the saved models (scores are computed once per user and
   scored against several target definitions):
   - test without bulk-upload targets (posting time may differ from visit time);
   - test by period: Aug-Oct 2020, Nov 2020-Jan 2021, Feb-Apr 2021 (same 2020-08-01 snapshot,
     so later periods are also a longer horizon);
   - validation window split at the COVID shutdown, using the train-window-only models:
     pre-COVID (Dec 2019 - Feb 2020) vs COVID (Mar - Jul 2020).
Outputs: outputs/rankers/test_intervals.csv, paired_tests.csv, robustness.csv, EVALUATION.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, TEST_START, VAL_START
from ranker_core import Data, Snapshot, eligible, metrics

OUT = OUTPUTS / "rankers"
B = 2000
SEED = 42
MODELS = ["Popularity", "Recent popularity", "Item-CF", "Logistic regression", "LightGBM"]
PAIRS = [("Logistic regression", "Item-CF"), ("LightGBM", "Item-CF"), ("LightGBM", "Logistic regression")]


def intervals(df, rng):
    rows = []
    users = df.user_id.unique()
    idx = rng.integers(0, len(users), size=(B, len(users)))
    for m in MODELS:
        d = df[df.model == m].set_index("user_id").loc[users]
        for metric in ("recall", "ndcg"):
            v = d[metric].to_numpy()
            boot = v[idx].mean(axis=1)
            rows.append(dict(model=m, metric=metric, mean=v.mean(), lo=np.quantile(boot, 0.025),
                             hi=np.quantile(boot, 0.975), users=len(v)))
    return pd.DataFrame(rows)


def paired(df, rng):
    from scipy.stats import wilcoxon
    rows = []
    wide = {m: df[df.model == m].set_index("user_id").sort_index() for m in MODELS}
    n = len(wide[MODELS[0]])
    idx = rng.integers(0, n, size=(B, n))
    for a, b in PAIRS:
        for metric in ("recall", "ndcg"):
            diff = wide[a][metric].to_numpy() - wide[b][metric].to_numpy()
            boot = diff[idx].mean(axis=1)
            nz = diff[diff != 0]
            p = wilcoxon(nz).pvalue if len(nz) else 1.0
            rows.append(dict(model_a=a, model_b=b, metric=metric, mean_diff=diff.mean(),
                             lo=np.quantile(boot, 0.025), hi=np.quantile(boot, 0.975),
                             relative=diff.mean() / wide[b][metric].mean(), wilcoxon_p=p,
                             users_a_better=int((diff > 0).sum()), users_b_better=int((diff < 0).sum())))
    return pd.DataFrame(rows)


def multi_evaluate(snap, variants, scorers, k=10, batch=64):
    """Score each user once, then compute metrics for every target variant they are eligible in."""
    cases = {name: dict(eligible(snap, t)[0]) for name, t in variants.items()}
    users = sorted(set().union(*[set(c) for c in cases.values()]))
    sums = {(v, m): np.zeros(2) for v in variants for m in MODELS}
    counts = {v: len(c) for v, c in cases.items()}
    for b0 in range(0, len(users), batch):
        chunk = users[b0:b0 + batch]
        feats, cands, cfs = [], [], []
        for u in chunk:
            f, seen, cf = snap.user(u)
            avail = snap.available.copy()
            avail[seen] = False
            ids = np.flatnonzero(avail)
            feats.append(f[ids]); cands.append(ids); cfs.append(cf[ids])
        stacked = np.vstack(feats)
        bounds = np.cumsum([0] + [len(c) for c in cands])
        scores = {name: fn(stacked) for name, fn in scorers.items()}
        for j, u in enumerate(chunk):
            ids = cands[j]
            top = {"Popularity": ids[np.lexsort((ids, -snap.pop[ids]))][:k],
                   "Recent popularity": ids[np.lexsort((ids, -snap.recent_pop[ids]))][:k],
                   "Item-CF": ids[np.lexsort((ids, -snap.pop[ids], -cfs[j]))][:k]}
            for name, sc in scores.items():
                top[name] = ids[np.lexsort((ids, -sc[bounds[j]:bounds[j + 1]]))][:k]
            for v, c in cases.items():
                if u in c:
                    for m in MODELS:
                        sums[(v, m)] += metrics(top[m].tolist(), c[u], k)
        if (b0 // batch) % 30 == 0:
            print(f"  {min(b0 + batch, len(users))}/{len(users)} users", flush=True)
    return [dict(variant=v, model=m, users=counts[v], **{"Recall@10": sums[(v, m)][0] / max(counts[v], 1),
                                                          "NDCG@10": sums[(v, m)][1] / max(counts[v], 1)})
            for v in variants for m in MODELS]


def scorers_from(lr_path, gb_path):
    import joblib
    import lightgbm as lgb
    lr, gb = joblib.load(lr_path), lgb.Booster(model_file=str(gb_path))
    sc, lin = lr[0], lr[-1]
    return {"Logistic regression": lambda f: ((f - sc.mean_) / sc.scale_) @ lin.coef_[0] + lin.intercept_[0],
            "LightGBM": lambda f: gb.predict(f, raw_score=True)}


def main():
    rng = np.random.default_rng(SEED)
    per_user = pd.read_csv(OUT / "test_per_user.csv")
    ci = intervals(per_user, rng)
    pt = paired(per_user, rng)
    ci.to_csv(OUT / "test_intervals.csv", index=False)
    pt.to_csv(OUT / "paired_tests.csv", index=False)
    print(ci.round(4).to_string(index=False)); print(pt.round(4).to_string(index=False), flush=True)

    data = Data()
    s = data.interactions
    users = set(s.user_id)
    test = s[s.split == "test"]
    tsnap = Snapshot(data, TEST_START, users)
    variants = {"test (all)": test, "test without bulk-day targets": test[~test.bulk_day],
                "test Aug-Oct 2020": test[test.time < "2020-11-01"],
                "test Nov 2020-Jan 2021": test[(test.time >= "2020-11-01") & (test.time < "2021-02-01")],
                "test Feb-Apr 2021": test[test.time >= "2021-02-01"]}
    rows = multi_evaluate(tsnap, variants, scorers_from(OUT / "logistic_pipeline.joblib", OUT / "lightgbm_model.txt"))
    del tsnap
    val = s[s.split == "val"]
    vsnap = Snapshot(data, VAL_START, users)
    vvariants = {"validation pre-COVID (Dec 2019-Feb 2020), train-only models": val[val.time < "2020-03-01"],
                 "validation COVID (Mar-Jul 2020), train-only models": val[val.time >= "2020-03-01"]}
    rows += multi_evaluate(vsnap, vvariants, scorers_from(CLEAN / "logistic_trainonly.joblib",
                                                          CLEAN / "lightgbm_trainonly.txt"))
    rob = pd.DataFrame(rows)
    rob.to_csv(OUT / "robustness.csv", index=False)
    print(rob.round(4).to_string(index=False))

    fmt = lambda r: f"{r['mean']:.4f} [{r.lo:.4f}, {r.hi:.4f}]"
    md = ["# Ranker evaluation", "", "Generated by `src/14_evaluate.py` from step 13 outputs.", "",
          "## Test (2020-08-01 to 2021-04-30), 95% user bootstrap intervals", "",
          "| Model | Recall@10 | NDCG@10 |", "|---|---|---|"]
    for m in MODELS:
        r = ci[(ci.model == m) & (ci.metric == "recall")].iloc[0]
        n = ci[(ci.model == m) & (ci.metric == "ndcg")].iloc[0]
        md.append(f"| {m} | {fmt(r)} | {fmt(n)} |")
    md += ["", f"Users: {int(ci.users.iloc[0]):,}.", "", "## Paired comparisons", "",
           "| A vs B | Metric | Mean difference [95% CI] | Relative | Wilcoxon p | Users A better / B better |",
           "|---|---|---|---|---|---|"]
    for _, r in pt.iterrows():
        md.append(f"| {r.model_a} vs {r.model_b} | {r.metric} | {r.mean_diff:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] | "
                  f"{r.relative:+.1%} | {r.wilcoxon_p:.2g} | {r.users_a_better:,} / {r.users_b_better:,} |")
    md += ["", "## Robustness (NDCG@10)", "", "| Variant | Users | " + " | ".join(MODELS) + " |",
           "|---|---|" + "---|" * len(MODELS)]
    for v, g in rob.groupby("variant", sort=False):
        md.append(f"| {v} | {int(g.users.iloc[0]):,} | " + " | ".join(f"{x:.4f}" for x in g.set_index('model').loc[MODELS, 'NDCG@10']) + " |")
    (OUT / "EVALUATION.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
