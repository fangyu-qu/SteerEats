#!/usr/bin/env python3
"""Step 14: uncertainty and robustness for the rankers of step 13 (--protocol full) or
step 15 (--protocol two_stage).

1. Test metrics with 95% user-level bootstrap intervals (2,000 resamples) and paired
   comparisons (bootstrap interval of the mean difference, Wilcoxon signed-rank test).
2. Robustness, re-ranking with the saved models (scores are computed once per user and
   scored against several target definitions):
   - test without bulk-upload targets (posting time may differ from visit time);
   - test by period: Aug-Oct 2020, Nov 2020-Jan 2021, Feb-Apr 2021 (same 2020-08-01 snapshot,
     so later periods are also a longer horizon);
   - validation window split at the COVID shutdown, using the train-window-only models:
     pre-COVID (Dec 2019 - Feb 2020) vs COVID (Mar - Jul 2020).
Outputs (in the protocol's folder): test_intervals.csv, paired_tests.csv, robustness.csv, EVALUATION.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, TEST_START, VAL_START
from ranker_core import Data, Snapshot, candidates, eligible, metrics

B = 2000
SEED = 42
BASELINES = ["Popularity", "Recent popularity", "Item-CF"]
PROTOCOLS = {
    "full": dict(out=OUTPUTS / "rankers", title="full catalog", two_stage=False,
                 models=["Logistic regression", "LightGBM"],
                 final=dict(lr="logistic_pipeline.joblib", gb="lightgbm_model.txt"),
                 trainonly=dict(lr="logistic_trainonly.joblib", gb="lightgbm_trainonly.txt")),
    "two_stage": dict(out=OUTPUTS / "rankers_two_stage", title="two-stage (200 candidates)", two_stage=True,
                      models=["Logistic regression", "LightGBM", "LightGBM LambdaRank (supplementary)"],
                      final=dict(lr="logistic_pipeline.joblib", gb="lightgbm_model.txt", rk="lambdarank_model.txt"),
                      trainonly=dict(lr="logistic_trainonly_two_stage.joblib", gb="lightgbm_trainonly_two_stage.txt")),
}
NAMES = dict(lr="Logistic regression", gb="LightGBM", rk="LightGBM LambdaRank (supplementary)")
PAIRS = [("Logistic regression", "Item-CF"), ("LightGBM", "Item-CF"), ("LightGBM", "Logistic regression")]


def intervals(df, models, rng):
    rows = []
    users = np.sort(df.user_id.unique())
    idx = rng.integers(0, len(users), size=(B, len(users)))
    for m in models:
        d = df[df.model == m].set_index("user_id").loc[users]
        for metric in ("recall", "ndcg"):
            v = d[metric].to_numpy()
            if np.isnan(v).all():
                continue
            boot = v[idx].mean(axis=1)
            rows.append(dict(model=m, metric=metric, mean=v.mean(), lo=np.quantile(boot, 0.025),
                             hi=np.quantile(boot, 0.975), users=len(v)))
    return pd.DataFrame(rows)


def paired(df, rng, pairs=PAIRS):
    from scipy.stats import wilcoxon
    rows = []
    wide = {m: df[df.model == m].set_index("user_id").sort_index() for m in set(sum(pairs, ()))}
    n = len(wide["Item-CF"])
    idx = rng.integers(0, n, size=(B, n))
    for a, b in pairs:
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


def multi_evaluate(snap, variants, scorers, two_stage, k=10, batch=64):
    """Score each user once, then compute metrics for every target variant they are eligible in."""
    cases = {name: dict(eligible(snap, t)[0]) for name, t in variants.items()}
    users = sorted(set().union(*[set(c) for c in cases.values()]))
    names = BASELINES + list(scorers) + (["Stage 1 candidates"] if two_stage else [])
    sums = {(v, m): np.zeros(2) for v in variants for m in names}
    for b0 in range(0, len(users), batch):
        chunk = users[b0:b0 + batch]
        feats, cands, fulls, cfs = [], [], [], []
        for u in chunk:
            f, seen, cf = snap.user(u)
            avail = snap.available.copy()
            avail[seen] = False
            full = np.flatnonzero(avail)
            ids = candidates(snap, f, seen, cf) if two_stage else full
            feats.append(f[ids]); cands.append(ids); fulls.append(full); cfs.append(cf[full])
        stacked = np.vstack(feats)
        bounds = np.cumsum([0] + [len(c) for c in cands])
        scores = {name: fn(stacked) for name, fn in scorers.items()}
        for j, u in enumerate(chunk):
            ids, full = cands[j], fulls[j]
            top = {"Popularity": full[np.lexsort((full, -snap.pop[full]))][:k],
                   "Recent popularity": full[np.lexsort((full, -snap.recent_pop[full]))][:k],
                   "Item-CF": full[np.lexsort((full, -snap.pop[full], -cfs[j]))][:k]}
            for name, sc in scores.items():
                top[name] = ids[np.lexsort((ids, -sc[bounds[j]:bounds[j + 1]]))][:k]
            for v, c in cases.items():
                if u in c:
                    for m in top:
                        sums[(v, m)] += metrics(top[m].tolist(), c[u], k)
                    if two_stage:
                        sums[(v, "Stage 1 candidates")] += [len(c[u] & set(ids.tolist())) / len(c[u]), np.nan]
        if (b0 // batch) % 30 == 0:
            print(f"  {min(b0 + batch, len(users))}/{len(users)} users", flush=True)
    return [dict(variant=v, model=m, users=len(cases[v]), **{"Recall@10": sums[(v, m)][0] / max(len(cases[v]), 1),
                                                              "NDCG@10": sums[(v, m)][1] / max(len(cases[v]), 1)})
            for v in variants for m in names]


def scorers_from(paths, folder):
    import joblib
    import lightgbm as lgb
    out = {}
    for key, name in paths.items():
        path = folder / name
        if key == "lr":
            m = joblib.load(path)
            sc, lin = m[0], m[-1]
            out[NAMES[key]] = (lambda sc, lin: lambda f: ((f - sc.mean_) / sc.scale_) @ lin.coef_[0] + lin.intercept_[0])(sc, lin)
        else:
            b = lgb.Booster(model_file=str(path))
            out[NAMES[key]] = (lambda b: lambda f: b.predict(f, raw_score=True))(b)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", choices=PROTOCOLS, default="full")
    cfg = PROTOCOLS[ap.parse_args().protocol]
    out, two = cfg["out"], cfg["two_stage"]
    models = BASELINES + cfg["models"] + (["Stage 1 candidates"] if two else [])
    rng = np.random.default_rng(SEED)
    per_user = pd.read_csv(out / "test_per_user.csv")
    ci = intervals(per_user, models, rng)
    extra = [("LightGBM LambdaRank (supplementary)", "Logistic regression")] if two else []
    pt = paired(per_user, rng, PAIRS + extra)
    ci.to_csv(out / "test_intervals.csv", index=False)
    pt.to_csv(out / "paired_tests.csv", index=False)
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
    rows = multi_evaluate(tsnap, variants, scorers_from(cfg["final"], out), two)
    del tsnap
    val = s[s.split == "val"]
    vsnap = Snapshot(data, VAL_START, users)
    vvariants = {"validation pre-COVID (Dec 2019-Feb 2020), train-only models": val[val.time < "2020-03-01"],
                 "validation COVID (Mar-Jul 2020), train-only models": val[val.time >= "2020-03-01"]}
    rows += multi_evaluate(vsnap, vvariants, scorers_from(cfg["trainonly"], CLEAN), two)
    rob = pd.DataFrame(rows)
    rob.to_csv(out / "robustness.csv", index=False)
    print(rob.round(4).to_string(index=False))

    fmt = lambda r: f"{r['mean']:.4f} [{r.lo:.4f}, {r.hi:.4f}]"
    md = ["# Ranker evaluation: " + cfg["title"], "", "Generated by `src/14_evaluate.py`.", "",
          "## Test (2020-08-01 to 2021-04-30), 95% user bootstrap intervals", "",
          "| Model | Recall@10 | NDCG@10 |", "|---|---|---|"]
    for m in models:
        r = ci[(ci.model == m) & (ci.metric == "recall")]
        n = ci[(ci.model == m) & (ci.metric == "ndcg")]
        md.append(f"| {m} | {fmt(r.iloc[0])} | {fmt(n.iloc[0]) if len(n) else '—'} |")
    if two:
        md += ["", "Stage 1 candidates: share of relevant restaurants that reach the 200 candidates "
               "(an upper bound on re-ranker recall, not a top-10 metric)."]
    md += ["", f"Users: {int(ci.users.iloc[0]):,}.", "", "## Paired comparisons", "",
           "| A vs B | Metric | Mean difference [95% CI] | Relative | Wilcoxon p | Users A better / B better |",
           "|---|---|---|---|---|---|"]
    for _, r in pt.iterrows():
        md.append(f"| {r.model_a} vs {r.model_b} | {r.metric} | {r.mean_diff:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] | "
                  f"{r.relative:+.1%} | {r.wilcoxon_p:.2g} | {r.users_a_better:,} / {r.users_b_better:,} |")
    cols = [m for m in models if m != "Stage 1 candidates"]
    md += ["", "## Robustness (NDCG@10; train-only models have no LambdaRank)", "",
           "| Variant | Users | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for v, g in rob.groupby("variant", sort=False):
        g = g.set_index("model")
        md.append(f"| {v} | {int(g.users.iloc[0]):,} | " + " | ".join(
            f"{g.loc[m, 'NDCG@10']:.4f}" if m in g.index else "—" for m in cols) + " |")
    (out / "EVALUATION.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
