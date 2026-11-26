#!/usr/bin/env python3
"""Step 33: how do group strategies change who is satisfied and who travels? (RQ2)

Synthetic three-member groups of test users, three kinds (fixed seed):
  similar   an anchor user and the two users whose cuisine mix is closest to theirs
  random    three users drawn at random
  diverse   an anchor, the user least like them, and the user least like both
Cuisine mix = share of each cuisine group among the restaurants a user reviewed before the snapshot.

For every group, model and strategy (src/group.py):
  pool           union of the members' stage-1 candidates at the test snapshot
  satisfaction   each member's raw model score, min-max over the pool (no steering: default settings)
  list           top 10 by the strategy's group score
Reported per list: the worst-off member's satisfaction and the satisfaction gap (mean over the ten
places), the longest trip and the travel gap in km from each member's usual area, each member's NDCG@10
on the group list against their own held-out visits (mean and worst member), and for "average without
misery" how many pool items pass the threshold. Sensitivity: thresholds 0.30 and 0.50, and satisfaction
as rank percentile instead of min-max. Intervals are bootstrap over groups; strategy differences are
paired within groups.

Outputs: outputs/rq2/RQ2.md, rq2_summary.csv, rq2_differences.csv; web/public/data/rq2.json
Run: .venv/bin/python src/33_rq2_groups.py            # --groups 1000 per kind
"""
import argparse
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
from group import STRATEGIES, STRATEGY_LABEL, TAU, group_score, rank_pool, satisfaction
from models import logistic_scorer
from ranker_core import Data, Snapshot, candidates, eligible, haversine_km, metrics

OUT = OUTPUTS / "rq2"
MODELS_DIR = OUTPUTS / "rankers_two_stage"
KINDS = ("similar", "random", "diverse")
METRICS = ["min_sat", "sat_gap", "max_km", "km_gap", "ndcg_mean", "ndcg_worst"]
METRIC_LABEL = {"min_sat": "Worst-off member's satisfaction", "sat_gap": "Satisfaction gap (best minus worst member)",
                "max_km": "Longest trip (km)", "km_gap": "Travel gap (km)",
                "ndcg_mean": "Member NDCG@10 on the group list, mean", "ndcg_worst": "Member NDCG@10, worst member"}


def make_groups(shares, n, rng):
    """Index triples into the eligible users for each kind."""
    unit = shares / np.maximum(np.linalg.norm(shares, axis=1, keepdims=True), 1e-12)
    anchors = rng.choice(len(shares), size=n, replace=False)
    out = {"similar": [], "diverse": [], "random": []}
    for a in anchors:
        sim = unit @ unit[a]
        sim[a] = np.inf
        order = np.argsort(-np.where(np.isinf(sim), -np.inf, sim), kind="stable")
        out["similar"].append((a, order[0], order[1]))
        far = np.where(np.isinf(sim), np.inf, sim)
        b = int(np.argmin(far))
        both = far + unit @ unit[b]
        both[b] = np.inf
        out["diverse"].append((a, b, int(np.argmin(both))))
    for _ in range(n):
        out["random"].append(tuple(rng.choice(len(shares), size=3, replace=False)))
    return out, unit


def boot(values, rng, n=2000):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=int, default=1000, help="groups per kind")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    data = Data()
    s = data.interactions
    snap = Snapshot(data, TEST_START, set(s.user_id.unique()))
    cases, _ = eligible(snap, s[s.split == "test"])
    users = [u for u, _ in cases]
    rel = dict(cases)
    shares = np.array([snap.profile[u]["cuisine"] / max(snap.profile[u]["n"], 1) for u in users], float)
    groups, unit = make_groups(shares, args.groups, rng)
    print(f"{len(users):,} eligible test users; {args.groups} groups per kind ({time.time() - t0:.0f}s)", flush=True)

    lr = joblib.load(MODELS_DIR / "logistic_pipeline.joblib")
    gb = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_model.txt"))
    rk = lgb.Booster(model_file=str(MODELS_DIR / "lambdarank_model.txt"))
    scorers = {"Logistic regression": logistic_scorer(lr), "LightGBM": lambda f: gb.predict(f, raw_score=True),
               "LambdaRank": lambda f: rk.predict(f, raw_score=True)}

    cache = {}

    def member(j):
        if j not in cache:
            u = users[j]
            f, seen, cf = snap.user(u)
            p = snap.profile[u]
            cache[j] = (f, candidates(snap, f, seen, cf), p["lat"], p["lon"])
            if len(cache) > 400:
                cache.pop(next(iter(cache)))
        return cache[j]

    variants = [(st, TAU, "minmax") for st in STRATEGIES]
    variants += [("average_without_misery", 0.30, "minmax"), ("average_without_misery", 0.50, "minmax")]
    variants += [(st, TAU, "percentile") for st in STRATEGIES]
    rows = []
    for kind in KINDS:
        for gi, trio in enumerate(groups[kind]):
            mem = [member(int(j)) for j in trio]
            pool = np.unique(np.concatenate([m[1] for m in mem]))
            feats = np.vstack([m[0][pool] for m in mem])
            km = np.vstack([haversine_km(m[2], m[3], data.lat[pool], data.lon[pool]) for m in mem])
            cos = float(np.mean([unit[trio[a]] @ unit[trio[b]] for a, b in ((0, 1), (0, 2), (1, 2))]))
            rels = [rel[users[int(j)]] for j in trio]
            for model, fn in scorers.items():
                sc = fn(feats).reshape(3, len(pool))
                own = []
                for m in range(3):                         # each member's own top 10 from their own candidates
                    c = mem[m][1]
                    mine = sc[m][np.searchsorted(pool, c)]
                    own.append(metrics(c[np.lexsort((c, -mine))][:10].tolist(), rels[m])[1])
                sats = {how: satisfaction(sc, how) for how in ("minmax", "percentile")}
                for strategy, tau, how in variants:
                    sat = sats[how]
                    score, feasible = group_score(sat, strategy, tau)
                    top = rank_pool(score, feasible, pool)[:10]
                    row = dict(kind=kind, group=gi, model=model, strategy=strategy, tau=tau, how=how, pool=len(pool),
                               feasible=int(feasible.sum()), listed=len(top), cuisine_cosine=cos,
                               own_ndcg_mean=float(np.mean(own)), own_ndcg_worst=float(np.min(own)))
                    if len(top):
                        nd = [metrics(pool[top].tolist(), r)[1] for r in rels]
                        row |= dict(min_sat=float(sat[:, top].min(axis=0).mean()),
                                    sat_gap=float((sat[:, top].max(axis=0) - sat[:, top].min(axis=0)).mean()),
                                    max_km=float(km[:, top].max(axis=0).mean()),
                                    km_gap=float((km[:, top].max(axis=0) - km[:, top].min(axis=0)).mean()),
                                    ndcg_mean=float(np.mean(nd)), ndcg_worst=float(np.min(nd)))
                    else:
                        row |= dict(min_sat=np.nan, sat_gap=np.nan, max_km=np.nan, km_gap=np.nan, ndcg_mean=0.0, ndcg_worst=0.0)
                    rows.append(row)
            if gi % 250 == 0:
                print(f"  {kind}: {gi}/{args.groups} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    (ROOT / "data" / "clean" / "rq2").mkdir(parents=True, exist_ok=True)
    df.to_parquet(ROOT / "data" / "clean" / "rq2" / "groups.parquet", index=False)

    brng = np.random.default_rng(SEED)
    summ = []
    for (kind, model, strategy, tau, how), g in df.groupby(["kind", "model", "strategy", "tau", "how"], sort=False):
        rec = dict(kind=kind, model=model, strategy=strategy, tau=tau, how=how, groups=len(g),
                   feasible_mean=float(g.feasible.mean()), share_ten=float((g.feasible >= 10).mean()),
                   share_none=float((g.listed == 0).mean()), pool_mean=float(g.pool.mean()),
                   cuisine_cosine=float(g.cuisine_cosine.mean()))
        for m in METRICS + ["own_ndcg_mean", "own_ndcg_worst"]:
            rec[m], rec[m + "_lo"], rec[m + "_hi"] = boot(g[m], brng)
        summ.append(rec)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "rq2_summary.csv", index=False)

    main_df = df[(df.tau == TAU) & (df.how == "minmax")]
    diffs = []
    for (kind, model), g in main_df.groupby(["kind", "model"], sort=False):
        wide = {st: g[g.strategy == st].set_index("group") for st in STRATEGIES}
        for other in STRATEGIES[1:]:
            for m in METRICS:
                d = (wide[other][m] - wide["average"][m]).to_numpy()
                mean, lo, hi = boot(d, brng)
                diffs.append(dict(kind=kind, model=model, strategy=other, metric=m, diff=mean, lo=lo, hi=hi,
                                  groups=int(np.isfinite(d).sum())))
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(OUT / "rq2_differences.csv", index=False)

    def cell(r, m, digits=3):
        return f"{r[m]:.{digits}f} [{r[m + '_lo']:.{digits}f}, {r[m + '_hi']:.{digits}f}]"

    L = ["# RQ2: group strategies", "",
         f"{args.groups:,} three-member groups per kind, drawn from {len(users):,} eligible test users "
         f"(snapshot {TEST_START}, seed {SEED}). Pool = union of the members' stage-1 candidates. Satisfaction = raw "
         "model score, min-max over the pool, no steering. Lists are the top 10 by group score. "
         f"\"Average without misery\" drops places where any member is below {TAU:.2f}. "
         "Brackets are 95% bootstrap intervals over groups.", "",
         "| Kind | Mean cuisine cosine between members | Mean pool size |", "|---|---|---|"]
    first = summ[(summ.model == "LambdaRank") & (summ.strategy == "average") & (summ.how == "minmax")]
    L += [f"| {r.kind} | {r.cuisine_cosine:.3f} | {r.pool_mean:.0f} |" for r in first.itertuples()]
    for model in scorers:
        L += ["", f"## {model}", ""]
        for kind in KINDS:
            part = summ[(summ.model == model) & (summ.kind == kind) & (summ.tau == TAU) & (summ.how == "minmax")]
            L += [f"### {kind.capitalize()} groups", "",
                  "| Strategy | Worst-off satisfaction | Satisfaction gap | Longest trip (km) | Travel gap (km) | "
                  "Member NDCG@10, mean | Member NDCG@10, worst |", "|---|---|---|---|---|---|---|"]
            for _, r in part.iterrows():
                L.append(f"| {STRATEGY_LABEL[r.strategy]} | {cell(r, 'min_sat')} | {cell(r, 'sat_gap')} | {cell(r, 'max_km', 2)} | "
                         f"{cell(r, 'km_gap', 2)} | {cell(r, 'ndcg_mean', 4)} | {cell(r, 'ndcg_worst', 4)} |")
            r = part.iloc[0]
            L += ["", f"Each member's own list, for reference: NDCG@10 mean {cell(r, 'own_ndcg_mean', 4)}, "
                      f"worst member {cell(r, 'own_ndcg_worst', 4)}.", ""]
    L += ["## Paired differences from \"Average\" (same groups)", "",
          "| Model | Kind | Strategy | " + " | ".join(METRIC_LABEL[m] for m in METRICS) + " |", "|---|---|---|" + "---|" * len(METRICS)]
    for (model, kind, strategy), g in diffs.groupby(["model", "kind", "strategy"], sort=False):
        g = g.set_index("metric")
        L.append(f"| {model} | {kind} | {STRATEGY_LABEL[strategy]} | " + " | ".join(
            f"{g.loc[m, 'diff']:+.4f} [{g.loc[m, 'lo']:+.4f}, {g.loc[m, 'hi']:+.4f}]" for m in METRICS) + " |")
    L += ["", "## Feasibility of \"average without misery\"", "",
          "| Model | Kind | Threshold | Places passing (mean) | Groups with at least ten | Groups with none |", "|---|---|---|---|---|---|"]
    awm = summ[(summ.strategy == "average_without_misery") & (summ.how == "minmax")].sort_values(["model", "kind", "tau"])
    L += [f"| {r.model} | {r.kind} | {r.tau:.2f} | {r.feasible_mean:.1f} | {r.share_ten:.1%} | {r.share_none:.1%} |"
          for r in awm.itertuples()]
    L += ["", "## Sensitivity: satisfaction as rank percentile", "",
          "| Model | Kind | Strategy | Worst-off satisfaction | Member NDCG@10, mean | Member NDCG@10, worst | Groups with at least ten |",
          "|---|---|---|---|---|---|---|"]
    pc = summ[summ.how == "percentile"]
    L += [f"| {r.model} | {r.kind} | {STRATEGY_LABEL[r.strategy]} | {r.min_sat:.3f} | {r.ndcg_mean:.4f} | {r.ndcg_worst:.4f} | {r.share_ten:.1%} |"
          for r in pc.itertuples()]
    L += ["", "Written by `src/33_rq2_groups.py`."]
    (OUT / "RQ2.md").write_text("\n".join(L) + "\n")

    web = dict(groups_per_kind=args.groups, users=len(users), tau=TAU, kinds=list(KINDS), models=list(scorers),
               strategies=[dict(id=k, label=v) for k, v in STRATEGY_LABEL.items()],
               metrics=[dict(id=k, label=v) for k, v in METRIC_LABEL.items()],
               summary=json.loads(summ.round(5).to_json(orient="records")),
               differences=json.loads(diffs.round(5).to_json(orient="records")))
    (ROOT / "web" / "public" / "data" / "rq2.json").write_text(json.dumps(web, separators=(",", ":")))
    print("\n".join(L[:40]))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
