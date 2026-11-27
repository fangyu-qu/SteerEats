#!/usr/bin/env python3
"""Step 22: how differently do the rankers rank the same candidates? (view V6)

For every eligible test user at the 2020-08-01 snapshot (the 8,025 users of the two-stage evaluation),
the three final two-stage models (logistic regression, LightGBM, LambdaRank) score the same 200 stage-1
candidates. Per user and pair of models:
  - overlap@10: restaurants the two top-10 lists share (0-10);
  - Kendall tau-b between the two scores over all candidates (how alike the whole pool is ordered);
  - NDCG@10 of each model, so disagreement can be related to which model found the later-liked place.
Ties break by candidate index, as in src/steering.py and the browser. LambdaRank joined on 2026-11-26; the
numbers of the logistic regression and LightGBM pair are unchanged by it.

Outputs:
  outputs/model_lens/MODEL_LENS.md       summary
  outputs/model_lens/lens_per_user.csv   per-user values (git-ignored)
  web/public/data/model_lens.json        per pair, histograms for the interface; plus the SHAP group shares
                                         of steps 17 and 39 (300 test users), so the browser can place a
                                         case user in the population.
  tests/lens_expected.json               per demo case, from the exported (rounded) file: overlap@10,
                                         Kendall tau-b and SHAP group shares, which the browser must match
                                         (tests/check_lens.mjs). `--cases-only` writes just this file.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS, RANKERS, ROOT, TEST_START
from ranker_core import Data, Snapshot, candidates, eligible, metrics
import steering

MODELS = OUTPUTS / "rankers_two_stage"
OUT = OUTPUTS / "model_lens"
WEB = ROOT / "web" / "public" / "data" / "model_lens.json"
EXPECTED = ROOT / "tests" / "lens_expected.json"
TAU_BINS = np.round(np.linspace(-0.2, 1.0, 25), 4)   # 0.05 wide
PAIRS = [("lr", "lgbm"), ("lr", "lmr"), ("lgbm", "lmr")]
SHARE_COLUMN = {"lr": "lr_share", "lgbm": "lgbm_share", "lmr": "lambdarank_share"}


def group_shares(c, model):
    """Share of mean |SHAP| per feature group over the case's candidates (as in step 17, summed by feature)."""
    imp = {f: float(np.mean(np.abs(c[model]["shap"][f]))) for f in c["features"]}
    tot = sum(imp.values())
    return {g: sum(imp[f] for f in fs) / tot for g, fs in c["feature_groups"].items()}


def case_expectations():
    out = {}
    for path in sorted((ROOT / "web" / "public" / "data" / "cases").glob("demo_*.json")):
        c = json.loads(path.read_text())
        ranks = {m: steering.final_ranks(c, steering.THETA0, m) for m in RANKERS}
        raw = {m: np.array(c[m]["raw"]) for m in RANKERS}
        out[c["id"]] = dict(pairs={f"{a}|{b}": dict(overlap=int(((ranks[a] <= 10) & (ranks[b] <= 10)).sum()),
                                                    tau=float(kendalltau(raw[a], raw[b]).statistic)) for a, b in PAIRS},
                            shares={m: group_shares(c, m) for m in RANKERS})
    EXPECTED.write_text(json.dumps(out, indent=1))
    print(f"wrote {EXPECTED.relative_to(ROOT)} for {len(out)} cases")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-only", action="store_true")
    if ap.parse_args().cases_only:
        return case_expectations()
    import joblib
    import lightgbm as lgb
    t0 = time.time()
    data = Data()
    test = data.interactions[data.interactions.split == "test"]
    snap = Snapshot(data, TEST_START, set(test.user_id))
    cases, audit = eligible(snap, test)
    lr = joblib.load(MODELS / "logistic_pipeline.joblib")
    w = lr[-1].coef_[0] / lr[0].scale_
    b_lr = lr[-1].intercept_[0] - lr[0].mean_ @ w
    gb = lgb.Booster(model_file=str(MODELS / "lightgbm_model.txt"))
    rk = lgb.Booster(model_file=str(MODELS / "lambdarank_model.txt"))

    rows = []
    for k, (u, rel) in enumerate(cases):
        f, seen, cf = snap.user(u)
        ids = candidates(snap, f, seen, cf)
        x = f[ids].astype(np.float64)
        sc = {"lr": x @ w + b_lr, "lgbm": gb.predict(x, raw_score=True), "lmr": rk.predict(x, raw_score=True)}
        top = {m: ids[steering.order(v)[:10]] for m, v in sc.items()}
        row = dict(user_id=u, in_candidates=len(rel & set(ids.tolist())) > 0)
        for m in RANKERS:
            row[f"ndcg_{m}"] = metrics(top[m].tolist(), rel, 10)[1]
        for a, b in PAIRS:
            row[f"overlap_{a}_{b}"] = len(set(top[a]) & set(top[b]))
            row[f"tau_{a}_{b}"] = kendalltau(sc[a], sc[b]).statistic
        rows.append(row)
        if k % 2000 == 0:
            print(f"  {k}/{len(cases)} users", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "lens_per_user.csv", index=False)

    n = len(df)
    pairs = {}
    for a, b in PAIRS:
        ov_col, tau_col = df[f"overlap_{a}_{b}"], df[f"tau_{a}_{b}"]
        nd_a, nd_b = df[f"ndcg_{a}"], df[f"ndcg_{b}"]
        ov = ov_col.value_counts().reindex(range(11), fill_value=0)
        tau_hist, _ = np.histogram(tau_col.clip(TAU_BINS[0], TAU_BINS[-1]), bins=TAU_BINS)
        hit_a, hit_b = nd_a > 0, nd_b > 0
        by_overlap = []
        for lo, hi, label in ((0, 3, "0-3 shared"), (4, 6, "4-6 shared"), (7, 10, "7-10 shared")):
            g = (ov_col >= lo) & (ov_col <= hi)
            by_overlap.append(dict(label=label, users=int(g.sum()),
                                   ndcg_a=float(nd_a[g].mean()) if g.any() else None, ndcg_b=float(nd_b[g].mean()) if g.any() else None,
                                   only_a=int((g & hit_a & ~hit_b).sum()), only_b=int((g & hit_b & ~hit_a).sum())))
        pairs[f"{a}|{b}"] = {
            "a": a, "b": b,
            "overlap": {"counts": ov.tolist(), "mean": float(ov_col.mean()), "median": float(ov_col.median())},
            "tau": {"edges": TAU_BINS.tolist(), "counts": tau_hist.tolist(), "median": float(tau_col.median()),
                    "q25": float(tau_col.quantile(0.25)), "q75": float(tau_col.quantile(0.75)), "min": float(tau_col.min())},
            "hits": {"both": int((hit_a & hit_b).sum()), "only_a": int((hit_a & ~hit_b).sum()),
                     "only_b": int((~hit_a & hit_b).sum()), "neither": int((~hit_a & ~hit_b).sum())},
            "by_overlap": by_overlap}
    groups = pd.read_csv(OUTPUTS / "shap" / "lambdarank_group_importance.csv")
    shap_summary = json.loads((OUTPUTS / "shap" / "summary.json").read_text())
    lens = {
        "cutoff": TEST_START, "users": n, "models": dict(RANKERS), "ndcg": {m: float(df[f"ndcg_{m}"].mean()) for m in RANKERS},
        "pairs": pairs,
        "shap_groups": {"users": shap_summary["users"],
                        "rows": {r.group: {m: float(getattr(r, col)) for m, col in SHARE_COLUMN.items()} for r in groups.itertuples()}},
    }
    WEB.write_text(json.dumps(lens, indent=1))

    pct = lambda v: f"{100 * v / n:.1f}%"
    md = ["# Model lens: how alike do the three rankers rank? (two-stage, test)", "",
          f"Generated by `src/22_model_lens.py`. {n:,} eligible test users at the {TEST_START} snapshot; the final "
          "two-stage models re-rank the same 200 stage-1 candidates per user. Ties break by candidate index.", ""]
    for a, b in PAIRS:
        P, A, Bn = pairs[f"{a}|{b}"], RANKERS[a], RANKERS[b]
        md += [f"## {A} and {Bn}", "",
               f"- Shared restaurants in the top 10: mean {P['overlap']['mean']:.2f}, median {P['overlap']['median']:.0f}.", "",
               "| Shared | Users | Share |", "|---|---|---|",
               *[f"| {s} | {c:,} | {pct(c)} |" for s, c in enumerate(P["overlap"]["counts"])], "",
               f"- Kendall tau-b over all candidates: median {P['tau']['median']:.3f} (IQR {P['tau']['q25']:.3f} to "
               f"{P['tau']['q75']:.3f}, min {P['tau']['min']:.3f}).",
               f"- At least one later-liked place in the top 10: both models {pct(P['hits']['both'])}, only {A} "
               f"{pct(P['hits']['only_a'])}, only {Bn} {pct(P['hits']['only_b'])}, neither {pct(P['hits']['neither'])}.", "",
               f"| Lists share | Users | NDCG@10 {A} | NDCG@10 {Bn} | Hit by {A} only | Hit by {Bn} only |", "|---|---|---|---|---|---|",
               *[f"| {x['label']} | {x['users']:,} | {x['ndcg_a']:.4f} | {x['ndcg_b']:.4f} | {x['only_a']:,} | {x['only_b']:,} |"
                 for x in P["by_overlap"] if x["users"]], ""]
    md += [f"Eligibility audit: {audit}. Runtime {time.time() - t0:.0f} s."]
    (OUT / "MODEL_LENS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    case_expectations()


if __name__ == "__main__":
    main()
