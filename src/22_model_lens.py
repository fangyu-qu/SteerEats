#!/usr/bin/env python3
"""Step 22: how differently do logistic regression and LightGBM rank the same candidates? (view V6)

For every eligible test user at the 2020-08-01 snapshot (the 8,025 users of the two-stage evaluation),
both final two-stage models score the same 200 stage-1 candidates. Per user:
  - overlap@10: restaurants the two top-10 lists share (0-10);
  - Kendall tau-b between the two scores over all candidates (how alike the whole pool is ordered);
  - NDCG@10 of each model, so disagreement can be related to which model found the later-liked place.
Ties break by candidate index, as in src/steering.py and the browser.

Outputs:
  outputs/model_lens/MODEL_LENS.md       summary
  outputs/model_lens/lens_per_user.csv   per-user values (git-ignored)
  web/public/data/model_lens.json        histograms for the interface, plus the SHAP group shares of
                                         step 17 (300 test users), so the browser can place a case user
                                         in the population.
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
from common import OUTPUTS, ROOT, TEST_START
from ranker_core import Data, Snapshot, candidates, eligible, metrics
import steering

MODELS = OUTPUTS / "rankers_two_stage"
OUT = OUTPUTS / "model_lens"
WEB = ROOT / "web" / "public" / "data" / "model_lens.json"
EXPECTED = ROOT / "tests" / "lens_expected.json"
TAU_BINS = np.round(np.linspace(-0.2, 1.0, 25), 4)   # 0.05 wide


def group_shares(c, model):
    """Share of mean |SHAP| per feature group over the case's candidates (as in step 17, summed by feature)."""
    imp = {f: float(np.mean(np.abs(c[model]["shap"][f]))) for f in c["features"]}
    tot = sum(imp.values())
    return {g: sum(imp[f] for f in fs) / tot for g, fs in c["feature_groups"].items()}


def case_expectations():
    out = {}
    for path in sorted((ROOT / "web" / "public" / "data" / "cases").glob("demo_*.json")):
        c = json.loads(path.read_text())
        r_lr, r_gb = (steering.final_ranks(c, steering.THETA0, m) for m in ("lr", "lgbm"))
        s_lr, s_gb = np.array(c["lr"]["raw"]), np.array(c["lgbm"]["raw"])
        out[c["id"]] = dict(overlap=int(((r_lr <= 10) & (r_gb <= 10)).sum()),
                            tau=float(kendalltau(s_lr, s_gb).statistic),
                            shares={m: group_shares(c, m) for m in ("lr", "lgbm")})
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

    rows = []
    for k, (u, rel) in enumerate(cases):
        f, seen, cf = snap.user(u)
        ids = candidates(snap, f, seen, cf)
        x = f[ids].astype(np.float64)
        s_lr, s_gb = x @ w + b_lr, gb.predict(x, raw_score=True)
        o_lr, o_gb = steering.order(s_lr), steering.order(s_gb)
        top_lr, top_gb = ids[o_lr[:10]], ids[o_gb[:10]]
        tau = kendalltau(s_lr, s_gb).statistic
        _, nd_lr = metrics(top_lr.tolist(), rel, 10)
        _, nd_gb = metrics(top_gb.tolist(), rel, 10)
        rows.append(dict(user_id=u, overlap=len(set(top_lr) & set(top_gb)), tau=tau,
                         ndcg_lr=nd_lr, ndcg_lgbm=nd_gb, in_candidates=len(rel & set(ids.tolist())) > 0))
        if k % 2000 == 0:
            print(f"  {k}/{len(cases)} users", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "lens_per_user.csv", index=False)

    n = len(df)
    ov = df.overlap.value_counts().reindex(range(11), fill_value=0)
    tau_hist, _ = np.histogram(df.tau.clip(TAU_BINS[0], TAU_BINS[-1]), bins=TAU_BINS)
    hit_lr, hit_gb = df.ndcg_lr > 0, df.ndcg_lgbm > 0
    by_overlap = []
    for lo, hi, label in ((0, 3, "0-3 shared"), (4, 6, "4-6 shared"), (7, 10, "7-10 shared")):
        g = df[(df.overlap >= lo) & (df.overlap <= hi)]
        by_overlap.append(dict(label=label, users=len(g), ndcg_lr=float(g.ndcg_lr.mean()), ndcg_lgbm=float(g.ndcg_lgbm.mean()),
                               only_lr=int(((g.ndcg_lr > 0) & (g.ndcg_lgbm == 0)).sum()),
                               only_lgbm=int(((g.ndcg_lgbm > 0) & (g.ndcg_lr == 0)).sum())))
    groups = pd.read_csv(OUTPUTS / "shap" / "group_importance.csv")
    shap_summary = json.loads((OUTPUTS / "shap" / "summary.json").read_text())
    lens = {
        "cutoff": TEST_START, "users": n,
        "overlap": {"counts": ov.tolist(), "mean": float(df.overlap.mean()), "median": float(df.overlap.median())},
        "tau": {"edges": TAU_BINS.tolist(), "counts": tau_hist.tolist(), "median": float(df.tau.median()),
                "q25": float(df.tau.quantile(0.25)), "q75": float(df.tau.quantile(0.75)), "min": float(df.tau.min())},
        "hits": {"both": int((hit_lr & hit_gb).sum()), "only_lr": int((hit_lr & ~hit_gb).sum()),
                 "only_lgbm": int((~hit_lr & hit_gb).sum()), "neither": int((~hit_lr & ~hit_gb).sum())},
        "by_overlap": by_overlap,
        "shap_groups": {"users": shap_summary["users"],
                        "rows": {r.group: {"lr": float(r.lr_share), "lgbm": float(r.lgbm_share)} for r in groups.itertuples()}},
    }
    WEB.write_text(json.dumps(lens, indent=1))

    pct = lambda v: f"{100 * v / n:.1f}%"
    md = [
        "# Model lens: logistic regression vs LightGBM (two-stage, test)",
        "",
        f"Generated by `src/22_model_lens.py`. {n:,} eligible test users at the {TEST_START} snapshot; both final "
        "two-stage models re-rank the same 200 stage-1 candidates per user. Ties break by candidate index.",
        "",
        "## Do the two top-10 lists agree?",
        "",
        f"- Shared restaurants in the top 10: mean {df.overlap.mean():.2f}, median {df.overlap.median():.0f}.",
        "",
        "| Shared | Users | Share |", "|---|---|---|",
        *[f"| {s} | {c:,} | {pct(c)} |" for s, c in ov.items()],
        "",
        f"- Kendall tau-b over all candidates: median {df.tau.median():.3f} (IQR {df.tau.quantile(0.25):.3f} to "
        f"{df.tau.quantile(0.75):.3f}, min {df.tau.min():.3f}).",
        "",
        "## Does disagreement matter for accuracy?",
        "",
        f"- At least one later-liked place in the top 10: both models {pct(lens['hits']['both'])}, only LR "
        f"{pct(lens['hits']['only_lr'])}, only LightGBM {pct(lens['hits']['only_lgbm'])}, neither {pct(lens['hits']['neither'])}.",
        "",
        "| Lists share | Users | NDCG@10 LR | NDCG@10 LightGBM | Hit by LR only | Hit by LightGBM only |",
        "|---|---|---|---|---|---|",
        *[f"| {b['label']} | {b['users']:,} | {b['ndcg_lr']:.4f} | {b['ndcg_lgbm']:.4f} | {b['only_lr']:,} | {b['only_lgbm']:,} |"
          for b in by_overlap],
        "",
        f"Eligibility audit: {audit}. Runtime {time.time() - t0:.0f} s.",
    ]
    (OUT / "MODEL_LENS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    case_expectations()


if __name__ == "__main__":
    main()
