#!/usr/bin/env python3
"""Step 23: calibration of the two-stage rankers (view V7).

The models are trained on one visit to ten sampled candidates, so their raw probabilities describe that
mix, not the real one (a few visits among 200 candidates). Platt scaling fits P(visit) = sigmoid(a*s + b)
on the raw score s. It is monotone per user, so Recall@10 and NDCG@10 do not change; it matters only where
probabilities are compared across users or models.

Rows: every (eligible test user, stage-1 candidate) pair at the 2020-08-01 snapshot (8,025 users, about
200 candidates each); label 1 if the user reviewed the restaurant in the test period (the evaluation's
relevance definition).

Main result, the deployed models: the final models were refit on train + validation, so validation is not
held out for them. Platt is cross-fitted on test instead: users split into 5 folds, each fold calibrated
by a fit on the other four.
Check, as the plan specified: train-only models, Platt fitted on validation candidates (2019-12-01
snapshot), evaluated on test.
Reference: popularity (log all-time review count) through the same Platt fit, a "calibrated but weak"
score. Constant base rate: perfectly calibrated (ECE 0) and useless (AUC 0.5).

Scores: Brier, Brier skill (1 - Brier / Brier of the base rate), log-loss, ECE with 15 equal-mass bins
(and 15 equal-width bins, to show how binning moves ECE), pooled ROC AUC. Reliability bins are
log-spaced in predicted probability, with counts and Wilson 95% intervals.
Outputs: outputs/calibration/CALIBRATION.md, calibration.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, TEST_START, VAL_START
from ranker_core import FEATURES, Data, Snapshot, candidates, eligible

SEED = 42
FOLDS = 5
MODELS = OUTPUTS / "rankers_two_stage"
OUT = OUTPUTS / "calibration"
EDGES = np.round(10 ** np.linspace(-5, 0, 21), 10)   # 4 bins per decade
POP = FEATURES.index("log_popularity")


def rows(snap, target, scorers):
    """Scores and labels for every candidate of every eligible user."""
    cases, _ = eligible(snap, target)
    users, ys, xs = [], [], []
    for u, rel in cases:
        f, seen, cf = snap.user(u)
        ids = candidates(snap, f, seen, cf)
        xs.append(f[ids].astype(np.float64))
        ys.append(np.isin(ids, list(rel)).astype(np.int8))
        users.append(np.full(len(ids), len(users)))
    x = np.vstack(xs)
    out = {name: fn(x) for name, fn in scorers.items()}
    out["Popularity"] = x[:, POP]
    return out, np.concatenate(ys), np.concatenate(users), len(cases)


def platt(s, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


sigmoid = lambda z: 1 / (1 + np.exp(-z))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def ece(p, y, bins, mass=True):
    if mass:
        idx = np.array_split(np.argsort(p, kind="stable"), bins)
    else:
        b = np.minimum((p * bins).astype(int), bins - 1)
        idx = [np.flatnonzero(b == k) for k in range(bins)]
    return float(sum(len(i) * abs(p[i].mean() - y[i].mean()) for i in idx if len(i)) / len(p))


def scores(p, y):
    from sklearn.metrics import roc_auc_score
    eps = 1e-12
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    return dict(brier=brier, brier_skill=1 - brier / float(base * (1 - base)),
                log_loss=float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))),
                ece_mass15=ece(p, y, 15, True), ece_width15=ece(p, y, 15, False),
                auc=float(roc_auc_score(y, p)), mean_p=float(p.mean()), base_rate=float(base))


def reliability(p, y):
    out = []
    b = np.clip(np.searchsorted(EDGES, p, side="right") - 1, 0, len(EDGES) - 2)
    for k in range(len(EDGES) - 1):
        m = b == k
        n = int(m.sum())
        if not n:
            continue
        pos = int(y[m].sum())
        lo, hi = wilson(pos, n)
        out.append(dict(lo_edge=float(EDGES[k]), hi_edge=float(EDGES[k + 1]), n=n, positives=pos,
                        mean_p=float(p[m].mean()), observed=pos / n, ci=[lo, hi]))
    return out


def main():
    import joblib
    import lightgbm as lgb
    from models import logistic_scorer
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    data = Data()
    s = data.interactions

    def load(lr_path, gb_path):
        gb = lgb.Booster(model_file=str(gb_path))
        return {"Logistic regression": logistic_scorer(joblib.load(lr_path)),
                "LightGBM": lambda f, gb=gb: gb.predict(f, raw_score=True)}

    test = s[s.split == "test"]
    tsnap = Snapshot(data, TEST_START, set(test.user_id))
    final, y, user, n_users = rows(tsnap, test, load(MODELS / "logistic_pipeline.joblib", MODELS / "lightgbm_model.txt"))
    trainonly, y2, _, _ = rows(tsnap, test, load(CLEAN / "logistic_trainonly_two_stage.joblib", CLEAN / "lightgbm_trainonly_two_stage.txt"))
    assert np.array_equal(y, y2)
    del tsnap
    print(f"test rows {len(y):,}, positives {int(y.sum()):,}, users {n_users:,} ({time.time() - t0:.0f}s)", flush=True)

    # Deployed models: Platt cross-fitted over user folds
    fold = rng.permutation(n_users)[user] % FOLDS
    main_res, coefs = {}, {}
    for name, sc in final.items():
        p = np.empty(len(y))
        coefs[name] = []
        for k in range(FOLDS):
            a, b = platt(sc[fold != k], y[fold != k])
            p[fold == k] = sigmoid(a * sc[fold == k] + b)
            coefs[name].append([a, b])
        res = dict(after=scores(p, y), reliability=reliability(p, y), platt_folds=coefs[name])
        if name != "Popularity":
            raw = sigmoid(sc)
            res.update(before=scores(raw, y), reliability_before=reliability(raw, y))
        main_res[name] = res

    # Check: train-only models, Platt fitted on validation
    val = s[s.split == "val"]
    vsnap = Snapshot(data, VAL_START, set(val.user_id))
    vscores, vy, _, v_users = rows(vsnap, val, load(CLEAN / "logistic_trainonly_two_stage.joblib", CLEAN / "lightgbm_trainonly_two_stage.txt"))
    del vsnap
    check = {}
    for name in trainonly:
        a, b = platt(vscores[name], vy)
        p = sigmoid(a * trainonly[name] + b)
        check[name] = dict(platt=[a, b], after=scores(p, y))
    base = y.mean()
    const = scores(np.full(len(y), base), y)
    result = dict(cutoff=TEST_START, users=n_users, rows=int(len(y)), positives=int(y.sum()), folds=FOLDS,
                  edges=EDGES.tolist(), deployed=main_res, base_rate_predictor=const,
                  validation_check=dict(users=v_users, rows=int(len(vy)), positives=int(vy.sum()), models=check))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(result, indent=1))

    f4 = lambda v: f"{v:.4f}"
    md = ["# Calibration of the two-stage rankers", "",
          f"Generated by `src/23_calibration.py`. {len(y):,} (user, candidate) rows from {n_users:,} test users at the "
          f"{TEST_START} snapshot; {int(y.sum()):,} positives (base rate {base:.5f}). ECE uses 15 equal-mass bins "
          "unless marked width. AUC is pooled over all rows.", "",
          f"## Deployed models, Platt cross-fitted over {FOLDS} user folds", "",
          "| Score | Brier | Brier skill | Log-loss | ECE | ECE (width) | AUC | Mean predicted |", "|---|---|---|---|---|---|---|---|"]
    for name, r in main_res.items():
        for when in ("before", "after"):
            if when not in r:
                continue
            m = r[when]
            label = f"{name}, {'raw probability' if when == 'before' else 'after Platt'}"
            md.append(f"| {label} | {m['brier']:.6f} | {m['brier_skill']:.4f} | {f4(m['log_loss'])} | {m['ece_mass15']:.5f} | "
                      f"{m['ece_width15']:.5f} | {f4(m['auc'])} | {m['mean_p']:.5f} |")
    md.append(f"| Constant base rate | {const['brier']:.6f} | {const['brier_skill']:.4f} | {f4(const['log_loss'])} | "
              f"{const['ece_mass15']:.5f} | {const['ece_width15']:.5f} | {f4(const['auc'])} | {const['mean_p']:.5f} |")
    md += ["", "Platt coefficients (a, b) per fold:", ""]
    md += [f"- {name}: " + "; ".join(f"({a:.3f}, {b:.3f})" for a, b in r["platt_folds"]) for name, r in main_res.items()]
    md += ["", f"## Check: train-only models, Platt fitted on validation ({v_users:,} users, {len(vy):,} rows, "
           f"{int(vy.sum()):,} positives), evaluated on test", "",
           "| Score | Brier skill | ECE | AUC | Mean predicted | Platt (a, b) |", "|---|---|---|---|---|---|"]
    md += [f"| {name} | {r['after']['brier_skill']:.4f} | {r['after']['ece_mass15']:.5f} | {f4(r['after']['auc'])} | "
           f"{r['after']['mean_p']:.5f} | ({r['platt'][0]:.3f}, {r['platt'][1]:.3f}) |" for name, r in check.items()]
    md += ["", f"Runtime {time.time() - t0:.0f} s."]
    (OUT / "CALIBRATION.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
