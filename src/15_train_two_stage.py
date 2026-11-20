#!/usr/bin/env python3
"""Step 15: two-stage protocol. Stage 1 retrieves about 200 candidates per user
(ranker_core.candidates); stage 2 re-ranks them with logistic regression or LightGBM.

Differences from step 13, applied identically to both models:
  - training negatives are sampled from the user's stage-1 candidates (10 per positive), so the
    models learn to separate a visit from the plausible alternatives, not from random restaurants;
  - evaluation re-ranks the candidates; baselines still rank the full catalog, and the share of
    relevant items reaching the candidates (stage-1 recall) is reported as an upper bound.
Same features, windows, cutoffs, labels and selection rule (NDCG@10 on 3,000 validation users).
Supplementary: LightGBM with a LambdaRank objective on the same candidate groups. It is listwise,
so it is reported separately from the pointwise comparison the proposal fixed.
Outputs: outputs/rankers_two_stage/
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, INTERACTIONS_CLEAN, OUTPUTS, TEST_START, VAL_START
from models import SEED, fit_lambdarank, fit_lgbm, fit_logistic, lgbm_params, logistic_scorer
from ranker_core import (CANDIDATE_QUOTAS, FEATURE_GROUPS, FEATURES, Data, Snapshot, evaluate,
                         examples_two_stage)

C_GRID = [0.1, 1.0, 10.0]
LEAVES_GRID = [15, 31, 63]
VAL_SELECTION_USERS = 3000


def summarize(recs, period):
    df = pd.DataFrame(recs)
    out = df.groupby("model")[["recall", "ndcg"]].mean().rename(columns={"recall": "Recall@10", "ndcg": "NDCG@10"})
    out.insert(0, "users", df.groupby("model").user_id.nunique())
    out.insert(0, "period", period)
    return out.reset_index(), df


def raw(b):
    return lambda f: b.predict(f, raw_score=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUTPUTS / "rankers_two_stage")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--negative-ratio", type=int, default=10)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="sampling seed (negatives, selection users); non-default seeds write metrics only")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    if args.seed != SEED:
        args.out = args.out / "seeds" / f"seed_{args.seed}"
        args.out.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    data = Data()
    s = data.interactions
    users = np.sort(s.user_id.unique())
    if args.smoke:
        users = np.sort(rng.choice(users, size=args.smoke, replace=False))
        s = s[s.user_id.isin(users)]
    users = set(users)
    train, val, test = (s[s.split == k] for k in ("train", "val", "test"))

    cuts = sorted({train.time.quantile(q).to_period("M").to_timestamp() for q in (0.4, 0.6, 0.8)}
                  | {pd.Timestamp(VAL_START)})
    xs, ys, gs, windows = [], [], [], []
    for start, end in zip(cuts[:-1], cuts[1:]):
        snap = Snapshot(data, start, users)
        x, y, g = examples_two_stage(snap, train[(train.time >= start) & (train.time < end)], rng, args.negative_ratio)
        xs.append(x); ys.append(y); gs.append(np.char.add(f"{start.date()}|", g.astype(str)))
        windows.append(dict(cutoff=str(start.date()), end=str(end.date()), rows=int(len(y)), positives=int(y.sum())))
        print(f"window {start.date()} -> {end.date()}: {len(y):,} rows ({time.time() - t0:.0f}s)", flush=True)
    x_tr, y_tr, g_tr = np.vstack(xs), np.concatenate(ys), np.concatenate(gs)
    vsnap = Snapshot(data, VAL_START, users)
    x_va, y_va, g_va = examples_two_stage(vsnap, val, rng, args.negative_ratio)
    print(f"validation examples: {len(y_va):,} ({time.time() - t0:.0f}s)", flush=True)

    sel_users = set(rng.choice(np.sort(val.user_id.unique()), size=min(VAL_SELECTION_USERS, val.user_id.nunique()),
                               replace=False))
    val_sel = val[val.user_id.isin(sel_users)]
    logit = {c: fit_logistic(x_tr, y_tr, c) for c in C_GRID}
    lgbm = {k: fit_lgbm(x_tr, y_tr, k, valid=(x_va, y_va)) for k in LEAVES_GRID}
    rank = {k: fit_lambdarank(x_tr, y_tr, g_tr, k, valid=(x_va, y_va, g_va)) for k in LEAVES_GRID}
    scorers = {f"Logistic C={c}": logistic_scorer(m) for c, m in logit.items()}
    scorers |= {f"LightGBM leaves={k}": raw(b) for k, b in lgbm.items()}
    scorers |= {f"LambdaRank leaves={k}": raw(b) for k, b in rank.items()}
    print(f"fitted candidates ({time.time() - t0:.0f}s)", flush=True)
    recs, val_audit = evaluate(vsnap, val_sel, scorers, two_stage=True)
    val_metrics, _ = summarize(recs, "validation (selection subset)")
    print(val_metrics.to_string(index=False), flush=True)

    def pick(prefix):
        return val_metrics[val_metrics.model.str.startswith(prefix)].sort_values(
            ["NDCG@10", "Recall@10"], ascending=False).iloc[0].model
    best_c = float(pick("Logistic").split("=")[1])
    best_leaves = int(pick("LightGBM").split("=")[1])
    best_rank = int(pick("LambdaRank").split("=")[1])
    rounds = int(lgbm[best_leaves].best_iteration)
    rank_rounds = int(rank[best_rank].best_iteration)

    x_all, y_all, g_all = np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va]), np.concatenate([g_tr, np.char.add("val|", g_va.astype(str))])
    final_lr = fit_logistic(x_all, y_all, best_c)
    final_gb = fit_lgbm(x_all, y_all, best_leaves, rounds=rounds)
    final_rk = fit_lambdarank(x_all, y_all, g_all, best_rank, rounds=rank_rounds)
    tsnap = Snapshot(data, TEST_START, users)
    print(f"final models fitted; evaluating test ({time.time() - t0:.0f}s)", flush=True)
    recs, test_audit = evaluate(tsnap, test, {"Logistic regression": logistic_scorer(final_lr),
                                              "LightGBM": raw(final_gb),
                                              "LightGBM LambdaRank (supplementary)": raw(final_rk)}, two_stage=True)
    test_metrics, per_user = summarize(recs, "test")
    print(test_metrics.to_string(index=False), flush=True)

    tag = "smoke_" if args.smoke else ""
    pd.concat([val_metrics, test_metrics]).to_csv(args.out / f"{tag}metrics.csv", index=False)
    per_user.to_csv(args.out / f"{tag}test_per_user.csv", index=False)
    if args.smoke or args.seed != SEED:
        print(f"metrics-only run done in {time.time() - t0:.0f}s")
        return
    import joblib
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    pd.DataFrame({"feature": FEATURES, "group": [group_of[f] for f in FEATURES],
                  "standardized_coefficient": final_lr[-1].coef_[0]}).to_csv(args.out / "coefficients.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "group": [group_of[f] for f in FEATURES],
                  "gain": final_gb.feature_importance("gain"), "splits": final_gb.feature_importance("split")}
                 ).sort_values("gain", ascending=False).to_csv(args.out / "lightgbm_importance.csv", index=False)
    joblib.dump(final_lr, args.out / "logistic_pipeline.joblib")
    final_gb.save_model(str(args.out / "lightgbm_model.txt"))
    final_rk.save_model(str(args.out / "lambdarank_model.txt"))
    joblib.dump(logit[best_c], CLEAN / "logistic_trainonly_two_stage.joblib")
    lgbm[best_leaves].save_model(str(CLEAN / "lightgbm_trainonly_two_stage.txt"), num_iteration=rounds)
    bg = rng.choice(len(y_all), size=min(5000, len(y_all)), replace=False)
    np.savez_compressed(CLEAN / "shap_background_two_stage.npz", x=x_all[bg], y=y_all[bg], features=np.array(FEATURES))

    import lightgbm, sklearn
    meta = dict(seed=SEED, protocol="two-stage", candidate_quotas=CANDIDATE_QUOTAS, negative_ratio=args.negative_ratio,
                negatives="uniform from the user's stage-1 candidates", features=FEATURES, training_windows=windows,
                validation_examples=int(len(y_va)), selection_users=len(sel_users),
                selected=dict(logistic_C=best_c, lightgbm_num_leaves=best_leaves, lightgbm_rounds=rounds,
                              lambdarank_num_leaves=best_rank, lambdarank_rounds=rank_rounds),
                lightgbm_params=lgbm_params(best_leaves), validation_audit=val_audit, test_audit=test_audit,
                input_sha256=hashlib.sha256(INTERACTIONS_CLEAN.read_bytes()).hexdigest(),
                versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                              lightgbm=lightgbm.__version__),
                runtime_seconds=round(time.time() - t0))
    (args.out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
