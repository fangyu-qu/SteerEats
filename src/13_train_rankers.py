#!/usr/bin/env python3
"""Step 13: train logistic regression and LightGBM on identical time-safe features, labels,
negatives and candidates, and evaluate them with the three baselines.

Protocol (same for both models)
  - Training examples come from disjoint target windows inside TRAIN: for each window
    [cut_k, cut_k+1) a Snapshot built strictly before cut_k supplies features, positives are the
    sample's 4-5 star first reviews in the window, and each positive gets 4 sampled negatives
    (half uniform, half popularity-matched) from restaurants available at cut_k.
  - Model choice uses VALIDATION (snapshot at 2019-12-01): logistic C in {0.1, 1, 10};
    LightGBM num_leaves in {15, 31, 63} with early stopping on validation examples. The ranking
    metric (NDCG@10, then Recall@10) on a fixed random subset of validation users decides.
  - Final models refit on train windows + the validation window, then rank every available,
    unseen restaurant for each TEST user (snapshot at 2020-08-01; test ends 2021-04-30).
Outputs: outputs/rankers/ (metrics, coefficients, importance, models, metadata, summary).
Per-user metrics stay local (git-ignored) and feed src/14_evaluate.py.
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
from common import INTERACTIONS_CLEAN, OUTPUTS, CLEAN, TEST_START, VAL_START
from ranker_core import FEATURES, FEATURE_GROUPS, Data, Snapshot, evaluate, examples

SEED = 42
C_GRID = [0.1, 1.0, 10.0]
LEAVES_GRID = [15, 31, 63]
VAL_SELECTION_USERS = 3000


def fit_logistic(x, y, c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, solver="lbfgs"))
    m.fit(x, y)
    if m[-1].n_iter_.max() >= 2000:
        raise ValueError("Logistic solver hit the iteration limit")
    return m


def logistic_scorer(m):
    scaler, lin = m[0], m[-1]
    w, b = lin.coef_[0], lin.intercept_[0]
    return lambda f: ((f - scaler.mean_) / scaler.scale_) @ w + b


def lgbm_params(leaves):
    return dict(objective="binary", learning_rate=0.05, num_leaves=leaves, min_child_samples=100,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                seed=SEED, deterministic=True, force_row_wise=True, verbose=-1, num_threads=0)


def fit_lgbm(x, y, leaves, rounds=None, valid=None):
    import lightgbm as lgb
    train = lgb.Dataset(x, y, feature_name=FEATURES, free_raw_data=False)
    if valid is None:
        return lgb.train(lgbm_params(leaves), train, num_boost_round=rounds)
    v = lgb.Dataset(valid[0], valid[1], reference=train)
    return lgb.train(lgbm_params(leaves), train, num_boost_round=3000, valid_sets=[v],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def summarize(recs, period, audit):
    df = pd.DataFrame(recs)
    out = df.groupby("model")[["recall", "ndcg"]].mean().rename(columns={"recall": "Recall@10", "ndcg": "NDCG@10"})
    out.insert(0, "users", df.groupby("model").user_id.nunique())
    out.insert(0, "period", period)
    return out.reset_index(), df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUTPUTS / "rankers")
    ap.add_argument("--smoke", type=int, default=0, help="use only this many sample users (quick check)")
    ap.add_argument("--negative-ratio", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    args.out.mkdir(parents=True, exist_ok=True)

    data = Data()
    s = data.interactions
    users = np.sort(s.user_id.unique())
    if args.smoke:
        users = np.sort(rng.choice(users, size=args.smoke, replace=False))
        s = s[s.user_id.isin(users)]
    users = set(users)
    train, val, test = (s[s.split == k] for k in ("train", "val", "test"))
    print(f"loaded in {time.time() - t0:.0f}s: {len(s):,} interactions, {len(users):,} users", flush=True)

    cuts = sorted({train.time.quantile(q).to_period("M").to_timestamp() for q in (0.4, 0.6, 0.8)}
                  | {pd.Timestamp(VAL_START)})
    xs, ys, windows = [], [], []
    for start, end in zip(cuts[:-1], cuts[1:]):
        snap = Snapshot(data, start, users)
        x, y, _ = examples(snap, train[(train.time >= start) & (train.time < end)], rng, args.negative_ratio)
        xs.append(x); ys.append(y)
        windows.append(dict(cutoff=str(start.date()), end=str(end.date()), rows=int(len(y)),
                            positives=int(y.sum()), history_reviews=snap.history_rows))
        print(f"window {start.date()} -> {end.date()}: {len(y):,} rows ({time.time() - t0:.0f}s)", flush=True)
    x_tr, y_tr = np.vstack(xs), np.concatenate(ys)

    vsnap = Snapshot(data, VAL_START, users)
    x_va, y_va, _ = examples(vsnap, val, rng, args.negative_ratio)
    print(f"validation examples: {len(y_va):,} ({time.time() - t0:.0f}s)", flush=True)

    # Model selection on a fixed subset of validation users
    sel_users = set(rng.choice(np.sort(val.user_id.unique()), size=min(VAL_SELECTION_USERS, val.user_id.nunique()),
                               replace=False))
    val_sel = val[val.user_id.isin(sel_users)]
    logit = {c: fit_logistic(x_tr, y_tr, c) for c in C_GRID}
    lgbm = {k: fit_lgbm(x_tr, y_tr, k, valid=(x_va, y_va)) for k in LEAVES_GRID}
    scorers = {f"Logistic C={c}": logistic_scorer(m) for c, m in logit.items()}
    scorers |= {f"LightGBM leaves={k}": (lambda b: lambda f: b.predict(f, raw_score=True))(b) for k, b in lgbm.items()}
    print(f"fitted candidates ({time.time() - t0:.0f}s); evaluating on {len(sel_users):,} validation users", flush=True)
    recs, val_audit = evaluate(vsnap, val_sel, scorers)
    val_metrics, _ = summarize(recs, "validation (selection subset)", val_audit)
    pick = lambda prefix: val_metrics[val_metrics.model.str.startswith(prefix)].sort_values(
        ["NDCG@10", "Recall@10"], ascending=False).iloc[0].model
    best_c = float(pick("Logistic").split("=")[1])
    best_leaves = int(pick("LightGBM").split("=")[1])
    best_rounds = int(lgbm[best_leaves].best_iteration)
    print(val_metrics.to_string(index=False), flush=True)

    # Final fit on train windows + validation window
    x_all, y_all = np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va])
    final_lr = fit_logistic(x_all, y_all, best_c)
    final_gb = fit_lgbm(x_all, y_all, best_leaves, rounds=best_rounds)
    tsnap = Snapshot(data, TEST_START, users)
    print(f"final models fitted; evaluating test ({time.time() - t0:.0f}s)", flush=True)
    recs, test_audit = evaluate(tsnap, test, {"Logistic regression": logistic_scorer(final_lr),
                                              "LightGBM": lambda f: final_gb.predict(f, raw_score=True)})
    test_metrics, per_user = summarize(recs, "test", test_audit)
    print(test_metrics.to_string(index=False), flush=True)

    # Outputs
    tag = "smoke_" if args.smoke else ""
    pd.concat([val_metrics, test_metrics]).to_csv(args.out / f"{tag}metrics.csv", index=False)
    per_user.to_csv(args.out / f"{tag}test_per_user.csv", index=False)
    if args.smoke:
        print(f"smoke run done in {time.time() - t0:.0f}s")
        return
    import joblib
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    pd.DataFrame({"feature": FEATURES, "group": [group_of[f] for f in FEATURES],
                  "standardized_coefficient": final_lr[-1].coef_[0]}).to_csv(args.out / "coefficients.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "group": [group_of[f] for f in FEATURES],
                  "gain": final_gb.feature_importance("gain"), "splits": final_gb.feature_importance("split")}
                 ).sort_values("gain", ascending=False).to_csv(args.out / "lightgbm_importance.csv", index=False)
    pd.DataFrame(np.corrcoef(x_all.T), index=FEATURES, columns=FEATURES).round(3).to_csv(
        args.out / "feature_correlations.csv")
    joblib.dump(final_lr, args.out / "logistic_pipeline.joblib")
    final_gb.save_model(str(args.out / "lightgbm_model.txt"))
    # Train-window-only models (selected configs): used for evaluation windows inside validation,
    # e.g. the pre-COVID check, where the final models would have seen the labels.
    joblib.dump(logit[best_c], CLEAN / "logistic_trainonly.joblib")
    lgbm[best_leaves].save_model(str(CLEAN / "lightgbm_trainonly.txt"), num_iteration=best_rounds)
    bg = rng.choice(len(y_all), size=min(5000, len(y_all)), replace=False)
    np.savez_compressed(CLEAN / "shap_background.npz", x=x_all[bg], y=y_all[bg], features=np.array(FEATURES))

    import lightgbm, sklearn
    meta = dict(seed=SEED, negative_ratio=args.negative_ratio, negatives="half uniform, half popularity-matched",
                features=FEATURES, feature_groups=FEATURE_GROUPS, training_windows=windows,
                validation_examples=int(len(y_va)), selection_users=len(sel_users),
                selected=dict(logistic_C=best_c, lightgbm_num_leaves=best_leaves, lightgbm_rounds=best_rounds),
                lightgbm_params=lgbm_params(best_leaves), validation_audit=val_audit, test_audit=test_audit,
                test_start=TEST_START, validation_start=VAL_START,
                input_sha256=hashlib.sha256(INTERACTIONS_CLEAN.read_bytes()).hexdigest(),
                versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                              lightgbm=lightgbm.__version__),
                runtime_seconds=round(time.time() - t0))
    (args.out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
