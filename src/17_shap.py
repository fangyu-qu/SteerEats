#!/usr/bin/env python3
"""Step 17: SHAP explanations for the two-stage logistic regression and LightGBM rankers.

What is explained: each model's RAW score (log-odds) for the stage-1 candidates of a random set
of test users at the 2020-08-01 snapshot. SHAP explains scores, not final ranks; rank changes
from steering are decomposed separately.

Value function (week 5): interventional SHAP for BOTH models with the SAME background, so the
attributions answer the same question ("contribution relative to a typical candidate").
  - background: 200 candidate rows drawn at random from the explained set;
  - logistic regression: closed form phi_j = w_j (x_j - E_bg[x_j]) on the raw-feature scale
    (w_j = coef_j / scale_j), checked against shap.LinearExplainer;
  - LightGBM: shap.TreeExplainer(interventional) with an explicit Independent masker (shap
    otherwise silently cuts the background to 100 rows); also path-dependent TreeSHAP
    (LightGBM pred_contrib), whose value function differs, to show how much that choice matters.
Checks: additivity (base + sum phi = raw score) for every row and model. shap's interventional
TreeSHAP on this LightGBM model leaves a small residual (about 1.4e-4 log-odds on almost every row,
up to about 0.1 on roughly 0.1% of rows), so shap's hard stop is turned off, the residual
distribution is reported, and rows above 1e-3 are flagged and excluded from agreement statistics.
Comparisons: global importance (mean |phi|, share of total), per-feature sign agreement and
rank correlation between the models, per-candidate Spearman correlation and top-3 overlap.

Outputs: outputs/shap/ (aggregates, checks, SHAP.md); data/clean/shap/ (row-level values with
user ids, local only, used later by the demo export).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, TEST_START
from ranker_core import FEATURE_GROUPS, FEATURES, Data, Snapshot, candidates, eligible

SEED = 42
N_USERS = 300
N_BACKGROUND = 200
MODELS = OUTPUTS / "rankers_two_stage"
OUT = OUTPUTS / "shap"
LOCAL = CLEAN / "shap"


def explained_rows(data, rng):
    test = data.interactions[data.interactions.split == "test"]
    pool = np.sort(test[test.label == 1].user_id.unique())
    users = set(rng.choice(pool, size=min(3 * N_USERS, len(pool)), replace=False))
    snap = Snapshot(data, TEST_START, users)
    cases = dict(eligible(snap, test[test.user_id.isin(users)])[0])
    chosen = rng.choice(np.array(sorted(cases)), size=min(N_USERS, len(cases)), replace=False)
    xs, meta = [], []
    for u in chosen:
        f, seen, cf = snap.user(u)
        ids = candidates(snap, f, seen, cf)
        xs.append(f[ids])
        meta.append(pd.DataFrame({"user_id": u, "gmap_id": data.items[ids], "relevant": np.isin(ids, list(cases[u]))}))
    return np.vstack(xs).astype(np.float64), pd.concat(meta, ignore_index=True)


def spearman_rows(a, b):
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(float)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def main():
    import joblib
    import lightgbm as lgb
    import shap
    from scipy.stats import spearmanr
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True); LOCAL.mkdir(parents=True, exist_ok=True)
    data = Data()
    X, meta = explained_rows(data, rng)
    bg = X[rng.choice(len(X), size=N_BACKGROUND, replace=False)]
    print(f"explaining {len(X):,} candidate rows of {meta.user_id.nunique()} users ({time.time() - t0:.0f}s)", flush=True)

    # Logistic regression: closed form on the raw-feature scale
    lr = joblib.load(MODELS / "logistic_pipeline.joblib")
    scaler, lin = lr[0], lr[-1]
    w = lin.coef_[0] / scaler.scale_
    raw_lr = X @ w + (lin.intercept_[0] - scaler.mean_ @ w)
    mu = bg.mean(axis=0)
    phi_lr = (X - mu) * w
    base_lr = mu @ w + (lin.intercept_[0] - scaler.mean_ @ w)
    expl = shap.LinearExplainer(lin, shap.maskers.Independent(scaler.transform(bg), max_samples=N_BACKGROUND))
    phi_lr_shap = expl(scaler.transform(X)).values
    linear_check = dict(max_abs_diff_closed_form_vs_shap=float(np.abs(phi_lr - phi_lr_shap).max()),
                        max_additivity_error=float(np.abs(base_lr + phi_lr.sum(1) - raw_lr).max()),
                        base_value=float(base_lr))

    # LightGBM: interventional (same background) and path-dependent
    gb = lgb.Booster(model_file=str(MODELS / "lightgbm_model.txt"))
    raw_gb = gb.predict(X, raw_score=True)
    tree = shap.TreeExplainer(gb, data=shap.maskers.Independent(bg, max_samples=N_BACKGROUND),
                              feature_perturbation="interventional", model_output="raw")
    phi_gb = tree.shap_values(X, check_additivity=False)
    base_gb = float(np.atleast_1d(tree.expected_value)[0])
    resid = np.abs(base_gb + phi_gb.sum(1) - raw_gb)
    ok = resid <= 1e-3
    contrib = gb.predict(X, pred_contrib=True)
    phi_gb_path, base_gb_path = contrib[:, :-1], contrib[:, -1]
    tree_check = dict(max_additivity_error_interventional=float(resid.max()),
                      additivity_error_interventional_quantiles={q: float(np.quantile(resid, q)) for q in (0.5, 0.9, 0.99)},
                      rows_flagged_over_1e_3=int((~ok).sum()),
                      max_additivity_error_path_dependent=float(np.abs(base_gb_path + phi_gb_path.sum(1) - raw_gb).max()),
                      base_value_interventional=base_gb, base_value_path_dependent=float(base_gb_path[0]))
    print(f"SHAP done ({time.time() - t0:.0f}s)", flush=True)

    # Global importance
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    imp = pd.DataFrame({"feature": FEATURES, "group": [group_of[f] for f in FEATURES],
                        "lr_mean_abs": np.abs(phi_lr).mean(0), "lgbm_mean_abs": np.abs(phi_gb).mean(0),
                        "lgbm_path_mean_abs": np.abs(phi_gb_path).mean(0)})
    for c in ("lr", "lgbm", "lgbm_path"):
        imp[f"{c}_share"] = imp[f"{c}_mean_abs"] / imp[f"{c}_mean_abs"].sum()
        imp[f"{c}_rank"] = imp[f"{c}_mean_abs"].rank(ascending=False).astype(int)
    imp = imp.sort_values("lgbm_mean_abs", ascending=False)
    imp.to_csv(OUT / "global_importance.csv", index=False)
    groups = imp.groupby("group")[["lr_share", "lgbm_share", "lgbm_path_share"]].sum().sort_values("lgbm_share", ascending=False)
    groups.to_csv(OUT / "group_importance.csv")

    # Agreement between models (both interventional, same background)
    phi_lr_all, phi_gb_all, phi_gb_path_all = phi_lr, phi_gb, phi_gb_path
    phi_lr, phi_gb, phi_gb_path = phi_lr[ok], phi_gb[ok], phi_gb_path[ok]
    active = (np.abs(phi_lr) > 1e-9) | (np.abs(phi_gb) > 1e-9)
    by_feat = pd.DataFrame({
        "feature": FEATURES,
        "sign_agreement": [(np.sign(phi_lr[active[:, j], j]) == np.sign(phi_gb[active[:, j], j])).mean()
                           for j in range(len(FEATURES))],
        "value_correlation": [np.corrcoef(phi_lr[:, j], phi_gb[:, j])[0, 1] if phi_gb[:, j].std() > 0 else np.nan
                              for j in range(len(FEATURES))],
        "lgbm_interventional_vs_path_correlation": [np.corrcoef(phi_gb[:, j], phi_gb_path[:, j])[0, 1] if phi_gb[:, j].std() > 0 else np.nan
                                        for j in range(len(FEATURES))]})
    by_feat.to_csv(OUT / "agreement_by_feature.csv", index=False)
    rho_rows = spearman_rows(np.abs(phi_lr), np.abs(phi_gb))
    top3 = lambda p: np.argsort(-np.abs(p), axis=1)[:, :3]
    overlap = np.array([len(set(a) & set(b)) for a, b in zip(top3(phi_lr), top3(phi_gb))])
    rho_vf = spearman_rows(np.abs(phi_gb), np.abs(phi_gb_path))
    summary = dict(
        rows=len(X), rows_in_agreement=int(ok.sum()), users=int(meta.user_id.nunique()), background=N_BACKGROUND,
        global_rank_spearman_lr_vs_lgbm=float(spearmanr(imp.lr_mean_abs, imp.lgbm_mean_abs).statistic),
        global_rank_spearman_interventional_vs_path=float(spearmanr(imp.lgbm_mean_abs, imp.lgbm_path_mean_abs).statistic),
        per_candidate_spearman_median=float(np.nanmedian(rho_rows)),
        per_candidate_spearman_iqr=[float(np.nanquantile(rho_rows, 0.25)), float(np.nanquantile(rho_rows, 0.75))],
        top3_overlap_distribution={str(k): float((overlap == k).mean()) for k in range(4)},
        value_function_per_candidate_spearman_median=float(np.nanmedian(rho_vf)),
        raw_score_correlation_lr_vs_lgbm=float(np.corrcoef(raw_lr, raw_gb)[0, 1]),
        linear_check=linear_check, tree_check=tree_check, runtime_seconds=round(time.time() - t0))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Feature correlations that make attribution ambiguous (week 5: correlated features)
    corr = pd.DataFrame(np.corrcoef(X.T), index=FEATURES, columns=FEATURES)
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:]
             if abs(corr.loc[a, b]) >= 0.7]
    pd.DataFrame(pairs, columns=["feature_a", "feature_b", "correlation"]).sort_values(
        "correlation", key=abs, ascending=False).to_csv(OUT / "correlated_pairs.csv", index=False)

    # Row-level values stay local (user ids)
    cols = {f"lr__{f}": phi_lr_all[:, j] for j, f in enumerate(FEATURES)}
    cols |= {f"lgbm__{f}": phi_gb_all[:, j] for j, f in enumerate(FEATURES)}
    cols |= {f"lgbm_path__{f}": phi_gb_path_all[:, j] for j, f in enumerate(FEATURES)}
    cols |= {f"x__{f}": X[:, j] for j, f in enumerate(FEATURES)}
    rows = pd.concat([meta, pd.DataFrame(cols), pd.DataFrame({"raw_lr": raw_lr, "raw_lgbm": raw_gb, "lgbm_shap_residual": resid})], axis=1)
    rows.to_parquet(LOCAL / "candidate_shap.parquet", index=False)
    np.savez_compressed(LOCAL / "background.npz", x=bg, features=np.array(FEATURES))

    # Summary page
    md = ["# SHAP explanations: logistic regression vs LightGBM (two-stage)", "",
          "Generated by `src/17_shap.py`. Raw scores (log-odds) of stage-1 candidates for "
          f"{summary['users']} random test users ({summary['rows']:,} rows) at the 2020-08-01 snapshot. "
          f"Interventional SHAP for both models, same {N_BACKGROUND}-row background.", "",
          "## Checks", "",
          f"- Logistic closed form vs `shap.LinearExplainer`: max difference {linear_check['max_abs_diff_closed_form_vs_shap']:.1e}.",
          f"- Additivity (base + sum = raw score), max error: logistic {linear_check['max_additivity_error']:.1e}, "
          f"LightGBM interventional median {tree_check['additivity_error_interventional_quantiles'][0.5]:.1e}, "
          f"99th percentile {tree_check['additivity_error_interventional_quantiles'][0.99]:.1e}, "
          f"max {tree_check['max_additivity_error_interventional']:.1e} "
          f"({tree_check['rows_flagged_over_1e_3']} rows above 1e-3, excluded from agreement statistics), "
          f"path-dependent {tree_check['max_additivity_error_path_dependent']:.1e}.", "",
          "## Global importance (share of mean |SHAP|)", "",
          "| Feature | Group | LR share (rank) | LightGBM share (rank) |", "|---|---|---|---|"]
    for _, r in imp.head(15).iterrows():
        md.append(f"| {r.feature} | {r.group} | {r.lr_share:.1%} ({r.lr_rank}) | {r.lgbm_share:.1%} ({r.lgbm_rank}) |")
    md += ["", "| Group | LR | LightGBM | LightGBM path-dependent |", "|---|---|---|---|"]
    for g, r in groups.iterrows():
        md.append(f"| {g} | {r.lr_share:.1%} | {r.lgbm_share:.1%} | {r.lgbm_path_share:.1%} |")
    o = summary["top3_overlap_distribution"]
    md += ["", "## Do the models explain candidates the same way?", "",
           f"- Raw-score correlation: {summary['raw_score_correlation_lr_vs_lgbm']:.3f}.",
           f"- Global importance rank correlation (Spearman): {summary['global_rank_spearman_lr_vs_lgbm']:.3f}.",
           f"- Per candidate, Spearman of |SHAP| across features: median {summary['per_candidate_spearman_median']:.3f} "
           f"(IQR {summary['per_candidate_spearman_iqr'][0]:.3f} to {summary['per_candidate_spearman_iqr'][1]:.3f}).",
           f"- Top-3 features shared per candidate: 3 of 3 {o['3']:.1%}, 2 {o['2']:.1%}, 1 {o['1']:.1%}, 0 {o['0']:.1%}.",
           f"- Value function (LightGBM interventional vs path-dependent): global rank correlation "
           f"{summary['global_rank_spearman_interventional_vs_path']:.3f}, per-candidate median "
           f"{summary['value_function_per_candidate_spearman_median']:.3f}.", "",
           "Per-feature sign agreement and correlations: `agreement_by_feature.csv`. Feature pairs with "
           "|r| >= 0.7, whose credit SHAP can split arbitrarily: `correlated_pairs.csv`."]
    (OUT / "SHAP.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
