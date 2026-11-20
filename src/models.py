"""Model fitting shared by the full-catalog (step 13) and two-stage (step 15) protocols."""
import numpy as np

from ranker_core import FEATURES

SEED = 42


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


def fit_lambdarank(x, y, groups, leaves, rounds=None, valid=None):
    """LightGBM with a listwise objective (supplementary: not part of the pointwise comparison)."""
    import lightgbm as lgb

    def sizes(g):
        _, idx, counts = np.unique(g, return_index=True, return_counts=True)
        return counts[np.argsort(idx)]
    params = lgbm_params(leaves) | dict(objective="lambdarank", metric="ndcg", eval_at=[10],
                                         lambdarank_truncation_level=20)
    train = lgb.Dataset(x, y, group=sizes(groups), feature_name=FEATURES, free_raw_data=False)
    if valid is None:
        return lgb.train(params, train, num_boost_round=rounds)
    v = lgb.Dataset(valid[0], valid[1], group=sizes(valid[2]), reference=train)
    return lgb.train(params, train, num_boost_round=3000, valid_sets=[v],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
