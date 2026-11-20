"""Steering and change accounting over one user's stage-1 candidates (reference implementation).

web/src/steer.js implements the same arithmetic in the same order; tests/replay.mjs checks that both
produce identical rankings for every exported case.

Controls theta = (w_h, alpha, beta, lam), each in [0, 1]:
  w_h    history weight: scales the Collaborative group's SHAP contribution to the model score.
         For logistic regression this is exactly scaling the CF coefficients toward the background;
         for LightGBM it removes the CF group's interventional SHAP credit. 1 = the model as trained.
  alpha  distance penalty on the normalized score
  beta   popularity penalty on the normalized score
  lam    MMR diversity weight over cuisine group and neighbourhood (0 = no re-ranking)
theta0 = (1, 0, 0, 0) reproduces the model's own ranking.

Fixed stage order (update, "Explanation accounting"):
  S0 model     z = base_noncollab + w_h * collab           (log-odds), then min-max to [0, 1]
  S1 distance  S0n - alpha * distance_n                     distance_n = min(km, 15) / 15
  S2 popular   S1 - beta * popularity_n                     popularity_n = percentile among candidates
  S3 diverse   MMR on min-max(S2) with weight lam
Ranks are positions (1 = best) in the full candidate ordering at each stage; ties break by candidate
index (the stage-1 order), so the ordering is deterministic. Rank change between stages telescopes,
so the fixed-order decomposition has zero residual by construction; the Shapley decomposition over
the four controls (16 subsets) is order-independent and also sums exactly to the total change.
"""
from itertools import combinations
from math import factorial

import numpy as np

CONTROLS = ("w_h", "alpha", "beta", "lam")
THETA0 = {"w_h": 1.0, "alpha": 0.0, "beta": 0.0, "lam": 0.0}
DIST_CAP_KM = 15.0
SIM_CUISINE, SIM_NTA = 0.5, 0.5


def minmax(v):
    lo, hi = float(v.min()), float(v.max())
    return np.zeros_like(v) if hi - lo <= 0 else (v - lo) / (hi - lo)


def order(scores):
    """Indices from best to worst; ties keep candidate order."""
    return np.lexsort((np.arange(len(scores)), -scores))


def ranks_from_order(o):
    r = np.empty(len(o), dtype=np.int64)
    r[o] = np.arange(1, len(o) + 1)
    return r


def similarity(case):
    c, n = np.asarray(case["cuisine"]), np.asarray(case["nta"])
    return SIM_CUISINE * (c[:, None] == c[None, :]) + SIM_NTA * (n[:, None] == n[None, :])


def mmr_order(rel, sim, lam):
    """Greedy MMR over all candidates: argmax (1 - lam) * rel_i - lam * max_{j chosen} sim_ij."""
    n = len(rel)
    if lam == 0:
        return order(rel)
    chosen, left = [], list(range(n))
    max_sim = np.zeros(n)
    for _ in range(n):
        best, best_val = -1, -np.inf
        for i in left:                                   # left stays in candidate order -> ties break by index
            v = (1 - lam) * rel[i] - lam * max_sim[i]
            if v > best_val:
                best, best_val = i, v
        chosen.append(best)
        left.remove(best)
        max_sim = np.maximum(max_sim, sim[best])
    return np.array(chosen)


def stages(case, theta, model, sim=None):
    """Scores and ranks at each stage for one model ("lr" or "lgbm")."""
    sim = similarity(case) if sim is None else sim
    z = np.asarray(case[model]["noncollab"]) + theta["w_h"] * np.asarray(case[model]["collab"])
    s0 = minmax(z)
    s1 = s0 - theta["alpha"] * np.asarray(case["distance_n"])
    s2 = s1 - theta["beta"] * np.asarray(case["popularity_n"])
    o3 = mmr_order(minmax(s2), sim, theta["lam"])
    return {"S0": ranks_from_order(order(s0)), "S1": ranks_from_order(order(s1)),
            "S2": ranks_from_order(order(s2)), "S3": ranks_from_order(o3), "score": {"S0": s0, "S1": s1, "S2": s2}}


def final_ranks(case, theta, model, sim=None):
    return stages(case, theta, model, sim)["S3"]


def stage_log(case, theta, model, sim=None):
    """Fixed-order decomposition of each candidate's rank change from theta0's final ranking to theta's.

    Steps: model change (w_h moves S0), then distance, popularity, diversity. Returns per-candidate
    arrays that sum exactly to final_rank(theta) - final_rank(theta0).
    """
    sim = similarity(case) if sim is None else sim
    before = final_ranks(case, THETA0, model, sim)
    st = stages(case, theta, model, sim)
    steps = {"model": st["S0"] - before, "distance": st["S1"] - st["S0"],
             "popularity": st["S2"] - st["S1"], "diversity": st["S3"] - st["S2"]}
    total = st["S3"] - before
    residual = total - sum(steps.values())
    return dict(before=before, after=st["S3"], steps=steps, total=total, residual=residual)


def shapley_controls(case, theta, model, sim=None):
    """Exact Shapley attribution of each candidate's final-rank change to the four controls."""
    sim = similarity(case) if sim is None else sim
    n = len(CONTROLS)
    cache = {}

    def rank_with(subset):
        key = tuple(sorted(subset))
        if key not in cache:
            t = dict(THETA0)
            for c in key:
                t[c] = theta[c]
            cache[key] = final_ranks(case, t, model, sim).astype(float)
        return cache[key]

    phi = {c: np.zeros(len(case["cuisine"])) for c in CONTROLS}
    for c in CONTROLS:
        others = [o for o in CONTROLS if o != c]
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for s in combinations(others, k):
                phi[c] += w * (rank_with(s + (c,)) - rank_with(s))
    total = rank_with(CONTROLS) - rank_with(())
    residual = total - sum(phi.values())
    return dict(phi=phi, total=total, residual=residual)
