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
THETA0 = {"w_h": 1.0, "alpha": 0.0, "beta": 0.0, "lam": 0.0, "boost": 0.0, "boost_nta": -1}
BOOST_GRID = [round(k * 0.02, 4) for k in range(101)]   # neighbourhood boost, normalized-score units
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


def boost_vector(case, theta):
    """Neighbourhood boost (map control, 2026-11-20): +boost for candidates in neighbourhood boost_nta."""
    b, n = theta.get("boost", 0.0), theta.get("boost_nta", -1)
    nta = np.asarray(case["nta"])
    return b * (nta == n) if b and n >= 0 else np.zeros(len(nta))


def stages(case, theta, model, sim=None):
    """Scores and ranks at each stage for one model ("lr" or "lgbm")."""
    sim = similarity(case) if sim is None else sim
    z = np.asarray(case[model]["noncollab"]) + theta["w_h"] * np.asarray(case[model]["collab"])
    s0 = minmax(z)
    s1 = s0 - theta["alpha"] * np.asarray(case["distance_n"])
    s2 = s1 - theta["beta"] * np.asarray(case["popularity_n"])
    s2b = s2 + boost_vector(case, theta)
    o3 = mmr_order(minmax(s2b), sim, theta["lam"])
    return {"S0": ranks_from_order(order(s0)), "S1": ranks_from_order(order(s1)),
            "S2": ranks_from_order(order(s2)), "S2b": ranks_from_order(order(s2b)), "S3": ranks_from_order(o3),
            "score": {"S0": s0, "S1": s1, "S2": s2, "S2b": s2b}}


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
             "popularity": st["S2"] - st["S1"], "boost": st["S2b"] - st["S2"], "diversity": st["S3"] - st["S2b"]}
    total = st["S3"] - before
    residual = total - sum(steps.values())
    return dict(before=before, after=st["S3"], steps=steps, total=total, residual=residual)


def shapley_controls(case, theta, model, sim=None):
    """Exact Shapley attribution of each candidate's final-rank change to the controls
    (the four sliders, plus the neighbourhood boost when it is on)."""
    sim = similarity(case) if sim is None else sim
    controls = CONTROLS + (("boost",) if theta.get("boost", 0) else ())
    n = len(controls)
    cache = {}

    def rank_with(subset):
        key = tuple(sorted(subset))
        if key not in cache:
            t = dict(THETA0)
            for c in key:
                t[c] = theta[c]
                if c == "boost":
                    t["boost_nta"] = theta["boost_nta"]
            cache[key] = final_ranks(case, t, model, sim).astype(float)
        return cache[key]

    phi = {c: np.zeros(len(case["cuisine"])) for c in controls}
    for c in controls:
        others = [o for o in controls if o != c]
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for s in combinations(others, k):
                phi[c] += w * (rank_with(s + (c,)) - rank_with(s))
    total = rank_with(controls) - rank_with(())
    residual = total - sum(phi.values())
    return dict(phi=phi, total=total, residual=residual)


# ---------------------------------------------------------------------------------------------
# Outcome targets (interface decision 2026-11-20, "Option B"): the user states what the top 10
# should look like; the controls are found per model. Mirrored by solveTargets in web/src/steer.js.
# ---------------------------------------------------------------------------------------------
LAM_MAX = 0.5
SOLVER_GRIDS = {"alpha": [round(k * 0.02, 4) for k in range(51)],
                "beta": [round(k * 0.02, 4) for k in range(51)],
                "lam": [round(k * 0.01, 4) for k in range(51)]}
NO_TARGET = {"km": None, "popularity": None, "cuisines": None}


def top_k(case, theta, model, k=10):
    """First k picks of the final ranking (equal to final_ranks' top k, without the full MMR)."""
    z = np.asarray(case[model]["noncollab"]) + theta["w_h"] * np.asarray(case[model]["collab"])
    s0 = minmax(z)
    s1 = s0 - theta["alpha"] * np.asarray(case["distance_n"])
    s2 = s1 - theta["beta"] * np.asarray(case["popularity_n"]) + boost_vector(case, theta)
    rel = minmax(s2)
    lam = theta["lam"]
    if lam == 0:
        return order(rel)[:k]
    cu, nt = np.asarray(case["cuisine"]), np.asarray(case["nta"])
    max_sim = np.zeros(len(rel))
    taken = np.zeros(len(rel), bool)
    out = []
    for _ in range(k):
        v = (1 - lam) * rel - lam * max_sim
        v[taken] = -np.inf
        i = int(np.argmax(v))
        out.append(i)
        taken[i] = True
        max_sim = np.maximum(max_sim, SIM_CUISINE * (cu == cu[i]) + SIM_NTA * (nt == nt[i]))
    return np.array(out)


def outcomes(case, top):
    # Plain left-to-right sums, the same order as the browser, so threshold checks agree exactly. Not sum():
    # from Python 3.12 it compensates rounding, which can land the mean on the other side of a target.
    km = pop = 0.0
    for i in top:
        km += float(case["candidates"]["distance_km"][i])
        pop += float(case["popularity_n"][i])
    km /= len(top)
    pop /= len(top)
    return {"km": km, "popularity": pop,
            "cuisines": len(set(np.asarray(case["cuisine"])[top].tolist()))}


def _meets(o, targets, key):
    t = targets.get(key)
    if t is None:
        return True
    return o["cuisines"] >= t if key == "cuisines" else o[key] <= t


def solve_targets(case, model, targets, w_h=1.0, passes=2):
    """Smallest alpha / beta / lam (scanned upward on fixed grids, two coordinate passes) meeting
    targets {"km": max mean km, "popularity": max mean percentile, "cuisines": min distinct groups};
    None means no target. Returns theta, top-10 indices, outcomes and which targets are met."""
    theta = dict(THETA0, w_h=w_h)
    knobs = (("alpha", "km"), ("beta", "popularity"), ("lam", "cuisines"))
    for _ in range(passes):
        for key, target in knobs:
            if targets.get(target) is None:
                continue
            grid = SOLVER_GRIDS[key]
            chosen = grid[-1]
            for v in grid:
                theta[key] = v
                if _meets(outcomes(case, top_k(case, theta, model)), targets, target):
                    chosen = v
                    break
            theta[key] = chosen
    top = top_k(case, theta, model)
    o = outcomes(case, top)
    return {"theta": theta, "top": [int(i) for i in top], "outcomes": o,
            "met": {k: _meets(o, targets, k) for k in ("km", "popularity", "cuisines")}}


def solve_boost(case, model, theta, nta):
    """Smallest boost on BOOST_GRID that puts a restaurant from neighbourhood `nta` into the top 10, given
    the other controls in theta. Returns (theta with boost, reached?)."""
    t = dict(theta, boost_nta=int(nta))
    codes = np.asarray(case["nta"])
    for b in BOOST_GRID:
        t["boost"] = b
        if np.any(codes[top_k(case, t, model)] == nta):
            return t, True
    t["boost"] = BOOST_GRID[-1]
    return t, False
