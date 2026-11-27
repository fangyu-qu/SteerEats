"""Diner app engine: reference implementation (web/src/app/engine.js mirrors it; change both or neither).

From a party (one or two sample taste profiles with start points) and its settings to the list of K places.
See the header of web/src/app/engine.js for the steps. Plain Python floats and left-to-right sums are used
where the browser's arithmetic order matters, so thresholds and ties agree exactly (tests/check_app.py).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from group import TAU, group_score, rank_pool, satisfaction  # noqa: E402
from steering import minmax  # noqa: E402
from server.nl import pool_mask  # noqa: E402

K = 6
SIM_CUISINE = SIM_NTA = 0.5
THETA0 = {"w_h": 1.0, "alpha": 0.0, "beta": 0.0, "lam": 0.0}
SOLVER_GRIDS = {"alpha": [round(k * 0.02, 4) for k in range(51)], "beta": [round(k * 0.02, 4) for k in range(51)],
                "lam": [round(k * 0.01, 4) for k in range(51)]}
STRATEGY = {"fair": "average_without_misery", "floor": "least_misery", "solo": "average"}


def km(lat0, lon0, lat, lon):
    p = math.pi / 180
    a = math.sin((lat - lat0) * p / 2) ** 2 + math.cos(lat0 * p) * math.cos(lat * p) * math.sin((lon - lon0) * p / 2) ** 2
    return math.floor(12742.0 * math.asin(math.sqrt(a)) * 1e6 + 0.5) / 1e6


def build_pool(places, cuisine_names, members, filters):
    union = sorted({i for m in members for i in m["profile"]["candidates"]})
    view = {"cuisine": [places["cuisine"][i] for i in union], "cuisine_names": cuisine_names,
            "candidates": {"price": [places["price"][i] for i in union], "borough": [places["borough"][i] for i in union],
                           "nta_name": [places["nta_name"][i] for i in union]},
            "feature_values": {f"attr_{a}": [col[i] for i in union] for a, col in places["needs"].items()}}
    keep = pool_mask(view, filters)
    lo = filters.get("min_rating")
    return [i for i, k in zip(union, keep) if k and (lo is None or places["rating"][i] >= lo)]


def popularity(places, ids):
    n = len(ids)
    order = sorted(range(n), key=lambda k: (places["recent_reviews"][ids[k]], k))
    out = [0.0] * n
    for rank, k in enumerate(order):
        out[k] = rank / (n - 1) if n > 1 else 0.0
    return out


def prepare(places, cap, members, ids):
    dist = [[km(m["start"]["lat"], m["start"]["lon"], places["lat"][i], places["lon"][i]) for i in ids] for m in members]
    return {"ids": ids, "n": len(ids), "dist": dist, "dist_n": [[min(d, cap) / cap for d in row] for row in dist],
            "pop": popularity(places, ids), "cuisine": [places["cuisine"][i] for i in ids], "nta": [places["nta"][i] for i in ids]}


def member_scores(members, pre, theta):
    out = []
    for k, m in enumerate(members):
        p = m["profile"]
        z = np.array([(p["raw"][i] - p["collab"][i]) + theta["w_h"] * p["collab"][i] for i in pre["ids"]], dtype=float)
        s0 = minmax(z) if len(z) else z
        out.append(np.array([s0[j] - theta["alpha"] * pre["dist_n"][k][j] - theta["beta"] * pre["pop"][j] for j in range(pre["n"])]))
    return out


def list_of(pre, sat, strategy, lam, k=K):
    score, feasible = group_score(sat, strategy, TAU, None)
    order = [int(j) for j in rank_pool(score, feasible, pre["ids"])]
    if lam == 0 or not order:
        return order[:k], len(order)
    rel = minmax(np.array([score[j] for j in order], dtype=float))
    taken, max_sim, top = [False] * len(order), [0.0] * len(order), []
    for _ in range(min(k, len(order))):
        best, best_val = -1, -math.inf
        for a in range(len(order)):
            if taken[a]:
                continue
            v = (1 - lam) * rel[a] - lam * max_sim[a]
            if v > best_val:
                best, best_val = a, v
        taken[best] = True
        jb = order[best]
        top.append(jb)
        for a, j in enumerate(order):
            s = SIM_CUISINE * (1 if pre["cuisine"][j] == pre["cuisine"][jb] else 0) + SIM_NTA * (1 if pre["nta"][j] == pre["nta"][jb] else 0)
            if s > max_sim[a]:
                max_sim[a] = s
    return top, len(order)


def outcomes(pre, top):
    if not top:
        return {"km": 0.0, "popularity": 0.0, "cuisines": 0}
    worst = 0.0
    for row in pre["dist"]:
        s = 0.0
        for j in top:
            s += row[j]
        if s / len(top) > worst:
            worst = s / len(top)
    pop = 0.0
    for j in top:
        pop += pre["pop"][j]
    return {"km": worst, "popularity": pop / len(top), "cuisines": len({pre["cuisine"][j] for j in top})}


def _meets(o, t, key):
    if t.get(key) is None:
        return True
    return o["cuisines"] >= t[key] if key == "cuisines" else o[key] <= t[key]


def run(members, pre, theta, strategy, k=K):
    if pre["n"] == 0:
        return {"theta": dict(theta), "top": [], "count": 0, "sat": [], "outcomes": outcomes(pre, [])}
    sat = satisfaction(np.vstack(member_scores(members, pre, theta)))
    top, count = list_of(pre, sat, strategy, theta["lam"], k)
    return {"theta": dict(theta), "top": top, "count": count, "sat": sat, "outcomes": outcomes(pre, top)}


def solve(members, pre, targets, w_h, strategy, k=K, passes=2):
    theta = dict(THETA0, w_h=w_h)
    for _ in range(passes):
        for key, target in (("alpha", "km"), ("beta", "popularity"), ("lam", "cuisines")):
            if targets.get(target) is None:
                continue
            g = SOLVER_GRIDS[key]
            chosen = g[-1]
            for v in g:
                theta[key] = v
                if _meets(run(members, pre, theta, strategy, k)["outcomes"], targets, target):
                    chosen = v
                    break
            theta[key] = chosen
    res = run(members, pre, theta, strategy, k)
    res["met"] = {t: _meets(res["outcomes"], targets, t) for t in ("km", "popularity", "cuisines")}
    return res
