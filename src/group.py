"""Group aggregation: reference implementation (web/src/group.js mirrors it; change both or neither).

Inputs are plain arrays so the same arithmetic runs in the browser:
  scores[m][i]   member m's (steered) score for pool item i; higher is better
  ids[i]         a stable integer id per pool item, used only to break ties

satisfaction()   turns each member's scores into [0, 1] over the pool
group_score()    combines members with one of three strategies
rank_pool()      orders the feasible items, best first, ties by id
"""
import numpy as np

TAU = 0.40                      # misery threshold of "average without misery", fixed in advance
STRATEGIES = ("average", "least_misery", "average_without_misery")
STRATEGY_LABEL = {"average": "Average", "least_misery": "Least misery",
                  "average_without_misery": "Average without misery"}


def satisfaction(scores, how="minmax"):
    """Member satisfaction in [0, 1] over the pool.

    minmax      (score - min) / (max - min) over the pool; a member whose scores are all equal is
                indifferent and gets 1 everywhere, so they never block a place
    percentile  1 for the member's best item, 0 for the worst, ties sharing their mean rank
    """
    s = np.asarray(scores, dtype=float)
    if how == "minmax":
        lo, hi = s.min(axis=1, keepdims=True), s.max(axis=1, keepdims=True)
        span = hi - lo
        return np.where(span > 0, (s - lo) / np.where(span > 0, span, 1.0), 1.0)
    if how == "percentile":
        n = s.shape[1]
        if n == 1:
            return np.ones_like(s)
        out = np.empty_like(s)
        for m, row in enumerate(s):
            order = np.argsort(row, kind="stable")
            ranks = np.empty(n)
            ranks[order] = np.arange(n)
            for v in np.unique(row):                      # ties share their mean rank
                tie = row == v
                ranks[tie] = ranks[tie].mean()
            out[m] = ranks / (n - 1)
        return out
    raise ValueError(how)


def group_score(sat, strategy, tau=TAU, allowed=None):
    """(score per item, feasible mask). `allowed` carries hard vetoes: an item any member vetoes is out."""
    sat = np.asarray(sat, dtype=float)
    feasible = np.ones(sat.shape[1], bool) if allowed is None else np.asarray(allowed, bool).copy()
    worst = sat.min(axis=0)
    if strategy == "average":
        score = sat.mean(axis=0)
    elif strategy == "least_misery":
        score = worst
    elif strategy == "average_without_misery":
        score = sat.mean(axis=0)
        feasible &= worst >= tau
    else:
        raise ValueError(strategy)
    return score, feasible


def rank_pool(score, feasible, ids):
    """Positions of the feasible items, best first; ties broken by the smaller id."""
    ids = np.asarray(ids)
    pos = np.flatnonzero(feasible)
    return pos[np.lexsort((ids[pos], -np.asarray(score)[pos]))]


def group_list(scores, ids, strategy, k=10, tau=TAU, how="minmax", allowed=None):
    sat = satisfaction(scores, how)
    score, feasible = group_score(sat, strategy, tau, allowed)
    order = rank_pool(score, feasible, ids)[:k]
    return order, sat, feasible
