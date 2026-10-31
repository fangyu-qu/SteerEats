"""Step 5: first baselines on the time split: popularity, recent popularity and item-based CF.

For validation, models use the training period; for test, they use everything before the test period.
A user is evaluated if they have history before the period and at least one 4-5 star review in the period of a
restaurant they had not reviewed before. Candidates are restaurants seen before the period, minus the user's
history. Item-CF: cosine similarity on the binary user-restaurant matrix, top 50 neighbors per restaurant, user
score = sum of similarities to the restaurants in their history; if it scores fewer than 10, popularity fills in.
Output: outputs/05_baselines.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS, PROCESSED

K = 10
NEIGHBORS = 50


def ndcg_recall(ranked, relevant):
    hits = np.isin(ranked[:K], list(relevant))
    dcg = (hits / np.log2(np.arange(2, K + 2))).sum()
    idcg = (1 / np.log2(np.arange(2, min(len(relevant), K) + 2))).sum()
    return dcg / idcg, hits.sum() / len(relevant)


def top_neighbors(x):
    """Item-item cosine on a binary users x items matrix, keeping the top NEIGHBORS per item."""
    norms = np.sqrt(np.asarray(x.sum(axis=0)).ravel())
    xn = x @ sp.diags(1 / np.maximum(norms, 1e-12))
    sim = (xn.T @ xn).tocsr()
    sim.setdiag(0)
    sim.eliminate_zeros()
    rows, cols, vals = [], [], []
    for i in range(sim.shape[0]):
        start, end = sim.indptr[i], sim.indptr[i + 1]
        if end == start:
            continue
        idx, data = sim.indices[start:end], sim.data[start:end]
        keep = np.argsort(-data)[:NEIGHBORS]
        rows.extend([i] * len(keep))
        cols.extend(idx[keep])
        vals.extend(data[keep])
    return sp.csr_matrix((vals, (rows, cols)), shape=sim.shape)


def evaluate(history, target, period_start):
    items = np.sort(history.gmap_id.unique())
    item_ix = {g: i for i, g in enumerate(items)}
    users = np.sort(history.user_id.unique())
    user_ix = {u: i for i, u in enumerate(users)}
    x = sp.csr_matrix((np.ones(len(history)), (history.user_id.map(user_ix), history.gmap_id.map(item_ix))),
                      shape=(len(users), len(items)))
    x.data[:] = 1

    pop_all = np.asarray(x.sum(axis=0)).ravel()
    recent = history[history.time >= period_start - pd.DateOffset(months=12)]
    pop_recent = np.bincount(recent.gmap_id.map(item_ix), minlength=len(items)).astype(float)
    neighbors = top_neighbors(x)

    seen = history.groupby("user_id").gmap_id.agg(set)
    positives = target[target.rating >= 4].groupby("user_id").gmap_id.agg(set)
    eval_users = [u for u in positives.index if u in user_ix and positives[u] - seen[u]]
    order_all, order_recent = np.argsort(-pop_all), np.argsort(-pop_recent)

    scores = {"Popularity (all history)": [], "Popularity (last 12 months)": [], "Item-CF": []}
    for u in eval_users:
        relevant = positives[u] - seen[u]
        hist_ix = x[user_ix[u]].indices
        exclude = set(hist_ix)
        for name, order in [("Popularity (all history)", order_all), ("Popularity (last 12 months)", order_recent)]:
            ranked = items[[i for i in order[:K + len(exclude)] if i not in exclude][:K]]
            scores[name].append(ndcg_recall(ranked, relevant))
        cf = (x[user_ix[u]] @ neighbors).toarray().ravel()
        cf[hist_ix] = -np.inf
        top = [i for i in np.argsort(-cf)[:K] if cf[i] > 0]
        fill = [i for i in order_all if i not in exclude and i not in top][:K - len(top)]
        scores["Item-CF"].append(ndcg_recall(items[top + fill], relevant))

    return len(eval_users), {name: np.mean(vals, axis=0) for name, vals in scores.items()}


def main():
    s = pd.read_parquet(PROCESSED / "sample_interactions.parquet")
    val_start, test_start = s[s.split == "val"].time.min(), s[s.split == "test"].time.min()
    rows = []
    for period, history, target, start in [
        ("validation", s[s.split == "train"], s[s.split == "val"], val_start),
        ("test", s[s.split != "test"], s[s.split == "test"], test_start),
    ]:
        n_users, results = evaluate(history, target, start)
        for model, (ndcg, recall) in results.items():
            rows.append({"period": period, "users": n_users, "model": model,
                         "Recall@10": round(recall, 4), "NDCG@10": round(ndcg, 4)})
    out = pd.DataFrame(rows)
    OUTPUTS.mkdir(exist_ok=True)
    out.to_csv(OUTPUTS / "05_baselines.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
