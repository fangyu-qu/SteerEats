#!/usr/bin/env python3
"""Pointwise logistic restaurant ranking, with strictly historical features.

Requires the existing sample_interactions.parquet; never manufactures a result.
Training negatives are unlabeled non-interactions, not known dislikes. Scores
are ranking scores and are NOT calibrated probabilities of liking a restaurant.
Evaluation ranks the entire warm catalog, not sampled negatives.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def metrics(ranked, relevant, k=10):
    if not relevant:
        raise ValueError("Metrics require at least one eligible relevant item")
    hit = [int(x in relevant) for x in ranked[:k]]
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hit))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return sum(hit) / len(relevant), dcg / ideal


def neighbors(x, n=50):
    norm = np.sqrt(np.asarray(x.sum(axis=0)).ravel())
    z = x @ sp.diags(1 / np.maximum(norm, 1e-12))
    sim = (z.T @ z).tocsr()
    sim.setdiag(0)
    sim.eliminate_zeros()
    rows, cols, vals = [], [], []
    for i in range(sim.shape[0]):
        a, b = sim.indptr[i:i + 2]
        ids, scores = sim.indices[a:b], sim.data[a:b]
        order = np.lexsort((ids, -scores))[:n]
        rows.extend([i] * len(order))
        cols.extend(ids[order])
        vals.extend(scores[order])
    return sp.csr_matrix((vals, (rows, cols)), shape=sim.shape)


FEATURES = ['cf_all', 'cf_liked', 'log_popularity', 'log_recent_popularity',
            'shrunk_rating', 'shrunk_recent_rating', 'rating_change',
            'recent_share', 'months_since_last_review', 'recent_cf_interaction']


class Snapshot:
    """All aggregates and similarities are recomputed before one cutoff."""
    def __init__(self, history, cutoff):
        if history.empty or not (history.time < cutoff).all():
            raise ValueError("Snapshot history must be nonempty and before cutoff")
        self.cutoff = cutoff
        self.items = np.sort(history.gmap_id.unique())
        self.users = np.sort(history.user_id.unique())
        self.ii = {v: i for i, v in enumerate(self.items)}
        self.ui = {v: i for i, v in enumerate(self.users)}
        self.catalog = set(self.items)
        u = history.user_id.map(self.ui).to_numpy()
        j = history.gmap_id.map(self.ii).to_numpy()
        self.x = sp.csr_matrix((np.ones(len(history)), (u, j)),
                               shape=(len(self.users), len(self.items)))
        self.x.data[:] = 1
        liked = history.rating.to_numpy() >= 4
        self.like = sp.csr_matrix((np.ones(liked.sum()), (u[liked], j[liked])),
                                  shape=self.x.shape)
        self.like.data[:] = 1
        self.sim = neighbors(self.x)
        self.pop = np.asarray(self.x.sum(axis=0)).ravel()
        recent = history[history.time >= cutoff - pd.DateOffset(months=12)]
        a = history.groupby('gmap_id').rating.agg(['count', 'sum']).reindex(self.items)
        b = recent.groupby('gmap_id').rating.agg(['count', 'sum']).reindex(self.items).fillna(0)
        self.recent_pop = b['count'].to_numpy()
        prior, strength = history.rating.mean(), 10
        self.mean = ((a['sum'] + strength * prior) / (a['count'] + strength)).to_numpy()
        self.recent_mean = ((b['sum'] + strength * self.mean) /
                            (b['count'] + strength)).to_numpy()
        last = history.groupby('gmap_id').time.max().reindex(self.items)
        age = (cutoff - last).dt.total_seconds().to_numpy() / (86400 * 30.4375)
        self.static = np.column_stack([
            np.log1p(self.pop), np.log1p(self.recent_pop), self.mean,
            self.recent_mean, self.recent_mean - self.mean,
            self.recent_pop / np.maximum(self.pop, 1), np.log1p(age)])

    def user(self, user_id):
        row = self.ui[user_id]
        seen = self.x[row].indices
        cf = (self.x[row] @ self.sim).toarray().ravel()
        liked_cf = (self.like[row] @ self.sim).toarray().ravel()
        features = np.column_stack([np.log1p(cf), np.log1p(liked_cf), self.static,
                                     np.log1p(liked_cf) * (self.recent_mean - 3)])
        assert features.shape[1] == len(FEATURES)
        if not np.isfinite(features).all():
            raise ValueError('Non-finite historical features')
        return features, seen, cf


def eligible(snapshot, target):
    """Count cold items separately; relevance is restricted to rankable items."""
    if not target.empty and not (target.time >= snapshot.cutoff).all():
        raise ValueError('Target precedes snapshot cutoff')
    pos = target[target.rating >= 4].groupby('user_id').gmap_id.agg(set)
    cases, cold, total, no_history, no_warm = [], 0, 0, 0, 0
    for u, items in pos.items():
        if u not in snapshot.ui:
            no_history += 1
            continue
        seen = set(snapshot.items[snapshot.x[snapshot.ui[u]].indices])
        unseen = items - seen
        total += len(unseen)
        cold += len(unseen - snapshot.catalog)
        rel = unseen & snapshot.catalog
        if rel:
            cases.append((u, rel))
        else:
            no_warm += 1
    return cases, dict(users=len(cases), cold_positive_items=cold,
                       total_positive_items=total, users_without_history=no_history,
                       users_without_warm_positive=no_warm)


def examples(snapshot, target, rng, ratio):
    cases, _ = eligible(snapshot, target)
    observed = target.groupby('user_id').gmap_id.agg(set)
    xs, ys = [], []
    for u, rel in cases:
        f, seen, _ = snapshot.user(u)
        positive = np.array(sorted(snapshot.ii[v] for v in rel))
        mask = np.ones(len(snapshot.items), dtype=bool)
        mask[seen] = False
        # Do not mislabel any observed item in this horizon as unobserved.
        known = [snapshot.ii[v] for v in observed[u] if v in snapshot.ii]
        mask[known] = False
        pool = np.flatnonzero(mask)
        n = min(len(pool), ratio * len(positive))
        negative = rng.choice(pool, size=n, replace=False)
        # Skip one-class users with no available sampled negatives.
        if not n:
            continue
        xs.extend([f[positive], f[negative]])
        ys.extend([np.ones(len(positive)), np.zeros(n)])
    if not xs:
        raise ValueError('No training pairs; need earlier history and later warm positives')
    return np.vstack(xs), np.concatenate(ys)


def evaluate(snapshot, target, models, period, out):
    cases, audit = eligible(snapshot, target)
    if not cases:
        raise ValueError('No eligible evaluation users')
    sums = {name: np.zeros(2) for name in ['Popularity', 'Recent popularity', 'Item-CF', *models]}
    records = []
    for n, (u, relevant) in enumerate(cases):
        f, seen, cf = snapshot.user(u)
        available = np.ones(len(snapshot.items), dtype=bool)
        available[seen] = False
        ids = np.flatnonzero(available)
        base = {'Popularity': snapshot.pop, 'Recent popularity': snapshot.recent_pop}
        # Match original CF fallback: positive CF first, then popularity.
        cf_order = ids[np.lexsort((ids, -snapshot.pop[ids], -cf[ids]))]
        orders = {'Item-CF': cf_order}
        for name, scores in base.items():
            orders[name] = ids[np.lexsort((ids, -scores[ids]))]
        for name, model in models.items():
            # Explicit linear contraction avoids platform BLAS matmul warnings.
            # This is the fitted pipeline's exact binary decision function.
            standardized = model.steps[0][1].transform(f[ids])
            linear = model.steps[-1][1]
            score = np.einsum('ij,j->i', standardized, linear.coef_[0]) + linear.intercept_[0]
            if not np.isfinite(score).all():
                raise ValueError('Non-finite logistic ranking scores')
            orders[name] = ids[np.lexsort((ids, -score))]
        for name, order in orders.items():
            recall, ndcg = metrics(snapshot.items[order[:10]], relevant)
            sums[name] += [recall, ndcg]
            records.append(dict(user_id=u, model=name, recall=recall, ndcg=ndcg))
        if (n + 1) % 1000 == 0:
            print(f'{period}: evaluated {n + 1}/{len(cases)} users', flush=True)
    pd.DataFrame(records).to_csv(out / f'{period}_per_user.csv', index=False)
    rows = [dict(period=period, model=name, users=len(cases),
                 **{'Recall@10': float(value[0] / len(cases)),
                    'NDCG@10': float(value[1] / len(cases))}) for name, value in sums.items()]
    return rows, audit


def fit(x, y, c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if not np.isfinite(x).all():
        raise ValueError('Non-finite training features')
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=c, max_iter=1000, solver='liblinear', random_state=42)).fit(x, y)
    linear = model.steps[-1][1]
    if not np.isfinite(linear.coef_).all() or not np.isfinite(linear.intercept_).all():
        raise ValueError('Non-finite logistic coefficients')
    if linear.n_iter_.max() >= 1000:
        raise ValueError('Logistic solver reached the iteration limit')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interactions', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('ranker_outputs'))
    parser.add_argument('--negative-ratio', type=int, default=5)
    args = parser.parse_args()
    if not args.interactions.is_file():
        parser.error(f'Missing real interaction data: {args.interactions}')
    if args.negative_ratio < 1:
        parser.error('--negative-ratio must be positive')
    global np, pd, sp
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import sklearn
    import joblib
    s = pd.read_parquet(args.interactions)
    required = {'user_id', 'gmap_id', 'rating', 'time', 'split'}
    if not required.issubset(s.columns):
        raise ValueError(f'Missing columns: {required - set(s.columns)}')
    s['time'] = pd.to_datetime(s.time)
    if s[list(required)].isna().any().any() or not s.rating.between(1, 5).all():
        raise ValueError('Null fields or invalid ratings')
    if s.duplicated(['user_id', 'gmap_id']).any():
        raise ValueError('Expected one first interaction per user/restaurant')
    if set(s.split) != {'train', 'val', 'test'}:
        raise ValueError('Expected train/val/test labels')
    train = s[s.split == 'train'].copy()
    val = s[s.split == 'val'].copy()
    test = s[s.split == 'test'].copy()
    val_start, test_start = val.time.min(), test.time.min()
    assert train.time.max() < val_start and val.time.max() < test_start
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    # Disjoint target blocks inside TRAIN only; no holdout labels fit the initial ranker.
    cuts = sorted(set([train.time.quantile(q).to_period('M').to_timestamp()
                       for q in [.4, .6, .8]] + [val_start]))
    blocks, targets = [], []
    for start, end in zip(cuts[:-1], cuts[1:]):
        print(f'Training snapshot: {start} -> {end}', flush=True)
        snap = Snapshot(train[train.time < start], start)
        x, y = examples(snap, train[(train.time >= start) & (train.time < end)],
                        rng, args.negative_ratio)
        blocks.append(x)
        targets.append(y)
    x, y = np.vstack(blocks), np.concatenate(targets)
    fitted = {f'Logistic C={c}': fit(x, y, c) for c in [.1, 1., 10.]}
    snap = Snapshot(train, val_start)
    val_rows, val_audit = evaluate(snap, val, fitted, 'validation', args.out)
    candidates = [r for r in val_rows if r['model'].startswith('Logistic')]
    best = max(candidates, key=lambda r: (r['NDCG@10'], r['Recall@10']))
    c = float(best['model'].split('=')[1])
    # Once C is selected, validation labels may join training for the final TEST fit.
    xv, yv = examples(snap, val, rng, args.negative_ratio)
    final = fit(np.vstack([x, xv]), np.concatenate([y, yv]), c)
    test_snap = Snapshot(s[s.split != 'test'], test_start)
    test_rows, test_audit = evaluate(test_snap, test, {'Logistic regression': final},
                                     'test', args.out)
    pd.DataFrame(val_rows + test_rows).to_csv(args.out / 'metrics.csv', index=False)
    linear = final.steps[-1][1]
    pd.DataFrame({'feature': FEATURES, 'standardized_coefficient': linear.coef_[0]}).to_csv(
        args.out / 'coefficients.csv', index=False)
    joblib.dump(final, args.out / 'logistic_pipeline.joblib')
    metadata = dict(seed=42, negative_ratio=args.negative_ratio, selected_C=c,
                    feature_names=FEATURES, training_snapshots=[str(cut) for cut in cuts],
                    validation_start=str(val_start), test_start=str(test_start),
                    validation_audit=val_audit, test_audit=test_audit,
                    input_sha256=hashlib.sha256(args.interactions.read_bytes()).hexdigest(),
                    solver='liblinear', solver_iterations=linear.n_iter_.tolist(),
                    versions=dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
                    note='Warm full-catalog ranking; sampled non-interaction training negatives; '
                         'no calibrated preference probabilities; no 2026 metadata features.')
    (args.out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    r = next(r for r in test_rows if r['model'] == 'Logistic regression')
    sentence = (f"On the warm-catalog test ({r['users']:,} users), logistic regression achieves "
                f"Recall@10 = {r['Recall@10']:.4f} and NDCG@10 = {r['NDCG@10']:.4f}; "
                "baselines use the same candidate catalog and eligibility rules.")
    (args.out / 'report_sentence.txt').write_text(sentence + '\n')
    print(sentence)


if __name__ == '__main__':
    main()
