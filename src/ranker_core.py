"""Shared, time-safe feature construction and evaluation for the SteerEats rankers.

A Snapshot holds everything known strictly before one cutoff:
  - item statistics from ALL clean NYC restaurant reviews before the cutoff (2M+ users),
    so popularity and ratings are measured on the city, not only on the 20,000-user sample;
  - item-item cosine similarity (top 50 neighbours) from the same history;
  - each sample user's own history (all their reviews before the cutoff, also pre-2017).
Static restaurant attributes come from data/clean/restaurant_attributes.parquet (see
src/12_restaurant_attributes.py for the stability assumption).

Candidates at a cutoff are restaurants with at least one review in the previous 12 months
("available"), minus the user's own history. Relevant items that are not available are counted
as cold and excluded, for every model alike.
"""
import math

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import INTERACTIONS_CLEAN, RESTAURANT_ATTRS, REVIEWS_CLEAN

K_NEIGHBORS = 50
PRIOR_STRENGTH = 10       # rating shrinkage toward the global mean
CUISINE_PRIOR = 5         # cuisine-affinity smoothing toward the city-wide share
DIST_CAP_KM = 15.0        # used by steering, not by the model features
ATTR_FEATURES = ["attr_vegetarian_options", "attr_halal", "attr_late_night", "attr_alcohol",
                 "attr_reservations", "attr_good_for_groups", "attr_good_for_kids", "attrs_missing"]

FEATURES = [
    # collaborative signal
    "cf_all", "cf_liked",
    # item popularity and quality, measured before the cutoff on all NYC reviews
    "log_popularity", "log_recent_popularity", "log_popularity_3m", "recent_share",
    "shrunk_rating", "shrunk_recent_rating", "rating_change", "months_since_last_review",
    "log_item_age_months", "recent_cf_interaction",
    # user-item context
    "log_distance_km", "same_nta", "same_borough", "cuisine_affinity",
    "price_gap", "item_price_missing", "user_price_unknown", "is_chain", "chain_affinity",
    "log_user_history",
    # static attributes
    *ATTR_FEATURES,
]
FEATURE_GROUPS = {
    "Collaborative": ["cf_all", "cf_liked", "recent_cf_interaction"],
    "Popularity": ["log_popularity", "log_recent_popularity", "log_popularity_3m", "recent_share",
                   "months_since_last_review", "log_item_age_months"],
    "Rating": ["shrunk_rating", "shrunk_recent_rating", "rating_change"],
    "Location": ["log_distance_km", "same_nta", "same_borough"],
    "Taste": ["cuisine_affinity", "price_gap", "item_price_missing", "user_price_unknown",
              "is_chain", "chain_affinity", "log_user_history"],
    "Attributes": ATTR_FEATURES,
}


def metrics(ranked, relevant, k=10):
    hit = [int(x in relevant) for x in ranked[:k]]
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hit))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return sum(hit) / len(relevant), dcg / ideal


def haversine_km(lat0, lon0, lat, lon):
    p = np.pi / 180
    a = (np.sin((lat - lat0) * p / 2) ** 2
         + np.cos(lat0 * p) * np.cos(lat * p) * np.sin((lon - lon0) * p / 2) ** 2)
    return 12742.0 * np.arcsin(np.sqrt(a))


class Data:
    """Clean inputs, loaded once and indexed by a fixed restaurant order."""

    def __init__(self):
        A = pd.read_parquet(RESTAURANT_ATTRS).reset_index(drop=True)
        self.attrs = A
        self.items = A.gmap_id.to_numpy()
        self.ii = {g: i for i, g in enumerate(self.items)}
        self.lat, self.lon = A.latitude.to_numpy(), A.longitude.to_numpy()
        self.nta = pd.factorize(A.nta2020)[0]
        self.boro = pd.factorize(A.boroname)[0]
        self.cuisine, self.cuisine_names = pd.factorize(A.cuisine_group)
        self.price = A.price_level.to_numpy(float)
        self.price_missing = A.price_missing.to_numpy(bool)
        self.is_chain = A.is_chain.to_numpy(float)
        self.attr = A[ATTR_FEATURES].to_numpy(float)

        r = pd.read_parquet(REVIEWS_CLEAN, columns=["user_id", "gmap_id", "time", "rating"])
        r = r.sort_values("time", kind="stable").drop_duplicates(["user_id", "gmap_id"])
        r["item"] = r.gmap_id.map(self.ii)
        if r["item"].isna().any():
            raise ValueError("Review of a restaurant missing from the attribute table")
        r["item"] = r["item"].astype(np.int32)
        self.reviews = r[["user_id", "item", "time", "rating"]].reset_index(drop=True)

        s = pd.read_parquet(INTERACTIONS_CLEAN)
        s["item"] = s.gmap_id.map(self.ii).astype(np.int32)
        self.interactions = s


class Snapshot:
    def __init__(self, data, cutoff, users):
        self.d, self.cutoff = data, pd.Timestamp(cutoff)
        n = len(data.items)
        h = data.reviews[data.reviews.time < self.cutoff]
        if h.empty:
            raise ValueError("Empty history")
        self.history_rows = len(h)
        t = h.time

        items = h.item.to_numpy()

        def count(mask):
            return np.bincount(items[mask], minlength=n).astype(float)

        everything = np.ones(len(h), bool)
        self.pop = count(everything)
        self.recent_pop = count((t >= self.cutoff - pd.DateOffset(months=12)).to_numpy())
        self.pop_3m = count((t >= self.cutoff - pd.DateOffset(months=3)).to_numpy())
        rating = h.rating.to_numpy(float)
        rsum = np.bincount(items, weights=rating, minlength=n)
        recent = (t >= self.cutoff - pd.DateOffset(months=12)).to_numpy()
        rsum_recent = np.bincount(items[recent], weights=rating[recent], minlength=n)
        prior = h.rating.mean()
        self.mean = (rsum + PRIOR_STRENGTH * prior) / (self.pop + PRIOR_STRENGTH)
        self.recent_mean = (rsum_recent + PRIOR_STRENGTH * self.mean) / (self.recent_pop + PRIOR_STRENGTH)
        days = (self.cutoff - t).dt.total_seconds().to_numpy() / 86400
        last = np.full(n, np.inf)
        np.minimum.at(last, items, days)
        first = np.zeros(n)
        np.maximum.at(first, items, days)
        months_last = np.where(np.isfinite(last), last, 120 * 30.4375) / 30.4375
        self.available = self.recent_pop > 0
        self.static = np.column_stack([
            np.log1p(self.pop), np.log1p(self.recent_pop), np.log1p(self.pop_3m),
            self.recent_pop / np.maximum(self.pop, 1), self.mean, self.recent_mean,
            self.recent_mean - self.mean, np.log1p(months_last), np.log1p(first / 30.4375)])
        self.static[:, 7] = months_last          # months since last review, unlogged as in 07
        self.sim = self._similarity(h, n)
        self.global_cuisine = np.bincount(data.cuisine[items], minlength=len(data.cuisine_names)) / len(h)
        self._profiles(h[h.user_id.isin(users)])

    def _similarity(self, h, n, block=1500):
        """Cosine item-item similarity on binary user histories, top K neighbours per item."""
        u_codes, _ = pd.factorize(h.user_id)
        x = sp.csr_matrix((np.ones(len(h), np.float32), (u_codes, h.item.to_numpy())),
                          shape=(u_codes.max() + 1, n))
        x = x[np.diff(x.indptr) >= 2]                        # one-review users add no co-occurrence
        norm = np.sqrt(np.asarray(x.sum(axis=0)).ravel())
        z = (x @ sp.diags(1 / np.maximum(norm, 1e-12))).astype(np.float32).tocsc()
        zt = z.T.tocsr()
        rows, cols, vals = [], [], []
        for s0 in range(0, n, block):
            s1 = min(n, s0 + block)
            sim = (zt[s0:s1] @ z).toarray()
            sim[np.arange(s1 - s0), np.arange(s0, s1)] = 0
            top = np.argpartition(-sim, K_NEIGHBORS, axis=1)[:, :K_NEIGHBORS]
            v = np.take_along_axis(sim, top, axis=1)
            keep = v > 0
            rows.append(np.repeat(np.arange(s0, s1), K_NEIGHBORS)[keep.ravel()])
            cols.append(top[keep])
            vals.append(v[keep])
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, n))

    def _profiles(self, h):
        d = self.d
        self.profile = {}
        for u, g in h.groupby("user_id", sort=False):
            items = g.item.to_numpy()
            liked = items[g.rating.to_numpy() >= 4]
            priced = d.price[items][~d.price_missing[items]]
            self.profile[u] = dict(
                seen=items, liked=liked, n=len(items),
                lat=float(np.median(d.lat[items])), lon=float(np.median(d.lon[items])),
                nta=int(np.bincount(d.nta[items]).argmax()), boro=int(np.bincount(d.boro[items]).argmax()),
                cuisine=np.bincount(d.cuisine[items], minlength=len(d.cuisine_names)),
                price=float(priced.mean()) if len(priced) else np.nan,
                chain=float(d.is_chain[items].mean()))

    def user(self, u):
        """Feature matrix (all restaurants x FEATURES), the user's seen items, and raw CF scores."""
        d, p = self.d, self.profile[u]
        n = len(d.items)
        row = np.zeros(n, np.float32)
        row[p["seen"]] = 1
        cf = self.sim.T.dot(row)
        row_l = np.zeros(n, np.float32)
        row_l[p["liked"]] = 1
        cf_l = self.sim.T.dot(row_l)
        dist = haversine_km(p["lat"], p["lon"], d.lat, d.lon)
        aff = (p["cuisine"][d.cuisine] + CUISINE_PRIOR * self.global_cuisine[d.cuisine]) / (p["n"] + CUISINE_PRIOR)
        unknown = np.isnan(p["price"])
        gap = np.where(d.price_missing | unknown, 0.0, np.abs(d.price - (0 if unknown else p["price"])))
        f = np.column_stack([
            np.log1p(cf), np.log1p(cf_l), self.static[:, :8], self.static[:, 8],
            np.log1p(cf_l) * (self.recent_mean - 3),
            np.log1p(dist), d.nta == p["nta"], d.boro == p["boro"], aff,
            gap, d.price_missing, np.full(n, float(unknown)), d.is_chain, d.is_chain * p["chain"],
            np.full(n, np.log1p(p["n"])),
            d.attr]).astype(np.float32)
        if f.shape[1] != len(FEATURES) or not np.isfinite(f).all():
            raise ValueError("Bad feature matrix")
        return f, p["seen"], cf


def eligible(snap, target):
    """(user, set of relevant item ids) for users with history and an available, unseen positive."""
    if (target.time < snap.cutoff).any():
        raise ValueError("Target precedes snapshot cutoff")
    pos = target[target.label == 1].groupby("user_id").item.agg(set)
    cases, audit = [], dict(no_history=0, no_available_positive=0, positives=0, cold_positives=0)
    for u, items in pos.items():
        if u not in snap.profile:
            audit["no_history"] += 1
            continue
        unseen = items - set(snap.profile[u]["seen"].tolist())
        rel = {i for i in unseen if snap.available[i]}
        audit["positives"] += len(unseen)
        audit["cold_positives"] += len(unseen) - len(rel)
        if rel:
            cases.append((u, rel))
        else:
            audit["no_available_positive"] += 1
    audit["users"] = len(cases)
    return cases, audit


def examples(snap, target, rng, ratio=4):
    """Positives plus `ratio` sampled negatives each: half uniform, half popularity-matched."""
    cases, _ = eligible(snap, target)
    observed = target.groupby("user_id").item.agg(set)
    xs, ys, groups = [], [], []
    for u, rel in cases:
        f, seen, _ = snap.user(u)
        pos = np.array(sorted(rel))
        mask = snap.available.copy()
        mask[seen] = False
        mask[list(observed[u])] = False
        pool = np.flatnonzero(mask)
        n_neg = min(len(pool), ratio * len(pos))
        if not n_neg:
            continue
        n_uni = n_neg // 2
        uni = rng.choice(pool, size=n_uni, replace=False)
        rest = np.setdiff1d(pool, uni)
        w = snap.pop[rest] / snap.pop[rest].sum()
        popm = rng.choice(rest, size=min(len(rest), n_neg - n_uni), replace=False, p=w)
        neg = np.concatenate([uni, popm])
        xs.extend([f[pos], f[neg]])
        ys.extend([np.ones(len(pos)), np.zeros(len(neg))])
        groups.extend([u] * (len(pos) + len(neg)))
    return np.vstack(xs), np.concatenate(ys), np.array(groups)


def evaluate(snap, target, scorers, k=10, batch=64, log_every=2000):
    """Full-candidate ranking for every eligible user.

    scorers: name -> function(feature_matrix) -> scores. Baselines are added here.
    Returns per-user records (list of dicts) and the eligibility audit.
    """
    cases, audit = eligible(snap, target)
    recs = []
    for b0 in range(0, len(cases), batch):
        chunk = cases[b0:b0 + batch]
        feats, cands, cfs = [], [], []
        for u, _ in chunk:
            f, seen, cf = snap.user(u)
            avail = snap.available.copy()
            avail[seen] = False
            ids = np.flatnonzero(avail)
            feats.append(f[ids]); cands.append(ids); cfs.append(cf[ids])
        stacked = np.vstack(feats)
        bounds = np.cumsum([0] + [len(c) for c in cands])
        scores = {name: fn(stacked) for name, fn in scorers.items()}
        for j, (u, rel) in enumerate(chunk):
            ids = cands[j]
            orders = {
                "Popularity": ids[np.lexsort((ids, -snap.pop[ids]))],
                "Recent popularity": ids[np.lexsort((ids, -snap.recent_pop[ids]))],
                "Item-CF": ids[np.lexsort((ids, -snap.pop[ids], -cfs[j]))],
            }
            for name, sc in scores.items():
                s = sc[bounds[j]:bounds[j + 1]]
                orders[name] = ids[np.lexsort((ids, -s))]
            for name, order in orders.items():
                r, nd = metrics(order[:k].tolist(), rel, k)
                recs.append(dict(user_id=u, model=name, recall=r, ndcg=nd, n_relevant=len(rel)))
        if log_every and (b0 // batch) % max(1, log_every // batch) == 0:
            print(f"  evaluated {min(b0 + batch, len(cases))}/{len(cases)} users", flush=True)
    return recs, audit
