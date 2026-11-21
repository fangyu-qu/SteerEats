#!/usr/bin/env python3
"""Step 26: similarity map (stretch view S1). Is the restaurant space structured by cuisine and place?

Item factors: every review before the 2020-08-01 snapshot (all reviewers, not only the sample; nothing after
the cutoff), restaurants with at least MIN_REVIEWERS distinct reviewers, users with at least 2 such reviews.
Binary user x restaurant matrix, each user's row scaled by 1/sqrt(reviews) so heavy reviewers do not dominate;
TruncatedSVD to K dimensions; restaurant factors = V * sigma, L2-normalized (so Euclidean = cosine order).

Projections (course week 7 checklist): UMAP (n_neighbors 15, min_dist 0.1, cosine) with two seeds, PCA beside
it, and the same UMAP on same-size Gaussian noise as a structure-free baseline. Scores, on a fixed random
subsample of SUB restaurants: trustworthiness and continuity (Venna & Kaski, k = 15; continuity = trustworthiness
with the two spaces swapped), and over all restaurants k-NN accuracy (k = 15, majority vote, self excluded) for
cuisine group and borough, with the majority-class rate as chance. Per cuisine group: neighbour purity (share of
the 15 nearest restaurants in the same group) in the factor space and in UMAP, against the group's share.
Parameter sweep (appendix figure): n_neighbors {5, 15, 50} x min_dist {0, 0.1, 0.5} on the subsample.

Writes web/public/data/similarity_map.json (positions, labels, scores; ids only for demo-case candidates),
outputs/similarity_map/SIMILARITY_MAP.md, metrics.csv, purity.csv, sweep.png.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS, RESTAURANT_ATTRS, REVIEWS_CLEAN, ROOT, TEST_START

SEED, SEED2 = 42, 7
MIN_REVIEWERS = 20
K = 64
NN, MIN_DIST = 15, 0.1
SUB = 3000
OUT = OUTPUTS / "similarity_map"
WEB = ROOT / "web" / "public" / "data" / "similarity_map.json"
CASES = ROOT / "web" / "public" / "data" / "cases"


def factors():
    r = pd.read_parquet(REVIEWS_CLEAN, columns=["user_id", "gmap_id", "local_date"])
    r = r[pd.to_datetime(r.local_date) < pd.Timestamp(TEST_START)].drop_duplicates(["user_id", "gmap_id"])
    n_rev = r.groupby("gmap_id").user_id.nunique()
    items = np.sort(n_rev[n_rev >= MIN_REVIEWERS].index.to_numpy())
    r = r[r.gmap_id.isin(items)]
    per_user = r.groupby("user_id").gmap_id.transform("size")
    r = r[per_user >= 2]
    users, ui = np.unique(r.user_id.to_numpy(), return_inverse=True)
    ii = np.searchsorted(items, r.gmap_id.to_numpy())
    w = 1 / np.sqrt(np.bincount(ui)[ui])
    X = sp.csr_matrix((w, (ui, ii)), shape=(len(users), len(items)))
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(K, random_state=SEED, n_iter=7).fit(X)
    F = svd.components_.T * svd.singular_values_
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    return items, F, dict(reviews=int(len(r)), users=int(len(users)), restaurants=int(len(items)),
                          explained=float(svd.explained_variance_ratio_.sum()))


def umap2(X, seed, nn=NN, md=MIN_DIST):
    import umap
    return umap.UMAP(n_neighbors=nn, min_dist=md, metric="cosine" if X.shape[1] > 2 else "euclidean",
                     random_state=seed, n_jobs=1).fit_transform(X)


def knn_acc(Z, labels, k=NN):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    nb = labels[idx]
    pred = np.array([np.bincount(row).argmax() for row in nb])
    same = (nb == labels[:, None]).mean(axis=1)
    return float((pred == labels).mean()), same


def trust_cont(X, Y, k=NN):
    from sklearn.manifold import trustworthiness
    return float(trustworthiness(X, Y, n_neighbors=k)), float(trustworthiness(Y, X, n_neighbors=k))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    items, F, info = factors()
    print(f"factors: {info} ({time.time() - t0:.0f}s)", flush=True)
    attrs = pd.read_parquet(RESTAURANT_ATTRS).set_index("gmap_id").loc[items]
    cuisines = sorted(attrs.cuisine_group.fillna("Unknown").unique())
    boroughs = sorted(attrs.boroname.fillna("Unknown").unique())
    cu = np.searchsorted(cuisines, attrs.cuisine_group.fillna("Unknown").to_numpy())
    bo = np.searchsorted(boroughs, attrs.boroname.fillna("Unknown").to_numpy())

    from sklearn.decomposition import PCA
    emb = {"UMAP": umap2(F, SEED), f"UMAP, seed {SEED2}": umap2(F, SEED2), "PCA": PCA(2, random_state=SEED).fit_transform(F)}
    noise = rng.standard_normal(F.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    emb_noise = umap2(noise, SEED)
    print(f"projections done ({time.time() - t0:.0f}s)", flush=True)

    sub = np.sort(rng.choice(len(items), size=min(SUB, len(items)), replace=False))
    rows = []
    for name, Z in [("Item factors (64-d)", F), *emb.items(), ("UMAP of same-size noise", emb_noise)]:
        src = noise if name.startswith("UMAP of same-size noise") else F
        tc = (None, None) if Z is F else trust_cont(src[sub], Z[sub])
        a_cu, s_cu = knn_acc(Z, cu)
        a_bo, s_bo = knn_acc(Z, bo)
        rows.append(dict(space=name, trustworthiness=tc[0], continuity=tc[1], knn_cuisine=a_cu, knn_borough=a_bo))
        if name == "Item factors (64-d)":
            pur_f = s_cu
        if name == "UMAP":
            pur_u, pur_bo_u = s_cu, s_bo
        print(f"  {name}: {rows[-1]}", flush=True)
    chance = dict(knn_cuisine=float(np.bincount(cu).max() / len(cu)), knn_borough=float(np.bincount(bo).max() / len(bo)))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    purity = pd.DataFrame(dict(cuisine=cuisines, restaurants=np.bincount(cu, minlength=len(cuisines))))
    purity["share"] = purity.restaurants / len(cu)
    purity["purity_factors"] = [float(pur_f[cu == g].mean()) if (cu == g).any() else np.nan for g in range(len(cuisines))]
    purity["purity_umap"] = [float(pur_u[cu == g].mean()) if (cu == g).any() else np.nan for g in range(len(cuisines))]
    purity["lift_factors"] = purity.purity_factors / purity.share
    purity = purity.sort_values("lift_factors", ascending=False)
    purity.to_csv(OUT / "purity.csv", index=False)

    # Parameter sweep on the subsample (appendix figure)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sweep = []
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for r_, nn in enumerate((5, 15, 50)):
        for c_, md in enumerate((0.0, 0.1, 0.5)):
            Z = umap2(F[sub], SEED, nn, md)
            t, cn = trust_cont(F[sub], Z)
            sweep.append(dict(n_neighbors=nn, min_dist=md, trustworthiness=t, continuity=cn))
            ax = axes[r_, c_]
            ax.scatter(Z[:, 0], Z[:, 1], s=1.5, c="#b8bfc7", linewidths=0)
            big = np.bincount(cu[sub]).argsort()[::-1][:1]
            m = np.isin(cu[sub], big)
            ax.scatter(Z[m, 0], Z[m, 1], s=1.5, c="#15191e", linewidths=0)
            ax.set_title(f"n_neighbors {nn}, min_dist {md}\nT {t:.3f} · C {cn:.3f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"UMAP parameter sweep on {len(sub):,} restaurants (dark: {cuisines[big[0]]})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "sweep.png", dpi=150)
    pd.DataFrame(sweep).to_csv(OUT / "sweep.csv", index=False)
    print(f"sweep done ({time.time() - t0:.0f}s)", flush=True)

    # Web export: positions scaled to 0..1000 (integers), labels as indices, ids only for demo candidates
    def scale(Z):
        lo, hi = Z.min(axis=0), Z.max(axis=0)
        return np.round((Z - lo) / (hi - lo).max() * 1000).astype(int)
    U, P = scale(emb["UMAP"]), scale(emb["PCA"])
    wanted = set()
    for f in CASES.glob("demo_*.json"):
        wanted.update(json.loads(f.read_text())["candidates"]["gmap_id"])
    pos = {g: int(k) for k, g in enumerate(items) if g in wanted}
    out = dict(cutoff=TEST_START, params=dict(min_reviewers=MIN_REVIEWERS, k=K, n_neighbors=NN, min_dist=MIN_DIST, seed=SEED,
                                              metric="cosine", subsample=len(sub)),
               info=info, cuisines=cuisines, boroughs=boroughs, chance=chance,
               metrics=metrics.replace({np.nan: None}).to_dict("records"),
               purity=purity.replace({np.nan: None}).to_dict("records"),
               x=U[:, 0].tolist(), y=U[:, 1].tolist(), px=P[:, 0].tolist(), py=P[:, 1].tolist(),
               cu=cu.tolist(), bo=bo.tolist(), ids=pos)
    WEB.write_text(json.dumps(out, separators=(",", ":")))

    f3 = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    md = ["# Similarity map (S1)", "",
          f"Generated by `src/26_similarity_map.py`. {info['reviews']:,} reviews before {TEST_START} by {info['users']:,} users "
          f"(at least 2 reviews each) of {info['restaurants']:,} restaurants (at least {MIN_REVIEWERS} reviewers). "
          f"TruncatedSVD, {K} dimensions ({info['explained']:.1%} of variance). UMAP n_neighbors {NN}, min_dist {MIN_DIST}, cosine, "
          f"seed {SEED}. Trustworthiness and continuity at k = {NN} on {len(sub):,} restaurants; k-NN accuracy (k = {NN}) on all.", "",
          "| Space | Trustworthiness | Continuity | k-NN cuisine | k-NN borough |", "|---|---|---|---|---|"]
    md += [f"| {r['space']} | {f3(r['trustworthiness'])} | {f3(r['continuity'])} | {f3(r['knn_cuisine'])} | {f3(r['knn_borough'])} |" for r in rows]
    md += [f"| Chance (majority class) | | | {chance['knn_cuisine']:.3f} | {chance['knn_borough']:.3f} |", "",
           "## Neighbour purity by cuisine group (share of 15 nearest restaurants in the same group)", "",
           "| Cuisine group | Restaurants | Share | Purity, factors | Purity, UMAP | Lift, factors |", "|---|---|---|---|---|---|"]
    md += [f"| {r.cuisine} | {r.restaurants:,} | {r.share:.3f} | {f3(r.purity_factors)} | {f3(r.purity_umap)} | {r.lift_factors:.1f} |"
           for r in purity.itertuples()]
    md += ["", "## Parameter sweep (subsample)", "", "| n_neighbors | min_dist | Trustworthiness | Continuity |", "|---|---|---|---|"]
    md += [f"| {s['n_neighbors']} | {s['min_dist']} | {s['trustworthiness']:.3f} | {s['continuity']:.3f} |" for s in sweep]
    md += ["", f"Demo candidates placed: {len(pos):,} of {len(wanted):,} (the rest have fewer than {MIN_REVIEWERS} reviewers before the cutoff).",
           "", f"Runtime {time.time() - t0:.0f} s."]
    (OUT / "SIMILARITY_MAP.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
