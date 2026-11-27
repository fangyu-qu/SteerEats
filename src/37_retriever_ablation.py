#!/usr/bin/env python3
"""Step 37: what does each stage-1 retriever contribute to candidate recall?

Stage 1 (ranker_core.candidates) takes 100 candidates from item-CF, 40 from liked-history CF, 40 nearby
and popular, and 20 trending. For every eligible test user this script measures
  - recall of each retriever's own list (its quota, on its own);
  - the held-out visits only that retriever brings into the 200 (credited in the order the union is built);
  - stage-1 recall when the retriever is removed and the 200 are filled up from item-CF
    (from liked-history CF when item-CF itself is removed);
  - recall of 200 candidates from item-CF alone, as the single-retriever alternative.
The union rebuilt here is asserted equal to ranker_core.candidates for every user.

Outputs: outputs/retrievers/RETRIEVERS.md, retrievers.csv; web/public/data/retrievers.json
Run: .venv/bin/python src/37_retriever_ablation.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS, ROOT, SEED, TEST_START
from ranker_core import CANDIDATE_QUOTAS, FEATURES, N_CANDIDATES, Data, Snapshot, candidates, eligible

OUT = OUTPUTS / "retrievers"
COL = {name: j for j, name in enumerate(FEATURES)}
LABEL = {"item_cf": "Item-CF (everything reviewed)", "liked_cf": "Liked-history CF (4-5 stars)",
         "nearby_popular": "Nearby and popular", "trending": "Trending (last 3 months)"}


def orders(snap, f, seen, cf):
    avail = snap.available.copy()
    avail[seen] = False
    ids = np.flatnonzero(avail)
    dist = np.expm1(f[ids, COL["log_distance_km"]])
    scores = {"item_cf": cf[ids] + 1e-9 * snap.pop[ids],
              "liked_cf": f[ids, COL["cf_liked"]] + 1e-9 * snap.pop[ids],
              "nearby_popular": np.log1p(snap.recent_pop[ids]) - 0.5 * dist,
              "trending": snap.pop_3m[ids]}
    return {name: ids[np.lexsort((ids, -sc))] for name, sc in scores.items()}


def union(order, quotas, fill, n=N_CANDIDATES):
    chosen, source = [], {}
    for name, q in quotas.items():
        have = set(chosen)
        new = [i for i in order[name][: q * 3] if i not in have][:q]
        chosen.extend(new)
        source.update({i: name for i in new})
    if len(chosen) < n:
        have = set(chosen)
        extra = [i for i in order[fill] if i not in have][: n - len(chosen)]
        chosen.extend(extra)
        source.update({i: "fill" for i in extra})
    return chosen[:n], source


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data = Data()
    s = data.interactions
    snap = Snapshot(data, TEST_START, set(s.user_id.unique()))
    cases, _ = eligible(snap, s[s.split == "test"])
    names = list(CANDIDATE_QUOTAS)
    rec = {k: [] for k in ["all", "item_cf_200"] + [f"alone_{n}" for n in names] + [f"without_{n}" for n in names]}
    only = {n: 0 for n in names} | {"fill": 0}
    retrieved = positives = 0
    for j, (u, rel) in enumerate(cases):
        f, seen, cf = snap.user(u)
        od = orders(snap, f, seen, cf)
        full, source = union(od, CANDIDATE_QUOTAS, "item_cf")
        assert np.array_equal(np.array(full), candidates(snap, f, seen, cf)), "union differs from ranker_core.candidates"
        hits = rel & set(full)
        positives += len(rel)
        retrieved += len(hits)
        for i in hits:
            only[source[i]] += 1
        rec["all"].append(len(hits) / len(rel))
        rec["item_cf_200"].append(len(rel & set(od["item_cf"][:N_CANDIDATES].tolist())) / len(rel))
        for n in names:
            rec[f"alone_{n}"].append(len(rel & set(od[n][:CANDIDATE_QUOTAS[n]].tolist())) / len(rel))
            rest = {k: q for k, q in CANDIDATE_QUOTAS.items() if k != n}
            loo, _ = union(od, rest, "liked_cf" if n == "item_cf" else "item_cf")
            rec[f"without_{n}"].append(len(rel & set(loo)) / len(rel))
        if j % 2000 == 0:
            print(f"  {j}/{len(cases)} ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(SEED)

    def ci(v):
        v = np.asarray(v)
        m = v[rng.integers(0, len(v), size=(2000, len(v)))].mean(axis=1)
        return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))

    allr = ci(rec["all"])
    rows = []
    for n in names:
        a, w = ci(rec[f"alone_{n}"]), ci(rec[f"without_{n}"])
        d = ci(np.asarray(rec[f"without_{n}"]) - np.asarray(rec["all"]))
        rows.append(dict(retriever=n, label=LABEL[n], quota=CANDIDATE_QUOTAS[n], recall_alone=a[0], alone_lo=a[1], alone_hi=a[2],
                         credited_positives=only[n], credited_share=only[n] / retrieved,
                         recall_without=w[0], without_lo=w[1], without_hi=w[2], change=d[0], change_lo=d[1], change_hi=d[2]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "retrievers.csv", index=False)
    cf200 = ci(rec["item_cf_200"])
    L = ["# Stage-1 retrievers: what each one contributes", "",
         f"{len(cases):,} eligible test users, snapshot {TEST_START}; {positives:,} held-out visits, {retrieved:,} of them among the "
         f"200 candidates. Stage-1 recall (mean per user): **{allr[0]:.4f}** [{allr[1]:.4f}, {allr[2]:.4f}]. "
         "Brackets are 95% bootstrap intervals over users.", "",
         "| Retriever | Quota | Recall of its own list | Retrieved visits credited to it | Stage-1 recall without it | Change |",
         "|---|---|---|---|---|---|"]
    L += [f"| {r.label} | {r.quota} | {r.recall_alone:.4f} [{r.alone_lo:.4f}, {r.alone_hi:.4f}] | {r.credited_positives:,} ({r.credited_share:.1%}) | "
          f"{r.recall_without:.4f} [{r.without_lo:.4f}, {r.without_hi:.4f}] | {r.change:+.4f} [{r.change_lo:+.4f}, {r.change_hi:+.4f}] |"
          for r in df.itertuples()]
    L += ["", f"Visits that entered only through the item-CF fill-up: {only['fill']:,}.",
          f"Item-CF alone with 200 candidates: recall {cf200[0]:.4f} [{cf200[1]:.4f}, {cf200[2]:.4f}].", "",
          "\"Credited\" follows the order in which the union is built (item-CF first), so a visit found by two retrievers "
          "counts for the earlier one. \"Without it\" removes the retriever and fills the 200 from item-CF "
          "(from liked-history CF when item-CF is the one removed).", "", "Written by `src/37_retriever_ablation.py`."]
    (OUT / "RETRIEVERS.md").write_text("\n".join(L) + "\n")
    (ROOT / "web" / "public" / "data" / "retrievers.json").write_text(json.dumps(
        dict(users=len(cases), recall=allr[0], recall_lo=allr[1], recall_hi=allr[2], item_cf_200=cf200[0],
             rows=json.loads(df.round(5).to_json(orient="records"))), separators=(",", ":")))
    print("\n".join(L))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
