#!/usr/bin/env python3
"""Live profiles: the server's answer equals the offline pipeline, and a new profile behaves as it should.

Needs server/bundle (src/31_export_serving_bundle.py) and the exported app files (src/41_export_app.py).
  1. Sample tastes alone: for each of the 24, the server-built candidates, scores, collaborative part and six
     SHAP groups equal web/public/data/app/profile_XX.json.
  2. Sample tastes in pairs: the same over the union of two candidate sets, and the app engine
     (src/app_engine.py) lists the same six places from the server's answer as from the exported files.
  3. New profiles: a taste built from picked places returns 200 candidates without the picked places in under
     2 seconds (the plan's bar); bad requests are refused. SHAP parts must add up to the score: interventional
     TreeSHAP leaves a remainder on a few rows of this model (5 of 60,000 above 0.001 in step 39, largest 0.029),
     which the app shows as its own line, so the check allows at most 1% of rows above 0.001 and none above 0.1.
  4. Search: every result holds every word of the query; nothing a request returns names a reviewer.
Run: .venv/bin/python tests/check_serving.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

import app_engine as eng  # noqa: E402
from server import live  # noqa: E402

APP = ROOT / "web" / "public" / "data" / "app"
NUMBERS = ["raw", "collab"]
FILTERS = dict(cuisines=[], price_max=None, price_min=None, needs=[], borough=None, area=None, min_rating=None)
NO_TARGETS = {"km": None, "popularity": None, "cuisines": None}


def main():
    t0 = time.time()
    b = live.Bundle()
    print(f"bundle loaded in {time.time() - t0:.1f}s: {b.n:,} restaurants, {len(b.searchable):,} searchable")
    index = json.loads((APP / "index.json").read_text())
    places = json.loads((APP / "places.json").read_text())
    exported = {p["id"]: json.loads((APP / p["file"]).read_text()) for p in index["profiles"]}
    home = {p["id"]: {"lat": p["home_lat"], "lon": p["home_lon"]} for p in index["profiles"]}
    fails = 0

    def compare(pids, ans):
        """Largest difference between the server's numbers and the exported ones over the answer's places."""
        rows = [b.app_row[i] for i in ans["places"]["rid"]]       # every place of a sample party is an app place
        worst = 0.0
        for pid, m in zip(pids, ans["members"]):
            e = exported[pid]
            if [rows[j] for j in m["candidates"]] != e["candidates"]:
                return None
            for key in NUMBERS:
                worst = max(worst, float(np.abs(np.array(m[key]) - np.array(e[key])[rows]).max()))
            for g in index["groups"]:
                worst = max(worst, float(np.abs(np.array(m["groups"][g]) - np.array(e["groups"][g])[rows]).max()))
        for key in ("name", "lat", "lon", "borough", "nta", "nta_name", "cuisine", "price", "rating", "reviews", "recent_reviews", "ref"):
            if ans["places"][key] != [places[key][r] for r in rows]:
                return None
        for a, col in places["needs"].items():
            if ans["places"]["needs"][a] != [col[r] for r in rows]:
                return None
        return worst

    # 1. every sample taste alone
    worst = 0.0
    for pid in exported:
        ans = b.party([{"profile": pid}])
        d = compare([pid], ans)
        if d is None or d > 1e-6 or abs(ans["base"] - index["base"]) > 1e-9:
            fails += 1; print(f"FAIL {pid}: the server's answer differs from the exported profile ({d})")
        else:
            worst = max(worst, d)
    print(f"1. {len(exported)} sample tastes alone: largest difference from the exported files {worst:.1e}")

    # 2. pairs, and the engine on both sources
    rng = np.random.default_rng(5)
    ids = list(exported)
    worst, same = 0.0, 0
    pairs = [[str(x) for x in rng.choice(ids, size=2, replace=False)] for _ in range(8)]
    for pair in pairs:
        ans = b.party([{"profile": p} for p in pair])
        d = compare(pair, ans)
        if d is None or d > 1e-6:
            fails += 1; print(f"FAIL {pair}: pair differs from the exported profiles ({d})"); continue
        worst = max(worst, d)
        lists = []
        for table, profs in ((places, [exported[p] for p in pair]), (ans["places"], ans["members"])):
            members = [{"profile": q, "start": home[p]} for p, q in zip(pair, profs)]
            pool = eng.build_pool(table, index["cuisine_names"], members, FILTERS)
            pre = eng.prepare(table, index["dist_cap_km"], members, pool)
            res = eng.solve(members, pre, NO_TARGETS, 1.0, "average_without_misery")
            lists.append([table["name"][pool[j]] for j in res["top"]])
        if lists[0] != lists[1]:
            fails += 1; print(f"FAIL {pair}: the engine lists {lists[1]} from the server, {lists[0]} from the files")
        else:
            same += 1
    print(f"2. {len(pairs)} pairs of sample tastes: largest difference {worst:.1e}; the same six places in {same}")

    # 3. new profiles
    times, resid, off, rows_n = [], 0.0, 0, 0
    queries = [["pizza", "ramen", "taco"], ["shake shack", "sushi", "diner", "bagel", "thai"], ["steak", "dim sum", "burger", "cafe", "noodle", "bbq", "deli", "curry", "wine", "tapas"]]
    for words in queries:
        liked = []
        for w in words:
            hit = b.search(w)
            if not hit:
                fails += 1; print(f"FAIL search '{w}' found nothing"); continue
            liked.append(hit[0]["id"])
        t = time.time()
        ans = b.party([{"liked": liked}])
        times.append(time.time() - t)
        m, rid = ans["members"][0], ans["places"]["rid"]
        own = [rid[j] for j in m["candidates"]]
        total = ans["base"] + np.sum([m["groups"][g] for g in index["groups"]], axis=0)
        gap = np.abs(total - np.array(m["raw"]))
        resid = max(resid, float(gap.max())); off += int((gap > 1e-3).sum()); rows_n += len(gap)
        if len(own) != 200 or len(set(own)) != 200 or set(own) & set(liked) or m["history_size"] != len(set(liked)):
            fails += 1; print(f"FAIL new profile from {words}: candidates")
        if not all(b.available[i] for i in own):
            fails += 1; print(f"FAIL new profile from {words}: a candidate with no review in the year before the snapshot")
    t = time.time()
    two = b.party([{"liked": [b.search("pizza")[0]["id"], b.search("ramen")[0]["id"], b.search("taco")[0]["id"]]}, {"profile": ids[3]}])
    t_two = time.time() - t
    if len(two["members"]) != 2 or any(len(m["raw"]) != len(two["places"]["rid"]) for m in two["members"]):
        fails += 1; print("FAIL party of a new profile and a sample taste")
    if resid > 0.1 or off > 0.01 * rows_n:
        fails += 1; print(f"FAIL SHAP additivity for new profiles: {off} of {rows_n} rows above 0.001, largest {resid:.2e}")
    if max(times) > 2.0:
        fails += 1; print(f"FAIL a new profile took {max(times):.1f}s (the bar is 2s)")
    for bad in ([{"liked": []}], [{"liked": [-1, 2, 3]}], [{"liked": [b.n + 5]}], [{"profile": "profile_99"}], [], [{"liked": list(range(60))}]):
        try:
            b.party(bad)
            fails += 1; print(f"FAIL accepted a bad request: {str(bad)[:60]}")
        except ValueError:
            pass
    print(f"3. new profiles from {', '.join(str(len(q)) for q in queries)} picked places: {', '.join(f'{x:.2f}s' for x in times)} each; "
          f"with a sample taste {t_two:.2f}s; |base + six groups - raw| above 0.001 on {off} of {rows_n} rows (largest {resid:.1e}); 6 bad requests refused")

    # 4. search, and nothing private in an answer
    checked = 0
    for q in ("Joe's Pizza", "shake shack", "café", "XI'AN", "halal guys", "zz-no-such-place"):
        words = live.plain(q).split()
        for hit in b.search(q):
            checked += 1
            if not all(w in live.plain(hit["name"]) for w in words) or b.pop[hit["id"]] <= 0:
                fails += 1; print(f"FAIL search '{q}' returned {hit['name']}")
    if b.search("zz-no-such-place") or b.search("a"):
        fails += 1; print("FAIL search returned results for a query that should find nothing")
    text = json.dumps(two) + json.dumps(b.search("pizza"))
    for word in ("user_id", "seen", "liked", "reviewer", "gmap"):
        if f'"{word}' in text:
            fails += 1; print(f"FAIL an answer holds '{word}'")
    print(f"4. search: {checked} results hold every word of their query; answers hold no ids or visit lists")
    print(f"{fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
