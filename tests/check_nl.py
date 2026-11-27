#!/usr/bin/env python3
"""One-sentence steering: the browser (web/src/nl.js) and Python (server/nl.py, src/steering.py) agree.

1. The slot vocabularies are the same on both sides, and the cuisine keys are those of src/taxonomy.py and of
   the case files.
2. normalize() gives the same slots for the same raw answer (valid, invalid and hostile inputs).
3. Filters keep the same candidates, and steering on the filtered pool gives the same final ranks, the same
   top 10 and the same solved settings in the browser as in Python, for every demo case.
No model call, no cost.  Run: .venv/bin/python tests/check_nl.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

import steering  # noqa: E402
from server import nl  # noqa: E402

CASES = ROOT / "web" / "public" / "data" / "cases"


def node(request):
    r = subprocess.run(["node", str(ROOT / "tests" / "nl_node.mjs")], input=json.dumps(request), capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def main():
    fails = 0
    # 1. vocabularies
    js = node({"cmd": "constants"})
    py = dict(cuisines=nl.CUISINES, needs=nl.NEEDS, boroughs=nl.BOROUGHS, popularity=nl.POPULARITY, variety=nl.VARIETY,
              history=nl.HISTORY, km_range=list(nl.KM_RANGE), km_step=nl.KM_STEP, max_sentence=nl.MAX_SENTENCE)
    for k in py:
        a, b = py[k], js[k]
        same = sorted(a) == sorted(b) if isinstance(a, list) and a and isinstance(a[0], str) else a == b
        if not same:
            fails += 1
            print(f"FAIL vocabulary {k}: python {a} != browser {b}")
    files = sorted(CASES.glob("demo_*.json"))
    cases = {f.name: json.loads(f.read_text()) for f in files}
    for name, c in cases.items():
        if sorted(c["cuisine_names"]) != sorted(nl.CUISINES):
            fails += 1
            print(f"FAIL {name}: cuisine names differ from the taxonomy")
        if not set(c["candidates"]["borough"]) <= set(nl.BOROUGHS):
            fails += 1
            print(f"FAIL {name}: unknown borough {set(c['candidates']['borough']) - set(nl.BOROUGHS)}")
        if any(f"attr_{a}" not in c["feature_values"] for a in nl.NEEDS):
            fails += 1
            print(f"FAIL {name}: an attribute of NEEDS is missing")
    print(f"1. vocabularies: {len(py)} lists, {len(cases)} case files")

    # 2. normalize
    rng = np.random.default_rng(7)
    raw = [None, {}, [], "x", {"in_scope": False, "max_km": 2, "cuisines": ["pizza"]},
           {"in_scope": True, "max_km": 0, "popularity": "none", "variety": "none", "history": "none", "cuisines": [], "price_max": 0,
            "price_min": 0, "needs": [], "borough": "none", "area": "", "unsupported": []},
           {"max_km": 1.13, "cuisines": ["japanese", "x", "japanese", 3], "price_max": 2, "price_min": 3, "needs": ["halal", "halal", "wifi"],
            "borough": "Brooklyn", "area": "  East   Village!! <script>", "unsupported": ["quiet", "a" * 60, "", 5, "x", "y", "z"]},
           {"max_km": 0.1}, {"max_km": 99}, {"max_km": 0.375}, {"max_km": 0.625}, {"max_km": 2.125}, {"max_km": "2"}, {"max_km": -1},
           {"max_km": True}, {"price_max": 2.5}, {"price_max": 5}, {"price_max": True}, {"price_min": 4, "price_max": 4},
           {"borough": "brooklyn"}, {"area": 12}, {"area": "Bedford-Stuyvesant (East)"}, {"area": "Hell's Kitchen"},
           {"popularity": "hidden_gems", "variety": "lots", "history": "new"}, {"popularity": "__proto__"}, {"history": "toString"},
           {"cuisines": "pizza"}, {"needs": {"halal": 1}}, {"unsupported": "quiet"}, {"area": "under_score  café 9"}]
    for _ in range(300):
        raw.append({"in_scope": bool(rng.random() > 0.1), "max_km": float(rng.choice([0, rng.uniform(0, 9)])),
                    "popularity": str(rng.choice(["none"] + nl.POPULARITY)), "variety": str(rng.choice(["none"] + nl.VARIETY)),
                    "history": str(rng.choice(["none"] + nl.HISTORY)),
                    "cuisines": [str(x) for x in rng.choice(nl.CUISINES + ["nope"], size=int(rng.integers(0, 4)))],
                    "price_max": int(rng.integers(0, 6)), "price_min": int(rng.integers(0, 6)),
                    "needs": [str(x) for x in rng.choice(nl.NEEDS, size=int(rng.integers(0, 3)))],
                    "borough": str(rng.choice(["none"] + nl.BOROUGHS)), "area": str(rng.choice(["", "SoHo", "Park Slope"])),
                    "unsupported": [str(x) for x in rng.choice(["quiet", "no pizza", "open now"], size=int(rng.integers(0, 3)))]})
    got = node({"cmd": "normalize", "raw": raw})
    bad = [(r, a, b) for r, a, b in zip(raw, got, (nl.normalize(r) for r in raw)) if a != b]
    for r, a, b in bad[:5]:
        print(f"FAIL normalize {r!r}\n   browser {a}\n   python  {b}")
    fails += len(bad)
    print(f"2. normalize: {len(raw)} raw answers, {len(bad)} differ")

    # 3. filters and steering on the filtered pool
    filters = [dict(cuisines=["japanese"]), dict(cuisines=["american", "pizza"], price_max=2), dict(price_max=1), dict(price_min=3),
               dict(needs=["halal"]), dict(needs=["vegetarian_options", "late_night"], borough="Brooklyn"), dict(borough="Manhattan", price_max=2),
               dict(area="village"), dict(area="harlem"), dict(cuisines=["korean"], area="nowhere at all"), dict(cuisines=["mexican"], needs=["alcohol"])]
    thetas = [dict(steering.THETA0), dict(steering.THETA0, alpha=0.5), dict(steering.THETA0, w_h=0.5, beta=0.3, lam=0.25),
              dict(steering.THETA0, w_h=0, alpha=1, beta=1, lam=0.5)]
    targets = [dict(km=2, popularity=None, cuisines=None), dict(km=None, popularity=0.4, cuisines=6), dict(km=1, popularity=0.2, cuisines=8)]
    full = lambda f: dict(dict(cuisines=[], price_max=None, price_min=None, needs=[], borough=None, area=None), **f)
    jobs = [dict(case=name, filters=full(f), thetas=thetas, targets=targets) for name in cases for f in filters]
    got = node({"cmd": "pool", "jobs": jobs})
    lists = pools = small = 0
    for job, g in zip(jobs, got):
        c = cases[job["case"]]
        keep = nl.pool_mask(c, job["filters"])
        index = [i for i, k in enumerate(keep) if k]
        tag = f"{job['case']} {json.dumps({k: v for k, v in job['filters'].items() if v not in (None, [])})}"
        if index != g["index"]:
            fails += 1
            print(f"FAIL pool {tag}: python keeps {len(index)}, browser {len(g['index'])}")
            continue
        if not index:
            continue
        pools += 1
        small += len(index) < 10
        sub = nl.filter_case(c, keep)
        runs = iter(g["runs"])
        for model in ("lr", "lgbm"):
            for theta in thetas:
                r = next(runs)
                ok = np.array_equal(steering.final_ranks(sub, theta, model), r["ranks"]) and np.array_equal(steering.top_k(sub, theta, model), r["top"])
                lists += 1
                if not ok:
                    fails += 1
                    print(f"FAIL ranks {tag} {model} {theta}")
            for t in targets:
                r = next(runs)
                s = steering.solve_targets(sub, model, t)
                ok = all(abs(s["theta"][k] - r["theta"][k]) < 1e-9 for k in ("w_h", "alpha", "beta", "lam")) and np.array_equal(s["top"], r["top"])
                lists += 1
                if not ok:
                    fails += 1
                    print(f"FAIL solver {tag} {model} {t}: python {s['theta']} browser {r['theta']}")
    print(f"3. filters: {len(jobs)} case-and-filter pairs, {pools} with a non-empty pool ({small} with fewer than 10 places), {lists} lists compared")
    print(f"{fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
