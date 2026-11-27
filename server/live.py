"""Live profiles: a taste built from places a person picks, scored like the offline pipeline (W3).

The bundle of src/31_export_serving_bundle.py stands in for ranker_core.Data and ranker_core.Snapshot at the
2020-08-01 snapshot, so the functions of ranker_core are used as they are: profile_of -> features -> candidates,
then the app's ranker (LambdaRank) and interventional SHAP on the step-17 background. For the 24 sample tastes
the result equals web/public/data/app/profile_XX.json (tests/check_serving.py).

A new profile is "these places, all of them liked", the same profile the cold-start study measured
(src/32_cold_start.py). Nothing a person picks is stored: a request is scored and forgotten.

`party(members)` answers for one or two members at once, because the app ranks the union of the members'
candidates and needs every member's score for every place in it.
"""
import json
import re
import sys
import threading
import unicodedata
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ranker_core import ATTR_FEATURES, FEATURE_GROUPS, FEATURES, candidates, features, profile_of  # noqa: E402
import steering  # noqa: E402

BUNDLE = ROOT / "server" / "bundle"
DECIMALS = 6                 # as in src/41_export_app.py
MIN_LIKED, MAX_LIKED = 1, 40
MAX_QUERY, MAX_RESULTS = 60, 12
GROUPS = list(FEATURE_GROUPS)
COLLAB = [FEATURES.index(f) for f in FEATURE_GROUPS["Collaborative"]]


def r(v, d=DECIMALS):
    return np.round(np.asarray(v, dtype=float), d).tolist()


def plain(text):
    """Lowercase letters and digits only, accents removed: "Café Habana!" -> "cafe habana"."""
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower().replace("&", " and ")
    s = re.sub(r"['’‘`]", "", s)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


class Bundle:
    """The arrays of one snapshot. It is passed to ranker_core both as the Data and as the Snapshot."""

    def __init__(self, path=BUNDLE):
        import lightgbm as lgb
        import shap
        a = np.load(path / "arrays.npz")
        meta = json.loads((path / "meta.json").read_text())
        if meta["features"] != FEATURES:
            raise ValueError("The bundle was made for another feature list: run src/31_export_serving_bundle.py")
        for key in a.files:
            setattr(self, key, a[key])
        self.n = len(self.lat)
        self.items = np.arange(self.n)                      # ranker_core.features only takes its length
        self.cuisine_names = meta["cuisine_names"]
        self.sim = sp.load_npz(path / "sim.npz").tocsr()
        self.meta = meta
        self.needs = meta["needs"]
        self.samples = json.loads((path / "profiles.json").read_text())
        self.app_row = {int(i): k for k, i in enumerate(meta["app_places"])}
        self.model = lgb.Booster(model_file=str(path / "lambdarank_model.txt"))
        bg = np.load(path / "background.npz")["x"]
        self.tree = shap.TreeExplainer(self.model, data=shap.maskers.Independent(bg, max_samples=len(bg)),
                                       feature_perturbation="interventional", model_output="raw")
        self.base = float(np.atleast_1d(self.tree.expected_value)[0])
        # search: restaurants with at least one review before the snapshot, most reviewed first
        known = np.flatnonzero(self.pop > 0)
        self.searchable = known[np.lexsort((known, -self.pop[known]))]
        self.plain = [plain(n) for n in meta["name"]]
        self.lock = threading.Lock()                        # LightGBM and SHAP objects are not shared between threads

    # ------------------------------------------------------------------------------------------ search
    def search(self, query, limit=MAX_RESULTS):
        """Restaurants whose name holds every word of the query; names that start with it first, then by reviews."""
        q = plain(query)[:MAX_QUERY]
        words = q.split()
        if len(q) < 2:
            return []
        first, rest = [], []
        for i in self.searchable:
            name = self.plain[i]
            if all(w in name for w in words):
                (first if name.startswith(q) else rest).append(int(i))
                if len(first) >= limit:
                    break
        return [self.card(i) for i in (first + rest)[:limit]]

    def card(self, i):
        m = self.meta
        return {"id": int(i), "name": m["name"][i], "area": m["nta_name"][i], "borough": m["borough"][i],
                "cuisine": m["cuisine_label"][i], "price": m["price"][i], "reviews": int(self.pop[i])}

    def known(self, ids):
        """Picked restaurant ids as an index array: in range, reviewed before the snapshot, without repeats."""
        out = []
        for i in ids:
            if not (isinstance(i, int) and 0 <= i < self.n and self.pop[i] > 0):
                raise ValueError("Unknown place")
            if i not in out:
                out.append(i)
        if not MIN_LIKED <= len(out) <= MAX_LIKED:
            raise ValueError(f"Pick between {MIN_LIKED} and {MAX_LIKED} places")
        return np.array(out, dtype=np.int64)

    # ------------------------------------------------------------------------------------------ scoring
    def profile(self, member):
        """ranker_core profile of a member: {"profile": "profile_07"} (a sample taste) or {"liked": [ids]}."""
        if member.get("profile") is not None:
            q = self.samples.get(member["profile"])
            if q is None:
                raise ValueError("Unknown sample taste")
            return profile_of(self, np.array(q["seen"], dtype=np.int64), np.array(q["liked"], dtype=np.int64))
        items = self.known(member.get("liked") or [])
        return profile_of(self, items, items)

    def party(self, members):
        """Places (the union of the members' candidates) and, per member, scores and SHAP parts over them.

        The shapes are those of web/public/data/app/places.json and profile_XX.json, so the browser's engine
        (web/src/app/engine.js) runs on a live party as it does on the exported files.
        """
        if not 1 <= len(members) <= 2:
            raise ValueError("A party has one or two members")
        m = self.meta
        with self.lock:
            built = []
            for member in members:
                p = self.profile(member)
                f, seen, cf = features(self, self, p)
                built.append((p, f, candidates(self, f, seen, cf)))
            pool = np.unique(np.concatenate([ids for _, _, ids in built]))     # restaurant indices, ascending
            where = {int(i): k for k, i in enumerate(pool)}
            out = []
            for p, f, ids in built:
                x = f[pool].astype(np.float64)
                raw = self.model.predict(x, raw_score=True)
                phi = self.tree.shap_values(x, check_additivity=False)
                own = raw[[where[int(i)] for i in ids]]
                cuisine = {self.cuisine_names[k]: int(c) for k, c in enumerate(p["cuisine"]) if c}
                out.append({
                    "candidates": [where[int(i)] for i in ids], "raw": r(raw), "collab": r(phi[:, COLLAB].sum(axis=1)),
                    "groups": {g: r(phi[:, [FEATURES.index(n) for n in FEATURE_GROUPS[g]]].sum(axis=1)) for g in GROUPS},
                    "shap_residual_max": float(np.abs(self.base + phi.sum(axis=1) - raw).max()),
                    "home_lat": round(p["lat"], 2), "home_lon": round(p["lon"], 2), "home_borough": m["boroughs"][p["boro"]],
                    "history_size": int(p["n"]), "cuisine_profile": cuisine,
                    "top": [m["name"][i] for i in ids[steering.order(own)[:3]]]})
        places = {
            "rid": [int(i) for i in pool], "name": [m["name"][i] for i in pool], "lat": r(self.lat[pool]), "lon": r(self.lon[pool]),
            "borough": [m["borough"][i] for i in pool], "nta": [int(v) for v in self.nta[pool]],
            "nta_name": [m["nta_name"][i] for i in pool], "cuisine": [int(v) for v in self.cuisine[pool]],
            "price": [m["price"][i] for i in pool], "is_chain": [m["chain"][i] for i in pool],
            "rating": r(self.mean[pool], 3), "reviews": self.pop[pool].astype(int).tolist(),
            "recent_reviews": self.recent_pop[pool].astype(int).tolist(),
            "needs": {a: [int(v) for v in self.attr[pool, ATTR_FEATURES.index(f"attr_{a}")]] for a in self.needs},
            # rating history and review themes exist for the places of the exported app only
            "ref": [m["app_ref"][self.app_row[int(i)]] if int(i) in self.app_row else None for i in pool],
        }
        return {"cutoff": m["cutoff"], "base": round(self.base, DECIMALS), "places": places, "members": out}
