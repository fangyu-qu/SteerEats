#!/usr/bin/env python3
"""Writes tests/group_expected.json: random group cases and what src/group.py returns for them.
tests/check_group.mjs then checks that web/src/group.js returns the same. Run with --write after changing group.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group import STRATEGIES, group_score, rank_pool, satisfaction

rng = np.random.default_rng(7)
cases = []
for c in range(120):
    m, n = int(rng.integers(2, 6)), int(rng.integers(1, 61))
    scores = rng.normal(size=(m, n))
    if c % 3 == 0:
        scores = np.round(scores, 1)                      # ties within a member
    if c % 10 == 0:
        scores[0] = 0.25                                  # an indifferent member
    ids = rng.permutation(10 * n)[:n]
    allowed = (rng.random(n) > 0.2) if c % 4 == 0 else None
    tau = float(rng.choice([0.3, 0.4, 0.5]))
    out = {}
    for how in ("minmax", "percentile"):
        sat = satisfaction(scores, how)
        for st in STRATEGIES:
            score, feasible = group_score(sat, st, tau, allowed)
            order = rank_pool(score, feasible, ids)[:10]
            out[f"{how}|{st}"] = dict(order=[int(i) for i in order], score=[float(v) for v in score],
                                      feasible=[bool(v) for v in feasible])
    cases.append(dict(scores=scores.tolist(), ids=[int(i) for i in ids], tau=tau,
                      allowed=None if allowed is None else [bool(v) for v in allowed], expected=out))
path = Path(__file__).with_name("group_expected.json")
if "--write" in sys.argv:
    path.write_text(json.dumps(cases, separators=(",", ":")))
    print(f"wrote {len(cases)} cases to {path.name}")
else:
    old = json.loads(path.read_text())
    assert json.dumps(old, separators=(",", ":")) == json.dumps(cases, separators=(",", ":")), "group_expected.json is stale: run with --write"
    print(f"group_expected.json matches src/group.py ({len(cases)} cases)")
