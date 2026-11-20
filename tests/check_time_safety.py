#!/usr/bin/env python3
"""Checks that the ranker features cannot see the future and that evaluation is well formed.

1. Truncation test: a Snapshot built from all reviews and one built from reviews truncated at the
   cutoff must give identical feature matrices. Any difference means a feature used later data.
2. Eligibility: every relevant item is available at the cutoff and unseen by the user.
3. No snapshot-only metadata (Google rating, review count, open state) among the features.
Run: .venv/bin/python tests/check_time_safety.py
"""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import VAL_START
from ranker_core import FEATURES, Data, Snapshot, eligible


def main(n_users=300, n_check=60, seed=0):
    rng = np.random.default_rng(seed)
    data = Data()
    s = data.interactions
    users = set(rng.choice(np.sort(s.user_id.unique()), size=n_users, replace=False))

    full = Snapshot(data, VAL_START, users)
    cut = copy.copy(data)
    cut.reviews = data.reviews[data.reviews.time < full.cutoff].reset_index(drop=True)
    trunc = Snapshot(cut, VAL_START, users)
    checked = 0
    for u in sorted(full.profile)[:n_check]:
        fa, sa, _ = full.user(u)
        fb, sb, _ = trunc.user(u)
        assert np.array_equal(sa, sb), f"seen items differ for {u}"
        assert np.allclose(fa, fb, atol=1e-6), f"features differ for {u}: " + ", ".join(
            FEATURES[j] for j in np.flatnonzero(~np.isclose(fa, fb, atol=1e-6).all(axis=0)))
        checked += 1
    print(f"truncation test passed for {checked} users ({len(FEATURES)} features)")

    val = s[(s.split == "val") & s.user_id.isin(users)]
    cases, audit = eligible(full, val)
    for u, rel in cases:
        seen = set(full.profile[u]["seen"].tolist())
        assert not (rel & seen), "relevant item already seen"
        assert all(full.available[i] for i in rel), "relevant item not available"
        assert all(full.d.reviews.time[full.d.reviews.item == i].min() < full.cutoff for i in list(rel)[:3])
    print(f"eligibility test passed for {len(cases)} users; audit {audit}")

    banned = [f for f in FEATURES if any(w in f for w in ("google", "avg_rating", "num_reviews", "state", "inspection"))]
    assert not banned, f"snapshot-only fields used as features: {banned}"
    print("no snapshot-only metadata among features")


if __name__ == "__main__":
    main()
