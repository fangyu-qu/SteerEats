#!/usr/bin/env python3
"""Step 24: gather the evaluation results for the interface's evaluation panel (view V7).

Reads (does not recompute):
  outputs/rankers_two_stage/test_intervals.csv   mean Recall@10 / NDCG@10 with 95% user-bootstrap intervals
  outputs/rankers_two_stage/paired_tests.csv     paired differences, intervals, Wilcoxon p
  outputs/rankers_two_stage/robustness.csv       test periods and validation windows (train-only models)
  outputs/rankers_two_stage/{,seeds/seed_*/}metrics.csv   test NDCG@10 per sampling seed
  outputs/calibration/calibration.json           step 23
Writes web/public/data/evaluation.json (aggregates only; no per-user rows).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS, ROOT, TEST_START

R = OUTPUTS / "rankers_two_stage"
WEB = ROOT / "web" / "public" / "data" / "evaluation.json"
PERIODS = [
    ("validation pre-COVID (Dec 2019-Feb 2020), train-only models", "Dec–Feb", "validation"),
    ("validation COVID (Mar-Jul 2020), train-only models", "Mar–Jul", "validation"),
    ("test Aug-Oct 2020", "Aug–Oct", "test"),
    ("test Nov 2020-Jan 2021", "Nov–Jan", "test"),
    ("test Feb-Apr 2021", "Feb–Apr", "test"),
]


def main():
    ci = pd.read_csv(R / "test_intervals.csv")
    rankers = {}
    for r in ci.itertuples():
        rankers.setdefault(r.model, {})[r.metric] = dict(mean=r.mean, lo=r.lo, hi=r.hi)
    users = int(ci.users.iloc[0])
    paired = pd.read_csv(R / "paired_tests.csv").to_dict("records")
    rob = pd.read_csv(R / "robustness.csv")
    periods = []
    for key, label, split in PERIODS:
        sub = rob[rob.variant == key]
        periods.append(dict(label=label, split=split, users=int(sub.users.iloc[0]),
                            ndcg={m: float(v) for m, v in zip(sub.model, sub["NDCG@10"]) if pd.notna(v)},
                            recall={m: float(v) for m, v in zip(sub.model, sub["Recall@10"]) if pd.notna(v)}))
    runs = {42: R / "metrics.csv"} | {int(p.parent.name.split("_")[1]): p for p in sorted((R / "seeds").glob("seed_*/metrics.csv"))}
    seeds = {}
    for k, p in sorted(runs.items()):
        d = pd.read_csv(p)
        for m, v in zip(d[d.period == "test"].model, d[d.period == "test"]["NDCG@10"]):
            if pd.notna(v):
                seeds.setdefault(m, {})[str(k)] = float(v)
    cal = json.loads((OUTPUTS / "calibration" / "calibration.json").read_text())
    out = dict(cutoff=TEST_START, users=users, rankers=rankers, paired=paired, periods=periods, seeds=seeds, calibration=cal)
    WEB.write_text(json.dumps(out, indent=1))
    print(f"wrote {WEB.relative_to(ROOT)}: {len(rankers)} rankers, {len(paired)} paired tests, {len(periods)} periods, "
          f"seeds {sorted(runs)}")
    for m, v in seeds.items():
        print(f"  {m}: {v}")


if __name__ == "__main__":
    main()
