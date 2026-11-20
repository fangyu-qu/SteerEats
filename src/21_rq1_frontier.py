#!/usr/bin/env python3
"""Step 21: compare the models at matched outcomes rather than matched slider values.

Scores are min-max normalized per user before the penalties, so the same slider value moves the two
models' lists by different amounts. Here each model gets the best NDCG@10 it can reach anywhere on the
192-setting joint grid (step 20, 2,000 users) while meeting a target on the steered quantity, plus the
Pareto frontier of NDCG@10 against each quantity (for the trade-off view).
Outputs: outputs/rq1/matched_targets.csv, outputs/rq1/frontiers.csv, section appended to RQ1.md.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUTS

OUT = OUTPUTS / "rq1"
NAME = {"lr": "Logistic regression", "lgbm": "LightGBM"}
TARGETS = [  # (quantity, direction, threshold, description)
    ("distance_km", "max", 2.0, "mean distance at most 2 km"),
    ("distance_km", "max", 1.5, "mean distance at most 1.5 km"),
    ("distance_km", "max", 1.0, "mean distance at most 1 km"),
    ("popularity", "max", 0.40, "mean popularity percentile at most 0.40"),
    ("popularity", "max", 0.25, "mean popularity percentile at most 0.25"),
    ("cuisines", "min", 6.0, "at least 6 cuisine groups"),
    ("cuisines", "min", 8.0, "at least 8 cuisine groups"),
    ("neighbourhoods", "min", 7.0, "at least 7 neighbourhoods"),
    ("outside_home_borough", "min", 0.25, "at least 25% outside the home borough"),
]


def frontier(g, q, better_low):
    g = g.sort_values(q, ascending=better_low)
    best, keep = -1.0, []
    for _, r in g.iterrows():
        if r.ndcg > best:
            keep.append(r); best = r.ndcg
    return pd.DataFrame(keep)


def main():
    grid = pd.read_csv(OUT / "joint_grid.csv")
    rows = []
    for q, direction, thr, text in TARGETS:
        for m in ("lr", "lgbm"):
            g = grid[grid.model == m]
            ok = g[g[q] <= thr] if direction == "max" else g[g[q] >= thr]
            base = g[(g.w_h == 1) & (g.alpha == 0) & (g.beta == 0) & (g.lam == 0)].iloc[0]
            if ok.empty:
                rows.append(dict(target=text, model=NAME[m], reachable=False))
                continue
            b = ok.sort_values(["ndcg", "l1_change"], ascending=[False, True]).iloc[0]
            rows.append(dict(target=text, model=NAME[m], reachable=True, best_ndcg=b.ndcg, default_ndcg=base.ndcg,
                             ndcg_retained=b.ndcg / base.ndcg, w_h=b.w_h, alpha=b.alpha, beta=b.beta, lam=b.lam,
                             l1_change=b.l1_change, achieved=b[q]))
    mt = pd.DataFrame(rows)
    mt.to_csv(OUT / "matched_targets.csv", index=False)

    fr = []
    for q, low in (("distance_km", True), ("popularity", True), ("cuisines", False), ("neighbourhoods", False),
                   ("outside_home_borough", False)):
        for m in ("lr", "lgbm"):
            f = frontier(grid[grid.model == m], q, low)
            fr.append(f.assign(quantity=q, model_name=NAME[m])[["quantity", "model_name", q, "ndcg", "w_h", "alpha", "beta", "lam"]]
                      .rename(columns={q: "value"}))
    pd.concat(fr).to_csv(OUT / "frontiers.csv", index=False)

    md = ["", "## Matched outcomes (joint grid, 2,000 users; added by `src/21_rq1_frontier.py`)", "",
          "Best NDCG@10 each model reaches while meeting the target, and the share of its default NDCG@10 kept. "
          "Comparing at equal outcomes avoids reading slider values as equal amounts of steering.", "",
          "| Target | Model | Best NDCG@10 | Kept | Setting (w_h, alpha, beta, lam) |", "|---|---|---|---|---|"]
    for r in mt.itertuples():
        if not r.reachable:
            md.append(f"| {r.target} | {r.model} | not reachable on the grid | — | — |")
        else:
            md.append(f"| {r.target} | {r.model} | {r.best_ndcg:.4f} | {r.ndcg_retained:.0%} | "
                      f"({r.w_h:g}, {r.alpha:g}, {r.beta:g}, {r.lam:g}) |")
    text = (OUT / "RQ1.md").read_text()
    marker = "\n## Matched outcomes"
    text = text.split(marker)[0].rstrip("\n") + "\n" + "\n".join(md) + "\n"
    (OUT / "RQ1.md").write_text(text)
    print("\n".join(md))


if __name__ == "__main__":
    main()
