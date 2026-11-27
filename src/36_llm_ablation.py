#!/usr/bin/env python3
"""Step 36: do features read from review text improve the rankers? (RQ3)

Feature sets, all trained and evaluated with the two-stage protocol of step 15 (same windows, labels,
candidate sampling, grids, selection rule and test users):
  base   the 30 features of ranker_core
  bow    base + bag-of-words text features (src/34_text_sample.py)
  llm    base + LLM aspect features of the same text sample (added when data/clean/text_features_llm.npz exists)

Within one seed every set sees exactly the same training rows and the same test users; only the columns
differ, so differences are paired per user. The seed drives negative sampling and the selection users
(as in step 15); seed 42 with the base set repeats step 15.

Text features of a restaurant are static and come from text before TEXT_CUTOFF. The user-side affinity
uses only restaurants the user reviewed before the snapshot.

Run:  .venv/bin/python src/36_llm_ablation.py --seed 42      # one seed, every available set
      .venv/bin/python src/36_llm_ablation.py --report       # outputs/llm_features/ABLATION.md from all seeds
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, TEST_START, TEXT_CUTOFF, VAL_START
from models import SEED, fit_lambdarank, fit_lgbm, fit_logistic, logistic_scorer
from ranker_core import FEATURE_GROUPS, FEATURES, Data, Snapshot, candidates, eligible, evaluate, examples_two_stage

C_GRID = [0.1, 1.0, 10.0]
LEAVES_GRID = [15, 31, 63]
VAL_SELECTION_USERS = 3000
OUT = OUTPUTS / "llm_features"
MODELS = ["Logistic regression", "LightGBM", "LambdaRank"]
SET_LABEL = {"base": "Base (30 features)", "bow": "Base + bag-of-words text", "llm": "Base + LLM aspects"}


def load_block(name):
    z = np.load(CLEAN / f"text_features_{name}.npz", allow_pickle=False)
    return z["emb"].astype(np.float32), z["missing"].astype(bool), [str(n) for n in z["names"]]


class TextSnapshot(Snapshot):
    """A Snapshot whose feature matrix carries extra text columns after the 30 base features."""

    def __init__(self, data, cutoff, users, blocks):
        if pd.Timestamp(TEXT_CUTOFF) > pd.Timestamp(cutoff):
            raise ValueError("Text cutoff is later than the snapshot cutoff")
        super().__init__(data, cutoff, users)
        self.blocks = blocks

    def user(self, u):
        f, seen, cf = super().user(u)
        cols = [f]
        for emb, missing, _ in self.blocks:
            mine = emb[seen][~missing[seen]]
            prof = mine.mean(axis=0) if len(mine) else np.zeros(emb.shape[1], np.float32)
            norm = float(np.linalg.norm(prof))
            aff = emb @ (prof / norm) if norm > 0 else np.zeros(len(emb), np.float32)
            cols += [emb, missing[:, None], aff[:, None]]
        return np.column_stack(cols).astype(np.float32), seen, cf


def raw(b):
    return lambda f: b.predict(f, raw_score=True)


def run(args):
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    out = OUT / ("smoke" if args.smoke else f"seeds/seed_{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    sets = [s for s in args.sets.split(",") if s == "base" or (CLEAN / f"text_features_{s}.npz").exists()]
    blocks = [load_block(s) for s in sets if s != "base"]
    cols, names, at = {"base": list(range(len(FEATURES)))}, {"base": list(FEATURES)}, len(FEATURES)
    for s, (emb, _, nm) in zip([s for s in sets if s != "base"], blocks):
        width = emb.shape[1] + 2
        cols[s] = cols["base"] + list(range(at, at + width))
        names[s] = list(FEATURES) + nm
        at += width
    print(f"sets: {sets}; columns {at}", flush=True)

    data = Data()
    s = data.interactions
    users = np.sort(s.user_id.unique())
    if args.smoke:
        users = np.sort(rng.choice(users, size=args.smoke, replace=False))
        s = s[s.user_id.isin(users)]
    users = set(users)
    train, val, test = (s[s.split == k] for k in ("train", "val", "test"))
    cuts = sorted({train.time.quantile(q).to_period("M").to_timestamp() for q in (0.4, 0.6, 0.8)}
                  | {pd.Timestamp(VAL_START)})
    xs, ys, gs = [], [], []
    for start, end in zip(cuts[:-1], cuts[1:]):
        snap = TextSnapshot(data, start, users, blocks)
        x, y, g = examples_two_stage(snap, train[(train.time >= start) & (train.time < end)], rng, 10)
        xs.append(x); ys.append(y); gs.append(np.char.add(f"{start.date()}|", g.astype(str)))
        print(f"window {start.date()} -> {end.date()}: {len(y):,} rows ({time.time() - t0:.0f}s)", flush=True)
    x_tr, y_tr, g_tr = np.vstack(xs), np.concatenate(ys), np.concatenate(gs)
    vsnap = TextSnapshot(data, VAL_START, users, blocks)
    x_va, y_va, g_va = examples_two_stage(vsnap, val, rng, 10)
    sel_users = set(rng.choice(np.sort(val.user_id.unique()), size=min(VAL_SELECTION_USERS, val.user_id.nunique()),
                               replace=False))
    val_sel = val[val.user_id.isin(sel_users)]

    def on(c, fn):
        return lambda f: fn(f[:, c])

    fitted, scorers = {}, {}
    for name in sets:
        c, nm = cols[name], names[name]
        logit = {k: fit_logistic(x_tr[:, c], y_tr, k) for k in C_GRID}
        lgbm = {k: fit_lgbm(x_tr[:, c], y_tr, k, valid=(x_va[:, c], y_va), names=nm) for k in LEAVES_GRID}
        rank = {k: fit_lambdarank(x_tr[:, c], y_tr, g_tr, k, valid=(x_va[:, c], y_va, g_va), names=nm) for k in LEAVES_GRID}
        fitted[name] = (logit, lgbm, rank)
        scorers |= {f"{name}|Logistic C={k}": on(c, logistic_scorer(m)) for k, m in logit.items()}
        scorers |= {f"{name}|LightGBM leaves={k}": on(c, raw(b)) for k, b in lgbm.items()}
        scorers |= {f"{name}|LambdaRank leaves={k}": on(c, raw(b)) for k, b in rank.items()}
        print(f"fitted grid for {name} ({time.time() - t0:.0f}s)", flush=True)
    recs, _ = evaluate(vsnap, val_sel, scorers, two_stage=True, log_every=0)
    vm = pd.DataFrame(recs).groupby("model")[["recall", "ndcg"]].mean()

    def pick(prefix):
        return vm[vm.index.str.startswith(prefix)].sort_values(["ndcg", "recall"], ascending=False).index[0]

    x_all, y_all = np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va])
    g_all = np.concatenate([g_tr, np.char.add("val|", g_va.astype(str))])
    final, chosen, test_scorers = {}, [], {}
    for name in sets:
        c, nm = cols[name], names[name]
        logit, lgbm, rank = fitted[name]
        best_c = float(pick(f"{name}|Logistic").split("=")[1])
        best_l = int(pick(f"{name}|LightGBM").split("=")[1])
        best_r = int(pick(f"{name}|LambdaRank").split("=")[1])
        rounds, rrounds = int(lgbm[best_l].best_iteration), int(rank[best_r].best_iteration)
        lr = fit_logistic(x_all[:, c], y_all, best_c)
        gb = fit_lgbm(x_all[:, c], y_all, best_l, rounds=rounds, names=nm)
        rk = fit_lambdarank(x_all[:, c], y_all, g_all, best_r, rounds=rrounds, names=nm)
        final[name] = (lr, gb, rk)
        chosen.append(dict(set=name, logistic_C=best_c, lightgbm_leaves=best_l, lightgbm_rounds=rounds,
                           lambdarank_leaves=best_r, lambdarank_rounds=rrounds, features=len(c)))
        test_scorers |= {f"Logistic regression [{name}]": on(c, logistic_scorer(lr)),
                         f"LightGBM [{name}]": on(c, raw(gb)), f"LambdaRank [{name}]": on(c, raw(rk))}
    del fitted
    print(f"final models fitted; evaluating test ({time.time() - t0:.0f}s)", flush=True)
    tsnap = TextSnapshot(data, TEST_START, users, blocks)
    recs, audit = evaluate(tsnap, test, test_scorers, two_stage=True, log_every=0)
    per_user = pd.DataFrame(recs)
    metrics = per_user.groupby("model")[["recall", "ndcg"]].mean().rename(
        columns={"recall": "Recall@10", "ndcg": "NDCG@10"})
    metrics.insert(0, "users", per_user.groupby("model").user_id.nunique())
    print(metrics.to_string(), flush=True)
    metrics.reset_index().to_csv(out / "metrics.csv", index=False)
    per_user.to_csv(out / "test_per_user.csv", index=False)
    pd.DataFrame(chosen).to_csv(out / "selected.csv", index=False)

    # Share of each model's attribution that falls on the text columns (candidate rows of 400 test users).
    cases, _ = eligible(tsnap, test)
    pick_users = np.random.default_rng(SEED).choice(len(cases), size=min(400, len(cases)), replace=False)
    feats = []
    for j in pick_users:
        f, seen, cf = tsnap.user(cases[j][0])
        feats.append(f[candidates(tsnap, f, seen, cf)])
    xc = np.vstack(feats)
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    rows = []
    for name in sets:
        c, nm = cols[name], names[name]
        lr, gb, rk = final[name]
        scaler, lin = lr[0], lr[-1]
        contrib = {"Logistic regression": np.abs(((xc[:, c] - scaler.mean_) / scaler.scale_) * lin.coef_[0]).mean(axis=0),
                   "LightGBM": np.abs(gb.predict(xc[:, c], pred_contrib=True)[:, :-1]).mean(axis=0),
                   "LambdaRank": np.abs(rk.predict(xc[:, c], pred_contrib=True)[:, :-1]).mean(axis=0)}
        for model, v in contrib.items():
            share = pd.Series(v / v.sum(), index=nm).groupby(lambda f: group_of.get(f, "Text")).sum()
            rows += [dict(set=name, model=model, group=g, share=float(sh)) for g, sh in share.items()]
    pd.DataFrame(rows).to_csv(out / "attribution_share.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s; audit {audit}", flush=True)


def boot_ci(d, rng, n=2000):
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def report():
    seeds = sorted(int(p.name.split("_")[1]) for p in (OUT / "seeds").glob("seed_*") if (p / "test_per_user.csv").exists())
    if not seeds:
        raise SystemExit("No finished seeds in outputs/llm_features/seeds")
    rng = np.random.default_rng(SEED)
    per = {s: pd.read_csv(OUT / f"seeds/seed_{s}/test_per_user.csv") for s in seeds}
    sets = [s for s in ("base", "bow", "llm") if any(per[seeds[0]].model == f"LightGBM [{s}]")]
    L = ["# RQ3: do text features improve ranking?", "",
         f"Two-stage protocol of step 15, test period from {TEST_START}. Seeds: {', '.join(map(str, seeds))} "
         "(the seed drives negative sampling and the selection users). Within a seed every feature set uses the "
         "same training rows and the same test users, so differences are paired per user.",
         f"All text is from before {TEXT_CUTOFF} (see TEXT_SAMPLE.md).", ""]
    if "llm" not in sets:
        L += ["**The LLM aspect features are not in this table yet**: they need an API key and the owner's "
              "go-ahead on cost. This run covers the base features and the bag-of-words comparator.", ""]
    n_users = per[seeds[0]][per[seeds[0]].model == "LightGBM [base]"].user_id.nunique()
    L += [f"## NDCG@10 on test ({n_users:,} users), mean over seeds [min, max]", "",
          "| Model | " + " | ".join(SET_LABEL[s] for s in sets) + " |", "|---|" + "---|" * len(sets)]
    table = {}
    for m in MODELS:
        cells = []
        for s in sets:
            v = [per[sd][per[sd].model == f"{m} [{s}]"].ndcg.mean() for sd in seeds]
            table[(m, s)] = v
            cells.append(f"{np.mean(v):.4f} [{min(v):.4f}, {max(v):.4f}]")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## Recall@10 on test, mean over seeds", "",
          "| Model | " + " | ".join(SET_LABEL[s] for s in sets) + " |", "|---|" + "---|" * len(sets)]
    for m in MODELS:
        L.append(f"| {m} | " + " | ".join(
            f"{np.mean([per[sd][per[sd].model == f'{m} [{s}]'].recall.mean() for sd in seeds]):.4f}" for s in sets) + " |")
    rows = []
    pairs = [(s, "base") for s in sets[1:]] + ([("llm", "bow")] if {"llm", "bow"} <= set(sets) else [])
    for s, ref in pairs:
        L += ["", f"## {SET_LABEL[s]} minus {'base' if ref == 'base' else SET_LABEL[ref].lower()}: paired difference in NDCG@10", "",
              "Per seed: mean difference over test users with a 95% bootstrap interval (2,000 resamples of users).", "",
              "| Model | " + " | ".join(f"seed {sd}" for sd in seeds) + " | mean of seeds |", "|---|" + "---|" * (len(seeds) + 1)]
        for m in MODELS:
            cells, means = [], []
            for sd in seeds:
                p = per[sd]
                a = p[p.model == f"{m} [{s}]"].set_index("user_id").ndcg
                b = p[p.model == f"{m} [{ref}]"].set_index("user_id").ndcg
                d = (a - b.reindex(a.index)).to_numpy()
                lo, hi = boot_ci(d, rng)
                means.append(d.mean())
                cells.append(f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
                rows.append(dict(set=s, against=ref, model=m, seed=sd, diff=d.mean(), lo=lo, hi=hi, users=len(d)))
            L.append(f"| {m} | " + " | ".join(cells) + f" | {np.mean(means):+.4f} |")
    pd.DataFrame(rows).to_csv(OUT / "ablation_differences.csv", index=False)
    pd.DataFrame([dict(model=m, set=s, seed=sd, ndcg=v) for (m, s), vs in table.items() for sd, v in zip(seeds, vs)]
                 ).to_csv(OUT / "ablation_ndcg.csv", index=False)
    share = pd.concat([pd.read_csv(OUT / f"seeds/seed_{sd}/attribution_share.csv").assign(seed=sd) for sd in seeds])
    text = share[(share.group == "Text") & (share.set != "base")].groupby(["set", "model"]).share.mean()
    if len(text):
        L += ["", "## Share of attribution on the text columns", "",
              "Mean absolute contribution of the text columns as a share of all columns, on the candidate rows of "
              "400 test users, mean over seeds (logistic regression: standardized coefficient times value; "
              "tree models: path-dependent TreeSHAP of the raw score).", "", "| Set | Model | Share |", "|---|---|---|"]
        L += [f"| {SET_LABEL[s]} | {m} | {v:.1%} |" for (s, m), v in text.items()]
        share.groupby(["set", "model", "group"]).share.mean().reset_index().to_csv(OUT / "attribution_share.csv", index=False)
    sel = pd.concat([pd.read_csv(OUT / f"seeds/seed_{sd}/selected.csv").assign(seed=sd) for sd in seeds])
    L += ["", "## Selected settings", "", "| Seed | Set | Features | Logistic C | LightGBM leaves / rounds | LambdaRank leaves / rounds |",
          "|---|---|---|---|---|---|"]
    L += [f"| {r.seed} | {r.set} | {r.features} | {r.logistic_C:g} | {r.lightgbm_leaves} / {r.lightgbm_rounds} | "
          f"{r.lambdarank_leaves} / {r.lambdarank_rounds} |" for r in sel.itertuples()]
    web = dict(seeds=seeds, users=int(n_users), sets=[dict(id=s, label=SET_LABEL[s]) for s in sets], models=MODELS,
               llm_pending="llm" not in sets, text_cutoff=TEXT_CUTOFF,
               ndcg=[dict(model=m, set=s, seed=int(sd), ndcg=float(v)) for (m, s), vs in table.items() for sd, v in zip(seeds, vs)],
               differences=[{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in rows],
               text_share=[dict(set=s, model=m, share=float(v)) for (s, m), v in text.items()])
    import json
    from common import ROOT
    (ROOT / "web" / "public" / "data" / "text_ablation.json").write_text(json.dumps(web, separators=(",", ":"), default=int))
    L += ["", "Written by `src/36_llm_ablation.py --report`."]
    (OUT / "ABLATION.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sets", default="base,bow,llm")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a)
