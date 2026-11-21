# Handoff: Fangyu (and Fangyu's AI assistant)

Read with `CLAUDE.md` (rules) and `docs/PLAN.md` (design). Everything below is current as of 2026-11-21.

## State

All of Yila's work is on branch `Yila`: pipeline `src/10`–`28`, views V1–V7 and the similarity map (S1),
case studies (`outputs/case_studies/`), the user-study kit (`docs/USER_STUDY.md`), the Pages workflow.
`main` is behind; a pull request from `Yila` into `main` brings it in (open one if none exists).

## Fangyu's tasks (the report can wait)

1. **Restaurant detail view (V8).** For the selected restaurant: rolling mean rating with an uncertainty
   band, review count, price (missing shown as missing), and an LLM theme summary with quoted excerpts.
   - Data: a new script `src/29_export_restaurant_detail.py` writing aggregates for the candidates of the 24
     demo cases (`web/public/data/cases/demo_*.json`, key `candidates.gmap_id`) to
     `web/public/data/detail/`. Use only reviews before the case's snapshot (2020-08-01) for anything the
     model could see; label later reviews as later. No per-user rows, no reviewer names or ids.
   - View: `web/src/views/detail.js`, mounted from `web/src/main.js`; refresh it in `update()` and in
     `select(i)` (candidate index `i`, same as the ledger). Colors only via tokens in `web/src/styles.css`;
     do not use the three effect hues (`--model`, `--control`, `--diversity`) for anything else.
   - Decide: which snapshots V8 shows over time.
2. **Publish the site.** As repository admin:
   1. Settings > Pages > Build and deployment > Source: **GitHub Actions**.
   2. Review and merge the pull request from `Yila` into `main`. `.github/workflows/deploy-pages.yml` then runs
      the checks, builds and publishes to https://fangyu-qu.github.io/SteerEats/. Later: Actions >
      Deploy to GitHub Pages > Run workflow. Fallback: Netlify import (reads `netlify.toml`).

## Setup on a new machine

```bash
git clone https://github.com/fangyu-qu/SteerEats && cd SteerEats && git checkout Yila
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd web && npm install && npm run dev        # http://localhost:5173; needs no local data
```

Python steps need `data/raw` and `data/processed` (git-ignored) in `data/`. Run `src/10`–`12` once to make
`data/clean/`; trained models are committed, so `src/13`–`28` are only needed after changing data or models
(commands and runtimes in `README.md`).

## Before every push

```bash
node tests/replay.mjs && node tests/check_ledger.mjs && node tests/check_lens.mjs && node tests/check_tradeoff.mjs && node tests/check_preview.mjs
.venv/bin/python tests/check_steering.py
cd web && npm run build
```

## Rules for an AI assistant working here

- Never type a number into the UI, README or docs that a script in this repo did not produce on the real data.
- Features use only observations before the snapshot cutoff. SHAP explains raw model scores, not ranks.
- Browser steering (`web/src/steer.js`) mirrors `src/steering.py` exactly; change both or neither, and keep
  the checks above at 0 failures.
- Work on a branch, open a pull request into `main`; end commits with the AI co-author line you use.
