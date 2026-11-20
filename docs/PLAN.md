# SteerEats: working plan (Yila's branch)

Written November 18, 2026. Final presentations are **December 1 and 8**; the 8-page report is due
**December 14**. This plan covers the work Yila owns (logistic regression, LightGBM, the explanation
pipeline, and the rank-change, SHAP, control, diversity and map views) and the joint work
(integration, case studies, evaluation, slides, report).

---

## 0. Where we stand

| Item | Status | Evidence |
|---|---|---|
| NYC filtering, sample, time split | Done (Fangyu) | `src/01`–`04`, README |
| Inspection matching | Done (Fangyu) | `src/03`, `outputs/03_*` |
| Popularity and item-CF baselines | Done | `outputs/logistic_ranker/metrics.csv` |
| Logistic ranker, 10 historical features | Done | test Recall@10 0.0484, NDCG@10 0.0357 |
| Ratings over time | Done (Fangyu) | `outputs/06_rating_over_time.csv` |
| LightGBM under the same protocol | **Not in the repo** (planned Nov 3–9) | — |
| Rank decomposition / explanation pipeline | **Not in the repo** (planned Nov 3–9) | — |
| Interface | **Not in the repo** (planned Nov 10–16) | — |
| Significance of the LR vs item-CF gain | Not tested (update says so) | — |

We are about two weeks behind the update's timeline, with 13 days to the first presentation slot.
The plan below is ordered so that **a working, honest demo exists by November 30** and everything
after that improves evidence and polish rather than adding risk.

---

## 1. What the professor grades, and how we aim above it

From the course site (syllabus, schedule, project-discussion slides):

| Component | Weight in project | What the slides ask for | How we exceed it |
|---|---|---|---|
| Technical quality | 60% | Sound approach, appropriate evaluation, depth, innovation beyond baseline | Two rankers under one protocol, bootstrap CIs and a paired test, an exact-replay check with zero residual, an order-independent (Shapley) attribution of rank change across pipeline stages |
| Presentation | 15% | Clear slides, demo/visualization quality, Q&A | Live, client-side demo (no server latency); one scripted case per research question |
| Write-up | 20% | Motivation, contributions, related work, effective figures, discussion | Paper-grade figures exported from the same code as the demo; a design-rationale section tied to perception and color principles |
| Teamwork | 5% | Division of labor documented | Contribution table in the report; commits on named branches |

Course-wide emphasis to keep in view:

- **Start from a question whose answer could come out either way.** Ours: *Does a linked,
  stage-wise before-and-after view explain ranking changes that aggregate metrics (Recall,
  NDCG, diversity) cannot?* and *Do logistic regression and LightGBM trade relevance for distance
  and diversity differently under the same controls?*
- **Visualization must serve an ML question.** The slide's in-scope examples include "how two
  models differ" and "whether two explanation methods agree"; both are central here.
- **Evaluation:** a quantitative metric, a baseline comparison, *or* a 3–5 person user study.
  We plan all three, kept small.
- **Reproduce, then extend:** reproduce TasteWeights' linked input–model–output layout on our
  data, then add the evaluated extension (entered/exited items and stage-wise attribution).
- **Modest and well-evaluated beats ambitious and broken.** Group recommendation, steering cost
  and UMAP stay stretch goals.
- **Disclose AI use** in the report (syllabus policy).

---

## 2. System design

### 2.1 Why the demo runs in the browser

Every control in the update (history weight, distance, popularity, diversity) acts *after* the
learned score. So for each case user we can precompute, offline in Python:

1. ~200 candidates (union of LR top-k, LightGBM top-k, item-CF top-k and recent-popular),
2. both models' raw scores and per-feature SHAP values for every candidate,
3. the distance and popularity terms and the cuisine/neighborhood similarity matrix for MMR.

The browser then recomputes `final = s − α·distance − β·popularity` and the MMR re-rank in a few
milliseconds per slider move. Benefits: instant feedback (the core of a steering interface), no
server to keep alive during the presentation, and a static site we can deploy to GitHub Pages or
Netlify. FastAPI stays optional (only needed if we later add the shared-room stretch goal).

```
data/processed/*.parquet          (Fangyu's pipeline, local only)
        │
src/07_logistic_ranker.py         existing
src/08_lightgbm_ranker.py         new: same snapshots, features, negatives, metrics
src/09_evaluate.py                new: bootstrap CIs, paired test, overlap, calibration
src/explain/                      new: candidates, stage log, SHAP, replay check
src/10_export_cases.py            new: one JSON per case user  →  web/public/data/cases/
        │
web/                              Vite + D3 v7 static app (all steering computed client-side)
```

### 2.2 Decisions to settle on day one

1. **Score scale.** LR and LightGBM raw scores (log-odds) are on different scales, so
   `α·distance` would mean different things per model. Normalize each model's score per user over
   its candidate set to [0, 1] (min–max on the logit, or percentile rank) before applying controls.
   Record the choice in the report.
2. **History weight `w_h`.** For LightGBM it cannot be applied by scaling one coefficient.
   Proposed definition for both models: `s(w_h) = w_h · s_full + (1 − w_h) · s_context`, where
   `s_context` is the same model class trained without the two CF features. It is model-agnostic
   and easy to explain ("how much your own history counts").
3. **Distance.** Haversine km from the user's training-period visit medoid (more robust than the
   centroid when a user reviews in two boroughs), normalized by a fixed cap (e.g. 15 km → 1).
4. **MMR.** `λ` is the diversity strength, with `λ = 0` meaning no re-ranking. Similarity
   between two restaurants = 0.5·cuisine match + 0.5·same NTA (cuisine from DOHMH where matched,
   Google categories otherwise).
5. **Time cutoff control.** Precompute 3 snapshots per case user (e.g. 2019-06-01, 2019-12-01,
   2020-08-01) rather than arbitrary dates.
6. **Deploy target.** GitHub Pages on `fangyu-qu/SteerEats` needs Fangyu to enable Pages; Netlify
   from the repo is the fallback.

### 2.3 Explanation accounting (the method contribution)

For one user and one control setting, log ranks over the *same* candidate set at four stages:

| Stage | Score |
|---|---|
| S0 model | normalized ranker score (with `w_h`) |
| S1 distance | S0 − α·distance |
| S2 popularity | S1 − β·popularity |
| S3 diversity | MMR(S2, λ) |

- Per item, `Δrank = Σ stage differences` telescopes exactly, so **replay residual = 0** is a
  test we run in CI on every case file, not a hope.
- Stage-wise attribution is order-dependent (the update says so). **Extension beyond the
  update:** with only three post-model stages there are 3! = 6 orderings, so we can compute the
  exact **Shapley value of each control's contribution to an item's rank change** cheaply and show
  both the fixed-order and the order-independent view. This answers the reviewer's
  order-dependence concern directly.
- SHAP explains S0 (the raw model score), never final ranks. The UI labels this.
- For LR, check SHAP against the closed form `β_j · (x_j − E[x_j])` (should match to float error).
  For LightGBM use `TreeExplainer` (exact for trees).

---

## 3. The views

Each view answers one question. One visual language across all of them: model effects, control
effects and diversity effects always use the same three hues everywhere.

| # | View | Question it answers | Encoding and interaction | Course link |
|---|---|---|---|---|
| V1 | **Steering console** | What can I change, and what will it cost? | Four sliders, a ghost tick at the starting value θ₀, and a small precomputed sparkline under each slider showing how relevance, distance and diversity move across its range | Wk 2 perception; TasteWeights |
| V2 | **Rank flow** | Which restaurants entered, left or moved, and at which stage? | Bump chart across S0→S1→S2→S3 (and before→after); exited items stay visible as hollow, faded lines; hover links to V3 and V4 | Novel (central claim) |
| V3 | **Score ledger** | Why is this restaurant here? | Horizontal waterfall: SHAP feature bars (model hue), then distance and popularity terms (control hue), then the MMR rank effect (diversity hue); toggle fixed-order vs Shapley-over-stages | Wk 5 SHAP; Wk 4 white-box (LR) |
| V4 | **Map** | Where did the list move geographically? Which neighborhoods can I reach, at what cost? | d3-geo NTA polygons for NYC, user medoid, distance rings, numbered top-10 markers, exited items hollow; NTA fill shows steering cost (smallest control change that brings the neighborhood into the top 10, hatched if unreachable). **Decided 2026-11-20:** clicking a neighborhood boosts it directly (a fifth, map-based control), because RQ1 showed the four indirect controls reach only 33–41% of other-borough neighborhoods | Spatial context; RQ1 steering cost |
| V5 | **Trade-off space** | What do I give up when steering? | Scatter of relevance vs mean distance from the sweep, colored by diversity, Pareto frontier drawn, current θ marked; click a point to load its θ | Wk 3 model assessment |
| V6 | **Model lens: LR vs LightGBM** | Do the two models rank and explain differently? | Side-by-side SHAP beeswarms; per-user top-10 overlap and Kendall τ distributions; attribution-agreement scatter | Wk 5; "how two models differ" |
| V7 | **Evaluation panel** | Is the learned ranker actually better? | Dot plot of Recall@10 / NDCG@10 with 95% bootstrap CIs for all five rankers; reliability diagram after Platt scaling | Wk 3 calibration |
| V8 | **Restaurant detail** (Fangyu) | What do the stars mean, and did they change? | Rolling mean rating with uncertainty band, review count, price (missing shown as missing), LLM theme summary with cited excerpts | Wk 11 time series; Wk 9 NLP |
| S1 | Stretch: **similarity map** | Is the restaurant space structured by cuisine and place? | UMAP of item-CF or SVD item factors, trustworthiness and continuity reported, linked brushing to V4 | Wk 7 DR |
| S2 | Stretch: **steering cost** | Which neighborhoods are reachable, at what L1 change in θ? | NTA choropleth of `C_m(n)` per model, ∞ shown as hatched | Proposal §4.2 |

Layout: one page with two densities. The default "diner" layout shows V1, V2, V4 and a one-line
plain-language summary ("3 places entered because you lowered the distance limit; 2 Brooklyn
places left"). An "analyst" toggle reveals V3, V5, V6, V7. This keeps the update's single-app scope
while preserving the proposal's user/researcher split.

### 3.1 Design system (so it does not look like a template)

- **Color:** tokens defined in OKLCH; three semantic hues for model / control / diversity; a
  sequential scale for the map; a diverging scale for rank change (up vs down). Every palette
  checked for color-vision deficiency (Sim Daltonism, or a script) and contrast (WCAG 2.1 AA).
- **Type:** one text face with tabular figures for ranks and scores; restaurant names in the body
  face, never in small caps.
- **Motion:** rank changes animate along the bump-chart paths (object constancy), 250–400 ms,
  with a `prefers-reduced-motion` fallback.
- **Skills installed for this:** `impeccable` (dashboard / Operate-mode critique and polish),
  `hallmark` and `design-taste-frontend` (landing/explainer page and anti-template checks),
  `diagram-design` (pipeline and architecture figures for the slides and report). The `dataviz`
  skill covers chart color and mark specs.

---

## 4. Evaluation

| RQ | Evidence | Planned output |
|---|---|---|
| Ranker quality | Recall@10, NDCG@10 on the 8,185 test users; 1,000-sample user-level bootstrap CIs; paired permutation (or Wilcoxon) test LR vs item-CF and LightGBM vs LR | Table 1, V7 |
| Calibration | Brier score, ECE (15 bins), reliability diagram, after Platt scaling on validation | Fig., V7 |
| RQ1 trade-offs | Sweep each control over a grid for all case users; report top-10 overlap with θ₀, mean distance, cuisine and NTA diversity (unique count, intra-list distance), catalog coverage, popularity, per model | Fig. (small multiples), V5 |
| Model difference | Per-user overlap@10 and Kendall τ between LR and LightGBM; SHAP rank agreement | V6 |
| Explanation fidelity | Replay residual = 0 for every case file; LR SHAP equals closed form | Automated test |
| Usefulness | 3–5 participant think-aloud study, 3 tasks (e.g. "Why did X drop?", "Get one Queens option without losing more than 2 of your current top 10", "Which model relies more on popularity?"); measure task success and time; compare with a list-only baseline (TasteWeights-like current list) | Report §Evaluation; follow NYU participant-review rules |
| Case studies | 3 scripted users: a Manhattan-centric user steered outward, a user whose list is popularity-dominated, a user where LR and LightGBM disagree | Slides and report |

Never quote a number that a script did not produce in this repository.

---

## 5. Timeline (from today)

| Dates | Yila | Fangyu (per update) | Joint |
|---|---|---|---|
| **Wed 11/18 – Thu 11/19** | Get `data/processed/*.parquet` from Fangyu; refactor shared code out of `07` into `src/ranker_core.py`; train LightGBM (`08`) with identical snapshots, features, negatives; validation-tuned | Data dictionary | Settle the six decisions in §2.2 |
| **Fri 11/20** | `09_evaluate.py`: bootstrap CIs, paired tests, Platt + reliability; SHAP for both models | Rating-history data export | — |
| **Sat 11/21 – Sun 11/22** | `src/explain/`: candidates, stage log, Shapley-over-stages, replay test; `10_export_cases.py` and JSON schema | LLM theme summaries for case restaurants | Pick case users |
| **Mon 11/23 – Wed 11/25** | `web/`: Vite + D3 skeleton, design tokens, V1, V2, V4 working on one case file | V8 | Daily merge to `main` via PR |
| **Thu 11/26 – Fri 11/27** (Thanksgiving) | V3, V5 (light days) | — | — |
| **Sat 11/28 – Mon 11/30** | V6, V7; design critique and polish pass; deploy | Integrate V8 | Slides, 10-minute script, dry run |
| **Tue 12/1** | Presentation slot I (if assigned) | | |
| **Wed 12/2 – Mon 12/7** | User study (3–5), case studies, stretch S1/S2 if time | Coverage views | Fix issues found |
| **Tue 12/8** | Presentation slot II (if assigned) | | |
| **Tue 12/8 – Mon 12/14** | Methods, explanation and evaluation sections; figures | Data and rating sections | Intro, related work, discussion, AI-use and contribution statements; **submit 12/14** |

Ask the instructor which day Team 4 presents; if it is December 1, the polish pass moves into the
Thanksgiving days.

---

## 6. Start here (first session)

1. Open the folder in VS Code; install the recommended extensions when prompted.
2. Select the interpreter `.venv/bin/python` (Python 3.13; `lightgbm`, `shap`, `umap-learn`,
   `fastapi` are installed on top of `requirements.txt`).
3. Copy Fangyu's `data/processed/` (four parquet files) and `data/raw/nyc_nta_2020.geojson` into
   `data/`. They are git-ignored. Alternatively, put the raw files in `~/Downloads` and run
   `scripts/download_data.py --from-directory ~/Downloads` then `src/01`–`04`.
4. Reproduce the logistic result: `bash scripts/run_first_ranker.sh` and check that
   `outputs/logistic_ranker/metrics.csv` matches the committed one.
5. Start on `08_lightgbm_ranker.py`.
6. Before each view, send the matching lecture slides so the view uses the course's vocabulary:
   Week 3 (model assessment, calibration) before V7; Week 5 (PDP, LIME, SHAP) before V3/V6;
   Week 2 (perception, color) before the design tokens; Week 7 (DR) before S1; Week 11 (time
   series) before V8; Week 8 lab (paper writing) before the report.

## 7. Git workflow

- Work on `Yila`; open small pull requests into `main` (one per script or view) so Fangyu can
  review, as the update promises ("Both review code").
- Data, per-user CSVs and private case files stay out of git (`.gitignore`). Only an anonymized
  `demo_*.json` case file is committed so the deployed demo works.
- Commit messages say what changed and which number it produced.

## 8. Risks

| Risk | Mitigation |
|---|---|
| LightGBM does not beat LR | That is a valid result; the comparison is about *how* they rank, not which wins |
| Controls barely change the top 10 | Report it; widen ranges; it is itself an RQ1 finding |
| Map clutter with 200 candidates | Show only top 10 + exited items; candidates on demand |
| User data privacy in the demo | Hash user IDs; show only restaurant data and aggregates |
| Presentation on December 1 | Freeze features November 29 |

## 9. Open questions

- For the instructor: presentation day for Team 4; final report template (the slides only say
  "conference paper format", 8 pages); whether a live demo link is expected.
- For Fangyu: enabling GitHub Pages; the inspection cuisine field for MMR; the time-cutoff
  snapshots she wants exposed in V8.
