# SteerEats

Course project for DS-GA 3001 Visualization for Machine Learning (NYU, Fall 2026, Prof. Claudio Silva).
Team 4: Fangyu Qu and Yila Cao. Course site: https://ctsilva.github.io/2026-VisML-CDS/

**What it is:** an app that helps friends pick a restaurant in New York together and shows each of them
why the pick is fair to them. Diners first; explanations one step in; model analysis in a separate Lab.

The working plan is `docs/PLAN.md` (v2, November 25, 2026). Read it before starting a task. The proposal
and project update are references, not contracts.

## Layout of the repository

- `src/` numbered pipeline scripts (Python 3.13, `.venv`); `src/steering.py` and `src/group.py` are the
  reference implementations of steering and group aggregation.
- `web/` the site: Vite, D3 v7. Two pages: `index.html` is the diner app (`web/src/app/`), `lab.html` the
  results dashboard and explorer (`web/src/main.js`). MapLibre GL is planned, not installed. `server/` the optional API: FastAPI (one-sentence steering in `nl.py`, live taste profiles in
  `live.py`; rooms are not built). It also serves `web/dist`, so one process can be the whole site. `tests/` parity and safety checks. `outputs/` reports written by scripts.

## Results and data

- Never fabricate or estimate a result. Every number in the app, README, slides or report comes from a
  script in this repository run on the real data. If data is missing, stop and say so.
- `data/` is git-ignored: raw and processed data, per-user files, private case files, model answers.
  `server/bundle/` and the rooms database are git-ignored too.
- Predictive features use only observations before the snapshot cutoff. Features from review text use
  text from before the **earliest training cutoff** (2018-07-01). `tests/check_time_safety.py` must pass.
- SHAP explains raw model scores, not ranks. Rank changes are accounted for separately and must replay
  with zero residual.
- The browser mirrors Python exactly: `web/src/steer.js` = `src/steering.py`, `web/src/group.js` =
  `src/group.py`, `web/src/app/engine.js` = `src/app_engine.py`. Change both or neither; the checks in `docs/PLAN.md` §8 stay at 0 failures.
- The live server does not re-implement features: `server/live.py` calls `src/ranker_core.py` on the bundle of
  `src/31_export_serving_bundle.py`. For the sample tastes its answer equals the exported app files
  (`tests/check_serving.py`, 0 failures). Three rankers have the keys `lr`, `lgbm`, `lmr` (LambdaRank) everywhere.

## People's data

- No accounts, passwords or real names. A room is a short code; its data lives at most 24 hours.
- Places a person picks as liked are scored by the server and forgotten: no log, no cache, never in a URL. The
  list stays in that browser. `server/bundle/profiles.json` holds the sample reviewers' visit lists: it stays
  on the server, is never returned, and is git-ignored.
- Location only after an explicit tap, rounded to about 100 m before leaving the device; a person can
  always set their start point by hand instead.
- Published files hold restaurant aggregates only: no user ids, no reviewer names, no review rows.
  Quotes shown from reviews are checked against the cited review and carry no names or contact details.

## LLM use

- API keys live in the server's environment, never in `web/`, the repo, or a URL.
- The LLM reads and labels; it does not rank and does not invent places. One-sentence steering returns
  targets that the app shows for confirmation and the solver applies.
- Print the cost (`--plan`) and get the owner's go-ahead before sending a batch.
- Load the `claude-api` skill before writing or changing an LLM call.

## Design

- Three layers: **App** (who, where, what, the pick), **Why** (reasons, ledger, history, who gives up
  what), **Lab** (analysis). A view goes in the shallowest layer where a diner needs it; nothing an
  analyst wants is on the first screen.
- Colours, type, radii and spacing only through tokens in `web/src/tokens.css` (imported by `styles.css` for the
  Lab and `app/app.css` for the app). Palette roles are in
  `docs/PLAN.md` §2.4: tomato for actions and the user's own steering; herb green for secondary actions
  and coloured text; salad, chips, pumpkin, prawn and tofu are fills and surfaces only; text is warm brown; cobalt blue
  `#1F5FBF` is for water on the map only.
- The three effect roles (model, control, variety) keep the same colour, glyph and texture in every
  view. They are never encoded by colour alone, because the palette cannot separate them for
  colour-blind readers. Run the `dataviz` validator on any new chart colours.
- Cuisine pictures and member avatars are watercolour stickers generated with Magic Patterns, in
  true-to-life food colours (`web/public/illustrations/`, one per cuisine group; prompt template and
  rules in `docs/PLAN.md` §2.4). They appear on recommendation cards and the place drawer only; on the
  map a restaurant is a tomato-red pin. Say in the report that they are AI-generated.
- Small interface marks (pins, chart glyphs, controls) are our own SVG, coloured through tokens. Food
  shapes replace plain marks only in charts of about 30 marks or fewer, centred on the value and equal
  in size. Every illustrated control is a real button with a text label.
- Design phones first for App screens, a wide projector for the Lab. Respect reduced motion.
- Use the installed skills: `impeccable` (shape, distill, layout, typeset, colorize, adapt, onboard,
  critique, audit, polish), `dataviz` for charts, `hallmark` / `design-taste-frontend` for new pages,
  `diagram-design` for figures.

## Working

- Branch from `main`; merge through pull requests; `main` deploys the site to GitHub Pages.
- New dependencies and anything that costs money need Yila's or Fangyu's OK first.
- Disclose AI assistance in the report.
