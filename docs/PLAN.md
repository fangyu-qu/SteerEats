# SteerEats: working plan v2 (the app redesign)

Written November 25, 2026 on branch `app-redesign`. It replaces the plan of November 18 (in git history),
whose pipeline and views are built and merged. Final presentations are **December 1 and 8** (our slot is
still unknown); the 8-page report is due **December 14**.

The proposal (`../proposal/Team4.pdf`) and the project update (`../project update/Team4.pdf`) are
references, not contracts. This plan goes back to two things the proposal had and the update dropped:
group recommendation (RQ2) and a shared room on phones.

---

## 0. Where we stand

| Built and checked (on `main`) | Evidence |
|---|---|
| Data: 3 sources joined, cleaned, time split, 30 time-safe features | `src/01`–`12`, `outputs/cleaning/` |
| Rankers: 3 baselines, LR, LightGBM, LambdaRank; two-stage protocol | test NDCG@10 0.0481 / 0.0482 / 0.0508, `outputs/rankers_two_stage/` |
| Explanations: SHAP (LR, LightGBM), steering pipeline, exact rank accounting | `src/17`–`19`, `src/steering.py`, 0 replay mismatches |
| RQ1: sweep, matched outcomes, frontiers, steering cost, calibration | `outputs/rq1/`, `outputs/calibration/` |
| Interface: V1–V8, similarity map, case studies, user-study kit, Pages workflow | `web/`, 7 browser checks at 0 failures |
| LLM theme summaries with verified quotes (Fangyu) | 2,804 restaurants, `outputs/theme_summaries/` |

**Added on `app-redesign` (November 25, evening), all from scripts run on the real data:**

| Piece | Where | State |
|---|---|---|
| RQ2 group engine and offline evaluation (1,000 groups per kind, three models) | `src/group.py`, `src/33_rq2_groups.py`, `outputs/rq2/RQ2.md` | run |
| Cold-start study (k = 1, 3, 5, 10 liked places) | `src/32_cold_start.py`, `outputs/cold_start/COLD_START.md` | run |
| Retriever contribution (W6) | `src/37_retriever_ablation.py`, `outputs/retrievers/RETRIEVERS.md` | run |
| RQ3 text sample and bag-of-words comparator, three seeds | `src/34_text_sample.py`, `src/36_llm_ablation.py`, `outputs/llm_features/ABLATION.md` | run |
| RQ3 LLM aspect features (18,085 restaurants, `claude-haiku-5-5`, one message batch, $2.76 at list price) | `src/35_llm_aspects.py`, `outputs/llm_features/ABLATION.md` | run: base, bag-of-words and LLM compared over three seeds |
| Dashboard: Overview (one chart and one sentence per question), Explorer, Analysis; swatch palette | `web/index.html`, `web/src/views/overview.js`, `web/src/styles.css` | built, 7 browser checks at 0 failures |
| Time-safety test covers the text cutoff | `tests/check_time_safety.py` | passes |
| W5 one-sentence steering in the Explorer: say-it box, chips to confirm, targets to the solver, filters (cuisine, price, needs, borough, neighbourhood) as a narrower candidate pool | `server/nl.py`, `server/app.py` (`/api/parse`), `web/src/nl.js`, `web/src/views/sayit.js` | built; `tests/check_nl.py` at 0 failures (3,136 filtered lists, browser = Python); keyword baseline scored, model run on the 64 sentences waits for the go-ahead (`src/40_nl_eval.py --plan`: about $0.02) |

**Added in the night of November 25 to 26:**

| Piece | Where | State |
|---|---|---|
| One-sentence evaluation: 64 sentences, language model against keyword rules | `src/40_nl_eval.py`, `outputs/nl_steering/NL_EVAL.md`, Lab Overview panel 7 | run: 56 of 64 against 45 of 64 read completely right; $0.018 |
| App data scored with LambdaRank (the best ranker on test) for the 24 sample tastes over the union of their candidates, with SHAP per feature group | `src/41_export_app.py`, `web/public/data/app/` | run |
| Diner app, option C: start screen, party bar, say-it box, quick filters, six cards with member satisfaction, map with pins, place drawer (reasons, numbers, stars over time, review themes, compare), "How is this ranked?", "About the data" | `web/index.html`, `web/src/app/` | built for one or two people on one device |
| App engine in the browser = Python | `web/src/app/engine.js`, `src/app_engine.py`, `tests/check_app.py` | 0 failures |
| Pages: the app is `index.html`; the dashboard moved to `lab.html` (old `#case=` and `#study=` links are forwarded) | `web/vite.config.js` | built |

**Added on November 26 (day):**

| Piece | Where | State |
|---|---|---|
| W3 live profiles: a taste built from places a person picks, scored with `ranker_core` on a saved snapshot | `src/31_export_serving_bundle.py`, `server/live.py`, `server/app.py` (`/api/search`, `/api/party`) | built; `tests/check_serving.py` at 0 failures: for the 24 sample tastes the server's answer equals the exported files (difference 0); a new taste answers in 0.6 to 1.1 seconds, a party of two in 2.5 to 3.8 (the plan's bar of 2 seconds is met for one person, not for two) |
| Onboarding in the app: "Places I like" (search by name, pick at least 3), next to the sample tastes; picks stay in the browser | `web/src/app/main.js`, `web/src/nlClient.js` | built; needs the server, the site says so when it is not there |
| One process for the whole site: the API also serves `web/dist` | `server/app.py` | built; `http://127.0.0.1:8787/` |
| W2 LambdaRank as the third model of the Lab: case files, replay, sweep, frontiers, calibration, trade-off, model lens for any pair, evaluation view | `src/19`–`25`, `web/src/views/`, `tests/` | run; every earlier number of logistic regression and LightGBM is unchanged (the result files only gained rows) |

Not started: rooms across phones, avatars for a third to fifth person, the MapLibre street map (the app draws
neighbourhood shapes with D3), picking cuisines instead of places in the onboarding, hosting the server
somewhere public (it runs on this computer; see the README).

What is wrong with it (Yila, November 25): the page is one long list of analyst views with no clear
goal; it reads as an analysis tool, not as something a person would use to choose dinner.

---

## 1. Direction

**SteerEats helps friends pick a restaurant in New York together, and shows each of them why the pick
is fair to them.** It is an app for diners first. The explanation and the model analysis are still
there, one step further in, for whoever asks "why?".

Three layers, each one step deeper than the last (progressive disclosure):

| Layer | For | Holds |
|---|---|---|
| **App** | a diner, alone or in a group, on a phone or laptop | who is eating, where from, what they want; cards and map; the pick |
| **Why** | a diner who asks "why this place?" or "why did it change?" | reasons in plain words, the ledger, rating history, review themes, who gives up what |
| **Lab** | the course audience, analysts, us at Q&A | trade-off space, model comparison, similarity map, evaluation, RQ2 and LLM results |

Why this still fits the course (technical quality 60%, presentation 15%, write-up 20%, teamwork 5%;
"visualization must serve an ML question"; "modest and well-evaluated beats ambitious and broken"):
every new feature below ships with a script-run evaluation, and each view answers one ML question.

Research questions:

- **RQ1 (done):** how do the controls trade relevance for distance, popularity and variety, and do the
  models trade differently?
- **RQ2 (new, from the proposal):** how do group strategies (average, least misery, average without
  misery) change member satisfaction, the satisfaction gap, travel burden and feasibility?
- **RQ3 (new):** do features an LLM reads out of review text improve ranking beyond collaborative
  signals, and beyond plain bag-of-words text features? The answer can come out either way.
- **Supporting evaluations:** how many liked places a new user must pick before recommendations are
  useful (cold start), and how accurately one sentence is turned into steering targets.

---

## 2. Design brief

### 2.1 Job, audience, success

- **Who:** two to five friends deciding where to eat tonight, each on their own phone, often in
  different places; also one person alone. Second audience: the class, on a projector.
- **Primary task:** agree on one place quickly. Success: a group reaches a pick everyone can accept
  and each person can say why it suits them.
- **Product truth:** recommendations come from a 2020 snapshot of public reviews (data ends 2021).
  The app says so; it is a research prototype, not a live directory.

### 2.2 What we take from OpenTable's search page (studied November 25)

| Pattern there | What we do |
|---|---|
| One context bar on top (date, time, party size) | One party bar: who is eating (avatars), where each starts from, solo or group |
| One row of quick-filter chips with icons; "All filters" opens a panel with collapsed groups | Chips for Closer, Less mainstream, More variety, Price, Dietary; each opens a small popover with its slider and preview; everything else in one panel |
| List on the left, map on the right; hovering a card raises its pin, clicking a pin shows a mini card | Same split and the same linking; the list is the top 10, the map shows start points and travel |
| Card: picture, name, one line of facts, one line of proof, actions | Card: cuisine illustration, name, "rating · price · cuisine · neighbourhood", one plain reason, group satisfaction marks |
| A quiet "How are results ranked?" link beside the count | Our door into the Why layer: opens "How the list changed" |
| One accent colour, used only for actions and pins | Tomato red for actions, selection and the user's own steering; nothing else competes |
| Phone: list first, a floating "Map" button swaps to the map | Same |

### 2.3 Screens

1. **Start:** name and food avatar; "eating alone" or "with friends" (create or join a room by code or
   QR); taste: pick places you like, or pick cuisines, or borrow a sample taste profile; where from:
   use my location, drop a pin, or choose a neighbourhood.
2. **Discover:** party bar, a "say it" box (one-sentence steering), filter chips, cards and map.
3. **Place (drawer over Discover):** illustration and facts; three reasons in plain words; in a group,
   each member's satisfaction and travel; rating history and themes (V8); "show the full ledger";
   "compare with another place".
4. **Room (group):** lobby with members and their status; each sets preferences privately on their own
   device; results with a strategy switch ("fair to everyone", "best on average", "nobody unhappy")
   and a view of who gains and who gives up what.
5. **Lab:** the analyst views, the RQ2 and RQ3 results, the case-study deep links, study mode.

States that must be designed, not left to chance: first run, no location permission, a member who has
not finished, no place satisfies everyone (say which constraint blocks, offer to relax it), server
unreachable (fall back to sample profiles, say so), loading, phone and projector widths.

### 2.4 Colour

Yila's swatches, with roles set by measurement (contrast on the cream page background):

| Token | Colour | Contrast | Role |
|---|---|---|---|
| `--tomato` | `#E52113` | 4.36 | Primary action, selection, pins, brand; the user's own steering in charts |
| `--stew` | `#D94841` | 4.01 | Hover and pressed states of tomato |
| `--herb` | `#556B2F` | 5.63 | Secondary action, links, small coloured text |
| `--salad` | `#93AC4E` | 2.41 | Fills only (parks, positive fills) |
| `--chips` `#FFCC64`, `--pumpkin` `#F4B860` | | 1.4–1.7 | Fills only (highlights, rating marks with a dark outline) |
| `--prawn` `#EFD9CD`, `--tofu` `#F1F0BA` | | 1.1–1.3 | Surfaces: cards, bands, map land |
| `--cream` | about `#FBF9EC` | | Page background |
| `--ink` | about `#4A3526` | 10.9 | Text (warm brown, not black) |

Findings that constrain the design:

- **Palette only (Yila, November 25):** every colour in the interface comes from the swatches above.
  The only additions are the two neutrals (cream paper, warm-brown ink) and one map colour: cobalt
  blue `#1F5FBF`, used for water on the map and nowhere else. No purple or grey. This rule covers the
  interface itself (surfaces, text, controls, charts, map); the food stickers and avatars are pictures
  and use true-to-life food colours (Yila, November 25).
- Tomato red and herb green are almost the same to a red-blind reader (OKLab difference 3.9; 8 is the
  target), so that pair never separates two kinds of data.
- Tomato, salad and chips stay apart for red-, green- and blue-blind readers (smallest difference
  10.4), but salad and chips are too light to stand on the page alone (contrast 2.4 and 1.4). So every
  chart mark is a palette fill **with a brown ink outline**.
- The three effect roles are **never colour alone**. Each has a colour, a small food glyph and a fill
  texture, and a legend is always present:

| Effect role | Fill | Glyph | Texture |
|---|---|---|---|
| Model (what the ranker saw) | salad `#93AC4E` | chef's hat | solid |
| Control (your steering) | tomato `#E52113` | fork | diagonal hatch |
| Variety (diversity re-ranking) | chips `#FFCC64` | mixed plate | dots |

- **Look (option C "Printed map" in the mock-ups, after Yila's reference; awaiting her final yes):** flat colour blocks without outlines or
  shadows on cards, a paper-grain layer over the whole interface, caps headings with an underline, hand
  lettering for map labels and reasons. Herb green is the dark colour (header band, dark buttons); the
  map is pumpkin land with tofu streets, salad parks and blue water; restaurants on the map are
  tomato-red pins, and the food stickers appear only on the recommendation cards.
- **Cuisine stickers (decided by Yila, November 25):** one watercolour die-cut sticker per cuisine
  group, 25 in all, generated with Magic Patterns' image tool from one prompt template and one style
  reference, in true-to-life food colours. Files: `web/public/illustrations/cuisine/<cuisine_group>.jpg`
  (1024 px, cream background). Each shows a dish chosen from what the group contains
  (`outputs/cleaning/cuisine_category_map.csv`): American cheeseburger and fries, Chinese soup
  dumplings, Pizza, Mexican tacos, Cafe latte and strawberry cake, General plated dinner, Italian
  spaghetti and meatballs, Latin American empanadas, Japanese sushi, Middle Eastern kebab with hummus
  and pita, European roast chicken leg with potatoes, Vegetarian salad bowl, Southeast Asian pho,
  Caribbean jerk chicken with rice and peas, Fast food fried chicken bucket, South Asian curry with
  rice and naan, Sandwich and deli pastrami on rye, Seafood prawns and mussels, Asian (other) takeout
  noodles, Bar beer and cocktail, Korean bibimbap, Jewish and kosher bagel with lox, African jollof
  rice with plantain and suya, Breakfast pancakes, Venue party platter.
- **Where they go:** on the restaurant recommendation cards and the place drawer only. On the map a
  restaurant is a tomato-red pin, never a sticker.
- **Avatars:** Tomato and Tofu are made the same way (`web/public/illustrations/avatars/`); the other
  three (prawn, chips, salad) are still to be made.
- The stickers are AI-generated images, not our drawings. The report says so.

Final values are fixed in milestone M1 with the `dataviz` palette validator, on the real backgrounds.

### 2.5 Illustration, type, motion, map

- **Illustrations:** the 25 cuisine stickers and the five member avatars (tofu, prawn, chips, salad,
  tomato) are watercolour stickers generated with Magic Patterns (§2.4). Small interface marks (map
  pins, slider thumbs, rating marks, chart glyphs, empty and loading states) stay our own SVG,
  coloured through tokens.
- **Food shapes in charts, with rules:** only where a chart has about 30 marks or fewer; the value sits
  at the glyph's centre; all glyphs have the same optical size; dense plots (beeswarm, scatter of 192
  settings, similarity map) keep plain dots. Every illustrated control is a real button with a text
  label.
- **Type:** one rounded display face for names and headings, one plain face for text and numbers
  (tabular figures). Chosen from two mock-ups, not from this document.
- **Motion:** existing rank-flow motion stays; add small, short reactions (a pin grows when its card is
  hovered, a member's avatar nods when they are ready). All of it off under reduced motion.
- **Map:** MapLibre GL with vector tiles, styled from the tokens (tofu land, salad parks, soft water,
  rounded labels), so street names appear as you zoom in. Candidate tile source: OpenFreeMap (no key);
  checked in a first-day spike, with Protomaps as the alternative. The neighbourhood steering-cost
  shading and the distance rings become layers on it. Without a network the current D3 map is the
  fallback.
- **Restaurant pictures:** the cuisine stickers, on cards only. The dataset's reviewer photo links (2.2% of sampled
  reviews) are not used: they belong to their uploaders and can disappear.

### 2.6 Anti-goals

No second dashboard on the first screen. No chart on the first screen that needs a legend to be read.
No colour-only encoding. No real names, accounts or passwords for the course version.

---

## 3. Architecture

```
            static site (GitHub Pages)                    optional API (FastAPI, one process)
┌────────────────────────────────────────────┐      ┌──────────────────────────────────────────────┐
│ web/  Vite + D3 v7 + MapLibre              │      │ server/                                      │
│  app screens, Why layer, Lab               │ ───► │  /api/case     profile -> case JSON          │
│  steer.js, group.js: all steering and      │      │  /api/rooms    shared rooms (SQLite, polling) │
│  aggregation in the browser                │ ◄─── │  /api/parse    one sentence -> targets (LLM)  │
│  24 demo cases (sample taste profiles)     │      │  /api/search   restaurant names (onboarding)  │
└────────────────────────────────────────────┘      └──────────────────────────────────────────────┘
        works alone, without the API                   reads server/bundle/ (built by src/31, no user rows)
```

- **Case builder.** The API turns any profile (liked places, start location, price habit) into a case
  file with the same schema as the demo cases: 200 candidates, three models' scores and SHAP values,
  steering inputs. Everything the browser already does (steering, accounting, views) then works for a
  new user unchanged, and the JS ↔ Python parity checks keep their meaning.
- **Serving bundle** (`src/31_export_serving_bundle.py` → `server/bundle/`, git-ignored): restaurant
  features at the 2020-08-01 snapshot, the top-50 item similarity matrix, retriever inputs, the three
  models, the SHAP background, Platt coefficients. Aggregates only; no user ids or review rows.
- **Rooms.** A room is a short code. Each member's device sends its nickname, avatar, taste profile,
  preferences and start point; the server stores them in SQLite for at most 24 hours and returns the
  room state on polling. Aggregation runs in every browser from the same state, so all phones show the
  same pick. No accounts, no passwords.
- **Location.** GPS only after the person taps "use my location"; otherwise a dropped pin or a chosen
  neighbourhood. Coordinates are rounded to about 100 m before they leave the device. Two people in
  the same room can therefore choose different start points for a demo.
- **LLM.** The API key stays on the server (environment variable), never in the site or the repo.
  Without the API the site falls back to a keyword parser and sample profiles, and says so.
- **Demo hosting.** The API runs on a laptop behind an HTTPS tunnel (phones need HTTPS for location).
  The static site alone remains a complete fallback.

New dependencies, each needing Yila's OK before install: `maplibre-gl` (npm); `fastapi`, `uvicorn`
(pip); a tunnel tool for the demo; a QR-code generator (npm, small).

---

## 4. Workstreams

Each lists what is built, where, and the check that says it is done.

### W1. Design system and app shell

- `PRODUCT.md` and `DESIGN.md` (impeccable `init`, then `document` at the end).
- Two clickable mock-ups of Discover + Place (artifact) → Yila picks → tokens, type, components.
- New layout and routes: `#/` app, `#/place`, `#/room/CODE`, `#/lab`; existing views moved, not rewritten.
- Illustration set; MapLibre map with our style; cards; chips with popovers; drawer.
- Charts re-coloured through tokens, effect roles double-encoded, validated with `dataviz`.
- **Done when:** all existing browser checks pass; phone (375 px) and projector (1440 px) screenshots
  reviewed; impeccable `critique` and `audit` findings fixed; no colour pair fails the validator
  without a second encoding.

### W2. LambdaRank as the app's ranker

- SHAP and additivity check for LambdaRank (`src/17`); third model in case files (`src/19`); sweep,
  frontier, lens, calibration and exports (`src/20`–`25`); steering parity for three models.
- Its raw scores are not log-odds: report Platt calibration quality before using "predicted visits"
  for it; if poor, show measured NDCG only for that model.
- App uses LambdaRank; Lab compares any two of the three models.
- **Done when:** `replay`, `check_ledger`, `check_lens`, `check_tradeoff`, `check_preview` pass with
  three models; evaluation view shows all three with intervals.

### W3. Live profiles (new users)

- `src/31_export_serving_bundle.py`, `server/live.py`, `server/app.py` (`/api/party`, `/api/search`, `/api/health`).
- Onboarding: search and pick liked places, or pick cuisines; start location by GPS, pin or neighbourhood.
- `src/32_cold_start.py`: for test users, keep only their k = 3, 5, 10 most recent liked places as
  history and report NDCG@10 against the full-history result, with intervals. This sets the minimum
  number of picks the onboarding asks for and is shown in the Lab.
- **Done when:** for the demo users (local data), the server-built case equals the exported case
  (`tests/check_serving.py`); a new profile returns a case in under 2 seconds.

### W4. Group dining (RQ2)

- **Engine** (`src/group.py`, mirrored by `web/src/group.js`):
  - Pool: union of members' 200 candidates, minus hard vetoes (price ceiling, dietary flags; a place
    is out if any member vetoes it).
  - Each member steers their own scores with their own targets and their own start point; the steered
    score, min-max over the pool, is that member's **satisfaction** in [0, 1].
  - Group score: average, least misery, or average without misery (τ = 0.40, fixed in advance;
    sensitivity at 0.30 and 0.50, and with rank-percentile satisfaction instead of min-max).
  - Variety is a property of the list, so it is one group setting (the largest any member asks for),
    applied by MMR after aggregation.
  - Travel burden per member: km from their start point; maximum and gap are reported.
- **Views:** G1 per-place member satisfaction and travel (avatars on a shared 0–1 line, never a bare
  number); G2 "who gains, who gives up" across strategies (the rank flow with strategies as columns);
  G3 group map (start points, travel lines, the pick); G4 fairness summary.
- **Rooms:** `server/rooms.py`, lobby, join by code or QR, per-device preferences, live status.
- **Offline evaluation** (`src/33_rq2_groups.py`): three-member groups of test users with similar,
  random and diverse tastes (fixed seed); per strategy and model: minimum satisfaction, satisfaction
  gap, maximum travel, travel gap, feasible-candidate count, share of groups with ten feasible places,
  and each member's NDCG@10 on the group list (mean and worst), with paired bootstrap intervals.
- **Done when:** `tests/check_group.mjs` shows browser = Python for every strategy; two phones in one
  room reach the same pick; `outputs/rq2/RQ2.md` exists.

### W5. LLM work

- **One-sentence steering.** `/api/parse` (model `claude-haiku-5-5`, schema-constrained output) turns
  "somewhere quiet, not too pricey, close to both of us" into targets and filters. The app shows what
  it understood as removable chips before anything changes; the solver then does the steering, so the
  LLM never ranks or invents a place. `tests/nl_cases.json` (64 sentences with
  expected targets, including unclear, out-of-scope and Chinese ones) and `src/40_nl_eval.py` report accuracy
  per slot against a keyword-rule baseline.
- **LLM features (RQ3).** `src/34_llm_aspects.py`: for each restaurant, a sample of text reviews from
  **before 2018-07-01** (the earliest training cutoff, so the same text is safe for every training,
  validation and test window) → structured aspects (food, service, value, ambience, wait, noise) and
  occasion tags, as a message batch. `src/36_llm_ablation.py`: retrain the three rankers with and
  without these features, and with a bag-of-words baseline from the same text; same protocol, three
  seeds, paired bootstrap on test. Cost is printed by `--plan` before anything is sent (the earlier
  pilot measured about $2 per 1,000 restaurants for Haiku and $19 for Sonnet in single calls);
  sending needs Yila's go-ahead and an API key.
- Lab view L1: NDCG@10 with intervals for the three feature sets, and the SHAP share of the new group.
- **Done when:** `tests/check_time_safety.py` covers the text cutoff; `outputs/llm_features/ABLATION.md`
  and `outputs/nl_steering/NL_EVAL.md` exist; the app works with the API off.

### W6. Evaluation and study

- Fix the user-study tasks (both sets currently share their answers) and rewrite them for the new
  app: conditions become app-with-Why versus app-without-Why; add one group task.
- Refresh the three case studies for LambdaRank; add a group case study.
- Missing evidence from the first build: literature references for the features, and a retriever
  ablation (each retriever's share of stage-1 recall).

### W7. Presentation and report

- Ten-minute script: the problem, a two-phone group demo, one "why" walk-through, RQ1–RQ3 in the Lab.
- A recorded backup of the demo. Report sections updated; AI assistance and contributions stated.

---

## 5. Where every view goes

| View | Now | In the redesign |
|---|---|---|
| V1 Console | left rail, always open | Filter chips with popovers (slider + preview); "say it" box above them |
| V2 Rank flow | first panel | Why layer, behind "How is this ranked?"; reused as G2 with strategies as columns |
| V3 Ledger | panel | Place drawer: three plain reasons first, full ledger on request |
| V4 Map | panel | Right half of Discover, on real streets; start points and travel added |
| V8 Rating history and themes | panel | Place drawer |
| V5 Trade-off, V6 Model lens, S1 Similarity, V7 Evaluation | "For analysts" on the same page | Lab page |
| G1–G4 group views | — | Room results and Place drawer |
| L1 LLM features, cold-start curve, RQ2 results | — | Lab page |

---

## 6. Schedule

Today is Wednesday, November 25. The order is the same for either slot; only the cut line moves.

| Milestone | Content | Slot Dec 8 | Slot Dec 1 |
|---|---|---|---|
| M0 | Plan agreed (Yila, Fangyu); mock-ups chosen; presentation day asked; dependencies and LLM cost approved | Wed 11/25 | Wed 11/25 |
| M1 | W1 shell: tokens, layout, cards, map, illustrations; W2 pipeline runs in the background; LLM batch sent | Thu 11/26 – Sat 11/28 | Thu 11/26 – Fri 11/27 |
| M2 | W2 finished in the interface; W3 live profiles and location | Sun 11/29 – Mon 11/30 | Sat 11/28 |
| M3 | W4 group engine, views, rooms; RQ2 run | Tue 12/1 – Thu 12/3 | Sun 11/29 |
| M4 | W5 one-sentence steering; RQ3 ablation in the Lab | Fri 12/4 | Mon 11/30 (morning) |
| M5 | Critique, audit, polish; two-phone dry run; recorded backup; slides | Sat 12/5 – Mon 12/7, freeze Sun 12/6 | Mon 11/30, freeze that night |
| M6 | User study (3–5 people), report | to Mon 12/14 | Wed 12/2 – Mon 12/14 |

**If the slot is December 1 the list above does not all fit safely.** Cut in this order, top first:
(1) Lab restyle beyond tokens; (2) cold-start study; (3) RQ3 view (results go on a slide);
(4) live rooms (demo two members on one screen instead); (5) one-sentence steering falls back to the
keyword parser. Never cut: passing checks, and numbers that come only from scripts.

---

## 7. Skills

| Skill | Used for | When |
|---|---|---|
| `impeccable shape` | this brief | M0 |
| `artifact-design` | two clickable mock-ups to choose from | M0 |
| `impeccable` new-work, `colorize`, `typeset`, `layout` | the new visual system | M1 |
| `impeccable distill` | removing clutter from each moved view | M1–M2 |
| `impeccable onboard` | Start screen and room lobby | M2–M3 |
| `impeccable adapt` | phone layouts | M1–M3 |
| `impeccable animate`, `delight` | small reactions, empty states | M5 |
| `impeccable critique`, `audit`, `polish` | review before the freeze | M5 |
| `dataviz` | every chart: form, colour roles, validator, hover | M1–M4 |
| `hallmark`, `design-taste-frontend` | Start screen as a new page; check against a templated look | M2 |
| `diagram-design` | architecture and pipeline figures for slides and report | M5–M6 |
| `claude-api` | before writing any LLM call | M4 |

---

## 8. Checks before every push

```bash
node tests/replay.mjs && node tests/check_ledger.mjs && node tests/check_lens.mjs
node tests/check_tradeoff.mjs && node tests/check_preview.mjs
node tests/check_detail.mjs && node tests/check_themes.mjs
node tests/check_group.mjs            # new in W4
.venv/bin/python tests/check_steering.py && .venv/bin/python tests/check_time_safety.py
.venv/bin/python tests/check_nl.py && .venv/bin/python tests/check_app.py
.venv/bin/python tests/check_serving.py   # W3, needs server/bundle (src/31_export_serving_bundle.py)
cd web && npm run build
```

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Too much for the time, especially a December 1 slot | Fixed order and cut list in §6; the current site stays deployable throughout |
| Classroom network blocks the tunnel or the map tiles | Static site with sample profiles; D3 map fallback; recorded backup |
| New users have little history | Cold-start study sets the minimum picks; sample taste profiles as an alternative; the app shows how sure it is |
| LLM features do not help | That is a valid RQ3 result; report it |
| Review text leaks the future into features | Text cutoff before the earliest training window; time-safety test |
| The LLM misreads a sentence | Chips show what was understood; nothing changes until the solver runs; accuracy is measured |
| Location and nicknames are personal data | Rounded coordinates, no accounts, 24-hour room life, SQLite file git-ignored |
| The warm palette cannot separate three chart roles by colour | Glyph and texture on every effect role; validator in the checks |
| Data ends in 2021 | Stated in the app and the report |

---

## 10. Decisions and approvals needed

- **Instructor:** which day Team 4 presents.
- **Fangyu:** agreement with this direction; who takes which workstream (recorded in the report's
  contribution table).
- **Yila:** the mock-up direction (type, third chart colour, illustration style); the new
  dependencies in §3; the LLM batch cost once `--plan` prints it; whose API key the server uses.

---

## 11. Git and housekeeping

- Branch from `main` (`app-redesign` now); small pull requests into `main`; `main` deploys the site.
- Remove the untracked duplicate files named "… 2" after checking them; add `web/.vite/`, `server/bundle/`
  and `server/*.sqlite` to `.gitignore`.
- Update `README.md`, `web/README.md`, `docs/HANDOFF.md` and `docs/USER_STUDY.md` as each workstream lands.
