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
  results dashboard and explorer (`web/src/main.js`). Both maps (the Lab's Explorer and the app) use Leaflet 1.9 on OpenStreetMap tiles (Yila's OK, 2026-11-26). `server/` the optional API: FastAPI (one-sentence steering in `nl.py`, live taste profiles in
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
- Both maps load street tiles from `tile.openstreetmap.org` while they are looked at: that server sees the
  visitor's IP address and the area in view. In the app the area in view can be around the visitor's start point;
  the start point itself is never sent, and "About the data" says this. The attribution stays on the map. Nothing
  else on the site calls an outside service without being asked.
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
  Lab and `app/app.css` for the app). The two pages have different palettes, both in that file.
- **App look** (changed by Yila on 2026-11-26): the diner app shares the Lab's world. A canvas of coral, light
  yellow and light blue; frosted cards on it (70% white, blurred), no paper grain, no flat colour blocks. Deep
  coral `#A63A2E` (`--tomato`) is for actions, selection and pins; teal `#1F7872` (`--herb`) is the second colour;
  text is warm ink. The app's old token names are kept and only their values changed (`web/src/tokens.css`, base
  block). The map is a zoomable street map with deep-coral numbered pins. A start is moved on the avatar itself (Yila, 2026-11-27; the "Move Tomato" buttons are gone): drag it, or tap it to hold it and then tap the map as often as wanted (Done, Esc or a second tap lets go), or Enter and the arrow keys; the view stays put after a move unless a pin leaves it. Every way ends in the same handler, which rounds the start to about 100 m and refuses a point outside the rectangle around the city that the location button also uses. Fredoka, Nunito and Patrick Hand stay as
  the app's faces. Buttons and fields are rectangles (6px corners) in the app and the Lab; cards stay rounded.
- **The mark** of the app and the Lab is two friends with their arms round each other's shoulders, in black line
  (Yila chose it on 2026-11-27 from five candidates; it replaced a watercolour bowl with two spoons, whose files are kept)
  (`web/public/illustrations/brand/`, AI-generated with Magic Patterns). Tomato and Tofu are people, so neither is
  ever used as the app's mark.
- **Lab look** (chosen by Yila on 2026-11-26; the `:root[data-site="lab"]` block): frosted cards (70% white over
  a blurred canvas of coral, light yellow and light blue), type in Urbanist. Chart marks use her six colours:
  terracotta, mist blue, teal, grape, forest, mustard. Three colour rules hold on every Lab page (agreed with Yila
  on 2026-11-26). Rankers: `lr` mist blue, `lgbm` grape, `lmr` terracotta, and nothing else wears those colours on
  the Analysis page; marks are circles wherever each series has its own row or panel. The baselines are mustard
  (Popularity), teal (Recent popularity) and forest (Item-CF). Quantities: one hue, light teal to deep teal
  (`--seq-lo` to `--seq-hi`), never a ranker colour. The Explorer map is the exception Yila asked for: its
  neighbourhood shading is hot coral (`--pop`, stronger = cheaper to reach) and its place bubbles sunny yellow
  (`--pop-base`) with dark numbers, so shading and places stay apart. Emphasis: the red circle
  (`--signal-red`) is the reference or the selected item only; guide lines are the soft blue `--frontier-line`.
  What moved a place in the Explorer (roles, not rankers, so no ranker colour): the model's own reasons teal
  `#1F7872`, solid; your controls mustard, dashed, with an ink edge on bars; the variety step the page's deep
  coral, dotted; start and total bars a quiet sand (`--total`). The rank flow on the Explorer ("How your top 10
  changed") is the exception to dashes (Yila, 2026-11-26): every line is solid, in its step's colour where the rank
  changed at that step (`--model-line` teal `#008B7D`, `--control-line` ochre `#B98B1E`, `--diversity-line` deep
  coral; validated together) and a thinner grey (`--flow-same`) where it did not. The step is never read from colour
  alone there: it is the gap between two named columns, each column has a bar in its colour, and the sentence under
  the chart names the steps that moved the hovered or selected restaurant. Card titles are 24px on every Lab page (Yila,
  2026-11-26; 20px before). That card has the entered and left counts as two tinted tags under its title; the Explorer has no summary line outside the cards. "Your top 10" shows the list first (rank, name at 17px,
  a status in words: "No change", "↑ 2 places", "↓ 2 places", "Entered top 10") and the explanation one press in: a
  row opens its detail under it (before and now, and the Shapley split on a bar chart that grows right for "up" and
  left for "down" from a zero line; values to one decimal, never rounded to "no effect"). One detail is open at a
  time. The places no longer in the top 10 are a card of their own under the map, and the map takes the height left
  over, so the map column and the list column always end level (Yila found unequal cards weird). "Why the model recommends this
  place" has a different job from the list and never shares a chart or a scale with it: the list splits a rank
  change across the person's settings, in places; this card explains the model's own score, for the model as
  trained, in the model's score units (group SHAP values; base + six groups = the raw score). It has two columns that start level, one
  chart each and nothing repeated (Yila, 2026-11-26): on the left the selected restaurant, one generated sentence and
  its whole score as a sum read down the rows (the typical candidate's score, every factor group as a bar from one
  zero line, right adds and left takes away, then "= the model score"; a running-total waterfall was tried and Yila
  could not tell which bars subtract); on the right a picker for a second
  restaurant, one generated sentence and every factor group's difference on bars from a zero line, with the total.
  No "top three" chart before the full one, no nested disclosures. What a bar is (SHAP, the inputs and groups, the
  typical candidate) is one paragraph on a teal tint between the title and the restaurant's name; the only
  disclosure, at the bottom, holds the table of single inputs (Yila cut the other method notes on 2026-11-26). The person's settings are not explained here; what they do to the selected
  restaurant's score (the 0–1 scale) is in the controls' "Model details". Two scores are called "similar" when
  they are less than twice the median gap between neighbours in that model's top 10 for that user.
  Never present a steered explanation as the original model's, and never turn a contribution into a probability. Under it,
  "What reviewers say about this place" shows Fangyu's review themes (`web/public/data/themes/`, an LLM's reading of
  reviews from before the snapshot) for the same selected restaurant: every theme the file holds (never a top few,
  which would hide the rarer complaints), with its tone in words, how many sampled reviews mention it, and one
  checked quote. It is a card of its own, not part of the list or of the model card, because the rankers do not
  read these summaries, and the card says so; it names the model that read the reviews (Claude Sonnet 5.5) in its
  first line, which is its AI disclosure (Yila removed the separate note on 2026-11-26). Long official area names are written as their first neighbourhood alone in a row, with nothing after it (the
  full name is the row's tooltip; Yila removed the area line from the detail); there is no verified list of short names, so nothing is abbreviated or renamed. Yila finds dark brown and mist blue ugly as large
  fills: do not fill bars with ink or with mist blue, and feel free to add colours that suit the page. The group card on the Overview keeps blue, orange and purple for its three rules (Yila's request). Its top right corner holds a teal-tinted box, "How a group's list is ranked": min-max satisfaction per member over the pool, then the three rules as formulas (mean; lowest; mean after dropping places where a member is below 0.40), then the 10 best (src/group.py). The legend under the lead carries the rule names only, since the box says what each does. A highlighted box in a card's top right corner is the Lab's place for "how this is computed" (Analysis cards use it too). Mustard and teal are
  too light to stand alone, so every mark has an ink outline and every series is named next to its mark; text
  in a series colour uses `inkOf()` (the colour mixed into ink). Do not make cards more transparent than 70%.
- Estimates with intervals follow the ranker chart (`intervals` with `band: true` in
  `web/src/views/overview.js`): the name on the left, the value and interval on the right, and between them a
  shaded bar that spans the interval (the one part taken from a newspaper poll chart) with a solid circle on the
  estimate; a legend, and a stated axis start when it is not zero. Yila asked for this layout on 2026-11-26:
  do not write names above the bars or vary the mark shape there.
- Categories are never encoded by colour alone (shape or dash, plus a legend). Run the `dataviz` validator on
  any new chart colours, on the surface they will sit on.
- The Lab shows a short set of charts, most of them about the models. Analysis has three cards, each with all
  three rankers or none: what each ranker relies on (three bar charts on shared rows and one scale from zero; a bar
  is a group's share of mean |SHAP| pooled over 300 test users, so it has no sign; bars carry a clear top-to-bottom
  gradient (lighter top, the ranker's colour below) and no shadow, and since 2026-11-26 every bar on the Explorer has
  the same gradient in its own colour (list detail, the two model-score charts, the single-input table); the selected user is a ring on the bar's centre
  line, and a figure that the ring would cover steps just past it; the rings show only while
  "Compare with selected user" is on; the card's title states the finding and is written from the data),
  calibration ("Do predicted probabilities match actual visits?": three reliability diagrams on the same log axes;
  after calibration in the ranker's colour with a solid line and filled points, before calibration in a neutral
  grey with a thin line and hollow points, the diagonal dashed, 95% Wilson intervals thin and light, no gradient or
  shadow on lines; a bin without visits is drawn in a row marked 0 under the axis, never moved to a small positive
  value; LambdaRank has no "before" curve because its raw score is not a probability; method, numbers and the
  count of predictions per bin sit in one "Method & data" disclosure), and the similarity map ("Restaurants cluster more by
  borough than by cuisine": one layout drawn twice, every restaurant a small round warm-grey dot (`--sim-other`
  `#C9BBB1`, 60%), the picked borough on the left and cuisine group on the right in brick red (`--sim-sel`); Yila
  chose these on 2026-11-26 in place of the sunny yellow and hot coral she had asked for earlier that day. Above the
  pictures, two rows of "Observed" against "Chance baseline": the share of restaurants whose own borough, or cuisine
  group, is the most common among their 15 nearest restaurants in the 64-number reviewer space, against always
  answering with the largest group. Never word it as "its neighbours share its borough 92% of the time", and never
  call the layout a map of the city). It has no model switch and no pairwise comparison. Before adding one, say which question it
  answers that no chart there answers; results that do not earn a chart stay in their `outputs/` reports and the
  README, not in the Lab (Yila removed the Overview's "More results" table on 2026-11-26).
- Cuisine pictures and member avatars are watercolour stickers generated with Magic Patterns, in
  true-to-life food colours (`web/public/illustrations/`, one per cuisine group; prompt template and
  rules in `docs/PLAN.md` §2.4). They appear on recommendation cards and the place drawer only; on the
  map a restaurant is a deep-coral pin. Say in the report that they are AI-generated.
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
