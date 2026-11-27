// SteerEats app: loads a demo case, solves the user's outcome targets per model, and renders the linked views.
import * as d3 from "d3";
import { LAM_MAX, THETA0, outcomes, shapleyControls, similarity, solveBoost, solveTargets, stageLog, stages, steeringCost, topK } from "./steer.js";
import { MODEL_NAMES, STEPS, VERSUS, esc } from "./views/common.js";
import { createOverview } from "./views/overview.js";
import { mountConsole, renderSolution } from "./views/console.js";
import { highlightFlow, renderFlow, renderFormula } from "./views/flow.js";
import { createMap } from "./views/map.js";
import { highlightList, markSelected, renderList } from "./views/list.js";
import { renderWhy as renderWhyCard } from "./views/ledger.js";
import { renderReviews } from "./views/reviews.js";
import { createLens } from "./views/lens.js";
import { createEvaluation } from "./views/evaluation.js";
import { renderPreviews } from "./views/preview.js";
import { createSimilarity } from "./views/similarity.js";
import { createStudy } from "./study.js";
import { previewSweep } from "./preview.js";

const STORY = { local_manhattan: "Manhattan regular", popularity_dominated: "Popular-places fan", models_disagree: "Models disagree" };
const state = {
  tab: "overview", model: "lmr", versus: "lgbm", w_h: 1, kmDraft: 2, targets: { km: null, popularity: null, cuisines: null },
  c: null, sim: null, caseId: null, hover: null, result: null, selected: null, contrast: null,
  boostNta: null, costCache: new Map(), ntaCodes: [],
  thetaOverride: null, // a grid setting from a link (th=); targets are paused while it is set
  showPreviews: false,
  themes: undefined,   // this test user's review summaries, one entry per candidate: undefined while loading, null if there is no file
  themesMeta: null,    // how the summaries were made (themes/index.json)
  listOpen: false,     // the selected restaurant's detail is open in "Your top 10" (closed until someone selects one)
  picking: false,      // "Choose on map" was pressed: the map waits for a neighbourhood
};
const el = (id) => document.getElementById(id);
// data files live next to the page, wherever the site is served from (vite.config.js sets a relative base)
const DATA = `${import.meta.env.BASE_URL}data/`;

async function boot() {
  const [index, nta, boroughs, ntaCodes, lensPop, evaluation, tradeoff, simMap] = await Promise.all(["cases/index.json", "nyc_nta.geojson", "nyc_boroughs.geojson", "nta_index.json", "model_lens.json", "evaluation.json", "tradeoff.json", "similarity_map.json"]
    .map((f) => DATA + f).map((u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`Could not load ${u}`); return r.json(); })));
  state.ntaCodes = ntaCodes;
  el("cutoff").textContent = `Snapshot ${index.cutoff} · test period Aug 2020 – Apr 2021`;
  // optional result files for the overview: a panel says so when its file has not been produced yet
  const opt = (f) => fetch(DATA + f).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const [overview, rq2] = await Promise.all(["overview.json", "rq2.json"].map(opt));
  views.overview = createOverview(el("overview"), { overview, evaluation, tradeoff, rq2 });
  document.querySelectorAll(".tabs button[data-tab]").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
  const scripted = index.cases.filter((x) => x.story !== "random"), others = index.cases.filter((x) => x.story === "random");
  el("case-select").innerHTML =
    `<optgroup label="Stories">${scripted.map((x) => `<option value="${x.file}">${STORY[x.story] ?? x.story} · ${esc(x.home_borough)}</option>`).join("")}</optgroup>` +
    `<optgroup label="Other test users">${others.map((x) => `<option value="${x.file}">User ${x.id.slice(-2)} · ${esc(x.home_borough)}</option>`).join("")}</optgroup>`;
  el("case-select").addEventListener("change", (e) => loadCase(e.target.value));
  document.querySelectorAll(".segmented button[data-model]").forEach((b) => b.addEventListener("click", () => {
    setModel(b.dataset.model);
    acted();
    update({ refit: true });
  }));
  views.map = createMap(el("map"), { nta, boroughs, codes: ntaCodes }, { hover: setHover, select, boost: toggleBoost });
  state.platt = tradeoff.platt;
  views.sim = createSimilarity(el("similarity"), simMap, { title: el("sim-title"), stats: el("sim-stats") });
  views.lens = createLens(el("lens"), lensPop, { title: el("lens-title"), sub: el("lens-sub"), method: el("lens-method"), tools: el("lens-tools") });
  views.evaluation = createEvaluation(el("evaluation"), evaluation);
  const fromHash = readHash(index);
  views.console = mountConsole(el("console"), state, () => { acted(); schedule(); }, { clearBoost: () => toggleBoost(null), pickOnMap: () => pickOnMap(!state.picking), clearLoaded: () => { state.thetaOverride = null; update({ refit: false }); } });
  setModel(state.model);
  window.addEventListener("resize", () => { schedule(false); views.evaluation.render(); views.sim.redraw(); views.overview.render(); });
  // the previews follow the rail's width (it changes at the one-column breakpoint)
  let railW = 0;
  new ResizeObserver(([e]) => { const w = Math.round(e.contentRect.width); if (w !== railW) { railW = w; if (state.c) previews(); } }).observe(el("console"));
  await loadCase(fromHash ?? index.cases[0].file);
  setTab(state.study ? "explorer" : state.tab);
  if (state.study) {
    const spec = await fetch(`${DATA}study.json`).then((r) => r.json());
    study = createStudy(state.study, spec, {
      lockCase: (on) => { el("case-select").disabled = on; },
      setCondition: (cond) => { document.body.classList.toggle("study-list", cond === "list"); window.dispatchEvent(new Event("resize")); },
      applyState: async (t) => {
        const s = t.state;
        state.thetaOverride = null; state.boostNta = null; state.selected = null;
        state.w_h = s.wh ?? 1;
        state.targets = { km: s.km ?? null, popularity: s.pop ?? null, cuisines: s.cui ?? null };
        if (s.km != null) state.kmDraft = s.km;
        setModel(s.model);
        views.console.sync();
        await loadCase(`${t.case}.json`);
      },
      outcome: () => ({ top: state.result.top, defTop: state.result.defTop, model: state.model, c: state.c }),
    });
  }
}

// Hash keys: case=demo_03, model=lr|lgbm|lmr, wh=0.5, km=2, pop=0.4, cui=6, th=w_h,alpha,beta,lam (loaded grid setting),
function readHash(index) {
  const h = new URLSearchParams(location.hash.slice(1));
  const num = (k) => (h.has(k) && h.get(k) !== "" && Number.isFinite(+h.get(k)) ? +h.get(k) : null);
  // links made before the sections existed carry a case and no tab: they mean the explorer
  if (["overview", "explorer", "analysis"].includes(h.get("tab"))) state.tab = h.get("tab");
  else if (h.has("case")) state.tab = "explorer";
  if (h.get("model") in MODEL_NAMES) { state.model = h.get("model"); state.versus = VERSUS[state.model]; }
  if (/^P\d{1,2}$/.test(h.get("study") ?? "")) { state.study = h.get("study"); state.showPreviews = true; }
  if (num("wh") != null) state.w_h = Math.min(1, Math.max(0, num("wh")));
  if (num("km") != null) { state.kmDraft = num("km"); state.targets.km = num("km"); }
  if (num("pop") != null) state.targets.popularity = num("pop");
  if (num("cui") != null) state.targets.cuisines = num("cui");
  if (num("sel") != null) state.pendingSel = num("sel");
  if (num("boost") != null) state.pendingBoost = num("boost");
  const th = (h.get("th") ?? "").split(",").map(Number);
  if (th.length === 4 && th.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)) {
    state.thetaOverride = { w_h: th[0], alpha: th[1], beta: th[2], lam: Math.min(th[3], LAM_MAX) };
    state.w_h = th[0]; state.targets = { km: null, popularity: null, cuisines: null };
  }
  const want = h.get("case");
  const hit = index.cases.find((x) => x.id === want || x.file === want);
  return hit ? hit.file : null;
}

function writeHash() {
  const h = new URLSearchParams();
  h.set("tab", state.tab);
  if (state.study) h.set("study", state.study);
  h.set("case", state.caseId.replace(".json", ""));
  h.set("model", state.model);
  if (state.thetaOverride) { const t = state.thetaOverride; h.set("th", [t.w_h, t.alpha, t.beta, t.lam].join(",")); }
  else if (state.w_h !== 1) h.set("wh", state.w_h);
  if (state.targets.km != null) h.set("km", state.targets.km);
  if (state.targets.popularity != null) h.set("pop", state.targets.popularity);
  if (state.targets.cuisines != null) h.set("cui", state.targets.cuisines);
  if (state.boostNta != null) h.set("boost", state.boostNta);
  if (state.selected != null) h.set("sel", state.selected);
  history.replaceState(null, "", "#" + h.toString());
}

// Sections: hidden views have no width, so they are redrawn when their section is shown.
function setTab(t) {
  state.tab = t;
  document.body.dataset.tab = t;
  document.querySelectorAll(".tabs button[data-tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t)));
  // what the top bar's choices do on this page (Analysis shows all three rankers, so it has no model switch)
  el("subbar-note").textContent = t === "analysis" ? "The first chart can mark this test user. The other two cover all test users and all restaurants."
    : "One test user's 200 candidates. Everything below follows this user and model.";
  if (state.caseId) writeHash();
  // drawn directly, not on the next animation frame: a page in a background tab gets no frames
  if (t === "overview") views.overview?.render();
  setTimeout(() => window.dispatchEvent(new Event("resize")), 0);
}

// The model in view, and the one the console's note compares it with (common.js VERSUS).
function setModel(m) {
  state.model = m;
  state.versus = VERSUS[m];
  document.querySelectorAll(".segmented button[data-model]").forEach((x) => x.setAttribute("aria-checked", String(x.dataset.model === m)));
}

async function loadCase(file) {
  const c = await fetch(`${DATA}cases/${file}`).then((r) => { if (!r.ok) throw new Error(`Could not load ${file}`); return r.json(); });
  state.c = c; state.caseId = file;
  // the review summaries come after the page is up: the card says "loading" until they are in
  state.themes = undefined;
  Promise.all([fetch(`${DATA}themes/${file}`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    state.themesMeta ?? fetch(`${DATA}themes/index.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null)])
    .then(([t, meta]) => { if (state.caseId !== file) return; state.themesMeta = meta; state.themes = t?.restaurants ?? null; renderWhy(); });
  state.sim = similarity(c);
  state.contrast = null;
  state.listOpen = false;
  state.selected = state.pendingSel != null && state.pendingSel < state.c.cuisine.length ? state.pendingSel : null;
  state.pendingSel = null;
  state.boostNta = state.pendingBoost != null && state.c.nta.includes(state.pendingBoost) ? state.pendingBoost : null;
  state.pendingBoost = null;
  el("case-select").value = file;
  update({ refit: true });
}

const views = {};
let pending = null, study = null;
const acted = () => study?.action();   // study mode counts the participant's interactions
function schedule(refit = true) {
  if (pending) return;
  pending = requestAnimationFrame(() => { pending = null; update({ refit }); });
}

function compute() {
  const { c, sim, model } = state;
  const otherModel = state.versus;
  const solved = solveOrLoad(c, model);
  const other = solveOrLoad(c, otherModel);
  let theta = solved.theta, otherTheta = other.theta, boost = null;
  if (state.boostNta != null) {
    const b = solveBoost(c, model, theta, state.boostNta);
    const ob = solveBoost(c, otherModel, other.theta, state.boostNta);
    theta = b.theta; otherTheta = ob.theta;
    boost = { nta: state.boostNta, amount: b.theta.boost, reached: b.reached, other: ob.theta.boost, otherReached: ob.reached };
    // targets are judged on the final list, after the boost
    const fin = stages(c, theta, model, sim).S3;
    const o = outcomes(c, [...Array(c.cuisine.length).keys()].filter((i) => fin[i] <= 10));
    const t = state.targets;
    solved.outcomes = o;
    solved.met = { km: t.km == null || o.km <= t.km, popularity: t.popularity == null || o.popularity <= t.popularity, cuisines: t.cuisines == null || o.cuisines >= t.cuisines };
  }
  const st = stages(c, theta, model, sim);
  const log = stageLog(c, theta, model, sim);
  const sh = shapleyControls(c, theta, model, sim);
  const before = log.before, after = log.after;
  const n = before.length;
  const top = [...Array(n).keys()].filter((i) => after[i] <= 10).sort((a, b) => after[a] - after[b]);
  const defTop = [...Array(n).keys()].filter((i) => before[i] <= 10).sort((a, b) => before[a] - before[b]);
  const gone = defTop.filter((i) => after[i] > 10);
  const key = `${c.id}|${model}`;
  if (!state.costCache.has(key)) state.costCache.set(key, steeringCost(c, model));
  return { solved, other, otherModel, theta, boost, st, before, after, top, defTop, gone, phi: sh.phi, cost: state.costCache.get(key) };
}

// A setting loaded from a link (th=) is used as is, for both models; otherwise the targets are solved.
function solveOrLoad(c, m) {
  if (!state.thetaOverride) return solveTargets(c, m, state.targets, state.w_h);
  const theta = { ...THETA0, ...state.thetaOverride }, top = topK(c, theta, m);
  return { theta, top, outcomes: outcomes(c, top), met: { km: true, popularity: true, cuisines: true }, loaded: true };
}

function stepNotes(theta) {
  const f = (v) => v.toFixed(2);
  return {
    default: MODEL_NAMES[state.model],
    model: theta.w_h === 1 ? "off" : `${Math.round(theta.w_h * 100)}% weight`,
    distance: theta.alpha === 0 ? "off" : `α ${f(theta.alpha)}`,
    popularity: theta.beta === 0 ? "off" : `β ${f(theta.beta)}`,
    boost: !theta.boost ? "off" : `+${f(theta.boost)}`,
    diversity: theta.lam === 0 ? "off" : `λ ${f(theta.lam / LAM_MAX)}`,
  };
}

// How many places the steering brought into the top 10 and how many it pushed out: the model's own list against
// the final one (not a count of the moves at each step in between).
function counts(r) {
  const entered = r.top.filter((i) => r.before[i] > 10).length, left = r.gone.length;
  const tag = (n, word) => `<span class="tag${n ? "" : " none"}"><b>${n}</b> ${word}</span>`;
  return `${tag(entered, "entered")}<span class="sep" aria-hidden="true">·</span>${tag(left, "left")}`;
}

function update({ refit }) {
  if (!state.c) return;
  const r = compute();
  state.result = r;
  writeHash();
  const c = state.c, liked = c.later_liked || new Array(c.cuisine.length).fill(false);
  const ranks = { default: r.before, model: r.st.S0, distance: r.st.S1, popularity: r.st.S2, boost: r.st.S2b, diversity: r.st.S3 };
  r.ranks = ranks;
  if (state.selected == null) state.selected = r.top[0];
  renderSolution(el("console"), state, r);
  el("flow-counts").innerHTML = counts(r);
  renderFormula(el("flow-formula"), { theta: r.theta, modelName: MODEL_NAMES[state.model], candidates: state.c.cuisine.length });
  renderFlow(el("flow"), { c, ranks, stepNotes: stepNotes(r.theta), selected: state.selected, read: el("flow-read") }, { hover: setHover, select });
  renderList(el("list"), el("exited"), { c, top: r.top, before: r.before, after: r.after, phi: r.phi, liked, gone: r.gone, open: state.listOpen ? state.selected : null }, { hover: setHover, select: selectInList });
  views.map.update({ c, top: r.top, gone: r.gone, liked, selected: state.selected, cost: r.cost, boostNta: state.boostNta,
    home: { lat: c.user.home_lat, lon: c.user.home_lon } }, { refit });
  markSelected([el("list"), el("exited")], state.selected);
  renderWhy();
  views.sim.update();
  previews();
  views.lens.update({ c, userName: el("case-select").selectedOptions[0]?.textContent ?? "" });
  setHover(state.hover);
}

// Console previews: each control swept over its range with the others held (web/src/preview.js). The sweeps are
// recomputed shortly after the controls settle; meanwhile the last ones are drawn (dimmed if they are out of date).
const previewCache = new Map();
let previewTimer = null, lastSweeps = null;
function previews() {
  if (!state.showPreviews) return;
  const { c, model, w_h, targets } = state;
  const key = `${c.id}|${model}|${w_h}|${targets.km}|${targets.popularity}|${targets.cuisines}`;
  const current = { w_h, km: targets.km == null ? undefined : targets.km, popularity: targets.popularity, cuisines: targets.cuisines };
  const draw = (sweeps, stale) => renderPreviews(el("console"), sweeps, current, pickPreview, { stale, boost: state.boostNta != null });
  if (previewCache.has(key)) { lastSweeps = previewCache.get(key); draw(lastSweeps, false); return; }
  draw(lastSweeps?.caseKey === `${c.id}|${model}` ? lastSweeps : null, true);
  clearTimeout(previewTimer);
  previewTimer = setTimeout(() => {
    const sweeps = Object.fromEntries(["w_h", "km", "popularity", "cuisines"].map((k) => [k, previewSweep(c, model, state.platt, w_h, targets, k)]));
    Object.defineProperty(sweeps, "caseKey", { value: `${c.id}|${model}`, enumerable: false });
    if (previewCache.size > 200) previewCache.clear();
    previewCache.set(key, sweeps);
    if (state.c === c && state.model === model) previews();
  }, 120);
}

// Clicking a preview sets that control (and leaves a setting loaded from a link).
function pickPreview(key, v) {
  acted();
  state.thetaOverride = null;
  if (key === "w_h") state.w_h = v;
  else if (key === "km") { state.kmDraft = v; state.targets.km = v; }
  else state.targets[key] = v;
  views.console.sync();
  update({ refit: false });
}

// "Why the model recommends this place": the selected restaurant's model score, explained, and compared with a
// second restaurant once one is picked among the places in the list and those that left it.
function renderWhy() {
  const r = state.result, c = state.c, a = state.selected;
  if (!r) return;
  if (state.contrast === a) state.contrast = null;
  // (a restaurant already being compared stays on offer when a new model or setting takes it off both lists)
  const options = [...new Set([...r.top, ...r.gone, ...(state.contrast != null ? [state.contrast] : [])])].filter((i) => i !== a)
    .map((i) => [i, `${r.after[i] <= 10 ? `#${r.after[i]}` : "Outside top 10"} · ${c.candidates.name[i]}`]);
  renderWhyCard(el("why"), { c, model: state.model, i: a, b: state.contrast, options }, { pick: (i) => { acted(); state.contrast = i; renderWhy(); } });
  renderReviews(el("reviews"), { name: a == null ? null : c.candidates.name[a], entry: state.themes == null ? state.themes : state.themes[a] ?? null,
    meta: state.themesMeta, missing: state.themes === null });
}

// Clicking a neighbourhood boosts it; clicking it again (or the console's remove button) turns the boost off.
function toggleBoost(nta) {
  acted();
  state.boostNta = nta == null || state.boostNta === nta ? null : nta;
  state.picking = false; views.map.pick(false);
  update({ refit: false });
}

// "Choose on map" in the console: bring the map into view and say what to click. The click itself is the map's
// own (toggleBoost); pressing the button again or Esc leaves without choosing.
function pickOnMap(on) {
  state.picking = on;
  views.map.pick(on);
  renderSolution(el("console"), state, state.result);
  if (on) {
    el("map").scrollIntoView({ block: "center", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    el("map").focus({ preventScroll: true });
  } else el("console").querySelector("#boost-pick")?.focus({ preventScroll: true });
}
window.addEventListener("keydown", (e) => { if (e.key === "Escape" && state.picking) pickOnMap(false); });

// Selecting a restaurant anywhere (chart, map, list) selects it everywhere and opens its detail in the list.
function select(i) {
  acted();
  state.selected = i;
  state.listOpen = true;
  update({ refit: false });
}

// In the list a row is also its own way out: pressing the open row closes its detail (it stays selected).
function selectInList(i) {
  if (state.selected === i && state.listOpen) { acted(); state.listOpen = false; update({ refit: false }); }
  else select(i);
}

function setHover(id) {
  state.hover = id;
  highlightFlow(el("flow"), id);
  highlightList([el("list"), el("exited")], id);
  views.map.highlight(id);
}

// The blur under the sticky bar belongs to scrolling: none while the page is at the top, growing over the first
// 90 px of scroll. Its height follows the bar, so it always ends a little below it.
{
  const blur = document.querySelector(".scroll-blur"), bar = document.querySelector(".subbar");
  if (blur && bar) {
    const sync = () => {
      blur.style.setProperty("--on", Math.min(1, window.scrollY / 90).toFixed(3));
      const bottom = bar.getBoundingClientRect().bottom;
      if (bottom > 0) blur.style.height = `${Math.round(bottom + 44)}px`;
    };
    window.addEventListener("scroll", sync, { passive: true });
    window.addEventListener("resize", sync);
    new ResizeObserver(sync).observe(bar);
    sync();
  }
}

boot().catch((err) => {
  el("boot-error").hidden = false;
  el("boot-error").textContent = `The app could not start: ${err.message}.` + (import.meta.env.DEV ? ' Run it with "npm run dev" from the web folder.' : " Reload the page to try again.");
  console.error(err);
});
