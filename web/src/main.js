// SteerEats app: loads a demo case, solves the user's outcome targets per model, and renders the linked views.
import * as d3 from "d3";
import { LAM_MAX, THETA0, finalRanks, outcomes, shapleyControls, similarity, solveBoost, solveTargets, stageLog, stages, steeringCost, topK } from "./steer.js";
import { MODEL_NAMES, STEPS, esc, hueVar } from "./views/common.js";
import { mountConsole, renderSolution } from "./views/console.js";
import { highlightFlow, renderFlow } from "./views/flow.js";
import { createMap } from "./views/map.js";
import { highlightList, markSelected, renderList } from "./views/list.js";
import { renderContrast, renderLedger } from "./views/ledger.js";
import { createLens } from "./views/lens.js";
import { createEvaluation } from "./views/evaluation.js";
import { createTradeoff } from "./views/tradeoff.js";
import { renderPreviews } from "./views/preview.js";
import { previewSweep } from "./preview.js";
import { predictedVisits, visitProbs } from "./tradeoff.js";

const STORY = { local_manhattan: "Manhattan regular", popularity_dominated: "Popular-places fan", models_disagree: "Models disagree" };
const state = {
  model: "lgbm", w_h: 1, kmDraft: 2, targets: { km: null, popularity: null, cuisines: null },
  c: null, sim: null, caseId: null, hover: null, result: null, selected: null, contrast: null,
  boostNta: null, costCache: new Map(), ntaCodes: [],
  thetaOverride: null, // a grid setting loaded from the trade-off view; targets are paused while it is set
};
const el = (id) => document.getElementById(id);
// data files live next to the page, wherever the site is served from (vite.config.js sets a relative base)
const DATA = `${import.meta.env.BASE_URL}data/`;

async function boot() {
  const [index, nta, boroughs, ntaCodes, lensPop, evaluation, tradeoff] = await Promise.all(["cases/index.json", "nyc_nta.geojson", "nyc_boroughs.geojson", "nta_index.json", "model_lens.json", "evaluation.json", "tradeoff.json"]
    .map((f) => DATA + f).map((u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`Could not load ${u}`); return r.json(); })));
  state.ntaCodes = ntaCodes;
  el("cutoff").textContent = `Snapshot ${index.cutoff} · test period Aug 2020 – Apr 2021`;
  const scripted = index.cases.filter((x) => x.story !== "random"), others = index.cases.filter((x) => x.story === "random");
  el("case-select").innerHTML =
    `<optgroup label="Stories">${scripted.map((x) => `<option value="${x.file}">${STORY[x.story] ?? x.story} · ${esc(x.home_borough)}</option>`).join("")}</optgroup>` +
    `<optgroup label="Other test users">${others.map((x) => `<option value="${x.file}">User ${x.id.slice(-2)} · ${esc(x.home_borough)}</option>`).join("")}</optgroup>`;
  el("case-select").addEventListener("change", (e) => loadCase(e.target.value));
  document.querySelectorAll(".segmented button[data-model]").forEach((b) => b.addEventListener("click", () => {
    setModel(b.dataset.model);
    update({ refit: true });
  }));
  el("flow-key").innerHTML = [["model", "History weight moved it"], ["control", "Distance, popularity or neighborhood moved it"],
    ["diversity", "Variety moved it"], [null, "No change at that step"]]
    .map(([h, t]) => `<li><i style="background:${hueVar(h)}"></i>${t}</li>`).join("");
  views.map = createMap(el("map"), { nta, boroughs, codes: ntaCodes }, { hover: setHover, select, boost: toggleBoost });
  state.platt = tradeoff.platt;
  views.tradeoff = createTradeoff(el("tradeoff"), tradeoff, { load: loadTheta });
  views.lens = createLens(el("lens"), lensPop, { hover: setHover, select });
  views.evaluation = createEvaluation(el("evaluation"), evaluation);
  el("contrast-select").addEventListener("change", (e) => { state.contrast = e.target.value === "" ? null : +e.target.value; renderWhy(); });
  const fromHash = readHash(index);
  views.console = mountConsole(el("console"), state, () => schedule(), { clearBoost: () => toggleBoost(null), clearLoaded: () => { state.thetaOverride = null; update({ refit: false }); } });
  setModel(state.model);
  window.addEventListener("resize", () => { schedule(false); views.evaluation.render(); views.tradeoff.render(); });
  // the previews follow the rail's width (it changes at the one-column breakpoint)
  let railW = 0;
  new ResizeObserver(([e]) => { const w = Math.round(e.contentRect.width); if (w !== railW) { railW = w; if (state.c) previews(); } }).observe(el("console"));
  await loadCase(fromHash ?? index.cases[0].file);
}

// Hash keys: case=demo_03, model=lr|lgbm, wh=0.5, km=2, pop=0.4, cui=6, th=w_h,alpha,beta,lam (loaded grid setting)
function readHash(index) {
  const h = new URLSearchParams(location.hash.slice(1));
  const num = (k) => (h.has(k) && h.get(k) !== "" && Number.isFinite(+h.get(k)) ? +h.get(k) : null);
  if (h.get("model") === "lr" || h.get("model") === "lgbm") state.model = h.get("model");
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

function setModel(m) {
  state.model = m;
  document.querySelectorAll(".segmented button[data-model]").forEach((x) => x.setAttribute("aria-checked", String(x.dataset.model === m)));
}

async function loadCase(file) {
  const c = await fetch(`${DATA}cases/${file}`).then((r) => { if (!r.ok) throw new Error(`Could not load ${file}`); return r.json(); });
  state.c = c; state.sim = similarity(c); state.caseId = file; state.contrast = null;
  state.selected = state.pendingSel != null && state.pendingSel < c.cuisine.length ? state.pendingSel : null;
  state.pendingSel = null;
  state.boostNta = state.pendingBoost != null && c.nta.includes(state.pendingBoost) ? state.pendingBoost : null;
  state.pendingBoost = null;
  el("case-select").value = file;
  update({ refit: true });
}

const views = {};
let pending = null;
function schedule(refit = true) {
  if (pending) return;
  pending = requestAnimationFrame(() => { pending = null; update({ refit }); });
}

function compute() {
  const { c, sim, model } = state;
  const otherModel = model === "lr" ? "lgbm" : "lr";
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
  const key = `${state.caseId}|${model}`;
  if (!state.costCache.has(key)) state.costCache.set(key, steeringCost(c, model));
  // Model lens: both models' final lists under the same targets, and their own (unsteered) lists.
  const dkey = `${state.caseId}|defaults`;
  if (!state.costCache.has(dkey)) state.costCache.set(dkey, { lr: finalRanks(c, THETA0, "lr", sim), lgbm: finalRanks(c, THETA0, "lgbm", sim) });
  const finals = { [model]: after, [otherModel]: finalRanks(c, otherTheta, otherModel, sim) };
  const off = (t) => t.w_h !== 1 || t.alpha || t.beta || t.lam || t.boost;
  const lens = { finals, defaults: state.costCache.get(dkey), steered: off(theta) || off(otherTheta) };
  return { solved, other, theta, boost, st, before, after, top, defTop, gone, phi: sh.phi, cost: state.costCache.get(key), lens };
}

// A setting loaded from the trade-off view is used as is, for both models; otherwise the targets are solved.
function solveOrLoad(c, m) {
  if (!state.thetaOverride) return solveTargets(c, m, state.targets, state.w_h);
  const theta = { ...THETA0, ...state.thetaOverride }, top = topK(c, theta, m);
  return { theta, top, outcomes: outcomes(c, top), met: { km: true, popularity: true, cuisines: true }, loaded: true };
}

function loadTheta(t) {
  state.thetaOverride = { w_h: t.w_h, alpha: t.alpha, beta: t.beta, lam: t.lam };
  state.w_h = t.w_h;
  state.targets = { km: null, popularity: null, cuisines: null };
  views.console.sync();
  update({ refit: false });
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

function summarize(r) {
  const c = state.c, liked = c.later_liked || [];
  const mean = (ids, f) => ids.reduce((a, i) => a + f(i), 0) / ids.length;
  const km0 = mean(r.defTop, (i) => c.candidates.distance_km[i]), km1 = mean(r.top, (i) => c.candidates.distance_km[i]);
  const cu0 = new Set(r.defTop.map((i) => c.cuisine[i])).size, cu1 = new Set(r.top.map((i) => c.cuisine[i])).size;
  const lk0 = r.defTop.filter((i) => liked[i]).length, lk1 = r.top.filter((i) => liked[i]).length;
  const changed = r.gone.length;
  const head = changed === 0 ? "Your top 10 matches the model's list."
    : `<b>${changed} ${changed === 1 ? "place" : "places"} entered</b> and ${changed} left your top 10.`;
  // the model's own estimate of later visits in the list (Platt-calibrated, as in the trade-off view)
  const pk = `${state.caseId}|${state.model}|probs`;
  if (!state.costCache.has(pk)) state.costCache.set(pk, visitProbs(c, state.model, state.platt));
  const p = state.costCache.get(pk), kept = predictedVisits(p, r.top) / predictedVisits(p, r.defTop);
  const keptText = changed === 0 ? "" : ` It keeps <span class="num">${kept < 1 && Math.round(kept * 100) >= 100 ? (Math.floor(kept * 1000) / 10).toFixed(1) : Math.round(kept * 100)}%</span> of the visits the model predicts for this user.`;
  return `${head} Average distance <span class="num">${km0.toFixed(1)}&nbsp;km → ${km1.toFixed(1)}&nbsp;km</span>, cuisine groups <span class="num">${cu0} → ${cu1}</span>, places this user later liked <span class="num">${lk0} → ${lk1}</span>.${keptText}`;
}

function update({ refit }) {
  if (!state.c) return;
  const r = compute();
  state.result = r;
  writeHash();
  const c = state.c, liked = c.later_liked || new Array(c.cuisine.length).fill(false);
  renderSolution(el("console"), state, r);
  el("summary").innerHTML = summarize(r);
  const ranks = { default: r.before, model: r.st.S0, distance: r.st.S1, popularity: r.st.S2, boost: r.st.S2b, diversity: r.st.S3 };
  r.ranks = ranks;
  if (state.selected == null) state.selected = r.top[0];
  renderFlow(el("flow"), { c, ranks, stepNotes: stepNotes(r.theta), selected: state.selected }, { hover: setHover, select });
  renderList(el("list"), el("exited"), { c, top: r.top, before: r.before, after: r.after, phi: r.phi, liked, gone: r.gone }, { hover: setHover, select });
  views.map.update({ c, top: r.top, gone: r.gone, liked, selected: state.selected, cost: r.cost, boostNta: state.boostNta,
    home: { lat: c.user.home_lat, lon: c.user.home_lon } }, { refit });
  markSelected([el("list"), el("exited")], state.selected);
  renderWhy();
  views.tradeoff.update({ c, model: state.model, theta: r.theta });
  previews();
  views.lens.update({ c, model: state.model, selected: state.selected, ...r.lens });
  setHover(state.hover);
}

// Console previews: each control swept over its range with the others held (web/src/preview.js). The sweeps are
// recomputed shortly after the controls settle; meanwhile the last ones are drawn (dimmed if they are out of date).
const previewCache = new Map();
let previewTimer = null, lastSweeps = null;
function previews() {
  const { c, model, w_h, targets } = state;
  const key = `${state.caseId}|${model}|${w_h}|${targets.km}|${targets.popularity}|${targets.cuisines}`;
  const current = { w_h, km: targets.km == null ? undefined : targets.km, popularity: targets.popularity, cuisines: targets.cuisines };
  const draw = (sweeps, stale) => renderPreviews(el("console"), sweeps, current, pickPreview, { stale, boost: state.boostNta != null });
  if (previewCache.has(key)) { lastSweeps = previewCache.get(key); draw(lastSweeps, false); return; }
  draw(lastSweeps?.caseKey === `${state.caseId}|${model}` ? lastSweeps : null, true);
  clearTimeout(previewTimer);
  previewTimer = setTimeout(() => {
    const sweeps = Object.fromEntries(["w_h", "km", "popularity", "cuisines"].map((k) => [k, previewSweep(c, model, state.platt, w_h, targets, k)]));
    Object.defineProperty(sweeps, "caseKey", { value: `${state.caseId}|${model}`, enumerable: false });
    if (previewCache.size > 200) previewCache.clear();
    previewCache.set(key, sweeps);
    if (state.c === c && state.model === model) previews();
  }, 120);
}

// Clicking a preview sets that control (and leaves a setting loaded from the trade-off view).
function pickPreview(key, v) {
  state.thetaOverride = null;
  if (key === "w_h") state.w_h = v;
  else if (key === "km") { state.kmDraft = v; state.targets.km = v; }
  else state.targets[key] = v;
  views.console.sync();
  update({ refit: false });
}

// Default comparison: a place that left the list (best default rank first), else the next place down.
function defaultContrast(r, a) {
  if (r.gone.length) return r.gone.find((i) => i !== a) ?? null;
  const pos = r.top.indexOf(a);
  return pos >= 0 && pos < r.top.length - 1 ? r.top[pos + 1] : r.top[0] === a ? r.top[1] : r.top[0];
}

function renderWhy() {
  const r = state.result, c = state.c, a = state.selected;
  if (!r || a == null) return;
  const options = [...r.top, ...r.gone].filter((i) => i !== a);
  if (state.contrast == null || state.contrast === a || !options.includes(state.contrast)) state.contrast = defaultContrast(r, a);
  el("contrast-select").innerHTML = options.map((i) =>
    `<option value="${i}">${r.after[i] <= 10 ? `#${r.after[i]}` : "left"} · ${esc(c.candidates.name[i])}</option>`).join("");
  if (state.contrast != null) el("contrast-select").value = state.contrast;
  const ctx = { c, model: state.model, theta: r.theta, ranks: r.ranks };
  renderLedger(el("ledger"), { ...ctx, i: a, notes: el("ledger-notes"), detail: el("ledger-detail") });
  renderContrast(el("contrast"), { ...ctx, a, b: state.contrast, caption: el("contrast-caption") });
}

// Clicking a neighbourhood boosts it; clicking it again (or the console's remove button) turns the boost off.
function toggleBoost(nta) {
  state.boostNta = nta == null || state.boostNta === nta ? null : nta;
  update({ refit: false });
}

function select(i) {
  state.selected = i;
  update({ refit: false });
}

function setHover(id) {
  state.hover = id;
  highlightFlow(el("flow"), id);
  highlightList([el("list"), el("exited")], id);
  views.map.highlight(id);
  views.lens?.highlight(id);
}

boot().catch((err) => {
  el("summary").textContent = `The app could not start: ${err.message}.` + (import.meta.env.DEV ? ' Run it with "npm run dev" from the web folder.' : " Reload the page to try again.");
  console.error(err);
});
