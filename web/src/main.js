// SteerEats app: loads a demo case, solves the user's outcome targets per model, and renders the linked views.
import * as d3 from "d3";
import { LAM_MAX, THETA0, shapleyControls, similarity, solveTargets, stageLog, stages } from "./steer.js";
import { MODEL_NAMES, STEPS, esc, hueVar } from "./views/common.js";
import { mountConsole, renderSolution } from "./views/console.js";
import { highlightFlow, renderFlow } from "./views/flow.js";
import { createMap } from "./views/map.js";
import { highlightList, renderList } from "./views/list.js";

const STORY = { local_manhattan: "Manhattan regular", popularity_dominated: "Popular-places fan", models_disagree: "Models disagree" };
const state = {
  model: "lgbm", w_h: 1, kmDraft: 2, targets: { km: null, popularity: null, cuisines: null },
  c: null, sim: null, caseId: null, hover: null, result: null,
};
const el = (id) => document.getElementById(id);

async function boot() {
  const [index, nta, boroughs] = await Promise.all(["/data/cases/index.json", "/data/nyc_nta.geojson", "/data/nyc_boroughs.geojson"]
    .map((u) => fetch(u).then((r) => { if (!r.ok) throw new Error(`Could not load ${u}`); return r.json(); })));
  el("cutoff").textContent = `Snapshot ${index.cutoff} · test period Aug 2020 – Apr 2021`;
  const scripted = index.cases.filter((x) => x.story !== "random"), others = index.cases.filter((x) => x.story === "random");
  el("case-select").innerHTML =
    `<optgroup label="Stories">${scripted.map((x) => `<option value="${x.file}">${STORY[x.story] ?? x.story} · ${esc(x.home_borough)}</option>`).join("")}</optgroup>` +
    `<optgroup label="Other test users">${others.map((x) => `<option value="${x.file}">User ${x.id.slice(-2)} · ${esc(x.home_borough)}</option>`).join("")}</optgroup>`;
  el("case-select").addEventListener("change", (e) => loadCase(e.target.value));
  document.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => {
    setModel(b.dataset.model);
    update({ refit: true });
  }));
  el("flow-key").innerHTML = [["model", "History weight moved it"], ["control", "Distance or popularity moved it"],
    ["diversity", "Variety moved it"], [null, "No change at that step"]]
    .map(([h, t]) => `<li><i style="background:${hueVar(h)}"></i>${t}</li>`).join("");
  views.map = createMap(el("map"), { nta, boroughs }, { hover: setHover });
  const fromHash = readHash(index);
  views.console = mountConsole(el("console"), state, () => schedule());
  setModel(state.model);
  window.addEventListener("resize", () => schedule(false));
  await loadCase(fromHash ?? index.cases[0].file);
}

// Hash keys: case=demo_03, model=lr|lgbm, wh=0.5, km=2, pop=0.4, cui=6
function readHash(index) {
  const h = new URLSearchParams(location.hash.slice(1));
  const num = (k) => (h.has(k) && h.get(k) !== "" && Number.isFinite(+h.get(k)) ? +h.get(k) : null);
  if (h.get("model") === "lr" || h.get("model") === "lgbm") state.model = h.get("model");
  if (num("wh") != null) state.w_h = Math.min(1, Math.max(0, num("wh")));
  if (num("km") != null) { state.kmDraft = num("km"); state.targets.km = num("km"); }
  if (num("pop") != null) state.targets.popularity = num("pop");
  if (num("cui") != null) state.targets.cuisines = num("cui");
  const want = h.get("case");
  const hit = index.cases.find((x) => x.id === want || x.file === want);
  return hit ? hit.file : null;
}

function writeHash() {
  const h = new URLSearchParams();
  h.set("case", state.caseId.replace(".json", ""));
  h.set("model", state.model);
  if (state.w_h !== 1) h.set("wh", state.w_h);
  if (state.targets.km != null) h.set("km", state.targets.km);
  if (state.targets.popularity != null) h.set("pop", state.targets.popularity);
  if (state.targets.cuisines != null) h.set("cui", state.targets.cuisines);
  history.replaceState(null, "", "#" + h.toString());
}

function setModel(m) {
  state.model = m;
  document.querySelectorAll(".segmented button").forEach((x) => x.setAttribute("aria-checked", String(x.dataset.model === m)));
}

async function loadCase(file) {
  const c = await fetch(`/data/cases/${file}`).then((r) => r.json());
  state.c = c; state.sim = similarity(c); state.caseId = file;
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
  const solved = solveTargets(c, model, state.targets, state.w_h);
  const other = solveTargets(c, model === "lr" ? "lgbm" : "lr", state.targets, state.w_h);
  const theta = solved.theta;
  const st = stages(c, theta, model, sim);
  const log = stageLog(c, theta, model, sim);
  const sh = shapleyControls(c, theta, model, sim);
  const before = log.before, after = log.after;
  const n = before.length;
  const top = [...Array(n).keys()].filter((i) => after[i] <= 10).sort((a, b) => after[a] - after[b]);
  const defTop = [...Array(n).keys()].filter((i) => before[i] <= 10).sort((a, b) => before[a] - before[b]);
  const gone = defTop.filter((i) => after[i] > 10);
  return { solved, other, theta, st, before, after, top, defTop, gone, phi: sh.phi };
}

function stepNotes(theta) {
  const f = (v) => v.toFixed(2);
  return {
    default: MODEL_NAMES[state.model],
    model: theta.w_h === 1 ? "off" : `${Math.round(theta.w_h * 100)}% weight`,
    distance: theta.alpha === 0 ? "off" : `α ${f(theta.alpha)}`,
    popularity: theta.beta === 0 ? "off" : `β ${f(theta.beta)}`,
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
  return `${head} Average distance <span class="num">${km0.toFixed(1)}&nbsp;km → ${km1.toFixed(1)}&nbsp;km</span>, cuisine groups <span class="num">${cu0} → ${cu1}</span>, places this user later liked <span class="num">${lk0} → ${lk1}</span>.`;
}

function update({ refit }) {
  if (!state.c) return;
  const r = compute();
  state.result = r;
  writeHash();
  const c = state.c, liked = c.later_liked || new Array(c.cuisine.length).fill(false);
  renderSolution(el("console"), state, r);
  el("summary").innerHTML = summarize(r);
  const ranks = { default: r.before, model: r.st.S0, distance: r.st.S1, popularity: r.st.S2, diversity: r.st.S3 };
  renderFlow(el("flow"), { c, ranks, stepNotes: stepNotes(r.theta) }, { hover: setHover });
  renderList(el("list"), el("exited"), { c, top: r.top, before: r.before, after: r.after, phi: r.phi, liked, gone: r.gone }, { hover: setHover });
  views.map.update({ c, top: r.top, gone: r.gone, liked, home: { lat: c.user.home_lat, lon: c.user.home_lon } }, { refit });
  setHover(state.hover);
}

function setHover(id) {
  state.hover = id;
  highlightFlow(el("flow"), id);
  highlightList([el("list"), el("exited")], id);
  views.map.highlight(id);
}

boot().catch((err) => {
  el("summary").textContent = `The app could not start: ${err.message}. Run it with "npm run dev" from the web folder.`;
  console.error(err);
});
