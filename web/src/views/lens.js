// Model lens: what does each of the three rankers rely on?
// Three small bar charts side by side, on the same rows and the same scale from zero. A bar is a feature group's
// share of the model's mean |SHAP|: for every input the absolute SHAP value is averaged over the candidates, the
// inputs of a group are added, and the sum is divided by the total over all inputs (src/17_shap.py and
// src/39_lambdarank_shap.py: 300 test users drawn at random with a fixed seed, their candidates pooled before the
// average). Switched on, "Compare with selected user" adds a ring per bar: the same share over that one user's
// candidates (web/src/lens.js `groupShares`, checked against Python by tests/check_lens.mjs). A share says how
// strongly a group moves scores, in either direction: it is not a part of the score and has no sign.
// The rows are HTML, so the three charts line up row by row, a whole row lights up together, and on a narrow
// screen the charts go one under the other with their own labels. Nothing is scaled down.
import { MODEL_NAMES, esc } from "./common.js";
import { FEATURE, GROUP_LABEL } from "./ledger.js";
import { groupShares } from "../lens.js";

const ORDER = ["lr", "lgbm", "lmr"];
const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const count = (n) => WORDS[n] ?? String(n);
const pct = (v) => `${(v * 100).toFixed(1)}%`;                       // one decimal everywhere: 44.2%, 1.5%
const lower = (g) => (GROUP_LABEL[g] ?? g).replace(/^Your /, "").replace(/^\w/, (ch) => ch.toLowerCase());
const list = (xs) => (xs.length < 3 ? xs.join(" and ") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

// els: { title, sub, method, tools } are the card's heading, the line under it, the box in its top right corner and the
// place for the switch under that line, filled from the data
export function createLens(root, pop, els = {}) {
  const P = pop.shap_groups.rows, N = pop.shap_groups.users;
  // one order for all three charts: the group that matters most to any model first
  const groups = Object.keys(P).sort((a, b) => Math.max(...ORDER.map((m) => P[b][m])) - Math.max(...ORDER.map((m) => P[a][m])));
  const ranked = (m) => [...groups].sort((a, b) => P[b][m] - P[a][m]);
  const top = ORDER.map((m) => ranked(m)[0]), second = Object.fromEntries(ORDER.map((m) => [m, ranked(m)[1]]));
  const sameTop = top.every((g) => g === top[0]);
  if (els.title) els.title.textContent = sameTop ? `All ${count(ORDER.length)} models rely most on ${lower(top[0])}` : "What does each ranker rely on?";
  if (els.sub) els.sub.textContent = `Compare the importance of ${count(groups.length)} feature groups across ${count(ORDER.length)} ranking models.`;
  // the switch and its legend sit in the card's heading, under the line of text and beside the method box (els.tools);
  // without that place they go above the chart
  const tools = `<label class="switch" for="rel-compare"><input id="rel-compare" type="checkbox" role="switch"><span class="sw-track" aria-hidden="true"></span>Compare with selected user</label>
      <p class="rel-legend" id="rel-legend" hidden></p>`;
  if (els.tools) els.tools.innerHTML = tools;
  root.innerHTML = `
    ${els.tools ? "" : `<div class="rel-tools">${tools}</div>`}
    <div class="rel" id="rel"></div>
    <p class="rel-read" id="rel-read" aria-live="polite"></p>
    <p class="rel-foot">Shares show the strength of each feature group’s influence, not whether it raises or lowers scores.</p>`;
  const $ = (id) => root.querySelector("#" + id) ?? els.tools?.querySelector("#" + id);
  const state = { c: null, userName: "", compare: false, hover: null, focus: null, pinned: null, user: null, drawn: null };

  function draw() {
    const { c } = state;
    if (!c) return;
    const user = (state.user?.id === c.id ? state.user : (state.user = { id: c.id, of: Object.fromEntries(ORDER.map((m) => [m, groupShares(c, m)])) })).of;
    // one scale for the three charts, from zero: 0–50% unless a bar or a ring needs more, then the next 10%
    const most = Math.max(...groups.flatMap((g) => ORDER.flatMap((m) => [P[g][m], user[m][g] ?? 0])));
    const max = Math.max(0.5, Math.ceil(most * 10 - 1e-9) / 10), at = (v) => `${((100 * v) / max).toFixed(3)}%`;
    const ticks = Array.from({ length: Math.round(max * 10) + 1 }, (_, k) => k / 10);
    const label = (g) => GROUP_LABEL[g] ?? g;
    const say = (g) => `${label(g)}: ${ORDER.map((m) => `${MODEL_NAMES[m]} ${pct(P[g][m])}`).join(", ")}`;
    $("rel").style.setProperty("--step", at(0.1));
    $("rel").classList.toggle("comparing", state.compare);
    $("rel").innerHTML = `
      <div class="rel-row rel-head"><span class="rel-name"></span>${ORDER.map((m, j) => `<h3 class="rel-cell rel-model" style="--m:${j};--r:0">${MODEL_NAMES[m]}</h3>`).join("")}</div>
      ${groups.map((g, r) => `<div class="rel-row${g === top[0] && sameTop ? " lead" : ""}" data-g="${g}" tabindex="0" role="group" aria-label="${esc(say(g))}">
        <span class="rel-name">${label(g)}</span>
        ${ORDER.map((m, j) => `<div class="rel-cell" data-g="${g}" style="--m:${j};--r:${r + 1}"><span class="rel-cell-name">${label(g)}</span>
          <div class="rel-plot"><i class="rel-bar ${m}" style="width:${at(P[g][m])}"></i>
            <b class="rel-pct" style="left:${at(P[g][m])}">${pct(P[g][m])}${second[m] === g ? '<small>2nd</small>' : ""}</b>
            <i class="rel-ring" style="left:${at(user[m][g] ?? 0)}"></i></div></div>`).join("")}
      </div>`).join("")}
      <div class="rel-row rel-axis-row" aria-hidden="true"><span class="rel-name"></span>${ORDER.map((_, j) => `<div class="rel-cell" style="--m:${j};--r:${groups.length + 1}"><div class="rel-axis">
        ${ticks.map((t) => `<span style="left:${at(t)}">${Math.round(t * 100)}${t === 0 ? "" : "%"}</span>`).join("")}</div></div>`).join("")}</div>`;
    $("rel-legend").hidden = !state.compare;
    $("rel-legend").innerHTML = `<i class="rel-key-bar" aria-hidden="true"></i>Bars: Across ${N} test users <span aria-hidden="true">·</span> <i class="rel-key-ring" aria-hidden="true"></i>Selected user: <b>${esc(state.userName)}</b>`;
    // how a share is calculated, in the card's top right corner (no longer folded away)
    if (els.method) els.method.innerHTML = `
      <p>A share is a feature group’s part of the model’s mean absolute SHAP value. For each of the model’s ${c.features.length} inputs, the absolute SHAP value is averaged over the candidates; the inputs of a group are added together and divided by the total over all inputs. So a model’s ${count(groups.length)} shares add up to 100%.</p>
      <p><b>${N}</b> test users, each with <b>${c.cuisine.length}</b> candidate restaurants.</p>`;
    read();
    mark();
    place();
  }

  // The ring is on the bar's centre line, at the user's own share. The figure normally sits 9px after the end of its
  // bar; when the ring would land on it, the figure steps to just past the ring (the ring is never moved).
  function place() {
    for (const plot of $("rel").querySelectorAll(".rel-plot")) {
      const w = plot.clientWidth, fig = plot.querySelector(".rel-pct"), ring = plot.querySelector(".rel-ring");
      if (!fig || !ring) continue;
      fig.style.marginLeft = "";
      if (!state.compare || !w) continue;
      const end = (parseFloat(fig.style.left) / 100) * w, at = (parseFloat(ring.style.left) / 100) * w, R = 6.5;
      const from = end + 9, to = from + fig.offsetWidth;
      if (at + R > from - 2 && at - R < to + 2) fig.style.marginLeft = `${(at + R + 6 - end).toFixed(1)}px`;
    }
  }
  new ResizeObserver(place).observe($("rel"));

  // The line under the chart: the row in hand (pointer, keyboard or tap) with exact values and the inputs it holds;
  // otherwise a short hint.
  const active = () => state.hover ?? state.focus ?? state.pinned;
  function read() {
    const g = active(), c = state.c;
    if (!c) return;
    if (!g) { $("rel-read").innerHTML = `<span class="rel-hint">Point at a row, or tap it, for exact values.</span>`; return; }
    const user = state.user.of, label = GROUP_LABEL[g] ?? g;
    const inputs = (c.feature_groups[g] ?? []).map((f) => FEATURE[f]?.[0] ?? f);
    $("rel-read").innerHTML = `<b>${label}.</b> Across ${N} test users: ${ORDER.map((m) => `${MODEL_NAMES[m]} ${pct(P[g][m])}`).join(", ")}.`
      + (state.compare ? ` ${esc(state.userName)}: ${ORDER.map((m) => pct(user[m][g] ?? 0)).join(", ")}.` : "")
      + ` <span class="rel-inputs">Inputs: ${esc(inputs.join("; ").toLowerCase())}.</span>`;
  }
  function mark() {
    const g = active();
    for (const el of $("rel").querySelectorAll("[data-g]")) el.classList.toggle("on", el.dataset.g === g);
  }
  const set = (key, g) => { if (state[key] === g) return; state[key] = g; read(); mark(); };
  const groupOf = (e) => e.target.closest?.("[data-g]")?.dataset.g ?? null;
  $("rel").addEventListener("pointerover", (e) => { if (e.pointerType !== "touch") set("hover", groupOf(e)); });
  $("rel").addEventListener("pointerleave", () => set("hover", null));
  $("rel").addEventListener("focusin", (e) => set("focus", groupOf(e)));
  $("rel").addEventListener("focusout", () => set("focus", null));
  $("rel").addEventListener("click", (e) => { const g = groupOf(e); set("pinned", g && state.pinned !== g ? g : null); });   // a tap keeps the row; a second tap lets go
  $("rel-compare").addEventListener("change", (e) => { state.compare = e.target.checked; draw(); });

  function update({ c, userName }) {
    const key = `${c.id}|${userName}`;
    state.c = c; state.userName = userName ?? "";
    if (key === state.drawn) return;
    state.drawn = key;
    draw();
  }
  return { update, highlight() {} };
}
