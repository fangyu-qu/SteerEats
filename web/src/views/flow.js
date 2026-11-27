// Rank flow (V2): a bump chart from the model's default ranking through each steering step.
// Lines keep their identity across steps (object constancy). Every line is solid: a segment takes its step's colour
// where the rank changed at that step and is a thinner grey where it did not. Which step a segment belongs to is
// also read from where it sits (between two named columns), from the colour bar under each column's name, from the
// sentence under the chart and from each segment's tooltip, so colour is never the only code. The selected
// restaurant's whole path is thick and the others step back.
// When a control changes, each line glides to its new ranks (320 ms, ease-out); places that join or leave the
// chart fade. No motion on a new case or width, or when the reader asks for reduced motion.
import * as d3 from "d3";
import { STEPS, esc, hueLine } from "./common.js";
import { LAM_MAX } from "../steer.js";

const TOPN = 10, DURATION = 320;
const ROW = 34, HEAD = 58, TOP = HEAD + 10;   // header: the step's name, its setting, its colour bar
const RANK_W = 30, MIN_NAME = 96;
const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

// text widths are measured in the page, at the size the stylesheet gives them (and again once the web font is in)
const widths = new Map();
document.fonts?.ready.then(() => widths.clear());
let last = null;   // what the chart shows now: for the sentence under it

export function renderFlow(root, data, handlers) {
  const { c, ranks, stepNotes } = data;          // ranks: { default, model, distance, popularity, boost, diversity } (1-based)
  const svg = d3.select(root).selectAll("svg").data([0]).join("svg").attr("role", "img")
    .attr("aria-label", "Bump chart of the top 10 from the original list through each steering step");
  const measure = (cls, s) => {
    const key = `${cls}|${s}`;
    if (widths.has(key)) return widths.get(key);
    const t = svg.append("text").attr("class", cls).text(s), w = t.node().getComputedTextLength();
    t.remove();
    if (w > 0) widths.set(key, w);   // (a hidden section measures 0: not kept, measured again when it is shown)
    return w;
  };
  // a name that does not fit its column is cut at a whole character, with an ellipsis
  const fit = (cls, s, room) => {
    if (measure(cls, s) <= room) return s;
    let k = s.length - 1;
    while (k > 1 && measure(cls, s.slice(0, k).trimEnd() + "…") > room) k--;
    return s.slice(0, k).trimEnd() + "…";
  };

  const n = c.candidates.name.length;
  const ids = [];
  for (let i = 0; i < n; i++) if (STEPS.some((s) => ranks[s.key][i] <= TOPN)) ids.push(i);
  const first = STEPS[0].key, final = STEPS[STEPS.length - 1].key;

  // Room. The step names come first: two neighbours must not touch, which sets how close the columns may sit. They
  // are written in full where that leaves the restaurant names enough room, else shortened (first the two long
  // ones, then "Neighborhood" as well). The names share what is left. Below that the chart keeps its spacing and
  // scrolls sideways.
  const box = Math.max(root.clientWidth, 320), gaps = STEPS.length - 1;
  const need = (pick) => d3.max(d3.range(gaps), (k) => (measure("col-label", pick(STEPS[k])) + measure("col-label", pick(STEPS[k + 1]))) / 2 + 12);
  const widestName = (key) => d3.max(ids.filter((i) => ranks[key][i] <= TOPN), (i) => measure("name", c.candidates.name[i])) ?? 0;
  const wantL = Math.min(210, widestName(first)), wantR = Math.min(260, widestName(final)), fixed = RANK_W + 14 + 16;
  const forms = [(s) => s.label, (s) => (s.key === "boost" ? s.label : s.short ?? s.label), (s) => s.short ?? s.label];
  const comfy = Math.min(wantL, 115) + Math.min(wantR, 135);   // names kept readable before a step name is shortened
  const label = forms.find((f) => box - fixed - gaps * need(f) >= comfy) ?? forms[forms.length - 1];
  // the columns get a quarter more than they need before the names take more than "readable"; then the names
  // grow to their full length, and what is left after that goes back to the columns
  const plotMin = gaps * need(label), spare = Math.max(box - fixed - plotMin, 2 * MIN_NAME);
  const budget = Math.min(wantL + wantR, Math.max(Math.min(spare, comfy), box - fixed - plotMin * 1.25));
  const leftRoom = Math.max(MIN_NAME, budget * wantL / (wantL + wantR || 1)), rightRoom = Math.max(MIN_NAME, budget - leftRoom);
  const left = RANK_W + leftRoom + 14, right = rightRoom + 16;
  const width = Math.max(box, Math.ceil(left + right + gaps * need(label)));
  const belowY = TOP + TOPN * ROW + 24;
  const height = belowY + 30;
  const x = d3.scalePoint(STEPS.map((s) => s.key), [left, width - right]);
  const y = (r) => (r <= TOPN ? TOP + (r - 0.5) * ROW : belowY + Math.min(r - TOPN - 1, 30) * 0.35);

  const animate = root.dataset.case === c.id && +root.dataset.width === width && !reducedMotion();
  root.dataset.case = c.id; root.dataset.width = width;
  const T = animate ? d3.transition("flow").duration(DURATION).ease(d3.easeCubicOut) : null;
  const tw = (sel) => (T ? sel.transition(T) : sel);
  svg.attr("width", width).attr("height", height);

  // Column headers and guides
  const cols = svg.selectAll("g.col").data(STEPS, (d) => d.key).join((e) => {
    const g = e.append("g").attr("class", "col");
    g.append("line").attr("class", "guide");
    g.append("text").attr("class", "col-label").attr("text-anchor", "middle");
    g.append("text").attr("class", "col-sub").attr("text-anchor", "middle");
    g.append("rect").attr("class", "col-hue").attr("height", 4).attr("rx", 2);
    return g;
  });
  cols.select("line").attr("x1", (d) => x(d.key)).attr("x2", (d) => x(d.key)).attr("y1", HEAD).attr("y2", belowY + 10);
  cols.select(".col-label").attr("x", (d) => x(d.key)).attr("y", 15).text(label);
  cols.select(".col-sub").attr("x", (d) => x(d.key)).attr("y", 35).text((d) => stepNotes[d.key]);
  // the colour of the lines a step moves, under its name (the first column is the starting list and moves nothing)
  cols.select(".col-hue").attr("x", (d) => x(d.key) - 15).attr("y", 44).attr("width", 30)
    .attr("fill", (d) => (d.hue ? hueLine(d.hue) : "none"));

  const ticks = svg.selectAll("text.rank-tick").data(d3.range(1, TOPN + 1)).join("text").attr("class", "rank-tick")
    .attr("x", 4).attr("y", (r) => y(r)).attr("dy", "0.35em").text((r) => r);
  ticks.raise();
  svg.selectAll("text.below-label").data([0]).join("text").attr("class", "below-label")
    .attr("x", 4).attr("y", belowY).attr("dy", "0.35em").text("below 10");

  // Items
  const link = (a, b) => {
    const xa = x(a.key), xb = x(b.key);
    return (i) => { const ya = y(ranks[a.key][i]), yb = y(ranks[b.key][i]), mx = (xa + xb) / 2;
      return `M${xa},${ya} C${mx},${ya} ${mx},${yb} ${xb},${yb}`; };
  };
  const segs = STEPS.slice(1).map((s, k) => ({ from: STEPS[k], to: s, path: link(STEPS[k], s) }));
  const kind = (s, i) => {
    const a = ranks[s.from.key][i], b = ranks[s.to.key][i];
    return a > TOPN && b > TOPN ? "out" : a !== b ? "moved" : "same";
  };
  const items = svg.selectAll("g.item").data(ids, (i) => i).join((e) => {
    const g = e.append("g").attr("class", "item").style("opacity", T ? 0 : null);
    // (the fade-in leaves no inline opacity behind: the stylesheet dims and lifts items after it)
    if (T) g.transition(T).style("opacity", 1).on("end", function () { d3.select(this).style("opacity", null); });
    segs.forEach(() => g.append("path").attr("class", "seg").append("title"));
    g.append("g").attr("class", "dots");
    g.append("text").attr("class", "name left").attr("text-anchor", "end").attr("dy", "0.35em");
    g.append("text").attr("class", "name right").attr("text-anchor", "start").attr("dy", "0.35em");
    return g;
  }, (u) => u, (ex) => (T ? ex.interrupt("flow").transition(T).style("opacity", 0).remove() : ex.remove()));
  items.attr("data-id", (i) => i).classed("selected", (i) => i === data.selected)
    .on("pointerenter", (_, i) => handlers.hover(i)).on("pointerleave", () => handlers.hover(null))
    .on("click", (_, i) => handlers.select?.(i));
  items.each(function (i) {
    const g = d3.select(this), name = c.candidates.name[i], bold = i === data.selected ? 0.93 : 1;   // the selected name is bold, so wider
    tw(g.selectAll("path.seg").data(segs)).attr("d", (s) => s.path(i));
    g.selectAll("path.seg").attr("class", (s) => `seg ${kind(s, i)}`).attr("stroke-dasharray", null)
      .attr("stroke", (s) => ({ out: "var(--line)", moved: hueLine(s.to.hue), same: "var(--flow-same)" })[kind(s, i)])
      .each(function (s) {
        const a = ranks[s.from.key][i], b = ranks[s.to.key][i];
        d3.select(this).select("title").text(a === b ? `${name}: #${a}, not moved by ${s.to.label}` : `${name}: #${a} → #${b} at ${s.to.label}`);
      });
    const dots = g.select(".dots").selectAll("circle").data(STEPS).join("circle").attr("cx", (s) => x(s.key));
    tw(dots).attr("cy", (s) => y(ranks[s.key][i]));
    dots.attr("r", (s) => (s.key === final && ranks[final][i] <= TOPN ? 5 : 3.2))
      .attr("fill", (s) => (s.key === final && ranks[final][i] <= TOPN ? "var(--ink)" : "var(--panel)"))
      .attr("stroke", (s) => (ranks[s.key][i] <= TOPN ? "var(--ink-2)" : "var(--line-strong)")).attr("stroke-width", 1.5);
    const d0 = ranks[first][i], d1 = ranks[final][i];
    tw(g.select(".name.left")).attr("y", y(d0));
    g.select(".name.left").attr("x", left - 12).text(d0 <= TOPN ? fit("name", name, leftRoom * bold) : "").classed("muted", d1 > TOPN);
    tw(g.select(".name.right")).attr("y", y(d1));
    g.select(".name.right").attr("x", width - right + 12).text(d1 <= TOPN ? fit("name", name, rightRoom * bold) : "");
  });
  // the selected restaurant is drawn last, over the others, which step back (styles.css)
  root.classList.toggle("has-selected", ids.includes(data.selected));
  items.filter((i) => i === data.selected).raise();

  last = { c, ranks, segs, first, final, selected: data.selected, read: data.read };
  describe(null);
}

// The sentence under the chart: where the hovered (or else the selected) restaurant started, where it is now, and
// which steps moved it, in words.
function describe(hover) {
  if (!last?.read) return;
  const { c, ranks, segs, first, final, read } = last, i = hover ?? last.selected;
  if (i == null) { read.textContent = ""; return; }
  const d0 = ranks[first][i], d1 = ranks[final][i];
  const what = d0 <= TOPN && d1 <= TOPN ? (d0 === d1 ? `stays at #${d0}` : `went from #${d0} to #${d1}`)
    : d0 <= TOPN ? `left the top 10 (#${d0} to #${d1})` : d1 <= TOPN ? `entered the top 10 (#${d0} to #${d1})` : d0 === d1 ? `stays outside the top 10, at #${d0}` : `is outside the top 10 (#${d0} to #${d1})`;
  const moves = segs.filter((s) => ranks[s.from.key][i] !== ranks[s.to.key][i])
    .map((s) => `<span class="step"><i style="background:${hueLine(s.to.hue)}"></i>${esc(s.to.label)} #${ranks[s.from.key][i]} → #${ranks[s.to.key][i]}</span>`);
  read.innerHTML = `<b>${esc(c.candidates.name[i])}</b> ${what}. ${moves.length ? `Moved by ${moves.join(", ")}.` : "No step moved it."}`;
}

export function highlightFlow(root, id) {
  const shown = id != null && !!root.querySelector(`g.item[data-id="${id}"]`);
  root.classList.toggle("has-hover", shown);
  root.querySelectorAll("g.item").forEach((g) => {
    const on = id != null && +g.dataset.id === id;
    g.classList.toggle("hover", on);
    if (on) g.parentNode.appendChild(g);
  });
  describe(id);
}

// The score behind the chart, written out with the weights now in use (web/src/steer.js `stages`, the same steps in
// the same order as the columns). The numbers are the solver's; the fine print sits behind the "?".
// DIST_CAP_KM and the two similarity weights are the constants of src/steering.py (15 km; 0.5 and 0.5).
export function renderFormula(root, { theta, modelName, candidates }) {
  if (!root.dataset.built) {
    root.dataset.built = "1";
    root.innerHTML = `
      <div class="ff-top"><h3>How the score is built</h3>
        <button type="button" class="help-btn" aria-expanded="false" aria-controls="help-formula" aria-label="About the score formula">?</button></div>
      <p class="ctl-help" id="help-formula" hidden></p>
      <ol class="ff-lines">
        <li><span class="ff-op">score =</span><span><span data-k="model"></span> score, its history part × <b class="chip model" data-k="w_h"></b></span></li>
        <li><span class="ff-op">−</span><span><b class="chip control" data-k="alpha"></b> × distance</span></li>
        <li><span class="ff-op">−</span><span><b class="chip control" data-k="beta"></b> × popularity</span></li>
        <li><span class="ff-op">+</span><span><b class="chip control" data-k="boost"></b> if in the chosen neighborhood</span></li>
        <li class="ff-pick"><span class="ff-op">top 10 =</span><span>one at a time, the best (1 − ${LAM_MAX}λ) × score − ${LAM_MAX}λ × similarity to the places already picked, <b class="chip diversity" data-k="lam"></b></span></li>
      </ol>`;
    root.querySelector(".help-btn").addEventListener("click", (e) => {
      const open = e.currentTarget.getAttribute("aria-expanded") !== "true";
      e.currentTarget.setAttribute("aria-expanded", open);
      root.querySelector("#help-formula").hidden = !open;
    });
  }
  const f = (v) => v.toFixed(2), set = (k, text) => { root.querySelector(`[data-k="${k}"]`).textContent = text; };
  set("model", modelName);
  set("w_h", f(theta.w_h)); set("alpha", `α ${f(theta.alpha)}`); set("beta", `β ${f(theta.beta)}`);
  set("boost", f(theta.boost || 0)); set("lam", `λ ${f(theta.lam / LAM_MAX)}`);
  root.querySelector("#help-formula").textContent = `The ${modelName} score is split (by SHAP) into a history part, from places like ones this user reviewed, and the rest. `
    + `After the history weight it is rescaled to 0–1 over this user's ${candidates} candidates. Distance is kilometres ÷ 15, at most 1. `
    + `Popularity is the place's percentile among the candidates by reviews in the past year. The score is then rescaled to 0–1 again `
    + `and the top 10 is picked one place at a time; similarity is 0.5 for the same cuisine group plus 0.5 for the same neighborhood, `
    + `counted against the most similar place already picked.`;
}
