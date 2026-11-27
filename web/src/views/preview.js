// Console previews (V1): a small chart under each control. Across: the control's range, aligned with its
// slider (or its options). Up: the share of the model's predicted visits for this user that the top 10 keeps
// (100% = the model's own list), on one scale for all four charts. Grey where the target cannot be met.
// Hover reads out a value; click sets the control to it.
import * as d3 from "d3";
import { CONTROL_INFO, hueVar, hueLine } from "./common.js";

const H = 40, T = 5, B = 7;      // chart height, top and bottom padding of the plot band
const HUE = { w_h: CONTROL_INFO.w_h.hue, km: CONTROL_INFO.alpha.hue, popularity: CONTROL_INFO.beta.hue, cuisines: CONTROL_INFO.lam.hue };
const ord = (n) => `${n}${[11, 12, 13].includes(n % 100) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" })[n % 10] ?? "th"}`;
const LABEL = {
  w_h: (v) => `history ${Math.round(v * 100)}%`,
  km: (v) => `within ${v} km`,
  popularity: (v) => (v == null ? "no preference" : `below the ${ord(Math.round(v * 100))} pct`),
  cuisines: (v) => (v == null ? "no preference" : `at least ${v} groups`),
};
const TICK = { popularity: (v) => (v == null ? "any" : `${Math.round(v * 100)}`), cuisines: (v) => (v == null ? "any" : `${v}`) };
const pct = (v) => (v < 1 && Math.round(v * 100) >= 100 ? `${Math.floor(v * 1000) / 10}%` : `${Math.round(v * 100)}%`);

// current: the control's value now (undefined when the control is off); onPick(key, value) sets it.
export function renderPreviews(root, sweeps, current, onPick, { stale = false, boost = false } = {}) {
  const note = root.querySelector("#spark-note");
  if (note) note.textContent = boost ? "The small charts leave out the neighborhood boost." : "";
  if (!sweeps) return;
  // one scale for all four charts: from just below the lowest value (and at least 20 points of range) to 100%
  const lo = Math.min(0.8, Math.floor((d3.min(Object.values(sweeps).flat(), (d) => d.kept) - 0.02) * 20) / 20);
  for (const [key, pts] of Object.entries(sweeps)) {
    const box = root.querySelector(`#spark-${key}`);
    if (!box) continue;
    box.classList.toggle("stale", stale);
    const width = box.clientWidth || 264, discrete = key in TICK, h = H + (discrete ? 12 : 0);
    const x = discrete ? d3.scalePoint(pts.map((_, k) => k), [10, width - 10]) : d3.scaleLinear(d3.extent(pts, (d) => d.v), [8, width - 8]);
    const xv = (d, k) => (discrete ? x(k) : x(d.v));
    const y = d3.scaleLinear([lo, 1], [H - B, T]);
    const svg = d3.select(box.querySelector(".sp-svg")).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", h)
      .attr("role", "img").attr("aria-label", `Predicted visits kept across ${key === "w_h" ? "history weight" : key} settings`);
    svg.selectAll("*").remove();
    svg.append("line").attr("class", "sp-base").attr("x1", 0).attr("x2", width).attr("y1", y(1)).attr("y2", y(1));
    const line = d3.line().x((d) => d.x).y((d) => d.y).curve(discrete ? d3.curveLinear : d3.curveMonotoneX);
    const P = pts.map((d, k) => ({ ...d, k, x: xv(d, k), y: y(d.kept) }));
    svg.append("path").attr("class", "sp-all").attr("d", line(P));
    // solid in the control's hue where the target is met
    svg.append("path").attr("class", "sp-met").style("stroke", hueLine(HUE[key])).attr("d", line.defined((d) => d.met)(P));
    if (discrete) {
      svg.append("g").selectAll("circle").data(P).join("circle").attr("cx", (d) => d.x).attr("cy", (d) => d.y).attr("r", 2.6)
        .attr("class", (d) => (d.met ? "sp-dot" : "sp-dot miss")).style("fill", (d) => (d.met ? hueVar(HUE[key]) : null));
      svg.append("g").selectAll("text").data(P).join("text").attr("class", "sp-tick").attr("x", (d) => d.x).attr("y", h - 1)
        .attr("text-anchor", "middle").text((d) => TICK[key](d.v));
    }
    const cur = current[key] === undefined ? null : P.find((d) => d.v === current[key]) ?? null;
    const mark = svg.append("g").attr("class", "sp-mark");
    // readout floats under the chart while hovering, so the console does not shift
    const read = box.querySelector(".spark-read");
    const say = (d) => {
      read.hidden = !d;
      if (!d) return;
      const o = d.outcomes;
      read.innerHTML = `${LABEL[key](d.v)}: ${d.met ? `keeps <b>${pct(d.kept)}</b>` : `<span class="miss">not reachable</span>, keeps ${pct(d.kept)}`}`
        + `<br><span class="sp-o">${o.km.toFixed(1)} km · ${ord(Math.round(o.popularity * 100))} pct popularity · ${o.cuisines} cuisine groups</span>`;
    };
    const place = (d, cls) => {
      mark.selectAll(`.${cls}`).remove();
      if (!d) return;
      mark.append("line").attr("class", `${cls} sp-guide`).attr("x1", d.x).attr("x2", d.x).attr("y1", 1).attr("y2", H - 2);
      mark.append("circle").attr("class", `${cls} sp-cur`).attr("cx", d.x).attr("cy", d.y).attr("r", 3.6);
    };
    place(cur, "now");
    say(null);
    const nearest = (px) => P[d3.minIndex(P, (d) => Math.abs(d.x - px))];
    svg.append("rect").attr("class", "sp-hit").attr("width", width).attr("height", h)
      .on("pointermove", (e) => { const d = nearest(d3.pointer(e)[0]); place(d, "hov"); say(d); })
      .on("pointerleave", () => { place(null, "hov"); say(null); })
      .on("click", (e) => onPick(key, nearest(d3.pointer(e)[0]).v));
  }
}
