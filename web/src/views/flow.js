// Rank flow (V2): a bump chart from the model's default ranking through each steering step.
// Lines keep their identity across steps (object constancy); the step that moved a restaurant colors its segment.
import * as d3 from "d3";
import { STEPS, hueVar, truncate } from "./common.js";

const ROW = 26, TOP = 46, TOPN = 10;

export function renderFlow(root, data, handlers) {
  const { c, ranks, stepNotes } = data;          // ranks: { default, model, distance, popularity, diversity } (1-based)
  const width = Math.max(root.clientWidth, 640);
  const left = 196, right = 200;
  const belowY = TOP + TOPN * ROW + 22;
  const height = belowY + 34;
  const x = d3.scalePoint(STEPS.map((s) => s.key), [left, width - right]);
  const y = (r) => (r <= TOPN ? TOP + (r - 0.5) * ROW : belowY + Math.min(r - TOPN - 1, 30) * 0.35);

  const n = c.candidates.name.length;
  const ids = [];
  for (let i = 0; i < n; i++) if (STEPS.some((s) => ranks[s.key][i] <= TOPN)) ids.push(i);

  const svg = d3.select(root).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", height)
    .attr("role", "img").attr("aria-label", "Bump chart of the top 10 across the steering steps");

  // Column headers and guides
  const cols = svg.selectAll("g.col").data(STEPS, (d) => d.key).join((e) => {
    const g = e.append("g").attr("class", "col");
    g.append("line").attr("class", "guide");
    g.append("text").attr("class", "col-label").attr("text-anchor", "middle");
    g.append("text").attr("class", "col-sub").attr("text-anchor", "middle");
    return g;
  });
  cols.select("line").attr("x1", (d) => x(d.key)).attr("x2", (d) => x(d.key)).attr("y1", TOP - 6).attr("y2", belowY + 10);
  cols.select(".col-label").attr("x", (d) => x(d.key)).attr("y", 14).text((d) => d.label);
  cols.select(".col-sub").attr("x", (d) => x(d.key)).attr("y", 30).text((d) => stepNotes[d.key]);

  const ticks = svg.selectAll("text.rank-tick").data(d3.range(1, TOPN + 1)).join("text").attr("class", "rank-tick")
    .attr("x", 6).attr("y", (r) => y(r)).attr("dy", "0.35em").text((r) => r);
  ticks.raise();
  svg.selectAll("text.below-label").data([0]).join("text").attr("class", "below-label")
    .attr("x", 6).attr("y", belowY).attr("dy", "0.35em").text("below 10");

  // Items
  const link = (a, b) => {
    const xa = x(a.key), xb = x(b.key);
    return (i) => { const ya = y(ranks[a.key][i]), yb = y(ranks[b.key][i]), mx = (xa + xb) / 2;
      return `M${xa},${ya} C${mx},${ya} ${mx},${yb} ${xb},${yb}`; };
  };
  const segs = STEPS.slice(1).map((s, k) => ({ from: STEPS[k], to: s, path: link(STEPS[k], s) }));
  const items = svg.selectAll("g.item").data(ids, (i) => i).join((e) => {
    const g = e.append("g").attr("class", "item");
    segs.forEach(() => g.append("path").attr("class", "seg"));
    g.append("g").attr("class", "dots");
    g.append("text").attr("class", "name left").attr("text-anchor", "end").attr("dy", "0.35em");
    g.append("text").attr("class", "name right").attr("text-anchor", "start").attr("dy", "0.35em");
    return g;
  });
  items.attr("data-id", (i) => i).classed("selected", (i) => i === data.selected)
    .on("pointerenter", (_, i) => handlers.hover(i)).on("pointerleave", () => handlers.hover(null))
    .on("click", (_, i) => handlers.select?.(i));
  items.each(function (i) {
    const g = d3.select(this);
    g.selectAll("path.seg").data(segs).attr("d", (s) => s.path(i)).attr("stroke", (s) => {
      const a = ranks[s.from.key][i], b = ranks[s.to.key][i];
      if (a > TOPN && b > TOPN) return "var(--line)";
      return a !== b ? hueVar(s.to.hue) : "var(--track)";
    });
    g.select(".dots").selectAll("circle").data(STEPS).join("circle").attr("cx", (s) => x(s.key))
      .attr("cy", (s) => y(ranks[s.key][i])).attr("r", (s) => (s.key === "diversity" && ranks.diversity[i] <= TOPN ? 4.5 : 3))
      .attr("fill", (s) => (s.key === "diversity" && ranks.diversity[i] <= TOPN ? "var(--ink)" : "var(--panel)"))
      .attr("stroke", (s) => (ranks[s.key][i] <= TOPN ? "var(--ink-2)" : "var(--line-strong)")).attr("stroke-width", 1.5);
    const d0 = ranks.default[i], d1 = ranks.diversity[i];
    g.select(".name.left").attr("x", left - 10).attr("y", y(d0)).text(d0 <= TOPN ? truncate(c.candidates.name[i], 21) : "")
      .classed("muted", d1 > TOPN);
    g.select(".name.right").attr("x", width - right + 10).attr("y", y(d1)).text(d1 <= TOPN ? truncate(c.candidates.name[i], 26) : "");
  });
}

export function highlightFlow(root, id) {
  root.classList.toggle("has-hover", id != null);
  root.querySelectorAll("g.item").forEach((g) => {
    const on = id != null && +g.dataset.id === id;
    g.classList.toggle("hover", on);
    if (on) g.parentNode.appendChild(g);
  });
}
