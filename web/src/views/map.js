// Map (V4): NYC neighbourhoods, the user's usual area with distance rings, the new top 10 and the places that left.
// The basemap scales with zoom; markers stay a constant size and are repositioned on every zoom step.
import * as d3 from "d3";
import { esc } from "./common.js";

const RINGS_KM = [1, 2, 5];
// Steering-cost bins (L1 change of the four controls); "in list" = already in the model's top 10.
export const COST_BINS = [[0, "in your list now"], [0.25, "≤ 0.25"], [0.5, "≤ 0.5"], [1, "≤ 1"], [Infinity, "> 1"]];
const costClass = (v) => (v == null ? "" : !Number.isFinite(v) ? " unreach" : ` cost-${COST_BINS.findIndex(([t]) => v <= t)}`);

export function createMap(root, geo, handlers) {
  const svg = d3.select(root).append("svg").attr("role", "img").attr("aria-label", "Map of New York City neighbourhoods with the top 10 restaurants");
  const base = svg.append("g").attr("class", "base");
  const ringsG = base.append("g");
  const overlay = svg.append("g").attr("class", "overlay");
  const note = d3.select(root).append("div").attr("class", "map-note").text("Scroll to zoom, drag to pan, click a neighborhood to boost it");
  const codeIdx = new Map(geo.codes.map((code, k) => [code, k]));
  svg.append("defs").html(`<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <line x1="0" y1="0" x2="0" y2="6" class="hatch-line"></line></pattern>`);
  let projection, path, transform = d3.zoomIdentity, last = null, w = 0, h = 0;

  const zoom = d3.zoom().scaleExtent([1, 40]).on("zoom", (e) => {
    transform = e.transform;
    base.attr("transform", transform);
    svg.select("#hatch").attr("patternTransform", `scale(${1 / transform.k}) rotate(45)`);   // keep hatching fine at any zoom
    place();
  });
  svg.call(zoom);

  function layout() {
    w = root.clientWidth; h = root.clientHeight;
    svg.attr("viewBox", `0 0 ${w} ${h}`);
    projection = d3.geoMercator().fitExtent([[8, 8], [w - 8, h - 8]], geo.boroughs);
    path = d3.geoPath(projection);
    base.selectAll("path.nta").data(geo.nta.features).join("path").attr("class", "nta").attr("d", path)
      .on("click", (e, d) => { const k = codeIdx.get(d.properties.nta2020); if (last?.cost.cost.has(k)) handlers.boost(k); })
      .append("title");
    styleNta();
    base.selectAll("path.boro").data(geo.boroughs.features).join("path").attr("class", "boro").attr("d", path);
    ringsG.raise();
  }

  const xy = (lon, lat) => transform.apply(projection([lon, lat]));

  function styleNta() {
    const cost = last?.cost.cost, how = last?.cost.how;
    base.selectAll("path.nta")
      .attr("class", (d) => {
        const k = codeIdx.get(d.properties.nta2020);
        const has = cost?.has(k);
        return "nta" + (d.properties.ntatype === "9" ? " park" : "") + (has ? " cand" + costClass(cost.get(k)) : "")
          + (has && last.boostNta === k ? " boosted" : "");
      })
      .select("title").text((d) => {
        const k = codeIdx.get(d.properties.nta2020), name = `${d.properties.ntaname}, ${d.properties.boroname}`;
        if (!cost?.has(k)) return `${name}: no candidates for this user`;
        const v = cost.get(k);
        if (!Number.isFinite(v)) return `${name}: not reachable with the four controls. Click to boost it directly.`;
        if (v === 0) return `${name}: already in the model's top 10`;
        const h = how.get(k), parts = [];
        if (h.w_h < 1) parts.push(`history ${h.w_h}`); if (h.alpha) parts.push(`α ${h.alpha}`);
        if (h.beta) parts.push(`β ${h.beta}`); if (h.lam) parts.push(`λ ${(h.lam / 0.5).toFixed(2)}`);
        return `${name}: steering cost ${v.toFixed(2)} (${parts.join(", ")}). Click to boost it directly.`;
      });
  }

  function place() {
    if (!last) return;
    const data = last;
    const { c, top, gone, home, liked } = last;
    overlay.selectAll("g.home").data([home]).join((e) => {
      const g = e.append("g").attr("class", "home");
      g.append("circle").attr("class", "halo").attr("r", 11);
      g.append("circle").attr("class", "dot").attr("r", 5);
      return g;
    }).attr("transform", (d) => `translate(${xy(d.lon, d.lat)})`);
    overlay.selectAll("text.ring-label").data(RINGS_KM).join("text").attr("class", "ring-label")
      .attr("text-anchor", "middle").attr("dy", "-0.3em")
      .attr("transform", (km) => `translate(${xy(home.lon, home.lat + km / 110.574)})`).text((km) => `${km} km`);
    const goneSel = overlay.selectAll("g.gone").data(gone, (i) => i).join((e) => {
      const g = e.append("g").attr("class", "gone");
      g.append("circle").attr("r", 5);
      g.append("title");
      return g;
    });
    goneSel.attr("data-id", (i) => i).attr("transform", (i) => `translate(${xy(c.candidates.lon[i], c.candidates.lat[i])})`)
      .on("pointerenter", (_, i) => handlers.hover(i)).on("pointerleave", () => handlers.hover(null))
      .on("click", (_, i) => handlers.select(i));
    goneSel.select("title").text((i) => `${c.candidates.name[i]} (left the top 10)`);
    const pins = overlay.selectAll("g.pin").data(top, (i) => i).join((e) => {
      const g = e.append("g").attr("class", "pin");
      g.append("circle").attr("r", 9);
      g.append("text");
      g.append("title");
      return g;
    });
    pins.attr("data-id", (i) => i).classed("liked", (i) => liked[i])
      .attr("transform", (i) => `translate(${xy(c.candidates.lon[i], c.candidates.lat[i])})`)
      .on("pointerenter", (_, i) => handlers.hover(i)).on("pointerleave", () => handlers.hover(null))
      .on("click", (_, i) => handlers.select(i)).classed("selected", (i) => i === data.selected);
    pins.select("text").text((i) => top.indexOf(i) + 1);
    pins.select("title").text((i) => `${top.indexOf(i) + 1}. ${c.candidates.name[i]}, ${c.candidates.distance_km[i].toFixed(1)} km`);
    pins.raise();
  }

  let pendingFit = null;
  new ResizeObserver(() => {
    if (!root.clientWidth || !root.clientHeight) return;
    if (root.clientWidth !== w || root.clientHeight !== h) { layout(); place(); }
    if (pendingFit) { const p = pendingFit; pendingFit = null; fitTo(p); }
  }).observe(root);

  function fitTo(points) {
    if (!w || !h) { pendingFit = points; return; }
    const xs = points.map((p) => projection(p));
    const [x0, x1] = d3.extent(xs, (p) => p[0]), [y0, y1] = d3.extent(xs, (p) => p[1]);
    const pad = 48;
    const k = Math.min(24, 0.9 / Math.max((x1 - x0 + pad) / w, (y1 - y0 + pad) / h));
    const t = d3.zoomIdentity.translate(w / 2, h / 2).scale(Math.max(1, k)).translate(-(x0 + x1) / 2, -(y0 + y1) / 2);
    svg.transition().duration(450).ease(d3.easeCubicOut).call(zoom.transform, t);
  }

  return {
    update(data, { refit }) {
      if (!projection || (root.clientWidth && (root.clientWidth !== w || root.clientHeight !== h))) layout();
      last = data;
      styleNta();
      const { c, top, gone, home } = data;
      ringsG.selectAll("path.ring").data(RINGS_KM).join("path").attr("class", "ring")
        .attr("d", (km) => path(d3.geoCircle().center([home.lon, home.lat]).radius(km / 111.32)()));
      place();
      if (refit) {
        const pts = [[home.lon, home.lat], ...top.concat(gone).map((i) => [c.candidates.lon[i], c.candidates.lat[i]])];
        fitTo(pts);
      }
    },
    highlight(id) {
      root.classList.toggle("has-hover", id != null);
      overlay.selectAll("g.pin, g.gone").classed("hover", (i) => i === id);
      if (id != null) overlay.selectAll("g.pin, g.gone").filter((i) => i === id).raise();
    },
    note,
  };
}

export const describePlace = (c, i) => `${esc(c.candidates.name[i])}`;
