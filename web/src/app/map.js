// The app's map: pumpkin land with tofu borders, salad parks, blue water. Restaurants are numbered tomato-red
// pins (real buttons); each person's start point is their avatar. The food stickers never appear here.
import * as d3 from "d3";
import { esc } from "../views/common.js";

const PIN = '<svg aria-hidden="true" viewBox="0 0 32 42"><path d="M16 40.5C16 40.5 2.5 24.5 2.5 15.2A13.5 13.5 0 0 1 29.5 15.2C29.5 24.5 16 40.5 16 40.5Z"/></svg>';
const MAX_SCALE = 420000;   // closest zoom: the frame is never narrower than about 2 km

// handlers: select(i), hover(i | null), drop(lat, lon)
export function createMap(el, nta, avatar, handlers) {
  el.innerHTML = `<svg role="img" aria-label="Map of the listed places and where each person starts"><g class="g-land"></g><g class="g-route"></g></svg><div class="g-over"></div>`;
  const svg = d3.select(el).select("svg"), over = el.querySelector(".g-over");
  const land = nta.features.filter((f) => f.properties.ntatype !== "9"), parks = nta.features.filter((f) => f.properties.ntatype === "9");
  const named = nta.features.filter((f) => f.properties.ntatype === "0").map((f) => ({ name: f.properties.ntaname.split("-")[0].replace(/\s*\(.*?\)/, ""), full: f.properties.ntaname, at: d3.geoCentroid(f), area: d3.geoArea(f) }));
  let ctx = null, projection = null, wide = false;

  function draw() {
    if (!ctx) return;
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    const pts = [...ctx.items.map((p) => [p.lon, p.lat]), ...ctx.starts.map((s) => [s.lon, s.lat])];
    projection = d3.geoMercator();
    const pad = 46;
    if (wide || !pts.length) projection.fitExtent([[8, 8], [w - 8, h - 8]], nta);
    else {
      projection.fitExtent([[pad, pad + 14], [w - pad, h - pad]], { type: "MultiPoint", coordinates: pts });
      if (projection.scale() > MAX_SCALE) {
        const c = d3.geoCentroid({ type: "MultiPoint", coordinates: pts });
        projection.scale(MAX_SCALE).center(c).translate([w / 2, h / 2]);
      }
    }
    const path = d3.geoPath(projection);
    svg.attr("viewBox", `0 0 ${w} ${h}`);
    svg.select(".g-land").selectAll("path").data([...land, ...parks]).join("path")
      .attr("class", (f) => (f.properties.ntatype === "9" ? "land park" : "land")).attr("d", path);
    const sel = ctx.items.find((p) => p.i === ctx.selected);
    svg.select(".g-route").selectAll("path").data(sel ? ctx.starts : []).join("path").attr("class", "route")
      .attr("d", (s) => { const a = projection([s.lon, s.lat]), b = projection([sel.lon, sel.lat]); return `M${a[0]},${a[1]}L${b[0]},${b[1]}`; });

    // neighbourhood names: those that hold a listed place or a start first, then the largest in view; none overlapping
    const inView = ([x, y]) => x > 40 && x < w - 40 && y > 24 && y < h - 24;
    const want = new Set(ctx.items.map((p) => p.nta));
    const cand = named.map((n) => ({ ...n, xy: projection(n.at) })).filter((n) => inView(n.xy))
      .sort((a, b) => (want.has(b.full) - want.has(a.full)) || b.area - a.area);
    const pins = ctx.items.map((p) => projection([p.lon, p.lat])), homes = ctx.starts.map((s) => projection([s.lon, s.lat]));
    // a pin stands on its point and reaches 60px up; an avatar is centred on its point
    const clear = (n) => pins.every((q) => Math.abs(q[0] - n.xy[0]) > 64 || n.xy[1] < q[1] - 72 || n.xy[1] > q[1] + 12)
      && homes.every((q) => Math.abs(q[0] - n.xy[0]) > 74 || Math.abs(q[1] - n.xy[1]) > 38);
    const placed = [];
    for (const n of cand) {
      if (placed.length >= 9) break;
      const far = (q, dx, dy) => Math.abs(q[0] - n.xy[0]) > dx || Math.abs(q[1] - n.xy[1]) > dy;
      if (placed.every((q) => far(q.xy, 96, 26)) && clear(n)) placed.push(n);
    }
    const order = [...ctx.items].sort((a, b) => b.rank - a.rank);   // rank 1 is drawn last, on top
    over.innerHTML = placed.map((n) => `<span class="lbl" style="left:${n.xy[0]}px;top:${n.xy[1]}px">${esc(n.name)}</span>`).join("")
      + ctx.starts.map((s) => { const [x, y] = projection([s.lon, s.lat]); return `<img class="startpin" src="${avatar(s.key)}" alt="${esc(s.name)} starts here" style="left:${x}px;top:${y}px">`; }).join("")
      + order.map((p) => { const [x, y] = projection([p.lon, p.lat]);
        return `<button type="button" class="pin${p.i === ctx.selected ? " sel" : ""}${p.i === ctx.hot ? " hot" : ""}" data-pin="${p.i}" aria-pressed="${p.i === ctx.selected}"
          aria-label="Number ${p.rank}, ${esc(p.name)}, ${esc(p.nta)}" style="left:${x}px;top:${y}px"><span>${PIN}<b>${p.rank}</b></span></button>`; }).join("")
      + (ctx.pinning ? `<div class="mapnote"><span>Tap the map where ${esc(ctx.pinning.name)} starts.</span><button type="button" class="btn quiet" data-maptool="cancel">Cancel</button></div>` : "")
      + `<div class="maptools"><button type="button" class="btn quiet" data-maptool="zoom">${wide ? "Zoom to the list" : "Whole city"}</button>
        ${ctx.starts.map((s) => `<button type="button" class="btn quiet" data-maptool="move:${s.key}">Move ${esc(s.name)}</button>`).join("")}</div>`;
    el.classList.toggle("pinning", !!ctx.pinning);
  }

  el.addEventListener("click", (e) => {
    const pin = e.target.closest("[data-pin]"), tool = e.target.closest("[data-maptool]");
    if (tool) {
      const t = tool.dataset.maptool;
      if (t === "zoom") { wide = !wide; draw(); }
      else if (t === "cancel") handlers.pinMode(null);
      else handlers.pinMode(t.split(":")[1]);
      return;
    }
    if (e.target.closest(".mapnote, .maptools")) return;
    if (ctx?.pinning && projection) {
      const r = el.getBoundingClientRect(), [lon, lat] = projection.invert([e.clientX - r.left, e.clientY - r.top]);
      handlers.drop(lat, lon);
      return;
    }
    if (pin) handlers.select(+pin.dataset.pin);
  });
  el.addEventListener("mouseover", (e) => { const pin = e.target.closest("[data-pin]"); if (pin) handlers.hover(+pin.dataset.pin); });
  el.addEventListener("mouseout", (e) => { if (e.target.closest("[data-pin]")) handlers.hover(null); });
  new ResizeObserver(() => draw()).observe(el);

  return {
    update(next) { ctx = next; draw(); },
    // only the pin states change: no redraw of the land
    mark(selected, hot) {
      if (!ctx) return;
      ctx.hot = hot;
      over.querySelectorAll("[data-pin]").forEach((b) => b.classList.toggle("hot", +b.dataset.pin === hot));
    },
  };
}
