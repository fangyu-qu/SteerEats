// Map (V4): the user's usual area with distance rings, the new top 10 and the places that left, on a street map.
// Leaflet draws OpenStreetMap tiles (loaded from tile.openstreetmap.org when the map is looked at; the page says
// so in its attribution). On top: the neighbourhoods shaded by steering cost, see-through so the streets stay
// readable, and numbered bubbles with a white edge and a shadow for the places. Buttons zoom and reset the view.
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { esc } from "./common.js";

const RINGS_KM = [1, 2, 5];
// Steering-cost bins (L1 change of the four controls); "in list" = already in the model's top 10.
export const COST_BINS = [[0, "in your list now"], [0.25, "≤ 0.25"], [0.5, "≤ 0.5"], [1, "≤ 1"], [Infinity, "> 1"]];
const costClass = (v) => (v == null ? "" : !Number.isFinite(v) ? " unreach" : ` cost-${COST_BINS.findIndex(([t]) => v <= t)}`);
const NYC = [[40.49, -74.27], [40.92, -73.68]];

export function createMap(root, geo, handlers) {
  root.setAttribute("role", "application");
  root.setAttribute("aria-label", "Street map of New York City with the top 10 restaurants");
  const map = L.map(root, { zoomControl: true, zoomSnap: 0.25, minZoom: 9, maxZoom: 18 });
  map.attributionControl.setPrefix('<a href="https://leafletjs.com" target="_blank" rel="noopener">Leaflet</a>');
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19,
    attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors' }).addTo(map);
  map.fitBounds(NYC);
  const codeIdx = new Map(geo.codes.map((code, k) => [code, k]));
  let last = null, pendingFit = null, hovered = null;

  // neighbourhoods: see-through shading by steering cost; a click on one with candidates boosts it
  const ntaLayer = L.geoJSON(geo.nta, { style: { className: "nta", weight: 0.8 },
    onEachFeature: (f, layer) => {
      layer.bindTooltip("", { sticky: true, className: "map-tip", direction: "top", offset: [0, -6] });
      layer.on("click", () => { const k = codeIdx.get(f.properties.nta2020); if (last?.cost.cost.has(k)) handlers.boost(k); });
    } }).addTo(map);
  // the hatching of neighbourhoods the four controls cannot reach
  const overlaySvg = map.getPanes().overlayPane.querySelector("svg");
  overlaySvg?.insertAdjacentHTML("afterbegin", `<defs><pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <line x1="0" y1="0" x2="0" y2="7" class="hatch-line"></line></pattern></defs>`);
  const ringLayer = L.layerGroup().addTo(map), placeLayer = L.layerGroup().addTo(map);
  const marks = new Map();   // candidate index -> marker

  // "Reset view": back to the user's area and places
  const Reset = L.Control.extend({ options: { position: "topright" }, onAdd() {
    const b = L.DomUtil.create("button", "map-reset");
    b.type = "button"; b.textContent = "↺ Reset view";
    L.DomEvent.disableClickPropagation(b);
    b.addEventListener("click", () => fit(true));
    return b;
  } });
  map.addControl(new Reset());
  const note = L.DomUtil.create("div", "map-note", root);
  const NOTE = "Scroll to zoom, drag to pan, click a shaded neighborhood to include it";
  note.textContent = NOTE;

  function styleNta() {
    const cost = last?.cost.cost, how = last?.cost.how;
    ntaLayer.eachLayer((layer) => {
      const p = layer.feature.properties, k = codeIdx.get(p.nta2020), has = cost?.has(k), name = `${p.ntaname}, ${p.boroname}`;
      layer.getElement()?.setAttribute("class", "leaflet-interactive nta" + (has ? " cand" + costClass(cost.get(k)) : "") + (has && last.boostNta === k ? " boosted" : ""));
      let text = `${name}: no candidates for this user`;
      if (has) {
        const v = cost.get(k);
        if (!Number.isFinite(v)) text = `${name}: not reachable with the four controls. Click to boost it directly.`;
        else if (v === 0) text = `${name}: already in the model's top 10`;
        else {
          const h = how.get(k), parts = [];
          if (h.w_h < 1) parts.push(`history ${h.w_h}`); if (h.alpha) parts.push(`α ${h.alpha}`);
          if (h.beta) parts.push(`β ${h.beta}`); if (h.lam) parts.push(`λ ${(h.lam / 0.5).toFixed(2)}`);
          text = `${name}: steering cost ${v.toFixed(2)} (${parts.join(", ")}). Click to boost it directly.`;
        }
      }
      layer.setTooltipContent(esc(text));
    });
  }

  const icon = (cls, html, size) => L.divIcon({ className: `map-mark ${cls}`, html, iconSize: [size, size], iconAnchor: [size / 2, size / 2] });
  function place() {
    if (!last) return;
    const { c, top, gone, home, liked, selected } = last;
    ringLayer.clearLayers(); placeLayer.clearLayers(); marks.clear();
    for (const km of RINGS_KM) {
      L.circle([home.lat, home.lon], { radius: km * 1000, className: "ring", fill: false, interactive: false }).addTo(ringLayer);
      L.marker([home.lat + km / 110.574, home.lon], { icon: L.divIcon({ className: "ring-label", html: `${km} km`, iconSize: [44, 16], iconAnchor: [22, 16] }), interactive: false, keyboard: false }).addTo(ringLayer);
    }
    L.marker([home.lat, home.lon], { icon: icon("home", '<i></i>', 22), interactive: false, keyboard: false, zIndexOffset: 200 }).addTo(ringLayer);
    const add = (i, cls, html, size, title, z) => {
      const m = L.marker([c.candidates.lat[i], c.candidates.lon[i]], { icon: icon(cls, html, size), title, alt: title, zIndexOffset: z, riseOnHover: true }).addTo(placeLayer);
      m.on("mouseover", () => handlers.hover(i)).on("mouseout", () => handlers.hover(null)).on("click", () => handlers.select(i));
      marks.set(i, m);
    };
    gone.forEach((i) => add(i, "gone", "<i></i>", 16, `${c.candidates.name[i]} (left the top 10)`, 300));
    top.forEach((i, r) => add(i, `pin${liked[i] ? " liked" : ""}${i === selected ? " selected" : ""}`, `<i>${r + 1}</i>`, 28,
      `${r + 1}. ${c.candidates.name[i]}, ${c.candidates.distance_km[i].toFixed(1)} km`, 600 - r));
    highlight(hovered);
  }

  function fit(animate) {
    if (!last) return;
    if (!root.clientWidth || !root.clientHeight) { pendingFit = true; return; }
    const { c, top, gone, home } = last;
    const pts = [[home.lat, home.lon], ...top.concat(gone).map((i) => [c.candidates.lat[i], c.candidates.lon[i]])];
    map.fitBounds(L.latLngBounds(pts), { padding: [46, 46], maxZoom: 15, animate });
  }

  function highlight(id) {
    hovered = id;
    root.classList.toggle("has-hover", id != null && marks.has(id));
    for (const [i, m] of marks) m.getElement()?.classList.toggle("hover", i === id);
  }

  // a hidden section has no size: Leaflet measures again when the map is shown or resized
  let w = 0, h = 0;
  new ResizeObserver(() => {
    if (!root.clientWidth || !root.clientHeight) return;
    const shown = !w || !h;
    if (root.clientWidth !== w || root.clientHeight !== h) { w = root.clientWidth; h = root.clientHeight; map.invalidateSize({ pan: !shown }); }
    if (pendingFit || shown) { pendingFit = null; requestAnimationFrame(() => { map.invalidateSize({ pan: false }); fit(false); }); }
  }).observe(root);

  return {
    update(data, { refit }) {
      last = data;
      styleNta();
      place();
      if (refit) fit(true);
    },
    setBrushed() {},
    // the console's "Choose on map": the map says what to click until a neighbourhood is chosen
    pick(on) {
      root.classList.toggle("picking", on);
      note.textContent = on ? "Click a shaded neighborhood to include one of its places" : NOTE;
    },
    highlight,
    note,
  };
}

export const describePlace = (c, i) => `${esc(c.candidates.name[i])}`;
