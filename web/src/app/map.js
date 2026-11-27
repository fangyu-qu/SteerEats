// The app's map: a street map that zooms from the whole city down to street and building names (Leaflet on
// OpenStreetMap tiles, fetched from tile.openstreetmap.org while the map is looked at). Restaurants are numbered
// deep-coral pins; each person's start point is their avatar; a dashed line runs from each start to the picked place.
// The food stickers never appear here.
// Moving a start is done on the avatar itself, three ways that all end in the same handler:
//   drag it: the dashed line follows while it moves, and the list is worked out again when it is let go;
//   tap it: it is "held", and every tap on the map moves it there, as often as wanted, until it is tapped again,
//           Done is pressed or Esc is hit;
//   keyboard: Tab to the avatar, Enter to hold it, arrow keys to nudge it, Enter or Esc to let go.
// After a move the view stays where it is unless a pin has gone out of sight.
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { esc } from "../views/common.js";

const PIN = '<svg aria-hidden="true" viewBox="0 0 32 42"><path d="M16 40.5C16 40.5 2.5 24.5 2.5 15.2A13.5 13.5 0 0 1 29.5 15.2C29.5 24.5 16 40.5 16 40.5Z"/></svg>';
const NYC = [[40.49, -74.27], [40.92, -73.68]];
const CLOSEST = 16;   // the view first fits the list no closer than this; the person can zoom in to 19
const NUDGE = 28;     // one arrow key moves a held avatar this many screen pixels

// handlers: select(i), hover(i | null), move(key, lat, lon) -> false when the point is refused, hold(key | null)
export function createMap(el, nta, avatar, handlers) {
  el.setAttribute("role", "application");
  el.setAttribute("aria-label", "Street map of the listed places and where each person starts");
  const map = L.map(el, { zoomControl: true, zoomSnap: 0.25, minZoom: 9, maxZoom: 19 });
  map.attributionControl.setPrefix('<a href="https://leafletjs.com" target="_blank" rel="noopener">Leaflet</a>');
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19,
    attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors' }).addTo(map);
  map.fitBounds(NYC);
  const routes = L.layerGroup().addTo(map), people = L.layerGroup().addTo(map), places = L.layerGroup().addTo(map);
  const pins = new Map(), routeOf = new Map(), startOf = new Map();   // place index -> marker; person -> line, marker
  let ctx = null, wide = false, fitKey = null, pendingFit = false, moved = false, refocus = null, refused = null;

  // the tools sit on the map and never pass their clicks to it
  const tools = L.DomUtil.create("div", "maptools", el), noteBox = L.DomUtil.create("div", "mapnote-wrap", el);
  for (const box of [tools, noteBox]) { L.DomEvent.disableClickPropagation(box); L.DomEvent.disableScrollPropagation(box); }
  // (a tool redraws itself when pressed, so its click is stopped here: once the pressed button has left the page,
  // Leaflet can no longer tell that the click came from a control and would read it as a tap on the map)
  tools.addEventListener("click", (e) => {
    e.stopPropagation();
    if (e.target.closest('[data-maptool="zoom"]')) { wide = !wide; fit(true); drawTools(); }
  });
  noteBox.addEventListener("click", (e) => { e.stopPropagation(); if (e.target.closest('[data-maptool="done"]')) handlers.hold(null); });
  // a held avatar goes wherever the map is tapped, again and again
  map.on("click", (e) => { if (ctx?.pinning) place(ctx.pinning.key, e.latlng.lat, e.latlng.lng); });

  // every way of moving a start ends here; a refused point (outside the city) says so for a moment
  function place(key, lat, lng) {
    moved = true;
    if (handlers.move(key, lat, lng) === false) {
      moved = false;
      clearTimeout(refused); el.classList.add("refused"); drawTools("That is too far from New York City. Pick a start closer in.");
      refused = setTimeout(() => { el.classList.remove("refused"); drawTools(); }, 2600);
    }
  }

  function drawTools(warn) {
    if (!ctx) return;
    const names = ctx.starts.map((s) => esc(s.name));
    tools.innerHTML = `<button type="button" class="btn quiet" data-maptool="zoom">${wide ? "Zoom to the list" : "Whole city"}</button>
      ${ctx.pinning || !names.length ? "" : `<span class="maphint">Drag ${names.join(" or ")} to move ${names.length > 1 ? "their" : "the"} start</span>`}`;
    noteBox.innerHTML = warn ? `<div class="mapnote warn" role="status"><span>${warn}</span></div>`
      : ctx.pinning ? `<div class="mapnote" role="status"><span>Tap the map to move ${esc(ctx.pinning.name)}, as often as you like. You can also drag.</span><button type="button" class="btn quiet" data-maptool="done">Done</button></div>` : "";
    el.classList.toggle("pinning", !!ctx.pinning);
  }

  const points = () => [...ctx.items.map((p) => [p.lat, p.lon]), ...ctx.starts.map((s) => [s.lat, s.lon])];
  function fit(animate) {
    if (!ctx) return;
    if (!el.clientWidth || !el.clientHeight) { pendingFit = true; return; }
    const pts = points();
    if (wide || !pts.length) map.fitBounds(NYC, { animate });
    else map.fitBounds(L.latLngBounds(pts), { paddingTopLeft: [44, 64], paddingBottomRight: [44, 64], maxZoom: CLOSEST, animate });
  }
  // after a move the view is left alone, unless a pin or a start is now out of sight: then it glides to show them
  function keepInSight() {
    const pts = points();
    if (!pts.length || !el.clientWidth) return;
    if (!map.getBounds().pad(-0.06).contains(L.latLngBounds(pts))) fit(true);
  }

  function draw() {
    if (!ctx) return;
    routes.clearLayers(); people.clearLayers(); places.clearLayers(); pins.clear(); routeOf.clear(); startOf.clear();
    const sel = ctx.items.find((p) => p.i === ctx.selected);
    for (const s of ctx.starts) {
      if (sel) routeOf.set(s.key, L.polyline([[s.lat, s.lon], [sel.lat, sel.lon]], { className: "route", interactive: false }).addTo(routes));
      const held = ctx.pinning?.key === s.key;
      const m = L.marker([s.lat, s.lon], { icon: L.divIcon({ className: `startpin${held ? " held" : ""}`, html: `<img src="${avatar(s.key)}" alt="">`, iconSize: [52, 52], iconAnchor: [26, 26] }),
        draggable: true, autoPan: true, autoPanPadding: [40, 40], keyboard: true, riseOnHover: true, riseOffset: 900, zIndexOffset: held ? 1200 : 100,
        title: held ? `${s.name}: tap the map or use the arrow keys to move` : `${s.name} starts here: drag to move, or tap to hold` }).addTo(people);
      const node = m.getElement();
      node?.setAttribute("aria-label", `${s.name}'s start. ${held ? "Held: arrow keys move it, Enter lets go." : "Press Enter to hold it, then move it with the arrow keys."}`);
      node?.setAttribute("aria-pressed", String(held));
      m.on("dragstart", () => { node?.classList.add("dragging"); })
        .on("drag", (e) => { const at = e.target.getLatLng(); if (sel) routeOf.get(s.key)?.setLatLngs([[at.lat, at.lng], [sel.lat, sel.lon]]); })   // the line follows the avatar
        .on("dragend", (e) => { const at = e.target.getLatLng(); place(s.key, at.lat, at.lng); })
        .on("click", () => { refocus = held ? null : s.key; handlers.hold(held ? null : s.key); });   // a tap (not a drag): hold it, or let it go
      // keyboard: arrows nudge a held avatar; the map itself must not pan under it
      node?.addEventListener("keydown", (e) => {
        const step = { ArrowLeft: [-NUDGE, 0], ArrowRight: [NUDGE, 0], ArrowUp: [0, -NUDGE], ArrowDown: [0, NUDGE] }[e.key];
        if (e.key === "Escape" && held) { e.stopPropagation(); refocus = s.key; handlers.hold(null); return; }
        // Enter or Space holds the avatar or lets it go (handled here, and the key's own press is cancelled, so the
        // map library does not add a click of its own on top)
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); refocus = s.key; handlers.hold(held ? null : s.key); return; }
        if (!step || !held) return;
        e.preventDefault(); e.stopPropagation();
        const to = map.containerPointToLatLng(map.latLngToContainerPoint([s.lat, s.lon]).add(step));
        refocus = s.key; place(s.key, to.lat, to.lng);
      });
      startOf.set(s.key, m);
    }
    for (const p of [...ctx.items].sort((a, b) => b.rank - a.rank)) {   // rank 1 is added last, on top
      const m = L.marker([p.lat, p.lon], { icon: L.divIcon({ className: `pin${p.i === ctx.selected ? " sel" : ""}${p.i === ctx.hot ? " hot" : ""}`, html: `<span>${PIN}<b>${p.rank}</b></span>`, iconSize: [34, 44], iconAnchor: [17, 44] }),
        zIndexOffset: 400 - p.rank + (p.i === ctx.selected ? 400 : 0), title: `${p.rank}. ${p.name}` }).addTo(places);
      // while an avatar is held, a tap on a pin moves the avatar there too (the map's own click does not see it)
      m.on("click", () => { if (ctx.pinning) place(ctx.pinning.key, p.lat, p.lon); else handlers.select(p.i); }).on("mouseover", () => handlers.hover(p.i)).on("mouseout", () => handlers.hover(null));
      const node = m.getElement();
      node?.setAttribute("aria-label", `Number ${p.rank}, ${p.name}, ${p.nta}`);
      node?.setAttribute("aria-pressed", String(p.i === ctx.selected));
      pins.set(p.i, m);
    }
    drawTools();
    if (refocus) { startOf.get(refocus)?.getElement()?.focus({ preventScroll: true }); refocus = null; }
    // the view moves only when the places or the start points change, so picking a place keeps the person's zoom;
    // and after a move it stays put as long as everything is still in sight
    const key = JSON.stringify([ctx.items.map((p) => [p.lat, p.lon]), ctx.starts.map((s) => [s.lat, s.lon])]);
    if (key !== fitKey) { fitKey = key; if (moved) keepInSight(); else fit(false); }
    moved = false;
  }

  let w = 0, h = 0;
  new ResizeObserver(() => {
    if (!el.clientWidth || !el.clientHeight) return;
    // coming out of hiding (the phone's list view) the map has no old centre worth keeping: measure, then fit on the
    // next frame, once Leaflet has settled; a later change of size keeps the centre
    const shown = !w || !h;
    if (el.clientWidth !== w || el.clientHeight !== h) { w = el.clientWidth; h = el.clientHeight; map.invalidateSize({ pan: !shown }); }
    if (pendingFit || shown) { pendingFit = false; requestAnimationFrame(() => { map.invalidateSize({ pan: false }); fit(false); }); }
  }).observe(el);

  return {
    update(next) { ctx = next; draw(); },
    // only the pin states change
    mark(selected, hot) {
      if (!ctx) return;
      ctx.hot = hot;
      for (const [i, m] of pins) m.getElement()?.classList.toggle("hot", i === hot);
    },
  };
}
