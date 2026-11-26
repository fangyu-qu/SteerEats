// Eat Buddy, the diner app. One or two people each bring a taste (a sample one, or one built from places they
// like when the server is connected), say what they feel like, and get six places with the reason each one
// suits each of them. Layers: App (this list and map), Why (the drawer and
// "How is this ranked?"), Lab (lab.html). All ranking happens in engine.js; a sentence only sets settings.
import * as d3 from "d3";
import { K, NO_TARGETS, STRATEGY, buildPool, mainReason, memberScores, prepare, solve } from "./engine.js";
import { TAU } from "../group.js";
import { BOROUGHS, CUISINES, HISTORY, MAX_SENTENCE, NEEDS, POPULARITY, chips as nlChips, emptyFilters, emptySlots, mergeFilters, withoutChip } from "../nl.js";
import { WHY, apiStatus, readSentence, scoreParty, searchPlaces } from "../nlClient.js";
import { esc } from "../views/common.js";
import { createMap } from "./map.js";
import { drawerHTML } from "./place.js";

const BASE = import.meta.env.BASE_URL, DATA = `${BASE}data/`;
const avatar = (key) => `${BASE}illustrations/avatars/${key}-cutout.png`;
const VARIETY = { some: 4, lots: 6 };                       // cuisine groups among the six
const NYC = { lat: [40.45, 40.95], lon: [-74.3, -73.65] };
const COUNT = ["No place", "One", "Two", "Three", "Four", "Five", "Six"];
const YOU = { Collaborative: "People with visits like yours come here", Taste: "It matches your taste", Popularity: "Lots of recent reviews",
  Rating: "Reviewers rate it highly", Location: "It is in your part of town", Attributes: "It offers what you usually look for" };
const THEM = { ...YOU, Collaborative: "People with visits like theirs come here", Taste: "It matches their taste", Location: "It is in their part of town", Attributes: "It offers what they usually look for" };
const CARD = { Collaborative: (n) => `people like ${n} come here`, Taste: (n) => `matches ${n}’s taste`, Popularity: (n) => `busy lately for ${n}`,
  Rating: (n) => `its ratings count for ${n}`, Location: (n) => `in ${n}’s part of town`, Attributes: (n) => `offers what ${n} looks for` };
const NONE = "No single reason stands out";
const MIN_LIKED = 3, MAX_LIKED = 30;                        // places needed for a taste of your own
const STORE = "eatbuddy.liked.v1";                          // the picked places, kept in this browser only
const FAIL = { loading: "The server is still starting. Try again in a few seconds.", busy: "The server is busy. Try again in a moment.",
  down: "The server did not answer." };

// D.base is the exported place table (sample tastes). D.places is the table in use: D.base, or the places the
// server sent for a party with a live taste. D.applied is the party the list is computed for.
const D = { index: null, base: null, places: null, nta: null, detailMeta: null, cold: null, profiles: new Map(), detail: new Map(), themes: new Map(), areas: [], applied: null };
const state = {
  screen: "start", two: true, api: null, busy: false, notice: null, search: [{ q: "", results: null, note: "" }, { q: "", results: null, note: "" }],
  members: [{ key: "tomato", name: "Tomato", profileId: null, start: null, source: "sample", liked: [], mine: null },
    { key: "tofu", name: "Tofu", profileId: null, start: null, source: "sample", liked: [], mine: null }],
  strategy: "fair", w_h: 1, targets: { ...NO_TARGETS }, filters: { ...emptyFilters(), min_rating: null },
  selected: null, picked: null, sheet: null, view: "list", pinning: null, draft: null, saying: false, hot: null, menu: null,
  open: {}, compareWith: null, gps: {}, r: null,
};
const root = document.getElementById("app");
const $ = (id) => document.getElementById(id);
const json = (path) => fetch(DATA + path).then((r) => { if (!r.ok) throw new Error(`Could not load ${path}`); return r.json(); });
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
const label = (m, k) => (k === 0 ? `${m.name} (you)` : m.name);
const profileOf = (id) => D.index.profiles.find((p) => p.id === id);
const sticker = (i) => `${BASE}illustrations/cuisine/${D.index.cuisine_names[D.places.cuisine[i]]}.jpg`;
const facts = (i) => { const P = D.places; return `${P.rating[i].toFixed(1)} stars · ${P.price[i] ? "$".repeat(P.price[i]) : "price not listed"} · ${CUISINES[D.index.cuisine_names[P.cuisine[i]]]} · ${P.nta_name[i].replace(/-/g, ", ")}`; };

// ------------------------------------------------------------------------------------------------ data
async function boot() {
  root.innerHTML = `<p class="start" role="status">Setting the table.</p>`;
  [D.index, D.base, D.nta, D.detailMeta, D.cold, state.api] = await Promise.all([json("app/index.json"), json("app/places.json"), json("nyc_nta.geojson"), json("detail/index.json"),
    fetch(`${DATA}cold_start.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null), apiStatus()]);
  D.places = D.base;
  D.areas = D.nta.features.filter((f) => f.properties.ntatype === "0")
    .map((f) => ({ code: f.properties.nta2020, name: f.properties.ntaname, borough: f.properties.boroname, at: d3.geoCentroid(f), f }))
    .sort((a, b) => a.borough.localeCompare(b.borough) || a.name.localeCompare(b.name));
  for (const p of D.index.profiles) {
    const hit = D.areas.find((a) => d3.geoContains(a.f, [p.home_lon, p.home_lat]));
    p.area = hit ? hit.name.split("-")[0] : p.home_borough;
    const top = Object.entries(p.cuisine_profile).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 2).map(([k]) => CUISINES[k]);
    p.taste = `${top.join(" and ")} · ${p.home_borough} · ${plural(p.history_size, "past review")}`;
  }
  // party from the address (profile ids and strategy only; a start point never goes into the address)
  const h = new URLSearchParams(location.hash.slice(1)), ids = D.index.profiles.map((p) => p.id);
  const first = ids.includes(h.get("a")) ? h.get("a") : ids[0];
  // places picked earlier in this browser; a taste of your own is restored only when the server is there to score it
  try {
    const kept = JSON.parse(localStorage.getItem(STORE) ?? "{}");
    for (const m of state.members) {
      const list = Array.isArray(kept[m.key]) ? kept[m.key].filter((x) => Number.isInteger(x?.id) && typeof x.name === "string").slice(0, MAX_LIKED) : [];
      m.liked = list.map((x) => ({ id: x.id, name: x.name.slice(0, 120), area: String(x.area ?? "").slice(0, 120) }));
    }
  } catch { /* private window or blocked storage: start without picks */ }
  if (liveReady()) state.members.forEach((m, k) => { if (h.get(k === 0 ? "a" : "b") === "mine" && m.liked.length >= MIN_LIKED) m.source = "mine"; });
  const mate = D.index.profiles.find((p) => p.id !== first && p.home_borough === profileOf(first).home_borough) ?? D.index.profiles.find((p) => p.id !== first);
  const second = ids.includes(h.get("b")) && h.get("b") !== first ? h.get("b") : mate.id;
  if (h.has("a")) { state.two = h.has("b"); state.screen = "discover"; }
  if (h.get("by") === "floor") state.strategy = "floor";
  setProfile(0, first); setProfile(1, second);
  if (state.screen === "discover" && !(await applyParty())) state.screen = "start";
  mount();
}

const isMine = (m) => m.source === "mine";
const active = () => state.members.slice(0, state.two ? 2 : 1);
const liveReady = () => state.api?.live === "ready";
const partyKey = () => active().map((m) => (isMine(m) ? `mine:${m.liked.map((x) => x.id).join(",")}` : m.profileId)).join("|");
const shortOf = (m) => active().find((x) => isMine(x) && x.liked.length < MIN_LIKED && (!m || x === m));
const areaOf = (lat, lon, fallback) => D.areas.find((a) => d3.geoContains(a.f, [lon, lat]))?.name.split("-")[0] ?? fallback;

// Where a member starts from unless they say otherwise: the usual area of the taste they use.
function usualStart(m) {
  if (isMine(m)) return m.mine ? { lat: m.mine.lat, lon: m.mine.lon, how: "usual", label: `around your places, ${m.mine.area}` } : { how: "usual", label: "around your places" };
  const p = profileOf(m.profileId);
  return { lat: p.home_lat, lon: p.home_lon, how: "usual", label: `usual area, ${p.area}` };
}

function setProfile(k, id) {
  const m = state.members[k];
  m.profileId = id;
  if (!m.start || m.start.how === "usual") m.start = usualStart(m);
}

function keepLiked() {
  try { localStorage.setItem(STORE, JSON.stringify(Object.fromEntries(state.members.map((m) => [m.key, m.liked])))); } catch { /* not kept: the picks last until the page closes */ }
}

// Make the tastes of the current party ready: the exported files for sample tastes, the server for a party
// with a taste of someone's own. Returns false, with state.notice set, when that is not possible; the list
// then stays on the party it was computed for.
async function applyParty() {
  const act = active(), key = partyKey();
  if (D.applied?.key === key) return true;
  state.notice = null;
  if (shortOf()) { state.notice = `Pick at least ${MIN_LIKED} places you like, or borrow a sample taste.`; return false; }
  try {
    if (!act.some(isMine)) {
      await Promise.all(act.map(async (m) => { if (!D.profiles.has(m.profileId)) D.profiles.set(m.profileId, await json(`app/${profileOf(m.profileId).file}`)); }));
      D.places = D.base;
      D.applied = { key, live: false, base: D.index.base, profiles: act.map((m) => D.profiles.get(m.profileId)) };
    } else {
      state.busy = true; showBusy();
      const j = await scoreParty(act.map((m) => (isMine(m) ? { liked: m.liked.map((x) => x.id) } : { profile: m.profileId })));
      D.places = j.places;
      D.applied = { key, live: true, base: j.base, profiles: j.members };
      act.forEach((m, k) => {
        if (!isMine(m)) return;
        const q = j.members[k];
        m.mine = { lat: q.home_lat, lon: q.home_lon, borough: q.home_borough, area: areaOf(q.home_lat, q.home_lon, q.home_borough), top: q.top };
        if (!m.start || m.start.how === "usual") m.start = usualStart(m);
        else if (m.start.lat == null) m.start = { ...m.start, lat: q.home_lat, lon: q.home_lon };
      });
    }
  } catch (err) {
    // a request that never arrived (no network, server stopped, timed out) is "down" as well
    state.notice = FAIL[err.message] ?? (err instanceof TypeError || err.name === "AbortError" ? FAIL.down : "The tastes could not be loaded.");
    return false;
  } finally { state.busy = false; }
  // the place table may have changed, and with it what a place number means
  state.selected = null; state.picked = null; state.compareWith = null; state.hot = null;
  D.members = JSON.stringify({ two: state.two, members: state.members });
  return true;
}

// The members the list is for: the current party or, while a change waits in "Who is eating?", the one before it.
function listMembers() {
  if (D.applied.key === partyKey()) return active();
  const was = JSON.parse(D.members);
  return was.members.slice(0, was.two ? 2 : 1);
}

// A change that needs no server (sample tastes only) shows in the list at once.
const settle = async () => { if (state.screen !== "start" && !active().some(isMine)) await applyParty(); };

// Back to the party the list was computed for (after a change that could not be applied).
function revertParty() {
  if (!D.members) return;
  const was = JSON.parse(D.members);
  state.two = was.two;
  was.members.forEach((m, k) => Object.assign(state.members[k], { profileId: m.profileId, source: m.source, liked: m.liked, mine: m.mine, start: m.start }));
  keepLiked();
}

function showBusy() {
  const el = state.screen === "start" ? $("start-status") : $("status");
  if (el) el.innerHTML = `<p role="status">Building your taste from the places you picked. This takes a few seconds.</p>`;
  // the button that asked for it says so too (the status line can be behind the sheet)
  document.querySelectorAll('[data-act="go"], .sheet .acts [data-act="closeSheet"]').forEach((b) => { b.disabled = true; b.textContent = "Building your taste…"; });
}

function writeHash() {
  const h = new URLSearchParams(), id = (m) => (isMine(m) ? "mine" : m.profileId);   // picked places never go into the address
  h.set("a", id(state.members[0]));
  if (state.two) { h.set("b", id(state.members[1])); h.set("by", state.strategy); }
  history.replaceState(null, "", "#" + h.toString());
}

// ------------------------------------------------------------------------------------------------ compute
function compute() {
  const members = listMembers().map((m, k) => ({ ...m, profile: D.applied.profiles[k] }));
  const pool = buildPool(D.places, D.index.cuisine_names, members, state.filters);
  const pre = prepare(D.places, D.index.dist_cap_km, members, pool.ids);
  const strategy = members.length === 1 ? STRATEGY.solo : STRATEGY[state.strategy];
  const res = pool.ids.length ? solve(members, pre, state.targets, state.w_h, strategy) : null;
  const base0 = pool.ids.length ? memberScores(members, pre, { w_h: 1, alpha: 0, beta: 0 }).map((x) => x.s0) : [];
  const items = !res ? [] : res.top.map((j, n) => {
    const i = pre.ids[j];
    return { j, i, rank: n + 1, name: D.places.name[i], known: members.findIndex((m) => isMine(m) && m.liked.some((x) => x.id === D.places.rid?.[i])), sat: members.map((_, k) => res.sat[k][j]), km: members.map((_, k) => pre.dist[k][j]),
      reasons: members.map((m) => mainReason(m.profile, D.index.groups, i)) };
  });
  state.r = { members, pool, pre, res, strategy, base0, items };
  if (!items.some((it) => it.i === state.selected)) state.selected = null;
  if (!items.some((it) => it.i === state.picked)) state.picked = null;
}

// One member's score for one listed place, part by part (for "Every reason, with numbers").
function parts(k, j) {
  const r = state.r, p = r.members[k].profile, i = r.pre.ids[j], th = r.res.theta;
  return { groups: D.index.groups.map((g) => [g, p.groups[g][i]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])),
    raw: p.raw[i], base: D.applied.base, history: r.res.ms[k].s0[j] - r.base0[k][j], distance: -th.alpha * r.pre.distN[k][j] || 0,
    popular: -th.beta * r.pre.pop[j] || 0, km: r.pre.dist[k][j], sat: r.res.sat[k][j] };
}

function cardReason(it) {
  const ms = state.r.members, g = it.reasons;
  if (ms.length === 1) return g[0] ? YOU[g[0]].replace(/^./, (c) => c.toLowerCase()) : NONE.toLowerCase();
  if (g[0] === g[1]) return g[0] ? `${YOU[g[0]].replace(/^./, (c) => c.toLowerCase())}, for both of you` : `${NONE.toLowerCase()}, for either of you`;
  return ms.map((m, k) => (g[k] ? CARD[g[k]](m.name) : `nothing stands out for ${m.name}`)).join("; ");
}

// ------------------------------------------------------------------------------------------------ rendering
// Replace a region's content, keeping keyboard focus on the same control.
function swap(el, html) {
  if (!el) return;
  const key = el.contains(document.activeElement) ? document.activeElement.dataset.fk : null;
  el.innerHTML = html;
  el.querySelectorAll("[data-act],[data-chg]").forEach((n) => { n.dataset.fk = `${n.dataset.act ?? n.dataset.chg}|${n.dataset.arg ?? ""}`; });
  if (key) el.querySelector(`[data-fk="${CSS.escape(key)}"]`)?.focus();
}

const bandTop = () => `<div class="top"><a class="brand" href="./"><img class="cut" src="${avatar("tomato")}" alt=""><span>Eat Buddy</span></a>
  <nav class="nav" aria-label="More"><a href="lab.html">Lab</a><button type="button" data-act="sheet" data-arg="data">About the data</button></nav></div>`;

// How much a short list of liked places is worth, from the cold-start study (src/32_cold_start.py).
function coldNote() {
  const rows = D.cold?.rows.filter((r) => r.model === "LambdaRank") ?? [], at = (k) => rows.find((r) => r.k === String(k));
  if (!at(3) || !at(5) || !at(10)) return "";
  const pc = (k) => `${Math.round(at(k).share_of_full * 100)}%`;
  return ` In our test on ${D.cold.users.toLocaleString("en-US")} reviewers, a taste built from 3 liked places kept ${pc(3)} of the ranking accuracy of their full history, 5 kept ${pc(5)} and 10 kept ${pc(10)}.`;
}

function resultsHTML(k) {
  const s = state.search[k], m = state.members[k];
  if (s.note) return `<p class="soft" role="status">${esc(s.note)}</p>`;
  if (!s.results) return "";
  if (!s.results.length) return `<p class="soft" role="status">No restaurant with that name in our data, which ends in ${esc(D.index.cutoff.slice(0, 4))}.</p>`;
  const full = m.liked.length >= MAX_LIKED;
  return `<ul class="results" aria-label="Search results">${s.results.map((x, n) => {
    const has = m.liked.some((y) => y.id === x.id);
    return `<li><button type="button" data-act="like" data-arg="${k}:${n}"${has || full ? " disabled" : ""}><b>${esc(x.name)}</b><span>${esc(x.cuisine)} · ${esc(x.area.replace(/-/g, ", "))} · ${x.reviews.toLocaleString("en-US")} reviews${has ? " · picked" : ""}</span></button></li>`;
  }).join("")}</ul>`;
}

function mineHTML(k) {
  const m = state.members[k], n = m.liked.length, s = state.search[k];
  const built = m.mine && D.applied?.live && D.applied.key === partyKey();
  return `<label class="field">Find a restaurant you like
      <input type="search" class="plain" data-search="${k}" data-chg="search" data-arg="${k}" autocomplete="off" spellcheck="false" maxlength="60" placeholder="Type a name, like Joe’s Pizza" value="${esc(s.q)}"></label>
    <div id="results-${k}" aria-live="polite">${resultsHTML(k)}</div>
    ${n ? `<ul class="liked" aria-label="Places you picked">${m.liked.map((x) => `<li class="tag"><span>${esc(x.name)}<small> · ${esc(x.area.split("-")[0])}</small></span><button type="button" data-act="unlike" data-arg="${k}:${x.id}" aria-label="Remove ${esc(x.name)}, ${esc(x.area.split("-")[0])}">Remove</button></li>`).join("")}</ul>` : ""}
    <p class="${n < MIN_LIKED ? "warn" : "soft"}" role="status">${n < MIN_LIKED ? `${n} picked. Pick at least ${MIN_LIKED}; more is better.` : `${n} picked${n >= MAX_LIKED ? ", the most a taste can hold" : ""}.`}<span class="soft">${coldNote()}</span></p>
    ${built ? `<p class="hand">Alone, this taste would go to ${esc(m.mine.top.join(", "))}.</p>` : ""}`;
}

function personHTML(k) {
  const m = state.members[k], mine = isMine(m), p = profileOf(m.profileId), how = m.start.how === "pin" ? "pin" : m.start.how;
  const gps = state.gps[k], live = state.api?.live;
  const usual = mine ? `Around my places${m.mine ? ` (${m.mine.area})` : ""}` : `This taste’s usual area (${p.area})`;
  return `<div class="person m${k}">
    <div class="ph"><img class="cut" src="${avatar(m.key)}" alt=""><b>${esc(label(m, k))}</b></div>
    <div class="seg" role="group" aria-label="Whose taste">
      <button type="button" data-act="source" data-arg="${k}:sample" aria-pressed="${!mine}">Borrow a sample taste</button>
      <button type="button" data-act="source" data-arg="${k}:mine" aria-pressed="${mine}"${liveReady() ? "" : " disabled"}>Places I like</button></div>
    ${liveReady() ? "" : `<p class="soft">${live === "loading" ? "The server is still starting, so “Places I like” is not ready yet." : "“Places I like” needs the Eat Buddy server, which is not connected to this site. Sample tastes work without it."}
      ${state.api || import.meta.env.DEV ? `<button type="button" class="btn line" data-act="recheck">Check again</button>` : ""}</p>`}
    ${mine ? mineHTML(k) : `<label class="field">Borrow a taste
      <select data-chg="profile" data-arg="${k}">${D.index.profiles.map((q, n) => `<option value="${q.id}"${q.id === m.profileId ? " selected" : ""}>Taste ${n + 1}: ${esc(q.taste)}</option>`).join("")}</select></label>
    <p class="hand">Alone, this taste would go to ${esc(p.top.slice(0, 3).join(", "))}.</p>`}
    <label class="field">Start from
      <select data-chg="startHow" data-arg="${k}">
        <option value="usual"${how === "usual" ? " selected" : ""}>${esc(usual)}</option>
        <option value="nta"${how === "nta" ? " selected" : ""}>A neighborhood I choose</option>
        ${k === 0 ? `<option value="gps"${how === "gps" ? " selected" : ""}>My location</option>` : ""}
        ${how === "pin" ? `<option value="pin" selected>The pin dropped on the map</option>` : ""}
      </select></label>
    ${how === "nta" ? `<label class="field">Neighborhood
      <select data-chg="startNta" data-arg="${k}">${BOROUGHS.map((b) => `<optgroup label="${b}">${D.areas.filter((a) => a.borough === b).map((a) => `<option value="${a.code}"${a.code === m.start.code ? " selected" : ""}>${esc(a.name)}</option>`).join("")}</optgroup>`).join("")}</select></label>` : ""}
    ${how === "gps" ? `<div class="acts"><button type="button" class="btn herb" data-act="gps" data-arg="${k}">${m.start.fix ? "Update my location" : "Use my location"}</button></div>
      <p class="${gps?.error ? "warn" : "soft"}" role="status">${gps?.error ? esc(gps.error) : gps?.busy ? "Asking your browser for your location." : m.start.fix ? "Using your location, rounded to about 100 m. It stays on this device." : "Your location is read only when you tap the button, is rounded to about 100 m and stays on this device. Until then the usual area is used."}</p>` : ""}
  </div>`;
}

function peopleHTML() {
  return `${personHTML(0)}${state.two ? personHTML(1) : ""}
    <div class="acts"><button type="button" class="btn sun" data-act="${state.two ? "solo" : "invite"}">${state.two ? "Eat alone instead" : "+ Invite a friend"}</button></div>`;
}

function mount() {
  document.body.dataset.view = "list"; document.body.dataset.drawer = "0";
  if (state.screen === "start") {
    root.innerHTML = `<header class="band"><div class="wrap">${bandTop()}</div></header>
      <main class="start"><div class="wrap">
        <h1 class="caps">Who is eating?</h1>
        <p class="lead">Pick a place to eat in New York together, and see why it is fair to each of you.</p>
        <div id="people" style="display:contents"></div>
        <div class="acts"><button type="button" class="btn go" data-act="go">Find our table</button></div>
        <div id="start-status" aria-live="polite">${state.notice ? `<p class="warn">${esc(state.notice)}</p>` : ""}</div>
        <p class="soft">Each person brings a taste: one built from places they like, or the taste of an anonymous sample diner. Recommendations come from public reviews up to ${esc(D.index.cutoff)}: a research prototype, not a live directory.</p>
      </div></main><div id="sheet"></div>`;
    swap($("people"), peopleHTML());
    renderSheet();
    return;
  }
  root.innerHTML = `<header class="band"><div class="wrap">${bandTop()}
      <div class="partyrow"><div class="party" id="party"></div>
        <form class="say" id="say"><label class="sr" for="say-in">Say what you feel like</label>
          <input id="say-in" type="text" maxlength="${MAX_SENTENCE}" autocomplete="off" placeholder="Say it: not too pricey, close to both of us">
          <button class="btn go" type="submit" id="say-go">Find our table</button></form></div>
    </div></header>
    <section class="heard" id="heard" aria-live="polite" hidden></section>
    <div class="page"><div class="wrap">
      <div class="filters" id="filters" role="group" aria-label="Quick filters"></div>
      <div class="status" id="status" aria-live="polite"></div>
      <div class="cols"><main class="listcol" id="list"></main>
        <aside class="aside" id="aside" aria-label="Map and reasons"><div class="mapframe"><div class="map" id="map"></div></div><div id="drawer"></div></aside></div>
    </div></div>
    <button type="button" class="btn herb fab" data-act="view" id="fab">Map</button>
    <div id="sheet"></div>`;
  map = createMap($("map"), D.nta, avatar, {
    select: (i) => { state.selected = i; state.compareWith = null; update(); focusDrawer(); },
    hover: (i) => { state.hot = i; markHot(); },
    pinMode: (key) => { state.pinning = key; update(); },
    drop: (lat, lon) => {
      const m = state.members.find((x) => x.key === state.pinning);
      if (lat < NYC.lat[0] || lat > NYC.lat[1] || lon < NYC.lon[0] || lon > NYC.lon[1]) return;
      m.start = { lat: +lat.toFixed(3), lon: +lon.toFixed(3), how: "pin", label: "dropped pin" };
      state.pinning = null; update();
    },
  });
  update();
}
let map = null;

function update() {
  compute();
  writeHash();
  const r = state.r, two = r.members.length === 2;
  swap($("party"), r.members.map((m, k) => `<button type="button" class="who on-herb" data-act="sheet" data-arg="party"><img class="cut" src="${avatar(m.key)}" alt="">${esc(label(m, k))}, ${esc(m.start.label)}</button>`).join("")
    + (two ? "" : `<button type="button" class="btn sun" data-act="invite">+ Invite a friend</button>`));
  $("say-in").placeholder = two ? "Say it: not too pricey, close to both of us" : "Say it: cheap sushi within walking distance";
  renderHeard(); renderFilters(); renderStatus(); renderList(); renderDrawer(); renderSheet();
  map.update({ items: r.items.map((it) => ({ i: it.i, rank: it.rank, lat: D.places.lat[it.i], lon: D.places.lon[it.i], name: it.name, nta: D.places.nta_name[it.i] })),
    starts: r.members.map((m) => ({ key: m.key, name: m.name, lat: m.start.lat, lon: m.start.lon })), selected: state.selected, hot: state.hot,
    pinning: state.pinning ? r.members.find((m) => m.key === state.pinning) ?? null : null });
  document.body.dataset.view = state.view; document.body.dataset.drawer = state.selected != null ? "1" : "0";
  $("fab").textContent = state.view === "map" ? "List" : "Map";
}

function markHot() {
  map.mark(state.selected, state.hot);
  document.querySelectorAll(".card").forEach((c) => c.classList.toggle("hot", +c.dataset.place === state.hot));
}

// What the sentence would set, in the app's words. Targets first, then filters.
function sayTags(slots) {
  const who = state.two ? "each of you" : "you", out = [];
  if (slots.max_km != null) out.push({ key: "max_km", text: `Closer: about ${slots.max_km} km on average for ${who}` });
  if (slots.popularity) out.push({ key: "popularity", text: slots.popularity === "hidden_gems" ? "Hidden gems" : "Less mainstream" });
  if (slots.variety) out.push({ key: "variety", text: slots.variety === "lots" ? "As much variety as possible" : "Some variety" });
  if (slots.history) out.push({ key: "history", text: { usual: "Places like the usual ones", mix: "Half usual, half new", new: "Something new" }[slots.history] });
  return [...out, ...nlChips(slots).filter((c) => c.kind === "filter").map((c) => ({ key: c.key, text: c.text, filter: true }))];
}

function renderHeard() {
  const el = $("heard"), d = state.draft;
  el.hidden = !(d || state.saying);
  if (state.saying) { swap(el, `<div class="wrap"><p role="status">Reading your sentence.</p></div>`); return; }
  if (!d) { swap(el, ""); return; }
  const tags = sayTags(d.slots), by = d.source === "llm" ? "Read by the language model." : `Read by keyword rules, because ${WHY[d.why]}.`;
  const un = d.slots.unsupported.length ? `<p>No setting for: ${d.slots.unsupported.map(esc).join(", ")}.</p>` : "";
  if (!tags.length) {
    swap(el, `<div class="wrap"><p>${!d.slots.in_scope ? "That does not read as a wish about where to eat." : "No setting was found in that sentence."} Nothing changed. <span class="soft">${by}</span></p>${un}
      <div class="acts"><button type="button" class="btn quiet" data-act="say-discard">Dismiss</button></div></div>`);
    return;
  }
  let count = "", blocked = false;
  if (tags.some((t) => t.filter)) {
    const merged = { ...mergeFilters(state.filters, d.slots), min_rating: state.filters.min_rating };
    const left = buildPool(D.places, D.index.cuisine_names, state.r.members, merged);
    blocked = left.ids.length === 0;
    count = blocked ? `<p class="warn">None of your ${state.r.pool.union} places passes all of these. Remove one to continue.</p>`
      : `<p>${plural(left.ids.length, "place")} of your ${left.union} would be left to choose from.</p>`;
    if (d.slots.area && !left.areas.length) count += `<p class="warn">None of your places is in a neighborhood named “${esc(d.slots.area)}”.</p>`;
  }
  swap(el, `<div class="wrap"><p><b>Is this what you meant?</b> <span class="soft">${by} Nothing changes until you apply it.</span></p>
    <ul>${tags.map((t) => `<li class="tag">${esc(t.text)}<button type="button" data-act="say-drop" data-arg="${esc(t.key)}" aria-label="Remove ${esc(t.text)}">Remove</button></li>`).join("")}</ul>
    ${count}${un}
    <div class="acts"><button type="button" class="btn go" data-act="say-apply"${blocked ? " disabled" : ""}>Apply</button><button type="button" class="btn quiet" data-act="say-discard">Discard</button></div></div>`);
}

function menu(id, title, value, options, shown) {
  const on = value != null;
  return `<details class="fchip${on ? " on" : ""}" data-menu="${id}"${state.menu === id ? " open" : ""}><summary data-fk="menu|${id}">${title}${on ? `: ${shown}` : ""}</summary>
    <div class="fmenu" role="group" aria-label="${title}">${options.map(([v, t]) => `<button type="button" data-act="set" data-arg="${id}:${v ?? ""}" aria-pressed="${v === value}">${t}</button>`).join("")}</div></details>`;
}

function renderFilters() {
  const t = state.targets, f = state.filters, diet = ["vegetarian_options", "halal"];
  const dietOn = f.needs.filter((n) => diet.includes(n));
  const extra = f.cuisines.length + (f.price_min != null) + f.needs.filter((n) => !diet.includes(n)).length + !!f.borough + !!f.area + (state.w_h !== 1);
  swap($("filters"),
    menu("km", "Closer", t.km, [[null, "Any distance"], [4, "Within 4 km"], [2, "Within 2 km"], [1, "Walking distance, 1 km"]], `${t.km} km`)
    + menu("popularity", "Less mainstream", t.popularity, [[null, "No preference"], [0.4, "Less mainstream"], [0.2, "Hidden gems"]], t.popularity === 0.2 ? "hidden gems" : "yes")
    + menu("cuisines", "More variety", t.cuisines, [[null, "No preference"], [4, "Some variety"], [6, "As much as possible"]], t.cuisines >= 6 ? "lots" : "some")
    + menu("price_max", "Price", f.price_max, [[null, "Any price"], [1, "$ only"], [2, "Up to $$"], [3, "Up to $$$"]], f.price_max ? `up to ${"$".repeat(f.price_max)}` : "")
    + `<details class="fchip${dietOn.length ? " on" : ""}" data-menu="diet"${state.menu === "diet" ? " open" : ""}><summary data-fk="menu|diet">Dietary${dietOn.length ? `: ${dietOn.length}` : ""}</summary>
        <div class="fmenu" role="group" aria-label="Dietary">${diet.map((n) => `<button type="button" data-act="need" data-arg="${n}" aria-pressed="${f.needs.includes(n)}">${NEEDS[n]}</button>`).join("")}</div></details>`
    + `<button type="button" class="fchip-btn" data-act="stars" aria-pressed="${f.min_rating != null}">4 stars and up</button>`
    + `<button type="button" class="fchip-btn all" data-act="sheet" data-arg="filters">All filters${extra ? ` (${extra} more on)` : ""}</button>`);
}

function renderStatus() {
  const r = state.r, t = state.targets, f = state.filters, rows = [];
  const who = r.members.length === 2 ? "whoever travels farther" : "you";
  if (r.res && r.items.length) {
    const o = r.res.outcomes, met = r.res.met;
    if (t.km != null) rows.push([met.km, met.km ? `Closer: this list averages ${o.km.toFixed(1)} km for ${who} (asked for ${t.km} km).` : `Asked for ${t.km} km, but the closest this list gets is ${o.km.toFixed(1)} km on average for ${who}.`]);
    if (t.popularity != null) rows.push([met.popularity, met.popularity ? `Less mainstream: the list sits at ${Math.round(o.popularity * 100)} of 100 for popularity among your places.` : `Asked for less mainstream places; this list still sits at ${Math.round(o.popularity * 100)} of 100 for popularity.`]);
    if (t.cuisines != null) rows.push([met.cuisines, met.cuisines ? `Variety: ${plural(o.cuisines, "kind")} of food in the list.` : `Asked for ${t.cuisines} kinds of food; this list reaches ${o.cuisines}.`]);
  }
  const tags = nlChips({ ...emptySlots(), ...f }).filter((c) => c.kind === "filter" && !["price_max", "needs:vegetarian_options", "needs:halal"].includes(c.key));
  if (state.w_h !== 1) tags.push({ key: "history", text: state.w_h === 0 ? "Something new" : "Half usual, half new" });
  swap($("status"), (state.notice ? `<p class="warn">${esc(state.notice)}</p>` : "") + (rows.length ? `<ul>${rows.map(([ok, text]) => `<li class="${ok ? "" : "miss"}">${ok ? "" : "<b>Not reached.</b> "}${text}</li>`).join("")}</ul>` : "")
    + (tags.length ? `<ul>${tags.map((c) => `<li class="tag">${esc(c.text)}<button type="button" data-act="unfilter" data-arg="${esc(c.key)}" aria-label="Remove ${esc(c.text)}">Remove</button></li>`).join("")}</ul>` : ""));
}

function renderList() {
  const r = state.r, two = r.members.length === 2, n = r.items.length;
  const head = `<div class="listhead"><h1 class="caps">${n ? `${COUNT[n]} for you${two ? " both" : ""}` : "Nothing to show yet"}</h1>
      <button type="button" class="howlink" data-act="sheet" data-arg="how">How is this ranked?</button></div>
    ${two ? `<div class="pickby" role="group" aria-label="How to combine your tastes"><span>pick by</span>
      <button type="button" data-act="by" data-arg="fair" aria-pressed="${state.strategy === "fair"}">Fair to everyone</button>
      <button type="button" data-act="by" data-arg="floor" aria-pressed="${state.strategy === "floor"}">Nobody unhappy</button></div>` : ""}`;
  let note = "";
  if (!r.pool.ids.length) note = `<div class="note"><h2>No place passes all your filters</h2><p>Together your tastes bring ${r.pool.union} places. None of them passes every filter that is on. Remove one above, or clear them all.</p><button type="button" class="btn herb" data-act="clear">Clear filters</button></div>`;
  else if (!n) note = `<div class="note"><h2>No place keeps you both happy enough</h2><p>“Fair to everyone” skips places where either of you is under ${TAU.toFixed(2)} out of 1, and with these settings that removes all ${r.pool.ids.length}. “Nobody unhappy” takes the place whose less happy person is happiest.</p><button type="button" class="btn herb" data-act="by" data-arg="floor">Pick by nobody unhappy</button></div>`;
  else if (n < K) note = `<div class="note"><p>Only ${plural(n, "place")} ${n === 1 ? "passes" : "pass"}${r.strategy === STRATEGY.fair && r.res.count < r.pool.ids.length ? ` with both of you at ${TAU.toFixed(2)} or more` : " your filters"}.</p></div>`;
  const pk = r.items.find((it) => it.i === state.picked);
  const picked = pk ? `<div class="note picked"><h2>Tonight: ${esc(pk.name)}</h2><p class="num">${r.members.map((m, k) => `Suits ${k === 0 ? "you" : esc(m.name)} ${pk.sat[k].toFixed(2)} out of 1, ${pk.km[k].toFixed(1)} km away`).join(" · ")}.</p><button type="button" class="btn quiet" data-act="unpick">Keep looking</button></div>` : "";
  const cards = r.items.map((it) => `<article class="card${it.i === state.selected ? " sel" : ""}" data-place="${it.i}">
      <span class="rank" aria-hidden="true">${it.rank}</span>
      <img class="sticker" src="${sticker(it.i)}" alt="" loading="lazy" width="100" height="100">
      <div class="body">
        <h2><span class="sr">Number ${it.rank}: </span>${esc(it.name)}</h2>
        <p class="facts">${esc(facts(it.i))}</p>
        <p class="hand">${esc(cardReason(it))}</p>
        ${it.known >= 0 && two ? `<p class="travel">${it.known === 0 ? `You picked this place as one you like; it is here through ${esc(r.members[1].name)}’s candidates.` : `${esc(r.members[1].name)} picked this place as one they like; it is here through your candidates.`}</p>` : ""}
        <div class="suits"><span class="lab">suits you${two ? " both" : ""}</span>
          <div class="suitline${two ? "" : " solo"}" role="img" aria-label="${r.members.map((m, k) => `Suits ${m.name} ${it.sat[k].toFixed(2)}`).join(", ")}, on a scale from 0 to 1"><i></i><i></i>
            ${r.members.map((m, k) => `<img class="cut m${k}" src="${avatar(m.key)}" alt="" style="left:${(it.sat[k] * 100).toFixed(1)}%">`).join("")}</div>
          <span class="val num">${it.sat.map((v) => v.toFixed(2)).join(" · ")}</span></div>
        <p class="travel num">${r.members.map((m, k) => `${it.km[k].toFixed(1)} km for ${k === 0 ? "you" : esc(m.name)}`).join(" · ")}</p>
      </div>
      <button type="button" class="btn herb" data-act="why" data-arg="${it.i}">See why<span class="sr">: ${esc(it.name)}</span></button>
    </article>`).join("");
  swap($("list"), head + picked + note + cards);
}

function renderDrawer() {
  const r = state.r, it = r.items.find((x) => x.i === state.selected);
  if (!it) { swap($("drawer"), ""); return; }
  // rating history and review themes were exported for the sample tastes' places; a live party may hold others
  const ref = D.places.ref[it.i], file = ref ? D.index.profiles[ref[0]].demo : null;
  const others = r.items.filter((x) => x.i !== it.i);
  if (state.compareWith == null || !others.some((x) => x.i === state.compareWith)) state.compareWith = others[0]?.i ?? null;
  const o = others.find((x) => x.i === state.compareWith);
  const why = (x, k) => (x.reasons[k] ? (k === 0 ? YOU : THEM)[x.reasons[k]] : NONE);
  if (file && ((state.open.stars && !D.detail.has(file)) || (state.open.reviews && !D.themes.has(file)))) loadPlaceFiles(file);
  swap($("drawer"), drawerHTML({
    place: { name: it.name, rating: D.places.rating[it.i] }, facts: facts(it.i), sticker: sticker(it.i), two: r.members.length === 2, open: state.open, picked: state.picked === it.i,
    members: r.members.map((m, k) => ({ name: m.name, label: label(m, k), avatar: avatar(m.key), sat: it.sat[k], km: it.km[k], reason: why(it, k), parts: parts(k, it.j) })),
    detailMeta: D.detailMeta, detail: file ? D.detail.get(file)?.[ref[1]] ?? null : null, themes: !file ? null : D.themes.has(file) ? (D.themes.get(file)?.[ref[1]] ?? null) : undefined,
    loading: !!file && (!D.detail.has(file) || !D.themes.has(file)),
    others: others.map((x) => ({ i: x.i, name: `${x.rank}. ${x.name}` })), compareWith: state.compareWith,
    other: o ? { name: o.name, members: r.members.map((_, k) => ({ sat: o.sat[k], km: o.km[k], reason: why(o, k) })) } : null,
  }));
}

// Rating history and review themes live in the demo case files of step 29 and 30; a place points at the first case that holds it.
const loadingFiles = new Set();
async function loadPlaceFiles(file) {
  if (loadingFiles.has(file)) return;
  loadingFiles.add(file);
  const get = (dir) => fetch(`${DATA}${dir}/${file}.json`).then((x) => (x.ok ? x.json() : null)).then((x) => x?.restaurants ?? null).catch(() => null);
  const [detail, themes] = await Promise.all([get("detail"), get("themes")]);
  D.detail.set(file, detail); D.themes.set(file, themes);
  loadingFiles.delete(file);
  renderDrawer();
}

function focusDrawer() { requestAnimationFrame(() => document.querySelector(".drawer")?.focus({ preventScroll: false })); }

// ------------------------------------------------------------------------------------------------ sheets
function strategyBlock(name, title, sub) {
  const r = state.r, res = solve(r.members, r.pre, state.targets, state.w_h, STRATEGY[name]);
  const rows = res.top.map((j, n) => ({ n: n + 1, name: D.places.name[r.pre.ids[j]], sat: r.members.map((_, k) => res.sat[k][j]) }));
  if (!rows.length) return `<div><h3>${title}</h3><p>${sub}</p><p>No place passes with these settings.</p></div>`;
  const low = Math.min(...rows.flatMap((x) => x.sat)), gap = Math.max(...rows.map((x) => Math.abs(x.sat[0] - x.sat[1])));
  return `<div><h3>${title}${state.strategy === name ? " (in use)" : ""}</h3><p>${sub}</p>
    <table class="cmp"><thead><tr><th></th><th>Place</th>${r.members.map((m) => `<th class="n">${esc(m.name)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((x) => `<tr><td class="n">${x.n}</td><td>${esc(x.name)}</td>${x.sat.map((v) => `<td class="n">${v.toFixed(2)}</td>`).join("")}</tr>`).join("")}</tbody></table>
    <p class="num">Lowest “suits” number on this list: ${low.toFixed(2)}. Largest gap between you at one place: ${gap.toFixed(2)}.</p></div>`;
}

function sheetBody() {
  const r = state.r, ix = D.index;
  if (state.sheet === "party") return ["Who is eating?", `${peopleHTML()}
    <div class="acts"><button type="button" class="btn herb" data-act="closeSheet">Show our table</button></div>
    <p class="soft">To start from an exact spot, close this and use “Move” on the map.</p>`];
  if (state.sheet === "data") {
    const k = ix.rankers, nd = (name) => k[name].ndcg.mean.toFixed(4);
    return ["About the data", `
      <p>Eat Buddy is a course research prototype (NYU DS-GA 3001, Visualization for Machine Learning), not a live directory. Everything here comes from public Google Local reviews of New York City restaurants and stops at ${esc(ix.cutoff)}. Places may have closed or changed since.</p>
      <p><b>Sample tastes.</b> A person can borrow the taste of one of ${ix.profiles.length} anonymous sample diners: real reviewers from the test period, published without ids, names or visit lists, with their usual area rounded to about 1 km.</p>
      <p><b>Your own taste.</b> ${liveReady() ? `With “Places I like” you pick at least ${MIN_LIKED} restaurants. Their numbers are sent to the Eat Buddy server, which scores them like a reviewer who liked exactly those places and keeps nothing. The list of picks stays in this browser until you remove it.${coldNote()}` : "Building a taste from places you like needs the Eat Buddy server, which is not connected to this site."}</p>
      <p><b>The model.</b> Scores come from ${esc(ix.model)}. On ${ix.test_users.toLocaleString("en-US")} test users it had the highest NDCG@10 of our three rankers: ${nd("LightGBM LambdaRank (supplementary)")}, against ${nd("LightGBM")} for LightGBM and ${nd("Logistic regression")} for logistic regression. The <a href="lab.html">Lab</a> shows the evaluation.</p>
      <p><b>Your sentence.</b> ${state.api?.llm ? "When you use “Say it”, the sentence alone is sent to Anthropic’s Claude to be read into settings. It never chooses places. Without the server, keyword rules read it on this page." : "On this site keyword rules read your sentence on this page; nothing is sent anywhere."}</p>
      <p><b>Your location.</b> Read only when you tap “Use my location”, rounded to about 100 m, and kept on this device.</p>
      <p><b>Pictures.</b> The food stickers and the Tomato and Tofu avatars are AI-generated images (Magic Patterns), one per cuisine group. They are not photos of the restaurants.</p>`];
  }
  if (state.sheet === "filters") {
    const f = state.filters;
    const box = (chg, arg, on, text) => `<label><input type="checkbox" data-chg="${chg}" data-arg="${arg}"${on ? " checked" : ""}>${esc(text)}</label>`;
    return ["All filters", `
      <h3>Kind of food</h3><div class="opts">${Object.entries(CUISINES).filter(([key]) => !["general", "venue"].includes(key)).map(([key, t]) => box("fcuisine", key, f.cuisines.includes(key), t)).join("")}</div>
      <h3>Must have</h3><div class="opts">${Object.entries(NEEDS).map(([key, t]) => box("fneed", key, f.needs.includes(key), t)).join("")}${box("fstars", "", f.min_rating != null, "4 stars and up")}</div>
      <h3>Price and place</h3>
      <div class="acts">
        <label class="field">Price from <select data-chg="fpricemin"><option value="">Any</option>${[2, 3, 4].map((v) => `<option value="${v}"${f.price_min === v ? " selected" : ""}>${"$".repeat(v)}</option>`).join("")}</select></label>
        <label class="field">Price up to <select data-chg="fpricemax"><option value="">Any</option>${[1, 2, 3].map((v) => `<option value="${v}"${f.price_max === v ? " selected" : ""}>${"$".repeat(v)}</option>`).join("")}</select></label>
        <label class="field">Borough <select data-chg="fborough"><option value="">Any</option>${BOROUGHS.map((b) => `<option${f.borough === b ? " selected" : ""}>${b}</option>`).join("")}</select></label>
      </div>
      <h3>Familiar or new</h3>
      <label class="field">How much past visits count <select data-chg="fhistory">${[[1, "Fully: places like the usual ones"], [0.5, "Half: some usual, some new"], [0, "Not at all: something new"]].map(([v, t]) => `<option value="${v}"${state.w_h === v ? " selected" : ""}>${t}</option>`).join("")}</select></label>
      <p class="num" role="status">${r.pool.ids.length ? `${plural(r.pool.ids.length, "place")} of your ${r.pool.union} pass.` : `<span class="warn">None of your ${r.pool.union} places passes all of these.</span>`} Places with no known price fail a price filter.</p>
      <div class="acts"><button type="button" class="btn herb" data-act="closeSheet">Show the list</button><button type="button" class="btn quiet" data-act="clear">Clear everything</button></div>`];
  }
  if (state.sheet === "how") {
    const two = r.members.length === 2, th = r.res?.theta;
    const set = th && (th.alpha || th.beta || th.lam || th.w_h !== 1);
    return ["How is this ranked?", `
      <ol class="steps">
        <li><b>A pool of places.</b> ${two ? `Your two tastes bring ${r.pool.union} places together` : `Your taste brings ${r.pool.union} places`}: each taste’s 200 candidates from the first stage of the recommender.${r.members.some(isMine) ? " A taste built from places you picked is scored as a reviewer who liked exactly those places. The picked places are left out of that person’s own candidates; one can still come in through the other person’s." : ""}${r.pool.ids.length < r.pool.union ? ` Your filters keep ${r.pool.ids.length} of them.` : ""}</li>
        <li><b>A score per person.</b> For every place the ranking model estimates how well it fits that person’s past visits. The score is built for the taste’s usual area; your start point changes the distances, not the taste.</li>
        <li><b>Your settings.</b> “Closer” takes points off for distance from each person’s own start; “Less mainstream” takes points off for popularity.${set ? ` Right now: distance weight ${th.alpha.toFixed(2)}, popularity weight ${th.beta.toFixed(2)}, variety ${(th.lam * 2).toFixed(2)}, past visits ${Math.round(th.w_h * 100)}%. The app uses the smallest weights that reach what you asked for, and its largest ones when a wish cannot be reached.` : " None is on right now."}</li>
        <li><b>“Suits you”, from 0 to 1.</b> Each person’s scores are stretched over the pool: 1 is that person’s best place, 0 the worst.</li>
        ${two ? `<li><b>Combining you.</b> “Fair to everyone” takes the average of your two numbers and skips any place where one of you is under ${TAU.toFixed(2)}. “Nobody unhappy” ranks by the lower of your two numbers.</li>` : ""}
        <li><b>Variety, if you asked.</b> The list is re-ordered so the same kind of food and the same neighborhood repeat less.</li>
      </ol>
      ${two && r.pool.ids.length ? `<h3>Who gains, who gives up</h3><p>The same pool and settings under each way of combining you. A number is how well the place suits that person.</p>
        ${strategyBlock("fair", "Fair to everyone", "Best on average, as long as nobody is left under the bar.")}
        ${strategyBlock("floor", "Nobody unhappy", "The less happy person decides.")}` : ""}
      <p class="soft">In our offline test of 1,000 simulated groups of three, “nobody unhappy” raised the least happy member’s number by 0.05 and cost the average member a little accuracy. The <a href="lab.html">Lab</a> has the full comparison.</p>`];
  }
  return null;
}

function renderSheet() {
  const el = $("sheet"), body = state.sheet ? sheetBody() : null;
  if (!body) { swap(el, ""); return; }
  swap(el, `<div class="scrim" data-scrim><div class="sheet" role="dialog" aria-modal="true" aria-labelledby="sheet-title">
    <div class="head"><h2 class="caps" id="sheet-title" tabindex="-1">${body[0]}</h2>
      <button type="button" class="btn herb x" data-act="closeSheet" aria-label="Close" style="width:var(--tap);padding:0"><svg viewBox="0 0 18 18" width="18" height="18" aria-hidden="true" stroke="currentColor" stroke-width="3" stroke-linecap="round"><path d="M3 3l12 12M15 3L3 15"/></svg></button></div>
    ${body[1]}</div></div>`);
}

function openSheet(name) {
  state.sheet = name; state.menu = null;
  refresh();
  requestAnimationFrame(() => $("sheet-title")?.focus());
}
const refresh = () => {
  if (state.screen !== "start") return update();
  swap($("people"), peopleHTML());
  swap($("start-status"), state.notice ? `<p class="warn">${esc(state.notice)}</p>` : "");
  renderSheet();
};

// ------------------------------------------------------------------------------------------------ actions
const clearAll = () => { state.filters = { ...emptyFilters(), min_rating: null }; state.targets = { ...NO_TARGETS }; state.w_h = 1; };

async function onAct(act, arg, el) {
  const f = state.filters;
  if (state.busy) return;                              // a party is being scored: one thing at a time
  if (act !== "closeSheet" && act !== "go" && act !== "sheet") state.notice = null;
  if (act === "sheet") return openSheet(arg);
  if (act === "closeSheet") {
    // leaving "Who is eating?" applies the party; a change that cannot be applied is undone, with the reason shown
    if (state.sheet === "party" && state.screen !== "start" && !(await applyParty())) { revertParty(); state.notice += " Your change to who is eating was undone."; }
    state.sheet = null; return refresh();
  }
  if (act === "go") {
    if (await applyParty()) state.screen = "discover";
    return mount();
  }
  if (act === "invite") { state.two = true; if (state.screen !== "start") state.sheet = "party"; await settle(); return refresh(); }
  if (act === "solo") { state.two = false; await settle(); return refresh(); }
  if (act === "recheck") { state.api = await apiStatus(true); return refresh(); }
  if (act === "source") {
    const [k, v] = arg.split(":"), m = state.members[+k];
    if (v === "mine" && !liveReady()) return;
    m.source = v;
    if (m.start.how === "usual") m.start = usualStart(m);
    state.notice = null;
    await settle();
    return refresh();
  }
  if (act === "like" || act === "unlike") {
    const [k, v] = arg.split(":"), m = state.members[+k], s = state.search[+k];
    if (act === "like") {
      const x = s.results?.[+v];
      if (x && m.liked.length < MAX_LIKED && !m.liked.some((y) => y.id === x.id)) m.liked = [...m.liked, { id: x.id, name: x.name, area: x.area }];
      state.search[+k] = { q: "", results: null, note: "" };
    } else m.liked = m.liked.filter((x) => x.id !== +v);
    keepLiked(); state.notice = null;
    refresh();
    if (act === "like") root.querySelector(`[data-search="${k}"]`)?.focus();
    return;
  }
  if (act === "gps") return locate(+arg);
  if (act === "view") { state.view = state.view === "map" ? "list" : "map"; return update(); }
  if (state.screen === "start") return;
  if (act === "set") {
    const [id, raw] = arg.split(":"), v = raw === "" ? null : +raw;
    if (id === "price_max") f.price_max = v; else state.targets[id] = v;
    if (f.price_max != null && f.price_min != null && f.price_min > f.price_max) f.price_min = null;
    state.menu = null;
  }
  if (act === "need") f.needs = f.needs.includes(arg) ? f.needs.filter((n) => n !== arg) : [...f.needs, arg];
  if (act === "stars") f.min_rating = f.min_rating == null ? 4 : null;
  if (act === "by") state.strategy = arg;
  if (act === "why") { state.selected = +arg; state.compareWith = null; update(); return focusDrawer(); }
  if (act === "close") { const i = state.selected; state.selected = null; update(); return document.querySelector(`[data-fk="why|${i}"]`)?.focus(); }
  if (act === "pick") state.picked = state.selected;
  if (act === "unpick") state.picked = null;
  if (act === "clear") clearAll();
  if (act === "unfilter") {
    if (arg === "history") state.w_h = 1;
    else state.filters = { ...withoutChip({ ...emptySlots(), ...f }, arg), min_rating: f.min_rating };
    for (const k of ["in_scope", "max_km", "popularity", "variety", "history", "unsupported"]) delete state.filters[k];
  }
  if (act === "say-drop" && state.draft) state.draft = { ...state.draft, slots: withoutChip(state.draft.slots, arg) };
  if (act === "say-discard") state.draft = null;
  if (act === "say-apply" && state.draft) {
    const s = state.draft.slots;
    if (s.max_km != null) state.targets.km = s.max_km;
    if (s.popularity) state.targets.popularity = POPULARITY[s.popularity];
    if (s.variety) state.targets.cuisines = VARIETY[s.variety];
    if (s.history) state.w_h = HISTORY[s.history];
    state.filters = { ...mergeFilters(f, s), min_rating: f.min_rating };
    state.draft = null; $("say-in").value = "";
  }
  update();
}

async function onChange(chg, arg, el) {
  const f = state.filters, v = el.value;
  if (chg === "search") return;                      // typing is handled by the input listener below
  state.notice = null;
  if (chg === "profile") { setProfile(+arg, v); await settle(); }
  if (chg === "startHow") {
    const m = state.members[+arg], home = isMine(m) ? m.mine?.borough : profileOf(m.profileId).home_borough;
    if (v === "usual") m.start = usualStart(m);
    if (v === "nta") { const a = D.areas.find((x) => x.borough === home) ?? D.areas[0]; m.start = { lat: +a.at[1].toFixed(3), lon: +a.at[0].toFixed(3), how: "nta", code: a.code, label: a.name.split("-")[0] }; }
    if (v === "gps") m.start = { ...m.start, how: "gps", fix: false };
  }
  if (chg === "startNta") { const a = D.areas.find((x) => x.code === v); state.members[+arg].start = { lat: +a.at[1].toFixed(3), lon: +a.at[0].toFixed(3), how: "nta", code: a.code, label: a.name.split("-")[0] }; }
  if (chg === "compare") { state.compareWith = +v; return renderDrawer(); }
  if (chg === "fcuisine") f.cuisines = el.checked ? [...f.cuisines, arg] : f.cuisines.filter((c) => c !== arg);
  if (chg === "fneed") f.needs = el.checked ? [...f.needs, arg] : f.needs.filter((c) => c !== arg);
  if (chg === "fstars") f.min_rating = el.checked ? 4 : null;
  if (chg === "fpricemin") { f.price_min = v === "" ? null : +v; if (f.price_max != null && f.price_min != null && f.price_min > f.price_max) f.price_max = null; }
  if (chg === "fpricemax") { f.price_max = v === "" ? null : +v; if (f.price_max != null && f.price_min != null && f.price_min > f.price_max) f.price_min = null; }
  if (chg === "fborough") f.borough = v || null;
  if (chg === "fhistory") state.w_h = +v;
  refresh();
}

// Search for a place to like: asked a moment after the typing stops, newest question wins.
let searchTimer = null, searchCtl = null;
root.addEventListener("input", (e) => {
  const box = e.target.closest("[data-search]");
  if (!box) return;
  const k = +box.dataset.search, q = box.value.trim(), s = state.search[k];
  s.q = box.value;
  clearTimeout(searchTimer); searchCtl?.abort();
  const show = () => swap($(`results-${k}`), resultsHTML(k));
  if (q.length < 2) { s.results = null; s.note = ""; return show(); }
  searchTimer = setTimeout(async () => {
    searchCtl = new AbortController();
    try { s.results = await searchPlaces(q, searchCtl.signal); s.note = ""; }
    catch (err) { if (err.name === "AbortError") return; s.results = null; s.note = FAIL[err.message] ?? FAIL.down; }
    if (state.search[k] === s) show();
  }, 250);
});

// Location: only after this tap, rounded to three decimals (about 100 m), never stored or sent.
function locate(k) {
  const m = state.members[k];
  if (!navigator.geolocation) { state.gps[k] = { error: "This browser cannot share a location. Choose a neighborhood instead." }; return refresh(); }
  state.gps[k] = { busy: true }; refresh();
  navigator.geolocation.getCurrentPosition((pos) => {
    const lat = +pos.coords.latitude.toFixed(3), lon = +pos.coords.longitude.toFixed(3);
    if (lat < NYC.lat[0] || lat > NYC.lat[1] || lon < NYC.lon[0] || lon > NYC.lon[1]) state.gps[k] = { error: "You seem to be outside New York City, where all these places are. Choose a neighborhood instead." };
    else { m.start = { lat, lon, how: "gps", fix: true, label: "my location" }; state.gps[k] = {}; }
    refresh();
  }, () => { state.gps[k] = { error: "The location was not shared. Choose a neighborhood, or move the start on the map." }; refresh(); }, { maximumAge: 60000, timeout: 15000 });
}

root.addEventListener("click", (e) => {
  const a = e.target.closest("[data-act]");
  if (a) { e.preventDefault(); onAct(a.dataset.act, a.dataset.arg, a); return; }
  if (e.target.matches("[data-scrim]")) onAct("closeSheet");
});
root.addEventListener("change", (e) => { const c = e.target.closest("[data-chg]"); if (c) onChange(c.dataset.chg, c.dataset.arg, c); });
root.addEventListener("submit", async (e) => {
  if (e.target.id !== "say") return;
  e.preventDefault();
  const sentence = $("say-in").value.trim().replace(/\s+/g, " ");
  if (!sentence) { $("list").scrollIntoView({ behavior: "smooth", block: "start" }); return; }
  state.saying = true; state.draft = null; $("say-go").disabled = true; renderHeard();
  state.draft = await readSentence(sentence);
  state.saying = false; $("say-go").disabled = false; renderHeard();
});
// quick-filter menus: one open at a time, remembered across re-renders, closed by a click elsewhere or Escape
root.addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.dataset?.menu) { if (d.open) { state.menu = d.dataset.menu; root.querySelectorAll("details[data-menu][open]").forEach((o) => { if (o !== d) o.open = false; }); } else if (state.menu === d.dataset.menu) state.menu = null; }
  // a section is filled when it is first opened; the re-render inserts an open section, whose own toggle event finds it filled
  if (d.dataset?.sec) { state.open[d.dataset.sec] = d.open; if (d.open && !d.querySelector(".inner").childElementCount) renderDrawer(); }
}, true);
document.addEventListener("click", (e) => { if (state.menu && !e.target.closest("details[data-menu]")) { state.menu = null; root.querySelectorAll("details[data-menu][open]").forEach((o) => { o.open = false; }); } });
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (state.sheet) onAct("closeSheet");
  else if (state.menu) { const id = state.menu; state.menu = null; root.querySelectorAll("details[data-menu][open]").forEach((o) => { o.open = false; }); root.querySelector(`[data-fk="menu|${id}"]`)?.focus(); }
  else if (state.pinning) { state.pinning = null; update(); }
  else if (state.selected != null) onAct("close");
});
root.addEventListener("mouseover", (e) => { const c = e.target.closest(".card"); if (c && +c.dataset.place !== state.hot) { state.hot = +c.dataset.place; markHot(); } });
root.addEventListener("mouseout", (e) => { const c = e.target.closest(".card"); if (c && !c.contains(e.relatedTarget)) { state.hot = null; markHot(); } });

boot().catch((err) => {
  root.innerHTML = `<p class="start" role="alert">Eat Buddy could not start: ${esc(err.message)}. ${import.meta.env.DEV ? "Run src/41_export_app.py to make the app data." : "Reload the page to try again."}</p>`;
  console.error(err);
});
