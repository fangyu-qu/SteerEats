// One-sentence steering: what a sentence can set, the keyword reader, and how the understood wishes become
// the app's own settings. No ranking happens here: targets go to the solver in steer.js, and filters only
// narrow the candidate pool the solver then ranks.
//
// The language model (server/nl.py) and the keyword reader below return the same `slots` object:
//   in_scope    false when the sentence is not about choosing a place to eat
//   max_km      number | null     top 10 within this average distance (solver target)
//   popularity  "less_mainstream" | "hidden_gems" | null   (solver target)
//   variety     "some" | "lots" | null                      (solver target)
//   history     "usual" | "mix" | "new" | null              (weight of past visits)
//   cuisines    [cuisine group key]           filter
//   price_max, price_min   1..4 | null        filter ($ to $$$$)
//   needs       [attribute key]               filter
//   borough     name | null                   filter
//   area        text | null                   filter: neighbourhoods of this user's candidates whose name holds it
//   unsupported [short phrases]               wishes the app has no setting for (shown, never applied)
// A slot that is null or empty means "not mentioned": the current setting stays as it is.

export const CUISINES = {
  chinese: "Chinese", korean: "Korean", japanese: "Japanese", southeast_asian: "Southeast Asian", south_asian: "South Asian",
  asian_other: "Asian, other", pizza: "Pizza", italian: "Italian", mexican: "Mexican", caribbean: "Caribbean",
  latin_american: "Latin American", middle_eastern: "Middle Eastern & Mediterranean", african: "African", european: "European",
  jewish_kosher: "Jewish & kosher", seafood: "Seafood", vegetarian_healthy: "Vegetarian & healthy", american: "American",
  breakfast_brunch: "Breakfast & brunch", cafe_dessert: "Cafe, bakery & dessert", sandwich_deli: "Sandwich & deli",
  fast_food: "Fast food & chicken", bar_pub: "Bar & pub", venue: "Venue, market & catering", general: "General",
};
export const NEEDS = {
  vegetarian_options: "Vegetarian options", halal: "Halal", late_night: "Open late", alcohol: "Serves alcohol",
  reservations: "Takes reservations", good_for_groups: "Good for groups", good_for_kids: "Good for kids",
};
export const BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"];
export const POPULARITY = { less_mainstream: 0.4, hidden_gems: 0.2 };   // top 10's mean popularity percentile, at most
export const VARIETY = { some: 6, lots: 8 };                            // cuisine groups in the top 10, at least
export const HISTORY = { usual: 1, mix: 0.5, new: 0 };                  // weight of past visits
export const KM_RANGE = [0.5, 6], KM_STEP = 0.25;
export const MAX_SENTENCE = 200;

export const emptySlots = () => ({ in_scope: true, max_km: null, popularity: null, variety: null, history: null, cuisines: [],
  price_max: null, price_min: null, needs: [], borough: null, area: null, unsupported: [] });
export const emptyFilters = () => ({ cuisines: [], price_max: null, price_min: null, needs: [], borough: null, area: null });
export const hasFilters = (f) => f.cuisines.length > 0 || f.price_max != null || f.price_min != null || f.needs.length > 0 || !!f.borough || !!f.area;

const price = (v) => (Number.isInteger(v) && v >= 1 && v <= 4 ? v : null);
const uniq = (list, allowed) => [...new Set(Array.isArray(list) ? list : [])].filter((k) => allowed.includes(k));
const cleanText = (s, max) => (typeof s === "string" ? s : "").replace(/[^\p{L}\p{N} '&.-]/gu, " ").replace(/\s+/g, " ").trim().slice(0, max);

// Whatever a reader returned, as valid slots: unknown keys and values are dropped, numbers are put on the
// console's own scales. server/nl.py applies the same rules before it answers.
export function normalize(raw) {
  const s = emptySlots(), r = raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
  s.in_scope = r.in_scope !== false;
  if (Number.isFinite(r.max_km) && r.max_km > 0)
    s.max_km = Math.min(KM_RANGE[1], Math.max(KM_RANGE[0], Math.round(r.max_km / KM_STEP) * KM_STEP));
  // own keys only: "toString" or "__proto__" are not choices
  const choice = (v, table) => (typeof v === "string" && Object.hasOwn(table, v) ? v : null);
  s.popularity = choice(r.popularity, POPULARITY);
  s.variety = choice(r.variety, VARIETY);
  s.history = choice(r.history, HISTORY);
  s.cuisines = uniq(r.cuisines, Object.keys(CUISINES));
  s.price_max = price(r.price_max);
  s.price_min = price(r.price_min);
  if (s.price_max != null && s.price_min != null && s.price_min > s.price_max) s.price_min = null;
  s.needs = uniq(r.needs, Object.keys(NEEDS));
  s.borough = BOROUGHS.includes(r.borough) ? r.borough : null;
  s.area = cleanText(r.area, 60) || null;
  s.unsupported = (Array.isArray(r.unsupported) ? r.unsupported : []).map((t) => cleanText(t, 40)).filter(Boolean).slice(0, 4);
  if (!s.in_scope) return { ...emptySlots(), in_scope: false };
  return s;
}

// ---------------------------------------------------------------------------------------------
// Keyword reader: the fallback when the language model is not reachable, and the baseline it is measured
// against (src/40_nl_eval.py runs this file through tests/nl_keyword.mjs). English only. The cuisine
// patterns follow src/taxonomy.py.
// ---------------------------------------------------------------------------------------------
const CUISINE_WORDS = [
  ["chinese", /hong kong|cantonese|sichuan|szechuan|hunan|shanghainese|dim sum|dumpling|chinese|taiwanese|hot ?pot/],
  ["korean", /korean|bibimbap|kimchi/],
  ["japanese", /japanese|sushi|ramen|izakaya|udon|soba|yakitori|tempura|omakase/],
  ["southeast_asian", /thai|vietnamese|\bpho\b|banh mi|malaysian|indonesian|filipino|burmese|southeast asian/],
  ["south_asian", /indian|pakistani|bangladeshi|nepal|sri lankan|tibetan|south asian|\bcurry\b/],
  ["pizza", /pizza|\bslice\b/],
  ["italian", /italian|pasta|trattoria/],
  ["mexican", /tex-mex|mexican|tacos?\b|burrito|taqueria/],
  ["caribbean", /jamaican|haitian|trinidad|caribbean|\bjerk\b/],
  ["latin_american", /latin|peruvian|colombian|dominican|salvadoran|ecuadorian|argentin|brazilian|venezuelan|cuban|empanada/],
  ["middle_eastern", /mediterranean|greek|middle eastern|lebanese|turkish|israeli|falafel|gyro|kebab|persian|shawarma|hummus/],
  ["african", /ethiopian|african|nigerian|senegalese/],
  ["european", /french|bistro|spanish|tapas|portuguese|german|polish|russian|ukrainian|georgian|european|british|irish/],
  ["jewish_kosher", /jewish|kosher/],
  ["seafood", /seafood|oysters?|\bcrab|lobster|\bfish\b/],
  ["vegetarian_healthy", /\bsalads?\b|health food|healthy/],
  ["american", /american|burgers?\b|diner|soul food|cajun|barbecue|\bbbq\b|steak/],
  ["breakfast_brunch", /breakfast|brunch|pancake|waffle/],
  ["cafe_dessert", /coffee|\bcafe\b|bakery|bagel|donut|dessert|ice cream|pastry|bubble tea|boba/],
  ["sandwich_deli", /sandwich|\bdeli\b/],
  ["fast_food", /fast food|fried chicken|\bwings\b/],
  ["bar_pub", /\bpub\b|\bbar\b|beer garden/],
];
const NEED_WORDS = [
  ["vegetarian_options", /vegetarian|vegan|meatless/],
  ["halal", /halal/],
  ["late_night", /late[- ]night|open late|after midnight|midnight/],
  ["alcohol", /cocktails?|\bwine\b|\bbeer\b|\bdrinks\b|alcohol/],
  ["reservations", /reserv|book a table|booking/],
  ["good_for_groups", /large group|big group|group of|party of|\b([6-9]|1\d) (people|of us)/],
  ["good_for_kids", /\bkids?\b|children|family[- ]friendly|toddler|stroller/],
];

export function keywordParse(sentence) {
  const text = String(sentence ?? "").slice(0, MAX_SENTENCE), t = text.toLowerCase();
  const r = { cuisines: [], needs: [] };
  for (const [key, re] of CUISINE_WORDS) if (re.test(t)) r.cuisines.push(key);
  for (const [key, re] of NEED_WORDS) if (re.test(t)) r.needs.push(key);
  const km = t.match(/(\d+(?:\.\d+)?)\s*(km|kilomet|mi\b|miles?)/);
  if (km) r.max_km = +km[1] * (km[2].startsWith("k") ? 1 : 1.6);
  else if (/walk/.test(t)) r.max_km = 1;
  else if (/not (too |that )?far/.test(t)) r.max_km = 4;
  else if (/\bclose\b|nearby|near me|around here|around the corner/.test(t)) r.max_km = 2;
  if (/cheap|budget|inexpensive|low[- ]cost/.test(t)) r.price_max = 1;
  else if (/not (too |that |very )?(pricey|expensive)|affordable|moderate|reasonabl/.test(t)) r.price_max = 2;
  else if (/mid[- ]range/.test(t)) { r.price_max = 3; r.price_min = 2; }
  if (/fancy|upscale|fine dining|splurge|high[- ]end|special occasion/.test(t)) r.price_min = 3;
  if (/hidden gem|hole[- ]in[- ]the[- ]wall|underrated|off the beaten/.test(t)) r.popularity = "hidden_gems";
  else if (/touristy|mainstream|less (popular|crowded|known)|not (too |so )?(popular|crowded)|avoid (the )?crowds/.test(t)) r.popularity = "less_mainstream";
  if (/wide variety|lots of variety|as many (different )?(cuisines|kinds)/.test(t)) r.variety = "lots";
  else if (/variety|\bmix of\b|different (kinds|cuisines|options)|diverse/.test(t)) r.variety = "some";
  if (/something (new|different)|surprise me|never (tried|been)|not my usual|out of my comfort/.test(t)) r.history = "new";
  else if (/my usual|like (the )?places i|same as always|my favou?rites/.test(t)) r.history = "usual";
  r.borough = BOROUGHS.find((b) => t.includes(b.toLowerCase())) ?? null;
  // a place name: capitalised words after "in", "near" or "around" that are not a borough
  const place = text.match(/\b(?:in|near|around)\s+((?:the\s+)?[A-Z][\w']*(?:\s+[A-Z][\w']*){0,3})/);
  if (place) {
    const name = place[1].replace(/^the\s+/i, "");
    if (!BOROUGHS.some((b) => b.toLowerCase() === name.toLowerCase()) && !/^(New York|NYC|Manhattan)/i.test(name)) r.area = name;
  }
  // rules cannot tell whether a sentence is about eating at all, so they never say it is not
  return normalize({ ...r, in_scope: true });
}

// ---------------------------------------------------------------------------------------------
// Filters narrow the pool: a place stays only if it passes every one. Unknown price fails a price filter.
// ---------------------------------------------------------------------------------------------
const words = (s) => s.toLowerCase().replace(/\(.*?\)/g, " ").replace(/[^a-z0-9]+/g, " ").trim();

// Neighbourhood names among this case's candidates that hold `area` as whole words ("harlem" -> Harlem (North), ...).
export function matchArea(c, area) {
  const want = ` ${words(area)} `, hit = new Set();
  if (want.trim() === "") return hit;
  for (const name of new Set(c.candidates.nta_name)) if (name != null && ` ${words(name)} `.includes(want)) hit.add(name);
  return hit;
}

export function poolMask(c, f) {
  const n = c.cuisine.length, cand = c.candidates, fv = c.feature_values, tests = [];
  if (f.cuisines.length) { const want = new Set(f.cuisines); tests.push(["cuisines", (i) => want.has(c.cuisine_names[c.cuisine[i]])]); }
  if (f.price_max != null) tests.push(["price_max", (i) => cand.price[i] != null && cand.price[i] <= f.price_max]);
  if (f.price_min != null) tests.push(["price_min", (i) => cand.price[i] != null && cand.price[i] >= f.price_min]);
  for (const a of f.needs) tests.push([`needs:${a}`, (i) => fv[`attr_${a}`]?.[i] === 1]);
  if (f.borough) tests.push(["borough", (i) => cand.borough[i] === f.borough]);
  let areas = null;
  if (f.area) { areas = matchArea(c, f.area); tests.push(["area", (i) => areas.has(cand.nta_name[i])]); }
  const keep = Array.from({ length: n }, (_, i) => tests.every(([, t]) => t(i)));
  const counts = {};
  for (const [k, t] of tests) { let m = 0; for (let i = 0; i < n; i++) if (t(i)) m++; counts[k] = m; }
  return { keep, size: keep.filter(Boolean).length, of: n, counts, areas: areas ? [...areas].sort() : [] };
}

// The case restricted to the kept candidates: every per-candidate array is cut the same way, so the solver,
// the accounting and the views run on it unchanged. Scales (distance_n, popularity_n) stay those of the full
// candidate set. `pool.index` gives each kept candidate's position in the full case.
const WHOLE = new Set(["features", "feature_groups", "cuisine_names", "models", "user"]);
export function filterCase(c, keep, key) {
  const n = c.cuisine.length, index = [];
  for (let i = 0; i < n; i++) if (keep[i]) index.push(i);
  const cut = (v, name) => {
    if (WHOLE.has(name)) return v;
    if (Array.isArray(v)) return v.length === n ? index.map((i) => v[i]) : v;
    if (v && typeof v === "object") return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, cut(x, k)]));
    return v;
  };
  const out = cut(c, "");
  out.id = `${c.id}~${key}`;
  out.pool = { of: n, index };
  return out;
}

// Only the six filter keys of an object (slots carry more).
export const asFilters = (o) => ({ cuisines: [...(o.cuisines ?? [])], price_max: o.price_max ?? null, price_min: o.price_min ?? null,
  needs: [...(o.needs ?? [])], borough: o.borough ?? null, area: o.area ?? null });

export const filterKey = (f) => [f.cuisines.join("+"), f.price_min ?? "", f.price_max ?? "", f.needs.join("+"), f.borough ?? "", f.area ?? ""].join("|");

// ---------------------------------------------------------------------------------------------
// From slots to the app's settings.
// ---------------------------------------------------------------------------------------------
const dollars = (v) => "$".repeat(v);
const ordinal = (p) => `${Math.round(p)}th`;

// One chip per understood wish, in the words the console uses. `key` identifies it for removal.
export function chips(slots) {
  const out = [];
  if (slots.max_km != null) out.push({ key: "max_km", kind: "target", text: `Keep it close: top 10 within ${slots.max_km} km on average` });
  if (slots.popularity) out.push({ key: "popularity", kind: "target", text: `Less mainstream: average below the ${ordinal(POPULARITY[slots.popularity] * 100)} percentile` });
  if (slots.variety) out.push({ key: "variety", kind: "target", text: `Cuisine variety: at least ${VARIETY[slots.variety]} groups` });
  if (slots.history) out.push({ key: "history", kind: "target", text: `Past visits count ${Math.round(HISTORY[slots.history] * 100)}%` });
  for (const k of slots.cuisines) out.push({ key: `cuisines:${k}`, kind: "filter", text: `Only ${CUISINES[k]}` });
  if (slots.price_max != null) out.push({ key: "price_max", kind: "filter", text: `Price up to ${dollars(slots.price_max)}` });
  if (slots.price_min != null) out.push({ key: "price_min", kind: "filter", text: `Price from ${dollars(slots.price_min)}` });
  for (const k of slots.needs) out.push({ key: `needs:${k}`, kind: "filter", text: NEEDS[k] });
  if (slots.borough) out.push({ key: "borough", kind: "filter", text: `In ${slots.borough}` });
  if (slots.area) out.push({ key: "area", kind: "filter", text: `Neighborhood: ${slots.area}` });
  return out;
}

export function withoutChip(slots, key) {
  const s = { ...slots, cuisines: [...slots.cuisines], needs: [...slots.needs] };
  const [name, item] = key.split(":");
  if (item) s[name] = s[name].filter((k) => k !== item);
  else s[name] = null;
  return s;
}

// The filters after a sentence: a filter the sentence mentions replaces the current one, the rest stay.
export function mergeFilters(current, slots) {
  const f = { ...current, cuisines: [...current.cuisines], needs: [...current.needs] };
  if (slots.cuisines.length) f.cuisines = [...slots.cuisines];
  if (slots.price_max != null) f.price_max = slots.price_max;
  if (slots.price_min != null) f.price_min = slots.price_min;
  if (f.price_max != null && f.price_min != null && f.price_min > f.price_max) {
    if (slots.price_max != null && slots.price_min == null) f.price_min = null; else f.price_max = null;
  }
  if (slots.needs.length) f.needs = [...new Set([...f.needs, ...slots.needs])];
  if (slots.borough) f.borough = slots.borough;
  if (slots.area) f.area = slots.area;
  return f;
}

// The console targets after a sentence, in the console's own units; null = leave as it is.
export function targetChanges(slots) {
  return {
    km: slots.max_km,
    popularity: slots.popularity ? POPULARITY[slots.popularity] : null,
    cuisines: slots.variety ? VARIETY[slots.variety] : null,
    w_h: slots.history ? HISTORY[slots.history] : null,
  };
}

// Active filters as removable chips (for the pool line under the say-it box).
export function filterChips(f) {
  return chips({ ...emptySlots(), ...f });
}
