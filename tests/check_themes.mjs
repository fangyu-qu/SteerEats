// Theme summaries (src/30_theme_summaries.py) as published for the detail view: every file must have the shape
// the view reads, cite only reviews from before the snapshot, and hold nothing but short quotes without contact
// details. Whether a quote is in the review it cites is checked by the export script, which has the review text;
// this check runs on the committed files alone.
//   node tests/check_themes.mjs
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dir = join(root, "web", "public", "data", "themes");
const read = (p) => JSON.parse(readFileSync(p, "utf8"));
if (!existsSync(join(dir, "index.json"))) {
  console.log("No theme summaries published (web/public/data/themes/index.json is missing); nothing to check.");
  process.exit(0);
}
const meta = read(join(dir, "index.json"));
const cutoffMonth = meta.cutoff.slice(0, 7);
const CONTACT = /https?:\/\/|www\.|\S+@\S+\.\S+|@\w{3,}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}/i;
const TONES = new Set(["positive", "negative", "mixed"]);
const MAX_QUOTE_CHARS = 220, MAX_QUOTE_WORDS = 32, MIN_QUOTE_WORDS = 4;

let failures = 0, quotes = 0, themes = 0;
const fail = (msg) => { failures++; if (failures <= 20) console.log("FAIL", msg); };
const seen = new Map();   // gmap_id -> its entry as JSON, to confirm a restaurant reads the same in every case
const tally = { summarised: 0, too_few: 0, missing: 0 };

for (const id of meta.cases) {
  const c = read(join(root, "web", "public", "data", "cases", `${id}.json`));
  const t = read(join(dir, `${id}.json`));
  const ids = c.candidates.gmap_id;
  if (t.id !== id || t.restaurants.length !== ids.length) { fail(`${id}: ${t.restaurants.length} entries for ${ids.length} candidates`); continue; }
  t.restaurants.forEach((r, i) => {
    const where = `${id}[${i}]`, text = JSON.stringify(r);
    if (seen.has(ids[i])) { if (seen.get(ids[i]) !== text) fail(`${where}: differs from the same restaurant in another case`); return; }
    seen.set(ids[i], text);
    if (r === null) { tally.missing++; return; }
    if (r.few) { tally.too_few++; if (!(r.n < meta.min_reviews)) fail(`${where}: marked few with ${r.n} text reviews`); return; }
    tally.summarised++;
    if (!(Number.isInteger(r.k) && r.k >= meta.min_reviews && r.k <= meta.sample && r.k <= r.n)) fail(`${where}: sample size ${r.k} of ${r.n}`);
    if (r.n > c.candidates.reviews[i]) fail(`${where}: more text reviews (${r.n}) than reviews before the snapshot (${c.candidates.reviews[i]})`);
    if (!(r.mean >= 1 && r.mean <= 5)) fail(`${where}: sample mean ${r.mean}`);
    if (!(Array.isArray(r.themes) && r.themes.length >= 1 && r.themes.length <= 5)) { fail(`${where}: ${r.themes?.length} themes`); return; }
    for (const th of r.themes) {
      themes++;
      if (!th.label || th.label.length > 60) fail(`${where}: label "${th.label}"`);
      if (!th.text || th.text.length > 300) fail(`${where}: summary of ${th.text?.length} characters`);
      if (!TONES.has(th.tone)) fail(`${where}: tone ${th.tone}`);
      if (!(Number.isInteger(th.n) && th.n >= 1 && th.n <= r.k)) fail(`${where}: ${th.n} of ${r.k} reviews`);
      if (!(th.stars >= 1 && th.stars <= 5)) fail(`${where}: mean rating ${th.stars}`);
      if (meta.quotes !== Array.isArray(th.quotes)) { fail(`${where}: quotes ${meta.quotes ? "missing" : "present although switched off"}`); continue; }
      if (!meta.quotes) continue;
      if (th.quotes.length < 1 || th.quotes.length > 2) fail(`${where}: ${th.quotes.length} quotes`);
      for (const [q, star, month, translated] of th.quotes) {
        quotes++;
        const words = q.split(" ").length;
        if (q.length > MAX_QUOTE_CHARS || words > MAX_QUOTE_WORDS || words < MIN_QUOTE_WORDS) fail(`${where}: quote of ${words} words, ${q.length} characters`);
        if (CONTACT.test(q)) fail(`${where}: quote with a link, handle or phone number`);
        if (![1, 2, 3, 4, 5].includes(star)) fail(`${where}: quote from a ${star}-star review`);
        if (!/^\d{4}-\d{2}$/.test(month) || !(month < cutoffMonth)) fail(`${where}: quote from ${month}, not before the snapshot`);
        if (translated !== 0 && translated !== 1) fail(`${where}: translated flag ${translated}`);
      }
    }
  });
}
for (const [k, v] of Object.entries(tally)) if (meta[k] !== v) fail(`index.json says ${meta[k]} ${k}, the files hold ${v}`);
if (meta.restaurants !== seen.size) fail(`index.json says ${meta.restaurants} restaurants, the files hold ${seen.size}`);

console.log(`${seen.size} restaurants (${tally.summarised} summarised, ${tally.too_few} too few reviews, ${tally.missing} without a summary), `
  + `${themes} themes, ${quotes} quotes: ${failures} failures`);
process.exit(failures ? 1 : 0);
