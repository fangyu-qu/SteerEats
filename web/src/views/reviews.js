// "What reviewers say about this place": the themes a language model found in a restaurant's written reviews
// (src/30_theme_summaries.py, published in web/public/data/themes/ and checked by tests/check_themes.mjs).
// It follows the restaurant selected anywhere on the page. It is a third question, kept apart from the two cards
// before it: not how the settings moved a rank, not what the model's score is made of, but what people who went
// there wrote. The rankers on this page do not read these summaries, and the card says so.
// An entry is null (no summary), { n, few } (too few written reviews) or { n, k, mean, themes: [{ label, tone,
// text, n, stars, quotes: [[text, stars, "YYYY-MM", translated]] }] }: n written reviews, k of them sampled.
import { esc } from "./common.js";

const STAR = '<span class="rv-star" aria-label="stars">★</span>';
const TONE = { positive: ["+", "Mostly positive"], negative: ["−", "Mostly negative"], mixed: ["±", "Mixed"] };
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const month = (ym) => `${MONTHS[+ym.slice(5, 7) - 1]} ${ym.slice(0, 4)}`;
// who read the reviews, by the name people know it by: "claude-sonnet-5-5" in the summaries' index is Claude Sonnet 5.5
const reader = (meta) => {
  const parts = (meta?.model ?? "").split("-"), k = parts.findIndex((x) => /^\d+$/.test(x));
  if (!meta?.model || k < 1) return "a language model";
  return `${parts.slice(0, k).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ")} ${parts.slice(k).join(".")}`;
};
const reviews = (n) => `${n.toLocaleString("en-US")} ${n === 1 ? "review" : "reviews"}`;

// ctx: { name, entry (see above; undefined while loading), meta (themes/index.json) or null, missing: no file for this user }
export function renderReviews(root, ctx) {
  const { name, entry, meta } = ctx;
  if (name == null) { root.innerHTML = `<p class="why-empty">Select a restaurant to see what its reviewers wrote.</p>`; return; }
  const head = `<h3 class="why-name">${esc(name)}</h3>`;
  if (ctx.missing) { root.innerHTML = `${head}<p class="rv-plain">No review summaries are published for this test user.</p>`; return; }
  if (entry === undefined) { root.innerHTML = `${head}<p class="rv-plain">Loading the review summary…</p>`; return; }
  if (entry === null) { root.innerHTML = `${head}<p class="rv-plain">No summary of the reviews was made for this place.</p>`; return; }
  if (entry.few || !entry.themes) {
    root.innerHTML = `${head}<p class="rv-plain">Only ${reviews(entry.n)} with text before the snapshot: too few to summarise${meta ? ` (at least ${meta.min_reviews} are needed)` : ""}.</p>`;
    return;
  }
  const before = meta ? ` from before ${LONG[+meta.cutoff.slice(5, 7) - 1]} ${meta.cutoff.slice(0, 4)}` : "";
  root.innerHTML = `${head}
    <p class="rv-scope">Themes ${esc(reader(meta))} found in <b>${entry.k}</b> of this place’s <b>${entry.n.toLocaleString("en-US")}</b> written reviews${before}. The rankers on this page do not read these summaries.</p>
    <ul class="rv-themes">${entry.themes.map((th) => {
      const [mark, tone] = TONE[th.tone] ?? ["", th.tone], q = th.quotes?.[0];
      return `<li>
        <p class="rv-head"><b>${esc(th.label)}</b><span class="rv-tone ${esc(th.tone)}"><i aria-hidden="true">${mark}</i>${tone}</span></p>
        <p class="rv-stat">${th.n} of the ${entry.k} sampled reviews · ${th.stars.toFixed(1)} ${STAR} on average (${entry.mean.toFixed(1)} ${STAR} over the whole sample)</p>
        <p class="rv-text ${esc(th.tone)}">${esc(th.text)}</p>
        ${q ? `<blockquote class="rv-quote"><q>${esc(q[0])}</q><span>${q[1]} ${STAR} review, ${month(q[2])}${q[3] ? ", translated" : ""}</span></blockquote>` : ""}</li>`;
    }).join("")}</ul>`;
}
