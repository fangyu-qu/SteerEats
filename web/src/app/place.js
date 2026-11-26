// Place drawer (the "Why" layer): who it suits and why in plain words, then, one tap further in, every reason
// with numbers, the place's stars over time, what reviewers wrote and a comparison with another listed place.
import * as d3 from "d3";
import { expand, monthLabel, rolling, snapshot } from "../detail.js";
import { esc } from "../views/common.js";

export const GROUP_WORDS = {
  Collaborative: "People with similar visits come here", Popularity: "How busy it has been", Rating: "Its star ratings",
  Location: "Where it is, seen from the usual area", Taste: "Cuisine, price and chain habits", Attributes: "What it offers (hours, groups, diet)",
};
const signed = (v, d = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
const num = (v, d = 2) => `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`;
const bar = (v, max, cls = "") => {
  const w = max > 0 ? Math.min(50, Math.abs(v) / max * 50) : 0;
  return `<div class="track"><div class="bar ${cls}" style="${v >= 0 ? `left:50%` : `right:50%`};width:${Math.max(w, 0.8)}%"></div></div>`;
};

// One member's score for one place, part by part. `x` holds the numbers (see main.js parts()).
function ledger(x, who, avatar, scale) {
  const rows = x.groups.map(([g, v]) => `<span>${GROUP_WORDS[g]}</span>${bar(v, scale.model)}<span class="v">${signed(v)}</span>`).join("");
  // the six parts add up to the score, except on rare rows where interventional TreeSHAP leaves a remainder
  const rest = x.raw - x.base - x.groups.reduce((a, [, v]) => a + v, 0);
  const settings = [];
  if (x.history !== 0) settings.push(["Past visits count less", x.history]);
  if (x.distance !== 0) settings.push([`Distance (${x.km.toFixed(1)} km)`, x.distance]);
  if (x.popular !== 0) settings.push(["Less mainstream", x.popular]);
  return `<div class="who2"><img class="cut" src="${avatar}" alt="">${esc(who)}</div>
    ${rows}
    <p class="sum">Model score ${num(x.raw)} = a typical place (${num(x.base)}) plus the six parts above${Math.abs(rest) >= 0.005 ? `, and ${signed(rest)} that the explanation method could not assign to a part` : ""}.</p>
    ${settings.map(([t, v]) => `<span>${t}</span>${bar(v, scale.set, "set")}<span class="v">${signed(v)}</span>`).join("")}
    <p class="sum">${settings.length ? "After the settings" : "No setting changes this score"}: suits ${esc(who)} ${x.sat.toFixed(2)} out of 1 (1 is this person's best place in the pool, 0 the worst).</p>`;
}

function starsChart(rest, meta, shown) {
  const h = expand(rest, meta.months), roll = rolling(h, 12, meta), snap = snapshot(rest, h, meta);
  const W = 360, H = 190, m = { l: 22, r: 8, t: 24, b: 26 }, from = meta.chart_start, cut = meta.cutoff_index;
  const x = d3.scaleLinear([from, meta.months - 1], [m.l, W - m.r]), y = d3.scaleLinear([1, 5], [H - m.b, m.t]);
  const line = d3.line().defined((d) => d[1] != null).x((d) => x(d[0])).y((d) => y(d[1].mean));
  const pts = roll.map((r, k) => [k, r]).filter(([k]) => k >= from);
  const years = d3.range(from, meta.months).filter((k) => monthLabel(meta, k).endsWith("-01"));
  const later = snap.later;
  return `<p>${snap.all ? `${snap.all.mean.toFixed(1)} stars on average from ${snap.all.n.toLocaleString("en-US")} reviews before the snapshot.` : "No reviews before the snapshot."}
      ${later ? ` Afterwards, ${later.n.toLocaleString("en-US")} more averaged ${later.mean.toFixed(1)}; the model never saw those.` : ""}</p>
    <p>The card shows ${shown.toFixed(1)}: the same reviews, pulled a little toward the city average, as the model sees them.</p>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Average star rating over the past twelve months, month by month">
      ${[1, 2, 3, 4, 5].map((s) => `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(s)}" y2="${y(s)}"/><text x="${m.l - 6}" y="${y(s) + 4}" text-anchor="end">${s}</text>`).join("")}
      ${years.map((k) => `<text x="${x(k)}" y="${H - 8}" text-anchor="middle">${monthLabel(meta, k).slice(0, 4)}</text>`).join("")}
      <path class="seen" d="${line(pts.filter(([k]) => k < cut)) ?? ""}"/>
      <path class="later" d="${line(pts.filter(([k]) => k >= cut - 1)) ?? ""}"/>
      <line class="cut" x1="${x(cut - 0.5)}" x2="${x(cut - 0.5)}" y1="4" y2="${H - m.b}"/>
      <text x="${x(cut - 0.5) - 6}" y="12" text-anchor="end">what the model saw</text>
      <text x="${Math.min(x(cut - 0.5) + 6, W - 34)}" y="12">later</text>
    </svg>
    <p class="soft">Line: average of the reviews of the twelve months up to each month. Dashed after the snapshot of ${esc(meta.cutoff)}.</p>`;
}

function themesBlock(t) {
  if (!t) return `<p>No summary of the reviews was made for this place.</p>`;
  if (t.few || !t.themes) return `<p>Only ${t.n} written ${t.n === 1 ? "review" : "reviews"} before the snapshot: too few to summarise.</p>`;
  return `<p class="soft">Themes a language model found in ${t.k} of this place's ${t.n.toLocaleString("en-US")} written reviews from before the snapshot. Every quote was checked against the review it comes from.</p>`
    + t.themes.map((th) => `<div class="theme"><p class="t">${esc(th.label)} <span class="soft">· ${esc(th.tone)} · ${th.n} reviews, ${th.stars.toFixed(1)} stars on average</span></p>
      <p>${esc(th.text)}</p>${th.quotes.slice(0, 1).map((q) => `<q>${esc(q[0])}</q>`).join("")}</div>`).join("");
}

// ctx: { place, facts, sticker, members:[{name, label, avatar, sat, km, reason, parts}], two, open:{}, detail, detailMeta, themes, loading,
//        others:[{i, name}], other:{name, members:[{sat, km, reason}]} | null, compareWith, picked }
export function drawerHTML(c) {
  const scale = { model: Math.max(...c.members.flatMap((m) => m.parts.groups.map(([, v]) => Math.abs(v))), 1e-9),
    set: Math.max(...c.members.flatMap((m) => [m.parts.history, m.parts.distance, m.parts.popular].map(Math.abs)), 1e-9) };
  const sec = (id, title, body) => `<details data-sec="${id}"${c.open[id] ? " open" : ""}><summary>${title}</summary><div class="inner">${c.open[id] ? body() : ""}</div></details>`;
  return `<section class="drawer" aria-label="Why this place" tabindex="-1">
    <div class="head">
      <div class="id"><img class="sticker" src="${c.sticker}" alt=""><div><h2>${esc(c.place.name)}</h2><p class="facts">${esc(c.facts)}</p></div></div>
      <button type="button" class="btn herb x" data-act="close" aria-label="Close"><svg viewBox="0 0 18 18" aria-hidden="true"><path d="M3 3l12 12M15 3L3 15"/></svg></button>
    </div>
    <h3 class="caps">${c.two ? "Why it works for you two" : "Why it works for you"}</h3>
    ${c.members.map((m, k) => `<div class="mem m${k}"><img class="cut" src="${m.avatar}" alt=""><div>
        <p class="nm">${esc(m.label)}</p><p class="hand">${esc(m.reason)}</p>
        <p class="fine num">Suits ${k === 0 ? "you" : esc(m.name)} ${m.sat.toFixed(2)} out of 1 · ${m.km.toFixed(1)} km from ${k === 0 ? "your" : "their"} start</p></div></div>`).join("")}
    <div class="more">
      ${sec("numbers", "Every reason, with numbers", () => `<div class="key2"><span><i></i>what the model saw (its own score units)</span><span><i class="set"></i>your settings (0 to 1 scale)</span></div>
        <div class="ledger num">${c.members.map((m) => ledger(m.parts, m.name, m.avatar, scale)).join("")}</div>
        <p class="soft">The model parts are SHAP values: how much each kind of information moved this place's score away from a typical place. They explain the score, not the rank.</p>`)}
      ${sec("stars", "Its stars over time", () => (c.detail ? `<div class="stars">${starsChart(c.detail, c.detailMeta, c.place.rating)}</div>` : `<p>${c.loading ? "Loading the rating history." : "No rating history is available for this place."}</p>`))}
      ${sec("reviews", "What reviewers wrote", () => (c.themes !== undefined ? themesBlock(c.themes) : `<p>${c.loading ? "Loading the review summary." : "No review summary is available for this place."}</p>`))}
      ${c.others.length ? sec("compare", "Compare with another place", () => `<label class="field">Compare with
          <select data-chg="compare">${c.others.map((o) => `<option value="${o.i}"${o.i === c.compareWith ? " selected" : ""}>${esc(o.name)}</option>`).join("")}</select></label>
        ${c.other ? `<table class="cmp"><thead><tr><th></th><th>${esc(c.place.name)}</th><th>${esc(c.other.name)}</th></tr></thead><tbody>
          ${c.members.map((m, k) => `<tr><th>Suits ${esc(m.name)}</th><td class="num">${m.sat.toFixed(2)}</td><td class="num">${c.other.members[k].sat.toFixed(2)}</td></tr>
            <tr><th>Distance for ${esc(m.name)}</th><td class="num">${m.km.toFixed(1)} km</td><td class="num">${c.other.members[k].km.toFixed(1)} km</td></tr>
            <tr><th>Main reason for ${esc(m.name)}</th><td>${esc(m.reason)}</td><td>${esc(c.other.members[k].reason)}</td></tr>`).join("")}
          </tbody></table>` : ""}`) : ""}
    </div>
    <button type="button" class="btn go" data-act="pick">${c.picked ? "This is tonight's pick" : "Let's go here"}</button>
  </section>`;
}
