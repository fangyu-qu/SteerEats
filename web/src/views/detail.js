// Restaurant detail (V8): what the selected place's stars say, and whether they changed.
// Reviews per month and a rolling mean rating with a 95% band, split at the snapshot: left of it is what the
// rankers could see, right of it is what arrived later. Stars are data, not the model's score, so the view
// also states the smoothed ratings the rankers used and how far they moved this place's score.
// Marks are ink and greys only (the three effect hues keep their meaning): solid = seen by the rankers,
// hollow or dashed = later.
import * as d3 from "d3";
import { expand, monthLabel, rolling, snapshot, toDate } from "../detail.js";
import { MODEL_NAMES, esc } from "./common.js";

const MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const PRICE = [null, "$", "$$", "$$$", "$$$$"];
const ORDINAL = ["largest", "2nd largest", "3rd largest", "4th largest", "5th largest", "smallest"];
const stars = (v) => `${v.toFixed(2)}&nbsp;★`;
const int = (v) => v.toLocaleString("en-US");
const signed = (v) => (Math.abs(v) < 0.005 ? "" : v > 0 ? "+" : "−") + Math.abs(v).toFixed(2);   // no sign on a rounded zero

export function createDetail(root, meta) {
  let ctx = null, win = meta.recent_months, hover = null;
  const when = (m, long = false) => { const [y, mo] = monthLabel(meta, m).split("-"); return `${long ? d3.timeFormat("%B")(new Date(+y, +mo - 1, 1)) : MONTH[+mo - 1]} ${y}`; };
  const last = when(meta.months - 1, true);

  root.innerHTML = `
    <p class="detail-lead" id="detail-lead"></p>
    <div class="detail-grid">
      <figure class="lens-fig">
        <figcaption><strong>Rating and reviews by month</strong>
          <span>Line: mean star rating over the trailing window. Band: 95% interval for that mean, wide when the window holds few reviews. Bars: reviews posted in the month. Where the window holds no review the line stops. Just after the snapshot a window still holds some earlier reviews.</span></figcaption>
        <div class="detail-controls">
          <span id="detail-win-label">Trailing window</span>
          <div class="segmented" role="radiogroup" aria-labelledby="detail-win-label">
            ${meta.windows.map((w) => `<button type="button" role="radio" data-window="${w}" aria-checked="${w === win}">${w} months</button>`).join("")}
          </div>
        </div>
        <ul class="key detail-key" id="detail-key"></ul>
        <div class="lens-svg" id="detail-chart"></div>
        <p class="sim-note" id="detail-read" aria-live="polite"></p>
      </figure>
      <div class="detail-side">
        <figure class="lens-fig">
          <figcaption><strong>Star mix</strong>
            <span>Share of reviews at each star level, on one scale. The first two periods end at the snapshot; the third runs from it to ${last}.</span></figcaption>
          <div class="lens-svg" id="detail-mix"></div>
        </figure>
        <table class="feat" id="detail-facts"></table>
      </div>
    </div>`;
  root.querySelectorAll("button[data-window]").forEach((b) => b.addEventListener("click", () => {
    win = +b.dataset.window;
    root.querySelectorAll("button[data-window]").forEach((x) => x.setAttribute("aria-checked", String(+x.dataset.window === win)));
    if (ctx) { ctx.roll = rolling(ctx.h, win, meta); render(); }
  }));

  function update({ c, rest, i, model, rank }) {
    if (!rest) { ctx = null; root.classList.add("is-empty"); root.querySelector("#detail-lead").textContent = "Rating history is not available for this test user."; return; }
    root.classList.remove("is-empty");
    const h = expand(rest, meta.months);
    ctx = { c, rest, i, model, rank, h, roll: rolling(h, win, meta), snap: snapshot(rest, h, meta), perMonth: h.map((x) => d3.sum(x)) };
    hover = null;
    render();
  }

  function render() {
    if (!ctx) return;
    lead(); chart(); mix(); facts();
  }

  function lead() {
    const { c, i, snap, model } = ctx, cand = c.candidates;
    const price = PRICE[cand.price[i] ?? 0];
    const head = `<b>${esc(cand.name[i])}</b> <span class="muted">· ${esc(cand.cuisine_label[i])} · ${esc(cand.nta_name[i])} · ${price ? `price ${price}` : "price not listed"}${cand.is_chain[i] ? " · chain" : ""}</span>`;
    const a = snap.all, r = snap.recent, l = snap.later;
    let text = a ? `Before the snapshot it had <span class="num">${int(a.n)}</span> ${a.n === 1 ? "review" : "reviews"} averaging <span class="num">${stars(a.mean)}</span>. ` : "It had no review before the snapshot. ";
    if (r) {
      const d = r.mean - a.mean;
      text += `The last ${meta.recent_months} months average <span class="num">${stars(r.mean)}</span> from <span class="num">${int(r.n)}</span> (95% band <span class="num">${r.lo.toFixed(2)}–${r.hi.toFixed(2)}</span>), ${Math.abs(d) < 0.05 ? "the same as" : `<span class="num">${Math.abs(d).toFixed(2)}</span> ${d < 0 ? "below" : "above"}`} all time. `;
    } else if (a) text += `None of them came in the last ${meta.recent_months} months. `;
    if (l) {
      const base = r ?? a;
      text += `By ${last} it received <span class="num">${int(l.n)}</span> more, averaging <span class="num">${stars(l.mean)}</span> (band <span class="num">${l.lo.toFixed(2)}–${l.hi.toFixed(2)}</span>)`;
      text += base ? `: <span class="num">${Math.abs(l.mean - base.mean).toFixed(2)}</span> ${l.mean < base.mean ? "lower" : "higher"} than ${r ? "the year before the snapshot" : "before"}. ` : ". ";
    } else text += `No review arrived between the snapshot and ${last}. `;
    // Stars are an input: how far the rating features moved this model's score for this place.
    const groups = Object.keys(c.feature_groups).map((g) => ({ g, v: c[model].shap_groups[g][i] })).sort((x, y) => Math.abs(y.v) - Math.abs(x.v));
    const pos = groups.findIndex((x) => x.g === "Rating");
    text += `Stars are an input to the models, not their score: the rating features move ${MODEL_NAMES[model]}'s score for this place by <span class="num">${signed(groups[pos].v)}</span> log-odds, the ${ORDINAL[pos]} of its six feature groups.`;
    root.querySelector("#detail-lead").innerHTML = `${head}<br>${text}`;
  }

  function chart() {
    const node = root.querySelector("#detail-chart"), { roll, perMonth, snap } = ctx;
    const width = Math.max(node.clientWidth, 300), narrow = width < 520;
    const M = { l: 34, r: 12, t: 22, b: 20 }, H1 = 156, GAP = 34, H2 = 54, height = M.t + H1 + GAP + H2 + M.b;
    const m0 = meta.chart_start, m1 = meta.months - 1, cut = meta.cutoff_index;
    const x = d3.scaleLinear([m0 - 0.5, m1 + 0.5], [M.l, width - M.r]);
    const y = d3.scaleLinear([1, 5], [M.t + H1, M.t]);
    const yc = d3.scaleLinear([0, Math.max(1, d3.max(perMonth.slice(m0)))], [height - M.b, M.t + H1 + GAP]).nice(2);
    const xc = x(cut - 0.5), months = d3.range(m0, m1 + 1);
    const svg = d3.select(node).html("").append("svg").attr("width", width).attr("height", height).attr("role", "img")
      .attr("aria-label", `Monthly reviews and ${win}-month mean rating of ${ctx.c.candidates.name[ctx.i]}`);

    svg.append("rect").attr("class", "d-later-bg").attr("x", xc).attr("y", M.t - 8).attr("width", width - M.r - xc + 4).attr("height", height - M.b - M.t + 8);
    svg.append("g").selectAll("line").data([1, 2, 3, 4, 5]).join("line").attr("class", "d-grid").attr("x1", M.l).attr("x2", width - M.r).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
    svg.append("g").selectAll("text").data([1, 2, 3, 4, 5]).join("text").attr("class", "tk").attr("x", M.l - 6).attr("y", (d) => y(d)).attr("dy", "0.32em").attr("text-anchor", "end").text((d) => `${d}★`);

    // Rolling mean and band, drawn in two pieces that meet at the last month before the snapshot.
    const pt = (m) => ({ m, s: roll[m] });
    const pieces = [["seen", months.filter((m) => m < cut).map(pt)], ["later", months.filter((m) => m >= cut - 1).map(pt)]];
    const area = d3.area().defined((d) => d.s).x((d) => x(d.m)).y0((d) => y(d.s.lo)).y1((d) => y(d.s.hi));
    const line = d3.line().defined((d) => d.s).x((d) => x(d.m)).y((d) => y(d.s.mean));
    for (const [cls, data] of pieces) {
      svg.append("path").attr("class", `d-band ${cls}`).attr("d", area(data));
      svg.append("path").attr("class", `d-line ${cls}`).attr("d", line(data));
    }
    // a month whose neighbours are both blank would not show as a line
    const alone = months.filter((m) => roll[m] && !roll[m - 1] && !(m < m1 && roll[m + 1]));
    svg.append("g").selectAll("circle").data(alone).join("circle").attr("class", (m) => `d-dot ${m < cut ? "seen" : "later"}`).attr("cx", (m) => x(m)).attr("cy", (m) => y(roll[m].mean)).attr("r", 2.5);

    // the all-time mean the snapshot saw, as a reference for "did the recent rating move?" (named in the key:
    // a label inside the plot would sit on the band)
    if (snap.all) svg.append("line").attr("class", "d-ref").attr("x1", M.l).attr("x2", xc).attr("y1", y(snap.all.mean)).attr("y2", y(snap.all.mean));
    root.querySelector("#detail-key").innerHTML = `<li><i class="k-line"></i>mean of the past ${win} months</li><li><i class="k-band"></i>95% band</li>
      <li><i class="k-later"></i>after the snapshot</li>${snap.all ? `<li><i class="k-ref"></i>all-time mean at the snapshot, ${snap.all.mean.toFixed(2)}&nbsp;★</li>` : ""}`;

    const bw = Math.max(1, x(1) - x(0) - (narrow ? 0.6 : 1.6));
    svg.append("g").selectAll("rect").data(months.filter((m) => perMonth[m] > 0)).join("rect").attr("class", (m) => `d-bar ${m < cut ? "seen" : "later"}`)
      .attr("x", (m) => x(m) - bw / 2).attr("width", bw).attr("y", (m) => yc(perMonth[m])).attr("height", (m) => yc(0) - yc(perMonth[m]));
    svg.append("line").attr("class", "zero").attr("x1", M.l).attr("x2", width - M.r).attr("y1", yc(0)).attr("y2", yc(0));
    svg.append("g").selectAll("text").data(yc.ticks(2).filter((d) => d > 0 && Number.isInteger(d))).join("text").attr("class", "tk").attr("x", M.l - 6).attr("y", (d) => yc(d)).attr("dy", "0.32em").attr("text-anchor", "end").text((d) => d);
    svg.append("text").attr("class", "sub").attr("x", M.l).attr("y", M.t + H1 + GAP - 8).text("reviews in the month");

    const years = months.filter((m) => monthLabel(meta, m).endsWith("-01"));
    svg.append("g").selectAll("text").data(years).join("text").attr("class", "tk").attr("x", (m) => x(m - 0.5) + 3).attr("y", height - 6).text((m) => monthLabel(meta, m).slice(0, 4));
    svg.append("g").selectAll("line").data(years).join("line").attr("class", "d-grid").attr("x1", (m) => x(m - 0.5)).attr("x2", (m) => x(m - 0.5)).attr("y1", yc(0)).attr("y2", yc(0) + 5);

    svg.append("line").attr("class", "d-cut").attr("x1", xc).attr("x2", xc).attr("y1", M.t - 8).attr("y2", yc(0));
    svg.append("text").attr("class", "d-cut-label").attr("x", xc - 6).attr("y", M.t - 10).attr("text-anchor", "end").text(narrow ? "seen by the models" : "what the models could see");
    svg.append("text").attr("class", "d-cut-label later").attr("x", xc + 6).attr("y", M.t - 10).text(narrow ? "later" : "later reviews");

    const guide = svg.append("line").attr("class", "d-hover").attr("y1", M.t - 8).attr("y2", yc(0)).style("display", "none");
    const dot = svg.append("circle").attr("class", "d-pick").attr("r", 4).style("display", "none");
    const show = (m) => {
      hover = m;
      const on = m != null && m >= m0;
      guide.style("display", on ? null : "none").attr("x1", x(m ?? 0)).attr("x2", x(m ?? 0));
      dot.style("display", on && roll[m] ? null : "none");
      if (on && roll[m]) dot.attr("cx", x(m)).attr("cy", y(roll[m].mean));
      read(on ? m : null);
    };
    svg.append("g").selectAll("rect").data(months).join("rect").attr("class", "hit").attr("x", (m) => x(m - 0.5)).attr("width", x(1) - x(0))
      .attr("y", M.t - 8).attr("height", yc(0) - M.t + 8).on("pointerenter pointermove", (_, m) => show(m)).on("pointerleave", () => show(null));
    show(hover);
  }

  // One line under the chart: the hovered month, or the last month before the snapshot.
  function read(m) {
    const { roll, perMonth, rest, h } = ctx, cut = meta.cutoff_index;
    const at = m ?? cut - 1, s = roll[at], t = toDate(rest, h, at);
    const lead = m == null ? `<b>${when(at)}</b>, the last month before the snapshot` : `<b>${when(at)}</b>${at >= cut ? ", later than the snapshot" : ""}`;
    const month = `<span class="num">${int(perMonth[at])}</span> ${perMonth[at] === 1 ? "review" : "reviews"} in the month`;
    const windowText = s ? `past ${win} months <span class="num">${stars(s.mean)}</span> from <span class="num">${int(s.n)}</span> (95% band <span class="num">${s.lo.toFixed(2)}–${s.hi.toFixed(2)}</span>)` : `no review in the past ${win} months`;
    const all = t.n ? `all reviews to date <span class="num">${stars(t.mean)}</span> from <span class="num">${int(t.n)}</span>` : "no review yet";
    root.querySelector("#detail-read").innerHTML = `${lead}: ${month}; ${windowText}; ${all}.${m == null ? " Hover the chart for another month." : ""}`;
  }

  function mix() {
    const node = root.querySelector("#detail-mix"), { snap } = ctx;
    const periods = [
      { key: "seen", title: "All time", s: snap.all },
      { key: "seen", title: `Last ${meta.recent_months} months`, s: snap.recent },
      { key: "later", title: "Later", s: snap.later },
    ];
    const width = Math.max(node.clientWidth, 260), G = 16, M = { l: 30, r: 4, t: 34, b: 18 }, H = 96, height = M.t + H + M.b;
    const gw = (width - M.l - M.r - 2 * G) / 3;
    const share = (p, k) => (p.s ? p.s.counts[k] / p.s.n : 0);
    const top = Math.max(0.5, d3.max(periods, (p) => d3.max(d3.range(5), (k) => share(p, k))));
    const y = d3.scaleLinear([0, top], [M.t + H, M.t]).nice(2);
    const xk = d3.scaleBand(d3.range(5), [0, gw]).paddingInner(0.18);
    const svg = d3.select(node).html("").append("svg").attr("width", width).attr("height", height).attr("role", "img")
      .attr("aria-label", "Share of reviews at each star level before and after the snapshot");
    const ticks = y.ticks(2);
    svg.append("g").selectAll("line").data(ticks).join("line").attr("class", "d-grid").attr("x1", M.l).attr("x2", width - M.r).attr("y1", (d) => y(d)).attr("y2", (d) => y(d));
    svg.append("g").selectAll("text").data(ticks).join("text").attr("class", "tk").attr("x", M.l - 6).attr("y", (d) => y(d)).attr("dy", "0.32em").attr("text-anchor", "end").text((d) => `${Math.round(d * 100)}%`);
    const g = svg.selectAll("g.d-mix").data(periods).join("g").attr("class", (p) => `d-mix ${p.key}`).attr("transform", (_, j) => `translate(${M.l + j * (gw + G)},0)`);
    g.append("text").attr("class", "lens-col").attr("x", 0).attr("y", 11).text((p) => p.title);
    g.append("text").attr("class", "sub").attr("x", 0).attr("y", 25).text((p) => (p.s ? `${int(p.s.n)} · mean ${p.s.mean.toFixed(2)}` : "no reviews"));
    g.selectAll("rect").data((p) => d3.range(5).map((k) => ({ k, v: share(p, k), n: p.s ? p.s.counts[k] : 0 }))).join("rect")
      .attr("x", (d) => xk(d.k)).attr("width", xk.bandwidth()).attr("y", (d) => y(d.v)).attr("height", (d) => y(0) - y(d.v))
      .append("title").text((d) => `${d.k + 1} ★: ${int(d.n)} (${Math.round(d.v * 100)}%)`);
    g.selectAll("text.tk").data(d3.range(5)).join("text").attr("class", "tk").attr("x", (k) => xk(k) + xk.bandwidth() / 2).attr("y", height - 5).attr("text-anchor", "middle").text((k) => k + 1);
    svg.append("line").attr("class", "zero").attr("x1", M.l).attr("x2", width - M.r).attr("y1", y(0)).attr("y2", y(0));
  }

  function facts() {
    const { c, i, snap, model, rank } = ctx, cand = c.candidates, n = cand.gmap_id.length;
    const price = PRICE[cand.price[i] ?? 0];
    const rows = [
      ["Price level", price ? `<span class="mono">${price}</span>` : `Not listed <span class="grp">Google shows no price level for this place</span>`],
      ["Reviews before the snapshot", `<span class="mono">${int(cand.reviews[i])}</span> <span class="grp">${int(cand.recent_reviews[i])} in the last ${meta.recent_months} months</span>`],
      [`Reviews after it, to ${last}`, `<span class="mono">${snap.later ? int(snap.later.n) : 0}</span> <span class="grp">no model saw these</span>`],
      ["Ratings the models used", `<span class="mono">${cand.rating[i].toFixed(2)}</span> ★ all time · <span class="mono">${cand.recent_rating[i].toFixed(2)}</span> ★ last ${meta.recent_months} months
        <span class="grp">smoothed toward the city-wide ${meta.prior_mean.toFixed(2)} ★ with the weight of ${meta.prior_strength} reviews</span>`],
      [`${MODEL_NAMES[model]}'s own rank`, `<span class="mono">#${rank}</span> of ${n} candidates <span class="grp">before any steering; ratings are 3 of its ${c.features.length} features</span>`],
    ];
    root.querySelector("#detail-facts").innerHTML = `<caption>At the snapshot (${c.cutoff})</caption><tbody>${rows.map(([k, v]) => `<tr><th scope="row">${k}</th><td>${v}</td></tr>`).join("")}</tbody>`;
  }

  return { update, render };
}
