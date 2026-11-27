// Similarity map: restaurants placed by who reviewed them (SVD item factors before the snapshot, then UMAP;
// src/26_similarity_map.py). The same layout twice, drawn the same way: every restaurant a small warm-grey dot, one
// group picked out in brick red on top of it: a borough on the left, a cuisine group on the right. Only the
// highlight changes with a choice; the positions never do.
// The two numbers above the pictures are not read off the picture. For each restaurant the script takes its 15
// nearest restaurants in the 64-number reviewer space (itself left out) and asks whether the restaurant's own borough
// (or cuisine group) is the most common one among them: "observed" is the share of restaurants for which it is.
// "Chance" is what always answering with the largest borough (or group) would score. Per-group neighbour shares
// exist for cuisine groups only, so they are in "Method & data", not beside one panel and missing from the other.
import * as d3 from "d3";
import { esc } from "./common.js";

const LABEL = (s) => s.replace(/_/g, " ").replace(/^\w/, (ch) => ch.toUpperCase());
const f3 = (v) => (v == null ? "—" : v.toFixed(3));
const pct = (v) => `${Math.round(v * 100)}%`, pct1 = (v) => `${(v * 100).toFixed(1)}%`;

// els: { title, stats } are the card's heading, written from the numbers, and the box in its top right corner for the statistics
export function createSimilarity(root, sm, els = {}) {
  const m = Object.fromEntries(sm.metrics.map((r) => [r.space, r]));
  const fac = m["Item factors (64-d)"], um = m.UMAP, pc = m.PCA, noise = m["UMAP of same-size noise"];
  const N = sm.x.length, nn = sm.params.n_neighbors;
  const count = (arr, k) => { let n = 0; for (let i = 0; i < N; i++) if (arr[i] === k) n++; return n; };
  const boN = sm.boroughs.map((_, k) => count(sm.bo, k)), cuN = sm.cuisines.map((_, k) => count(sm.cu, k));
  // start with Brooklyn, and with the cuisine group closest to it in size, so the two pictures hold a similar
  // number of red dots ("general" has no cuisine and is left out of the choice)
  const pick = { bo: Math.max(0, sm.boroughs.indexOf("Brooklyn")), cu: 0 };
  pick.cu = sm.cuisines.map((c, k) => k).filter((k) => sm.cuisines[k] !== "general").reduce((a, k) => (Math.abs(cuN[k] - boN[pick.bo]) < Math.abs(cuN[a] - boN[pick.bo]) ? k : a));
  // the heading says what the numbers say: how far above chance each label is
  const gain = { bo: fac.knn_borough - sm.chance.knn_borough, cu: fac.knn_cuisine - sm.chance.knn_cuisine };
  if (els.title) els.title.textContent = gain.bo > gain.cu ? "Restaurants cluster more by borough than by cuisine"
    : gain.cu > gain.bo ? "Restaurants cluster more by cuisine than by borough" : "How restaurants cluster by borough and by cuisine";

  const panel = (key, title, options) => `
    <section class="sim-panel" aria-labelledby="sim-${key}-h">
      <div class="sim-head"><h3 id="sim-${key}-h">${title}</h3><select id="sim-${key}" aria-labelledby="sim-${key}-h">${options}</select></div>
      <p class="sim-count" id="sim-${key}-count"></p>
      <div class="sim-plot" id="sim-${key}-plot"><canvas role="img"></canvas></div>
    </section>`;
  // the two numbers against chance: in the card's top right corner (els.stats) when the page gives that place
  const STATS = `
    <table class="sim-stats">
      <caption>For each restaurant: is its own borough, or cuisine group, the most common one among its ${nn} nearest restaurants?</caption>
      <thead><tr><td></td><th scope="col">Observed</th><th scope="col">Chance baseline</th></tr></thead>
      <tbody>
        <tr><th scope="row">Same borough</th><td>${pct1(fac.knn_borough)}</td><td>${pct1(sm.chance.knn_borough)}</td></tr>
        <tr><th scope="row">Same cuisine</th><td>${pct1(fac.knn_cuisine)}</td><td>${pct1(sm.chance.knn_cuisine)}</td></tr>
      </tbody>
    </table>`;
  if (els.stats) els.stats.innerHTML = STATS;
  root.innerHTML = `
${els.stats ? "" : STATS}
    <div class="sim-pair">
      ${panel("bo", "By borough", sm.boroughs.map((b, k) => `<option value="${k}"${k === pick.bo ? " selected" : ""}>${esc(b)}</option>`).join(""))}
      ${panel("cu", "By cuisine", sm.cuisines.map((c, k) => [c, k]).sort((a, b) => a[0].localeCompare(b[0]))
        .map(([c, k]) => `<option value="${k}"${k === pick.cu ? " selected" : ""}>${esc(LABEL(c))}</option>`).join(""))}
    </div>
    <ul class="sim-key" aria-label="Legend"><li><i class="sel" aria-hidden="true"></i>Selected group</li><li><i class="other" aria-hidden="true"></i>Other restaurants</li></ul>
    <p class="sim-foot">Nearby dots indicate similar reviewer patterns; distances between clusters should not be interpreted literally.</p>
    <details class="why-more sim-method"><summary>Method &amp; data</summary><div id="sim-method"></div></details>`;
  const $ = (id) => root.querySelector("#" + id);

  $("sim-method").innerHTML = `
    <p><b>Data.</b> Reviews written before ${esc(sm.cutoff)}: ${sm.info.reviews.toLocaleString()} reviews by ${sm.info.users.toLocaleString()} people of ${sm.info.restaurants.toLocaleString()} restaurants. A restaurant is included when at least ${sm.params.min_reviewers} people reviewed it; a reviewer when they reviewed at least two of those restaurants.</p>
    <p><b>Layout.</b> Each restaurant is first described by ${sm.params.k} numbers that summarise who reviewed it (a truncated SVD of the reviewer-by-restaurant table, ${(sm.info.explained * 100).toFixed(1)}% of its variance). UMAP (n_neighbors ${nn}, min_dist ${sm.params.min_dist}, cosine distance, seed ${sm.params.seed}) then places them in two dimensions. Both panels draw that one layout; choosing a group only changes which dots are red.</p>
    <p><b>Nearest neighbours.</b> The two rows above the pictures use each restaurant’s ${nn} nearest restaurants in the ${sm.params.k}-number space, not in the picture, with the restaurant itself left out. “Observed” is the share of all ${N.toLocaleString()} restaurants whose own borough (or cuisine group) is the most common one among those ${nn}. “Chance baseline” is the score of always answering with the largest borough (or cuisine group), that is, its share of all restaurants. The baseline matters because five boroughs and ${sm.cuisines.length} cuisine groups of very different sizes are not equally easy to match.</p>
    <p><b>How faithful the picture is.</b> Trustworthiness asks whether neighbours in the picture are real neighbours, continuity whether real neighbours stay together (both at k = ${nn}, on a random sample of ${sm.params.subsample.toLocaleString()} restaurants). The UMAP layout scores ${f3(um.trustworthiness)} and ${f3(um.continuity)}; a linear PCA projection ${f3(pc.trustworthiness)} and ${f3(pc.continuity)}; the same UMAP on random data of the same size ${f3(noise.trustworthiness)} and ${f3(noise.continuity)}. UMAP keeps near neighbours together, so the size of a clump and the gap between two clumps are not distances.</p>
    <div class="scroll-x"><table class="feat lens-acc sim-table">
      <thead><tr><th>Space</th><th>Trustworthiness</th><th>Continuity</th><th>Same cuisine</th><th>Same borough</th></tr></thead>
      <tbody>${sm.metrics.map((r) => `<tr><td>${esc(r.space)}</td><td class="mono">${f3(r.trustworthiness)}</td><td class="mono">${f3(r.continuity)}</td>
        <td class="mono">${f3(r.knn_cuisine)}</td><td class="mono">${f3(r.knn_borough)}</td></tr>`).join("")}
        <tr><td>Chance baseline (largest group)</td><td></td><td></td><td class="mono">${f3(sm.chance.knn_cuisine)}</td><td class="mono">${f3(sm.chance.knn_borough)}</td></tr></tbody></table></div>
    <p><b>By cuisine group.</b> For the restaurants of one cuisine group, the average share of their ${nn} nearest restaurants (same ${sm.params.k}-number space) that are in the same group, beside the group’s share of all restaurants. This was computed for cuisine groups only, so there is no matching column for boroughs.</p>
    <div class="scroll-x"><table class="feat lens-acc sim-table">
      <thead><tr><th>Cuisine group</th><th>Restaurants</th><th>Share of all</th><th>Same group among the ${nn} nearest</th></tr></thead>
      <tbody>${sm.purity.map((r) => `<tr><td>${esc(LABEL(r.cuisine))}</td><td class="mono">${r.restaurants.toLocaleString()}</td><td class="mono">${pct1(r.share)}</td>
        <td class="mono">${r.purity_factors == null ? "—" : pct1(r.purity_factors)}</td></tr>`).join("")}</tbody></table></div>
    <p>A second seed and a parameter sweep are in outputs/similarity_map/SIMILARITY_MAP.md.</p>`;

  // (a small group is not rounded down to "0% of all": below 1% the share keeps one decimal)
  const describe = (n) => `${n.toLocaleString()} restaurants · ${n / N < 0.01 ? pct1(n / N) : pct(n / N)} of all`;
  function counts() {
    $("sim-bo-count").textContent = describe(boN[pick.bo]);
    $("sim-cu-count").textContent = describe(cuN[pick.cu]);
  }

  // one picture: every restaurant in warm grey first, then the picked group in brick red on top; round dots of
  // nearly the same size, no edge. Drawn at the screen's own pixel density, so it is sharp on any display.
  function draw(id, labels, value, what) {
    const node = $(id), w = Math.max(node.clientWidth, 240), h = Math.round(Math.min(w * 0.82, 520));
    const [x0, x1] = d3.extent(sm.x), [y0, y1] = d3.extent(sm.y), pad = 10;
    const k = Math.min((w - 2 * pad) / (x1 - x0), (h - 2 * pad) / (y1 - y0));
    const ox = (w - k * (x1 - x0)) / 2, oy = (h - k * (y1 - y0)) / 2;
    const X = (v) => ox + (v - x0) * k, Y = (v) => h - oy - (v - y0) * k;
    const css = getComputedStyle(root);
    const other = css.getPropertyValue("--sim-other").trim(), sel = css.getPropertyValue("--sim-sel").trim();
    const cv = node.querySelector("canvas"), dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.width = `${w}px`; cv.style.height = `${h}px`;
    cv.setAttribute("aria-label", `Layout of ${N.toLocaleString()} restaurants by who reviewed them, with ${what} picked out: ${describe(labels === sm.bo ? boN[value] : cuN[value])}`);
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const dots = (r, on) => { g.beginPath(); for (let i = 0; i < N; i++) if ((labels[i] === value) === on) { const px = X(sm.x[i]), py = Y(sm.y[i]); g.moveTo(px + r, py); g.arc(px, py, r, 0, 6.2832); } g.fill(); };
    const R = w < 420 ? 1.05 : 1.3;                       // a little smaller on a narrow picture, where dots crowd
    g.globalAlpha = 0.6; g.fillStyle = other; dots(R, false);
    g.globalAlpha = 0.92; g.fillStyle = sel; dots(R + 0.2, true);
    g.globalAlpha = 1;
  }
  function drawAll() {
    draw("sim-bo-plot", sm.bo, pick.bo, sm.boroughs[pick.bo]);
    draw("sim-cu-plot", sm.cu, pick.cu, LABEL(sm.cuisines[pick.cu]));
    counts();
  }
  $("sim-bo").addEventListener("change", (e) => { pick.bo = +e.target.value; drawAll(); });
  $("sim-cu").addEventListener("change", (e) => { pick.cu = +e.target.value; drawAll(); });

  let drawnW = null;
  const sync = () => { const w = $("sim-bo-plot").clientWidth; if (w && w !== drawnW) { drawnW = w; drawAll(); } };
  return { update: sync, redraw() { drawnW = null; sync(); }, highlight() {} };
}
