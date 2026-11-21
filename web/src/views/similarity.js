// Similarity map (S1): restaurants placed by who reviewed them (SVD item factors before the snapshot, then UMAP,
// with PCA beside it as the reference; src/26_similarity_map.py). Grey: all 19,812 placed restaurants; this
// user's candidates on top, the current top 10 numbered. Brushing the UMAP panel shows the brushed candidates
// on the map (V4); hover links to every view through the candidate index. Positions are for asking questions,
// not evidence: the scores table says how faithful each projection is.
import * as d3 from "d3";
import { esc } from "./common.js";

const LABEL = (s) => s.replace(/_/g, " ").replace(/^\w/, (ch) => ch.toUpperCase());
const f3 = (v) => (v == null ? "—" : v.toFixed(3));

export function createSimilarity(root, sm, handlers) {
  const m = Object.fromEntries(sm.metrics.map((r) => [r.space, r]));
  const fac = m["Item factors (64-d)"], um = m.UMAP, pc = m.PCA;
  root.innerHTML = `
    <div class="eval-top">
      <p class="lens-headline">Restaurants sit close together when the same people reviewed them. Their 15 nearest neighbours share
        the borough <b>${Math.round(fac.knn_borough * 100)}%</b> of the time (chance ${Math.round(sm.chance.knn_borough * 100)}%) but the
        cuisine group only <b>${Math.round(fac.knn_cuisine * 100)}%</b> (chance ${Math.round(sm.chance.knn_cuisine * 100)}%): who eats where
        follows place more than cuisine.</p>
      <label class="field" for="sim-hl"><span>Highlight</span><select id="sim-hl"><option value="">Nothing</option>
        <optgroup label="Borough">${sm.boroughs.map((b, k) => `<option value="bo:${k}">${esc(b)}</option>`).join("")}</optgroup>
        <optgroup label="Cuisine group">${sm.cuisines.map((c, k) => [c, k]).sort((a, b) => a[0].localeCompare(b[0]))
          .map(([c, k]) => `<option value="cu:${k}">${esc(LABEL(c))}</option>`).join("")}</optgroup></select></label>
    </div>
    <p class="sim-note" id="sim-note"></p>
    <div class="sim-row">
      <figure class="lens-fig">
        <figcaption><b>UMAP</b><span>Drag a box to show those candidates on the map; click to clear. Cluster sizes and the distances between clusters carry no meaning.</span></figcaption>
        <div class="sim-plot" id="sim-umap"><canvas></canvas><svg></svg></div>
      </figure>
      <figure class="lens-fig">
        <figcaption><b>PCA</b><span>Same restaurants, linear projection: the reference UMAP has to beat.</span></figcaption>
        <div class="sim-plot" id="sim-pca"><canvas></canvas><svg></svg></div>
      </figure>
    </div>
    <ul class="key" id="sim-key"></ul>
    <p class="sim-cap" id="sim-cap"></p>
    <div class="scroll-x"><table class="feat lens-acc" id="sim-table"></table></div>
    <div class="lens-tip" id="sim-tip" role="tooltip" hidden></div>`;
  const $ = (id) => root.querySelector("#" + id);
  const tip = $("sim-tip");
  const N = sm.x.length;
  let ctx = null, hl = "", hovered = null, brushed = null;
  const panels = {};

  $("sim-key").innerHTML = `<li><svg width="12" height="12"><circle cx="6" cy="6" r="2" fill="var(--line-strong)"/></svg>Placed restaurant</li>
    <li><svg width="12" height="12"><circle cx="6" cy="6" r="3.5" fill="var(--panel)" stroke="var(--ink-3)" stroke-width="1.3"/></svg>This user's candidate</li>
    <li><svg width="16" height="16"><circle cx="8" cy="8" r="7" fill="var(--ink)"/></svg>In your top 10 (number = rank)</li>
    <li><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="var(--panel)" stroke="var(--ink)" stroke-width="1.6"/></svg>Left your top 10</li>`;
  $("sim-cap").innerHTML = `<b>How faithful is each picture?</b> Trustworthiness: are 2-D neighbours real neighbours; continuity: are real
      neighbours kept together (k = ${sm.params.n_neighbors}, ${sm.params.subsample.toLocaleString()} restaurants). k-NN: share of restaurants whose 15 nearest
      neighbours mostly share their label. Noise row: the same UMAP on random data of the same size.`;
  $("sim-table").innerHTML = `<thead><tr><th>Space</th><th>Trustworthiness</th><th>Continuity</th><th>k-NN cuisine</th><th>k-NN borough</th></tr></thead>
    <tbody>${sm.metrics.map((r) => `<tr><td>${esc(r.space)}</td><td class="mono">${f3(r.trustworthiness)}</td><td class="mono">${f3(r.continuity)}</td>
      <td class="mono">${f3(r.knn_cuisine)}</td><td class="mono">${f3(r.knn_borough)}</td></tr>`).join("")}
      <tr><td>Chance (most common label)</td><td></td><td></td><td class="mono">${f3(sm.chance.knn_cuisine)}</td><td class="mono">${f3(sm.chance.knn_borough)}</td></tr></tbody>
    <tfoot><tr><td colspan="5" class="note">${sm.info.reviews.toLocaleString()} reviews before ${sm.cutoff} by ${sm.info.users.toLocaleString()} people of
      ${sm.info.restaurants.toLocaleString()} restaurants with at least ${sm.params.min_reviewers} reviewers; ${sm.params.k}-dimensional SVD; UMAP n_neighbors
      ${sm.params.n_neighbors}, min_dist ${sm.params.min_dist}, cosine, seed ${sm.params.seed}. A second seed and a parameter sweep are in
      outputs/similarity_map/SIMILARITY_MAP.md.</td></tr></tfoot>`;
  $("sim-hl").addEventListener("change", (e) => { hl = e.target.value; drawBackground(); note(); });

  function note() {
    const c = ctx?.c;
    let s = "";
    if (c) {
      const placed = c.candidates.gmap_id.filter((g) => g in sm.ids).length;
      s = `${placed} of this user's ${c.candidates.gmap_id.length} candidates are placed; the rest had fewer than ${sm.params.min_reviewers} reviewers before ${sm.cutoff}.`;
    }
    if (hl.startsWith("cu:")) {
      const p = sm.purity.find((r) => r.cuisine === sm.cuisines[+hl.slice(3)]);
      if (p) s += ` ${LABEL(p.cuisine)}: ${p.restaurants.toLocaleString()} restaurants; ${Math.round(p.purity_factors * 100)}% of their nearest neighbours are also ${LABEL(p.cuisine).toLowerCase()}, against ${(p.share * 100).toFixed(1)}% of all restaurants.`;
    }
    if (brushed) s += ` ${brushed.length} candidate${brushed.length === 1 ? "" : "s"} brushed, shown on the map.`;
    $("sim-note").textContent = s;
  }

  function size(id) {
    const node = $(id), w = Math.max(node.clientWidth, 240);
    return { node, w, h: Math.round(Math.min(w * 0.8, 440)) };
  }

  function scales(key, w, h) {
    const xs = key === "umap" ? sm.x : sm.px, ys = key === "umap" ? sm.y : sm.py;
    const [x0, x1] = d3.extent(xs), [y0, y1] = d3.extent(ys), pad = 10;
    const k = Math.min((w - 2 * pad) / (x1 - x0), (h - 2 * pad) / (y1 - y0));
    const ox = (w - k * (x1 - x0)) / 2, oy = (h - k * (y1 - y0)) / 2;
    return { xs, ys, x: (v) => ox + (v - x0) * k, y: (v) => h - oy - (v - y0) * k };
  }

  function drawBackground() {
    const css = getComputedStyle(document.documentElement);
    const grey = css.getPropertyValue("--line-strong").trim(), ink = css.getPropertyValue("--ink").trim();
    const [kind, val] = hl ? [hl.slice(0, 2), +hl.slice(3)] : [null, null];
    const lab = kind === "cu" ? sm.cu : kind === "bo" ? sm.bo : null;
    for (const [key, id] of [["umap", "sim-umap"], ["pca", "sim-pca"]]) {
      const { node, w, h } = size(id);
      const S = scales(key, w, h);
      const cv = node.querySelector("canvas"), dpr = window.devicePixelRatio || 1;
      cv.width = w * dpr; cv.height = h * dpr; cv.style.width = `${w}px`; cv.style.height = `${h}px`;
      const g = cv.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      g.globalAlpha = lab ? 0.35 : 0.55;
      g.fillStyle = grey;
      for (let i = 0; i < N; i++) if (!lab || lab[i] !== val) g.fillRect(S.x(S.xs[i]) - 1, S.y(S.ys[i]) - 1, 2, 2);
      if (lab) {
        g.globalAlpha = 0.85; g.fillStyle = ink;
        for (let i = 0; i < N; i++) if (lab[i] === val) g.fillRect(S.x(S.xs[i]) - 1.2, S.y(S.ys[i]) - 1.2, 2.4, 2.4);
      }
      panels[key] = Object.assign(panels[key] ?? {}, S, { w, h, node });   // keep the overlay's positions and ring layer
    }
  }

  function drawOverlay() {
    if (!ctx) return;
    const { c, top, gone } = ctx;
    const cand = c.candidates.gmap_id.map((g, i) => [i, sm.ids[g]]).filter(([, k]) => k != null);
    for (const [key, P] of Object.entries(panels)) {
      const svg = d3.select(P.node.querySelector("svg")).attr("width", P.w).attr("height", P.h)
        .attr("role", "img").attr("aria-label", `${key === "umap" ? "UMAP" : "PCA"} of restaurants with this user's candidates`);
      svg.selectAll("*").remove();
      const pos = new Map(cand.map(([i, k]) => [i, [P.x(P.xs[k]), P.y(P.ys[k])]]));
      svg.append("g").selectAll("circle").data(cand.filter(([i]) => !top.includes(i) && !gone.includes(i))).join("circle")
        .attr("class", "sim-cand").attr("r", 3).attr("cx", ([i]) => pos.get(i)[0]).attr("cy", ([i]) => pos.get(i)[1]);
      svg.append("g").selectAll("circle").data(gone.filter((i) => pos.has(i))).join("circle").attr("class", "sim-gone").attr("r", 4.5)
        .attr("cx", (i) => pos.get(i)[0]).attr("cy", (i) => pos.get(i)[1]);
      const pins = svg.append("g").selectAll("g").data(top.filter((i) => pos.has(i))).join("g").attr("class", "sim-pin")
        .attr("transform", (i) => `translate(${pos.get(i)})`);
      pins.append("circle").attr("r", 8);
      pins.append("text").attr("dy", "0.35em").text((i) => top.indexOf(i) + 1);
      P.rings = svg.append("g");
      P.pos = pos;
      if (key === "umap") {
        const brush = d3.brush().extent([[0, 0], [P.w, P.h]]).on("end", ({ selection }) => {
          brushed = selection ? [...pos].filter(([, [x, y]]) => x >= selection[0][0] && x <= selection[1][0] && y >= selection[0][1] && y <= selection[1][1]).map(([i]) => i) : null;
          handlers.brush?.(brushed);
          note(); rings();
        });
        const bg = svg.append("g").attr("class", "sim-brush").call(brush);
        const ids = [...pos.keys()], del = d3.Delaunay.from(ids, (i) => pos.get(i)[0], (i) => pos.get(i)[1]);
        bg.on("pointermove.tip", (e) => {
          const [px, py] = d3.pointer(e, svg.node());
          const i = ids[del.find(px, py)], [x, y] = pos.get(i);
          if (Math.hypot(x - px, y - py) > 12) { handlers.hover(null); tip.hidden = true; return; }
          handlers.hover(i);
          const box = root.getBoundingClientRect();
          tip.innerHTML = `<b>${esc(c.candidates.name[i])}</b><br>${esc(c.candidates.cuisine_label[i])} · ${esc(c.candidates.nta_name[i])}, ${esc(c.candidates.borough[i])}`
            + `<br>${top.includes(i) ? `#${top.indexOf(i) + 1} in your list` : gone.includes(i) ? "left your top 10" : "candidate"}`;
          tip.hidden = false;
          tip.style.left = `${Math.min(e.clientX - box.left + 14, box.width - tip.offsetWidth - 4)}px`;
          tip.style.top = `${e.clientY - box.top + 16}px`;
        }).on("pointerleave.tip", () => { handlers.hover(null); tip.hidden = true; });
        if (brushed) brushed = brushed.filter((i) => pos.has(i));
      }
    }
    rings();
  }

  function rings() {
    for (const P of Object.values(panels)) {
      if (!P.rings) continue;
      const marks = [...(brushed ?? []).map((i) => [i, "brushed"]), ...(hovered != null ? [[hovered, "hov"]] : [])].filter(([i]) => P.pos.has(i));
      P.rings.selectAll("circle").data(marks).join("circle").attr("class", ([, cls]) => `ring ${cls}`)
        .attr("r", ([, cls]) => (cls === "hov" ? 11 : 6)).attr("cx", ([i]) => P.pos.get(i)[0]).attr("cy", ([i]) => P.pos.get(i)[1]);
    }
  }

  let drawnW = null, caseId = null;
  // the canvas takes its colours from the theme tokens when drawn, so redraw when the theme changes
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", () => { drawBackground(); });
  return {
    update(next) {
      if (next.c.id !== caseId) { brushed = null; handlers.brush?.(null); caseId = next.c.id; }
      ctx = next;
      const w = $("sim-umap").clientWidth + "|" + $("sim-pca").clientWidth;
      if (w !== drawnW) { drawnW = w; drawBackground(); }
      drawOverlay(); note();
    },
    redraw() { drawnW = null; if (ctx) this.update(ctx); },
    highlight(i) { hovered = i; rings(); },
  };
}
