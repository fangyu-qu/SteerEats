// Top-10 list with change attribution: each place's rank change split across the controls (exact Shapley values).
import { CONTROL_INFO, STAR, esc, hueVar } from "./common.js";

const CONTROLS = ["w_h", "alpha", "beta", "lam"];

function why(phi, i) {
  // phi[c][i] is the control's share of (new rank - default rank); negative = helped it move up.
  const parts = CONTROLS.concat(phi.boost ? ["boost"] : []).map((k) => ({ k, v: -phi[k][i] })).filter((p) => Math.abs(p.v) >= 0.5)
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const total = parts.reduce((a, p) => a + Math.abs(p.v), 0);
  const bar = total
    ? parts.map((p) => `<span style="width:${(100 * Math.abs(p.v)) / total}%;background:${hueVar(CONTROL_INFO[p.k].hue)}" title="${CONTROL_INFO[p.k].name} ${p.v > 0 ? "+" : ""}${p.v.toFixed(1)}"></span>`).join("")
    : "";
  const text = parts.slice(0, 3).map((p) => `${CONTROL_INFO[p.k].name} ${p.v > 0 ? "+" : "−"}${Math.round(Math.abs(p.v))}`).join(", ");
  return { bar, text };
}

export function renderList(listEl, exitedEl, data, handlers) {
  const { c, top, before, after, phi, liked, gone } = data;
  // keep keyboard focus on the same restaurant across the re-render that selecting it triggers
  const focused = [listEl, exitedEl].some((r) => r.contains(document.activeElement)) ? document.activeElement.dataset.id : null;
  listEl.innerHTML = top.map((i, k) => {
    const d = before[i] - after[i];
    const move = before[i] > 10
      ? `<span class="move up" title="was #${before[i]}">new · was ${before[i]}</span>`
      : d > 0 ? `<span class="move up">▲ ${d}</span>` : d < 0 ? `<span class="move down">▼ ${-d}</span>` : `<span class="move same">same</span>`;
    const w = why(phi, i);
    return `<li data-id="${i}" tabindex="0" role="button" aria-label="Explain ${esc(c.candidates.name[i])}, number ${k + 1}">
      <span class="pos">${k + 1}</span>
      <span class="nm"><strong><span>${esc(c.candidates.name[i])}</span>${liked[i] ? `<span class="liked" title="The user reviewed this place 4–5 stars during the test period">${STAR}later liked</span>` : ""}</strong>
        <span class="meta">${esc(c.candidates.cuisine_label[i])} · ${esc(c.candidates.nta_name[i])}</span></span>
      <span class="right">${move}<span class="km">${c.candidates.distance_km[i].toFixed(1)} km</span></span>
      ${w.text ? `<span class="why"><span class="why-bar">${w.bar}</span>${w.text}</span>` : ""}
    </li>`;
  }).join("");
  exitedEl.innerHTML = gone.length
    ? gone.map((i) => {
        const w = why(phi, i);
        return `<li data-id="${i}" tabindex="0" role="button" aria-label="Explain ${esc(c.candidates.name[i])}, now number ${after[i]}"><span>${esc(c.candidates.name[i])}${liked[i] ? " · later liked" : ""}</span><span class="mono">now #${after[i]}${w.text ? ` · ${w.text}` : ""}</span></li>`;
      }).join("")
    : `<li class="empty">None. Your top 10 holds the same places as the model's.</li>`;
  for (const el of [...listEl.children, ...exitedEl.children]) {
    if (el.dataset.id == null) continue;
    el.addEventListener("pointerenter", () => handlers.hover(+el.dataset.id));
    el.addEventListener("pointerleave", () => handlers.hover(null));
    el.addEventListener("click", () => handlers.select(+el.dataset.id));
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handlers.select(+el.dataset.id); }
    });
    el.addEventListener("focus", () => handlers.hover(+el.dataset.id));
    el.addEventListener("blur", () => handlers.hover(null));
  }
  if (focused != null) [...listEl.children, ...exitedEl.children].find((el) => el.dataset.id === focused)?.focus();
}

export function highlightList(roots, id) {
  for (const r of roots) for (const el of r.children) el.classList.toggle("hover", id != null && +el.dataset.id === id);
}

export function markSelected(roots, id) {
  for (const r of roots) for (const el of r.children) el.classList.toggle("selected", id != null && +el.dataset.id === id);
}
