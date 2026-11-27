// "Your top 10": the list first, the explanation one press in.
// A row shows the rank now, the restaurant, how its rank changed in plain words, and where it is. Pressing a row
// selects that restaurant everywhere on the page and opens its detail under it: the rank before and now, and the
// change split across the settings (exact Shapley values, web/src/steer.js `shapleyControls`). One detail is open
// at a time; pressing the open row closes it. The places that are no longer in the top 10 follow, built the same way.
import { STAR, esc, hueLine } from "./common.js";

// The settings a rank change is split across, in the order of the steps elsewhere on the page, with the names this
// card uses for them and the colour of their step.
const FACTORS = [["w_h", "Past preferences", "model"], ["alpha", "Distance preference", "control"], ["beta", "Popularity preference", "control"],
  ["boost", "Neighborhood", "control"], ["lam", "Cuisine variety", "diversity"]];
const CHEVRON = '<svg class="chev" viewBox="0 0 12 12" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 4.5 6 8l3.5-3.5"/></svg>';
const places = (n) => `${n} ${n === 1 ? "place" : "places"}`;
const signed = (v) => String(v).replace("-", "−");   // a real minus sign

// An official area name can list several neighbourhoods ("SoHo-Little Italy-Hudson Square"). In a row a long one is
// written as its first neighbourhood alone (Yila, 2026-11-26); the full name is the row's tooltip. Nothing is
// abbreviated or renamed. (Three names carry a hyphen of their own and are kept whole.)
const COMPOUND = ["Bedford-Stuyvesant", "Co-op City", "Green-Wood"];
export function shortArea(name) {
  if (name.length <= 24) return name;
  const parts = COMPOUND.reduce((s, w, k) => s.replace(w, `\u0001${k}`), name).split("-").map((p) => p.replace(/\u0001(\d)/, (_, k) => COMPOUND[+k]));
  return parts[0];
}

// How the rank changed, in words (the class only adds a colour).
function status(before, after) {
  if (after > 10) return ["out", `Now #${after}`];
  if (before > 10) return ["in", "Entered top 10"];
  const d = before - after;
  return d > 0 ? ["up", `↑ ${places(d)}`] : d < 0 ? ["down", `↓ ${places(-d)}`] : ["same", "No change"];
}

// One contribution, to one decimal everywhere. A value that rounds to 0.0 but is not zero is said to be small, not nil.
function amount(v) {
  const a = Math.abs(v);
  if (a === 0) return "0 places";
  const sign = v > 0 ? "+" : "−";
  if (a < 0.05) return `${sign}<0.1 place`;
  return `${sign}${a.toFixed(1)} ${a.toFixed(1) === "1.0" ? "place" : "places"}`;
}

function detail(c, i, before, after, phi) {
  // phi[k][i] is the setting's share of (rank now − rank before): negative when it helped the place up. Shown the
  // other way round, as places gained, so that "up" is positive and drawn to the right.
  const rows = FACTORS.filter(([k]) => phi[k]).map(([k, label, hue]) => ({ label, hue, v: -phi[k][i] }));
  const d = before[i] - after[i], max = Math.max(...rows.map((r) => Math.abs(r.v)));
  const shown = rows.reduce((a, r) => a + Math.sign(r.v) * +Math.abs(r.v).toFixed(1), 0);
  const overall = d > 0 ? `↑ ${places(d)} overall` : d < 0 ? `↓ ${places(-d)} overall` : "No change overall";
  const name = esc(c.candidates.name[i]);
  const table = max === 0
    ? `<p class="rd-none">No setting moved this restaurant: it holds the rank it had in the original list.</p>`
    : `<table class="rd-table">
        <thead><tr><th scope="col">Preference</th><th scope="col" colspan="2">Contribution to rank change</th></tr></thead>
        <tbody>${rows.map((r) => `<tr><th scope="row">${r.label}</th>
          <td class="rd-bar" aria-hidden="true">${r.v ? `<i class="${r.v > 0 ? "pos" : "neg"}" style="width:${Math.max(1.5, (50 * Math.abs(r.v)) / max)}%;--bar:${hueLine(r.hue)}"></i>` : ""}</td>
          <td class="rd-val">${amount(r.v)}</td></tr>`).join("")}</tbody>
        <tfoot aria-hidden="true"><tr><td></td><td class="rd-axis"><span>↓ down</span><span>up ↑</span></td><td></td></tr></tfoot>
      </table>
      <p class="rd-note">A positive value means the setting contributed that many places upward, toward #1; a negative value, downward.${
        Math.abs(shown - d) > 0.05 ? ` Values are rounded to one decimal, so these rows add up to ${signed(shown.toFixed(1))}, not exactly ${signed(d)}.` : ""}</p>`;
  return `<div class="row-detail" id="detail-${i}" role="region" aria-label="What contributed to the change for ${name}">
      <h4>What contributed to this change?</h4>
      <p class="rd-ranks"><span>Previously <b>#${before[i]}</b> → Now <b>#${after[i]}</b></span><span class="rd-overall">${overall}</span></p>
      ${table}
    </div>`;
}

export function renderList(listEl, exitedEl, data, handlers) {
  const { c, top, before, after, phi, liked, gone, open } = data;
  // keep keyboard focus on the same restaurant across the re-render that selecting it triggers
  const focused = [listEl, exitedEl].some((r) => r.contains(document.activeElement)) ? document.activeElement.closest("li")?.dataset.id : null;
  const row = (i, inner) => {
    const isOpen = i === open, [cls, text] = status(before[i], after[i]);
    return `<li data-id="${i}"${isOpen ? ' class="open"' : ""}>
      <button type="button" class="row" aria-expanded="${isOpen}" aria-controls="detail-${i}">${inner(cls, text)}${CHEVRON}</button>
      ${isOpen ? detail(c, i, before, after, phi) : ""}</li>`;
  };
  listEl.innerHTML = top.map((i, k) => row(i, (cls, text) => `
      <span class="pos">${k + 1}</span>
      <span class="nm"><strong>${esc(c.candidates.name[i])}</strong>${liked[i] ? `<span class="liked" title="The user reviewed this place 4–5 stars during the test period">${STAR}later liked</span>` : ""}
        <span class="meta" title="${esc(c.candidates.nta_name[i])}">${esc(c.candidates.cuisine_label[i])} · ${esc(shortArea(c.candidates.nta_name[i]))}</span></span>
      <span class="right"><span class="move ${cls}">${text}</span><span class="km">${c.candidates.distance_km[i].toFixed(1)} km</span></span>`)).join("");

  // the places that were in the original top 10 and are not in it now, each with its rank now
  const box = exitedEl.closest(".exited-box"), head = box?.querySelector(".subhead"), none = box?.querySelector(".exited-none");
  if (head) head.innerHTML = `No longer in your top 10 <span aria-hidden="true">·</span> <b>${gone.length}</b>`;
  if (box) { box.classList.toggle("empty", !gone.length); exitedEl.hidden = !gone.length; if (none) none.hidden = !!gone.length; }
  exitedEl.innerHTML = gone.map((i) => row(i, (cls, text) => `
      <span class="nm"><strong>${esc(c.candidates.name[i])}</strong>${liked[i] ? `<span class="liked">${STAR}later liked</span>` : ""}</span>
      <span class="right"><span class="move ${cls}">${text}</span></span>`)).join("");

  for (const li of [...listEl.children, ...exitedEl.children]) {
    const id = +li.dataset.id, btn = li.querySelector(".row");
    li.addEventListener("pointerenter", () => handlers.hover(id));
    li.addEventListener("pointerleave", () => handlers.hover(null));
    btn.addEventListener("click", () => handlers.select(id));     // a button: Enter and Space press it too
    btn.addEventListener("focus", () => handlers.hover(id));
    btn.addEventListener("blur", () => handlers.hover(null));
  }
  if (focused != null) [...listEl.children, ...exitedEl.children].find((li) => li.dataset.id === focused)?.querySelector(".row").focus({ preventScroll: true });
}

export function highlightList(roots, id) {
  for (const r of roots) for (const el of r.children) el.classList.toggle("hover", id != null && +el.dataset.id === id);
}

export function markSelected(roots, id) {
  for (const r of roots) for (const el of r.children) el.classList.toggle("selected", id != null && +el.dataset.id === id);
}
