// "Say it" box: one sentence in, the app's own settings out. A language model (server/nl.py, through
// /api/parse) or, when it cannot be reached, the keyword reader in ../nl.js turns the sentence into slots.
// The slots are shown as removable chips and nothing changes until the person applies them. The ranking is
// then done by the solver, as for every other control.
import { MAX_SENTENCE, chips, filterChips, hasFilters, mergeFilters, poolMask, withoutChip } from "../nl.js";
import { WHY, apiStatus, readSentence } from "../nlClient.js";
import { esc } from "./common.js";

const EXAMPLE = "cheap sushi within walking distance";
const read = readSentence;

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

// ctx: full() -> the loaded case with all its candidates; filters() -> active filters;
//      apply(slots) -> take the confirmed slots; setFilters(f) -> replace the active filters
export function mountSayit(root, ctx) {
  root.innerHTML = `
    <div class="ctl-top"><label for="say-input">Say what you want</label></div>
    <form class="say-row" id="say-form">
      <input id="say-input" type="text" maxlength="${MAX_SENTENCE}" autocomplete="off" spellcheck="true" placeholder="${EXAMPLE}">
      <button class="btn-primary" type="submit" id="say-go">Read</button>
    </form>
    <p class="ctl-help" id="say-help">Keyword rules turn your sentence into the settings below (English only). Nothing leaves this page.</p>
    <div class="say-result" id="say-result" aria-live="polite"></div>
    <div class="say-pool" id="say-pool"></div>`;
  const $ = (id) => root.querySelector("#" + id);
  let draft = null;   // { slots, source, why } waiting for confirmation
  apiStatus().then((s) => { if (s?.llm) $("say-help").textContent = "A language model reads your sentence into the settings below; the ranking model still chooses the places. The sentence goes to Anthropic's Claude, so leave out private details."; });

  function renderDraft() {
    const box = $("say-result");
    if (!draft) { box.innerHTML = ""; return; }
    const { slots, source, why } = draft, list = chips(slots);
    const by = source === "llm" ? "Read by the language model." : `Read by keyword rules, because ${WHY[why]}.`;
    const un = slots.unsupported.length ? `<p class="say-note">No setting for: ${slots.unsupported.map(esc).join(", ")}.</p>` : "";
    if (!list.length) {
      box.innerHTML = `<p class="say-note">${!slots.in_scope ? "This does not read as a request about where to eat." : "No setting was found in this sentence."} Nothing was changed. ${by}</p>${un}
        <div class="say-actions"><button class="btn-reset" type="button" data-say="discard">Dismiss</button></div>`;
      return;
    }
    const merged = mergeFilters(ctx.filters(), slots);
    let count = "", blocked = false;
    if (list.some((c) => c.kind === "filter")) {
      const m = poolMask(ctx.full(), merged);
      blocked = m.size === 0;
      const already = hasFilters(ctx.filters()) ? " together with the filters already set" : "";
      count = blocked ? `<p class="say-note warn">None of this user's ${m.of} candidates passes all of these${already}. Remove one to continue.</p>`
        : `<p class="say-note">With these filters, ${plural(m.size, "place")} of this user's ${m.of} candidates ${m.size === 1 ? "is" : "are"} left to rank.</p>`;
      if (slots.area && !m.areas.length) count += `<p class="say-note warn">No candidate is in a neighborhood named “${esc(slots.area)}”.</p>`;
    }
    box.innerHTML = `<p class="say-note">${by} Check what it understood:</p>
      <ul class="say-chips">${list.map((c) => `<li class="say-chip"><span><span class="say-kind">${c.kind === "filter" ? "Filter" : "Target"}</span>${esc(c.text)}</span>
        <button type="button" data-drop="${esc(c.key)}" aria-label="Remove ${esc(c.text)}">Remove</button></li>`).join("")}</ul>
      ${count}${un}
      <div class="say-actions"><button class="btn-primary" type="button" data-say="apply"${blocked ? " disabled" : ""}>Apply</button>
        <button class="btn-reset" type="button" data-say="discard">Discard</button></div>`;
  }

  // Active filters, removable one by one, and how many candidates they leave.
  function renderPool() {
    const f = ctx.filters(), box = $("say-pool");
    if (!hasFilters(f)) { box.innerHTML = ""; return; }
    const m = poolMask(ctx.full(), f);
    box.innerHTML = `<span class="lbl">Filters</span>
      <ul class="say-chips">${filterChips(f).map((c) => `<li class="say-chip"><span>${esc(c.text)}</span>
        <button type="button" data-unfilter="${esc(c.key)}" aria-label="Remove filter ${esc(c.text)}">Remove</button></li>`).join("")}</ul>
      <p class="ctl-help">${m.size > 0
        ? `${plural(m.size, "place")} of this user's ${m.of} candidates ${m.size === 1 ? "passes" : "pass"}. Everything on this page ranks and explains ${m.size === 1 ? "that one" : `those ${m.size}`}. Places with no known price fail a price filter.`
        : `None of this user's ${m.of} candidates passes, so the filters are not applied and all ${m.of} are shown.`}</p>`;
  }

  $("say-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const sentence = $("say-input").value.trim().replace(/\s+/g, " ");
    if (!sentence) { $("say-input").focus(); return; }
    const go = $("say-go");
    go.disabled = true; go.textContent = "Reading";
    $("say-result").innerHTML = `<p class="say-note">Reading your sentence.</p>`;
    draft = await read(sentence);
    go.disabled = false; go.textContent = "Read";
    renderDraft();
  });
  root.addEventListener("click", (e) => {
    const drop = e.target.closest("[data-drop]"), act = e.target.closest("[data-say]"), unf = e.target.closest("[data-unfilter]");
    if (drop && draft) { draft = { ...draft, slots: withoutChip(draft.slots, drop.dataset.drop) }; renderDraft(); }
    if (act?.dataset.say === "discard") { draft = null; renderDraft(); }
    if (act?.dataset.say === "apply" && draft) {
      const slots = draft.slots, n = chips(slots).length;
      draft = null;
      ctx.apply(slots);
      $("say-result").innerHTML = `<p class="say-note">Applied ${plural(n, "setting")}. The controls below show them.</p>`;
    }
    if (unf) ctx.setFilters(withoutChip({ ...ctx.filters(), unsupported: [] }, unf.dataset.unfilter));
  });
  return { render: renderPool, clearDraft: () => { draft = null; renderDraft(); } };
}
