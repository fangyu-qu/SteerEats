// Steering console (V1): outcome targets, not raw penalty weights. The solver finds each model's controls.
// Each group reads the same way: the setting's name and its current value, the control, what the list does now,
// one line of help. Names are the largest text after the title, things to press or drag come next, help is the
// smallest; a chosen option is filled, the others outlined.
import { CHECK, CROSS, MODEL_NAMES, esc } from "./common.js";
import { LAM_MAX } from "../steer.js";
import { ledgerData } from "./ledger.js";

// The targets the solver accepts: [value, the button, what it means in full].
const POP_OPTIONS = [[null, "Any", "No popularity target"], [0.5, "50th", "Average at or below the 50th percentile"],
  [0.4, "40th", "Average at or below the 40th percentile"], [0.3, "30th", "Average at or below the 30th percentile"],
  [0.2, "20th", "Average at or below the 20th percentile"]];
const CUI_OPTIONS = [[null, "Any", "No cuisine target"], [5, "5+", "At least 5 cuisine groups"], [6, "6+", "At least 6 cuisine groups"],
  [7, "7+", "At least 7 cuisine groups"], [8, "8+", "At least 8 cuisine groups"]];
const KM = { min: 0.5, max: 6, step: 0.25 };   // the average-distance target, in kilometres
const WH = { min: 0, max: 1, step: 0.05 };     // weight on the history part of the score; 1 is the model as trained

const PIN = '<svg class="ico-pin" viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M8 14.5S3 9.6 3 6.3a5 5 0 0 1 10 0c0 3.3-5 8.2-5 8.2Z"/><circle cx="8" cy="6.3" r="1.7"/></svg>';
const ordinal = (n) => `${n}${[11, 12, 13].includes(n % 100) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" })[n % 10] ?? "th"}`;
// help is one press away, behind a small "?" in the bottom right corner of each group (not beside the name, where it
// read as a question mark after the title), so the panel shows names, controls and values only
const helpBtn = (key, about) => `<button type="button" class="help-btn" aria-expanded="false" aria-controls="help-${key}" aria-label="About ${about}">?</button>`;
const seg = (name, labelId, options) => `<div class="seg" role="radiogroup" aria-labelledby="${labelId}">${options.map(([v, short, long]) =>
  `<label title="${long}"><input type="radio" name="${name}" value="${v ?? ""}" aria-label="${long}"><span>${short}</span></label>`).join("")}</div>`;

export function mountConsole(root, state, onChangeRaw, actions = {}) {
  // Any console control leaves a setting loaded from a link and returns to outcome targets.
  const onChange = () => { state.thetaOverride = null; onChangeRaw(); };
  root.innerHTML = `
    <h2>Steer your recommendations</h2>
    <p class="ctl-help spark-intro">The small chart under each control shows how much of the model's predicted visits for this user the top 10 keeps across that control's range (top line = the model's own list). Grey: the target cannot be met. Hover to read a value, click to set it. <span id="spark-note"></span></p>
    <div id="loaded-note"></div>
    <section class="ctl" id="km-ctl" aria-labelledby="lbl-km">
      <div class="ctl-top"><h3 id="lbl-km">Average distance target</h3><span class="ctl-value" id="val-km"></span></div>
      <label class="switch" for="ctl-km-on"><input id="ctl-km-on" type="checkbox" role="switch"><span class="sw-track" aria-hidden="true"></span>Limit the average distance</label>
      <input id="ctl-km" type="range" min="${KM.min}" max="${KM.max}" step="${KM.step}" aria-label="Average distance target in kilometres">
      <div class="ends" aria-hidden="true"><span>${KM.min} km, closer</span><span>farther, ${KM.max} km</span></div>
      <div class="spark" id="spark-km"><div class="sp-svg"></div><p class="spark-read" role="status" hidden></p></div>
      <div class="ctl-foot"><p class="ctl-status" id="st-km"></p>${helpBtn("km", "the average distance target")}</div>
      <p class="ctl-help" id="help-km" hidden>The average over the 10 places, from your usual area. A single place can be farther.</p>
    </section>
    <section class="ctl" aria-labelledby="lbl-pop">
      <div class="ctl-top"><h3 id="lbl-pop">Restaurant popularity</h3><span class="ctl-value" id="val-pop"></span></div>
      ${seg("ctl-pop", "lbl-pop", POP_OPTIONS)}
      <div class="spark" id="spark-popularity"><div class="sp-svg"></div><p class="spark-read" role="status" hidden></p></div>
      <div class="ctl-foot"><p class="ctl-status" id="st-pop"></p>${helpBtn("pop", "restaurant popularity")}</div>
      <p class="ctl-help" id="help-pop" hidden>Caps the average popularity percentile of the 10 places. Lower is less mainstream.</p>
    </section>
    <section class="ctl" aria-labelledby="lbl-cui">
      <div class="ctl-top"><h3 id="lbl-cui">Cuisine variety</h3><span class="ctl-value" id="val-cui"></span></div>
      ${seg("ctl-cui", "lbl-cui", CUI_OPTIONS)}
      <div class="spark" id="spark-cuisines"><div class="sp-svg"></div><p class="spark-read" role="status" hidden></p></div>
      <div class="ctl-foot"><p class="ctl-status" id="st-cui"></p>${helpBtn("cui", "cuisine variety")}</div>
      <p class="ctl-help" id="help-cui" hidden>The fewest cuisine groups the 10 places should cover.</p>
    </section>
    <section class="ctl" id="boost-ctl" aria-labelledby="lbl-nb">
      <div class="ctl-top"><h3 id="lbl-nb">Include a neighborhood</h3></div>
      <div id="boost-status"></div>
      <div class="ctl-foot solo">${helpBtn("nb", "including a neighborhood")}</div>
      <p class="ctl-help" id="help-nb" hidden>Brings at least one of its places into your top 10; the list is not limited to it. Your other settings stay, but the averages above can move.</p>
    </section>
    <section class="ctl" aria-labelledby="lbl-wh">
      <div class="ctl-top"><h3 id="lbl-wh">Past preferences</h3><span class="ctl-value" id="val-wh"></span></div>
      <input id="ctl-wh" type="range" min="${WH.min}" max="${WH.max}" step="${WH.step}" aria-label="Weight on past preferences">
      <div class="ends" aria-hidden="true"><span>0%, history ignored</span><span>100%, as the model learned</span></div>
      <div class="spark" id="spark-w_h"><div class="sp-svg"></div><p class="spark-read" role="status" hidden></p></div>
      <div class="ctl-foot">${helpBtn("wh", "past preferences")}</div>
      <p class="ctl-help" id="help-wh" hidden>Scales the part of each score that comes from places like ones you reviewed. 100% is the model unchanged, not a score made only of history.</p>
    </section>
    <button class="btn" type="button" id="ctl-reset">Reset to the model's list</button>
    <details class="solution" id="solution"><summary>Model details</summary><div id="solution-body"></div></details>`;

  const $ = (id) => root.querySelector("#" + id);
  const radios = (name) => [...root.querySelectorAll(`input[name="${name}"]`)];
  // the filled part of a slider's track ends under the middle of its thumb (styles.css reads --p)
  const fill = (input) => input.style.setProperty("--p", ((+input.value - +input.min) / (+input.max - +input.min)).toFixed(4));
  const value = (id, text, off) => { const v = $(id); v.textContent = text; v.classList.toggle("off", !!off); };
  const sync = () => {
    const pct = Math.round(state.w_h * 100);
    $("ctl-wh").value = state.w_h;
    $("ctl-wh").setAttribute("aria-valuetext", `${pct}% of the model's weight on past preferences`);
    fill($("ctl-wh"));
    value("val-wh", `${pct}%`);
    const on = state.targets.km != null;
    $("ctl-km-on").checked = on;
    $("ctl-km").disabled = !on;
    $("ctl-km").value = state.kmDraft;
    $("ctl-km").setAttribute("aria-valuetext", `${state.kmDraft} kilometres`);
    fill($("ctl-km"));
    $("km-ctl").classList.toggle("is-off", !on);
    value("val-km", on ? `${state.kmDraft} km` : "Off", !on);
    const pop = state.targets.popularity, cui = state.targets.cuisines;
    for (const r of radios("ctl-pop")) r.checked = r.value === String(pop ?? "");
    for (const r of radios("ctl-cui")) r.checked = r.value === String(cui ?? "");
    value("val-pop", pop == null ? "No target" : `≤ ${ordinal(Math.round(pop * 100))} pct`, pop == null);
    value("val-cui", cui == null ? "No target" : `${cui} or more`, cui == null);
  };
  $("ctl-wh").addEventListener("input", (e) => { state.w_h = +e.target.value; sync(); onChange(); });
  $("ctl-km-on").addEventListener("change", (e) => { state.targets.km = e.target.checked ? state.kmDraft : null; sync(); onChange(); });
  $("ctl-km").addEventListener("input", (e) => { state.kmDraft = +e.target.value; state.targets.km = state.kmDraft; sync(); onChange(); });
  const pickTarget = (key) => (e) => { state.targets[key] = e.target.value === "" ? null : +e.target.value; sync(); onChange(); };
  for (const r of radios("ctl-pop")) r.addEventListener("change", pickTarget("popularity"));
  for (const r of radios("ctl-cui")) r.addEventListener("change", pickTarget("cuisines"));
  $("ctl-reset").addEventListener("click", () => {
    state.w_h = 1; state.targets = { km: null, popularity: null, cuisines: null }; state.boostNta = null; sync(); onChange();
  });
  // the small previews under each control are drawn in study mode only (the checkbox that turned them on was removed)
  root.classList.toggle("show-previews", !!state.showPreviews);
  root.addEventListener("click", (e) => {
    const help = e.target.closest(".help-btn");
    if (help) { const open = help.getAttribute("aria-expanded") !== "true"; help.setAttribute("aria-expanded", open); root.querySelector("#" + help.getAttribute("aria-controls")).hidden = !open; }
    if (e.target.closest("#boost-pick")) actions.pickOnMap?.();
    if (e.target.closest("#boost-clear")) actions.clearBoost?.();
    if (e.target.closest("#loaded-clear")) actions.clearLoaded?.();
  });
  sync();
  return { sync };
}

// What the list does now, next to each control: every number here is read from the list just computed.
export function renderSolution(root, state, result) {
  const { solved, boost, top } = result;
  const $ = (id) => root.querySelector("#" + id);
  const t = state.targets, o = solved.outcomes, loaded = !!solved.loaded;
  const line = (cls, icon, html) => `<span class="ico ${cls}">${icon}</span><span>${html}</span>`;
  const status = (set, met, now) => {
    if (!set) return line("idle", "", now);
    if (loaded) return line("idle", "", `${now} · <b>Target paused</b>`);
    return met ? line("yes", CHECK, `${now} · <b>Target met</b>`) : line("no", CROSS, `${now} · <b>Not reached</b>`);
  };
  $("st-km").innerHTML = status(t.km != null, solved.met.km, `Current average: ${o.km.toFixed(1)} km`);
  $("st-pop").innerHTML = status(t.popularity != null, solved.met.popularity, `Current average: ${ordinal(Math.round(o.popularity * 100))} pct`);
  $("st-cui").innerHTML = status(t.cuisines != null, solved.met.cuisines, `Now: ${o.cuisines} cuisine ${o.cuisines === 1 ? "group" : "groups"}`);

  // the neighbourhood: a way in, what is chosen, and whether the list holds one of its places
  const picking = state.picking ? `<p class="ctl-status">${line("idle", "", "<b>Click a shaded neighborhood on the map.</b> Esc cancels.")}</p>` : "";
  if (!boost) {
    $("boost-status").innerHTML = `<button class="btn" type="button" id="boost-pick" aria-pressed="${!!state.picking}">${state.picking ? "Cancel" : `${PIN}Choose on map`}</button>
      ${picking || `<p class="ctl-status">${line("idle", "", "No neighborhood chosen")}</p>`}`;
  } else {
    const name = esc(state.c.candidates.nta_name[state.c.nta.indexOf(boost.nta)]);
    const n = top.filter((i) => state.c.nta[i] === boost.nta).length;
    $("boost-status").innerHTML = `<p class="nb-chosen">${PIN}<span>${name}</span></p>
      <div class="btn-row"><button class="btn" type="button" id="boost-pick" aria-pressed="${!!state.picking}">${state.picking ? "Cancel" : "Change on map"}</button>
        <button class="btn" type="button" id="boost-clear" aria-label="Remove ${name} from your settings">Remove</button></div>
      ${picking}
      <p class="ctl-status">${n ? line("yes", CHECK, `<b>${n} of your top 10</b> ${n === 1 ? "is" : "are"} in ${name}`)
        : line("no", CROSS, `<b>Not reached:</b> no place from ${name} enters the top 10, even with the largest boost`)}</p>`;
  }

  $("loaded-note").innerHTML = loaded ? `<div class="loaded-chip"><span>A setting loaded from a link is in use. Your targets are paused; changing any control goes back to them.</span><button class="btn" type="button" id="loaded-clear">Clear</button></div>` : "";

  // the coefficients the solver found, for readers who want the model's side
  const th = solved.theta, fmt = (v) => v.toFixed(2);
  const anyTarget = t.km != null || t.popularity != null || t.cuisines != null;
  $("solution-body").innerHTML = `
    <p>${loaded ? "The loaded setting" : anyTarget ? `To meet your targets, ${MODEL_NAMES[state.model]} uses` : `No targets set, so ${MODEL_NAMES[state.model]} uses`}</p>
    <ul class="theta">
      <li><span class="chip model">${fmt(th.w_h)}</span>History weight</li>
      <li><span class="chip control">α ${fmt(th.alpha)}</span>Distance penalty</li>
      <li><span class="chip control">β ${fmt(th.beta)}</span>Popularity penalty</li>
      <li><span class="chip diversity">λ ${fmt(th.lam / LAM_MAX)}</span>Variety step</li>
      ${boost ? `<li><span class="chip control">+${fmt(boost.amount)}</span>Neighborhood boost</li>` : ""}
    </ul>${scoreEffect(state, result)}`;
}

// What these settings do to the selected restaurant's score, on the scale they work on: the model score rescaled
// over this user's candidates (0 = the lowest, 1 = the highest), then each setting, then the adjusted score that
// orders the list before the variety step. (Moved here from the explanation card on 2026-11-26; `ledgerData`.)
function scoreEffect(state, result) {
  const i = state.selected, c = state.c, th = result.theta;
  if (i == null || !result.ranks) return "";
  const L1 = ledgerData(c, state.model, { ...th, w_h: 1 }, result.ranks, i), L = ledgerData(c, state.model, th, result.ranks, i);
  const two = (v) => v.toFixed(2).replace("-", "−");
  const step = (v) => (v === 0 ? "0" : `${v > 0 ? "+" : "−"}${Math.abs(v) < 0.005 ? "<0.01" : Math.abs(v).toFixed(2)}`);
  const row = (cls, value, label) => `<li><span class="chip ${cls}">${value}</span>${label}</li>`;
  return `<p class="effect-head">For <b>${esc(c.candidates.name[i])}</b>, on the 0–1 scale your settings work on:</p>
    <ul class="theta">
      ${row("plain", two(L1.s0), "Model score, rescaled")}
      ${row("model", th.w_h === 1 ? "off" : step(L.s0 - L1.s0), "Past preferences")}
      ${row("control", th.alpha === 0 ? "off" : step(L.dist), "Distance")}
      ${row("control", th.beta === 0 ? "off" : step(L.pop), "Popularity")}
      ${th.boost ? row("control", step(L.boost), "Neighborhood") : ""}
      ${row("plain", two(L.s2), "Adjusted score")}
    </ul>
    <p>0 is this user’s lowest-scoring candidate, 1 the highest. Cuisine variety then reorders the list; it does not change a score.</p>`;
}
