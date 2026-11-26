// Steering console (V1): outcome targets, not raw penalty weights. The solver finds each model's controls.
import { CHECK, CROSS, MODEL_NAMES } from "./common.js";
import { LAM_MAX } from "../steer.js";

const POP_OPTIONS = [[null, "No preference"], [0.5, "Average below the 50th percentile"], [0.4, "Below the 40th percentile"],
  [0.3, "Below the 30th percentile"], [0.2, "Below the 20th percentile"]];
const CUI_OPTIONS = [[null, "No preference"], [5, "At least 5 cuisine groups"], [6, "At least 6"], [7, "At least 7"], [8, "At least 8"]];

const ordinal = (n) => `${n}${[11, 12, 13].includes(n % 100) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" })[n % 10] ?? "th"}`;

export function mountConsole(root, state, onChangeRaw, actions = {}) {
  // Any console control leaves a setting loaded from the trade-off view and returns to outcome targets.
  const onChange = () => { state.thetaOverride = null; onChangeRaw(); };
  root.innerHTML = `
    <h2>Steer your recommendations</h2>
    <div class="ctl sayit" id="sayit"></div>
    <p class="ctl-help spark-intro">The small chart under each control shows how much of the model's predicted visits for this user the top 10 keeps across that control's range (top line = the model's own list). Grey: the target cannot be met. Hover to read a value, click to set it. <span id="spark-note"></span></p>
    <div class="ctl">
      <div class="ctl-top"><label for="ctl-wh">How much your past visits count</label><span class="ctl-value" id="val-wh"></span></div>
      <div class="range-wrap"><input id="ctl-wh" type="range" min="0" max="1" step="0.05"><span class="ghost" style="right:0" title="Model default"></span></div>
      <div class="spark" id="spark-w_h"><div class="sp-svg"></div><p class="spark-read" role="status" hidden></p></div>
      <p class="ctl-help">Scales the part of each score that comes from places like ones you reviewed.</p>
    </div>
    <div class="ctl">
      <div class="ctl-top"><label class="switch" for="ctl-km-on"><input id="ctl-km-on" type="checkbox"> Keep it close</label><span class="ctl-value" id="val-km"></span></div>
      <input id="ctl-km" type="range" min="0.5" max="6" step="0.25" aria-label="Maximum average distance in kilometres">
      <div class="spark" id="spark-km"><div class="sp-svg"></div><p class="spark-read" role="status" hidden></p></div>
      <p class="ctl-help">Average distance of the top 10 from your usual area.</p>
    </div>
    <div class="ctl">
      <label class="field" for="ctl-pop"><span>Less mainstream</span><select id="ctl-pop">${POP_OPTIONS.map(([v, t]) => `<option value="${v ?? ""}">${t}</option>`).join("")}</select></label>
      <div class="spark" id="spark-popularity"><div class="sp-svg"></div><p class="spark-read" role="status" hidden></p></div>
      <p class="ctl-help">Popularity is each place's percentile among your candidates, by reviews in the past year.</p>
    </div>
    <div class="ctl">
      <label class="field" for="ctl-cui"><span>Cuisine variety</span><select id="ctl-cui">${CUI_OPTIONS.map(([v, t]) => `<option value="${v ?? ""}">${t}</option>`).join("")}</select></label>
      <div class="spark" id="spark-cuisines"><div class="sp-svg"></div><p class="spark-read" role="status" hidden></p></div>
    </div>
    <div class="ctl" id="boost-ctl">
      <span class="ctl-top"><span class="lbl">Neighborhood</span></span>
      <div id="boost-status"></div>
      <p class="ctl-help">Click a neighborhood on the map to bring one of its places into your top 10. Shading shows how far your other controls would have to move to get there.</p>
    </div>
    <button class="btn-reset" type="button" id="ctl-reset">Reset to the model's list</button>
    <div class="solution" id="solution" aria-live="polite"></div>`;

  const $ = (id) => root.querySelector("#" + id);
  const sync = () => {
    $("ctl-wh").value = state.w_h;
    $("val-wh").textContent = `${Math.round(state.w_h * 100)}%`;
    const on = state.targets.km != null;
    $("ctl-km-on").checked = on;
    $("ctl-km").disabled = !on;
    $("ctl-km").value = state.kmDraft;
    $("val-km").textContent = on ? `within ${state.kmDraft} km` : "off";
    $("ctl-pop").value = state.targets.popularity ?? "";
    $("ctl-cui").value = state.targets.cuisines ?? "";
  };
  $("ctl-wh").addEventListener("input", (e) => { state.w_h = +e.target.value; sync(); onChange(); });
  $("ctl-km-on").addEventListener("change", (e) => { state.targets.km = e.target.checked ? state.kmDraft : null; sync(); onChange(); });
  $("ctl-km").addEventListener("input", (e) => { state.kmDraft = +e.target.value; state.targets.km = state.kmDraft; sync(); onChange(); });
  $("ctl-pop").addEventListener("change", (e) => { state.targets.popularity = e.target.value === "" ? null : +e.target.value; onChange(); });
  $("ctl-cui").addEventListener("change", (e) => { state.targets.cuisines = e.target.value === "" ? null : +e.target.value; onChange(); });
  $("ctl-reset").addEventListener("click", () => {
    state.w_h = 1; state.targets = { km: null, popularity: null, cuisines: null }; state.boostNta = null; actions.clearFilters?.(); sync(); onChange();
  });
  root.addEventListener("click", (e) => {
    if (e.target.closest("#boost-clear")) actions.clearBoost?.();
    if (e.target.closest("#loaded-clear")) actions.clearLoaded?.();
  });
  sync();
  return { sync };
}

export function renderSolution(root, state, result) {
  const { solved, other, boost } = result;
  const bs = root.querySelector("#boost-status");
  if (!boost) bs.innerHTML = `<p class="ctl-help">None selected.</p>`;
  else {
    const i = state.c.nta.indexOf(boost.nta), name = state.c.candidates.nta_name[i];
    bs.innerHTML = `<div class="boost-chip"><span><b>${name}</b><span class="mono"> +${boost.amount.toFixed(2)}</span>
      ${boost.reached ? "" : " · not reached"}</span><button type="button" id="boost-clear" aria-label="Remove neighborhood boost">Remove</button></div>`;
  }
  const t = state.targets;
  const badges = [];
  const badge = (ok, text) => `<span class="met ${ok ? "yes" : "no"}">${ok ? CHECK : CROSS}${text}</span>`;
  if (t.km != null) badges.push(badge(solved.met.km, `${solved.outcomes.km.toFixed(1)} km`));
  if (t.popularity != null) badges.push(badge(solved.met.popularity, `${ordinal(Math.round(solved.outcomes.popularity * 100))} pct`));
  if (t.cuisines != null) badges.push(badge(solved.met.cuisines, `${solved.outcomes.cuisines} cuisines`));
  const th = solved.theta, o = other.theta;
  const fmt = (v) => v.toFixed(2);
  const anyTarget = t.km != null || t.popularity != null || t.cuisines != null;
  const loaded = solved.loaded;
  const otherName = MODEL_NAMES[state.versus];
  root.querySelector("#solution").innerHTML = `
    ${loaded ? `<div class="boost-chip"><span>Setting loaded from the trade-off view. Targets are paused; changing any control goes back to them.</span><button type="button" id="loaded-clear">Clear</button></div>`
      : badges.length ? `<div class="chips">${badges.join("")}</div>` : `<p>No targets set. The list is the model's own ranking${state.w_h < 1 ? ", with your history weight applied" : ""}.</p>`}
    <p>${MODEL_NAMES[state.model]} used</p>
    <div class="chips theta"><span class="chip model">history ${fmt(th.w_h)}</span><span class="chip control">α ${fmt(th.alpha)}</span>
      <span class="chip control">β ${fmt(th.beta)}</span><span class="chip diversity">λ ${fmt(th.lam / LAM_MAX)}</span></div>
    ${boost ? `<p>Neighborhood boost <span class="num">+${boost.amount.toFixed(2)}</span>; ${otherName} would need <span class="num">+${boost.other.toFixed(2)}</span>.</p>` : ""}
    ${loaded ? `<p>${otherName} uses the same setting.</p>` : ""}
    ${anyTarget ? `<p>${otherName} would need α <span class="num">${fmt(o.alpha)}</span>, β <span class="num">${fmt(o.beta)}</span>,
      λ <span class="num">${fmt(o.lam / LAM_MAX)}</span> for the same targets.</p>` : ""}`;
}
