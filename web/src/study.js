// Study mode (user study, docs/USER_STUDY.md): open the app with #study=P1 (P1-P5). Two parts: the full interface
// and a list-only baseline (console and top 10, no explanation views), counterbalanced by participant number;
// three tasks per part from public/data/study.json, each timed, scored automatically, with interactions counted.
// After each part three 1-7 ratings. Results download as JSON (and stay in this browser's storage as a backup).
const KEY = (p) => `steereats-study-${p}`;
const SCALE = [1, 2, 3, 4, 5, 6, 7];

export function createStudy(participant, spec, app) {
  const n = Math.max(1, parseInt(participant.replace(/\D/g, ""), 10) || 1);
  const fullFirst = n % 2 === 1, aFirst = [1, 2].includes(((n - 1) % 4) + 1);
  const parts = [{ condition: fullFirst ? "full" : "list", set: aFirst ? "A" : "B" },
    { condition: fullFirst ? "list" : "full", set: aFirst ? "B" : "A" }];
  const result = { participant, started: new Date().toISOString(), order: parts, tasks: [], ratings: {} };
  let part = 0, step = -1, t0 = 0, actions = 0;

  const bar = document.createElement("section");
  bar.className = "study-bar";
  bar.setAttribute("aria-label", "Study task");
  document.body.append(bar);
  document.body.classList.add("study");
  app.lockCase(true);

  const save = () => { try { localStorage.setItem(KEY(participant), JSON.stringify(result)); } catch { /* storage may be unavailable */ } };
  const tasks = () => spec.sets[parts[part].set];
  const condName = (c) => (c === "full" ? "full interface" : "list only");

  function prompt(t) {
    if (t.kind === "why") return `<b>${t.place}</b> (${t.nta}) is in LightGBM's own top 10 but not in this list. Which control pushed it out?`;
    if (t.kind === "reach") return `Get at least one restaurant in <b>${t.borough}</b> into the top 10 while keeping at least ${t.keep} of the 10 places LightGBM lists first. Stay on LightGBM. Click Done when the list shows it.`;
    return `For this user, which model gives more weight to <b>${t.group_text}</b>?`;
  }

  function intro() {
    bar.innerHTML = `<div class="sb-main"><p class="sb-kicker">Session ${participant} · two parts, three tasks each</p>
      <p>Part ${part + 1}: ${condName(parts[part].condition)}. Think aloud as you work; there is no time limit.</p></div>
      <div class="sb-actions"><button type="button" class="btn-primary" id="sb-go">Start part ${part + 1}</button></div>`;
    bar.querySelector("#sb-go").onclick = () => { step = 0; app.setCondition(parts[part].condition); show(); };
  }

  async function show() {
    const t = tasks()[step];
    bar.innerHTML = `<div class="sb-main"><p class="sb-kicker">Part ${part + 1} (${condName(parts[part].condition)}) · task ${step + 1} of ${tasks().length}</p><p>Loading…</p></div>`;
    await app.applyState(t);
    actions = 0; t0 = performance.now();
    const opts = t.options ? `<fieldset class="sb-opts"><legend class="sr-only">Answer</legend>${t.options.map((o, k) =>
      `<label><input type="radio" name="sb-ans" value="${o}"${k === 0 ? "" : ""}> ${o}</label>`).join("")}</fieldset>` : "";
    bar.innerHTML = `<div class="sb-main"><p class="sb-kicker">Part ${part + 1} (${condName(parts[part].condition)}) · task ${step + 1} of ${tasks().length}</p>
      <p>${prompt(t)}</p>${opts}</div>
      <div class="sb-actions"><button type="button" class="btn-reset" id="sb-skip">Give up</button><button type="button" class="btn-primary" id="sb-done">Done</button></div>`;
    bar.querySelector("#sb-done").onclick = () => finish(t, false);
    bar.querySelector("#sb-skip").onclick = () => finish(t, true);
  }

  function finish(t, gaveUp) {
    const answer = bar.querySelector('input[name="sb-ans"]:checked')?.value ?? null;
    if (!gaveUp && t.options && !answer) { bar.querySelector(".sb-opts").classList.add("need"); return; }
    const o = app.outcome();
    let correct = false;
    if (!gaveUp) {
      if (t.kind === "reach") {
        const keep = o.top.filter((i) => o.defTop.includes(i)).length;
        correct = o.model === t.state.model && o.top.some((i) => o.c.candidates.borough[i] === t.borough) && keep >= t.keep;
      } else correct = answer === t.answer;
    }
    result.tasks.push({ part: part + 1, set: parts[part].set, condition: parts[part].condition, kind: t.kind, case: t.case, answer,
      gave_up: gaveUp, correct, seconds: +((performance.now() - t0) / 1000).toFixed(1), actions, final: { model: o.model, hash: location.hash } });
    save();
    step += 1;
    if (step < tasks().length) show(); else ratings();
  }

  function ratings() {
    const cond = parts[part].condition;
    bar.innerHTML = `<div class="sb-main"><p class="sb-kicker">Part ${part + 1} done: ${condName(cond)}. 1 = strongly disagree, 7 = strongly agree.</p>
      ${spec.likert.map((q, k) => `<fieldset class="sb-likert"><legend>${q}</legend>${SCALE.map((v) =>
        `<label><input type="radio" name="sb-q${k}" value="${v}"> ${v}</label>`).join("")}</fieldset>`).join("")}</div>
      <div class="sb-actions"><button type="button" class="btn-primary" id="sb-next">Continue</button></div>`;
    bar.querySelector("#sb-next").onclick = () => {
      const vals = spec.likert.map((_, k) => bar.querySelector(`input[name="sb-q${k}"]:checked`)?.value);
      if (vals.some((v) => v == null)) { bar.querySelectorAll(".sb-likert").forEach((f, k) => f.classList.toggle("need", vals[k] == null)); return; }
      result.ratings[cond] = Object.fromEntries(spec.likert.map((q, k) => [q, +vals[k]]));
      save();
      part += 1;
      if (part < parts.length) intro(); else end();
    };
  }

  function end() {
    result.finished = new Date().toISOString();
    save();
    app.setCondition("full");
    bar.innerHTML = `<div class="sb-main"><p class="sb-kicker">Session ${participant} complete</p><p>Thank you. Save the results file and put it in data/user_study/.</p></div>
      <div class="sb-actions"><button type="button" class="btn-primary" id="sb-save">Download results</button></div>`;
    bar.querySelector("#sb-save").onclick = () => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(new Blob([JSON.stringify(result, null, 1)], { type: "application/json" }));
      a.download = `steereats-study-${participant}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
    };
  }

  intro();
  return { action() { actions += 1; } };
}
