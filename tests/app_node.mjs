// The diner app's engine (web/src/app/engine.js), run from Python (tests/check_app.py).
// Reads {"jobs": [{profiles, starts, filters, targets, w_h, strategy}]} on stdin, writes one result per job.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildPool, prepare, solve } from "../web/src/app/engine.js";

const APP = join(dirname(fileURLToPath(import.meta.url)), "..", "web", "public", "data", "app");
const load = (f) => JSON.parse(readFileSync(join(APP, f), "utf8"));
const index = load("index.json"), places = load("places.json"), cache = new Map();
const profile = (id) => { if (!cache.has(id)) cache.set(id, load(`${id}.json`)); return cache.get(id); };
const { jobs } = JSON.parse(readFileSync(0, "utf8"));
const out = jobs.map((job) => {
  const members = job.profiles.map((id, k) => ({ profile: profile(id), start: job.starts[k] }));
  const pool = buildPool(places, index.cuisine_names, members, job.filters);
  if (!pool.ids.length) return { ids: [] };
  const pre = prepare(places, index.dist_cap_km, members, pool.ids);
  const res = solve(members, pre, job.targets, job.w_h, job.strategy);
  return { ids: pool.ids, top: res.top.map((j) => pool.ids[j]), count: res.count, theta: res.theta, outcomes: res.outcomes, met: res.met,
    sat: res.sat.map((row) => res.top.map((j) => row[j])), dist: pre.dist.map((row) => res.top.map((j) => row[j])) };
});
process.stdout.write(JSON.stringify(out));
