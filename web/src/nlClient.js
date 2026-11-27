// The site's link to the optional server (server/app.py): is it there, what does it offer, and the sentence
// reader. A sentence goes to the language model through /api/parse or, when that cannot be reached, to the
// keyword reader of nl.js. Used by the Explorer's say-it box and by the diner app.
import { keywordParse, normalize } from "./nl.js";

// Where the API is. The dev server proxies /api (vite.config.js). A built site looks for "api" next to its own
// page, which is where server/app.py puts it when it serves the site; VITE_API_BASE names another host at build
// time. A site with no server behind it (GitHub Pages) gets no answer from /health and works without one.
export const API = import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? "/api" : new URL("api", document.baseURI).href);
export const WHY = { off: "this site has no language model attached", busy: "the language model is busy", down: "the language model could not be reached" };

const post = (path, body, signal) => fetch(`${API}/${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal });

// What the server offers: { llm: true|false, live: "ready"|"loading"|"off"|"failed", ... }, or null without a server.
let status = null;
export function apiStatus(fresh = false) {
  if (!status || fresh) {
    status = fetch(`${API}/health`).then((r) => (r.ok ? r.json() : null)).then((j) => (j?.ok ? j : null)).catch(() => null);
  }
  return status;
}

export async function readSentence(sentence) {
  let why = "off";
  if ((await apiStatus())?.llm) {
    const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 45000);   // the first sentence after the API starts can take long
    try {
      // the sentence travels in the request body, never in the address
      const r = await post("parse", { sentence }, ctl.signal);
      if (r.ok) return { slots: normalize((await r.json()).slots), source: "llm" };
      why = r.status === 429 ? "busy" : "down";
    } catch { why = "down"; } finally { clearTimeout(timer); }
  }
  return { slots: keywordParse(sentence), source: "keyword", why };
}

// Restaurants by name, for picking places you like (live profiles). Returns [] when nothing matches.
export async function searchPlaces(q, signal) {
  const r = await post("search", { q }, signal);
  if (!r.ok) throw new Error(r.status === 503 ? "loading" : "down");
  return (await r.json()).results;
}

// A party of one or two: each member is { profile: "profile_07" } (a sample taste) or { liked: [place ids] }.
// The answer has the party's places and each member's scores, in the shape of the exported app files.
export async function scoreParty(members) {
  const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), 60000);
  try {
    const r = await post("party", { members }, ctl.signal);
    if (!r.ok) throw new Error(r.status === 503 ? "loading" : r.status === 429 ? "busy" : "down");
    return await r.json();
  } finally { clearTimeout(timer); }
}
