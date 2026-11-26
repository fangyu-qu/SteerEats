"""The optional API of the site: one-sentence steering and live taste profiles.

Run from the repository root (the key comes from the environment, never from a file in this repository):
  .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8787

  GET  /api/health   is the API up, is a key present, are live profiles ready, calls and tokens since it started
  POST /api/parse    {"sentence": "..."} -> {"slots": {...}, "source": "llm", "model": "..."}
  POST /api/search   {"q": "joe's pizza"} -> {"results": [{"id", "name", "area", "borough", "cuisine", ...}]}
  POST /api/party    {"members": [{"profile": "profile_07"} | {"liked": [ids]}]} (one or two members)
                     -> the party's places and each member's scores (server/live.py)

The site works without this server: sentences are then read with keyword rules and people borrow sample tastes.
One-sentence steering sends the sentence alone to the model: no user id, no location, no candidate list.
Live profiles need server/bundle (src/31_export_serving_bundle.py) and send nothing anywhere: the picked places
are scored on this machine and forgotten. Nothing a person types or picks is logged or stored; a small
in-memory cache keeps the answers of repeated sentences.

If web/dist exists (cd web && npm run build), the built site is served from the same address, so one process is
the whole app: open http://127.0.0.1:8787/. To let phones on the same Wi-Fi open it, start with --host 0.0.0.0
and use this computer's address; that also opens the API, including /api/parse, to everyone on that network.

Environment: ANTHROPIC_API_KEY (required for /api/parse), STEEREATS_ORIGINS (comma-separated origins allowed to
call the API from another host, e.g. the GitHub Pages site; the local dev site needs none, Vite proxies /api).
"""
import os
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import nl

ROOT = Path(__file__).resolve().parent.parent
PARTIES_PER_MINUTE = 40  # a party takes a second or two of this machine's time

PER_MINUTE = 60          # calls to the model per minute, for the whole server
CACHE_SIZE = 500

app = FastAPI(title="SteerEats API", docs_url=None, redoc_url=None)
origins = [o.strip() for o in os.environ.get("STEEREATS_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["content-type"])

_lock = threading.Lock()
_cache = OrderedDict()
_recent = deque()
_used = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0}
_client = None
_parties = deque()
# live profiles: the bundle loads in the background (LightGBM, SHAP and 20,000 restaurants take a few seconds)
_live = {"state": "off", "bundle": None}


def _load_bundle():
    try:
        from . import live
        _live["bundle"] = live.Bundle()
        _live["state"] = "ready"
    except Exception as err:                       # the API still reads sentences without live profiles
        _live["state"] = "failed"
        print(f"live profiles are off: {type(err).__name__}: {err}", flush=True)


if (ROOT / "server" / "bundle" / "meta.json").exists():
    _live["state"] = "loading"
    threading.Thread(target=_load_bundle, daemon=True).start()


def bundle():
    if _live["state"] == "loading":
        raise HTTPException(503, "Live profiles are still loading. Try again in a few seconds.")
    if _live["state"] != "ready":
        raise HTTPException(503, "Live profiles are not set up on this server.")
    return _live["bundle"]


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(timeout=20.0, max_retries=1)   # key from ANTHROPIC_API_KEY
    return _client


class Sentence(BaseModel):
    sentence: str = Field(min_length=1, max_length=nl.MAX_SENTENCE)


@app.get("/api/health")
def health():
    with _lock:
        used = dict(_used)
    return {"ok": True, "llm": bool(os.environ.get("ANTHROPIC_API_KEY")), "model": nl.MODEL, "live": _live["state"], **used,
            "cost_usd": round(nl.cost(used["input_tokens"], used["output_tokens"]), 6)}


@app.post("/api/parse")
def parse(body: Sentence):
    sentence = " ".join(body.sentence.split())
    if not sentence:
        raise HTTPException(422, "The sentence is empty.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "The language model is not configured on this server.")
    key = sentence.lower()
    now = time.monotonic()
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            _used["cached"] += 1
            return {"slots": _cache[key], "source": "llm", "model": nl.MODEL, "cached": True}
        while _recent and now - _recent[0] > 60:
            _recent.popleft()
        if len(_recent) >= PER_MINUTE:
            raise HTTPException(429, "Too many sentences in the last minute. Try again shortly.")
        _recent.append(now)
    try:
        slots, (tin, tout) = nl.read(client(), sentence)
    except (anthropic.APIError, ValueError):
        raise HTTPException(502, "The language model could not read this sentence.") from None
    with _lock:
        _used["calls"] += 1
        _used["input_tokens"] += tin
        _used["output_tokens"] += tout
        _cache[key] = slots
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return {"slots": slots, "source": "llm", "model": nl.MODEL, "cached": False}


class Query(BaseModel):
    q: str = Field(min_length=1, max_length=60)


class Member(BaseModel):
    profile: str | None = Field(default=None, max_length=20)
    liked: list[int] | None = Field(default=None, max_length=40)


class Party(BaseModel):
    members: list[Member] = Field(min_length=1, max_length=2)


@app.post("/api/search")
def search(body: Query):
    return {"results": bundle().search(body.q)}


@app.post("/api/party")
def party(body: Party):
    b = bundle()
    now = time.monotonic()
    with _lock:
        while _parties and now - _parties[0] > 60:
            _parties.popleft()
        if len(_parties) >= PARTIES_PER_MINUTE:
            raise HTTPException(429, "Too many requests in the last minute. Try again shortly.")
        _parties.append(now)
    try:
        return b.party([m.model_dump() for m in body.members])
    except ValueError as err:
        raise HTTPException(422, str(err)) from None


# The built site, when there is one. Mounted last, so /api/* stays with the routes above.
if (ROOT / "web" / "dist" / "index.html").exists():
    app.mount("/", StaticFiles(directory=ROOT / "web" / "dist", html=True), name="site")
