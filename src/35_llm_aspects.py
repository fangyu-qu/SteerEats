#!/usr/bin/env python3
"""Step 35: LLM aspect features from review text (RQ3). The model reads and labels; it never ranks.

For each restaurant with at least 5 sampled reviews, a Claude model reads the fixed text sample of
src/34_text_sample.py (all text from before TEXT_CUTOFF) and returns a fixed schema: six aspect scores
(food, service, value, ambience, wait, noise; 0 = not mentioned, 1 = bad ... 5 = excellent) and eight
occasion tags. The answers become static restaurant features for src/36_llm_ablation.py, next to the
bag-of-words features built from exactly the same text.

  python src/35_llm_aspects.py --plan       # no key, no call: requests, text volume, estimated cost
  python src/35_llm_aspects.py --count      # needs a key: exact input tokens for 50 requests (free endpoint)
  python src/35_llm_aspects.py --pilot      # needs a key: 20 restaurants, one call each; prints tokens and cost
  python src/35_llm_aspects.py --submit     # needs a key and the owner's go-ahead: one message batch (half price)
  python src/35_llm_aspects.py --collect    # fetch the batch when it has ended
  python src/35_llm_aspects.py --features   # answers -> data/clean/text_features_llm.npz; then run step 36

Rules: the key is read from the environment (ANTHROPIC_API_KEY or an `ant auth login` profile), never from
the repo. Only star rating, month and text are sent; no user ids. Answers stay in data/llm_aspects (not in
git). Needs `pip install anthropic` for everything except --plan and --features.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CLEAN, OUTPUTS, RESTAURANT_ATTRS, ROOT, TEXT_CUTOFF, TEXT_SAMPLE

MODEL = "claude-haiku-5-5"
# USD per million tokens (input, output) for prompts up to 100K tokens, list price cached 2026-11-03 in the
# claude-api skill. A message batch costs half.
PRICE_IN, PRICE_OUT, BATCH = 0.10, 0.50, 0.5
MIN_REVIEWS = 5
MAX_TOKENS = 2000
CACHE = ROOT / "data" / "llm_aspects"
OUT = OUTPUTS / "llm_features"
ASPECTS = ["food", "service", "value", "ambience", "wait", "noise"]
TAGS = ["date_night", "groups", "families", "quick_bite", "special_occasion", "solo", "late_night", "takeout"]
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ASPECTS + TAGS,
    "properties": {a: {"type": "integer", "enum": [0, 1, 2, 3, 4, 5]} for a in ASPECTS} | {t: {"type": "boolean"} for t in TAGS},
}
SYSTEM = """You label restaurants from customer reviews for a research study.

You are given a sample of reviews of one restaurant (star rating, month, text). Report only what the reviews say.

Aspect scores, each 0 to 5:
- 0 means the reviews do not mention the aspect at all. Use 0 freely; do not guess.
- 1 = reviewers are clearly negative, 3 = mixed or neutral, 5 = clearly very positive.
- food: taste and quality of food and drink. service: staff and service. value: prices for what you get.
  ambience: decor, atmosphere, comfort. wait: 5 = no wait or fast, 1 = long waits. noise: 5 = quiet, 1 = very loud.

Occasion tags are true only when at least one review supports them: date_night, groups (large parties),
families (children), quick_bite (fast, casual), special_occasion (celebrations), solo (eating alone),
late_night (open or visited late), takeout (takeout or delivery)."""


def requests_table():
    s = pd.read_parquet(TEXT_SAMPLE)
    n = s.groupby("gmap_id").size()
    s = s[s.gmap_id.isin(n[n >= MIN_REVIEWS].index)]
    return s, n


def prompt(group):
    return "\n\n".join(f"[{int(r.rating)} stars, {r.month}] {r.text}" for r in group.itertuples())


def params(text):
    return dict(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": text}])


def plan():
    s, n = requests_table()
    per = s.groupby("gmap_id").text.agg(lambda t: int(t.str.len().sum() + 24 * len(t)))
    chars = int(per.sum())
    tok_in = (chars + len(per) * (len(SYSTEM) + len(json.dumps(SCHEMA)))) / 4
    lines = [
        "# LLM aspect features: plan (nothing has been sent)", "",
        f"Model `{MODEL}`, one request per restaurant, as one message batch (half price). Text before {TEXT_CUTOFF} only.", "",
        "| Quantity | Value |", "|---|---|",
        f"| Restaurants with sampled text | {len(n):,} |",
        f"| Requests (restaurants with at least {MIN_REVIEWS} sampled reviews) | {len(per):,} |",
        f"| Review text sent | {chars:,} characters (mean {chars / len(per):,.0f} per request, largest {int(per.max()):,}) |",
        f"| Estimated input tokens (characters / 4, instructions and schema included) | about {tok_in / 1e6:.1f} million |",
        "",
        "Estimated cost of the batch. The output side depends on how much the model reasons before it answers, "
        "which only a real call shows, so three cases are given:", "",
        "| Output tokens per request (answer + reasoning) | Input | Output | Total |", "|---|---|---|---|",
    ]
    for out_tok in (150, 500, MAX_TOKENS):
        ci, co = tok_in * PRICE_IN / 1e6 * BATCH, len(per) * out_tok * PRICE_OUT / 1e6 * BATCH
        lines.append(f"| {out_tok:,}{' (the hard cap, max_tokens)' if out_tok == MAX_TOKENS else ''} | ${ci:.2f} | ${co:.2f} | ${ci + co:.2f} |")
    lines += ["", f"List price used: ${PRICE_IN:.2f} input and ${PRICE_OUT:.2f} output per million tokens, times {BATCH} for a batch. "
              "These are estimates from character counts, not measured token counts: `--count` measures input tokens "
              "exactly (free, needs a key) before anything is submitted.", "",
              "Written by `src/35_llm_aspects.py --plan`."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LLM_PLAN.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def client():
    import anthropic
    return anthropic.Anthropic()


def count():
    s, _ = requests_table()
    ids = s.gmap_id.drop_duplicates().sample(50, random_state=42)
    api, total, chars = client(), 0, 0
    for g in ids:
        p = params(prompt(s[s.gmap_id == g]))
        total += api.messages.count_tokens(model=p["model"], system=p["system"], messages=p["messages"]).input_tokens
        chars += len(p["messages"][0]["content"])
    print(f"50 requests: {total:,} input tokens ({total / 50:,.0f} each; {chars / total:.2f} characters of review text per token)")


def pilot(n=20):
    import anthropic
    (CACHE / "answers").mkdir(parents=True, exist_ok=True)
    s, _ = requests_table()
    done = {p.stem for p in (CACHE / "answers").glob("*.json")}
    ids = [g for g in s.gmap_id.drop_duplicates().sample(frac=1, random_state=7) if g not in done][:n]
    api, tin, tout, bad = client(), 0, 0, 0
    names = pd.read_parquet(RESTAURANT_ATTRS, columns=["gmap_id", "cuisine_group"]).set_index("gmap_id").cuisine_group
    for g in ids:
        try:
            msg = api.messages.create(**params(prompt(s[s.gmap_id == g])))
        except anthropic.AuthenticationError:
            raise SystemExit("The API key was rejected.")
        except anthropic.RateLimitError:
            raise SystemExit("Rate limited after retries; run --pilot again in a few minutes.")
        except anthropic.APIStatusError as e:
            raise SystemExit(f"The API refused the request ({e.status_code}): {e.message}")
        except anthropic.APIConnectionError as e:
            raise SystemExit(f"Could not reach the API: {e}")
        tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
        text = next((b.text for b in msg.content if b.type == "text"), "")
        if msg.stop_reason != "end_turn" or not text:
            bad += 1
            print(f"  {names.get(g, '?'):<28} stop_reason={msg.stop_reason}, no usable answer")
            continue
        a = json.loads(text)
        (CACHE / "answers" / f"{g}.json").write_text(text)
        print(f"  {names.get(g, '?'):<28} " + " ".join(f"{k}={a[k]}" for k in ASPECTS) + "  tags: " + (", ".join(t for t in TAGS if a[t]) or "none")
              + f"  [{msg.usage.output_tokens} out]")
    k = len(ids)
    print(f"{k} requests, {bad} without an answer: {tin:,} input and {tout:,} output tokens ({tin / k:,.0f} and {tout / k:,.0f} each); "
          f"${(tin * PRICE_IN + tout * PRICE_OUT) / 1e6:.4f} at list price (single calls, no batch discount).")
    n_all = s.gmap_id.nunique()
    print(f"At this size the full batch of {n_all:,} would cost about ${n_all * (tin / k * PRICE_IN + tout / k * PRICE_OUT) / 1e6 * BATCH:.2f}.")


def submit():
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    CACHE.mkdir(parents=True, exist_ok=True)
    s, _ = requests_table()
    done = {p.stem for p in (CACHE / "answers").glob("*.json")} if (CACHE / "answers").exists() else set()
    ids = [g for g in s.gmap_id.drop_duplicates() if g not in done]
    index = {f"r{i}": g for i, g in enumerate(ids)}          # custom ids must be short and plain
    reqs = [Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params(prompt(s[s.gmap_id == g]))))
            for cid, g in index.items()]
    try:
        batch = client().messages.batches.create(requests=reqs)
    except anthropic.AuthenticationError:
        raise SystemExit("The API key was rejected. Set ANTHROPIC_API_KEY (or run `ant auth login`) and try again.")
    except anthropic.RateLimitError:
        raise SystemExit("Rate limited after retries. Wait a few minutes and run --submit again.")
    except anthropic.APIStatusError as e:
        raise SystemExit(f"The API refused the batch ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError as e:
        raise SystemExit(f"Could not reach the API: {e}")
    (CACHE / "batch.json").write_text(json.dumps(dict(batch_id=batch.id, index=index)))
    print(f"submitted {len(reqs):,} requests as batch {batch.id}. Most batches end within an hour; then run --collect.")


def collect():
    rec = json.loads((CACHE / "batch.json").read_text())
    api = client()
    batch = api.messages.batches.retrieve(rec["batch_id"])
    if batch.processing_status != "ended":
        raise SystemExit(f"batch {batch.id} is {batch.processing_status}: {batch.request_counts}")
    (CACHE / "answers").mkdir(parents=True, exist_ok=True)
    ok = bad = tin = tout = 0
    for result in api.messages.batches.results(rec["batch_id"]):            # results come in any order
        g = rec["index"][result.custom_id]
        if result.result.type != "succeeded" or result.result.message.stop_reason != "end_turn":
            bad += 1
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        (CACHE / "answers" / f"{g}.json").write_text(text)
        ok += 1; tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
    cost = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6 * BATCH
    print(f"{ok:,} answers saved, {bad:,} failed or cut off (run --submit again for those). "
          f"{tin:,} input and {tout:,} output tokens: ${cost:.2f} at list batch price.")


def features():
    items = pd.read_parquet(RESTAURANT_ATTRS, columns=["gmap_id"]).gmap_id.to_numpy()
    emb = np.zeros((len(items), 2 * len(ASPECTS) + len(TAGS)), np.float32)
    missing = np.ones(len(items), bool)
    for i, g in enumerate(items):
        p = CACHE / "answers" / f"{g}.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        scores = np.array([a[k] for k in ASPECTS], float)
        emb[i] = np.concatenate([np.where(scores > 0, (scores - 3) / 2, 0), scores > 0, [a[t] for t in TAGS]])
        missing[i] = False
    names = [f"llm_{a}" for a in ASPECTS] + [f"llm_{a}_mentioned" for a in ASPECTS] + [f"llm_tag_{t}" for t in TAGS] + ["llm_missing", "llm_affinity"]
    np.savez_compressed(CLEAN / "text_features_llm.npz", emb=emb, missing=missing, names=np.array(names))
    print(f"LLM features for {int((~missing).sum()):,} of {len(items):,} restaurants -> text_features_llm.npz. Now run src/36_llm_ablation.py.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    for flag in ("plan", "count", "pilot", "submit", "collect", "features"):
        g.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    {"plan": plan, "count": count, "pilot": pilot, "submit": submit, "collect": collect, "features": features}[
        next(f for f in ("plan", "count", "pilot", "submit", "collect", "features") if getattr(a, f))]()
