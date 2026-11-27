#!/usr/bin/env python3
"""Step 40: how accurately is one sentence turned into steering settings?

tests/nl_cases.json holds sentences with the slots they should give (written before the model was run).
Two readers are scored on them:
  llm      server/nl.py: claude-haiku-5-5 with a JSON schema (what /api/parse returns)
  keyword  web/src/nl.js keywordParse: the browser's fallback when the API is off (English only)
A slot is right when it equals the expected value exactly (lists as sets, the area without regard to case).
A sentence is right when all eleven setting slots are right; `unsupported` changes nothing in the app and is
scored on its own (does the reader notice a wish that no setting covers).

Usage
  python src/40_nl_eval.py --plan      # no call: how many sentences, estimated cost
  python src/40_nl_eval.py --run       # needs ANTHROPIC_API_KEY and the owner's go-ahead: one call per sentence
  python src/40_nl_eval.py --report    # no call: score the saved answers, write the report
Model answers are kept in data/nl_eval/answers.json (not in git), so --report never calls the model.
Outputs: outputs/nl_steering/NL_EVAL.md, nl_eval.csv; web/public/data/nl_eval.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from common import OUTPUTS  # noqa: E402
from server import nl  # noqa: E402

CASES = ROOT / "tests" / "nl_cases.json"
CACHE = ROOT / "data" / "nl_eval" / "answers.json"
OUT = OUTPUTS / "nl_steering"
SETTINGS = ["in_scope", "max_km", "popularity", "variety", "history", "cuisines", "price_max", "price_min", "needs", "borough", "area"]
KINDS = {"single": "One setting", "multi": "Several settings", "vague": "Vague", "unsupported": "With a wish no setting covers",
         "out": "Not about where to eat", "zh": "Chinese"}
# Measured on the four calls made while the feature was built (2026-11-25): 9,368 input and 372 output tokens.
PLAN_IN, PLAN_OUT = 9368 / 4, 372 / 4


def load_cases():
    return json.loads(CASES.read_text())["cases"]


def expected(case):
    e = nl.empty()
    e.update({k: v for k, v in case["expect"].items() if k != "has_unsupported"})
    return e, bool(case["expect"].get("has_unsupported", False))


def same(slot, got, want):
    if slot in ("cuisines", "needs"):
        return set(got) == set(want)
    if slot == "area":
        return (got or "").casefold() == (want or "").casefold()
    if slot == "max_km":
        return got == want or (got is not None and want is not None and abs(got - want) < 1e-9)
    return got == want


def filled(slot, value):
    return value is False if slot == "in_scope" else value not in (None, [], "")


def keyword(sentences):
    r = subprocess.run(["node", str(ROOT / "tests" / "nl_node.mjs")], input=json.dumps({"cmd": "keyword", "sentences": sentences}),
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def plan():
    cases = load_cases()
    n = len(cases)
    cost = n * nl.cost(PLAN_IN, PLAN_OUT)
    print(f"{n} sentences in {CASES.relative_to(ROOT)}; one call each to {nl.MODEL} (no batch: answers are needed at once).")
    print(f"Per call about {PLAN_IN:,.0f} input tokens (system prompt and schema) and {PLAN_OUT:,.0f} output tokens, "
          "measured on four calls on 2026-11-25.")
    print(f"List price ${nl.PRICE_IN:.2f} input and ${nl.PRICE_OUT:.2f} output per million tokens: about ${cost:.3f} for the whole set.")
    print("Nothing was sent. Run with --run after the owner's go-ahead.")


def run():
    import anthropic
    cases = load_cases()
    client = anthropic.Anthropic(timeout=60.0, max_retries=2)
    saved = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    answers = saved.get("answers", {}) if saved.get("system") == nl.SYSTEM and saved.get("model") == nl.MODEL else {}
    tin = tout = calls = 0
    for c in cases:
        if answers.get(c["id"], {}).get("sentence") == c["sentence"]:
            continue
        try:
            slots, (a, b) = nl.read(client, c["sentence"])
        except (anthropic.APIError, ValueError) as err:
            print(f"  {c['id']}: failed ({type(err).__name__})")
            continue
        answers[c["id"]] = {"sentence": c["sentence"], "slots": slots, "input_tokens": a, "output_tokens": b}
        tin += a; tout += b; calls += 1
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"model": nl.MODEL, "system": nl.SYSTEM, "answers": answers}, ensure_ascii=False, indent=1))
    print(f"{calls} calls, {tin:,} input and {tout:,} output tokens: ${nl.cost(tin, tout):.4f} at list price. "
          f"{len(answers)} of {len(cases)} sentences have an answer.")


def report():
    import pandas as pd
    cases = load_cases()
    saved = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    current = saved.get("system") == nl.SYSTEM and saved.get("model") == nl.MODEL
    answers = saved.get("answers", {}) if current else {}
    llm = {c["id"]: answers[c["id"]]["slots"] for c in cases if answers.get(c["id"], {}).get("sentence") == c["sentence"]}
    kw = dict(zip((c["id"] for c in cases), keyword([c["sentence"] for c in cases])))
    readers = {"llm": llm, "keyword": kw} if len(llm) == len(cases) else {"keyword": kw}
    names = {"llm": f"Language model ({nl.MODEL})", "keyword": "Keyword rules"}
    rows = []
    for c in cases:
        want, want_un = expected(c)
        for r, got_all in readers.items():
            got = got_all[c["id"]]
            ok = {s: same(s, got[s], want[s]) for s in SETTINGS}
            rows.append(dict(id=c["id"], kind=c["kind"], reader=r, sentence=c["sentence"], exact=all(ok.values()),
                             unsupported_ok=bool(got["unsupported"]) == want_un,
                             **{f"ok_{s}": v for s, v in ok.items()}, **{f"want_{s}": filled(s, want[s]) for s in SETTINGS},
                             **{f"got_{s}": filled(s, got[s]) for s in SETTINGS}, want_unsupported=want_un,
                             got_json=json.dumps({k: v for k, v in got.items() if filled(k, v)}, ensure_ascii=False)))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "nl_eval.csv", index=False)
    n = len(cases)
    L = ["# One-sentence steering: how accurately is a sentence read?", "",
         f"{n} sentences from `tests/nl_cases.json`, each with the settings it should give (written before the model was run on "
         "them, by the same authors as the prompt, so this is a check of the instructions, not of unseen language). "
         "A sentence counts as right when all eleven setting slots equal the expected values. Brackets are 95% Wilson intervals.", ""]
    if "llm" not in readers:
        L += [f"**The language model has not been run on this set yet** ({len(llm)} of {n} answers saved for the current prompt). "
              "`python src/40_nl_eval.py --plan` prints the cost; `--run` needs the owner's go-ahead. Below: the keyword rules only.", ""]
    L += ["## Sentences read completely right", "", "| Kind | Sentences | " + " | ".join(names[r] for r in readers) + " |",
          "|---|---|" + "---|" * len(readers)]
    for kind, label in list(KINDS.items()) + [(None, "All")]:
        d = df if kind is None else df[df.kind == kind]
        cells = []
        for r in readers:
            x = d[d.reader == r]
            k, m = int(x.exact.sum()), len(x)
            lo, hi = wilson(k, m)
            cells.append(f"{k} of {m} ({k / m:.0%}) [{lo:.0%}, {hi:.0%}]" if kind is None else f"{k} of {m}")
        L.append(f"| {label} | {len(d) // len(readers)} | " + " | ".join(cells) + " |")
    L += ["", "## By setting", "",
          "\"Should be set\": sentences whose expected answer fills the slot; right = the reader gave exactly that value. "
          "\"Wrongly set\": sentences where the slot should stay empty and the reader filled it. For `in_scope` \"set\" means "
          "the sentence is marked as not about where to eat.", "",
          "| Setting | Should be set | " + " | ".join(f"{names[r]}: right" for r in readers) + " | "
          + " | ".join(f"{names[r]}: wrongly set" for r in readers) + " |", "|---|---|" + "---|" * (2 * len(readers))]
    for s in SETTINGS:
        right, wrong = [], []
        for r in readers:
            x = df[df.reader == r]
            should = x[x[f"want_{s}"]]
            rest = x[~x[f"want_{s}"]]
            right.append(f"{int(should[f'ok_{s}'].sum())} of {len(should)}")
            wrong.append(f"{int(rest[f'got_{s}'].sum())} of {len(rest)}")
        L.append(f"| {s} | {int(df[df.reader == next(iter(readers))][f'want_{s}'].sum())} | " + " | ".join(right) + " | " + " | ".join(wrong) + " |")
    if "llm" in readers:
        x = df[df.reader == "llm"]
        L += ["", "## Wishes no setting covers", "",
              f"The language model lists such wishes so the app can say \"no setting for: quiet\". It noticed one in "
              f"{int((x.want_unsupported & x.unsupported_ok).sum())} of {int(x.want_unsupported.sum())} sentences that hold one, and "
              f"reported one in {int((~x.want_unsupported & ~x.unsupported_ok).sum())} of {int((~x.want_unsupported).sum())} that hold none. "
              "The keyword rules cannot do this."]
        tin = sum(a["input_tokens"] for a in answers.values()); tout = sum(a["output_tokens"] for a in answers.values())
        L += ["", "## Cost", "", f"{len(answers)} calls, {tin:,} input and {tout:,} output tokens: ${nl.cost(tin, tout):.4f} at list price "
              f"(${nl.cost(tin, tout) / max(len(answers), 1):.5f} per sentence)."]
    wrong = df[~df.exact]
    L += ["", "## Sentences not read completely right", ""]
    for r in readers:
        w = wrong[wrong.reader == r]
        L += [f"### {names[r]} ({len(w)})", ""] + ([f"- `{x.id}` \"{x.sentence}\" → `{x.got_json}`" for x in w.itertuples()] or ["None."]) + [""]
    L += ["The keyword rules are English only, so every Chinese sentence that needs a setting fails for them by design. "
          "Written by `src/40_nl_eval.py`."]
    (OUT / "NL_EVAL.md").write_text("\n".join(L) + "\n")
    # the same numbers for the Lab's Overview
    web = {"sentences": n, "model": nl.MODEL, "llm_run": "llm" in readers, "readers": {}, "kinds": [], "slots": []}
    for r in readers:
        x = df[df.reader == r]
        k = int(x.exact.sum()); lo, hi = wilson(k, n)
        web["readers"][r] = {"label": names[r], "exact": k, "share": k / n, "lo": lo, "hi": hi}
    for kind, label in KINDS.items():
        d = df[df.kind == kind]
        web["kinds"].append({"kind": kind, "label": label, "n": len(d) // len(readers), **{r: int(d[d.reader == r].exact.sum()) for r in readers}})
    for s in SETTINGS:
        row = {"slot": s}
        for r in readers:
            x = df[df.reader == r]
            row[r] = {"should": int(x[f"want_{s}"].sum()), "right": int(x[x[f"want_{s}"]][f"ok_{s}"].sum()),
                      "rest": int((~x[f"want_{s}"]).sum()), "wrongly_set": int(x[~x[f"want_{s}"]][f"got_{s}"].sum())}
        web["slots"].append(row)
    if "llm" in readers:
        tin = sum(a["input_tokens"] for a in answers.values()); tout = sum(a["output_tokens"] for a in answers.values())
        web["cost_usd"] = round(nl.cost(tin, tout), 4); web["per_sentence_usd"] = round(nl.cost(tin, tout) / len(answers), 5)
    (ROOT / "web" / "public" / "data" / "nl_eval.json").write_text(json.dumps(web, separators=(",", ":")))
    print("\n".join(L[:40]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true"); g.add_argument("--run", action="store_true"); g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    plan() if a.plan else run() if a.run else report()
