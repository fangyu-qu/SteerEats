# LLM aspect features: plan (nothing has been sent)

Model `claude-haiku-5-5`, one request per restaurant, as one message batch (half price). Text before 2018-07-01 only.

| Quantity | Value |
|---|---|
| Restaurants with sampled text | 20,040 |
| Requests (restaurants with at least 5 sampled reviews) | 18,087 |
| Review text sent | 72,289,215 characters (mean 3,997 per request, largest 12,549) |
| Estimated input tokens (characters / 4, instructions and schema included) | about 25.9 million |

Estimated cost of the batch. The output side depends on how much the model reasons before it answers, which only a real call shows, so three cases are given:

| Output tokens per request (answer + reasoning) | Input | Output | Total |
|---|---|---|---|
| 150 | $1.30 | $0.68 | $1.97 |
| 500 | $1.30 | $2.26 | $3.56 |
| 2,000 (the hard cap, max_tokens) | $1.30 | $9.04 | $10.34 |

List price used: $0.10 input and $0.50 output per million tokens, times 0.5 for a batch. These are estimates from character counts, not measured token counts: `--count` measures input tokens exactly (free, needs a key) before anything is submitted.

Written by `src/35_llm_aspects.py --plan`.
