# RQ3: do text features improve ranking?

Two-stage protocol of step 15, test period from 2020-08-01. Seeds: 1, 2, 42 (the seed drives negative sampling and the selection users). Within a seed every feature set uses the same training rows and the same test users, so differences are paired per user.
All text is from before 2018-07-01 (see TEXT_SAMPLE.md).

## NDCG@10 on test (8,025 users), mean over seeds [min, max]

| Model | Base (30 features) | Base + bag-of-words text | Base + LLM aspects |
|---|---|---|---|
| Logistic regression | 0.0481 [0.0479, 0.0482] | 0.0486 [0.0485, 0.0487] | 0.0475 [0.0474, 0.0477] |
| LightGBM | 0.0490 [0.0482, 0.0498] | 0.0507 [0.0505, 0.0510] | 0.0486 [0.0482, 0.0491] |
| LambdaRank | 0.0502 [0.0497, 0.0508] | 0.0515 [0.0507, 0.0521] | 0.0511 [0.0509, 0.0513] |

## Recall@10 on test, mean over seeds

| Model | Base (30 features) | Base + bag-of-words text | Base + LLM aspects |
|---|---|---|---|
| Logistic regression | 0.0677 | 0.0685 | 0.0670 |
| LightGBM | 0.0683 | 0.0705 | 0.0677 |
| LambdaRank | 0.0700 | 0.0713 | 0.0707 |

## Base + bag-of-words text minus base: paired difference in NDCG@10

Per seed: mean difference over test users with a 95% bootstrap interval (2,000 resamples of users).

| Model | seed 1 | seed 2 | seed 42 | mean of seeds |
|---|---|---|---|---|
| Logistic regression | +0.0006 [-0.0007, +0.0019] | +0.0004 [-0.0008, +0.0017] | +0.0006 [-0.0007, +0.0019] | +0.0005 |
| LightGBM | +0.0016 [+0.0002, +0.0030] | +0.0012 [-0.0002, +0.0026] | +0.0022 [+0.0008, +0.0036] | +0.0017 |
| LambdaRank | +0.0024 [+0.0012, +0.0036] | +0.0016 [+0.0003, +0.0029] | -0.0001 [-0.0011, +0.0010] | +0.0013 |

## Base + LLM aspects minus base: paired difference in NDCG@10

Per seed: mean difference over test users with a 95% bootstrap interval (2,000 resamples of users).

| Model | seed 1 | seed 2 | seed 42 | mean of seeds |
|---|---|---|---|---|
| Logistic regression | -0.0005 [-0.0016, +0.0005] | -0.0006 [-0.0016, +0.0005] | -0.0007 [-0.0017, +0.0004] | -0.0006 |
| LightGBM | -0.0008 [-0.0019, +0.0003] | -0.0015 [-0.0025, -0.0004] | +0.0009 [-0.0003, +0.0020] | -0.0005 |
| LambdaRank | +0.0012 [-0.0000, +0.0022] | +0.0012 [+0.0001, +0.0023] | +0.0003 [-0.0008, +0.0015] | +0.0009 |

## Base + LLM aspects minus base + bag-of-words text: paired difference in NDCG@10

Per seed: mean difference over test users with a 95% bootstrap interval (2,000 resamples of users).

| Model | seed 1 | seed 2 | seed 42 | mean of seeds |
|---|---|---|---|---|
| Logistic regression | -0.0011 [-0.0025, +0.0003] | -0.0009 [-0.0022, +0.0003] | -0.0013 [-0.0026, +0.0001] | -0.0011 |
| LightGBM | -0.0024 [-0.0038, -0.0009] | -0.0027 [-0.0040, -0.0012] | -0.0013 [-0.0028, +0.0001] | -0.0022 |
| LambdaRank | -0.0012 [-0.0024, +0.0000] | -0.0003 [-0.0015, +0.0009] | +0.0004 [-0.0008, +0.0017] | -0.0004 |

## Share of attribution on the text columns

Mean absolute contribution of the text columns as a share of all columns, on the candidate rows of 400 test users, mean over seeds (logistic regression: standardized coefficient times value; tree models: path-dependent TreeSHAP of the raw score).

| Set | Model | Share |
|---|---|---|
| Base + bag-of-words text | LambdaRank | 19.3% |
| Base + bag-of-words text | LightGBM | 21.2% |
| Base + bag-of-words text | Logistic regression | 32.4% |
| Base + LLM aspects | LambdaRank | 6.8% |
| Base + LLM aspects | LightGBM | 8.8% |
| Base + LLM aspects | Logistic regression | 27.0% |

## Selected settings

| Seed | Set | Features | Logistic C | LightGBM leaves / rounds | LambdaRank leaves / rounds |
|---|---|---|---|---|---|
| 1 | base | 30 | 0.1 | 15 / 557 | 31 / 102 |
| 1 | bow | 48 | 1 | 31 / 418 | 15 / 300 |
| 1 | llm | 52 | 0.1 | 31 / 421 | 31 / 202 |
| 2 | base | 30 | 1 | 15 / 372 | 63 / 100 |
| 2 | bow | 48 | 10 | 31 / 376 | 31 / 248 |
| 2 | llm | 52 | 0.1 | 31 / 394 | 31 / 175 |
| 42 | base | 30 | 10 | 31 / 233 | 31 / 127 |
| 42 | bow | 48 | 10 | 15 / 662 | 31 / 154 |
| 42 | llm | 52 | 10 | 63 / 260 | 31 / 227 |

Written by `src/36_llm_ablation.py --report`.
