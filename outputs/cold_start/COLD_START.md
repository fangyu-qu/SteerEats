# Cold start: how many liked places does a new user need?

5,071 test users with at least 10 liked restaurants before 2020-08-01. Their profile is rebuilt from only the k most recent liked places; "full" is their whole history. Same users and held-out visits at every k. Brackets are 95% bootstrap intervals over users; the difference is paired per user.

## Logistic regression

| Liked places used | NDCG@10 | Share of full-history NDCG | Difference from full | Recall@10 | Stage-1 recall |
|---|---|---|---|---|---|
| 1 | 0.0210 [0.0188, 0.0233] | 44% | -0.0269 [-0.0304, -0.0234] | 0.0307 | 0.153 |
| 3 | 0.0274 [0.0247, 0.0301] | 57% | -0.0205 [-0.0239, -0.0170] | 0.0390 | 0.213 |
| 5 | 0.0318 [0.0290, 0.0347] | 66% | -0.0161 [-0.0194, -0.0127] | 0.0453 | 0.245 |
| 10 | 0.0394 [0.0360, 0.0427] | 82% | -0.0085 [-0.0110, -0.0059] | 0.0576 | 0.277 |
| full | 0.0479 [0.0445, 0.0514] | 100% | +0.0000 [+0.0000, +0.0000] | 0.0705 | 0.314 |

## LightGBM

| Liked places used | NDCG@10 | Share of full-history NDCG | Difference from full | Recall@10 | Stage-1 recall |
|---|---|---|---|---|---|
| 1 | 0.0236 [0.0213, 0.0259] | 49% | -0.0249 [-0.0285, -0.0213] | 0.0329 | 0.153 |
| 3 | 0.0310 [0.0281, 0.0338] | 64% | -0.0174 [-0.0206, -0.0140] | 0.0448 | 0.213 |
| 5 | 0.0340 [0.0311, 0.0369] | 70% | -0.0144 [-0.0176, -0.0115] | 0.0488 | 0.245 |
| 10 | 0.0417 [0.0383, 0.0452] | 86% | -0.0068 [-0.0092, -0.0045] | 0.0608 | 0.277 |
| full | 0.0485 [0.0447, 0.0520] | 100% | +0.0000 [+0.0000, +0.0000] | 0.0694 | 0.314 |

## LambdaRank

| Liked places used | NDCG@10 | Share of full-history NDCG | Difference from full | Recall@10 | Stage-1 recall |
|---|---|---|---|---|---|
| 1 | 0.0238 [0.0213, 0.0263] | 48% | -0.0260 [-0.0296, -0.0225] | 0.0336 | 0.153 |
| 3 | 0.0308 [0.0281, 0.0337] | 62% | -0.0189 [-0.0223, -0.0155] | 0.0452 | 0.213 |
| 5 | 0.0356 [0.0325, 0.0387] | 72% | -0.0141 [-0.0174, -0.0110] | 0.0512 | 0.245 |
| 10 | 0.0432 [0.0398, 0.0465] | 87% | -0.0066 [-0.0090, -0.0041] | 0.0623 | 0.277 |
| full | 0.0498 [0.0461, 0.0533] | 100% | +0.0000 [+0.0000, +0.0000] | 0.0722 | 0.314 |

Written by `src/32_cold_start.py`.
