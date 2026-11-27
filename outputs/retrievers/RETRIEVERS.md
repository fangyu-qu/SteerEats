# Stage-1 retrievers: what each one contributes

8,025 eligible test users, snapshot 2020-08-01; 29,886 held-out visits, 8,708 of them among the 200 candidates. Stage-1 recall (mean per user): **0.3107** [0.3031, 0.3184]. Brackets are 95% bootstrap intervals over users.

| Retriever | Quota | Recall of its own list | Retrieved visits credited to it | Stage-1 recall without it | Change |
|---|---|---|---|---|---|
| Item-CF (everything reviewed) | 100 | 0.2321 [0.2250, 0.2391] | 6,468 (74.3%) | 0.2980 [0.2902, 0.3058] | -0.0127 [-0.0158, -0.0097] |
| Liked-history CF (4-5 stars) | 40 | 0.1395 [0.1337, 0.1452] | 723 (8.3%) | 0.3132 [0.3050, 0.3210] | +0.0025 [+0.0006, +0.0042] |
| Nearby and popular | 40 | 0.0774 [0.0730, 0.0819] | 979 (11.2%) | 0.3102 [0.3022, 0.3183] | -0.0006 [-0.0042, +0.0032] |
| Trending (last 3 months) | 20 | 0.0167 [0.0148, 0.0188] | 280 (3.2%) | 0.3145 [0.3063, 0.3223] | +0.0038 [+0.0011, +0.0066] |

Visits that entered only through the item-CF fill-up: 258.
Item-CF alone with 200 candidates: recall 0.3105 [0.3022, 0.3186].

"Credited" follows the order in which the union is built (item-CF first), so a visit found by two retrievers counts for the earlier one. "Without it" removes the retriever and fills the 200 from item-CF (from liked-history CF when item-CF is the one removed).

Written by `src/37_retriever_ablation.py`.
