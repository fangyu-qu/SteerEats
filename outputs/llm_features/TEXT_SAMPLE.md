# Text sample for RQ3

All text is from before **2018-07-01** (latest sampled review: 2018-06-30).

| Quantity | Value |
|---|---|
| Restaurants in the clean table | 24,724 |
| Reviews before the cutoff (first per user and restaurant) | 2,371,755 |
| ... with at least 20 characters of text | 1,194,273 |
| Restaurants with sampled text | 20,040 (81.1%) |
| Restaurants without text before the cutoff | 4,684 |
| Sampled reviews (up to 30 per restaurant) | 429,562 |
| Restaurants with the full 30 | 10,170 |
| Median sampled reviews per restaurant | 30 |
| Characters in the sample | 62,869,982 (mean 146 per review) |

Written by `src/34_text_sample.py`. The sample itself is in `data/clean/` (not in git).

## Bag-of-words features

TF-IDF over 20,224 terms (English stop words removed, terms in at least 5 restaurants and at most half of them), reduced to 16 SVD components (explained variance 0.062).

| Component | Highest-weight terms |
|---|---|
| 0 | chicken, pizza, prices, ordered, price, atmosphere, menu, recommend |
| 1 | pizza, delivery, chinese, fast, customer, slice, chicken, clean |
| 2 | pizza, slice, italian, slices, crust, pizzas, pie, cheese |
| 3 | coffee, donuts, dunkin, starbucks, breakfast, location, employees, morning |
| 4 | coffee, sushi, sandwiches, breakfast, lunch, healthy, sandwich, donuts |
| 5 | mexican, tacos, taco, burrito, guacamole, burritos, margaritas, nachos |
| 6 | mexican, tacos, authentic, italian, taco, burrito, guacamole, pasta |
| 7 | sushi, japanese, rolls, roll, sashimi, mexican, tacos, pizza |
| 8 | chinese, coffee, donuts, dunkin, starbucks, mexican, tacos, pizza |
| 9 | chinese, sushi, delivery, breakfast, coffee, prices, neighborhood, diner |
| 10 | indian, clean, chicken, jerk, jamaican, curry, rice, music |
| 11 | thai, delivery, healthy, curry, pad, called, smoothies, juice |
| 12 | thai, donuts, dunkin, coffee, fries, burger, pad, burgers |
| 13 | breakfast, diner, soup, dominican, noodles, prices, spanish, family |
| 14 | thai, pad, fast, clean, lunch, quick, curry, delivery |
| 15 | sandwich, sandwiches, subway, italian, pasta, deli, donuts, bread |
