# RQ2: group strategies

1,000 three-member groups per kind, drawn from 8,025 eligible test users (snapshot 2020-08-01, seed 42). Pool = union of the members' stage-1 candidates. Satisfaction = raw model score, min-max over the pool, no steering. Lists are the top 10 by group score. "Average without misery" drops places where any member is below 0.40. Brackets are 95% bootstrap intervals over groups.

| Kind | Mean cuisine cosine between members | Mean pool size |
|---|---|---|
| similar | 0.939 | 503 |
| random | 0.529 | 530 |
| diverse | 0.000 | 501 |

## Logistic regression

### Similar groups

| Strategy | Worst-off satisfaction | Satisfaction gap | Longest trip (km) | Travel gap (km) | Member NDCG@10, mean | Member NDCG@10, worst |
|---|---|---|---|---|---|---|
| Average | 0.837 [0.833, 0.840] | 0.130 [0.127, 0.133] | 12.49 [12.08, 12.91] | 10.38 [10.01, 10.75] | 0.0226 [0.0197, 0.0257] | 0.0001 [0.0000, 0.0002] |
| Least misery | 0.856 [0.853, 0.858] | 0.070 [0.068, 0.072] | 12.35 [11.93, 12.75] | 9.46 [9.10, 9.82] | 0.0165 [0.0139, 0.0193] | 0.0000 [0.0000, 0.0000] |
| Average without misery | 0.837 [0.833, 0.841] | 0.130 [0.127, 0.134] | 12.49 [12.07, 12.91] | 10.38 [10.00, 10.79] | 0.0226 [0.0196, 0.0258] | 0.0001 [0.0000, 0.0002] |

Each member's own list, for reference: NDCG@10 mean 0.0486 [0.0440, 0.0533], worst member 0.0006 [0.0001, 0.0012].

### Random groups

| Strategy | Worst-off satisfaction | Satisfaction gap | Longest trip (km) | Travel gap (km) | Member NDCG@10, mean | Member NDCG@10, worst |
|---|---|---|---|---|---|---|
| Average | 0.825 [0.822, 0.828] | 0.139 [0.136, 0.142] | 13.82 [13.43, 14.20] | 11.45 [11.09, 11.84] | 0.0203 [0.0174, 0.0237] | 0.0001 [0.0000, 0.0002] |
| Least misery | 0.848 [0.845, 0.850] | 0.066 [0.064, 0.068] | 13.77 [13.39, 14.15] | 10.43 [10.11, 10.77] | 0.0115 [0.0094, 0.0139] | 0.0000 [0.0000, 0.0000] |
| Average without misery | 0.825 [0.822, 0.828] | 0.139 [0.136, 0.142] | 13.82 [13.45, 14.19] | 11.45 [11.08, 11.81] | 0.0203 [0.0174, 0.0232] | 0.0001 [0.0000, 0.0002] |

Each member's own list, for reference: NDCG@10 mean 0.0482 [0.0434, 0.0528], worst member 0.0011 [0.0004, 0.0019].

### Diverse groups

| Strategy | Worst-off satisfaction | Satisfaction gap | Longest trip (km) | Travel gap (km) | Member NDCG@10, mean | Member NDCG@10, worst |
|---|---|---|---|---|---|---|
| Average | 0.854 [0.852, 0.857] | 0.113 [0.111, 0.115] | 18.08 [17.74, 18.44] | 15.44 [15.10, 15.78] | 0.0114 [0.0094, 0.0136] | 0.0000 [0.0000, 0.0000] |
| Least misery | 0.870 [0.868, 0.872] | 0.060 [0.058, 0.061] | 17.93 [17.57, 18.29] | 14.15 [13.82, 14.52] | 0.0089 [0.0070, 0.0108] | 0.0000 [0.0000, 0.0000] |
| Average without misery | 0.854 [0.852, 0.857] | 0.113 [0.111, 0.115] | 18.08 [17.71, 18.44] | 15.44 [15.13, 15.77] | 0.0114 [0.0095, 0.0135] | 0.0000 [0.0000, 0.0000] |

Each member's own list, for reference: NDCG@10 mean 0.0530 [0.0495, 0.0568], worst member 0.0005 [0.0001, 0.0010].


## LightGBM

### Similar groups

| Strategy | Worst-off satisfaction | Satisfaction gap | Longest trip (km) | Travel gap (km) | Member NDCG@10, mean | Member NDCG@10, worst |
|---|---|---|---|---|---|---|
| Average | 0.653 [0.647, 0.658] | 0.284 [0.278, 0.290] | 12.78 [12.37, 13.19] | 10.04 [9.66, 10.42] | 0.0210 [0.0182, 0.0240] | 0.0000 [0.0000, 0.0000] |
| Least misery | 0.688 [0.683, 0.694] | 0.152 [0.149, 0.155] | 11.85 [11.52, 12.22] | 8.52 [8.22, 8.83] | 0.0146 [0.0123, 0.0172] | 0.0000 [0.0000, 0.0000] |
| Average without misery | 0.653 [0.647, 0.659] | 0.284 [0.278, 0.290] | 12.78 [12.38, 13.18] | 10.04 [9.67, 10.43] | 0.0210 [0.0181, 0.0240] | 0.0000 [0.0000, 0.0000] |

Each member's own list, for reference: NDCG@10 mean 0.0480 [0.0433, 0.0525], worst member 0.0004 [0.0000, 0.0010].

### Random groups

| Strategy | Worst-off satisfaction | Satisfaction gap | Longest trip (km) | Travel gap (km) | Member NDCG@10, mean | Member NDCG@10, worst |
|---|---|---|---|---|---|---|
| Average | 0.600 [0.596, 0.604] | 0.330 [0.325, 0.334] | 14.17 [13.81, 14.52] | 11.07 [10.71, 11.42] | 0.0165 [0.0142, 0.0190] | 0.0000 [0.0000, 0.0000] |
| Least misery | 0.644 [0.640, 0.648] | 0.157 [0.154, 0.160] | 13.09 [12.78, 13.43] | 9.20 [8.94, 9.48] | 0.0083 [0.0066, 0.0101] | 0.0000 [0.0000, 0.0000] |
| Average without misery | 0.600 [0.596, 0.604] | 0.329 [0.325, 0.334] | 14.16 [13.80, 14.51] | 11.07 [10.72, 11.39] | 0.0164 [0.0139, 0.0189] | 0.0000 [0.0000, 0.0000] |

Each member's own list, for reference: NDCG@10 mean 0.0450 [0.0407, 0.0492], worst member 0.0012 [0.0004, 0.0022].

### Diverse groups

| Strategy | Worst-off satisfaction | Satisfaction gap | Longest trip (km) | Travel gap (km) | Member NDCG@10, mean | Member NDCG@10, worst |
|---|---|---|---|---|---|---|
| Average | 0.579 [0.575, 0.582] | 0.341 [0.337, 0.345] | 18.37 [18.00, 18.74] | 15.21 [14.85, 15.58] | 0.0159 [0.0137, 0.0183] | 0.0000 [0.0000, 0.0000] |
| Least misery | 0.627 [0.624, 0.630] | 0.139 [0.137, 0.142] | 16.08 [15.82, 16.37] | 11.28 [11.03, 11.53] | 0.0066 [0.0051, 0.0083] | 0.0000 [0.0000, 0.0000] |
| Average without misery | 0.579 [0.575, 0.582] | 0.341 [0.337, 0.345] | 18.37 [17.99, 18.73] | 15.21 [14.86, 15.55] | 0.0159 [0.0136, 0.0184] | 0.0000 [0.0000, 0.0000] |

Each member's own list, for reference: NDCG@10 mean 0.0460 [0.0430, 0.0491], worst member 0.0009 [0.0004, 0.0015].


## LambdaRank

### Similar groups

| Strategy | Worst-off satisfaction | Satisfaction gap | Longest trip (km) | Travel gap (km) | Member NDCG@10, mean | Member NDCG@10, worst |
|---|---|---|---|---|---|---|
| Average | 0.598 [0.592, 0.605] | 0.332 [0.325, 0.338] | 12.93 [12.51, 13.33] | 10.17 [9.81, 10.57] | 0.0227 [0.0197, 0.0255] | 0.0000 [0.0000, 0.0000] |
| Least misery | 0.638 [0.632, 0.644] | 0.177 [0.173, 0.181] | 11.96 [11.58, 12.34] | 8.70 [8.37, 9.03] | 0.0149 [0.0125, 0.0173] | 0.0000 [0.0000, 0.0000] |
| Average without misery | 0.599 [0.592, 0.606] | 0.331 [0.324, 0.338] | 12.92 [12.49, 13.35] | 10.17 [9.78, 10.56] | 0.0226 [0.0198, 0.0255] | 0.0000 [0.0000, 0.0000] |

Each member's own list, for reference: NDCG@10 mean 0.0518 [0.0472, 0.0568], worst member 0.0011 [0.0003, 0.0021].

### Random groups

| Strategy | Worst-off satisfaction | Satisfaction gap | Longest trip (km) | Travel gap (km) | Member NDCG@10, mean | Member NDCG@10, worst |
|---|---|---|---|---|---|---|
| Average | 0.539 [0.534, 0.543] | 0.383 [0.378, 0.388] | 14.41 [14.03, 14.80] | 11.25 [10.89, 11.60] | 0.0175 [0.0151, 0.0201] | 0.0000 [0.0000, 0.0000] |
| Least misery | 0.587 [0.583, 0.591] | 0.179 [0.175, 0.182] | 13.10 [12.76, 13.47] | 9.35 [9.05, 9.66] | 0.0080 [0.0064, 0.0096] | 0.0000 [0.0000, 0.0000] |
| Average without misery | 0.540 [0.536, 0.544] | 0.381 [0.376, 0.386] | 14.38 [13.99, 14.78] | 11.21 [10.86, 11.57] | 0.0171 [0.0148, 0.0196] | 0.0000 [0.0000, 0.0000] |

Each member's own list, for reference: NDCG@10 mean 0.0483 [0.0440, 0.0530], worst member 0.0009 [0.0003, 0.0016].

### Diverse groups

| Strategy | Worst-off satisfaction | Satisfaction gap | Longest trip (km) | Travel gap (km) | Member NDCG@10, mean | Member NDCG@10, worst |
|---|---|---|---|---|---|---|
| Average | 0.539 [0.536, 0.542] | 0.377 [0.373, 0.381] | 19.22 [18.81, 19.62] | 15.65 [15.26, 16.03] | 0.0143 [0.0122, 0.0166] | 0.0000 [0.0000, 0.0000] |
| Least misery | 0.586 [0.583, 0.589] | 0.161 [0.158, 0.165] | 17.73 [17.36, 18.09] | 12.78 [12.47, 13.11] | 0.0091 [0.0077, 0.0107] | 0.0000 [0.0000, 0.0000] |
| Average without misery | 0.540 [0.536, 0.543] | 0.375 [0.371, 0.379] | 19.21 [18.82, 19.60] | 15.63 [15.27, 16.01] | 0.0141 [0.0120, 0.0164] | 0.0000 [0.0000, 0.0000] |

Each member's own list, for reference: NDCG@10 mean 0.0421 [0.0390, 0.0455], worst member 0.0003 [0.0001, 0.0006].

## Paired differences from "Average" (same groups)

| Model | Kind | Strategy | Worst-off member's satisfaction | Satisfaction gap (best minus worst member) | Longest trip (km) | Travel gap (km) | Member NDCG@10 on the group list, mean | Member NDCG@10, worst member |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | similar | Least misery | +0.0186 [+0.0177, +0.0195] | -0.0608 [-0.0633, -0.0582] | -0.1405 [-0.2499, -0.0374] | -0.9227 [-1.0472, -0.8056] | -0.0061 [-0.0083, -0.0040] | -0.0001 [-0.0002, +0.0000] |
| Logistic regression | similar | Average without misery | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| LightGBM | similar | Least misery | +0.0359 [+0.0346, +0.0372] | -0.1321 [-0.1366, -0.1278] | -0.9257 [-1.0494, -0.7997] | -1.5114 [-1.6659, -1.3584] | -0.0065 [-0.0088, -0.0044] | +0.0000 [+0.0000, +0.0000] |
| LightGBM | similar | Average without misery | +0.0000 [+0.0000, +0.0000] | -0.0000 [-0.0000, +0.0000] | +0.0002 [+0.0000, +0.0006] | -0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| LambdaRank | similar | Least misery | +0.0397 [+0.0382, +0.0413] | -0.1543 [-0.1593, -0.1494] | -0.9624 [-1.0818, -0.8501] | -1.4713 [-1.6056, -1.3322] | -0.0078 [-0.0104, -0.0054] | +0.0000 [+0.0000, +0.0000] |
| LambdaRank | similar | Average without misery | +0.0005 [+0.0003, +0.0006] | -0.0006 [-0.0009, -0.0004] | -0.0039 [-0.0098, +0.0015] | -0.0024 [-0.0089, +0.0030] | -0.0001 [-0.0003, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| Logistic regression | random | Least misery | +0.0227 [+0.0217, +0.0236] | -0.0732 [-0.0754, -0.0707] | -0.0483 [-0.1491, +0.0523] | -1.0220 [-1.1456, -0.8995] | -0.0088 [-0.0116, -0.0063] | -0.0001 [-0.0002, +0.0000] |
| Logistic regression | random | Average without misery | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| LightGBM | random | Least misery | +0.0441 [+0.0428, +0.0455] | -0.1723 [-0.1766, -0.1679] | -1.0743 [-1.2154, -0.9393] | -1.8716 [-2.0356, -1.7116] | -0.0082 [-0.0106, -0.0059] | +0.0000 [+0.0000, +0.0000] |
| LightGBM | random | Average without misery | +0.0001 [+0.0000, +0.0002] | -0.0002 [-0.0004, -0.0001] | -0.0041 [-0.0081, -0.0010] | -0.0040 [-0.0087, -0.0007] | -0.0002 [-0.0006, +0.0001] | +0.0000 [+0.0000, +0.0000] |
| LambdaRank | random | Least misery | +0.0484 [+0.0470, +0.0499] | -0.2044 [-0.2095, -0.1997] | -1.3056 [-1.4411, -1.1760] | -1.9010 [-2.0527, -1.7551] | -0.0095 [-0.0120, -0.0071] | +0.0000 [+0.0000, +0.0000] |
| LambdaRank | random | Average without misery | +0.0015 [+0.0012, +0.0019] | -0.0025 [-0.0032, -0.0019] | -0.0258 [-0.0432, -0.0106] | -0.0363 [-0.0538, -0.0206] | -0.0003 [-0.0010, +0.0002] | +0.0000 [+0.0000, +0.0000] |
| Logistic regression | diverse | Least misery | +0.0161 [+0.0154, +0.0168] | -0.0531 [-0.0549, -0.0514] | -0.1499 [-0.2671, -0.0311] | -1.2813 [-1.4309, -1.1269] | -0.0026 [-0.0042, -0.0011] | +0.0000 [+0.0000, +0.0000] |
| Logistic regression | diverse | Average without misery | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| LightGBM | diverse | Least misery | +0.0487 [+0.0473, +0.0500] | -0.2017 [-0.2064, -0.1971] | -2.2874 [-2.4524, -2.1265] | -3.9325 [-4.1690, -3.7085] | -0.0093 [-0.0115, -0.0071] | +0.0000 [+0.0000, +0.0000] |
| LightGBM | diverse | Average without misery | +0.0001 [+0.0000, +0.0001] | -0.0001 [-0.0003, -0.0000] | -0.0034 [-0.0076, +0.0001] | -0.0023 [-0.0051, -0.0002] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| LambdaRank | diverse | Least misery | +0.0469 [+0.0455, +0.0483] | -0.2156 [-0.2208, -0.2108] | -1.4908 [-1.6540, -1.3355] | -2.8636 [-3.0636, -2.6698] | -0.0052 [-0.0072, -0.0032] | +0.0000 [+0.0000, +0.0000] |
| LambdaRank | diverse | Average without misery | +0.0008 [+0.0006, +0.0011] | -0.0015 [-0.0020, -0.0011] | -0.0114 [-0.0249, +0.0010] | -0.0212 [-0.0354, -0.0090] | -0.0002 [-0.0004, -0.0000] | +0.0000 [+0.0000, +0.0000] |

## Feasibility of "average without misery"

| Model | Kind | Threshold | Places passing (mean) | Groups with at least ten | Groups with none |
|---|---|---|---|---|---|
| LambdaRank | diverse | 0.30 | 394.9 | 100.0% | 0.0% |
| LambdaRank | diverse | 0.40 | 292.4 | 100.0% | 0.0% |
| LambdaRank | diverse | 0.50 | 99.8 | 94.2% | 0.8% |
| LambdaRank | random | 0.30 | 402.5 | 100.0% | 0.0% |
| LambdaRank | random | 0.40 | 280.4 | 100.0% | 0.0% |
| LambdaRank | random | 0.50 | 78.7 | 86.9% | 1.7% |
| LambdaRank | similar | 0.30 | 391.7 | 100.0% | 0.0% |
| LambdaRank | similar | 0.40 | 286.3 | 100.0% | 0.0% |
| LambdaRank | similar | 0.50 | 99.7 | 93.1% | 0.9% |
| LightGBM | diverse | 0.30 | 444.3 | 100.0% | 0.0% |
| LightGBM | diverse | 0.40 | 388.5 | 100.0% | 0.0% |
| LightGBM | diverse | 0.50 | 205.5 | 100.0% | 0.0% |
| LightGBM | random | 0.30 | 466.3 | 100.0% | 0.0% |
| LightGBM | random | 0.40 | 411.8 | 100.0% | 0.0% |
| LightGBM | random | 0.50 | 229.2 | 99.5% | 0.1% |
| LightGBM | similar | 0.30 | 445.7 | 100.0% | 0.0% |
| LightGBM | similar | 0.40 | 399.4 | 100.0% | 0.0% |
| LightGBM | similar | 0.50 | 243.2 | 99.9% | 0.0% |
| Logistic regression | diverse | 0.30 | 490.4 | 100.0% | 0.0% |
| Logistic regression | diverse | 0.40 | 482.0 | 100.0% | 0.0% |
| Logistic regression | diverse | 0.50 | 467.1 | 100.0% | 0.0% |
| Logistic regression | random | 0.30 | 517.1 | 100.0% | 0.0% |
| Logistic regression | random | 0.40 | 506.8 | 100.0% | 0.0% |
| Logistic regression | random | 0.50 | 489.3 | 100.0% | 0.0% |
| Logistic regression | similar | 0.30 | 491.3 | 100.0% | 0.0% |
| Logistic regression | similar | 0.40 | 482.0 | 100.0% | 0.0% |
| Logistic regression | similar | 0.50 | 465.3 | 100.0% | 0.0% |

## Sensitivity: satisfaction as rank percentile

| Model | Kind | Strategy | Worst-off satisfaction | Member NDCG@10, mean | Member NDCG@10, worst | Groups with at least ten |
|---|---|---|---|---|---|---|
| Logistic regression | similar | Average | 0.785 | 0.0174 | 0.0000 | 100.0% |
| Logistic regression | similar | Least misery | 0.807 | 0.0142 | 0.0000 | 100.0% |
| Logistic regression | similar | Average without misery | 0.785 | 0.0174 | 0.0000 | 100.0% |
| LightGBM | similar | Average | 0.773 | 0.0172 | 0.0000 | 100.0% |
| LightGBM | similar | Least misery | 0.798 | 0.0140 | 0.0000 | 100.0% |
| LightGBM | similar | Average without misery | 0.773 | 0.0172 | 0.0000 | 100.0% |
| LambdaRank | similar | Average | 0.764 | 0.0183 | 0.0000 | 100.0% |
| LambdaRank | similar | Least misery | 0.791 | 0.0147 | 0.0001 | 100.0% |
| LambdaRank | similar | Average without misery | 0.764 | 0.0183 | 0.0000 | 100.0% |
| Logistic regression | random | Average | 0.759 | 0.0114 | 0.0000 | 100.0% |
| Logistic regression | random | Least misery | 0.783 | 0.0091 | 0.0000 | 100.0% |
| Logistic regression | random | Average without misery | 0.759 | 0.0114 | 0.0000 | 100.0% |
| LightGBM | random | Average | 0.726 | 0.0101 | 0.0000 | 100.0% |
| LightGBM | random | Least misery | 0.759 | 0.0074 | 0.0000 | 100.0% |
| LightGBM | random | Average without misery | 0.726 | 0.0101 | 0.0000 | 100.0% |
| LambdaRank | random | Average | 0.723 | 0.0108 | 0.0000 | 100.0% |
| LambdaRank | random | Least misery | 0.756 | 0.0072 | 0.0000 | 100.0% |
| LambdaRank | random | Average without misery | 0.724 | 0.0108 | 0.0000 | 100.0% |
| Logistic regression | diverse | Average | 0.792 | 0.0080 | 0.0000 | 100.0% |
| Logistic regression | diverse | Least misery | 0.810 | 0.0069 | 0.0000 | 100.0% |
| Logistic regression | diverse | Average without misery | 0.792 | 0.0080 | 0.0000 | 100.0% |
| LightGBM | diverse | Average | 0.722 | 0.0105 | 0.0000 | 100.0% |
| LightGBM | diverse | Least misery | 0.756 | 0.0060 | 0.0000 | 100.0% |
| LightGBM | diverse | Average without misery | 0.722 | 0.0105 | 0.0000 | 100.0% |
| LambdaRank | diverse | Average | 0.743 | 0.0096 | 0.0000 | 100.0% |
| LambdaRank | diverse | Least misery | 0.770 | 0.0080 | 0.0000 | 100.0% |
| LambdaRank | diverse | Average without misery | 0.743 | 0.0096 | 0.0000 | 100.0% |

Written by `src/33_rq2_groups.py`.
