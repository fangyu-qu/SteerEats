# Course notes applied to SteerEats

Digest of the DS-GA 3001 (Fall 2026) slides and labs for weeks 2, 3, 5, 7 and 11, written so
each rule points at the SteerEats component it constrains. Source pages are linked per section.
Where a page could not be read in full, that is stated.

---

## Week 2: Perception and color

Sources: [perception](https://ctsilva.github.io/2026-VisML-CDS/slides/perception.html),
[color](https://ctsilva.github.io/2026-VisML-CDS/slides/color.html),
[lab](https://ctsilva.github.io/2026-VisML-CDS/labs/perception-color/lab.html).

Rules from the slides:

- **Channel ranking** (Cleveland & McGill 1984; Heer & Bostock 2010; Munzner Fig. 5.6):
  position on a common scale > position on non-aligned scales > length, direction, angle >
  area > volume, curvature > shading, saturation. Put the most important quantity on position.
- **Discriminability:** about 3 line widths; no more than 5–6 hues or shapes. When categories
  exceed the channel, aggregate. The professor marks this warning as the important one.
- **Pop-out** needs a feature the distractors lack, in one channel; conjunctions are serial.
- **Gestalt strength:** enclosure > connection > similarity, proximity. "Color assignment is a
  claim": two things sharing a hue read as related.
- **Common baseline** is the only layout that answers a position comparison.
- **Two jobs for color:** quantify (vary luminance, fix hue) and label (vary hue, fix luminance).
  Hue is an identity channel, not a magnitude channel. "If the number matters, do not put it in a
  color."
- **Design in a perceptual space** (CIELAB, HCL; OKLCH is the modern equivalent); CSS HSL is not
  perceptual. Rainbow fails uniformity. Single-hue ramps give about 6–8 discriminable steps.
- **Diverging scales only when the midpoint means something.** Use blue–orange, not red–green.
- **No-data** is a neutral grey outside the ramp, never the bottom of the scale.
- **Color-vision deficiency:** about 8–10% of men. Never encode a critical distinction in
  red vs green alone; keep a luminance difference; run every palette through a simulator.
- **Test on the projector.** Lab adds WCAG contrast: text 4.5:1, large text and graphics 3:1.

Applied to SteerEats:

| Component | Decision |
|---|---|
| Rank flow (V2) | Rank on the y position; one accent hue for the focused restaurant, grey for others; entered/exited marked by shape and fill, not by a second hue |
| Score ledger (V3) | Bar length on a common zero baseline carries the SHAP value; sign by blue–orange; the three effect families (model / control / diversity) by grouping and labels first, hue second |
| Map (V4) | Single-hue sequential ramp for neighborhood reach; grey hatch for no data and unreachable |
| Tokens | Defined in OKLCH; every palette checked for CVD and contrast before use |
| Cuisine | Never one hue per cuisine (25 groups); show the top 5 and group the rest |

---

## Week 3: Model assessment

Source: [model assessment](https://ctsilva.github.io/2026-VisML-CDS/slides/model-assessment.html).
About 40% read verbatim; the calibration section came from a summary. Platt scaling and isotonic
regression were not found in what was read.

Rules from the slides:

- A single number cannot tell which failure you have; the majority-class guess is the baseline.
- A scoring model is a family of classifiers; "the fault is the threshold, not the model."
- **Ranking quality and probability quality are different** (Fawcett 2006, naive Bayes example).
  Raw scores from different model families are not comparable, which motivates calibration.
- ROC hides precision under imbalance; PR curves use average precision and state the prevalence
  baseline (Davis & Goadrich 2006; Saito & Rehmsmeier 2015).
- **Calibration:** reliability diagram (bins of predicted probability vs observed frequency);
  ECE, MCE, Brier, log-loss. ECE depends on binning. **Never report ECE alone:** a constant
  base-rate predictor has ECE 0. Pair it with a discrimination score or a proper score.
- Logistic regression trained with log-loss is usually near the diagonal; boosted trees often are not.
- Systems: Squares (Ren et al. 2016), Neo (Görtler et al. 2022), confusion wheel (Alsallakh et al.
  2014), EnsembleMatrix (2009). Design ideas: keep instances reachable from aggregates; answer the
  user's question directly; the visualization can be the control surface.

Applied to SteerEats:

- Report Recall@10 / NDCG@10 with bootstrap CIs, plus coverage, diversity and popularity, so
  no single number decides.
- Platt scaling is monotone per user, so it does not change Recall/NDCG; it matters only where
  probabilities are compared across users or models (score normalization before steering, group
  aggregation). State this in the report.
- Calibration panel (V7): reliability diagram with bin counts shown, Brier and ECE together,
  popularity included as the "calibrated but useless" reference.
- EnsembleMatrix's idea (visualization as control surface) is the lineage for the steering console.

---

## Week 5: Black-box interpretation

Sources: [slides](https://ctsilva.github.io/2026-VisML-CDS/slides/black-box-models.html),
[lab](https://ctsilva.github.io/2026-VisML-CDS/labs/black-box-explainability/lab.html). The lab page
truncates after its LIME section; its SHAP part was not readable.

Rules from the slides:

- **PDP** (Friedman 2001): averages hide heterogeneous effects; assumes feature independence.
- **LIME** (Ribeiro et al. 2016): local surrogate; kernel width is "the single most consequential
  parameter"; unstable across seeds by design.
- **Shapley values:** efficiency, symmetry, dummy, additivity. Taxi example A 5.5, B 15.5, C 30.
- **SHAP** (Lundberg & Lee 2017): absent features are marginalized, which assumes independence.
  TreeSHAP (Lundberg et al. 2020) is exact for trees. Background data choice matters.
- **For linear models, SHAP_j = β_j (x_j − E[x_j])** under independence.
- Plots: force (efficiency made visible), beeswarm (spread = importance, color = feature value),
  waterfall, dependence (color = interacting feature).
- **Pitfalls:** correlated features create unrealistic samples ("true to the model" vs "true to
  the data", Aas et al. 2021); `tree_path_dependent` / `correlation_dependent` change the value
  function; Slack et al. 2020 show explanations can be manipulated; Kumar et al. 2020 list
  Shapley problems; Miller 2019 says good explanations are contrastive.
- "SHAP values measure contribution to deviation from average, NOT the effect of removing
  features from training."

Applied to SteerEats:

- LR: verify `LinearExplainer` against β_j (x_j − E[x_j]) in a test (the course's own check failed
  as printed, so doing it right is a visible improvement).
- LightGBM: `TreeExplainer`, interventional with the same background sample as LR.
- Correlated features to watch: `cf_all` vs `cf_liked`, popularity vs recent popularity, the two
  smoothed ratings. Report their correlations next to the beeswarm.
- Compare models by rank order and sign of attributions on the same candidates, not by raw size.
- Miller's point is our motivation for contrastive before-and-after explanations: SHAP alone
  answers "why this score", not "why this instead of that".
- Our Shapley-over-stages attribution reuses the taxi-example logic on the three controls.

---

## Week 7: Dimensionality reduction (stretch view S1)

Sources: [slides](https://ctsilva.github.io/2026-VisML-CDS/slides/dimensionality-reduction.html),
[lab](https://ctsilva.github.io/2026-VisML-CDS/labs/dimensionality-reduction/lab.html).

Rules from the slides:

- t-SNE: perplexity must be below N; "run until the picture stops changing"; cluster sizes and
  inter-cluster distances mean nothing; random noise does not always look random.
- UMAP: `n_neighbors` (local vs global), `min_dist` (0–0.1 for clusters, 0.5+ to see points);
  "stable is not the same as right"; the global-structure advantage is mostly initialization
  (Kobak & Linderman 2021).
- **Trustworthiness and continuity** (Venna & Kaski), k = 15; local only.
- **Checklist before trusting a projection:** vary parameters and at least two seeds; compare
  with PCA and a same-size noise baseline; score trustworthiness, continuity and k-NN accuracy;
  report method, parameters, seed and preprocessing; use the projection to ask questions, not as
  evidence.
- Interactive DR: "what can the user change, and what stays fixed?"

Applied to SteerEats (if S1 is built): UMAP of item factors with a PCA panel beside it, a
parameter sweep figure in the appendix, trustworthiness/continuity/k-NN (cuisine label) reported,
and linked brushing to the map instead of reading distances in the projection.

---

## Week 11: Time series and temporal data

Sources: [slides](https://ctsilva.github.io/2026-VisML-CDS/slides/time-series.html),
[lab](https://ctsilva.github.io/2026-VisML-CDS/labs/time-series/lab.html). The last fifth of the
slides (drift monitoring and anomaly detection) came from a summary.

Rules from the slides:

- Distinguish **event time from record time**. A review timestamp is when it was posted.
- Counting events per interval turns event data into measurement data; keep the denominator visible.
- "Choose the granularity from the question: trends want coarse bins, anomalies want fine ones."
- Missing periods are not zero; incomplete last periods must be marked
  ("mark the last period with complete labels").
- Spaghetti → highlight with faded context, or small multiples on a common scale.
- Random CV on time-ordered data is over-optimistic; use rolling-origin backtesting and report
  error per fold.
- **Drift:** data drift changes P(x); concept drift changes P(y|x). Shapes: sudden, incremental,
  gradual, recurring (Gama et al. 2014). Monitoring views share a time axis, mark reference
  periods, thresholds and deployments, and slice by segment.
- Lab rule: **"any group whose data has timestamps should say in the final report how the data
  was split in time."** The lab also flags local-time artifacts (daylight-saving duplicates).

Applied to SteerEats:

- State the global time split and its dates in the report, with a timeline figure.
- Test spans the COVID regime and the collection's tail: report Recall/NDCG per test month or
  quarter, and mark months with incomplete collection.
- Add a pre-COVID evaluation window as a drift comparison (data drift in which restaurants are
  active; possible concept drift in what users choose).
- Convert timestamps to America/New_York before any hour or weekday feature.
- Rating history (V8): rolling mean with an uncertainty band and visible review counts; missing
  months as gaps, not zeros.
