# SteerEats

Course project for DS-GA 3001 Visualization for Machine Learning (NYU, Fall 2026, Prof. Claudio Silva).
Team 4: Fangyu Qu (data, baselines, rating history, LLM themes) and Yila Cao (logistic regression,
LightGBM, explanation pipeline, rank-change / SHAP / control / diversity / map views).
Course site: https://ctsilva.github.io/2026-VisML-CDS/

The working plan is `docs/PLAN.md`. Read it before starting a task.

## Conventions

- Never fabricate or estimate a result. Every number in the README, report or UI comes from a script in
  this repository run on the real sample. If data is missing, stop and say so.
- Raw and processed data live in `data/` and are git-ignored. Per-user CSVs and private case files too.
- Python: `.venv` (Python 3.13). Scripts in `src/` are numbered in pipeline order.
- Predictive features use only observations before the snapshot cutoff.
- SHAP explains raw model scores, not final ranks. Stage-wise rank attribution must replay to zero residual.
- Front end (`web/`): Vite + D3 v7, all steering computed client-side from precomputed case JSON.
  Colors and fonts only through tokens; model / control / diversity effects keep the same three hues
  in every view.
- Work happens on the `Yila` branch; merge to `main` through pull requests.
- Disclose AI assistance in the report.
