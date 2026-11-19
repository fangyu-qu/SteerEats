# src/explain

Explanation pipeline (Yila). See `docs/PLAN.md` §2.3.

- candidate generation (~200 per user)
- stage log S0 model → S1 distance → S2 popularity → S3 diversity (MMR) on a common candidate set
- Shapley-over-stages attribution of rank change
- SHAP for logistic regression (checked against the closed form) and LightGBM (TreeExplainer)
- replay test: reconstructed final ranks must match with zero residual
