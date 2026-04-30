# EstateMind Score Summary (March 29, 2026)

## Migration note (April 18, 2026)

- A Django unified backend scaffold was added under `django_backend/` to begin migration from FastAPI + Streamlit toward React + Django.
- A React frontend scaffold was added under `frontend_react/` with a Listings -> Add -> Valuation flow.
- The valuation view now focuses on valuation + explainability results (XAI panel) and no longer includes the generic service-console controls.
- The XAI panel now surfaces full explainability outputs from serving responses, including confidence interval/reasons, full SHAP/feature-impact entries, detailed comparables, and market-context diagnostics.
- Feature Impact and SHAP sections now include simple signed bar visualizations in the React valuation UI for quicker interpretation.
- A dashboard-style XAI illustration summary was added with visual meters for confidence, interval spread, completeness, and evidence quality.
- The valuation UI now includes summary report generation/download and keeps raw JSON payload blocks hidden until the user explicitly clicks Show Raw JSON.
- New compatibility endpoint in the Django scaffold: `POST /listings/add-from-valuation`, which runs valuation and persists listing records.
- Model scores and valuation formulas below remain unchanged by this scaffold-only migration step.

This document consolidates the latest measured model scores and the valuation formulas currently used by the serving pipeline.

## 1) Valuation formulas used in `ValuationService`

### 1.1 Heuristic estimate (used when model bundles are unavailable)

Source: `src/inference/valuation_service.py` (`_heuristic_estimate`).

- Base market price per m2:
  - `base_per_m2 = max(market_context.avg_m2, 1)`
- Type factor:
  - Terrain: 0.82
  - Maison: 1.00
  - Appartement: 1.08
- Condition factor:
  - New: 1.16
  - Excellent: 1.10
  - Good: 1.00
  - Fair: 0.92
  - Needs Renovation: 0.80
- Amenity factor:
  - `1 + 0.06*sea_view + 0.04*has_pool + 0.03*has_garden + 0.02*has_parking + 0.015*elevator`
- Room factor:
  - `1 + min(0.01*bedrooms + 0.008*bathrooms, 0.06)`

Computed values:

- `estimated_per_m2 = round(base_per_m2 * type_factor * condition_factor * amenity_factor * room_factor)`
- `estimated_price = round(estimated_per_m2 * surface_m2)`

### 1.2 Multimodal refinement (model + CV + sentiment)

Source: `src/inference/valuation_service.py` (`_refine_with_multimodal_signals`).

- CV signal:
  - `cv_signal = price_band_effect * price_band_confidence * 0.04`
- Sentiment signal:
  - `sentiment_signal = (description_sentiment - 0.5) * 0.03`
- Refinement factor:
  - `factor = clamp(1 + cv_signal + sentiment_signal, 0.92, 1.08)`

Refined outputs:

- `refined_price = round(estimated_price * factor)`
- `refined_price_per_m2 = round(price_per_m2 * factor)`

### 1.3 Confidence and prediction interval

Source: `src/explainability/confidence_service.py` (`estimate`), invoked by `ValuationService.estimate`.

Core blend:

- `combined = 0.35*base_quality + 0.25*input_completeness + 0.15*image_coverage + 0.10*text_score + 0.15*comparable_score`
- OOD penalty:
  - subtract `min(0.1, 0.03 * len(ood_flags))`
- Final confidence:
  - `confidence = clamp(round(combined * 100), 30, 92)`

Uncertainty interval:

- `uncertainty_ratio = clamp(0.34 - confidence/500, 0.08, 0.30)`
- If fallback uncertainty mode: `uncertainty_ratio >= 0.14`
- Bounds:
  - `lower_bound = round(estimated_price * (1 - uncertainty_ratio))`
  - `upper_bound = round(estimated_price * (1 + uncertainty_ratio))`

### 1.4 Upload-time CLIP consistency warning

Source: `src/api.py` (`_build_upload_consistency_warnings`, endpoint `/estimate-upload`).

- Uses image classification rows produced by `ImageTypeClassifierService.classify_many`.
- Consistency warning is evaluated only when CLIP fallback is active for uploaded images.
- If CLIP-inferred dominant property type conflicts with user-selected property type, the response warning includes:
  - `clip_property_type_mismatch:selected=<selected>,inferred=<inferred>`
- This is non-blocking: estimation still runs, warning is surfaced in response metadata.

## 2) Sentiment runtime model selection and measured scores

Primary/fallback decision is now performance-aware in `src/nlp/sentiment_service.py`:

- Reads `artifacts/reports/nlp_sentiment/best_sentiment_model_report.json`
- Uses selected best model as primary order
- Falls back to the other runtime if primary cannot run

Current benchmark evidence:

- Best selected model: `tfidf_char`
- Grouped CV macro-F1 mean: `0.8366`
- Grouped CV macro-F1 std: `0.1778`

Held-out test comparison:

- `tfidf_char_test_metrics.json`:
  - accuracy: `0.6957`
  - macro-F1: `0.6954`
  - weighted-F1: `0.6217`
- `distilbert_test_metrics.json`:
  - accuracy: `0.0000`
  - macro-F1: `0.0000`

Decision:

- Main sentiment runtime: `tfidf_primary`
- Fallback runtime: `transformer_fallback`

## 3) Retrained valuation model scores (transaction-aware features)

Retraining artifact script:

- `scripts/retrain_estateprocessor_models.py`

Output files:

- `artifacts/reports/ml_reports/training_estateprocessor_results.csv`
- `artifacts/reports/ml_reports/training_estateprocessor_manifest.json`

Quality gates used:

- `test_r2 >= 0.60`
- `overfit_gap <= 0.20` (train_r2 - test_r2)

Latest scores:

- By-type Appartement:
  - train_r2: `0.8259`
  - test_r2: `0.6394`
  - test_rmse: `135,589.21`
  - test_mae: `56,207.40`
  - overfit_gap: `0.1865`
  - status: `accepted`
- By-type Maison:
  - train_r2: `0.8641`
  - test_r2: `0.7182`
  - test_rmse: `249,761.54`
  - test_mae: `154,919.76`
  - overfit_gap: `0.1459`
  - status: `accepted`
- By-type Terrain:
  - approach used: `terrain_data_enrichment`
  - train_r2: `0.9641`
  - test_r2: `0.9408`
  - test_rmse: `98,627.78`
  - test_mae: `39,446.56`
  - overfit_gap: `0.0234`
  - status: `accepted`
- Global (accepted types only: Appartement + Maison):
  - train_r2: `0.7673`
  - test_r2: `0.7213`
  - test_rmse: `202,708.05`
  - test_mae: `93,749.80`
  - overfit_gap: `0.0461`
  - status: `accepted`

Terrain challenge remediation experiments (both executed in retraining):

- Approach A: `terrain_data_enrichment`
  - method: governorate-wise quantile clipping + synthetic densification for low-count governorates
  - rows_train: `23,584`
  - rows_test: `5,896`
  - train_r2: `0.9641`
  - test_r2: `0.9408`
  - overfit_gap: `0.0234`
  - outcome: `selected and accepted`
- Approach B: `terrain_specialized`
  - method: terrain-only target strategy (predict `price_per_m2`, then recover total price)
  - rows_train: `556`
  - rows_test: `140`
  - train_r2: `0.7221`
  - test_r2: `-0.0711`
  - overfit_gap: `0.7933`
  - outcome: `not selected`

Interpretation:

- The retrained accepted models satisfy the requested performance constraints for all supported scopes, including Terrain.
- Current accepted model count from retraining manifest: `4` (Appartement, Maison, Terrain, Global).

## 4) Fallback tabular model sample benchmark

Script: `tests/evaluate_fallback_model_sample.py`

Sample benchmark output (2,500 rows):

- Coverage: `100%`
- R2: `0.0478`
- MAE: `1,424,720.74`
- RMSE: `11,339,327.65`
- Median absolute error: `78,834.50`

Conclusion:

- Current fallback tabular artifacts are available for continuity, but quality is substantially lower than the newly accepted main models.
- Serving should prioritize accepted estateprocessor models, then fallback/heuristic modes only when needed.

## 5) Formula rationale (why these factors and bounds were chosen)

This section explains the design intent of the formulas currently used in serving.

## 6) Image-driven input validation and guidance

The backend now uses uploaded property images to support structured guidance before finalizing a valuation:

- When `property_type` is omitted, the vision pipeline can auto-detect the most likely property type and return a confirmation prompt.
- When `property_type` is manually selected but conflicts with image evidence, the pipeline returns a structured mismatch prompt so the UI can ask the user to correct the selection.
- Amenities visible in the image can be suggested automatically when not manually selected.
- Amenities manually selected but not visible in the image are surfaced as review prompts, with an option to upload more supporting photos.
- If the effective property type is `Terrain`, all amenities are force-disabled in backend serving and a guidance warning is emitted.
- Image-driven guidance is activated as soon as at least one input image is present.

The response now exposes this through:

- `vision_guidance`
- `vision_requires_confirmation`

### 5.1 Heuristic estimate rationale

- `base_per_m2 = max(market_context.avg_m2, 1)`:
  - Uses local market anchor instead of a fixed national constant.
  - `max(..., 1)` guarantees positivity and prevents invalid/zero anchors.
- Type factor (`Terrain: 0.82`, `Maison: 1.00`, `Appartement: 1.08`):
  - Encodes structural market differences between land, houses, and apartments.
  - Multiplicative scaling preserves proportional behavior with local market level.
- Condition factor (`New`, `Excellent`, `Good`, `Fair`, `Needs Renovation`):
  - Monotonic premium/discount based on expected market willingness-to-pay.
  - Chosen to be material but not large enough to dominate location/size effects.
- Amenity factor:
  - `1 + 0.06*sea_view + 0.04*has_pool + 0.03*has_garden + 0.02*has_parking + 0.015*elevator`
  - Additive increments keep contributions transparent and independently tunable.
  - Small coefficients avoid unstable jumps from binary flags.
- Room factor:
  - `1 + min(0.01*bedrooms + 0.008*bathrooms, 0.06)`
  - Captures utility uplift from room count with a hard cap to prevent over-amplification.
- Final composition:
  - Per-m2 is computed first, then scaled by surface area.
  - This matches standard market decomposition (`price = area * price_per_m2`).

### 5.2 Multimodal refinement rationale

- CV signal:
  - `cv_signal = price_band_effect * price_band_confidence * 0.04`
  - Confidence weighting ensures low-certainty image signals have limited effect.
  - `0.04` keeps CV influence supportive instead of dominant.
- Sentiment signal:
  - `sentiment_signal = (description_sentiment - 0.5) * 0.03`
  - Uses `0.5` as neutral midpoint; positive text nudges up and negative text nudges down.
  - `0.03` limits textual influence to a small correction.
- Refinement factor and clamp:
  - `factor = clamp(1 + cv_signal + sentiment_signal, 0.92, 1.08)`
  - Starts from neutral factor `1.0` and applies bounded multimodal deltas.
  - Clamp prevents noisy multimodal outputs from creating unrealistic valuation swings.
  - Operationally this caps refinement to approximately +/-8%.
- Refined outputs:
  - Applies same factor to total price and per-m2 to keep internal consistency.

### 5.3 Confidence and interval rationale

- Weighted confidence blend:
  - `combined = 0.35*base_quality + 0.25*input_completeness + 0.15*image_coverage + 0.10*text_score + 0.15*comparable_score`
  - Higher weights on model quality and completeness because they are most stable reliability drivers.
  - Image/text/comparables contribute secondary evidence.
- OOD penalty:
  - subtract `min(0.1, 0.03 * len(ood_flags))`
  - Penalizes out-of-distribution risk while capping penalty to avoid collapse.
- Final confidence:
  - `confidence = clamp(round(combined * 100), 30, 92)`
  - Converts to intuitive percent scale with lower/upper caps for robustness.
- Interval width:
  - `uncertainty_ratio = clamp(0.34 - confidence/500, 0.08, 0.30)`
  - Inverse relation: higher confidence yields narrower interval.
  - Bounds prevent unrealistically narrow or excessively wide intervals.
- Price bounds:
  - `lower_bound = round(estimated_price * (1 - uncertainty_ratio))`
  - `upper_bound = round(estimated_price * (1 + uncertainty_ratio))`
  - Symmetric around estimate for consistency and interpretability.

### 5.4 Explainability fallback scaling rationale

Source: `src/explainability/shap_service.py` (`_fallback`).

- Fallback feature contributions are now scaled as fractions of the current estimate instead of fixed absolute constants.
- This keeps SHAP fallback decomposition numerically aligned with low, mid, and high price regimes.
- Baseline is computed directly as:
  - `baseline = estimated_price - sum(contributions)`
- Result: reduced mismatch risk between displayed estimate and fallback SHAP final total.
