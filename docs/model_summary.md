# EstateMind Model Summary

Generated on: 2026-04-21

## 1. Executive Overview

EstateMind is a Tunisia-focused real-estate valuation system that combines:
- Structured/tabular property inputs
- Image understanding (property type and amenity evidence)
- Listing-description NLP and sentiment signals
- Comparable listing retrieval
- Confidence and uncertainty estimation
- Explainability outputs (feature impact and SHAP-like contribution views)

The production serving path is model-first (accepted CatBoost bundles), with graceful fallback to tabular fallback and then heuristic logic when required.

## 2. End-to-End Prediction Pipeline

### 2.1 Input Layer

Main inputs collected from API/UI:
- Property characteristics: type, transaction type, area, rooms, condition, amenities, location
- Optional free-text listing description
- Optional uploaded images

Primary entrypoint:
- src/api.py

### 2.2 Request Mapping Layer

Responsibility:
- Normalize client payloads into internal schema expected by inference services.

Main module:
- src/inference/request_mapper.py

### 2.3 Vision Processing Layer

Responsibilities:
- Image classification for property evidence
- Image quality and coverage scoring
- Aggregation of image-level predictions into listing-level guidance
- Consistency checks between user-provided fields and visual evidence

Main modules:
- src/vision/type_classifier.py
- src/vision/image_quality.py
- src/vision/feature_aggregation.py

Model behavior:
- Primary image classifier: ResNet family runtime
- Fallback semantic image model: CLIP (openai/clip-vit-base-patch32)

Serving behavior details:
- If property_type is missing, vision may auto-suggest a type.
- If user-selected property_type conflicts with inferred evidence, a warning is surfaced.
- If effective type is Terrain, non-applicable amenities are disabled and guidance is emitted.

### 2.4 NLP Processing Layer

Responsibilities:
- Description quality features
- Sentiment scoring
- Location sentiment priors

Main modules:
- src/nlp/description_analysis.py
- src/nlp/sentiment_service.py
- src/nlp/location_sentiment.py

Runtime sentiment strategy:
- Primary: tfidf_char (selected by benchmark)
- Fallback: transformer runtime

### 2.5 Feature Fusion Layer

Responsibility:
- Merge structured, vision, and NLP signals into inference-ready feature vectors.

Main module:
- src/inference/feature_fusion.py

### 2.6 Core Valuation Inference Layer

Responsibilities:
- Resolve active model handles from manifest
- Run estimator prediction
- Apply multimodal refinement
- Degrade safely if bundle/model unavailable

Main modules:
- src/inference/model_registry.py
- src/inference/inference_bundle.py
- src/inference/valuation_service.py
- src/inference/fallback_model.py

### 2.7 Explainability and Uncertainty Layer

Responsibilities:
- Comparable retrieval
- Confidence scoring and uncertainty interval
- SHAP/contribution outputs (with fallback scaling)
- Natural-language explanation

Main modules:
- src/explainability/comparables_service.py
- src/explainability/confidence_service.py
- src/explainability/shap_service.py
- src/explainability/explanation_service.py

### 2.8 Response Assembly Layer

Responsibility:
- Build normalized API/UI response contract.

Main module:
- src/reporting/response_builder.py

## 3. Model Registry and Artifacts

### 3.1 Runtime Registry Contract

Model discovery is manifest-driven.

Primary manifest:
- artifacts/reports/ml_reports/training_estateprocessor_manifest.json

Training results file:
- artifacts/reports/ml_reports/training_estateprocessor_results.csv

Model artifact directory:
- artifacts/models/models_estateprocessor/

### 3.2 Accepted Valuation Models (Latest Reported Snapshot)

From training_estateprocessor_results.csv:

| Scope | Property Type | Approach | train_r2 | test_r2 | test_rmse | test_mae | overfit_gap | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| by_type | Appartement | standard | 0.8259 | 0.6394 | 135589.21 | 56207.40 | 0.1865 | accepted |
| by_type | Maison | standard | 0.8641 | 0.7182 | 249761.54 | 154919.76 | 0.1459 | accepted |
| by_type | Terrain | terrain_data_enrichment | 0.9641 | 0.9408 | 98627.78 | 39446.56 | 0.0234 | accepted |
| global | ALL | - | 0.7673 | 0.7213 | 202708.05 | 93749.80 | 0.0461 | accepted |

Accepted model count:
- 4 models (Appartement, Maison, Terrain, Global)

### 3.3 Terrain Experiment Detail

Two terrain strategies were evaluated:

| Scope | Approach | train_r2 | test_r2 | overfit_gap |
| --- | --- | ---: | ---: | ---: |
| by_type_terrain_experiment | terrain_data_enrichment | 0.9641 | 0.9408 | 0.0234 |
| by_type_terrain_experiment | terrain_specialized | 0.7221 | -0.0711 | 0.7933 |

Interpretation:
- terrain_data_enrichment is clearly superior and was selected.
- terrain_specialized significantly underperformed and is not used in accepted serving models.

### 3.4 Quality Gates Used in Retraining

Configured acceptance criteria:
- test_r2 >= 0.60
- overfit_gap <= 0.20

All accepted serving models in the latest report satisfy these gates.

## 4. NLP Model Performance Summary

Based on the benchmark summary in artifacts/score.md:

Best selected sentiment model:
- tfidf_char

Reported benchmark values:
- Grouped CV macro-F1 mean: 0.8366
- Grouped CV macro-F1 std: 0.1778

Held-out comparison:
- tfidf_char test accuracy: 0.6957
- tfidf_char macro-F1: 0.6954
- tfidf_char weighted-F1: 0.6217
- distilbert fallback test metrics in that snapshot: 0.0000 across key fields

Serving decision:
- Primary sentiment runtime: tfidf_primary
- Fallback runtime: transformer_fallback

## 5. Computer Vision Model Performance Summary

EstateMind uses the vision stack to convert uploaded images into property-type evidence, amenity hints, and image-quality signals.

Primary and fallback vision runtimes:
- Primary image classifier: ResNet family runtime
- Fallback semantic model: CLIP (openai/clip-vit-base-patch32)

Reported performance snapshot from the visual summary used in this project:
- ResNet50 accuracy: 91.8%

Operational interpretation:
- The ResNet-based classifier is the primary computer-vision model for property-type recognition.
- The CLIP fallback is used when semantic feature inference is needed or when the primary image path is unavailable.
- Vision outputs are not used only for classification; they also feed guidance logic for property-type mismatches, amenity suggestions, and image-coverage checks.

Role in inference:
- Vision does not directly replace the main CatBoost valuation model.
- It provides auxiliary signals that improve input completeness, explainability, and confidence scoring.

## 6. Valuation Math and Processing Logic

The serving pipeline applies multiple layers of numeric processing.

### 6.1 Heuristic Base Estimate (Fallback Path)

When model bundles are unavailable, ValuationService applies a rule-based estimate.

Core terms (from artifacts/score.md):
- base_per_m2 = max(market_context.avg_m2, 1)
- type_factor in {Terrain: 0.82, Maison: 1.00, Appartement: 1.08}
- condition_factor in {New: 1.16, Excellent: 1.10, Good: 1.00, Fair: 0.92, Needs Renovation: 0.80}
- amenity factor and room factor to adjust utility premium

Computation:
- estimated_per_m2 = round(base_per_m2 * type_factor * condition_factor * amenity_factor * room_factor)
- estimated_price = round(estimated_per_m2 * surface_m2)

### 6.2 Multimodal Refinement Layer

Post-estimate refinement applies bounded adjustments from sentiment signals:
- sentiment_signal = (description_sentiment - 0.5) * 0.03
- factor = clamp(1.0 + sentiment_signal, 0.96, 1.04)

Note: Vision parameters are passed but currently unused in the active refinement strategy.

Outputs:
- refined_price = max(round(estimated_price * factor), 1)
- refined_price_per_m2 = max(round(price_per_m2 * factor), 1)

### 6.3 Confidence and Uncertainty Layer

Confidence blend:
- combined = 0.35*base_quality + 0.25*input_completeness + 0.15*image_coverage + 0.10*text_score + 0.15*comparable_score
- OOD penalty = min(0.1, 0.03 * len(ood_flags))
- confidence = clamp(round(combined * 100), 30, 92)

Interval:
- uncertainty_ratio = clamp(0.34 - confidence/500, 0.08, 0.30)
- lower_bound = round(estimated_price * (1 - uncertainty_ratio))
- upper_bound = round(estimated_price * (1 + uncertainty_ratio))

## 7. Explainability Outputs

Explainability layer returns:
- Feature impact contributions
- SHAP sequence or fallback contribution view
- Comparable listing evidence
- Confidence level, interval, and uncertainty reasons
- AI narrative explanation

Fallback explainability in SHAP service is estimate-scaled so contribution totals remain numerically aligned with predicted value ranges.

## 8. Fallback and Resilience Strategy

Serving priority order:
1. Accepted by-type/global estateprocessor CatBoost models
2. Fallback tabular model
3. Heuristic estimator

Fallback benchmark snapshot from artifacts/score.md indicates materially lower quality than accepted main models, so fallback remains continuity-first, not quality-first.

## 9. Processing Layers at a Glance

| Layer | Purpose | Primary Components |
| --- | --- | --- |
| Input/API | Receive request payloads and files | src/api.py |
| Mapping | Normalize request schema | src/inference/request_mapper.py |
| Vision | Infer visual evidence and guidance | src/vision/type_classifier.py, src/vision/image_quality.py, src/vision/feature_aggregation.py |
| NLP | Extract text and sentiment features | src/nlp/description_analysis.py, src/nlp/sentiment_service.py, src/nlp/location_sentiment.py |
| Fusion | Merge all modalities | src/inference/feature_fusion.py |
| Prediction | Run model or fallback estimate | src/inference/model_registry.py, src/inference/inference_bundle.py, src/inference/valuation_service.py |
| Explainability | Build comparables/confidence/SHAP/explanations | src/explainability/* |
| Response | Return normalized result contract | src/reporting/response_builder.py |

## 10. Current Strengths and Watch Areas

### Strengths
- Strong Terrain performance after data-enrichment strategy selection.
- Good generalization profile for accepted models (low overfit gaps relative to gates).
- Robust multimodal architecture with graceful fallback.
- Detailed explainability and uncertainty surfaces for UI and API consumers.

### Watch Areas
- By-type Appartement model has the lowest accepted test_r2 in the current accepted set.
- Fallback tabular path is substantially weaker than accepted primary bundles.
- Some NLP runtime snapshots indicate unstable transformer fallback metrics; monitoring is recommended.

## 11. Key Source Files for This Summary

- README.md
- artifacts/score.md
- artifacts/reports/ml_reports/training_estateprocessor_results.csv
- artifacts/reports/ml_reports/training_estateprocessor_manifest.json

## 12. Notebook Model Choice Rationale

### Why this model? notebook_ml.ipynb
The tabular valuation notebook prioritizes CatBoost because the feature space is mixed (numeric plus high-value categorical fields such as city, governorate, and property type), and CatBoost handles these interactions well without brittle manual encoding. The notebook does not assume CatBoost is best by default: it defines challenger pools for both quick and full experimentation and compares CatBoost against RandomForest, ExtraTrees, GradientBoosting, and Ridge through structured search, then keeps the highest-performing by-type and global choices under explicit overfit and quality constraints.

### Why this model? notebook_images.ipynb
The vision notebook uses an empirical backbone-selection strategy rather than architecture bias: it trains and compares ResNet50, EfficientNet-B0, and EfficientNetV2-S on the same prepared loaders and evaluates with class-sensitive metrics (including macro F1) in addition to accuracy. ResNet50 is retained as the best operational model in the reported run, then exported as a fallback artifact and label map for runtime reliability, which makes the choice both performance-driven and deployment-ready.

### Why this model? nlp_model_sentiment.ipynb
The sentiment notebook selects tfidf_char as primary because it is more robust to noisy, multilingual, spelling-variable real-estate text than word-only sparse baselines in this dataset, and that advantage is reflected in grouped cross-validation and held-out metrics. The notebook explicitly benchmarks majority baseline, word TF-IDF, char TF-IDF, and a multilingual transformer pipeline, then persists the best local classical model from grouped CV, so model selection is evidence-based and reproducible.

### Why this model? nlp_model_description.ipynb
The description-price notebook favors a Ridge-over-TF-IDF baseline family to control variance and keep behavior interpretable while testing text signal utility for regression. Character n-gram TF-IDF with Ridge is selected over word n-gram Ridge in the notebook summary because it better captures short-form listing language patterns (orthographic variants, abbreviations, and local spelling noise) while preserving stable linear-model generalization characteristics.

### Why this model? langgraph_real_estate_agent.ipynb
The agent notebook is designed around system robustness, not single-model score maximization: it orchestrates extraction and valuation calls with explicit fallback paths, including heuristic valuation when service calls fail, and surfaces model metadata when the production valuation bundle is available. This choice is appropriate for an end-to-end assistant notebook because it optimizes continuity, observability, and graceful degradation under real-world runtime constraints.

This summary is intended as a technical reference for model governance, serving behavior review, and handoff documentation.
