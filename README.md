# EstateMind

EstateMind is a Tunisia-focused real-estate valuation platform combining tabular modeling, image understanding, NLP sentiment, comparables, and confidence estimation.

It exposes:
- FastAPI backend for programmatic valuation
- Streamlit frontend for interactive estimation and reporting
- Offline training/evaluation scripts for model refresh and benchmarking

## 1. End-to-End Pipeline

1. Input capture
- Structured features: property type, transaction type, size, rooms, condition, amenities, location
- Optional listing description
- Optional property images

2. Request mapping
- `src/inference/request_mapper.py` normalizes API payloads into internal feature format

3. Vision processing
- `src/vision/type_classifier.py` classifies images (ResNet first, CLIP feature inference fallback)
- `src/vision/image_quality.py` evaluates image coverage/quality
- `src/vision/feature_aggregation.py` aggregates image-level signals to listing-level features
- If the user does not manually choose a property type, the backend can auto-detect it from uploaded images and return a structured guidance prompt.
- If the user manually selects a property type that conflicts with the image evidence, the backend returns a confirmation warning so the UI can prompt the user to correct it.
- Amenity selections are also cross-checked against uploaded photos: missing visible amenities are suggested, while selected amenities that are not visible trigger a prompt to review the selection or upload more images.
- If the effective property type is `Terrain`, the backend disables all amenities (`has_pool`, `has_garden`, `has_parking`, `sea_view`, `elevator`) and returns guidance indicating those amenities are not applicable.
- As soon as at least one image is uploaded, image-guidance checks are activated and surfaced in the response.

4. NLP processing
- `src/nlp/description_analysis.py` extracts description quality and key phrases
- `src/nlp/sentiment_service.py` computes sentiment with runtime primary/fallback ordering
- `src/nlp/location_sentiment.py` adds location sentiment priors

5. Feature fusion
- `src/inference/feature_fusion.py` merges structured, vision, and NLP features

6. Model inference
- `src/inference/model_registry.py` loads by-type/global bundles from manifest
- `src/inference/inference_bundle.py` performs estimator-backed predictions
- Fallbacks:
  - `src/inference/fallback_model.py` tabular fallback model
  - heuristic estimator inside `src/inference/valuation_service.py` when bundles are unavailable

7. Explainability and uncertainty
- `src/explainability/comparables_service.py` retrieves comparable listings
- `src/explainability/confidence_service.py` computes confidence and uncertainty interval
- `src/explainability/shap_service.py` returns SHAP or explainability fallback (fallback contributions are estimate-scaled for consistency)
- `src/explainability/explanation_service.py` builds AI explanation text

8. Response build
- `src/reporting/response_builder.py` returns final API response contract

## 2. Architecture

## 2.1 Runtime services
- Backend API: `src/api.py` (mounted via top-level `api.py` entrypoint)
- Frontend: `streamlit_app.py`
- Core orchestration: `src/inference/valuation_service.py`

## 2.2 Data and artifacts
- Preprocessed data: `data/csv/preprocessed/final_listings_preprocessed.csv`
- Trained models: `artifacts/models/models_estateprocessor/`
- Training reports: `artifacts/reports/ml_reports/`
- NLP reports: `artifacts/reports/nlp_sentiment/`

## 2.3 Model loading contract
`ModelRegistry` reads:
- `artifacts/reports/ml_reports/training_estateprocessor_manifest.json`

Each handle stores:
- scope (`by_type` or `global`)
- property type
- model artifact path
- offline metrics

## 3. Models in Use

## 3.1 Valuation models
Primary family:
- CatBoost regression bundles, trained transaction-aware with priors:
  - `bytype__appartement__catboost.joblib`
  - `bytype__maison__catboost.joblib`
  - `bytype__terrain__catboost.joblib`
  - `global__catboost.joblib`

Current accepted model set (latest retrain):
- Appartement by-type: accepted
- Maison by-type: accepted
- Terrain by-type: accepted
- Global fallback: accepted

Quality gates in retraining:
- `test_r2 >= 0.60`
- `overfit_gap <= 0.20`

Terrain remediation now included in retraining script:
- Approach A: data enrichment (selected when superior)
- Approach B: terrain-specialized objective (`price_per_m2` target)

Latest terrain selection behavior:
- The retraining script executes both approaches for Terrain and selects the best candidate by holdout quality.
- In the latest run, `terrain_data_enrichment` was selected and passed gates.

## 3.2 Vision models
- Primary classifier: ResNet image-type model
- CLIP model: `openai/clip-vit-base-patch32` for feature inference fallback and semantic tags

## 3.3 NLP models
- Primary sentiment runtime selected from benchmark report:
  - TF-IDF char model currently primary
- Transformer sentiment runtime used as fallback

Runtime selection details:
- Selection is read from `artifacts/reports/nlp_sentiment/best_sentiment_model_report.json`.
- Service order is performance-aware and degrades to neutral fallback only if both runtimes fail.

## 4. Services and Key Modules

- API contracts and endpoints:
  - `src/api.py`
  - `/estimate-upload` adds non-blocking CLIP property-type consistency warnings when image-inferred type conflicts with selected type
- Inference orchestration:
  - `src/inference/valuation_service.py`
- Model discovery/loading:
  - `src/inference/model_registry.py`
- Training outputs consumed at runtime:
  - `artifacts/reports/ml_reports/training_estateprocessor_manifest.json`
- Dynamic handle resolution:
  - by-type first, then global fallback when allowed
- Confidence/uncertainty:
  - `src/explainability/confidence_service.py`
- Comparables:
  - `src/explainability/comparables_service.py`
- Response formatting:
  - `src/reporting/response_builder.py`

## 5. Training and Evaluation Workflows

## 5.1 Retraining pipeline
- Script: `scripts/retrain_estateprocessor_models.py`
- Outputs:
  - `artifacts/reports/ml_reports/training_estateprocessor_results.csv`
  - `artifacts/reports/ml_reports/training_estateprocessor_manifest.json`
  - refreshed model artifacts under `artifacts/models/models_estateprocessor/`

Run:
```powershell
.\.venv\Scripts\python.exe scripts\retrain_estateprocessor_models.py
```

## 5.2 Terrain strategy experiments
- Transaction-scope benchmark: `tmp/experiment_tx_models.py`
- Target-strategy benchmark: `tmp/experiment_target_strategy.py`

Run:
```powershell
.\.venv\Scripts\python.exe tmp\experiment_tx_models.py
.\.venv\Scripts\python.exe tmp\experiment_target_strategy.py
```

## 5.3 Fallback benchmark
- Script: `tests/evaluate_fallback_model_sample.py`

## 6. Running the Application

## 6.1 Backend API
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Important:
- After retraining, restart the API process to ensure refreshed manifest/model handles are loaded.

Health check:
```powershell
.\.venv\Scripts\python.exe -c "import requests; print(requests.get('http://127.0.0.1:8000/health', timeout=10).json())"
```

## 6.2 Streamlit UI
```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## 6.3 Agent API (LangGraph Scrape + Valuation)

Agent modules:
- `src/agents/langgraph_agent_api.py`
- `src/agents/listing_graph_factory.py`
- `src/agents/run_agent_api.py`
- `src/agents/agent_source_runner.py`
- `src/agents/streamlit_agent_dashboard.py`

Run the agent API launcher:
```powershell
.\.venv\Scripts\python.exe -m src.agents.run_agent_api --host 127.0.0.1 --port 8010
```

Health check:
```powershell
.\.venv\Scripts\python.exe -c "import requests; print(requests.get('http://127.0.0.1:8010/health', timeout=10).json())"
```

Trigger one immediate scheduler cycle:
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8010/scheduler/run-once" -ContentType "application/json" -Body '{"target_per_source":20,"timeout_s":30,"sleep_s":0.0}'
```

Start recurring scheduler every 24h:
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8010/scheduler/start" -ContentType "application/json" -Body '{"interval_hours":24.0,"target_per_source":20,"timeout_s":30,"sleep_s":0.0}'
```

Check scheduler status:
```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8010/scheduler/status"
```

Stop scheduler:
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8010/scheduler/stop"
```

Run source runner (20 listings per source) and export CSV:
```powershell
.\.venv\Scripts\python.exe -m src.agents.agent_source_runner --api-base-url http://127.0.0.1:8010 --target-per-source 20 --csv-output artifacts/reports/agent_source_runner_listings.csv
```

Outputs:
- `artifacts/langgraph_listings.db`
- `artifacts/reports/agent_source_runner_report.json`
- `artifacts/reports/agent_source_runner_listings.csv`

## 6.4 Agent Streamlit Dashboard

The agent dashboard reads listing cards from SQLite first and falls back to CSV if SQLite is empty.

Run:
```powershell
.\.venv\Scripts\python.exe -m streamlit run src/agents/streamlit_agent_dashboard.py
```

## 7. Testing

Run targeted inference/API tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_model_registry.py tests\test_valuation_service.py tests\test_api_output.py -q
```

Run all tests:
```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run focused agent tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_listing_graph_factory.py tests\test_langgraph_agent_api.py tests\test_agent_source_runner.py -q --basetemp C:\Users\ASUS\AppData\Local\Temp\estatemind-pytest
```

## 8. Notes

- If property-type bundles are unavailable, serving degrades gracefully to fallback model or heuristic mode.
- Manifest-driven loading allows retraining refreshes without changing API code paths.
- Current score snapshot and formulas are maintained in `artifacts/score.md`.
- Vision pipeline now uses CLIP feature inference to autofill missing property type and amenities when not provided manually.
- Upload endpoint can emit `clip_property_type_mismatch:selected=...,inferred=...` warning while still returning an estimate.
- The valuation response now includes `vision_guidance` and `vision_requires_confirmation` so the frontend can show popups and inline prompts for property-type and amenity mismatches.
- If you observe identical predicted prices across many listings, verify you launched the API with `python -m src.agents.run_agent_api` (real graph factory) and not a temporary demo lambda runner.
