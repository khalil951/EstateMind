# Integration Guide — EstateMind Feature

This guide explains how to integrate the EstateMind feature (this repository) into a teammate's project. It is optimized for **Option B: vendor-copy / copy-paste integration**, where teammates merge the needed modules into their own codebase with Claude or another assistant.

**Important:** file paths below are relative to the repository root unless stated otherwise.

**Recommended integration style**
- Option B: copy the required modules into the teammate project and adapt imports there.
- Keep the model artifacts and manifests in a predictable runtime location.
- Use the API contract from this project as the source of truth during the merge.

**What this feature provides**
- FastAPI valuation API: `src/api.py` (top-level `api.py` can also be used)
- Streamlit UI: `streamlit_app.py`
- Agent service API and scheduler: `src/agent/langgraph_agent_api.py`, `src/agent/listing_graph_factory.py`, `src/agent/run_agent_api.py`, `src/agent/agent_source_runner.py`, and `src/agent/streamlit_agent_dashboard.py`
- Django unified API scaffold: `django_backend/manage.py` and `django_backend/estatemind_backend` (migration scaffold for listings/valuation)
- Inference modules and model registry: `src/inference/` and `src/reporting/response_builder.py`
- Model artifacts and manifests: `artifacts/models/` and `artifacts/reports/ml_reports/training_estateprocessor_manifest.json`

## 1) Prerequisites (local dev)
- Python 3.10+ recommended (this repo uses a `.venv` in the root)
- Create & activate virtualenv (example uses the project's `.venv`):

```powershell
Set-Location .
.\.venv\Scripts\Activate.ps1
.
```

- Install Python dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- Ensure model artifacts and manifests exist under `artifacts/` (see section 4 for required artifacts).

## 2) Key files to copy into the teammate project
- `src/api.py` — FastAPI entrypoint and API contract.
- `src/inference/` — request mapping, model loading, feature fusion, fallback prediction, valuation orchestration.
- `src/reporting/response_builder.py` — response shaping.
- `src/explainability/` — comparables, confidence, SHAP, explanation text.
- `src/nlp/` — description sentiment, location sentiment, text analysis.
- `src/vision/` — image classifier, quality scoring, feature aggregation.
- `src/agent/` — agent API, listing graph factory, runner, and dashboard.
- `src/data_preprocessing.py` and `src/data_wrangling_pipeline.py` — only if the teammate project also needs the upstream data prep pipeline.
- `frontend_react/src/api.ts`, `frontend_react/src/pages/ValuationPage.tsx`, `frontend_react/src/types.ts` — only if the teammate project is also merging the React UI.
- `django_backend/manage.py` and `django_backend/estatemind_backend/settings.py` — only if the teammate project wants the Django scaffold.
- `artifacts/reports/ml_reports/training_estateprocessor_manifest.json` — runtime manifest.
- `artifacts/models/` — trained model artifacts referenced by the manifest.
- `artifacts/langgraph_listings.db` — optional but required for agent listings persistence.

## 3) Suggested merge order for copy-paste integration
1. Copy the backend inference modules first:
   - `src/inference/`
   - `src/explainability/`
   - `src/nlp/`
   - `src/vision/`
   - `src/reporting/response_builder.py`

2. Copy the agent service modules next:
   - `src/agent/langgraph_agent_api.py`
   - `src/agent/listing_graph_factory.py`
   - `src/agent/run_agent_api.py`
   - `src/agent/agent_source_runner.py`
   - `src/agent/streamlit_agent_dashboard.py`

3. Copy the API entrypoint next:
   - `src/api.py`
   - any shared request/response types that the teammate project needs to call it cleanly

4. Copy the artifacts and manifests:
   - `artifacts/reports/ml_reports/training_estateprocessor_manifest.json`
   - `artifacts/models/`
   - `artifacts/langgraph_listings.db` if the agent service is part of the merged project

5. If the teammate project includes the frontend:
   - copy `frontend_react/src/api.ts`
   - copy `frontend_react/src/pages/ValuationPage.tsx`
   - copy `frontend_react/src/types.ts`
   - reconcile the API base URL in the frontend config

5. Run a small smoke test after each merge step so broken imports are caught early.

## 4) Key files to review when integrating
- `src/api.py` — FastAPI entrypoint used by the primary backend API.
- `api.py` — top-level script referenced in README for running the API via `uvicorn`.
- `streamlit_app.py` — Streamlit UI; optional but useful for manual testing and demos.
- `django_backend/manage.py` — run the Django scaffold (if your team uses Django routing).
- `django_backend/estatemind_backend/settings.py` — Django settings (DB path, installed apps).
- `src/inference/` — all inference orchestration and model-loading logic (ModelRegistry, inference bundles).
- `src/reporting/response_builder.py` — constructs API response contract consumed by frontends.
- `artifacts/models/` — trained model artifacts (CatBoost, CNN fallbacks, etc.).
- `artifacts/reports/ml_reports/training_estateprocessor_manifest.json` — model manifest used at runtime to discover models.
- `artifacts/langgraph_listings.db` — optional SQLite used by agent features and the agent service.
- `frontend_react/` and `start-frontend.ps1` — React frontend scaffold and start helper (if integrating UI).

When integrating with another codebase, copy or reference the above modules (or add this repo as a Git submodule) rather than re-implementing the model-loading logic.

## 5) Environment variables & runtime config
- `DJANGO_SETTINGS_MODULE` (Django): default already set in `django_backend/manage.py` to `estatemind_backend.settings`.
- `DJANGO_SECRET_KEY` (optional): set for non-dev deployments.
- `VITE_API_BASE` (optional): used by the React frontend to point to the API, e.g. `http://127.0.0.1:8001`.
- Paths: the Django settings reference a SQLite DB at `artifacts/django_unified.db` — ensure `artifacts/` is writeable.

## 6) Required artifacts (ensure present before running)
- `artifacts/reports/ml_reports/training_estateprocessor_manifest.json` — runtime manifest enumerating CatBoost and fallback models.
- `artifacts/models/*` — model files referenced by the manifest (CatBoost joblib files, fallback CNN `*.pt`, label JSONs).
- Optional: `artifacts/langgraph_listings.db` for agent/listings flows.

If artifacts are absent, the API will run but may fall back to heuristics or fallback models; integration tests should confirm expected behavior.

## 7) How to run locally (commands)
- Run main API (FastAPI / uvicorn):

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

- Run Django unified scaffold (port 8001):

```powershell
.\.venv\Scripts\python.exe django_backend\manage.py runserver 127.0.0.1:8001
```

- Run Streamlit demo UI:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

- Start the React frontend (uses `start-frontend.ps1` in repo root):

```powershell
.\start-frontend.ps1
```

## 8) Integration steps for Option B
1. Copy the modules into the teammate project.
   - Preserve the same folder layout where possible.
   - If you rename the package root, update all internal imports consistently.

2. Merge the shared request and response flow.
   - Use `src/inference/request_mapper.py` as the source of truth for request normalization.
   - Use `src/inference/valuation_service.py` as the orchestration layer.
   - Use `src/reporting/response_builder.py` as the final response contract.

3. Merge the support services.
   - Copy `src/explainability/`, `src/nlp/`, and `src/vision/` so the inference service keeps working end to end.
   - If the teammate project does not need all of them, keep the imports aligned with the features they actually expose.

4. Merge the agent service if the team wants listings scraping / scheduler capabilities.
   - Copy `src/agent/` as a unit, because the API, graph factory, scheduler, and dashboard share imports and runtime assumptions.
   - Keep `artifacts/langgraph_listings.db` available if the agent service needs to persist scraped listings.
   - If you only need part of the agent stack, keep `run_agent_api.py` and `langgraph_agent_api.py` together so the API and scheduler stay consistent.

5. Merge the API boundary.
   - Copy `src/api.py` and connect it to the teammate project's app entrypoint.
   - Keep the route names and payload shape stable unless the consuming UI also changes.

6. Merge the model assets.
   - Copy `artifacts/reports/ml_reports/training_estateprocessor_manifest.json`.
   - Copy the referenced files under `artifacts/models/`.
   - Preserve the relative paths used by the manifest, or update the manifest and loader together.

7. If the teammate project includes the UI, merge the React files next.
   - Copy `frontend_react/src/api.ts`, `frontend_react/src/pages/ValuationPage.tsx`, and `frontend_react/src/types.ts`.
   - Update the API base URL and disable/enable UI sections only after the backend contract is stable.

8. Verify model discovery:
   - Confirm that `training_estateprocessor_manifest.json` lists model artifacts with correct relative paths.
   - Run a small script to load and run a model through `src/inference/model_registry.py` to ensure there are no path or dependency errors.

9. Run smoke tests / health checks:
   - Health endpoint (if using FastAPI or agent API):

```powershell
.\.venv\Scripts\python.exe -c "import requests; print(requests.get('http://127.0.0.1:8000/health', timeout=10).json())"
```

   - For Django scaffold (if used):

```powershell
.\.venv\Scripts\python.exe -c "import requests; print(requests.get('http://127.0.0.1:8001/health', timeout=10).json())"
```

10. Run a functional test through your integration (example):
   - POST a minimal payload to `/estimate` or `/estimate-upload` (see `src/api.py` contract) and validate the response shape (confidence, price, feature impacts).

## 9) API endpoints teammates will likely call
- `POST /estimate` — synchronous estimate from structured features
- `POST /estimate-upload` — estimate with image upload (non-blocking guidance warnings possible)
- `POST /listings/add-from-valuation` — (Django unified API) add listing from valuation flow
- `GET /health` — basic service health check

Confirm exact contract shapes in `src/reporting/response_builder.py` and `src/api.py` before integrating.

## 10) PR checklist for integration
- [ ] Add or update `requirements.txt` in the consuming repo to include any new runtime deps used by `src/` (e.g., CatBoost, timm, torch if using CNN fallbacks).
- [ ] Ensure `artifacts/` with models and `training_estateprocessor_manifest.json` is accessible by runtime.
- [ ] Add environment vars/config entries for `VITE_API_BASE` or `DJANGO_SECRET_KEY` as needed.
- [ ] Add smoke test that runs one `/estimate` request against local or CI deployment.
- [ ] If vendoring code, update import paths and run `pytest` for the integrated test suite.
- [ ] Document in the consuming repo README how to run the EstateMind service locally (copy the short run commands above).

## 11) Troubleshooting notes
- If predictions are identical across many inputs, verify the manifest and that model files are correctly loaded (not falling back to a single heuristic path).
- If `torch` or `timm` is not available but CNN fallbacks are referenced, either install GPU/CPU PyTorch builds or remove fallback usage in config.
- For performance, enable `torch.cuda.amp` if GPU is available and tune batch sizes.
- If teammates are pasting modules into a different package root, the most common failure is stale internal imports. Fix those first before debugging model loading.
- If the merged project uses a different frontend route structure, keep the API payload shape stable and only remap the page-level state.
- If the agent service is included, verify that the scheduler endpoints and SQLite file path match the consuming project before wiring the UI.

---

If you want, I can:
- Produce a small `integration_test.py` script that performs a sample `/estimate` request and validates response keys.
- Add a Docker Compose sample that runs the API and exposes it for teammates.

Tell me which of the above you'd like next and I will add it to the repo.
