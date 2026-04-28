# Integration Guide — EstateMind Feature

This guide explains how to integrate the EstateMind feature (this repository) into a teammate's project. It lists the required files, runtime services, environment variables, and step-by-step actions to install, run, and verify the integration.

**Important:** file paths below are relative to the repository root unless stated otherwise.

**What this feature provides**
- FastAPI valuation API: `src/api.py` (top-level `api.py` can also be used)
- Streamlit UI: `streamlit_app.py`
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

## 2) Key files to include / review when integrating
- `src/api.py` — FastAPI entrypoint used by the primary backend API.
- `api.py` — top-level script referenced in README for running the API via `uvicorn`.
- `streamlit_app.py` — Streamlit UI; optional but useful for manual testing and demos.
- `django_backend/manage.py` — run the Django scaffold (if your team uses Django routing).
- `django_backend/estatemind_backend/settings.py` — Django settings (DB path, installed apps).
- `src/inference/` — all inference orchestration and model-loading logic (ModelRegistry, inference bundles).
- `src/reporting/response_builder.py` — constructs API response contract consumed by frontends.
- `artifacts/models/` — trained model artifacts (CatBoost, CNN fallbacks, etc.).
- `artifacts/reports/ml_reports/training_estateprocessor_manifest.json` — model manifest used at runtime to discover models.
- `artifacts/langgraph_listings.db` — optional SQLite used by agent features (if used).
- `frontend_react/` and `start-frontend.ps1` — React frontend scaffold and start helper (if integrating UI).

When integrating with another codebase, copy or reference the above modules (or add this repo as a Git submodule) rather than re-implementing the model-loading logic.

## 3) Environment variables & runtime config
- `DJANGO_SETTINGS_MODULE` (Django): default already set in `django_backend/manage.py` to `estatemind_backend.settings`.
- `DJANGO_SECRET_KEY` (optional): set for non-dev deployments.
- `VITE_API_BASE` (optional): used by the React frontend to point to the API, e.g. `http://127.0.0.1:8001`.
- Paths: the Django settings reference a SQLite DB at `artifacts/django_unified.db` — ensure `artifacts/` is writeable.

## 4) Required artifacts (ensure present before running)
- `artifacts/reports/ml_reports/training_estateprocessor_manifest.json` — runtime manifest enumerating CatBoost and fallback models.
- `artifacts/models/*` — model files referenced by the manifest (CatBoost joblib files, fallback CNN `*.pt`, label JSONs).
- Optional: `artifacts/langgraph_listings.db` for agent/listings flows.

If artifacts are absent, the API will run but may fall back to heuristics or fallback models; integration tests should confirm expected behavior.

## 5) How to run locally (commands)
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

## 6) Integration steps (recommended)
1. Decide integration approach:
   - Option A: Add this repo as a Git submodule to your teammate's project and import `src` modules directly.
   - Option B: Vendor-copy the specific modules you need (`src/inference`, `src/reporting`, `src/api.py`) into your codebase and adapt imports.
   - Option C: Run EstateMind as a separate service (recommended for minimal coupling) and call it over HTTP from teammates' code.

2. If choosing Option C (service approach):
   - Start the API (see commands above) in CI/dev.
   - Expose the API base URL to teammates (e.g., `http://dev-estatemind:8000`).
   - Ensure `artifacts/` with models is available to the service process.

3. If choosing Option A or B (library approach):
   - Ensure the teammate project matches the Python version and has the required dependencies installed.
   - Add `src/inference` and `src/reporting` into the import path (or install as an internal package).
   - Wire the teammate project's request mapping to use `src/inference/valuation_service.py` and `src/reporting/response_builder.py`.

4. Verify model discovery:
   - Confirm that `training_estateprocessor_manifest.json` lists model artifacts with correct relative paths.
   - Run a small script to load and run a model through `src/inference/model_registry.py` to ensure there are no path or dependency errors.

5. Run smoke tests / health checks:
   - Health endpoint (if using FastAPI or agent API):

```powershell
.\.venv\Scripts\python.exe -c "import requests; print(requests.get('http://127.0.0.1:8000/health', timeout=10).json())"
```

   - For Django scaffold (if used):

```powershell
.\.venv\Scripts\python.exe -c "import requests; print(requests.get('http://127.0.0.1:8001/health', timeout=10).json())"
```

6. Run a functional test through your integration (example):
   - POST a minimal payload to `/estimate` or `/estimate-upload` (see `src/api.py` contract) and validate the response shape (confidence, price, feature impacts).

## 7) API endpoints teammates will likely call
- `POST /estimate` — synchronous estimate from structured features
- `POST /estimate-upload` — estimate with image upload (non-blocking guidance warnings possible)
- `POST /listings/add-from-valuation` — (Django unified API) add listing from valuation flow
- `GET /health` — basic service health check

Confirm exact contract shapes in `src/reporting/response_builder.py` and `src/api.py` before integrating.

## 8) PR checklist for integration
- [ ] Add or update `requirements.txt` in the consuming repo to include any new runtime deps used by `src/` (e.g., CatBoost, timm, torch if using CNN fallbacks).
- [ ] Ensure `artifacts/` with models and `training_estateprocessor_manifest.json` is accessible by runtime.
- [ ] Add environment vars/config entries for `VITE_API_BASE` or `DJANGO_SECRET_KEY` as needed.
- [ ] Add smoke test that runs one `/estimate` request against local or CI deployment.
- [ ] If vendoring code, update import paths and run `pytest` for the integrated test suite.
- [ ] Document in the consuming repo README how to run the EstateMind service locally (copy the short run commands above).

## 9) Troubleshooting notes
- If predictions are identical across many inputs, verify the manifest and that model files are correctly loaded (not falling back to a single heuristic path).
- If `torch` or `timm` is not available but CNN fallbacks are referenced, either install GPU/CPU PyTorch builds or remove fallback usage in config.
- For performance, enable `torch.cuda.amp` if GPU is available and tune batch sizes.

---

If you want, I can:
- Produce a small `integration_test.py` script that performs a sample `/estimate` request and validates response keys.
- Add a Docker Compose sample that runs the API and exposes it for teammates.

Tell me which of the above you'd like next and I will add it to the repo.
