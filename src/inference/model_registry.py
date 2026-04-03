"""Model artifact discovery and lazy loading for valuation serving.

The registry reads the training manifest produced by the offline pipeline
and exposes model handles for by-type and global estimators. It is designed
to degrade gracefully when artifacts or dependencies are missing so the API
can continue operating in fallback mode while model export work is still in
progress.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.inference.inference_bundle import InferenceBundle, load_reference_dataset

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover
    joblib = None  # type: ignore[assignment]


def project_root() -> Path:
    """Return the repository root used to resolve relative artifact paths."""

    return Path(__file__).resolve().parents[2]


@dataclass
class ModelHandle:
    """Reference to a saved estimator together with serving metadata.

    Attributes:
        scope: Training scope such as ``by_type`` or ``global``.
        property_type: Property type associated with the estimator.
        model_name: Human-readable model identifier from the manifest.
        path: Absolute path to the serialized artifact.
        metrics: Offline evaluation metrics saved in the manifest.
        estimator: Lazily loaded estimator instance when available.
        load_error: Error captured during lazy loading, if any.
    """

    scope: str
    property_type: str
    model_name: str
    path: Path
    metrics: dict[str, Any]
    estimator: Any | None = None
    load_error: str | None = None
    bundle: InferenceBundle | None = None
    bundle_error: str | None = None

    @property
    def available(self) -> bool:
        """Whether the artifact exists and has not failed to load."""

        return self.path.exists() and self.load_error is None

    @property
    def bundle_available(self) -> bool:
        """Whether a serving bundle is ready for estimator-backed inference."""

        return self.bundle is not None and self.bundle_error is None


class ModelRegistry:
    """Load model metadata and lazily expose model handles."""

    def __init__(self, manifest_path: str | Path | None = None) -> None:
        """Initialize the registry from the training manifest if present."""

        root = project_root()
        default_manifest = root / "artifacts" / "reports" / "ml_reports" / "training_estateprocessor_manifest.json"
        self.manifest_path = Path(manifest_path) if manifest_path else default_manifest
        self.root = root
        self._handles = self._load_manifest()
        self._reference_df: Any | None = None

    def _load_manifest(self) -> list[ModelHandle]:
        """Parse the manifest file into ``ModelHandle`` records."""

        if not self.manifest_path.exists():
            return []
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        handles: list[ModelHandle] = []
        for item in payload:
            rel_path = Path(str(item.get("path", "")))
            resolved = self.root / rel_path
            if not resolved.exists() and "artifacts\\models_estateprocessor" in str(rel_path).replace("/", "\\"):
                alt_rel = Path(str(rel_path).replace("artifacts\\models_estateprocessor", "artifacts\\models\\models_estateprocessor"))
                alt_resolved = self.root / alt_rel
                if alt_resolved.exists():
                    resolved = alt_resolved
            handles.append(
                ModelHandle(
                    scope=str(item.get("scope", "")),
                    property_type=str(item.get("property_type", "")),
                    model_name=str(item.get("model_name", "")),
                    path=resolved,
                    metrics=dict(item.get("metrics", {})),
                )
            )
        return handles

    def list_handles(self) -> list[ModelHandle]:
        """Return every known handle from the loaded manifest."""

        return list(self._handles)

    def get_property_handle(self, property_type: str) -> ModelHandle | None:
        """Return the by-type model handle for a normalized property type."""

        wanted = str(property_type).strip().lower()
        for handle in self._handles:
            if handle.scope == "by_type" and handle.property_type.strip().lower() == wanted:
                return handle
        return None

    def get_global_handle(self) -> ModelHandle | None:
        """Return the global fallback model handle, if one is declared."""

        for handle in self._handles:
            if handle.scope == "global":
                return handle
        return None

    def get_best_handle(self, property_type: str) -> ModelHandle | None:
        """Prefer a by-type handle and fall back to the global model."""

        return self.get_property_handle(property_type) or self.get_global_handle()

    def maybe_load_estimator(self, handle: ModelHandle | None) -> ModelHandle | None:
        """Lazy-load the estimator referenced by ``handle`` when possible.

        The method is idempotent and stores any loading failure on the handle
        itself so callers can inspect the reason fallback mode was used.
        """

        if handle is None or handle.estimator is not None or handle.load_error is not None:
            return handle
        if not handle.path.exists():
            handle.load_error = f"Missing model artifact: {handle.path}"
            return handle
        if joblib is None:
            handle.load_error = "joblib is not installed"
            return handle
        try:
            handle.estimator = joblib.load(handle.path)
        except Exception as exc:  # pragma: no cover
            handle.load_error = str(exc)
        return handle

    def _get_reference_df(self) -> Any:
        if self._reference_df is None:
            self._reference_df = load_reference_dataset()
        return self._reference_df

    def maybe_load_bundle(self, handle: ModelHandle | None) -> ModelHandle | None:
        """Attach a serving bundle to a compatible handle when possible."""

        handle = self.maybe_load_estimator(handle)
        if handle is None or handle.bundle is not None or handle.bundle_error is not None:
            return handle
        if handle.load_error is not None:
            handle.bundle_error = handle.load_error
            return handle
        try:
            reference_df = self._get_reference_df()
            handle.bundle = InferenceBundle.from_handle(handle, reference_df)
        except Exception as exc:
            handle.bundle_error = str(exc)
        return handle
