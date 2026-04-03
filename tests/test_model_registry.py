from src.inference.model_registry import ModelRegistry


def test_model_registry_loads_manifest_handles() -> None:
    registry = ModelRegistry()
    handles = registry.list_handles()
    assert len(handles) >= 1
    assert any(handle.scope == "global" for handle in handles)


def test_best_handle_prefers_property_specific_model() -> None:
    registry = ModelRegistry()
    handle = registry.get_best_handle("Appartement")
    assert handle is not None
    assert handle.property_type in {"Appartement", "ALL"}


def test_catboost_handle_can_build_dynamic_bundle() -> None:
    registry = ModelRegistry()
    handle = registry.maybe_load_bundle(registry.get_best_handle("Appartement"))
    assert handle is not None
    assert handle.bundle_available is True
    assert handle.bundle is not None
    assert handle.bundle.version == "estatebundle-v1"
