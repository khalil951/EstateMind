import json
from pathlib import Path


def _load_notebook(path: str) -> dict:
    p = Path(path)
    assert p.exists(), f"Notebook not found: {path}"
    return json.loads(p.read_text(encoding="utf-8"))


def _all_sources(nb: dict) -> str:
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        chunks.append("".join(cell.get("source", [])))
    return "\n".join(chunks)


def test_notebook_ml_structure() -> None:
    nb = _load_notebook("notebooks/notebook_ml.ipynb")
    assert len(nb.get("cells", [])) > 0
    text = _all_sources(nb)

    # Core modeling/preprocessing elements expected in this notebook
    assert "EstatePreprocessor" in text or "EstateProcessor" in text
    assert "GridSearchCV" in text
    assert "train_models_by_property_type" in text


def test_notebook_images_structure() -> None:
    nb = _load_notebook("notebooks/notebook_images_preprocessing.ipynb")
    assert len(nb.get("cells", [])) > 0
    text = _all_sources(nb)

    # Image preprocessing + split/dataloader section
    assert "listing_images.csv" in text
    assert "Train/Val/Test Split + PyTorch DataLoader" in text
    assert "ListingImageDataset" in text
    assert "DataLoader" in text

