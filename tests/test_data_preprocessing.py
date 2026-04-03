import pandas as pd

from src.data_preprocessing import BasePreprocessor, PreprocessConfig


def _preprocessor() -> BasePreprocessor:
    return BasePreprocessor(PreprocessConfig(geocode_max_cities=10, geocode_sleep_sec=0.0, knn_neighbors=2))


def test_title_extraction_splus_and_studio_rules() -> None:
    pp = _preprocessor()
    df = pd.DataFrame(
        [
            {
                "record_id": "r1",
                "title": "A louer appartement S+2 avec parking",
                "price_tnd": 900.0,
                "city": "Tunis",
                "governorate": "Tunis",
            },
            {
                "record_id": "r2",
                "title": "Studio meuble a la marsa",
                "price_tnd": 1200.0,
                "city": "La Marsa",
                "governorate": "Tunis",
            },
        ]
    )

    out = pp._apply_title_extraction(df)

    row1 = out.loc[out["record_id"] == "r1"].iloc[0]
    assert int(row1["rooms"]) == 3
    assert int(row1["bedrooms"]) == 2
    assert int(row1["bathrooms"]) == 1
    assert row1["property_type"] == "Appartement"
    assert row1["transaction_type"] == "rent"

    row2 = out.loc[out["record_id"] == "r2"].iloc[0]
    assert int(row2["rooms"]) == 2
    assert int(row2["bedrooms"]) == 1
    assert int(row2["bathrooms"]) == 1
    assert row2["property_type"] == "Appartement"


def test_advanced_transaction_type_uses_price_when_no_title_cue() -> None:
    pp = _preprocessor()
    df = pd.DataFrame(
        [
            {"record_id": "a", "title": "bien immobilier", "price_tnd": 20.0, "transaction_type": None},
            {"record_id": "b", "title": "bien immobilier", "price_tnd": 80.0, "transaction_type": None},
            {"record_id": "c", "title": "bien immobilier", "price_tnd": 500.0, "transaction_type": None},
            {"record_id": "d", "title": "a vendre villa", "price_tnd": 70.0, "transaction_type": None},
        ]
    )

    out = pp._fill_transaction_type_advanced(df)
    values = dict(zip(out["record_id"], out["transaction_type"]))

    # For this distribution, very low prices should map to rent.
    assert values["a"] == "rent"
    assert values["c"] == "sale"
    # Title cue should override price heuristic.
    assert values["d"] == "sale"


def test_geocode_city_cache_reuses_calls_for_duplicate_cities() -> None:
    pp = _preprocessor()

    class _Loc:
        def __init__(self, lat: float, lon: float):
            self.latitude = lat
            self.longitude = lon

    class _FakeGeolocator:
        def __init__(self):
            self.calls: list[str] = []

        def geocode(self, query: str, timeout: float = 10.0):  # noqa: ARG002
            self.calls.append(query)
            mapping = {
                "Tunis, Tunisia": _Loc(36.8, 10.1),
                "Sfax, Tunisia": _Loc(34.7, 10.8),
            }
            return mapping.get(query)

    fake = _FakeGeolocator()
    pp._geolocator = fake

    df = pd.DataFrame(
        [
            {"record_id": "1", "city": "Tunis"},
            {"record_id": "2", "city": "Tunis"},
            {"record_id": "3", "city": "Sfax"},
        ]
    )

    out = pp._add_lat_lon_from_city(df)
    assert out["latitude"].notna().all()
    assert out["longitude"].notna().all()

    # Exactly one network call per unique city.
    assert fake.calls.count("Tunis, Tunisia") == 1
    assert fake.calls.count("Sfax, Tunisia") == 1


def test_preprocess_drops_columns_and_title_and_generates_description() -> None:
    pp = _preprocessor()
    pp._geolocator = None  # avoid external calls in unit test

    df = pd.DataFrame(
        [
            {
                "record_id": "x1",
                "source": "A",
                "source_file": "f.csv",
                "listing_url": "http://example.com",
                "title": "A louer appartement S+1 avec parking",
                "description": None,
                "transaction_type": None,
                "property_type": None,
                "price_tnd": 1000.0,
                "surface_m2": 80.0,
                "rooms": None,
                "bedrooms": None,
                "bathrooms": None,
                "governorate": "Tunis",
                "city": "Tunis",
                "neighborhood": "n1",
                "posted_at": "2025-01-01",
                "scraped_at": "2025-01-02",
                "currency": "TND",
                "image_url": None,
                "image_count": 1,
            }
        ]
    )

    out, report = pp.preprocess(df)

    assert "title" not in out.columns
    assert "source" not in out.columns
    assert "listing_url" not in out.columns
    assert "description" in out.columns
    assert out["description"].notna().all()
    assert report["row_count"] == 1
