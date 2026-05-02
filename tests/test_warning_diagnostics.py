"""
Diagnostic test to map exact field values to warning codes.
Shows which request fields trigger each uncertainty reason.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.api import PropertyRequest
from src.inference.valuation_service import ValuationService

# Initialize the service
service = ValuationService()

# Test scenarios
test_cases = [
    {
        "name": "Scenario 1: Known city (should have no warnings)",
        "request": PropertyRequest(
            property_type="Appartement",
            governorate="Tunis",
            city="Tunis",
            neighborhood="Medina",
            size_m2=100.0,
            bedrooms=2,
            bathrooms=1,
            condition="Good",
            has_pool=False,
            has_garden=False,
            has_parking=True,
            sea_view=False,
            elevator=False,
            description="Nice apartment",
            uploaded_images_count=0,
            image_refs=[],
        ),
    },
    {
        "name": "Scenario 2: Unknown city (triggers unknown_city_data_coverage + geo_lookup_missing)",
        "request": PropertyRequest(
            property_type="Appartement",
            governorate="Unknown_Gov",
            city="Unknown_City_XYZ_12345",
            neighborhood="",
            size_m2=100.0,
            bedrooms=2,
            bathrooms=1,
            condition="Good",
            has_pool=False,
            has_garden=False,
            has_parking=True,
            sea_view=False,
            elevator=False,
            description="",
            uploaded_images_count=0,
            image_refs=[],
        ),
    },
    {
        "name": "Scenario 3: City exists, governorate fallback needed (local_price_prior_fallback_governorate)",
        "request": PropertyRequest(
            property_type="Appartement",
            governorate="Sfax",
            city="Sfax",  # City-level prior might be weak
            neighborhood="",
            size_m2=100.0,
            bedrooms=2,
            bathrooms=1,
            condition="Good",
            has_pool=False,
            has_garden=False,
            has_parking=True,
            sea_view=False,
            elevator=False,
            description="",
            uploaded_images_count=0,
            image_refs=[],
        ),
    },
    {
        "name": "Scenario 4: Empty/missing city (triggers local_price_prior_city_missing_input)",
        "request": PropertyRequest(
            property_type="Appartement",
            governorate="Tunis",
            city="",  # Empty city
            neighborhood="",
            size_m2=100.0,
            bedrooms=2,
            bathrooms=1,
            condition="Good",
            has_pool=False,
            has_garden=False,
            has_parking=True,
            sea_view=False,
            elevator=False,
            description="",
            uploaded_images_count=0,
            image_refs=[],
        ),
    },
    {
        "name": "Scenario 5: Small governorate (might trigger governorate_data_coverage)",
        "request": PropertyRequest(
            property_type="Appartement",
            governorate="Kebili",  # Small governorate with weak data coverage
            city="Kebili",
            neighborhood="",
            size_m2=100.0,
            bedrooms=2,
            bathrooms=1,
            condition="Good",
            has_pool=False,
            has_garden=False,
            has_parking=True,
            sea_view=False,
            elevator=False,
            description="",
            uploaded_images_count=0,
            image_refs=[],
        ),
    },
]

print("=" * 80)
print("WARNING DIAGNOSTIC TEST: Field Values to Uncertainty Reasons")
print("=" * 80)

for scenario in test_cases:
    print(f"\n{scenario['name']}")
    print("-" * 80)
    
    req = scenario["request"]
    print(f"INPUT FIELDS:")
    print(f"  governorate: '{req.governorate}'")
    print(f"  city: '{req.city}'")
    print(f"  size_m2: {req.size_m2}")
    print(f"  property_type: '{req.property_type}'")
    
    try:
        result = service.estimate(req)
        print(f"\nOUTPUT WARNINGS/OOD FLAGS:")
        uncertainty = result.get("uncertainty_reasons", [])
        if uncertainty:
            for reason in uncertainty:
                print(f"  [+] {reason}")
        else:
            print(f"  (none)")
        print(f"\nESTIMATED PRICE: {result.get('estimated_price', 0):,} TND")
        print(f"CONFIDENCE: {result.get('confidence', 0) * 100:.1f}%")
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "=" * 80)
print("MAPPING SUMMARY")
print("=" * 80)
print("""
geo_lookup_missing:
  TRIGGERED BY: city NOT in city_geo_lookup AND latitude is missing
  COMMON CAUSES: Unknown/misspelled city, no coordinates provided
  FIELD VALUES: city = 'Unknown_City_XYZ_12345'

local_price_prior_fallback_governorate:
  TRIGGERED BY: No exact city+gov prior AND no city prior, but gov prior exists
  COMMON CAUSES: City-level data coverage is weak/missing
  FIELD VALUES: governorate = known, city = known but sparse
  OOD FLAG: ood:local_price_prior_city_data_coverage added

ood:unknown_city_data_coverage:
  TRIGGERED BY: city NOT in city_geo_lookup AND latitude is NaN
  COMMON CAUSES: City name is unknown to the system
  FIELD VALUES: city = 'Unknown_City_XYZ_12345', latitude = NaN

ood:local_price_prior_city_data_coverage:
  TRIGGERED BY: Fallback to governorate pricing because city-level prior unavailable
  COMMON CAUSES: Sparse historical data for the requested city
  FIELD VALUES: city exists but not well-represented in training data
  WHEN: Added together with local_price_prior_fallback_governorate

FIELD COMBINATION TRIGGERS:
  1. Unknown city + no lat/lon
     → geo_lookup_missing + ood:unknown_city_data_coverage
  
  2. Known city + weak history
     → local_price_prior_fallback_governorate + ood:local_price_prior_city_data_coverage
  
  3. Empty city + known governorate
     → Falls back to governorate prior (no OOD flag)
""")

print("\n" + "=" * 80)
print("PROPOSED FIXES")
print("=" * 80)
print("""
FIX 1: Populate city_geo_lookup with more cities
  - Add missing city coordinates to reference data
  - Location: Reference dataset city/latitude/longitude columns
  - Impact: Eliminates geo_lookup_missing + ood:unknown_city_data_coverage
  - Status: Data-driven; requires more listing data with coordinates

FIX 2: Improve city-level price prior coverage
  - Aggregate more transactions per city in training data
  - Location: src/inference/inference_bundle.py _resolve_price_prior()
  - Impact: Reduces fallback to governorate (eliminates local_price_prior_fallback_governorate)
  - Status: Data-driven; requires more historical listings per city

FIX 3: Add default coordinates for unmapped cities
  - Use governorate centroid or median city coords when exact city fails
  - Location: src/inference/inference_bundle.py, ~line 257
  - Example: if city_key not in lookup, use governorate's centroid
  - Impact: Provides better distance-based features even for unknown cities

FIX 4: Validate and normalize city/governorate at API boundary
  - Add pre-validation step to catch empty/obviously-wrong city values
  - Location: src/api.py PropertyRequest validation
  - Example: city must be non-empty, governorate must match known governorates
  - Impact: Fails fast on bad input rather than silent fallback

FIX 5: Enrich the reference dataset for sparse governorates
  - For Kebili, Tataouine, Medenine: actively collect more listings
  - Impact: Reduces uncertainty for rural/sparse regions
  - Status: Business/data collection task

IMMEDIATE WIN (FIX 3):
  If you add governorate centroids as fallback coordinates, you can:
  - Keep distance-based features non-null even for unknown cities
  - Reduce geo_lookup_missing from appearing twice
  - Provide more consistent pricing signals
""")
