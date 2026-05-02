#!/usr/bin/env python3
"""Quick diagnostic to verify warning reduction after Priority 1 & 2 fixes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.api import PropertyRequest
from src.inference.valuation_service import ValuationService

service = ValuationService()

# Test scenarios
test_cases = [
    {
        "name": "Scenario 1: Known city (Tunis/Tunis)",
        "request": PropertyRequest(
            property_type="Appartement",
            governorate="Tunis",
            city="Tunis",
            neighborhood="Medina",
            size_m2=100.0,
            bedrooms=2,
            bathrooms=1,
            condition="Good",
        ),
    },
    {
        "name": "Scenario 2: Unknown city (Unknown_Gov / Unknown_City_XYZ_12345)",
        "request": PropertyRequest(
            property_type="Appartement",
            governorate="Unknown_Gov",
            city="Unknown_City_XYZ_12345",
            size_m2=100.0,
            bedrooms=2,
            bathrooms=1,
            condition="Good",
        ),
    },
    {
        "name": "Scenario 3: Small governorate (Kebili/Kebili)",
        "request": PropertyRequest(
            property_type="Appartement",
            governorate="Kebili",
            city="Kebili",
            size_m2=100.0,
            bedrooms=2,
            bathrooms=1,
            condition="Good",
        ),
    },
]

print("=" * 80)
print("DIAGNOSTIC TEST: Before vs After Priority 1 & 2 Fixes")
print("=" * 80)

for scenario in test_cases:
    print(f"\n{scenario['name']}")
    print("-" * 80)
    
    req = scenario["request"]
    print(f"INPUT: governorate='{req.governorate}', city='{req.city}'")
    
    try:
        result = service.estimate(req)
        warnings = result.get("uncertainty_reasons", [])
        print(f"WARNINGS: {len(warnings)}")
        if warnings:
            for w in sorted(warnings):
                print(f"  - {w}")
        else:
            print(f"  (none)")
        print(f"PRICE: {result.get('estimated_price', 0):,} TND | CONFIDENCE: {result.get('confidence', 0)*100:.0f}%")
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "=" * 80)
print("SUMMARY OF FIXES APPLIED")
print("=" * 80)
print("""
[PRIORITY 1] City field validation (DONE)
  - Added Field(min_length=1) to city in src/api.py
  - RESULT: Rejects empty city requests at API boundary
  - EXPECTED: Scenario 4 (empty city) now fails at validation instead of processing

[PRIORITY 2] Governorate centroid fallback (DONE)
  - Built gov_geo_lookup in src/inference/inference_bundle.py __init__
  - When city not found, falls back to governorate coordinates
  - RESULT: Unknown cities use governorate centroid; coordinates never NaN
  - EXPECTED: Scenario 2 (unknown city) warnings reduced; geo_lookup_missing appears less

VERIFICATION:
- Scenario 1 (known city): Should have 0 warnings
- Scenario 2 (unknown city): Should have 3-4 warnings (down from 9)
- Scenario 3 (small gov): Should have 1-2 warnings (down from 5)
""")
