#!/usr/bin/env python3
"""Verify governorate centroid fallback is working for real scenarios."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.api import PropertyRequest
from src.inference.valuation_service import ValuationService

service = ValuationService()

# Test scenarios with REAL governorates/cities
test_cases = [
    {
        "name": "Test 1: Valid city in known governorate (Tunis/Tunis)",
        "request": PropertyRequest(
            property_type="Appartement",
            governorate="Tunis",
            city="Tunis",
            size_m2=100.0,
            bedrooms=2,
            bathrooms=1,
            condition="Good",
        ),
    },
    {
        "name": "Test 2: Unknown city, but known governorate (Sfax/Unknown_City_In_Sfax)",
        "request": PropertyRequest(
            property_type="Appartement",
            governorate="Sfax",
            city="Unknown_City_In_Sfax_XYZ",
            size_m2=100.0,
            bedrooms=2,
            bathrooms=1,
            condition="Good",
        ),
    },
    {
        "name": "Test 3: Small rural governorate (Kebili/Kebili)",
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

print("=" * 90)
print("VERIFICATION: Governorate Centroid Fallback")
print("=" * 90)
print("\nFIX: When city not in city_geo_lookup, fall back to governorate centroid")
print("EXPECTED: Unknown city in known governorate should NOT trigger geo_lookup_missing")
print()

for scenario in test_cases:
    print(f"\n{scenario['name']}")
    print("-" * 90)
    
    req = scenario["request"]
    print(f"  governorate: '{req.governorate}'")
    print(f"  city: '{req.city}'")
    
    try:
        result = service.estimate(req)
        warnings = result.get("uncertainty_reasons", [])
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in sorted(warnings):
            print(f"    - {w}")
        
        # Check if geo_lookup_missing is present
        if "geo_lookup_missing" in warnings:
            print(f"\n  [!] geo_lookup_missing PRESENT - centroid fallback did NOT work")
        else:
            print(f"\n  [OK] geo_lookup_missing ABSENT - centroid fallback working!")
            
        print(f"\n  Price: {result.get('estimated_price', 0):,} TND | Confidence: {result.get('confidence', 0)*100:.0f}%")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 90)
print("INTERPRETATION")
print("=" * 90)
print("""
Test 1 (Known city): Should have minimal warnings
  - Healthy case; city is in reference data

Test 2 (Unknown city, known gov): CRITICAL TEST FOR FIX
  - BEFORE FIX: geo_lookup_missing present (city not found, no coordinates)
  - AFTER FIX: geo_lookup_missing should NOT be present (uses gov centroid)
  - This proves the centroid fallback is working!

Test 3 (Small governorate): Degrades gracefully
  - May still have warnings about sparse data, but coordinates should be filled
  - geo_lookup_missing should be absent or minimal
""")
