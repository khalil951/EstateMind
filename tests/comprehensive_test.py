#!/usr/bin/env python3
"""Comprehensive test suite for API with multiple sample properties"""

import json
import sys
import io

# Fix encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.api import PropertyRequest, estimate_price

# Test cases covering various scenarios
TEST_CASES = [
    {
        "name": "Luxury Apartment in La Marsa",
        "property": PropertyRequest(
            property_type="Appartement",
            governorate="Tunis",
            city="La Marsa",
            neighborhood="Sidi Abdelaziz",
            size_m2=120,
            bedrooms=3,
            bathrooms=2,
            condition="Excellent",
            has_pool=False,
            has_garden=True,
            has_parking=True,
            sea_view=True,
            elevator=True,
            description="Appartement renove avec vue mer exceptionnelle, cuisine moderne et emplacement premium a La Marsa.",
            uploaded_images_count=2
        )
    },
    {
        "name": "Budget Apartment in Sfax",
        "property": PropertyRequest(
            property_type="Appartement",
            governorate="Sfax",
            city="Sfax",
            neighborhood="Cite Ennasr",
            size_m2=60,
            bedrooms=1,
            bathrooms=1,
            condition="Fair",
            has_pool=False,
            has_garden=False,
            has_parking=False,
            sea_view=False,
            elevator=False,
            description="petit appartement dans le centre ville de Sfax",
            uploaded_images_count=0
        )
    },
    {
        "name": "Family House in Sousse",
        "property": PropertyRequest(
            property_type="Maison",
            governorate="Sousse",
            city="Sousse",
            neighborhood="Ribat Khaled",
            size_m2=250,
            bedrooms=4,
            bathrooms=3,
            condition="Good",
            has_pool=True,
            has_garden=True,
            has_parking=True,
            sea_view=False,
            elevator=False,
            description="Maison spacieuse avec jardin prive, piscine et parking couvert nearby.",
            uploaded_images_count=5
        )
    },
    {
        "name": "Land Plot in Ariana",
        "property": PropertyRequest(
            property_type="Terrain",
            governorate="Ariana",
            city="Ariana",
            neighborhood="",
            size_m2=500,
            bedrooms=0,
            bathrooms=0,
            condition="New",
            has_pool=False,
            has_garden=False,
            has_parking=False,
            sea_view=False,
            elevator=False,
            description="Terrain urbain avec infrastructure existante, bon investissement.",
            uploaded_images_count=1
        )
    },
    {
        "name": "Renovated Apartment in Bizerte",
        "property": PropertyRequest(
            property_type="Appartement",
            governorate="Bizerte",
            city="Bizerte",
            neighborhood="Medina",
            size_m2=85,
            bedrooms=2,
            bathrooms=1,
            condition="Excellent",
            has_pool=False,
            has_garden=False,
            has_parking=True,
            sea_view=True,
            elevator=False,
            description="Appartement rénové récemment. Vue sur la mer. Proche du port.",
            uploaded_images_count=3
        )
    },
    {
        "name": "Minimal Info Property",
        "property": PropertyRequest(
            property_type="Appartement",
            governorate="Tunis",
            city="Tunis",
            neighborhood="",
            size_m2=100,
            bedrooms=0,
            bathrooms=0,
            condition="Fair",
            has_pool=False,
            has_garden=False,
            has_parking=False,
            sea_view=False,
            elevator=False,
            description="General apartment",
            uploaded_images_count=0
        )
    },
]

def validate_response(result_dict, test_name):
    """Validate that all expected fields are populated"""
    empty_fields = []
    expected_fields = [
        "estimated_price", "lower_bound", "upper_bound", "price_per_m2",
        "confidence", "confidence_level", "features_impact", "comparables",
        "ai_explanation", "image_analysis", "text_analysis", "market_context", "shap"
    ]

    for field in expected_fields:
        if field not in result_dict:
            empty_fields.append(f"    MISSING FIELD: {field}")
            continue

        value = result_dict[field]
        if value is None:
            empty_fields.append(f"    EMPTY: {field} = None")
        elif isinstance(value, (list, dict)) and len(value) == 0:
            empty_fields.append(f"    EMPTY: {field} = empty {type(value).__name__}")
        elif isinstance(value, str) and value.strip() == "":
            empty_fields.append(f"    EMPTY: {field} = empty string")

    return empty_fields

# Main test execution
print("=" * 80)
print("COMPREHENSIVE API TEST SUITE")
print("=" * 80)
print()

all_passed = True
results_summary = []

for i, test_case in enumerate(TEST_CASES, 1):
    test_name = test_case["name"]
    property_obj = test_case["property"]

    print(f"TEST {i}/{len(TEST_CASES)}: {test_name}")
    print("-" * 80)

    try:
        result = estimate_price(property_obj)
        result_dict = result.model_dump()
        empty_fields = validate_response(result_dict, test_name)

        if empty_fields:
            print("  STATUS: [FAIL] - Empty/Missing fields detected")
            for field in empty_fields:
                print(field)
            results_summary.append((test_name, "FAIL", len(empty_fields)))
            all_passed = False
        else:
            print("  STATUS: [PASS] - All fields populated")
            print(f"  Estimated Price: {result_dict['estimated_price']:,} TND")
            print(f"  Confidence: {result_dict['confidence']}% ({result_dict['confidence_level']})")
            print(f"  Comparables: {len(result_dict['comparables'])} properties found")
            results_summary.append((test_name, "PASS", 0))

    except Exception as e:
        print(f"  STATUS: [ERROR] - {str(e)}")
        results_summary.append((test_name, "ERROR", str(e)))
        all_passed = False

    print()

# Summary report
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
for test_name, status, issues in results_summary:
    status_icon = "[PASS]" if status == "PASS" else f"[{status}]"
    if status == "PASS":
        print(f"  {status_icon} {test_name}")
    else:
        issue_info = f" ({issues} issues)" if isinstance(issues, int) else f" ({issues})"
        print(f"  {status_icon} {test_name}{issue_info}")

print()
passed = sum(1 for _, s, _ in results_summary if s == "PASS")
total = len(results_summary)
print(f"RESULT: {passed}/{total} tests passed")

if all_passed:
    print("[SUCCESS] All tests passed - API output is complete!")
    sys.exit(0)
else:
    print("[FAILURE] Some tests failed")
    sys.exit(1)
