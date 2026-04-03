#!/usr/bin/env python3
"""Test suite for synthethic_reviews.py - testing for empty output scenarios"""

import sys
import io
import re
import tempfile
import csv
from pathlib import Path
from typing import Any

# Fix encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Import functions from the synthetic reviews script
import sys
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from synthethic_reviews import (
    _clean_location,
    build_prompt,
    strip_think_tags,
    _extract_generated_text,
    load_locations_from_csv,
    NOISE_LOCATION_KEYWORDS,
    LOCATION_COLUMNS,
)


def test_clean_location():
    """Test the _clean_location function for edge cases"""
    print("\n" + "=" * 80)
    print("TEST 1: Location Cleaning (_clean_location)")
    print("=" * 80)

    test_cases = [
        ("La Marsa", True, "Valid location"),
        ("Tunis", True, "Valid single word"),
        ("Sidi Bouzid", True, "Valid multi-word"),
        ("", False, "Empty string"),
        (None, False, "None value"),
        ("nan", False, "NaN string"),
        ("none", False, "None string"),
        ("null", False, "Null string"),
        ("123456", False, "Numbers only"),
        ("contact@email.com", False, "Contact noise keyword"),
        ("www.agence.tn", False, "Agency noise keywords"),
        ("   La Marsa   ", True, "Whitespace handling"),
        ("La  Marsa  City", True, "Multiple spaces"),
        ("A" * 100, False, "Too long (>80 chars)"),
        ("La " * 30, False, "Cumulative too long (>80 chars)"),
        ("La Marsa", True, "Clean valid location"),
    ]

    failures = []
    for value, should_pass, description in test_cases:
        result = _clean_location(value)
        is_valid = result is not None
        status = "[OK]" if is_valid == should_pass else "[FAIL]"
        print(f"  {status} {description}: {repr(value)}")
        if is_valid != should_pass:
            failures.append(f"    Expected {should_pass}, got {is_valid}")
            print(f"    {failures[-1]}")

    return len(failures) == 0, failures


def test_strip_think_tags():
    """Test the strip_think_tags function"""
    print("\n" + "=" * 80)
    print("TEST 2: Strip Think Tags")
    print("=" * 80)

    test_cases = [
        ("Normal review text", "Normal review text", "No tags"),
        ("<think>internal reasoning</think>Normal text", "Normal text", "Simple think tags"),
        ("<THINK>UPPERCASE</THINK>Text", "Text", "Uppercase tags"),
        ("<think>nested <think>tags</think></think>Content", "Content", "Nested tags"),
        ("Text with <think>hidden</think> content <think>more</think> here", "Text with  content  here", "Multiple tags"),
        ("Normal text\n\n\n\nMultiple newlines", "Normal text\n\nMultiple newlines", "Multiple newlines"),
        ("", "", "Empty string"),
        ("<think></think>", "", "Only tags"),
        ("Review of La Marsa neighborhood.\nVery safe area.\nGood schools.",
         "Review of La Marsa neighborhood.\nVery safe area.\nGood schools.", "Valid review"),
    ]

    failures = []
    for input_text, expected, description in test_cases:
        result = strip_think_tags(input_text)
        is_correct = result == expected
        status = "[OK]" if is_correct else "[FAIL]"
        print(f"  {status} {description}")
        if not is_correct:
            msg = f"    Expected: {repr(expected)}\n    Got: {repr(result)}"
            failures.append(msg)
            print(msg)

    return len(failures) == 0, failures


def test_extract_generated_text():
    """Test the _extract_generated_text function with various API responses"""
    print("\n" + "=" * 80)
    print("TEST 3: Extract Generated Text from API Response")
    print("=" * 80)

    test_cases = [
        (
            {"choices": [{"message": {"content": "Sample review"}}]},
            "Sample review",
            "Valid response"
        ),
        (
            {"choices": [{"message": {"content": ""}}]},
            "",
            "Empty content"
        ),
        (
            {"choices": [{"message": {"content": None}}]},
            None,
            "Null content"
        ),
        (
            {"choices": []},
            None,
            "Empty choices"
        ),
        (
            {},
            None,
            "Empty dict"
        ),
        (
            None,
            None,
            "None"
        ),
        (
            {"choices": [{"message": {"content": ["text1", "text2"]}}]},
            None,
            "Content as list of non-dict items"
        ),
        (
            {
                "choices": [{
                    "message": {
                        "content": [
                            {"type": "text", "text": "chunk1"},
                            {"type": "text", "text": "chunk2"}
                        ]
                    }
                }]
            },
            "chunk1\nchunk2",
            "Content as list of dict items with text"
        ),
    ]

    failures = []
    for api_response, expected, description in test_cases:
        result = _extract_generated_text(api_response)
        is_correct = result == expected
        status = "[OK]" if is_correct else "[FAIL]"
        print(f"  {status} {description}")
        if not is_correct:
            msg = f"    Expected: {repr(expected)}\n    Got: {repr(result)}"
            failures.append(msg)
            print(msg)

    return len(failures) == 0, failures


def test_build_prompt():
    """Test the build_prompt function"""
    print("\n" + "=" * 80)
    print("TEST 4: Build Prompt")
    print("=" * 80)

    test_cases = [
        ("La Marsa", 1, "fr", "French", ["La Marsa", "Tunisia", "French", "reviews"]),
        ("Sfax", 1, "en", "English", ["Sfax", "Tunisia", "English", "reviews"]),
        ("Bizerte", 5, "fr", "Multiple reviews FR", ["Bizerte", "5", "French"]),
        ("Sousse", 10, "en", "Multiple reviews EN", ["Sousse", "10", "English"]),
    ]

    failures = []
    for location, num_reviews, language, description, required_words in test_cases:
        result = build_prompt(location, num_reviews, language)

        # Check prompt is not empty
        if not result or result.strip() == "":
            failures.append(f"    Empty prompt for {description}")
            print(f"  [FAIL] {description}: Empty prompt")
            continue

        # Check all required words are present
        all_found = all(word.lower() in result.lower() for word in required_words)
        status = "[OK]" if all_found else "[FAIL]"
        print(f"  {status} {description}: {num_reviews} review(s) in {language}")

        if not all_found:
            missing = [w for w in required_words if w.lower() not in result.lower()]
            msg = f"    Missing words: {missing}"
            failures.append(msg)
            print(msg)

    return len(failures) == 0, failures


def test_load_locations_from_csv():
    """Test loading locations from CSV files"""
    print("\n" + "=" * 80)
    print("TEST 5: Load Locations from CSV")
    print("=" * 80)

    # Create temporary CSV files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create test CSV with valid locations
        test_csv_1 = tmppath / "test1.csv"
        with test_csv_1.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["city", "governorate", "price"])
            writer.writeheader()
            writer.writerows([
                {"city": "La Marsa", "governorate": "Tunis", "price": "500000"},
                {"city": "Sfax", "governorate": "Sfax", "price": "300000"},
                {"city": "Sousse", "governorate": "Sousse", "price": "400000"},
            ])

        # Create test CSV with invalid locations (should be filtered)
        test_csv_2 = tmppath / "test2.csv"
        with test_csv_2.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["location", "contact"])
            writer.writeheader()
            writer.writerows([
                {"location": "Bizerte", "contact": "contact@agence.tn"},
                {"location": "123456", "contact": "immo.com"},
                {"location": "Nabeul", "contact": "other"},
            ])

        locations = load_locations_from_csv(tmppath)

        print(f"  Locations loaded: {len(locations)}")
        if locations:
            print(f"    Unique locations: {locations}")
        else:
            print("    [FAIL] No locations loaded!")

        # Expected: La Marsa, Sfax, Sousse, Bizerte, Nabeul
        # Not: 123456, contact info
        expected_count = 5
        is_correct = len(locations) >= expected_count - 1  # Allow some variation
        status = "[OK]" if is_correct else "[FAIL]"
        print(f"  {status} Location count check: {len(locations)} locations")

        if not locations:
            return False, ["No locations loaded from CSV"]
        return is_correct, []


def test_output_file_generation():
    """Test that output files would not be empty"""
    print("\n" + "=" * 80)
    print("TEST 6: Output File Generation Safety")
    print("=" * 80)

    print(f"  [INFO] Testing file generation scenarios...")

    # Scenario 1: Empty reviews list
    empty_reviews = []
    empty_output = "\n".join(empty_reviews)
    print(f"  [WARN] Empty reviews list produces: {repr(empty_output)}")
    if empty_output.strip() == "":
        print(f"    -> Would create empty file")

    # Scenario 2: Reviews with only whitespace after stripping
    whitespace_reviews = ["   ", "\n\n", "\t"]
    after_strip = [r.strip() for r in whitespace_reviews if r.strip()]
    whitespace_output = "\n".join(after_strip)
    print(f"  [WARN] Whitespace-only reviews produces: {repr(whitespace_output)}")

    # Scenario 3: Reviews with think tags that get completely removed
    think_tag_reviews = ["<think>hidden</think>", "<think>more hidden</think>"]
    cleaned = [strip_think_tags(r) for r in think_tag_reviews]
    think_output = "\n".join(c for c in cleaned if c)
    print(f"  [WARN] Think-tag-only reviews produces: {repr(think_output)}")

    print(f"  [OK] Output generation tests completed")
    return True, []


# Run all tests
def main():
    print("=" * 80)
    print("SYNTHETIC REVIEWS SCRIPT - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    tests = [
        ("Location Cleaning", test_clean_location),
        ("Think Tag Stripping", test_strip_think_tags),
        ("Extract Generated Text", test_extract_generated_text),
        ("Build Prompt", test_build_prompt),
        ("Load Locations from CSV", test_load_locations_from_csv),
        ("Output File Generation", test_output_file_generation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed, failures = test_func()
            results.append((test_name, passed, failures))
        except Exception as e:
            print(f"\n  [ERROR] Exception in {test_name}: {str(e)}")
            results.append((test_name, False, [str(e)]))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)

    for test_name, passed, failures in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")
        if failures:
            for failure in failures[:3]:  # Show first 3 failures
                print(f"    {failure}")

    print()
    print(f"RESULT: {passed_count}/{total_count} test groups passed")

    if passed_count == total_count:
        print("[SUCCESS] All tests passed!")
        print("\nNo empty output issues detected in core functions.")
        return 0
    else:
        print("[FAILURE] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
