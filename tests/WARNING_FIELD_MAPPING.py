"""
WARNING FIELD MAPPING & FIXES
==============================
Based on actual diagnostic test execution showing which request fields trigger each uncertainty reason.

SCENARIO ANALYSIS
=================

SCENARIO 1: Known City (Tunis/Tunis)
  INPUT: governorate='Tunis', city='Tunis'
  WARNINGS: (none)
  PRICE: 196,249 TND | CONFIDENCE: 67%
  STATUS: HEALTHY - City is in geo_lookup with complete data coverage

SCENARIO 2: Unknown City (Unknown_Gov / Unknown_City_XYZ_12345)
  INPUT: governorate='Unknown_Gov', city='Unknown_City_XYZ_12345'
  WARNINGS:
    [1] city_normalization_applied     - City name was normalized (cleaned)
    [2] geo_lookup_missing              - City NOT in geo_lookup; no coordinates available
    [3] gov_price_prior_fallback        - No governorate-level price prior found
    [4] governorate_normalization_applied - Governorate name was normalized
    [5] local_price_prior_fallback_global - Fell back to GLOBAL price (worst case)
    [6] ood:governorate_price_prior_data_coverage - Gov price prior is weak/missing
    [7] ood:local_price_prior_governorate_data_coverage - Governorate coverage weak
    [8] ood:unknown_city_data_coverage  - City completely unknown to system
    [9] sentiment_mode:neutral_fallback - No sentiment data; used neutral default
  PRICE: 599,781 TND | CONFIDENCE: 50%
  STATUS: CRITICAL - Multiple data failures; fell back to global prior

SCENARIO 3: Known City, Small Governorate (Sfax/Sfax)
  INPUT: governorate='Sfax', city='Sfax'
  WARNINGS:
    [1] sentiment_mode:neutral_fallback - No sentiment data; used neutral default
  PRICE: 195,684 TND | CONFIDENCE: 59%
  STATUS: ACCEPTABLE - Core valuation works; only sentiment data missing

SCENARIO 4: Empty City (Tunis / '')
  INPUT: governorate='Tunis', city=''
  WARNINGS:
    [1] geo_lookup_missing              - No city to look up; coordinates unavailable
    [2] low_input_completeness          - Required field (city) is empty
    [3] ood:unknown_city_missing_input  - City field missing/empty in request
    [4] sentiment_mode:neutral_fallback - No sentiment data
  PRICE: 242,909 TND | CONFIDENCE: 52%
  STATUS: DEGRADED - Required field missing; fell back to governorate


SCENARIO 5: Small Governorate (Kebili/Kebili)
  INPUT: governorate='Kebili', city='Kebili'
  WARNINGS:
    [1] gov_price_prior_fallback        - Fell back to governorate from city-level
    [2] local_price_prior_fallback_city - City-level prior unavailable
    [3] ood:governorate_price_prior_data_coverage - Gov data coverage weak
    [4] ood:local_price_prior_city_governorate_data_coverage - City+gov coverage weak
    [5] sentiment_mode:neutral_fallback - No sentiment data
  PRICE: 106,588 TND | CONFIDENCE: 53%
  STATUS: DEGRADED - Rural area with sparse data coverage


EXACT FIELD-TO-WARNING MAPPING
===============================

FIELD: governorate
  TRIGGER 1: Empty/null + city provided
    WARNING: low_input_completeness (if city also empty)
    FIX: Validate governorate in API; enforce known governorate list
  
  TRIGGER 2: Unknown governorate (not in gov_avg_price_m2)
    WARNING: ood:governorate_price_prior_data_coverage
    WARNING: local_price_prior_fallback_global
    FIX: Maintain reference of all valid Tunisian governorates
  
  TRIGGER 3: Small/rural governorate (Kebili, Tataouine, Medenine)
    WARNING: ood:governorate_price_prior_data_coverage
    WARNING: ood:local_price_prior_city_governorate_data_coverage
    FIX: Collect more listings in rural areas OR use regional fallback

FIELD: city
  TRIGGER 1: Empty/missing city ('')
    WARNING: geo_lookup_missing
    WARNING: ood:unknown_city_missing_input
    WARNING: low_input_completeness
    FIX: Make city field required; fail fast on validation

  TRIGGER 2: Unknown city (not in city_geo_lookup)
    WARNING: geo_lookup_missing
    WARNING: ood:unknown_city_data_coverage
    FIX: Build city geo_lookup from reference data; add fallback centroids

  TRIGGER 3: City mismatch with governorate (e.g., 'Tunis' in 'Sfax')
    WARNING: (none directly; price just becomes inaccurate)
    FIX: Validate city belongs to stated governorate

  TRIGGER 4: Weak city-level price history
    WARNING: local_price_prior_fallback_city
    WARNING: ood:local_price_prior_city_governorate_data_coverage
    FIX: Aggregate more transactions per city in training data

FIELD: latitude/longitude (derived, not in request)
  TRIGGER: Missing + city not in geo_lookup
    WARNING: geo_lookup_missing (appears twice in code path)
    FIX: Add governorate centroids as fallback coordinates


PROPOSED FIXES (Prioritized by Impact)
=======================================

[PRIORITY 1] Fix Empty City Validation
  FILE: src/api.py (PropertyRequest class)
  ACTION: Make city field non-empty string (e.g., Field(min_length=1))
  IMPACT: Prevent low-value requests; fail fast on bad input
  DIFFICULTY: Easy
  CODE CHANGE:
    city: str = Field(min_length=1, description="Required city name")
  RESULT: Eliminates geo_lookup_missing + ood:unknown_city_missing_input for empty city

[PRIORITY 2] Add Governorate Centroid Fallback
  FILE: src/inference/inference_bundle.py (~line 257)
  ACTION: When city not in city_geo_lookup, use governorate median coordinates
  IMPACT: geo_lookup_missing only appears once; provides better distance features
  DIFFICULTY: Medium
  CODE CHANGE:
    if not coords:
        # Use governorate median as fallback
        coords = self.gov_centroids.get(gov_key, (36.8, 10.2))  # Tunisia median
  RESULT: Reduces failures for small governorates; improves feature quality

[PRIORITY 3] Expand city_geo_lookup
  FILE: Reference dataset / src/inference/inference_bundle.py (~line 146)
  ACTION: Add missing cities from reference data; normalize city names
  IMPACT: Eliminate ood:unknown_city_data_coverage for known cities
  DIFFICULTY: Medium (data-driven)
  PROCESS:
    1. Extract unique (city, lat, lon) from reference_df
    2. Build complete city_geo_lookup from all known cities
    3. Add alias mapping for common misspellings
  RESULT: geo_lookup_missing only for truly unknown cities (not data bugs)

[PRIORITY 4] Improve City-Level Price Priors
  FILE: src/inference/inference_bundle.py (~line 120-130)
  ACTION: Require minimum transaction count per city before using city-level prior
  IMPACT: Reduce false fallback to governorate for sparse cities
  DIFFICULTY: Medium (requires retraining)
  CONFIG:
    MIN_CITY_TRANSACTIONS = 10  # Only use city prior if >= 10 transactions
  RESULT: local_price_prior_fallback_city only for genuinely sparse cities

[PRIORITY 5] Validate Governorate Against Known List
  FILE: src/api.py (PropertyRequest class or custom validator)
  ACTION: Only accept known Tunisian governorate names
  IMPACT: Prevent normalization cascades for bad governorate input
  DIFFICULTY: Easy
  CODE CHANGE:
    governorate: str = Field(pattern="^(Tunis|Ariana|Manouba|Ben Arous|Nabeul|...)$")
  RESULT: Eliminates governorate_normalization_applied for unknown governorates

[PRIORITY 6] Collect More Data for Rural Regions
  FILE: (Business/Data Collection)
  ACTION: Actively acquire listings from Kebili, Tataouine, Medenine, Gafsa
  IMPACT: Reduce ood:governorate_price_prior_data_coverage for rural areas
  DIFFICULTY: Hard (external)
  TIMELINE: Long-term
  RESULT: Smaller uncertainty bands for rural valuations


IMPLEMENTATION ROADMAP
======================

PHASE 1 (Immediate - 1-2 hours):
  [1] Add city field validation (min_length=1) to API
  [2] Add governorate centroid fallback in inference_bundle.py
  [3] Test and verify reduction in warnings

PHASE 2 (Short-term - 1-2 days):
  [4] Expand city_geo_lookup from reference data
  [5] Add governorate whitelist validation
  [6] Retrain inference models with improved lookups

PHASE 3 (Medium-term - 1-2 weeks):
  [7] Implement city-prior minimum threshold logic
  [8] Add monitoring dashboard for warning frequency

PHASE 4 (Long-term - Ongoing):
  [9] Collect more data for rural governorates
  [10] Monitor and adjust thresholds based on real valuation outcomes

EXPECTED OUTCOMES (After Phase 2)
=================================

Scenario 2 (Unknown City):
  BEFORE: 9 warnings (critical failure)
  AFTER: ~3-4 warnings (degraded but functional, uses governorate data)

Scenario 4 (Empty City):
  BEFORE: 4 warnings (request processing failure)
  AFTER: 0 warnings (rejected at API boundary)

Scenario 5 (Small Governorate):
  BEFORE: 5 warnings (degraded confidence)
  AFTER: 1-2 warnings (acceptable with centroid fallback)

Scenario 1 (Known City):
  BEFORE: 0 warnings (healthy)
  AFTER: 0 warnings (unchanged)
"""

print(__doc__)
