"""Fast what-if scenario simulation for valuation explainability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


AMENITY_LABELS = {
    "has_pool": "swimming pool",
    "has_garden": "garden",
    "has_parking": "parking",
    "sea_view": "sea view",
    "elevator": "elevator",
}

CONDITION_ORDER = ["Needs Renovation", "Fair", "Good", "Excellent", "New"]


@dataclass(slots=True)
class ScenarioCard:
    """Normalized output for a simulated scenario."""

    scenario_name: str
    scenario_description: str
    modified_features: dict[str, Any]
    predicted_price: int
    price_delta: int
    delta_percentage: float
    confidence: float
    why: str


class ScenarioSimulator:
    """Build fast, feasible what-if scenarios from the current valuation state."""

    def _confidence_from_inputs(self, confidence: dict[str, Any] | None) -> float:
        base = float(confidence.get("confidence", 60) if confidence else 60) / 100.0
        return max(0.35, min(0.95, base * 0.92 + 0.05))

    @staticmethod
    def _percent_delta(base: int, delta: int) -> float:
        if base <= 0:
            return 0.0
        return round((delta / base) * 100.0, 2)

    @staticmethod
    def _is_feasible_amenity(mapped: dict[str, Any], field: str, estimated_price: int, market_context: dict[str, Any]) -> bool:
        property_type = str(mapped.get("property_type", "")).strip().lower()
        if property_type == "terrain":
            return False
        if field == "has_pool":
            minimum = max(int(market_context.get("avg_m2", 0) or 0) * 120, 220000)
            return estimated_price >= minimum
        return True

    def _scenario(
        self,
        *,
        name: str,
        description: str,
        modified_features: dict[str, Any],
        price_delta: int,
        estimated_price: int,
        confidence: float,
        why: str,
    ) -> ScenarioCard:
        predicted_price = max(1, int(round(estimated_price + price_delta)))
        return ScenarioCard(
            scenario_name=name,
            scenario_description=description,
            modified_features=modified_features,
            predicted_price=predicted_price,
            price_delta=int(price_delta),
            delta_percentage=self._percent_delta(estimated_price, int(price_delta)),
            confidence=round(confidence, 2),
            why=why,
        )

    def _amenity_scenarios(
        self,
        mapped: dict[str, Any],
        estimated_price: int,
        market_context: dict[str, Any],
        confidence: float,
        comparables: list[dict[str, Any]],
    ) -> list[ScenarioCard]:
        scenarios: list[ScenarioCard] = []
        similar_count = len(comparables)
        amenity_weights = {
            "has_pool": 0.11 if str(mapped.get("property_type", "")).strip().lower() in {"maison", "villa"} else 0.08,
            "has_garden": 0.05,
            "has_parking": 0.035,
            "sea_view": 0.09,
            "elevator": 0.025,
        }

        for field, weight in amenity_weights.items():
            if bool(mapped.get(field)):
                continue
            if not self._is_feasible_amenity(mapped, field, estimated_price, market_context):
                continue
            label = AMENITY_LABELS[field]
            boost = int(round(estimated_price * weight))
            if field == "has_pool" and estimated_price < 300000:
                boost = int(round(boost * 0.85))
            why = (
                f"Similar listings in this market scope price better when they include {label}. "
                f"Comparable support: {similar_count} listing(s)."
            )
            scenarios.append(
                self._scenario(
                    name=f"Add {label}",
                    description=f"Simulate adding a {label}.",
                    modified_features={field: True},
                    price_delta=boost,
                    estimated_price=estimated_price,
                    confidence=confidence,
                    why=why,
                )
            )
        return scenarios

    def _condition_scenario(
        self,
        mapped: dict[str, Any],
        estimated_price: int,
        confidence: float,
        market_context: dict[str, Any],
    ) -> ScenarioCard | None:
        current = str(mapped.get("condition", "Good")).strip()
        if current not in CONDITION_ORDER:
            current = "Good"
        current_index = CONDITION_ORDER.index(current)
        if current_index >= len(CONDITION_ORDER) - 1:
            return None
        upgraded = CONDITION_ORDER[current_index + 1]
        boost = int(round(estimated_price * 0.045))
        why = (
            f"Improving condition usually lifts buyer willingness and aligns the listing with stronger comparables. "
            f"Market scope median: {int(market_context.get('avg_m2', 0) or 0):,} TND/m2."
        )
        return self._scenario(
            name=f"Improve condition to {upgraded}",
            description=f"Simulate upgrading the property condition from {current} to {upgraded}.",
            modified_features={"condition": upgraded},
            price_delta=boost,
            estimated_price=estimated_price,
            confidence=confidence,
            why=why,
        )

    def _description_scenario(self, estimated_price: int, confidence: float, description_analysis: dict[str, Any]) -> ScenarioCard:
        score = float(description_analysis.get("description_score", 0.5))
        target = 0.82
        lift = max(target - score, 0.0)
        boost = int(round(estimated_price * lift * 0.12))
        return self._scenario(
            name="Improve description quality",
            description="Simulate a richer, clearer description with stronger marketing language.",
            modified_features={"description_score": round(target, 2)},
            price_delta=boost,
            estimated_price=estimated_price,
            confidence=confidence,
            why="Clearer descriptions usually improve engagement and support a stronger buyer perception.",
        )

    def _location_scenario(
        self,
        mapped: dict[str, Any],
        estimated_price: int,
        confidence: float,
        location_sentiment: dict[str, Any],
        market_context: dict[str, Any],
    ) -> ScenarioCard | None:
        """Generate location improvement scenario only if market trend data is available.
        
        This ensures we only propose relocation when we have evidence (trend, comparable locations)
        to substantiate the claim that a stronger area would improve value.
        """
        # Gate on market_context.trend and demand availability.
        trend = market_context.get("trend")
        demand = market_context.get("demand")
        if not trend or not demand:
            return None  # Insufficient market context to propose location improvement.
        
        sentiment = float(location_sentiment.get("sentiment", 0.5))
        target = max(sentiment + 0.12, 0.72)
        boost = int(round(estimated_price * max(target - sentiment, 0.0) * 0.1))
        why = (
            f"Properties in stronger-demand neighborhoods generally command higher values. "
            f"The current location sentiment is {sentiment:.2f} and the market trend is {str(trend).lower()}."
        )
        return self._scenario(
            name="Improve location profile",
            description="Simulate moving the listing into a stronger nearby area.",
            modified_features={
                "city": mapped.get("city"),
                "neighborhood": mapped.get("neighborhood"),
                "location_sentiment_target": round(target, 2),
            },
            price_delta=boost,
            estimated_price=estimated_price,
            confidence=confidence,
            why=why,
        )

    def generate(
        self,
        mapped: dict[str, Any],
        estimated_price: int,
        market_context: dict[str, Any],
        features_impact: list[dict[str, Any]],
        comparables: list[dict[str, Any]],
        confidence: dict[str, Any] | None,
        description_analysis: dict[str, Any],
        description_sentiment: dict[str, Any],
        location_sentiment: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Generate ranked scenario cards and recommendation summaries."""

        base_confidence = self._confidence_from_inputs(confidence)
        scenarios: list[ScenarioCard] = []
        scenarios.extend(self._amenity_scenarios(mapped, estimated_price, market_context, base_confidence, comparables))
        condition = self._condition_scenario(mapped, estimated_price, base_confidence, market_context)
        if condition is not None:
            scenarios.append(condition)
        scenarios.append(self._description_scenario(estimated_price, base_confidence, description_analysis))
        # Note: Transaction Type scenario is logically inconsistent (why suggest changing sale to rent?).
        # Intentionally excluded from scenario generation.
        location = self._location_scenario(mapped, estimated_price, base_confidence, location_sentiment, market_context)
        if location is not None:
            scenarios.append(location)

        if features_impact:
            strongest = max(features_impact, key=lambda item: abs(float(item.get("amount", 0) or 0)))
            strongest_amount = float(strongest.get("amount", 0) or 0)
            if strongest_amount != 0:
                label = str(strongest.get("feature", "market driver")).replace("_", " ").title()
                scenarios.append(
                    self._scenario(
                        name=f"Reinforce {label}",
                        description="Double down on the strongest existing market driver.",
                        modified_features={"focus_feature": strongest.get("feature")},
                        price_delta=int(round(strongest_amount * 0.25)),
                        estimated_price=estimated_price,
                        confidence=base_confidence,
                        why=f"{label} is already a major driver, so improving it gives a fast upside scenario.",
                    )
                )

        ordered = sorted(scenarios, key=lambda item: (item.price_delta, item.confidence), reverse=True)
        scenario_rows = [
            {
                "scenario_name": item.scenario_name,
                "scenario_description": item.scenario_description,
                "modified_features": item.modified_features,
                "predicted_price": item.predicted_price,
                "price_delta": item.price_delta,
                "delta_percentage": item.delta_percentage,
                "confidence": item.confidence,
                "why": item.why,
            }
            for item in ordered[:5]
        ]
        recommendations = [
            {
                "title": row["scenario_name"],
                "description": row["scenario_description"],
                "predicted_impact_tnd": row["price_delta"],
                "predicted_impact_pct": row["delta_percentage"],
                "confidence": row["confidence"],
                "justification": row["why"],
            }
            for row in scenario_rows
        ]
        return scenario_rows, recommendations