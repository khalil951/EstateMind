from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd


LABELS = ["negative", "neutral", "positive"]
LANGUAGES = ["en", "fr"]

ENGLISH_BANK: dict[str, dict[str, list[str]]] = {
    "positive": {
        "subject": [
            "Rental demand stays resilient",
            "Buyer interest remains active",
            "The neighborhood keeps attracting families",
            "Investor appetite remains healthy",
            "Occupancy risk stays limited",
            "Resale activity remains fluid",
        ],
        "market": [
            "because price signals are still supportive",
            "thanks to steady local demand",
            "with stable absorption across recent listings",
            "as comparable assets continue to move well",
            "while resale timelines remain acceptable",
            "given the area keeps drawing qualified buyers",
        ],
        "infrastructure": [
            "Road access is dependable and daily services are easy to reach",
            "Utilities are reliable and the street network is well connected",
            "Schools, shops, and transport options are easy to access",
            "Public services and commerce create a practical living environment",
            "Core amenities support convenient day to day use",
            "Infrastructure quality supports regular residential demand",
        ],
        "investment": [
            "That combination improves long term resale confidence",
            "This setup supports durable investment appeal",
            "These signals strengthen the asset's marketability",
            "That profile makes the property easier to position",
            "This context supports dependable value retention",
            "These conditions help preserve liquidity over time",
        ],
    },
    "neutral": {
        "subject": [
            "Demand is present but not especially strong",
            "The market looks balanced rather than heated",
            "Buyer activity is steady without clear acceleration",
            "The neighborhood performs in line with nearby areas",
            "Pricing appears aligned with current conditions",
            "The area shows moderate market consistency",
        ],
        "market": [
            "with no major pricing pressure in either direction",
            "and recent listings point to limited momentum",
            "while transaction conditions remain fairly ordinary",
            "and valuation signals are mostly mixed",
            "with neither scarcity nor oversupply dominating",
            "while purchase interest stays selective",
        ],
        "infrastructure": [
            "Amenities cover basic needs but do not stand out",
            "Roads and services are functional without being exceptional",
            "Transport and schools are adequate for routine use",
            "Infrastructure is serviceable though some upgrades would help",
            "Commercial access is acceptable but not a clear advantage",
            "The environment is workable for standard residential demand",
        ],
        "investment": [
            "Overall the asset looks manageable but not compelling",
            "This points to an ordinary investment profile",
            "The property may suit cautious buyers seeking stability",
            "That leaves the opportunity in a middle range position",
            "The outlook is reasonable without being especially attractive",
            "This supports a wait and compare decision more than urgency",
        ],
    },
    "negative": {
        "subject": [
            "Buyer confidence appears fragile",
            "The neighborhood raises several investment concerns",
            "Market perception remains cautious",
            "Resale confidence is under pressure",
            "Demand quality looks inconsistent",
            "The area shows multiple warning signs",
        ],
        "market": [
            "because comparable pricing is difficult to justify",
            "as recent listings suggest weak conversion",
            "while local demand appears hesitant",
            "given the market lacks clear momentum",
            "because transaction visibility remains limited",
            "as valuation signals stay hard to trust",
        ],
        "infrastructure": [
            "Administrative friction and legal opacity add execution risk",
            "Infrastructure gaps reduce everyday convenience",
            "Road quality and service reliability remain below expectation",
            "Weak amenity access hurts residential appeal",
            "Transport limitations lower the neighborhood's competitiveness",
            "Utility uncertainty makes the asset harder to defend",
        ],
        "investment": [
            "That combination weakens long term resale confidence",
            "This setup increases downside risk for investors",
            "These conditions reduce the property's liquidity",
            "That profile makes the asset difficult to position",
            "This context undermines pricing power",
            "These weaknesses can drag on exit value",
        ],
    },
}

FRENCH_BANK: dict[str, dict[str, list[str]]] = {
    "positive": {
        "subject": [
            "La demande locative reste solide",
            "L'interet des acheteurs demeure actif",
            "Le quartier attire encore les familles",
            "L'appetit des investisseurs reste sain",
            "Le risque de vacance reste limite",
            "La revente conserve une bonne fluidite",
        ],
        "market": [
            "car les signaux de prix restent favorables",
            "grace a une demande locale reguliere",
            "avec une absorption stable des annonces recentes",
            "puisque les biens comparables se vendent correctement",
            "alors que les delais de revente restent acceptables",
            "dans un contexte ou les acheteurs qualifies sont presents",
        ],
        "infrastructure": [
            "Les acces routiers sont fiables et les services du quotidien sont proches",
            "Les reseaux sont stables et la desserte du secteur est pratique",
            "Les ecoles, commerces et transports sont faciles a rejoindre",
            "Les services publics et commerces soutiennent un usage residentiel confortable",
            "Les equipements de base rendent la vie quotidienne simple",
            "La qualite des infrastructures soutient une demande residentielle reguliere",
        ],
        "investment": [
            "Cet ensemble renforce la confiance sur la revente a long terme",
            "Cette configuration soutient un potentiel d'investissement durable",
            "Ces signaux ameliorent la commercialisation du bien",
            "Ce profil rend l'actif plus facile a positionner",
            "Ce contexte soutient une bonne conservation de valeur",
            "Ces conditions aident a maintenir la liquidite dans le temps",
        ],
    },
    "neutral": {
        "subject": [
            "La demande est presente sans etre particulierement forte",
            "Le marche parait equilibre plutot que dynamique",
            "L'activite des acheteurs reste stable sans acceleration nette",
            "Le quartier evolue dans la moyenne des zones voisines",
            "Les prix semblent alignes sur les conditions actuelles",
            "La zone montre une regularite de marche moderee",
        ],
        "market": [
            "sans veritable pression de prix dans un sens ou dans l'autre",
            "et les annonces recentes montrent peu d'elan",
            "alors que les conditions de transaction restent ordinaires",
            "avec des signaux de valorisation assez mitiges",
            "sans rarete ni suroffre dominante",
            "pendant que l'interet d'achat reste selectif",
        ],
        "infrastructure": [
            "Les commodites couvrent les besoins de base sans se distinguer",
            "Les routes et services sont fonctionnels sans etre remarquables",
            "Les transports et ecoles sont adequats pour un usage courant",
            "Les infrastructures restent correctes meme si des ameliorations seraient utiles",
            "L'acces commercial est acceptable sans avantage clair",
            "L'environnement convient a une demande residentielle standard",
        ],
        "investment": [
            "Au final le bien parait gerable sans etre tres attractif",
            "Cela decrit un profil d'investissement plutot ordinaire",
            "Le bien peut convenir a un acheteur prudent qui cherche de la stabilite",
            "Cette situation place l'opportunite dans une zone intermediaire",
            "La perspective reste correcte sans reel atout marquant",
            "Ce contexte invite davantage a comparer qu'a se precipiter",
        ],
    },
    "negative": {
        "subject": [
            "La confiance des acheteurs parait fragile",
            "Le quartier souleve plusieurs risques d'investissement",
            "La perception du marche reste prudente",
            "La confiance dans la revente est sous pression",
            "La qualite de la demande semble irreguliere",
            "La zone presente plusieurs signaux d'alerte",
        ],
        "market": [
            "car les prix comparables sont difficiles a justifier",
            "puisque les annonces recentes suggerent une conversion faible",
            "alors que la demande locale parait hesitante",
            "dans un marche qui manque de momentum clair",
            "car la visibilite sur les transactions reste limitee",
            "puisque les signaux de valorisation restent peu fiables",
        ],
        "infrastructure": [
            "Les lenteurs administratives et l'opacite juridique augmentent le risque",
            "Les insuffisances d'infrastructure reduisent le confort quotidien",
            "La qualite des routes et des services reste en dessous des attentes",
            "Le faible acces aux amenites nuit a l'attractivite residentielle",
            "Les limites de transport affaiblissent la competitivite du secteur",
            "L'incertitude sur les services rend le bien plus difficile a defendre",
        ],
        "investment": [
            "Cet ensemble affaiblit la confiance sur la revente a long terme",
            "Cette configuration augmente le risque baissier pour l'investisseur",
            "Ces conditions reduisent la liquidite du bien",
            "Ce profil rend l'actif difficile a positionner",
            "Ce contexte deteriore le pouvoir de prix",
            "Ces faiblesses peuvent peser sur la valeur de sortie",
        ],
    },
}


def _template_text(language: str, subject: str, market: str, infrastructure: str, investment: str, idx: int) -> str:
    infrastructure_lower = infrastructure[:1].lower() + infrastructure[1:] if infrastructure else infrastructure
    if language == "fr":
        patterns = [
            f"{subject} {market}. {infrastructure}. {investment}.",
            f"{subject} {market}, et {infrastructure_lower}. {investment}.",
        ]
    else:
        patterns = [
            f"{subject} {market}. {infrastructure}. {investment}.",
            f"{subject} {market}, and {infrastructure_lower}. {investment}.",
        ]
    return patterns[idx % len(patterns)]


def _collect_combinations(language: str, label: str) -> list[str]:
    bank = ENGLISH_BANK if language == "en" else FRENCH_BANK
    rows: list[str] = []
    slots = bank[label]
    for idx, combo in enumerate(
        product(
            slots["subject"],
            slots["market"],
            slots["infrastructure"],
            slots["investment"],
        )
    ):
        text = _template_text(language, combo[0], combo[1], combo[2], combo[3], idx)
        rows.append(text)
    return rows


def synthesize_sentiment_metadata(
    target_groups_per_label: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    del seed  # deterministic ordering is enough here

    rows: list[dict[str, Any]] = []
    review_id = 1
    per_label_language_target = max(1, (target_groups_per_label + len(LANGUAGES) - 1) // len(LANGUAGES))

    for label in LABELS:
        for language in LANGUAGES:
            candidates = _collect_combinations(language, label)
            selected = candidates[:per_label_language_target]
            for text in selected:
                rows.append(
                    {
                        "review_id": review_id,
                        "city": "",
                        "language": "English" if language == "en" else "French",
                        "sentiment": label.capitalize(),
                        "source": "synthetic_template_generator",
                        "label_source": "synthetic_template_label",
                        "review_text": text,
                    }
                )
                review_id += 1

    return pd.DataFrame(rows)
