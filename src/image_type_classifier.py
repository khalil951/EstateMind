from __future__ import annotations

"""Classify real-estate image types with a pretrained CLIP model.

This module is intentionally standalone so it can be tested locally before
backend integration. It supports both local image paths and remote image URLs
and performs zero-shot classification over a curated set of real-estate image
types using Hugging Face CLIP.
"""

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import requests
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
except ModuleNotFoundError as exc:  # pragma: no cover
    missing = exc.name or "required dependency"
    raise ModuleNotFoundError(
        f"Missing dependency '{missing}'. Install pillow, requests, torch, and transformers to use image_type_classifier."
    ) from exc


DEFAULT_IMAGE_TYPE_PROMPTS: dict[str, str] = {
    "property_type_terrain": "a real estate image of an empty land plot or terrain lot",
    "property_type_maison": "a real estate image of a detached house or villa exterior",
    "property_type_appartement": "a real estate image of an apartment building exterior or apartment facade",
    "amenity_pool": "a real estate photo showing a swimming pool",
    "amenity_garden": "a real estate photo showing a private garden or yard",
    "amenity_parking": "a real estate photo showing a garage, parking space, or driveway",
    "amenity_sea_view": "a real estate photo showing a sea view or ocean view from the property",
    "amenity_elevator": "a real estate photo showing an apartment elevator or lift",
    "land_plot": "a real estate photo of an empty land plot or terrain for sale",
    "agricultural_land": "a real estate photo of agricultural land, farmland, or an open field",
    "villa_exterior": "a real estate photo of the exterior of a villa",
    "house_exterior": "a real estate photo of the exterior of a detached house",
    "apartment_building_exterior": "a real estate photo of the exterior of an apartment building",
    "residential_building_exterior": "a real estate photo of the facade of a residential building",
    "commercial_exterior": "a real estate photo of the exterior of a shop, office, or commercial property",
    "living_room": "a real estate photo of a living room",
    "bedroom": "a real estate photo of a bedroom",
    "bathroom": "a real estate photo of a bathroom",
    "kitchen": "a real estate photo of a kitchen",
    "dining_room": "a real estate photo of a dining room",
    "balcony_or_terrace": "a real estate photo of a balcony or terrace",
    "garden_or_yard": "a real estate photo of a garden, yard, or outdoor courtyard",
    "swimming_pool": "a real estate photo of a swimming pool",
    "garage_or_parking": "a real estate photo of a garage, driveway, or parking area",
    "stairs_or_hallway": "a real estate photo of stairs, a corridor, or a hallway inside a property",
    "empty_room": "a real estate photo of an empty unfurnished room",
    "floor_plan": "a floor plan, blueprint, map, or property layout image",
    "construction_site": "a real estate photo of a construction site or unfinished property",
    "other": "a real estate image that does not clearly fit the main categories",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a property image type with a pretrained Hugging Face CLIP model."
    )
    parser.add_argument("--image", required=True, help="Local image path or image URL.")
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to return.")
    parser.add_argument(
        "--labels-json",
        default="",
        help="Optional JSON file mapping label keys to CLIP prompt strings.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output JSON path for saving the classification result.",
    )
    return parser.parse_args()


def load_image(image_ref: str) -> Image.Image:
    """Load an image from a local path or URL and return it as RGB PIL image."""
    if image_ref.lower().startswith(("http://", "https://")):
        response = requests.get(image_ref, stream=True, timeout=20)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")#type: ignore
    return Image.open(Path(image_ref)).convert("RGB")


def load_prompt_map(labels_json: str | None) -> dict[str, str]:
    """Load a custom label->prompt map or fall back to the default set."""
    if not labels_json:
        return DEFAULT_IMAGE_TYPE_PROMPTS.copy()

    path = Path(labels_json)
    prompt_map = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prompt_map, dict) or not prompt_map:
        raise ValueError("labels-json must contain a non-empty JSON object of label-to-prompt mappings.")

    cleaned: dict[str, str] = {}
    for key, value in prompt_map.items():
        label = str(key).strip()
        prompt = str(value).strip()
        if not label or not prompt:
            continue
        cleaned[label] = prompt

    if not cleaned:
        raise ValueError("labels-json did not contain any usable label/prompt pairs.")
    return cleaned


class CLIPImageTypeClassifier:
    """Reusable CLIP-based zero-shot classifier for property images."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32") -> None:
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device) # type: ignore
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def classify_image(
        self,
        image: Image.Image,
        prompt_map: dict[str, str] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Classify a PIL image against the configured prompt map."""
        prompts = prompt_map or DEFAULT_IMAGE_TYPE_PROMPTS
        labels = list(prompts.keys())
        texts = list(prompts.values())

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt", # type: ignore
            padding=True, # type: ignore
        ) # type: ignore
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)
        top_k = max(1, min(int(top_k), len(labels)))
        top_probs, top_indices = probs.topk(top_k)

        predictions: list[dict[str, Any]] = []
        for score, idx in zip(top_probs.tolist(), top_indices.tolist()):
            predictions.append(
                {
                    "label": labels[idx],
                    "prompt": prompts[labels[idx]],
                    "score": float(score),
                }
            )

        return {
            "model_id": self.model_id,
            "device": self.device,
            "top_prediction": predictions[0],
            "predictions": predictions,
        }

    def classify_image_path(
        self,
        image_ref: str,
        prompt_map: dict[str, str] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Load an image from a path/URL and classify it."""
        image = load_image(image_ref)
        result = self.classify_image(image=image, prompt_map=prompt_map, top_k=top_k)
        result["image_ref"] = image_ref
        return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt_map = load_prompt_map(args.labels_json)
    classifier = CLIPImageTypeClassifier(model_id=str(args.model_id))
    result = classifier.classify_image_path(
        image_ref=str(args.image),
        prompt_map=prompt_map,
        top_k=int(args.top_k),
    )

    output_json = str(args.output_json).strip()
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
