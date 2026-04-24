from __future__ import annotations

import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from src.agent.langgraph_agent_api import (
	_invoke_graph,
	delete_listing_by_id,
	fetch_listing_by_id,
	fetch_recent_listings,
	search_listings,
	store_listing,
)
from src.api import PropertyRequest
from src.vision.feature_aggregation import PROPERTY_HINT_MAP
from unified_api.serializers import (
	BatchListingRequestSerializer,
	ListingRequestSerializer,
	PropertyRequestSerializer,
	SchedulerRunOnceSerializer,
	SchedulerStartSerializer,
)
from unified_api.services import get_db_path, get_listing_graph, get_scheduler, get_valuation_service


def _parse_bool(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _infer_from_clip_rows(rows: list[dict[str, Any]]) -> str:
	inferred_candidates: list[str] = []
	for row in rows:
		top = row.get("top_prediction") or {}
		label = str(top.get("label", "")).strip()
		if label in {"Terrain", "Maison", "Appartement"}:
			inferred_candidates.append(label)
			continue
		inferred = PROPERTY_HINT_MAP.get(label)
		if inferred:
			inferred_candidates.append(inferred)
	if not inferred_candidates:
		return ""
	return Counter(inferred_candidates).most_common(1)[0][0]


def _as_property_payload(data: dict[str, Any]) -> PropertyRequest:
	payload = dict(data)
	payload["size_m2"] = float(payload.get("size_m2") or 0.0)
	return PropertyRequest(**payload)


class HealthView(APIView):
	def get(self, request):
		return Response({"status": "ok"})


class EstimateView(APIView):
	def post(self, request):
		serializer = PropertyRequestSerializer(data=request.data)
		serializer.is_valid(raise_exception=True)
		payload = _as_property_payload(serializer.validated_data)
		result = get_valuation_service().estimate(payload)
		return Response(result)


class EstimateUploadView(APIView):
	def post(self, request):
		upload_dir = Path("artifacts") / "tmp" / "api_uploads" / str(uuid.uuid4())
		upload_dir.mkdir(parents=True, exist_ok=True)
		image_refs: list[str] = []
		try:
			for upload in request.FILES.getlist("images"):
				suffix = Path(upload.name or "image.jpg").suffix or ".jpg"
				target = upload_dir / f"{uuid.uuid4().hex}{suffix}"
				with target.open("wb") as handle:
					for chunk in upload.chunks():
						handle.write(chunk)
				image_refs.append(str(target))

			raw_payload = {
				"property_type": str(request.data.get("property_type", "") or "").strip(),
				"governorate": request.data.get("governorate", ""),
				"city": request.data.get("city", ""),
				"neighborhood": request.data.get("neighborhood", ""),
				"size_m2": float(request.data.get("size_m2", 0) or 0),
				"bedrooms": int(request.data.get("bedrooms", 0) or 0),
				"bathrooms": int(request.data.get("bathrooms", 0) or 0),
				"condition": request.data.get("condition", ""),
				"has_pool": _parse_bool(request.data.get("has_pool", False)),
				"has_garden": _parse_bool(request.data.get("has_garden", False)),
				"has_parking": _parse_bool(request.data.get("has_parking", False)),
				"sea_view": _parse_bool(request.data.get("sea_view", False)),
				"elevator": _parse_bool(request.data.get("elevator", False)),
				"description": request.data.get("description", ""),
				"uploaded_images_count": len(image_refs),
				"image_refs": image_refs,
				"transaction_type": request.data.get("transaction_type", "sale"),
			}

			serializer = PropertyRequestSerializer(data=raw_payload)
			serializer.is_valid(raise_exception=True)
			payload = _as_property_payload(serializer.validated_data)

			rows = get_valuation_service().image_type.classify_many(image_refs)
			inferred_property_type = _infer_from_clip_rows(rows)
			selected_property_type = str(payload.property_type or "").strip()
			has_conflict = bool(
				selected_property_type
				and inferred_property_type
				and selected_property_type != inferred_property_type
			)
			if has_conflict and not _parse_bool(request.data.get("confirm_visual_conflict", False)):
				return Response(
					{
						"detail": {
							"code": "property_type_conflict_requires_confirmation",
							"message": (
								f"Selected property type '{selected_property_type}' conflicts with image inference "
								f"('{inferred_property_type}'). Confirm correction before estimation."
							),
							"selected_property_type": selected_property_type,
							"inferred_property_type": inferred_property_type,
							"requires_confirmation": True,
						}
					},
					status=status.HTTP_409_CONFLICT,
				)

			external_warnings = []
			if has_conflict:
				external_warnings.append(
					f"clip_property_type_mismatch:selected={selected_property_type},inferred={inferred_property_type}"
				)
			result = get_valuation_service().estimate(payload, external_warnings=external_warnings)
			return Response(result)
		finally:
			shutil.rmtree(upload_dir, ignore_errors=True)


class IngestAndValueView(APIView):
	def post(self, request):
		serializer = ListingRequestSerializer(data=request.data)
		serializer.is_valid(raise_exception=True)
		url = serializer.validated_data["url"]
		try:
			out = _invoke_graph(listing_graph=get_listing_graph(), url=url)
			final_listing = out.get("final_listing") if isinstance(out, dict) else {}
			if not isinstance(final_listing, dict):
				raise RuntimeError("missing final listing")
			out["final_listing"] = store_listing(db_path=get_db_path(), final_listing=final_listing, source_url=url)
			return Response(out)
		except Exception as exc:
			return Response({"detail": f"graph execution failed: {exc}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class IngestAndValueBatchView(APIView):
	def post(self, request):
		serializer = BatchListingRequestSerializer(data=request.data)
		serializer.is_valid(raise_exception=True)
		items: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []

		for url in serializer.validated_data["urls"]:
			try:
				out = _invoke_graph(listing_graph=get_listing_graph(), url=url)
				listing = out.get("final_listing") if isinstance(out, dict) else {}
				if isinstance(listing, dict):
					listing = store_listing(db_path=get_db_path(), final_listing=listing, source_url=url)
				items.append({"url": url, "final_listing": listing})
			except Exception as exc:
				errors.append({"url": url, "error": str(exc)})

		return Response({"count": len(items), "error_count": len(errors), "items": items, "errors": errors})


class RecentListingsView(APIView):
	def get(self, request):
		limit = int(request.query_params.get("limit", 5) or 5)
		rows = fetch_recent_listings(db_path=get_db_path(), limit=max(1, min(limit, 100)))
		return Response({"count": len(rows), "items": rows})


class ListingsView(APIView):
	def get(self, request):
		city = request.query_params.get("city")
		governorate = request.query_params.get("governorate")
		property_type = request.query_params.get("property_type")
		min_price = request.query_params.get("min_price")
		max_price = request.query_params.get("max_price")
		limit = int(request.query_params.get("limit", 25) or 25)

		min_value = float(min_price) if min_price is not None else None
		max_value = float(max_price) if max_price is not None else None
		if min_value is not None and max_value is not None and min_value > max_value:
			return Response({"detail": "min_price cannot exceed max_price"}, status=status.HTTP_400_BAD_REQUEST)

		rows = search_listings(
			db_path=get_db_path(),
			city=city,
			governorate=governorate,
			property_type=property_type,
			min_price=min_value,
			max_price=max_value,
			limit=max(1, min(limit, 200)),
		)
		return Response(
			{
				"count": len(rows),
				"filters": {
					"city": city,
					"governorate": governorate,
					"property_type": property_type,
					"min_price": min_value,
					"max_price": max_value,
					"limit": limit,
				},
				"items": rows,
			}
		)


class AddFromValuationView(APIView):
	def post(self, request):
		serializer = PropertyRequestSerializer(data=request.data)
		serializer.is_valid(raise_exception=True)
		payload = _as_property_payload(serializer.validated_data)
		valuation = get_valuation_service().estimate(payload)
		final_listing = {
			"title": f"Manual {payload.property_type or 'Property'} in {payload.city}",
			"property_type": payload.property_type,
			"governorate": payload.governorate,
			"city": payload.city,
			"neighborhood": payload.neighborhood,
			"surface_m2": payload.size_m2,
			"bedrooms": payload.bedrooms,
			"bathrooms": payload.bathrooms,
			"condition": payload.condition,
			"predicted_price_tnd": float(valuation.get("estimated_price") or 0.0),
			"confidence_score": float(valuation.get("confidence") or 0.0),
			"payload": valuation,
			"transaction_type": payload.transaction_type,
		}
		stored = store_listing(db_path=get_db_path(), final_listing=final_listing, source_url="manual://valuation")
		return Response({"valuation": valuation, "final_listing": stored}, status=status.HTTP_201_CREATED)


class ListingByIdView(APIView):
	def get(self, request, listing_id: str):
		row = fetch_listing_by_id(db_path=get_db_path(), listing_id=listing_id)
		if row is None:
			return Response({"detail": "listing not found"}, status=status.HTTP_404_NOT_FOUND)
		return Response(row)

	def delete(self, request, listing_id: str):
		deleted = delete_listing_by_id(db_path=get_db_path(), listing_id=listing_id)
		if not deleted:
			return Response({"detail": "listing not found"}, status=status.HTTP_404_NOT_FOUND)
		return Response({"deleted": True, "id": listing_id})


class SchedulerStartView(APIView):
	def post(self, request):
		serializer = SchedulerStartSerializer(data=request.data)
		serializer.is_valid(raise_exception=True)
		data = serializer.validated_data
		result = get_scheduler().start(
			interval_hours=data["interval_hours"],
			target_per_source=data["target_per_source"],
			timeout_s=data["timeout_s"],
			sleep_s=data["sleep_s"],
		)
		return Response(result)


class SchedulerStopView(APIView):
	def post(self, request):
		return Response(get_scheduler().stop())


class SchedulerStatusView(APIView):
	def get(self, request):
		return Response(get_scheduler().status())


class SchedulerRunOnceView(APIView):
	def post(self, request):
		serializer = SchedulerRunOnceSerializer(data=request.data)
		serializer.is_valid(raise_exception=True)
		data = serializer.validated_data
		result = get_scheduler().run_once(
			target_per_source=data["target_per_source"],
			timeout_s=data["timeout_s"],
			sleep_s=data["sleep_s"],
		)
		return Response(result)


class ServicesView(APIView):
	def get(self, request):
		return Response(
			{
				"name": "EstateMind Unified API (Django)",
				"version": "0.1.0",
				"endpoints": [
					{"method": "GET", "path": "/health"},
					{"method": "POST", "path": "/estimate"},
					{"method": "POST", "path": "/estimate-upload"},
					{"method": "POST", "path": "/ingest-and-value"},
					{"method": "POST", "path": "/ingest-and-value/batch"},
					{"method": "GET", "path": "/recent-listings"},
					{"method": "GET", "path": "/listings"},
					{"method": "POST", "path": "/listings/add-from-valuation"},
					{"method": "GET", "path": "/listings/{listing_id}"},
					{"method": "DELETE", "path": "/listings/{listing_id}"},
					{"method": "POST", "path": "/scheduler/start"},
					{"method": "POST", "path": "/scheduler/stop"},
					{"method": "GET", "path": "/scheduler/status"},
					{"method": "POST", "path": "/scheduler/run-once"},
					{"method": "GET", "path": "/services"},
				],
			}
		)
