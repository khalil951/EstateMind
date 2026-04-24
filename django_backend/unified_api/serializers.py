from __future__ import annotations

from typing import Any

from rest_framework import serializers

from src.api import PropertyRequest


class PropertyRequestSerializer(serializers.Serializer):
    property_type = serializers.ChoiceField(
        choices=["", "Terrain", "Maison", "Appartement"],
        required=False,
        default="",
    )
    governorate = serializers.CharField()
    city = serializers.CharField()
    neighborhood = serializers.CharField(required=False, allow_blank=True, default="")
    size_m2 = serializers.FloatField(min_value=10.0000001)
    bedrooms = serializers.IntegerField(required=False, default=0)
    bathrooms = serializers.IntegerField(required=False, default=0)
    condition = serializers.ChoiceField(choices=["New", "Excellent", "Good", "Fair", "Needs Renovation"])
    has_pool = serializers.BooleanField(required=False, default=False)
    has_garden = serializers.BooleanField(required=False, default=False)
    has_parking = serializers.BooleanField(required=False, default=False)
    sea_view = serializers.BooleanField(required=False, default=False)
    elevator = serializers.BooleanField(required=False, default=False)
    description = serializers.CharField(required=False, allow_blank=True, default="")
    uploaded_images_count = serializers.IntegerField(required=False, default=0)
    image_refs = serializers.ListField(required=False, child=serializers.CharField(), default=list)
    transaction_type = serializers.ChoiceField(choices=["sale", "rent"], required=False, default="sale")

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        payload = dict(attrs)
        payload["size_m2"] = max(float(payload.get("size_m2") or 0.0), 10.0000001)
        PropertyRequest(**payload)
        return attrs


class ListingRequestSerializer(serializers.Serializer):
    url = serializers.URLField()


class BatchListingRequestSerializer(serializers.Serializer):
    urls = serializers.ListField(child=serializers.URLField())

    def validate_urls(self, value: list[str]) -> list[str]:
        if not value:
            raise serializers.ValidationError("urls must not be empty")
        if len(value) > 100:
            raise serializers.ValidationError("urls must contain at most 100 items")
        return value


class SchedulerStartSerializer(serializers.Serializer):
    interval_hours = serializers.FloatField(required=False, default=24.0, min_value=0.000001)
    target_per_source = serializers.IntegerField(required=False, default=20, min_value=1, max_value=200)
    timeout_s = serializers.IntegerField(required=False, default=30, min_value=5, max_value=120)
    sleep_s = serializers.FloatField(required=False, default=0.0, min_value=0.0, max_value=5.0)


class SchedulerRunOnceSerializer(serializers.Serializer):
    target_per_source = serializers.IntegerField(required=False, default=20, min_value=1, max_value=200)
    timeout_s = serializers.IntegerField(required=False, default=30, min_value=5, max_value=120)
    sleep_s = serializers.FloatField(required=False, default=0.0, min_value=0.0, max_value=5.0)
