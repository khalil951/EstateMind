import type { Listing, ListingSearchResponse, PropertyRequest } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8001";

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchListings(): Promise<Listing[]> {
  const data = await jsonRequest<ListingSearchResponse>("/listings");
  return data.items ?? [];
}

export async function addFromValuation(payload: PropertyRequest): Promise<{ valuation: Record<string, unknown>; final_listing: Listing }> {
  return jsonRequest("/listings/add-from-valuation", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function estimate(payload: PropertyRequest): Promise<Record<string, unknown>> {
  return jsonRequest("/estimate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function estimateUpload(
  payload: PropertyRequest,
  images: File[],
  confirmVisualConflict = false
): Promise<Record<string, unknown>> {
  const formData = new FormData();
  formData.append("property_type", payload.property_type);
  formData.append("governorate", payload.governorate);
  formData.append("city", payload.city);
  formData.append("neighborhood", payload.neighborhood);
  formData.append("size_m2", String(payload.size_m2));
  formData.append("bedrooms", String(payload.bedrooms));
  formData.append("bathrooms", String(payload.bathrooms));
  formData.append("condition", payload.condition);
  formData.append("has_pool", String(payload.has_pool));
  formData.append("has_garden", String(payload.has_garden));
  formData.append("has_parking", String(payload.has_parking));
  formData.append("sea_view", String(payload.sea_view));
  formData.append("elevator", String(payload.elevator));
  formData.append("description", payload.description);
  formData.append("transaction_type", payload.transaction_type);
  formData.append("confirm_visual_conflict", String(confirmVisualConflict));

  for (const image of images) {
    formData.append("images", image);
  }

  const res = await fetch(`${API_BASE}/estimate-upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json() as Promise<Record<string, unknown>>;
}

export async function health(): Promise<Record<string, unknown>> {
  return jsonRequest("/health");
}

export async function services(): Promise<Record<string, unknown>> {
  return jsonRequest("/services");
}

export async function recentListings(limit = 5): Promise<Record<string, unknown>> {
  return jsonRequest(`/recent-listings?limit=${limit}`);
}

export async function schedulerStatus(): Promise<Record<string, unknown>> {
  return jsonRequest("/scheduler/status");
}

export async function schedulerStart(payload?: {
  interval_hours?: number;
  target_per_source?: number;
  timeout_s?: number;
  sleep_s?: number;
}): Promise<Record<string, unknown>> {
  return jsonRequest("/scheduler/start", {
    method: "POST",
    body: JSON.stringify(payload ?? {}),
  });
}

export async function schedulerStop(): Promise<Record<string, unknown>> {
  return jsonRequest("/scheduler/stop", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export async function schedulerRunOnce(payload?: {
  target_per_source?: number;
  timeout_s?: number;
  sleep_s?: number;
}): Promise<Record<string, unknown>> {
  return jsonRequest("/scheduler/run-once", {
    method: "POST",
    body: JSON.stringify(payload ?? {}),
  });
}

export async function ingestAndValue(url: string): Promise<Record<string, unknown>> {
  return jsonRequest("/ingest-and-value", {
    method: "POST",
    body: JSON.stringify({ url }),
  });
}

export async function ingestAndValueBatch(urls: string[]): Promise<Record<string, unknown>> {
  return jsonRequest("/ingest-and-value/batch", {
    method: "POST",
    body: JSON.stringify({ urls }),
  });
}
