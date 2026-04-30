import { memo, useMemo } from "react";
import type { Listing } from "../types";
import propertyPlaceholder from "../assets/property-placeholder.svg";

type PropertyListingCardProps = {
  listing: Listing;
};

function normalizeText(value: unknown): string {
  if (typeof value !== "string") {
    return "";
  }
  return value
    .replace(/\bentregitrer\b/gi, "")
    .replace(/\bsignaler\b/gi, "")
    .replace(/\s+/g, " ")
    .trim();
}

function titleCase(value: string): string {
  if (!value) {
    return "";
  }
  return value
    .toLowerCase()
    .split(" ")
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(" ");
}

function translatePropertyTypeToDisplay(value: string): string {
  const normalized = normalizeText(value).toLowerCase();
  if (!normalized) {
    return "";
  }

  const map: Record<string, string> = {
    apartment: "Appartment",
    house: "House",
    land: "Land",
    apartments: "Appartments",
    houses: "Houses",
    lands: "Lands",
    villa: "Villa",
    duplex: "Duplex",
    studio: "Studio",
    commercial: "Commercial",
    office: "Office",
    bureau: "Office",
    "retail space": "Retail Space",
    local: "Local",
    appartement: "Appartment",
    maison: "House",
    terrain: "Land",
  };

  return map[normalized] ?? titleCase(normalized);
}

function translateTitleToDisplay(value: string): string {
  let text = normalizeText(value);
  if (!text) {
    return "";
  }

  const replacements: Array<[RegExp, string | ((...args: string[]) => string)]> = [
    [/\bappartement\b/gi, "Appartment"],
    [/\bmaison\b/gi, "House"],
    [/\bterrain\b/gi, "Land"],
    [/\bvilla\b/gi, "Villa"],
    [/\bduplex\b/gi, "Duplex"],
    [/\bstudio\b/gi, "Studio"],
    [/\bcommercial\b/gi, "Commercial"],
    [/\bfor\s+rent\b/gi, "For Rent"],
    [/\bfor\s+sale\b/gi, "For Sale"],
    [/\brental\b/gi, "Rental"],
    [/\bs\s*\+\s*(\d+)\b/gi, (_match: string, rooms: string) => `S+${rooms}`],
  ];

  for (const [pattern, replacement] of replacements) {
    if (typeof replacement === "string") {
      text = text.replace(pattern, replacement);
    } else {
      text = text.replace(pattern, (...args: unknown[]) => replacement(...(args as [string, string])));
    }
  }

  return titleCase(text);
}

function truncateWithEllipsis(value: string, maxLength: number): string {
  if (!value || value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 1).trimEnd()}…`;
}

function formatPrice(value: unknown): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "";
  }
  return `${Math.round(numeric).toLocaleString()} TND`;
}

function formatSurface(value: unknown): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return "";
  }
  return Number.isInteger(numeric) ? `${numeric} m2` : `${numeric.toFixed(1)} m2`;
}

function formatCount(value: unknown): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) {
    return "";
  }
  return `${Math.trunc(numeric)}`;
}

function PropertyListingCardComponent({ listing }: PropertyListingCardProps) {
  const view = useMemo(() => {
    const title = truncateWithEllipsis(translateTitleToDisplay(listing.title ?? ""), 68);
    const propertyType = translatePropertyTypeToDisplay(listing.property_type ?? "");
    const locationParts = [listing.city, listing.governorate, listing.neighborhood]
      .map((part) => titleCase(normalizeText(part)))
      .filter(Boolean);
    const location = truncateWithEllipsis(locationParts.join(", "), 72);

    const price = formatPrice(listing.predicted_price_tnd);
    const surface = formatSurface(listing.surface_m2);
    const bedrooms = formatCount(listing.bedrooms);
    const bathrooms = formatCount(listing.bathrooms);

    return {
      title,
      propertyType,
      location,
      price,
      bedrooms,
      bathrooms,
      surface,
    };
  }, [listing]);

  return (
    <article className="card">
      <div className="listing-image-placeholder" aria-hidden="true">
        <img src={propertyPlaceholder} alt="" loading="lazy" />
      </div>

      <div className="card-header">
        {view.title ? <h3 className="card-title">{view.title}</h3> : null}
        {view.location ? <p className="card-location">{view.location}</p> : null}
      </div>

      <div className="card-main-info">
        {view.price ? <p className="card-price">{view.price}</p> : null}
        {view.propertyType ? <span className="meta-chip">{view.propertyType}</span> : null}
      </div>

      <div className="card-details" aria-label="Listing details">
        {view.bedrooms ? <span className="detail-chip">Beds {view.bedrooms}</span> : null}
        {view.bathrooms ? <span className="detail-chip">Baths {view.bathrooms}</span> : null}
        {view.surface ? <span className="detail-chip">{view.surface}</span> : null}
      </div>
    </article>
  );
}

export const PropertyListingCard = memo(PropertyListingCardComponent);
