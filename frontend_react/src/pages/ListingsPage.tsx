import { useEffect, useState } from "react";
import { fetchListings } from "../api";
import type { Listing } from "../types";
import { PropertyListingCard } from "../components/PropertyListingCard";

export default function ListingsPage() {
  const [listings, setListings] = useState<Listing[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchListings()
      .then(setListings)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <section className="workspace-panel reveal-panel">
      <div className="panel-header">
        <p className="section-label">MARKET DISCOVERY</p>
        <h2 className="panel-title">Listings Intelligence Surface</h2>
        <p className="panel-description">
          Scan opportunities quickly with normalized cards, uniform media slots, and price-first emphasis.
        </p>
      </div>

      {loading && <p className="muted">Loading listings...</p>}
      {error && <p className="error">{error}</p>}
      <div className="cards">
        {listings.map((listing, index) => (
          <PropertyListingCard
            key={listing.id ?? listing.source_url ?? `${listing.city}-${listing.title}-${index}`}
            listing={listing}
          />
        ))}
      </div>
    </section>
  );
}
