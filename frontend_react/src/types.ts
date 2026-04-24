export type PropertyRequest = {
  property_type: "" | "Terrain" | "Maison" | "Appartement";
  governorate: string;
  city: string;
  neighborhood: string;
  size_m2: number;
  bedrooms: number;
  bathrooms: number;
  condition: "New" | "Excellent" | "Good" | "Fair" | "Needs Renovation";
  has_pool: boolean;
  has_garden: boolean;
  has_parking: boolean;
  sea_view: boolean;
  elevator: boolean;
  description: string;
  uploaded_images_count: number;
  image_refs: string[];
  transaction_type: "sale" | "rent";
};

export type Listing = {
  id?: string;
  title?: string;
  property_type?: string;
  governorate?: string;
  city?: string;
  neighborhood?: string;
  surface_m2?: number;
  bedrooms?: number;
  bathrooms?: number;
  condition?: string;
  predicted_price_tnd?: number;
  confidence?: number;
  confidence_score?: number;
  payload?: Record<string, unknown>;
  source_url?: string;
};

export type ListingSearchResponse = {
  count: number;
  items: Listing[];
};
