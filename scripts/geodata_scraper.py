import requests
import json
import time
from geopy.geocoders import Nominatim

# Initialize the geocoder (Nominatim is free and requires no API key)
geolocator = Nominatim(user_agent="tunisia_property_scraper")

def get_location_info(place_name):
    """Get location information using Nominatim"""
    try:
        location = geolocator.geocode(place_name)
        if location:
            return {
                "address": location.address, # pyright: ignore[reportAttributeAccessIssue]
                "latitude": location.latitude, # type: ignore
                "longitude": location.longitude # type: ignore
            }
        return None
    except Exception as e:
        print(f"Error geocoding {place_name}: {e}")
        return None

def search_nearby_places(lat, lon, radius=5000, amenity_type="neighborhood"):
    """
    Search for nearby places/neighborhoods using Overpass API (OpenStreetMap)
    Similar to gmaps.places_nearby() behavior
    """
    try:
        # Try multiple Overpass endpoints
        endpoints = [
            "http://overpass.osm.ch/api/interpreter",
            "http://lz4.overpass-api.de/api/interpreter",
            "http://overpass-api.de/api/interpreter"
        ]
        
        # More specific query for amenities and buildings
        overpass_query = f"""
        [bbox:{lat-0.05},{lon-0.05},{lat+0.05},{lon+0.05}];
        (
          node["amenity"];
          node["shop"];
          way["amenity"];
          way["shop"];
          way["building"];
        );
        out center;
        """
        
        for endpoint in endpoints:
            try:
                response = requests.post(endpoint, data=overpass_query, timeout=15)
                
                if response.status_code == 200 and response.text.strip():
                    try:
                        data = response.json()
                        places = []
                        
                        for element in data.get('elements', []):
                            place_info = {
                                "id": element.get('id'),
                                "name": element.get('tags', {}).get('name', 'Unknown'),
                                "type": element.get('tags', {}).get('amenity') or 
                                       element.get('tags', {}).get('shop') or 
                                       element.get('tags', {}).get('building', 'unknown'),
                                "latitude": element.get('lat') or element.get('center', {}).get('lat'),
                                "longitude": element.get('lon') or element.get('center', {}).get('lon'),
                                "description": element.get('tags', {}).get('description', ''),
                                "website": element.get('tags', {}).get('website', ''),
                                "phone": element.get('tags', {}).get('phone', ''),
                                "opening_hours": element.get('tags', {}).get('opening_hours', ''),
                            }
                            
                            if place_info['latitude'] and place_info['longitude']:
                                places.append(place_info)
                        
                        return places
                    except json.JSONDecodeError:
                        continue
                        
            except requests.exceptions.Timeout:
                continue
            except Exception:
                continue
        
        # If Overpass fails, return empty list (fallback)
        return []
            
    except Exception as e:
        print(f"  Warning: Could not retrieve nearby places ({str(e)[:50]})")
        return []

# Main execution
print("=== Fetching Geospatial Data ===\n")

places = ["Tunis, Tunisia", "Sfax, Tunisia", "Sousse, Tunisia"]
all_results = {}

# Get Location Info
for place in places:
    print(f"Fetching data for {place}...")
    info = get_location_info(place)
    if info:
        print(f"  Address: {info['address']}")
        print(f"  Coordinates: ({info['latitude']:.4f}, {info['longitude']:.4f})")
        print(f"  Searching for nearby places...")
        nearby_places = search_nearby_places(info['latitude'], info['longitude'], radius=5000)
        if nearby_places:
            print(f"  Found {len(nearby_places)} nearby places/amenities")
            info['nearby_places'] = nearby_places
            print(f"  Sample nearby place: {nearby_places[0]['name']} ({nearby_places[0]['type']})")
        else:
            print("  No nearby places found.")
        
        # Search for nearby places (neighborhoods, amenities, etc.)
        print(f"  Searching for nearby places...")
        nearby_places = search_nearby_places(info['latitude'], info['longitude'], radius=5000)
        
        if nearby_places:
            print(f"  Found {len(nearby_places)} nearby places/amenities")
            info['nearby_places'] = nearby_places
        
        all_results[place] = info
    print()
    time.sleep(1)  # Respect rate limits

# Save to file
print("Saving geospatial data to file...")
with open("data/geospatial_data.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
    
print(f"Data saved to data/geospatial_data.json")
print(f"Total places processed: {len(all_results)}")