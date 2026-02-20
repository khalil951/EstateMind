import requests
import json
import os

# Hugging Face free API - no payment required!
# Get your free API token from: https://huggingface.co/settings/tokens

# Set your HuggingFace API token here or via environment variable
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "YOUR_API_TOKEN_HERE")

# Use free models available on Hugging Face Inference API
# Mistral-7B is fast and free
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def generate_synthetic_reviews(location, num_reviews=5):
    """
    Generate synthetic reviews using Hugging Face free API
    Similar to OpenAI but completely free with no quota limits
    """
    
    prompt = f"""Generate {num_reviews} realistic neighborhood reviews for {location}, Tunisia.
Include both positive and negative aspects. Write in French.
Reviews should mention: safety, noise, amenities, schools, transport.
Format: numbered list of reviews.

Reviews:"""
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 1000,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    try:
        print(f"Generating {num_reviews} synthetic reviews for {location}...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                return generated_text
            else:
                return result
        else:
            print(f"Error: Status {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Hugging Face free API might be busy, try again later.")
        return None
    except Exception as e:
        print(f"Error generating synthetic reviews: {e}")
        return None

# Example usage
print("=" * 60)
print("Synthetic Review Generator (Free - Hugging Face)")
print("=" * 60)
print()

# Check if API token is set
if HF_API_TOKEN == "YOUR_API_TOKEN_HERE":
    print("⚠️  ERROR: API token not configured!")
    print()
    print("To use this script:")
    print("1. Create a free account at: https://huggingface.co/")
    print("2. Get your API token: https://huggingface.co/settings/tokens")
    print("3. Set it as environment variable:")
    print("   - Windows: set HF_API_TOKEN=your_token_here")
    print("   - Linux/Mac: export HF_API_TOKEN=your_token_here")
    print()
else:
    # Generate reviews for multiple locations
    locations = ["La Marsa", "Tunis", "Sfax"]
    
    all_reviews = {}
    
    for location in locations:
        reviews = generate_synthetic_reviews(location, num_reviews=3)
        if reviews:
            all_reviews[location] = reviews
            print(f"\n✓ Generated reviews for {location}")
            print("-" * 40)
            print(reviews[:500])  # Print first 500 chars
            print("...")
        else:
            print(f"\n✗ Failed to generate reviews for {location}")
    
    # Save to file
    print("\n" + "=" * 60)
    if all_reviews:
        with open("data/synthetic_reviews.json", "w", encoding="utf-8") as f:
            json.dump(all_reviews, f, indent=2, ensure_ascii=False)
        print("✓ Synthetic reviews saved to data/synthetic_reviews.json")
