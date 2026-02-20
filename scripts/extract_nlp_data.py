import pandas as pd
import json
import re
import os
from pathlib import Path

# Load the cleaned dataset
csv_path = "data/csv/data_prices_cleaned.csv"
df = pd.read_csv(csv_path)

print("=" * 60)
print("NLP Dataset Extractor")
print("=" * 60)
print(f"\nLoading dataset: {csv_path}")
print(f"Total records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Clean and extract descriptions
def clean_text(text):
    """Clean and normalize text for NLP training"""
    if pd.isna(text):
        return None
    
    text = str(text).strip()
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Remove phone numbers
    text = re.sub(r'\+?\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '[PHONE]', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special markers like "Afficher le numéro"
    text = re.sub(r'Afficher le numéro', '', text)
    
    return text.strip()

# Format 1: JSONL (one JSON object per line) - Best for streaming training
print("\n" + "=" * 60)
print("Creating JSONL format (for large-scale NLP training)...")

jsonl_data = []

for idx, row in df.iterrows():
    description = clean_text(row['descriptions'])
    title = clean_text(row['titles'])
    
    if description and len(description) > 20:  # Filter out very short descriptions
        record = {
            "id": idx,
            "text": description,
            "title": title,
            "category": row['category'],
            "location": row['location'],
            "city": row['city'],
            "price": row['price'],
            "features": {
                "bedrooms": row['chambres'],
                "bathrooms": row['salles_de_bains'],
                "area": row['superficie']
            }
        }
        jsonl_data.append(record)

# Save JSONL
jsonl_path = "data/nlp_training_data.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for record in jsonl_data:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✓ Saved {len(jsonl_data)} records to {jsonl_path}")

# Format 2: CSV for sklearn/pandas-based models
print("\n" + "=" * 60)
print("Creating CSV format (for structured analysis)...")

nlp_df = pd.DataFrame([
    {
        "id": idx,
        "text": clean_text(row['descriptions']),
        "title": clean_text(row['titles']),
        "category": row['category'],
        "location": row['location'],
        "city": row['city'],
        "price": row['price'],
        "bedrooms": row['chambres'],
        "bathrooms": row['salles_de_bains'],
        "area": row['superficie']
    }
    for idx, row in df.iterrows()
    if clean_text(row['descriptions']) and len(clean_text(row['descriptions'])) > 20 # pyright: ignore[reportArgumentType]
])

csv_output_path = "data/nlp_training_data.csv"
nlp_df.to_csv(csv_output_path, index=False, encoding="utf-8")
print(f"✓ Saved {len(nlp_df)} records to {csv_output_path}")

# Format 3: Plain Text (one description per line) - For simple models
print("\n" + "=" * 60)
print("Creating Plain Text format (for text generation models)...")

txt_path = "data/nlp_training_descriptions.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
        text = clean_text(row['descriptions'])
        if text and len(text) > 20:
            f.write(text + "\n")

print(f"✓ Saved descriptions to {txt_path}")

# Format 4: JSON with metadata (for HuggingFace datasets)
print("\n" + "=" * 60)
print("Creating JSON format (for HuggingFace/PyTorch loaders)...")

json_data = {
    "dataset_name": "tunisia_property_descriptions",
    "total_records": len(jsonl_data),
    "data": []
}

for record in jsonl_data:
    json_data["data"].append(record)

json_path = "data/nlp_training_data.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"✓ Saved {len(json_data['data'])} records to {json_path}")

# Summary Statistics
print("\n" + "=" * 60)
print("Dataset Summary")
print("=" * 60)
print(f"Total records extracted: {len(jsonl_data)}")
print(f"Average text length: {nlp_df['text'].str.len().mean():.0f} characters")
print(f"Min text length: {nlp_df['text'].str.len().min()}")
print(f"Max text length: {nlp_df['text'].str.len().max()}")
print(f"\nCategories distribution:")
print(nlp_df['category'].value_counts())
print(f"\nCities distribution:")
print(nlp_df['city'].value_counts().head(10))

print("\n" + "=" * 60)
print("✅ NLP Training data ready!")
print("=" * 60)
print("\nOutput files created:")
print(f"  1. {jsonl_path} - JSONL format (best for transformers/LLMs)")
print(f"  2. {csv_output_path} - CSV format (for pandas/sklearn)")
print(f"  3. {txt_path} - Plain text (for text generation models)")
print(f"  4. {json_path} - JSON format (for HuggingFace datasets)")
print("\nUsage examples:")
print("\n  # For HuggingFace transformers:")
print("  from datasets import load_dataset")
print("  dataset = load_dataset('json', data_files='data/nlp_training_data.jsonl')")
print("\n  # For Pandas:")
print("  import pandas as pd")
print("  df = pd.read_csv('data/nlp_training_data.csv')")
print("\n  # For PyTorch:")
print("  import json")
print("  with open('data/nlp_training_data.jsonl') as f:")
print("      data = [json.loads(line) for line in f]")
