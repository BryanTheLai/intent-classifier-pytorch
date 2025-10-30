"""Generate intent mapping document from CLINC150 dataset."""

import json
from datasets import load_dataset

dataset = load_dataset("DeepPavlov/clinc150", cache_dir="cache")

train_data = dataset["train"]
features = train_data.features

print("Dataset features:", features)
print("\nDataset sample:")
for i in range(3):
    print(f"  {i}: {train_data[i]}")

intent_col = None
text_col = None

for col_name, feature in features.items():
    print(f"\nColumn: {col_name}, Type: {type(feature)}, Feature: {feature}")
    if hasattr(feature, 'names'):
        intent_col = col_name
        print(f"  -> Found ClassLabel column: {col_name}")
        print(f"  -> Names: {feature.names[:10]}...")
    if col_name in ['text', 'utterance']:
        text_col = col_name

if intent_col and hasattr(features[intent_col], 'names'):
    intent_names = features[intent_col].names
    
    mapping = {
        "total_intents": len(intent_names),
        "intents": {}
    }
    
    for idx, name in enumerate(intent_names):
        mapping["intents"][str(idx)] = name
    
    with open("results/intent_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✓ Saved intent mapping with {len(intent_names)} intents")
    print(f"Sample: Intent 0 = {intent_names[0]}, Intent 83 = {intent_names[83]}")
else:
    print("\n❌ Could not find ClassLabel feature with intent names")
