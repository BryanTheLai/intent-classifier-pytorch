"""Generate intent mapping document from CLINC150 dataset."""

import json
from pathlib import Path
from datasets import load_dataset

print("Loading CLINC150 dataset with intent names...")
dataset = load_dataset("DeepPavlov/clinc150", cache_dir="cache")
intents_dataset = load_dataset("DeepPavlov/clinc150", "intents", cache_dir="cache")

print("\nMain dataset structure:")
print(f"  Columns: {dataset['train'].column_names}")
print(f"  Sample: {dataset['train'][0]}")

print("\nIntents dataset structure:")
print(f"  Columns: {intents_dataset['intents'].column_names}")
print(f"  Total intents: {len(intents_dataset['intents'])}")

id2label = {row['id']: row['name'] for row in intents_dataset['intents']}

mapping = {
    "total_intents": len(id2label),
    "intents": id2label
}

results_path = Path("results")
results_path.mkdir(exist_ok=True)

with open(results_path / "intent_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)

print(f"\n✓ Saved intent mapping with {len(id2label)} intents to results/intent_mapping.json")
print(f"\nSample mappings:")
for i in [0, 1, 9, 83, 113, 121, 130, 140]:
    print(f"  {i}: {id2label.get(i, 'N/A')}")
