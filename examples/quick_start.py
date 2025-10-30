"""Quick start example for intent classification."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformers import DistilBertTokenizer

from src.config import Config
from src.dataset import CLINC150DataLoader
from src.inference import IntentPredictor
from src.model import IntentClassifier
from src.utils import get_device


def quick_inference_example():
    print("\n" + "=" * 60)
    print("QUICK START: Intent Classification Inference")
    print("=" * 60)

    device = get_device()

    print("\n📦 Loading dataset to get label mappings...")
    data_loader = CLINC150DataLoader(cache_dir="cache")
    data_loader.load_dataset()

    print("\n🤖 Creating and loading model...")
    model_path = Path("models/best_model.pt")

    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        print("Please train the model first by running: python train.py")
        return

    num_intents = data_loader.get_num_labels()
    model = IntentClassifier(num_intents=num_intents)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    predictor = IntentPredictor(
        model=model,
        tokenizer=tokenizer,
        id2label=data_loader.id2label,
        device=device,
        confidence_threshold=0.7,
    )

    print("\n✅ Model loaded successfully!\n")

    examples = [
        "I need to check my bank balance",
        "What's the weather forecast for tomorrow?",
        "Book a table for two at 7pm",
        "Set an alarm for 6am",
        "Play some jazz music",
        "How do I reset my password?",
    ]

    print("🔍 Testing predictions on example queries:")
    print("=" * 60)

    for example in examples:
        intent, confidence, is_high_conf = predictor.predict_single(example)
        conf_indicator = "✓" if is_high_conf else "⚠️"

        print(f"\n📝 Query: '{example}'")
        print(f"   {conf_indicator} Intent: {intent}")
        print(f"   Confidence: {confidence:.4f}")


def batch_prediction_example():
    print("\n" + "=" * 60)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 60)

    device = get_device()
    data_loader = CLINC150DataLoader(cache_dir="cache")
    data_loader.load_dataset()

    model_path = Path("models/best_model.pt")
    if not model_path.exists():
        print(f"\n❌ Model not found at {model_path}")
        return

    predictor = IntentPredictor.from_pretrained(
        model_path=model_path,
        tokenizer_name="distilbert-base-uncased",
        id2label=data_loader.id2label,
        device=device,
    )

    batch_texts = [
        "Transfer $50 to John",
        "What's the temperature outside?",
        "Book a flight to Paris",
        "Turn on the bedroom lights",
    ]

    print("\n🚀 Processing batch of queries...")
    results = predictor.predict_batch(batch_texts)

    for text, (intent, confidence, is_high_conf) in zip(batch_texts, results):
        conf_indicator = "✓" if is_high_conf else "⚠️"
        print(f"\n📝 '{text}'")
        print(f"   {conf_indicator} {intent} (conf: {confidence:.4f})")


if __name__ == "__main__":
    quick_inference_example()
    print("\n" + "=" * 60 + "\n")
    batch_prediction_example()
