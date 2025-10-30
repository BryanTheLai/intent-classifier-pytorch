"""Inference script for intent classification."""

import argparse
import json
from pathlib import Path

import torch

from src.inference import IntentPredictor
from src.utils import get_device


def main(
    model_path: str,
    metadata_path: str,
    text: str = None,
    interactive: bool = False,
) -> None:
    device = get_device()

    print("\n" + "=" * 60)
    print("INTENT CLASSIFICATION INFERENCE")
    print("=" * 60)

    print("\n📦 Loading model metadata...")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    id2label = {int(k): v for k, v in metadata["label_mappings"]["id2label"].items()}

    print("\n🤖 Loading model...")
    predictor = IntentPredictor.from_pretrained(
        model_path=Path(model_path),
        tokenizer_name="distilbert-base-uncased",
        id2label=id2label,
        device=device,
        confidence_threshold=0.7,
    )
    print("✓ Model loaded successfully")

    if interactive:
        print("\n💬 Interactive mode (type 'quit' to exit)")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nEnter text: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye! 👋")
                    break

                if not user_input:
                    continue

                intent, confidence, is_high_conf = predictor.predict_single(user_input)
                conf_indicator = "✓" if is_high_conf else "⚠️"

                print(f"\n{conf_indicator} Predicted Intent: {intent}")
                print(f"   Confidence: {confidence:.4f}")

                print("\n   Top 5 predictions:")
                top_k = predictor.predict_with_top_k(user_input, k=5)
                for i, (label, prob) in enumerate(top_k, 1):
                    print(f"   {i}. {label}: {prob:.4f}")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    elif text:
        print(f"\n📝 Input text: {text}")
        intent, confidence, is_high_conf = predictor.predict_single(text)
        conf_indicator = "✓" if is_high_conf else "⚠️"

        print(f"\n{conf_indicator} Predicted Intent: {intent}")
        print(f"   Confidence: {confidence:.4f}")

        print("\n   Top 5 predictions:")
        top_k = predictor.predict_with_top_k(text, k=5)
        for i, (label, prob) in enumerate(top_k, 1):
            print(f"   {i}. {label}: {prob:.4f}")

    else:
        print("\n🔍 Running example predictions...")
        examples = [
            "I want to transfer money to my friend",
            "What's the weather like today?",
            "Book a flight to New York",
            "Turn off the lights in the living room",
            "What time is it in Tokyo?",
        ]

        for example in examples:
            print(f"\n📝 '{example}'")
            intent, confidence, is_high_conf = predictor.predict_single(example)
            conf_indicator = "✓" if is_high_conf else "⚠️"
            print(f"   {conf_indicator} {intent} (confidence: {confidence:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict intent from text")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="results/training_metadata.json",
        help="Path to training metadata",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to classify",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )

    args = parser.parse_args()
    main(args.model_path, args.metadata_path, args.text, args.interactive)
