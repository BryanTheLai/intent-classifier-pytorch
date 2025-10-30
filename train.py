"""Main training script for intent classification."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from src.config import Config
from src.dataset import CLINC150DataLoader, IntentDataset
from src.evaluate import Evaluator
from src.model import IntentClassifier
from src.trainer import Trainer
from src.utils import count_parameters, get_device, set_seed


def main(config_path: str) -> None:
    config = Config.from_yaml(Path(config_path))
    config.create_directories()

    set_seed(config.training.seed)
    device = get_device()

    print("\n" + "=" * 60)
    print("INTENT CLASSIFICATION TRAINING")
    print("=" * 60)

    print("\n📦 Loading dataset...")
    data_loader = CLINC150DataLoader(cache_dir=config.paths.cache_dir)
    data_loader.load_dataset()

    train_texts, train_labels = data_loader.get_split(config.data.train_split)
    val_texts, val_labels = data_loader.get_split(config.data.val_split)
    test_texts, test_labels = data_loader.get_split(config.data.test_split)

    num_intents = data_loader.get_num_labels()
    print(f"✓ Dataset loaded: {num_intents} intents")
    print(f"  Train samples: {len(train_texts)}")
    print(f"  Val samples: {len(val_texts)}")
    print(f"  Test samples: {len(test_texts)}")

    print("\n🔤 Initializing tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(config.model.name)

    print("\n📊 Creating datasets...")
    train_dataset = IntentDataset(
        train_texts, train_labels, tokenizer, config.model.max_length
    )
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer, config.model.max_length)
    test_dataset = IntentDataset(
        test_texts, test_labels, tokenizer, config.model.max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    print("\n🏗️ Building model...")
    model = IntentClassifier(
        num_intents=num_intents,
        model_name=config.model.name,
        dropout=config.model.dropout,
    )
    model.to(device)

    param_counts = count_parameters(model)
    print(f"✓ Model created")
    print(f"  Total parameters: {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")

    print("\n🎯 Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config.training.learning_rate,
        num_epochs=config.training.num_epochs,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        model_dir=config.paths.model_dir,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_min_delta=config.training.early_stopping_min_delta,
    )

    evaluator = Evaluator(
        model=model,
        device=device,
        id2label=data_loader.id2label,
    )

    print("\n🚀 Starting training...")
    trainer.train(evaluator)

    print("\n📈 Evaluating on test set...")
    test_metrics = evaluator.evaluate(test_loader)
    print("\nTest Set Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision: {test_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall: {test_metrics['macro_recall']:.4f}")

    results_path = Path(config.paths.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print("\n📊 Generating classification report...")
    report = evaluator.get_classification_report(
        test_loader,
        save_path=results_path / "classification_report.txt",
    )

    if config.evaluation.save_confusion_matrix:
        print("\n🎨 Generating confusion matrix...")
        evaluator.plot_confusion_matrix(
            test_loader,
            save_path=results_path / "confusion_matrix.png",
        )

    print("\n🔍 Analyzing confidence scores...")
    confidence_metrics = evaluator.analyze_confidence(
        test_loader,
        threshold=config.evaluation.confidence_threshold,
    )
    print(f"  Average confidence: {confidence_metrics['avg_confidence']:.4f}")
    print(
        f"  High confidence ratio: {confidence_metrics['high_confidence_ratio']:.4f}"
    )
    if "high_confidence_accuracy" in confidence_metrics:
        print(
            f"  High confidence accuracy: {confidence_metrics['high_confidence_accuracy']:.4f}"
        )

    metadata = {
        "num_intents": num_intents,
        "test_metrics": test_metrics,
        "confidence_metrics": confidence_metrics,
        "training_history": trainer.get_training_history(),
        "label_mappings": {
            "id2label": data_loader.id2label,
            "label2id": data_loader.label2id,
        },
    }

    with open(results_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Training completed successfully!")
    print(f"📁 Results saved to: {results_path}")
    print(f"💾 Best model saved to: {Path(config.paths.model_dir) / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train intent classification model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    main(args.config)
