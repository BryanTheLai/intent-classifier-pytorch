"""Evaluation metrics and utilities."""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        id2label: Dict[int, str],
    ):
        self.model = model
        self.device = device
        self.id2label = id2label

    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
    ) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = self._calculate_metrics(all_labels, all_preds)

        if return_predictions:
            return metrics, all_preds, all_labels, all_probs
        return metrics

    def _calculate_metrics(
        self,
        labels: List[int],
        predictions: List[int],
    ) -> Dict[str, float]:
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
            "weighted_f1": f1_score(labels, predictions, average="weighted"),
            "macro_precision": precision_score(labels, predictions, average="macro"),
            "macro_recall": recall_score(labels, predictions, average="macro"),
        }

    def get_classification_report(
        self,
        dataloader: DataLoader,
        save_path: Path = None,
    ) -> str:
        _, predictions, labels, _ = self.evaluate(dataloader, return_predictions=True)

        target_names = [self.id2label[i] for i in range(len(self.id2label))]
        report = classification_report(
            labels,
            predictions,
            target_names=target_names,
            digits=4,
        )

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report)
            print(f"Classification report saved to {save_path}")

        return report

    def plot_confusion_matrix(
        self,
        dataloader: DataLoader,
        save_path: Path,
        normalize: bool = True,
        figsize: Tuple[int, int] = (20, 20),
    ) -> None:
        _, predictions, labels, _ = self.evaluate(dataloader, return_predictions=True)

        cm = confusion_matrix(labels, predictions)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=False,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=[self.id2label[i] for i in range(len(self.id2label))],
            yticklabels=[self.id2label[i] for i in range(len(self.id2label))],
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    def analyze_confidence(
        self,
        dataloader: DataLoader,
        threshold: float = 0.7,
    ) -> Dict[str, float]:
        _, predictions, labels, probabilities = self.evaluate(
            dataloader, return_predictions=True
        )

        max_probs = np.max(probabilities, axis=1)
        high_confidence = max_probs >= threshold
        low_confidence = max_probs < threshold

        metrics = {
            "avg_confidence": float(np.mean(max_probs)),
            "high_confidence_ratio": float(np.mean(high_confidence)),
            "low_confidence_ratio": float(np.mean(low_confidence)),
        }

        if np.sum(high_confidence) > 0:
            high_conf_acc = accuracy_score(
                np.array(labels)[high_confidence],
                np.array(predictions)[high_confidence],
            )
            metrics["high_confidence_accuracy"] = float(high_conf_acc)

        if np.sum(low_confidence) > 0:
            low_conf_acc = accuracy_score(
                np.array(labels)[low_confidence],
                np.array(predictions)[low_confidence],
            )
            metrics["low_confidence_accuracy"] = float(low_conf_acc)

        return metrics
