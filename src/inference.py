"""Inference utilities for intent classification."""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import DistilBertTokenizer

from src.model import IntentClassifier


class IntentPredictor:
    def __init__(
        self,
        model: IntentClassifier,
        tokenizer: DistilBertTokenizer,
        id2label: Dict[int, str],
        device: torch.device,
        confidence_threshold: float = 0.7,
        max_length: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_length = max_length
        self.model.eval()

    def predict_single(
        self,
        text: str,
    ) -> Tuple[str, float, bool]:
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)

        predicted_label = self.id2label[predicted_class.item()]
        confidence_value = confidence.item()
        is_high_confidence = confidence_value >= self.confidence_threshold

        return predicted_label, confidence_value, is_high_confidence

    def predict_batch(
        self,
        texts: List[str],
    ) -> List[Tuple[str, float, bool]]:
        encodings = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predicted_classes = torch.max(probabilities, dim=-1)

        results = []
        for pred_class, conf in zip(predicted_classes, confidences):
            label = self.id2label[pred_class.item()]
            confidence_value = conf.item()
            is_high_confidence = confidence_value >= self.confidence_threshold
            results.append((label, confidence_value, is_high_confidence))

        return results

    def predict_with_top_k(
        self,
        text: str,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=-1)

        results = []
        for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
            label = self.id2label[idx.item()]
            results.append((label, prob.item()))

        return results

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        tokenizer_name: str,
        id2label: Dict[int, str],
        device: torch.device,
        confidence_threshold: float = 0.7,
        max_length: int = 128,
    ) -> "IntentPredictor":
        num_intents = len(id2label)
        model = IntentClassifier(num_intents=num_intents, model_name=tokenizer_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

        return cls(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            confidence_threshold=confidence_threshold,
            max_length=max_length,
        )
