"""Dataset loading and preprocessing for intent classification."""

from typing import Dict, List, Tuple

import torch
from datasets import ClassLabel, load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class IntentDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        if label is None:
            label = 0
        else:
            try:
                label = int(label)
            except Exception:
                label = 0

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class CLINC150DataLoader:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.dataset = None
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.value2id: dict = {}
        self.id2value: dict = {}
        self.text_col: str = "text"
        self.label_col: str = "intent"

    def load_dataset(self) -> None:
        self.dataset = load_dataset("DeepPavlov/clinc150", cache_dir=self.cache_dir)
        self._detect_columns()
        self._create_label_mappings()

    def _detect_columns(self) -> None:
        features = self.dataset["train"].features

        if "text" in features:
            self.text_col = "text"
        elif "utterance" in features:
            self.text_col = "utterance"
        else:
            for col, feat in features.items():
                if getattr(feat, "dtype", None) == "string":
                    self.text_col = col
                    break

        if "intent" in features:
            self.label_col = "intent"
        elif "label" in features:
            self.label_col = "label"
        else:
            for col, feat in features.items():
                if isinstance(feat, ClassLabel):
                    self.label_col = col
                    break
            else:
                for col, feat in features.items():
                    if col != self.text_col and getattr(feat, "dtype", None) in {"int32", "int64"}:
                        self.label_col = col
                        break

    def _create_label_mappings(self) -> None:
        features = self.dataset["train"].features
        label_feat = features[self.label_col]

        if isinstance(label_feat, ClassLabel):
            names = list(label_feat.names)
            self.label2id = {name: idx for idx, name in enumerate(names)}
            self.id2label = {idx: name for idx, name in enumerate(names)}
            self.value2id = {idx: idx for idx in range(len(names))}
            self.id2value = {idx: idx for idx in range(len(names))}
        else:
            all_labels = set()
            for split in ["train", "validation", "test"]:
                if split in self.dataset:
                    col_vals = [v for v in self.dataset[split][self.label_col] if v is not None]
                    all_labels.update(col_vals)

            unique_values = sorted(list(all_labels))
            self.value2id = {val: idx for idx, val in enumerate(unique_values)}
            self.id2value = {idx: val for idx, val in enumerate(unique_values)}
            self.id2label = {idx: str(val) for idx, val in self.id2value.items()}
            self.label2id = {str(val): idx for idx, val in enumerate(unique_values)}

    def get_split(self, split: str = "train") -> Tuple[List[str], List[int]]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        split_data = self.dataset[split]
        texts = split_data[self.text_col]

        features = self.dataset[split].features
        label_feat = features[self.label_col]

        raw_labels = split_data[self.label_col]

        clean_texts: List[str] = []
        clean_labels: List[int] = []

        if isinstance(label_feat, ClassLabel):
            for t, lbl in zip(texts, raw_labels):
                if lbl is None:
                    continue
                clean_texts.append(t)
                clean_labels.append(int(lbl))
        else:
            for t, lbl in zip(texts, raw_labels):
                if lbl is None:
                    continue
                mapped = self.value2id.get(lbl)
                if mapped is None:
                    mapped = self.value2id.get(str(lbl))
                if mapped is None:
                    continue
                clean_texts.append(t)
                clean_labels.append(int(mapped))

        return clean_texts, clean_labels

    def get_num_labels(self) -> int:
        return len(self.id2label)

    def get_label_name(self, label_id: int) -> str:
        return self.id2label.get(label_id, "unknown")
