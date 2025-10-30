"""Configuration management for the intent classifier."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "distilbert-base-uncased"
    dropout: float = 0.3
    max_length: int = 128


@dataclass
class DataConfig:
    dataset_name: str = "DeepPavlov/clinc150"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class TrainingConfig:
    num_epochs: int = 5
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0


@dataclass
class EvaluationConfig:
    confidence_threshold: float = 0.7
    save_confusion_matrix: bool = True


@dataclass
class PathsConfig:
    model_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"
    cache_dir: str = "cache"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            paths=PathsConfig(**config_dict.get("paths", {})),
        )

    def create_directories(self) -> None:
        Path(self.paths.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.paths.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.paths.logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.paths.cache_dir).mkdir(parents=True, exist_ok=True)
