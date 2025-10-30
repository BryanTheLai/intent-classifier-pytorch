"""Training pipeline for intent classification."""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.evaluate import Evaluator
from src.utils import format_time, save_checkpoint, save_model


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 2e-5,
        num_epochs: int = 5,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        model_dir: str = "models",
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, evaluator: Evaluator) -> tuple[float, dict]:
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        metrics = evaluator.evaluate(self.val_loader)

        return avg_loss, metrics

    def train(self, evaluator: Evaluator) -> None:
        print(f"\nStarting training for {self.num_epochs} epochs...")
        print(f"Total training batches per epoch: {len(self.train_loader)}")
        print(f"Total validation batches: {len(self.val_loader)}\n")

        start_time = time.time()
        no_improve_epochs = 0

        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*60}")

            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate(evaluator)

            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_accuracy"].append(val_metrics["accuracy"])

            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val Macro F1: {val_metrics['macro_f1']:.4f}")
            print(f"  Epoch Time: {format_time(epoch_time)}")

            improved = (self.best_val_loss - val_loss) > self.early_stopping_min_delta
            if improved:
                self.best_val_loss = val_loss
                best_model_path = self.model_dir / "best_model.pt"
                save_model(self.model, best_model_path)
                print(f"  ✓ New best model saved!")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if self.early_stopping_patience > 0 and no_improve_epochs >= self.early_stopping_patience:
                    print(
                        f"\nEarly stopping triggered after {no_improve_epochs} epochs without improvement"
                    )
                    break

            checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch + 1,
                val_loss,
                checkpoint_path,
            )

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {format_time(total_time)}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

    def get_training_history(self) -> dict:
        return self.training_history
