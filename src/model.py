"""Model architecture for intent classification."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import DistilBertModel


class IntentClassifier(nn.Module):
    def __init__(self, num_intents: int, model_name: str = "distilbert-base-uncased", dropout: float = 0.3):
        super(IntentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_intents)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities

    def freeze_bert_encoder(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
