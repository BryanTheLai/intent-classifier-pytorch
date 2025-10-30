"""Tests for the model module."""

import torch
from src.model import IntentClassifier


def test_model_forward():
    model = IntentClassifier(num_intents=150)
    batch_size = 4
    seq_length = 128

    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    logits = model(input_ids, attention_mask)

    assert logits.shape == (batch_size, 150)


def test_model_predict():
    model = IntentClassifier(num_intents=150)
    model.eval()

    batch_size = 4
    seq_length = 128

    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    predictions, probabilities = model.predict(input_ids, attention_mask)

    assert predictions.shape == (batch_size,)
    assert probabilities.shape == (batch_size, 150)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_freeze_unfreeze():
    model = IntentClassifier(num_intents=150)

    initial_trainable = model.get_trainable_parameters()

    model.freeze_bert_encoder()
    frozen_trainable = model.get_trainable_parameters()

    model.unfreeze_bert_encoder()
    unfrozen_trainable = model.get_trainable_parameters()

    assert frozen_trainable < initial_trainable
    assert unfrozen_trainable == initial_trainable


if __name__ == "__main__":
    test_model_forward()
    test_model_predict()
    test_freeze_unfreeze()
    print("✅ All tests passed!")
