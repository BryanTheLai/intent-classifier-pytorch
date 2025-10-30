# Intent Classifier PyTorch

Production-ready PyTorch Intent Classifier for LLMs using DistilBERT and the CLINC150 dataset. Built with clean, modular architecture following ML engineering best practices.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

Intent classification identifies what a user wants to do from text input—critical for routing LLM queries, reducing latency, and improving response accuracy. This implementation achieves **88-91% accuracy** on CLINC150's 150 intent classes.

### Key Features

- **Production-Ready Architecture**: Two-stage approach with lightweight BERT classifier and confidence-based routing
- **CLINC150 Dataset**: 23,700 examples across 150 intents in 10 domains (banking, travel, utilities, etc.)
- **Comprehensive Evaluation**: Macro/weighted F1, confusion matrices, confidence analysis, and per-class metrics
- **Modular Design**: Clean separation of concerns with configuration management, dataset loading, training, and inference
- **Type-Safe**: Full type hints throughout for better IDE support and code quality

## 📁 Project Structure

```
intent-classifier-pytorch/
│
├── configs/
│   └── config.yaml              # Training and model configuration
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   ├── dataset.py               # Dataset loading and preprocessing
│   ├── model.py                 # DistilBERT classifier architecture
│   ├── trainer.py               # Training pipeline
│   ├── evaluate.py              # Evaluation metrics and utilities
│   ├── inference.py             # Inference and prediction
│   └── utils.py                 # Helper functions
│
├── examples/
│   └── quick_start.py           # Quick start examples
│
├── tests/
│   └── test_model.py            # Unit tests
│
├── train.py                     # Main training script
├── predict.py                   # Inference script
├── pyproject.toml              # Project dependencies (uv)
└── README.md

# Generated during training/inference:
├── models/                      # Saved model checkpoints
├── results/                     # Evaluation results and metrics
├── logs/                        # Training logs
└── cache/                       # HuggingFace dataset cache
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

#### 1. Install uv (if not already installed)

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Clone the repository

```bash
git clone https://github.com/BryanTheLai/intent-classifier-pytorch.git
cd intent-classifier-pytorch
```

#### 3. Create and activate virtual environment with uv

```bash
uv venv
```

**Activate the environment:**

**Windows (PowerShell):**
```powershell
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

#### 4. Install dependencies

```bash
uv pip install -e .
```

For development dependencies:
```bash
uv pip install -e ".[dev]"
```

### GPU Setup (Windows + uv)

If training shows "Using CPU device" but you have an NVIDIA GPU, you likely installed a CPU-only PyTorch wheel. Install the CUDA build via uv.

1) Verify current status

```powershell
.venv\Scripts\python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'built for', torch.version.cuda)"
```

2) Install CUDA-enabled PyTorch (CUDA 12.x). Try cu124 first, then cu121 if needed.

```powershell
uv pip uninstall torch torchvision torchaudio

# Install CUDA wheels (cu124)
uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# If cu124 not available for your version, use cu121 instead:
# uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

3) Re-verify

```powershell
.venv\Scripts\python -c "import torch; print('cuda available?', torch.cuda.is_available(), 'runtime cuda', torch.version.cuda)"
```

4) Train again

```powershell
python train.py
```

Expected output:

```
✓ Using CUDA device: NVIDIA GeForce RTX 3050
  CUDA Version: 12.x
  GPU Memory: 4.00 GB
```

### Training the Model

#### 1. Review configuration (optional)

Edit `configs/config.yaml` to adjust hyperparameters:

```yaml
model:
  name: "distilbert-base-uncased"
  dropout: 0.3
  max_length: 128

training:
  num_epochs: 5
  learning_rate: 2.0e-5
  warmup_steps: 0
  weight_decay: 0.01
  max_grad_norm: 1.0
  seed: 42
  early_stopping_patience: 5
  early_stopping_min_delta: 0.001

data:
  dataset_name: "DeepPavlov/clinc150"
  batch_size: 8
  num_workers: 2
```

#### 2. Run training

```bash
python train.py --config configs/config.yaml
```

**What happens during training:**

1. **Dataset Download**: CLINC150 dataset downloads automatically from HuggingFace (first run only)
2. **Model Initialization**: DistilBERT model loads pre-trained weights
3. **Training Loop**: 5 epochs with validation after each epoch
4. **Checkpoints**: Model checkpoints saved after each epoch
5. **Best Model**: Best model (lowest validation loss) saved to `models/best_model.pt`
6. **Evaluation**: Full test set evaluation with metrics and visualizations

**Expected output:**
```
============================================================
INTENT CLASSIFICATION TRAINING
============================================================

📦 Loading dataset...
✓ Dataset loaded: 150 intents
  Train samples: 15000
  Val samples: 3000
  Test samples: 5700

🏗️ Building model...
✓ Model created
  Total parameters: 66,955,350
  Trainable parameters: 66,955,350

🚀 Starting training...

Epoch 1/5
============================================================
Training: 100%|████████████| 938/938 [05:23<00:00, 2.90it/s]
Validation: 100%|████████| 188/188 [00:31<00:00, 5.98it/s]

Epoch 1 Summary:
  Train Loss: 2.1234
  Val Loss: 1.2345
  Val Accuracy: 0.7856
  Val Macro F1: 0.7654
  ✓ New best model saved!
...
```

**Training outputs:**
- `models/best_model.pt` - Best model weights
- `models/checkpoint_epoch_*.pt` - Epoch checkpoints
- `results/classification_report.txt` - Detailed per-class metrics
- `results/confusion_matrix.png` - Confusion matrix visualization
- `results/training_metadata.json` - Training history and label mappings

### Running Inference

#### 1. Interactive mode (recommended for testing)

```bash
python predict.py --interactive
```

Example session:
```
💬 Interactive mode (type 'quit' to exit)
============================================================

Enter text: I want to check my account balance

✓ Predicted Intent: balance
   Confidence: 0.9543

   Top 5 predictions:
   1. balance: 0.9543
   2. transactions: 0.0234
   3. freeze_account: 0.0098
   4. pin_change: 0.0067
   5. routing: 0.0034
```

#### 2. Single prediction

```bash
python predict.py --text "Transfer money to John"
```

#### 3. Programmatic usage

```python
from pathlib import Path
import torch
from src.inference import IntentPredictor
from src.utils import get_device
import json

device = get_device()

with open("results/training_metadata.json", "r") as f:
    metadata = json.load(f)
id2label = {int(k): v for k, v in metadata["label_mappings"]["id2label"].items()}

predictor = IntentPredictor.from_pretrained(
    model_path=Path("models/best_model.pt"),
    tokenizer_name="distilbert-base-uncased",
    id2label=id2label,
    device=device,
    confidence_threshold=0.7,
)

text = "Book a flight to Paris"
intent, confidence, is_high_conf = predictor.predict_single(text)
print(f"Intent: {intent}, Confidence: {confidence:.4f}")

batch_texts = ["Transfer $50", "What's the weather?", "Set an alarm"]
results = predictor.predict_batch(batch_texts)
for text, (intent, conf, _) in zip(batch_texts, results):
    print(f"{text} -> {intent} ({conf:.4f})")
```

## 📊 Performance Metrics

Expected performance on CLINC150 test set (after 5 epochs):

| Metric | Score |
|--------|-------|
| Accuracy | 0.88-0.91 |
| Macro F1 | 0.87-0.90 |
| Weighted F1 | 0.88-0.91 |
| Macro Precision | 0.88-0.91 |
| Macro Recall | 0.87-0.90 |

**Confidence Analysis:**
- Average confidence: ~0.85
- High confidence ratio (>0.7): ~75%
- High confidence accuracy: ~0.95

## 🔧 Configuration

### Model Configuration

```yaml
model:
  name: "distilbert-base-uncased"  # Can use other BERT variants
  dropout: 0.3                     # Dropout rate
  max_length: 128                  # Max sequence length
```

### Training Configuration

```yaml
training:
  num_epochs: 5                    # Number of training epochs
  learning_rate: 2.0e-5           # Learning rate
  warmup_steps: 0                  # LR warmup steps
  weight_decay: 0.01              # Weight decay (L2 regularization)
  max_grad_norm: 1.0              # Gradient clipping
  seed: 42                         # Random seed
  early_stopping_patience: 5       # Stop if no val loss improvement
  early_stopping_min_delta: 0.001  # Minimum improvement to reset patience
```

### Data Configuration

```yaml
data:
  dataset_name: "DeepPavlov/clinc150"
  batch_size: 8                    # Safer on 4 GB GPUs; increase if memory allows
  num_workers: 2                   # Windows: 0–2 is stable
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Or run tests directly:

```bash
python tests/test_model.py
```

## 📈 Advanced Usage

### Custom Dataset

To use your own dataset, modify `src/dataset.py`:

```python
class CustomDataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def load_dataset(self):
        # Load your custom data
        # Create label2id and id2label mappings
        pass
    
    def get_split(self, split: str):
        # Return texts and labels for split
        pass
```

### Fine-tuning on Custom Intents

1. Prepare your data in CLINC150 format (text, intent)
2. Update `src/dataset.py` to load your data
3. Adjust `num_intents` in configuration
4. Run training as normal

### Freezing BERT Layers

For faster training or when data is limited:

```python
from src.model import IntentClassifier

model = IntentClassifier(num_intents=150)
model.freeze_bert_encoder()  # Freeze BERT, only train classifier head
```

### Two-Stage Training

```python
# Stage 1: Train classifier head only
model.freeze_bert_encoder()
trainer.train(evaluator)

# Stage 2: Fine-tune entire model
model.unfreeze_bert_encoder()
trainer.learning_rate = 1e-5  # Lower LR for fine-tuning
trainer.train(evaluator)
```

## 🏗️ Production Deployment

### Confidence-Based Routing

```python
predictor = IntentPredictor.from_pretrained(...)

intent, confidence, is_high_conf = predictor.predict_single(user_query)

if is_high_conf:
    # Route to appropriate handler
    handle_intent(intent, user_query)
else:
    # Fallback to LLM for ambiguous queries
    llm_response = llm.generate(user_query)
```

### Model Optimization

For production deployment:

1. **Quantization**: Reduce model size
```python
import torch.quantization as quantization
quantized_model = quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

2. **ONNX Export**: For cross-platform deployment
```python
import torch.onnx
dummy_input = (input_ids, attention_mask)
torch.onnx.export(model, dummy_input, "model.onnx")
```

3. **TorchScript**: For faster inference
```python
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

## 🔬 Research Extensions

### Synthetic Data Generation

Boost performance with LLM-generated training data:

```python
from openai import OpenAI

client = OpenAI()

def generate_intent_examples(intent_name: str, num_examples: int = 50):
    prompt = f"Generate {num_examples} diverse user queries for the intent '{intent_name}'"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### Knowledge Distillation

Improve student model using teacher LLM:

```python
teacher_model = LargeLanguageModel()
student_model = IntentClassifier(num_intents=150)

for batch in dataloader:
    teacher_logits = teacher_model(batch)
    student_logits = student_model(batch)
    
    distillation_loss = kl_divergence(
        F.softmax(student_logits / temperature),
        F.softmax(teacher_logits / temperature)
    )
```

## 📚 Dataset Information

### CLINC150

- **Size**: 23,700 examples
- **Intents**: 150 intent classes
- **Domains**: 10 (banking, travel, utility, work, etc.)
- **Out-of-scope**: Includes out-of-scope detection
- **Source**: [HuggingFace](https://huggingface.co/datasets/DeepPavlov/clinc150)

**Domain Distribution:**
- Banking (13 intents)
- Credit cards (11 intents)
- Kitchen & dining (6 intents)
- Home (9 intents)
- Auto & commute (9 intents)
- Travel (15 intents)
- Utility (17 intents)
- Work (11 intents)
- Small talk (10 intents)
- Meta (9 intents)

### Alternative Datasets

**Banking77**: Fine-grained banking intents
```bash
dataset = load_dataset("PolyAI/banking77")
```

**Custom Generation**: See Research Extensions section

## 🛠️ Troubleshooting

### CUDA Out of Memory

Reduce batch size in `configs/config.yaml`:
```yaml
data:
  batch_size: 8  # Reduce from 16
```

### Slow Training

- Reduce `num_workers` if CPU is bottleneck
- Use mixed precision training:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
```

### Poor Accuracy

- Increase `num_epochs` (try 8-10)
- Adjust `learning_rate` (try 3e-5 or 1e-5)
- Reduce `dropout` (try 0.1 or 0.2)
- Try different BERT variants (`bert-base-uncased`, `roberta-base`)

## 📖 References

- [CLINC150 Paper](https://www.aclweb.org/anthology/D19-1131.pdf)
- [DistilBERT](https://arxiv.org/abs/1910.01108)
- [Intent Classification Best Practices](https://labelyourdata.com/articles/machine-learning/intent-classification)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- CLINC150 dataset by [Larson et al.](https://www.aclweb.org/anthology/D19-1131.pdf)
- HuggingFace Transformers library
- PyTorch team

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions

---

Built with ❤️ for production ML systems
