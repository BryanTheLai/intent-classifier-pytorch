# Quick Start Guide

This guide walks you through training and running the intent classifier.

## Step-by-Step Setup

### 1. Install UV Package Manager

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup Environment

```bash
# Navigate to project directory
cd intent-classifier-pytorch

# Create virtual environment
uv venv

# Activate environment
# Windows PowerShell:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -e .

### GPU Setup (Windows + uv)

If training says "Using CPU device" but you have an NVIDIA GPU, install the CUDA-enabled PyTorch wheels.

1) Verify current status

```powershell
.venv\Scripts\python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'built for', torch.version.cuda)"
```

2) Install CUDA wheels (CUDA 12.x). Try cu124 first; if unavailable, use cu121.

```powershell
uv pip uninstall torch torchvision torchaudio

# CUDA 12.4 wheels
uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# Fallback (CUDA 12.1):
# uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

3) Re-verify

```powershell
.venv\Scripts\python -c "import torch; print('cuda available?', torch.cuda.is_available(), 'runtime cuda', torch.version.cuda)"
```

4) Train

```powershell
python train.py
```

Expected header:

```
✓ Using CUDA device: NVIDIA GeForce RTX 3050
  CUDA Version: 12.x
  GPU Memory: 4.00 GB
```

### 3. Train the Model

```bash
python train.py
```

**This will:**
- Download CLINC150 dataset (23,700 examples)
- Train DistilBERT classifier for 5 epochs
- Save best model to `models/best_model.pt`
- Generate evaluation metrics and visualizations

**Expected time:**
- **CPU only**: ~45-60 minutes
- **GPU (CUDA)**: ~15-20 minutes

### 4. Run Predictions

**Interactive mode:**
```bash
python predict.py --interactive
```

**Single prediction:**
```bash
python predict.py --text "I want to transfer money"
```

**Example predictions:**
```bash
python predict.py
```

## What You Get

After training, you'll have:

```
models/
├── best_model.pt                    # Best trained model
└── checkpoint_epoch_*.pt            # Epoch checkpoints

results/
├── classification_report.txt        # Per-class metrics
├── confusion_matrix.png            # Visual confusion matrix
└── training_metadata.json          # Training history & label mappings
```

## Expected Performance

- **Accuracy**: 88-91%
- **Macro F1**: 87-90%
- **Training time**: 15-60 minutes (depending on hardware)

## Common Commands

```bash
# Train with custom config
python train.py --config configs/config.yaml

# Interactive predictions
python predict.py --interactive

# Single prediction
python predict.py --text "Book a flight to NYC"

# Run tests
pytest tests/

# Format code
black src/ train.py predict.py
```

## Adjusting for Your Hardware

**Low Memory GPU?** Edit `configs/config.yaml`:
```yaml
data:
  batch_size: 8  # Reduce from 16
  num_workers: 2  # Reduce from 4
```

**CPU Only?**
- Training will work but take longer (~45-60 min)
- Consider reducing epochs to 3 for faster testing

**Powerful GPU?** Increase batch size:
```yaml
data:
  batch_size: 32  # Increase from 16
```

## Next Steps

1. **Test inference**: Run `python predict.py --interactive`
2. **Check results**: Open `results/confusion_matrix.png`
3. **Read full docs**: See [README.md](README.md)
4. **Customize**: Edit `configs/config.yaml` for your use case

## Troubleshooting

**Dataset download slow?**
- First run downloads ~50MB from HuggingFace
- Cached for future runs

**CUDA out of memory?**
- Reduce `batch_size` in config
- Close other GPU applications

**Import errors?**
- Ensure virtual environment is activated
- Run `uv pip install -e .` again

## Getting Help

- Check [README.md](README.md) for detailed documentation
- See [examples/quick_start.py](examples/quick_start.py) for code examples
- Review [configs/config.yaml](configs/config.yaml) for all options

---

**Need more help?** Open an issue on GitHub or check the full README.
