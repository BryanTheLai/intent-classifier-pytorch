# GPU Acceleration Enabled ✓

## What Was Done

### 1. GPU Detection & Optimization
Updated `src/utils.py` to automatically detect and use your NVIDIA RTX 3050:

```python
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device
```

### 2. Your GPU Specs
- **GPU**: NVIDIA GeForce RTX 3050
- **CUDA Version**: 12.7
- **Driver**: 565.90
- **Memory**: 4096 MB (4 GB)

### 3. Expected Speed Improvement
- **CPU Training**: ~45-60 minutes per 5 epochs
- **GPU Training**: ~10-15 minutes per 5 epochs
- **Speedup**: **3-5x faster** ⚡

## How to Use

### Next Training Run
Simply run as before:
```powershell
python train.py
```

The script will automatically detect and use your GPU. You'll see output like:
```
✓ Using CUDA device: NVIDIA GeForce RTX 3050
  CUDA Version: 12.7
  GPU Memory: 4.00 GB
```

### Monitor GPU Usage
In another terminal, run:
```powershell
nvidia-smi
```

Watch the GPU Memory and GPU-Util columns increase during training.

## Training Results (CPU)

Your first training run completed successfully on CPU:

### Performance Metrics
- **Test Accuracy**: 95.4% 🎯
- **Macro F1**: 0.954
- **Weighted F1**: 0.954
- **Macro Precision**: 0.956
- **Macro Recall**: 0.954

### Confidence Analysis
- **Average Confidence**: 0.952
- **High Confidence Predictions**: 95%
- **High Confidence Accuracy**: 97.8%

### Training History
| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|-----------|----------|--------------|
| 1 | 2.545 | 0.780 | 0.897 |
| 2 | 0.475 | 0.308 | 0.943 |
| 3 | 0.165 | 0.213 | 0.958 |
| 4 | 0.082 | 0.194 | 0.959 |
| 5 | 0.055 | 0.190 | 0.962 |

## Intent Mapping

See `results/INTENT_MAPPING.md` for the complete list of all 150 intents.

### Example: Intent 83
- **Name**: `work_pto_request_cancel`
- **Domain**: Work
- **Example**: "Cancel my time off request"

## Next Steps

1. **Re-train with GPU** for faster iterations:
   ```powershell
   python train.py
   ```

2. **Try inference**:
   ```powershell
   python predict.py --interactive
   ```

3. **Experiment with hyperparameters** in `configs/config.yaml`:
   - Increase `num_epochs` to 10 for potentially better accuracy
   - Adjust `batch_size` (try 32 or 64 with GPU)
   - Tune `learning_rate` for different convergence speeds

## Troubleshooting

### GPU Not Detected
If you see "Using CPU device" instead:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with CUDA support if needed

### Out of Memory
If you get CUDA OOM errors:
1. Reduce `batch_size` in `configs/config.yaml` (try 8 or 4)
2. Reduce `num_workers` to 0 or 2
3. Clear GPU cache: `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill -9`

### Performance Not Improving
- Verify GPU is actually being used: `nvidia-smi` during training
- Check GPU utilization should be >80%
- If low, increase `batch_size`

## Files Modified

- `src/utils.py` - Added GPU detection and CUDA memory optimization
- `results/INTENT_MAPPING.md` - Complete intent reference guide

---

**Ready to train faster!** 🚀 Run `python train.py` to see your GPU in action.
