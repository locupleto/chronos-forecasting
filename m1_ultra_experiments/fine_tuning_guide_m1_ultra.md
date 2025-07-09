
# Chronos Fine-Tuning Guide for M1 Ultra

## Overview
Fine-tuning allows you to adapt Chronos models to your specific domain and data patterns.
This can significantly improve performance for domain-specific forecasting tasks.

## Supported Models
- ✅ Chronos-T5 (tiny, mini, small, base, large)
- ⚠️  Chronos-Bolt (limited fine-tuning support)

## Prerequisites
```bash
pip install "chronos-forecasting[training]"
```

## Step 1: Prepare Your Data
Your time series data needs to be in GluonTS Arrow format:

```python
import numpy as np
from gluonts.dataset.arrow import ArrowWriter

def prepare_training_data(time_series_list, output_path):
    start = np.datetime64("2020-01-01 00:00", "D")
    dataset = [{"start": start, "target": ts} for ts in time_series_list]
    ArrowWriter(compression="lz4").write_to_file(dataset, path=output_path)
```

## Step 2: Create Training Configuration
Create a YAML config file:

```yaml
training_data_paths:
- "/path/to/your_data.arrow"
probability:
- 1.0
context_length: 128
prediction_length: 24
max_steps: 1000
learning_rate: 0.001
per_device_train_batch_size: 8
model_id: amazon/chronos-t5-tiny
random_init: false  # Fine-tune from pretrained
output_dir: ./fine_tuned_model/
```

## Step 3: Run Fine-Tuning
```bash
python scripts/training/train.py --config your_config.yaml
```

## M1 Ultra Optimization Tips
1. Use smaller batch sizes (4-8) for CPU training
2. Disable torch_compile for compatibility
3. Use regular AdamW optimizer instead of fused variants
4. Start with tiny/mini models for faster iteration
5. Monitor memory usage with Activity Monitor

## Expected Performance
- Tiny model: ~0.1s per training step
- Mini model: ~0.2s per training step
- Small model: ~0.5s per training step

## Post Fine-Tuning Usage
```python
from chronos import BaseChronosPipeline

# Load your fine-tuned model
pipeline = BaseChronosPipeline.from_pretrained(
    "./fine_tuned_model/checkpoint-1000",
    device_map="cpu"
)

# Use as normal
quantiles, mean = pipeline.predict_quantiles(context, prediction_length=12)
```
