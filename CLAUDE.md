# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Project Context

The main task we are working on is found in this Obsidian-compatible documentation located in the vault at `/Users/urban/Library/Mobile Documents/iCloud~md~obsidian/Documents/ObsidianVault/Projects/Chronos/Experiments.md`

Claude is working and updating this md-file together with the user. Always have this overall goal in mind: **Learn how to deploy an out-of-the-box Chronos-Bolt model on first an example dataset here and finally on a dataset of our own. We must use a local deployment on an Apple Silicon Mac (M1 Ultra 64GB RAM).**

When editing this file always use full Obsidian syntax (internal links `[[]]`, tags `#`, callouts `> [!note]`, Mermaid diagrams), and ask for clarification if location/scope is ambiguous by listing 2-3 alternatives. Include: title, abstract callout, tags, main content with hierarchy, cross-references, and metadata. Make it comprehensive, actionable, visual, and well-linked.

## Project Overview

Chronos is a family of pretrained time series forecasting models based on language model architectures. The repository contains:

- **Chronos-T5**: Original T5-based models for time series forecasting
- **Chronos-Bolt**: Newer, more efficient models that are 5% more accurate and up to 250x faster
- Training scripts for pretraining and fine-tuning models
- Evaluation scripts for benchmarking on various time series datasets

## Architecture

The codebase implements two main model architectures:

### Core Components (`src/chronos/`)

1. **Base Pipeline** (`base.py`):
   - `BaseChronosPipeline`: Abstract base class for all Chronos models
   - `ForecastType`: Enum for SAMPLES vs QUANTILES prediction types
   - `PipelineRegistry`: Metaclass for registering pipeline implementations

2. **Chronos-T5** (`chronos.py`):
   - `ChronosPipeline`: Main pipeline for T5-based models
   - `ChronosTokenizer`: Converts time series to tokens (MeanScaleUniformBins implementation)
   - `ChronosModel`: Wrapper around HuggingFace T5 models
   - Supports both encoder-decoder (seq2seq) and decoder-only (causal) architectures

3. **Chronos-Bolt** (`chronos_bolt.py`):
   - `ChronosBoltPipeline`: Pipeline for efficient Bolt models
   - `ChronosBoltModelForForecasting`: Custom T5-based model with patching
   - Uses instance normalization and patch-based input processing
   - Directly predicts quantiles rather than sampling

## Commands

### Installation and Setup

```bash
# Install from PyPI for inference only
pip install chronos-forecasting

# Install from source for development/training
pip install --editable ".[training]"

# Install for evaluation
pip install --editable ".[evaluation]"
```

### Running Tests

```bash
# Run all tests
pytest test/

# Run specific test files
pytest test/test_chronos.py
pytest test/test_chronos_bolt.py
pytest test/test_utils.py

# Run with additional dependencies
pytest test/ --verbose
```

### Type Checking

```bash
# Run mypy for type checking
mypy src/chronos/
```

### Training

```bash
# Train on single GPU
CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py --config path/to/config.yaml

# Train on multiple GPUs
torchrun --nproc-per-node=8 scripts/training/train.py --config path/to/config.yaml

# Fine-tune existing model
python scripts/training/train.py --config path/to/config.yaml \
    --model-id amazon/chronos-t5-small \
    --no-random-init \
    --max-steps 1000 \
    --learning-rate 0.001
```

### Evaluation

```bash
# In-domain evaluation
python scripts/evaluation/evaluate.py scripts/evaluation/configs/in-domain.yaml results.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20

# Zero-shot evaluation
python scripts/evaluation/evaluate.py scripts/evaluation/configs/zero-shot.yaml results.csv \
    --chronos-model-id "amazon/chronos-t5-small" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20
```

### Data Generation

```bash
# Generate synthetic time series using KernelSynth
python scripts/kernel-synth.py --num-series 1000000 --max-kernels 5
```

## Key Design Patterns

### Model Loading

Models are loaded using the factory pattern through `BaseChronosPipeline.from_pretrained()`:

```python
from chronos import BaseChronosPipeline

# Automatically detects model type (Chronos-T5 vs Chronos-Bolt)
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-t5-small")
```

### Tokenization

Time series are converted to tokens using scaling and quantization:

1. **Scaling**: Time series are normalized using mean scaling
2. **Quantization**: Values are mapped to discrete tokens using uniform bins
3. **Special tokens**: PAD and EOS tokens are added as needed

### Prediction Methods

- `predict()`: Returns sample trajectories (Chronos-T5) or quantiles (Chronos-Bolt)
- `predict_quantiles()`: Returns quantile forecasts and mean predictions
- `embed()`: Extracts encoder embeddings for analysis

### Context Handling

- Input context is automatically padded/truncated to model's context length
- Missing values (`torch.nan`) are handled appropriately
- Batch processing supports variable-length series with left-padding

## File Structure

```
src/chronos/
├── __init__.py          # Main exports
├── base.py              # Base pipeline and registry
├── chronos.py           # Chronos-T5 implementation
├── chronos_bolt.py      # Chronos-Bolt implementation
└── utils.py             # Utility functions

scripts/
├── training/
│   ├── train.py         # Training script
│   └── configs/         # Training configurations
├── evaluation/
│   ├── evaluate.py      # Evaluation script
│   └── configs/         # Evaluation configurations
└── kernel-synth.py      # Synthetic data generation

test/
├── test_chronos.py      # Tests for Chronos-T5
├── test_chronos_bolt.py # Tests for Chronos-Bolt
├── test_utils.py        # Utility tests
└── util.py              # Test utilities
```

## Model Configurations

### Chronos-T5 Models
- chronos-t5-tiny (8M parameters)
- chronos-t5-mini (20M parameters)
- chronos-t5-small (46M parameters)
- chronos-t5-base (200M parameters)
- chronos-t5-large (710M parameters)

### Chronos-Bolt Models
- chronos-bolt-tiny (9M parameters)
- chronos-bolt-mini (21M parameters)
- chronos-bolt-small (48M parameters)
- chronos-bolt-base (205M parameters)

## Common Development Patterns

1. **Loading models**: Always use `BaseChronosPipeline.from_pretrained()` for automatic type detection
2. **Handling context**: Use `_prepare_and_validate_context()` for input preprocessing
3. **Device management**: Models handle device placement automatically
4. **Configuration**: Model-specific configs are stored in `chronos_config` within HuggingFace config
5. **Scaling**: Both architectures use instance normalization for time series scaling

## Important Notes

- Models expect `torch.float32` input for time series data
- Use `torch.nan` for missing values in time series
- Chronos-Bolt is optimized for quantile prediction, Chronos-T5 for sampling
- Context length limits vary by model (typically 512-2048 time steps)
- Prediction length should not exceed model's training prediction length for best results