# Chronos M1 Ultra Experiments

This directory contains all the experiments, tests, and documentation for running Chronos time series forecasting models on Apple Silicon M1 Ultra Mac.

## 📁 Directory Structure

```
m1_ultra_experiments/
├── README.md                           # This file
├── test_m1_basic.py                    # Basic functionality tests
├── advanced_m1_test.py                 # Model comparison and stress tests
├── test_large_models.py                # Comprehensive all-model testing
├── custom_dataset_example.py           # Custom dataset integration examples
├── test_fine_tuning_m1.py              # Fine-tuning capability tests
├── working_fine_tuning_example.py      # Complete fine-tuning workflow
├── ultimate_chronos_m1_demo.py         # Complete ecosystem demonstration
├── chronos_m1_results.txt              # Performance benchmarks
├── comprehensive_performance_report.md # Detailed performance analysis
├── deployment_guide.md                 # Usage patterns and best practices
├── fine_tuning_guide_m1_ultra.md      # M1 Ultra fine-tuning guide
├── ultimate_chronos_m1_report.md      # Final comprehensive report
└── custom_datasets/                    # Sample datasets
    ├── stock_prices.csv
    ├── temperature_sensors.csv
    └── retail_sales.csv
```

## 🚀 Quick Start

### Run All Tests
```bash
# Navigate to the chronos-forecasting directory
cd /path/to/chronos-forecasting

# Run basic tests
python m1_ultra_experiments/test_m1_basic.py

# Run comprehensive model comparison
python m1_ultra_experiments/test_large_models.py

# Run the ultimate demo (everything combined)
python m1_ultra_experiments/ultimate_chronos_m1_demo.py
```

### Test Custom Datasets
```bash
# Test with custom dataset examples
python m1_ultra_experiments/custom_dataset_example.py
```

### Fine-Tuning Tests
```bash
# Test fine-tuning capabilities
python m1_ultra_experiments/test_fine_tuning_m1.py

# Complete fine-tuning workflow
python m1_ultra_experiments/working_fine_tuning_example.py
```

## 📊 Test Scripts Overview

### 1. `test_m1_basic.py`
- **Purpose**: Basic functionality verification
- **Tests**: AirPassengers dataset, synthetic data
- **Models**: chronos-bolt-tiny
- **Runtime**: ~30 seconds

### 2. `advanced_m1_test.py`
- **Purpose**: Model comparison and stress testing
- **Tests**: Multiple model sizes, batch processing, long sequences
- **Models**: tiny, mini, small
- **Runtime**: ~2 minutes

### 3. `test_large_models.py`
- **Purpose**: Comprehensive testing of all model sizes
- **Tests**: All 4 model sizes with memory monitoring
- **Models**: tiny, mini, small, base (9M to 205M parameters)
- **Runtime**: ~3 minutes

### 4. `custom_dataset_example.py`
- **Purpose**: Custom dataset integration demonstration
- **Tests**: Stock prices, temperature sensors, retail sales
- **Features**: Data preparation, error handling
- **Runtime**: ~1 minute

### 5. `test_fine_tuning_m1.py`
- **Purpose**: Fine-tuning capability testing
- **Tests**: Data preparation, configuration validation
- **Features**: GluonTS Arrow format conversion
- **Runtime**: ~30 seconds

### 6. `working_fine_tuning_example.py`
- **Purpose**: Complete fine-tuning workflow
- **Tests**: End-to-end fine-tuning preparation
- **Features**: Realistic training data, config generation
- **Runtime**: ~1 minute

### 7. `ultimate_chronos_m1_demo.py`
- **Purpose**: Complete ecosystem demonstration
- **Tests**: All models, datasets, fine-tuning prep
- **Features**: Comprehensive reporting
- **Runtime**: ~5 minutes

## 📈 Performance Results Summary

### Model Performance (M1 Ultra)
| Model | Parameters | Load Time | Inference Time | Memory Usage |
|-------|-----------|-----------|----------------|--------------|
| chronos-bolt-tiny | 9M | 0.61s | 0.051s | 0.45GB |
| chronos-bolt-mini | 21M | 0.82s | 0.119s | 0.50GB |
| chronos-bolt-small | 48M | 0.58s | 0.261s | 0.61GB |
| chronos-bolt-base | 205M | 0.64s | 1.384s | 1.22GB |

### Dataset Compatibility
- **E-commerce Sales**: 100 points → 14 predictions (0.012s)
- **Server Metrics**: 72 points → 24 predictions (0.008s)
- **Financial Returns**: 200 points → 30 predictions (0.008s)
- **Energy Consumption**: 168 points → 48 predictions (0.007s)

## 🔧 Requirements

### Python Dependencies
```bash
pip install chronos-forecasting pandas numpy psutil

# For fine-tuning tests
pip install "chronos-forecasting[training]"
```

### System Requirements
- **Hardware**: Apple Silicon Mac (M1/M1 Pro/M1 Max/M1 Ultra)
- **Memory**: 8GB+ (16GB+ recommended for larger models)
- **Storage**: 5GB+ free space for models and data

## 📚 Documentation

### Performance Reports
- `comprehensive_performance_report.md`: Detailed benchmarks
- `ultimate_chronos_m1_report.md`: Final comprehensive analysis
- `chronos_m1_results.txt`: Raw performance data

### Guides
- `deployment_guide.md`: Usage patterns and best practices
- `fine_tuning_guide_m1_ultra.md`: Complete fine-tuning guide

## 🎯 Key Findings

1. **Outstanding Performance**: All models run efficiently on M1 Ultra
2. **Memory Efficient**: Even 205M parameter model uses <2GB RAM
3. **Fast Inference**: Sub-second to ~1.4s prediction times
4. **Versatile**: Handles diverse dataset types and sizes
5. **Fine-Tuning Ready**: Complete workflow for domain adaptation

## 🛠️ Troubleshooting

### Common Issues
1. **Memory errors**: Use smaller models or reduce batch sizes
2. **Import errors**: Ensure all dependencies are installed
3. **Path errors**: Run scripts from chronos-forecasting root directory

### Performance Tips
- Use `chronos-bolt-tiny` for development/testing
- Use `chronos-bolt-small` for production balance
- Use `chronos-bolt-base` for maximum accuracy
- Monitor memory usage with Activity Monitor

## 🔄 Updates

This experiment suite is designed to be:
- **Extensible**: Easy to add new tests and datasets
- **Maintainable**: Clear structure and documentation
- **Reproducible**: All results can be replicated
- **Educational**: Learn M1 Ultra optimization techniques

## 📞 Support

For questions or issues:
1. Check the main [CLAUDE.md](../CLAUDE.md) file
2. Review performance reports in this directory
3. Run `ultimate_chronos_m1_demo.py` for a complete system check

---

*Created as part of comprehensive Chronos M1 Ultra testing and optimization.*