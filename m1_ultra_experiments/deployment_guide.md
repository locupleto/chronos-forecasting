
# Custom Dataset Deployment Guide

## 1. Data Preparation
- Ensure your time series data is in a single column
- Handle missing values (NaN) appropriately
- Consider data normalization if values are very large/small
- Time series should have regular intervals (daily, hourly, etc.)

## 2. Basic Usage Pattern
```python
import torch
from chronos import BaseChronosPipeline

# Load your data
data = pd.read_csv('your_data.csv')
context = torch.tensor(data['value_column'].values, dtype=torch.float32)

# Load model
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-tiny",  # Start with tiny for testing
    device_map="cpu",
    torch_dtype=torch.float32,
)

# Make predictions
quantiles, mean = pipeline.predict_quantiles(
    context=context,
    prediction_length=12,  # Adjust based on your needs
    quantile_levels=[0.1, 0.5, 0.9],
)
```

## 3. Model Selection
- **tiny**: Fast prototyping, good for initial tests
- **mini**: Better accuracy, still fast
- **small**: Highest accuracy tested, still CPU-friendly
- **base**: Use only if you need maximum accuracy

## 4. Performance Tips
- Use CPU for consistency on M1 Ultra
- Batch multiple series for efficiency
- Consider data preprocessing for better results
- Monitor memory usage with large datasets
