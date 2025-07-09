#!/usr/bin/env python3
"""
Custom dataset example for Chronos-Bolt on M1 Ultra Mac
Shows how to prepare and use your own time series data
"""

import torch
import pandas as pd
import numpy as np
from chronos import BaseChronosPipeline
import time
from pathlib import Path

def create_sample_datasets():
    """Create sample custom datasets for testing"""
    print("=== Creating Sample Custom Datasets ===")
    
    # Dataset 1: Financial time series (simulated stock prices)
    np.random.seed(42)
    days = 365
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    
    # Simulate stock price with trend and volatility
    returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
    price = 100 * np.exp(np.cumsum(returns))  # Geometric Brownian motion
    
    stock_data = pd.DataFrame({
        'date': dates,
        'price': price,
        'volume': np.random.lognormal(10, 1, days),  # Trading volume
        'returns': returns
    })
    
    # Dataset 2: IoT sensor data (simulated temperature readings)
    hours = 24 * 30  # 30 days of hourly data
    dates = pd.date_range('2023-01-01', periods=hours, freq='H')
    
    # Simulate temperature with daily and weekly patterns
    t = np.arange(hours)
    temp = (
        20 +  # Base temperature
        5 * np.sin(2 * np.pi * t / 24) +  # Daily cycle
        2 * np.sin(2 * np.pi * t / (24 * 7)) +  # Weekly cycle
        np.random.normal(0, 1, hours)  # Noise
    )
    
    temp_data = pd.DataFrame({
        'datetime': dates,
        'temperature': temp,
        'humidity': 50 + 20 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 5, hours)
    })
    
    # Dataset 3: Sales data (simulated retail sales)
    weeks = 104  # 2 years of weekly data
    dates = pd.date_range('2022-01-01', periods=weeks, freq='W')
    
    # Simulate sales with seasonal patterns
    t = np.arange(weeks)
    sales = (
        1000 +  # Base sales
        200 * np.sin(2 * np.pi * t / 52) +  # Annual seasonality
        100 * np.sin(2 * np.pi * t / 13) +  # Quarterly patterns
        50 * (t / weeks) * 100 +  # Growth trend
        np.random.normal(0, 50, weeks)  # Noise
    )
    
    sales_data = pd.DataFrame({
        'week': dates,
        'sales': np.maximum(sales, 0),  # Ensure positive sales
        'marketing_spend': 100 + 50 * np.sin(2 * np.pi * t / 26) + np.random.normal(0, 20, weeks)
    })
    
    # Save datasets
    datasets = {
        'stock_prices.csv': stock_data,
        'temperature_sensors.csv': temp_data,
        'retail_sales.csv': sales_data
    }
    
    data_dir = Path('m1_ultra_experiments/custom_datasets')
    data_dir.mkdir(exist_ok=True)
    
    for filename, df in datasets.items():
        filepath = data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"  ✅ Created {filepath} ({len(df)} records)")
    
    return datasets

def test_custom_dataset(dataset_name, df, value_column, prediction_length=12):
    """Test Chronos-Bolt on a custom dataset"""
    print(f"\n=== Testing {dataset_name} ===")
    
    # Prepare data
    values = df[value_column].values
    context = torch.tensor(values, dtype=torch.float32)
    
    print(f"Dataset: {len(values)} timesteps")
    print(f"Value range: {values.min():.2f} to {values.max():.2f}")
    print(f"Target column: {value_column}")
    
    # Use 80% for context, predict the rest
    split_point = int(len(values) * 0.8)
    train_context = context[:split_point]
    true_values = context[split_point:split_point + prediction_length]
    
    print(f"Training context: {len(train_context)} timesteps")
    print(f"Prediction length: {prediction_length} timesteps")
    
    try:
        # Load model
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        
        # Make predictions
        start_time = time.time()
        quantiles, mean = pipeline.predict_quantiles(
            context=train_context,
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        inference_time = time.time() - start_time
        
        # Calculate basic metrics if we have true values
        if len(true_values) >= prediction_length:
            mae = torch.mean(torch.abs(mean[0] - true_values[:prediction_length])).item()
            mse = torch.mean((mean[0] - true_values[:prediction_length]) ** 2).item()
            print(f"✅ Inference time: {inference_time:.3f}s")
            print(f"✅ MAE: {mae:.3f}")
            print(f"✅ MSE: {mse:.3f}")
        else:
            print(f"✅ Inference time: {inference_time:.3f}s")
            print(f"✅ Predictions generated successfully")
        
        # Show sample predictions
        print(f"✅ Sample predictions: {mean[0, :5].tolist()}")
        if len(true_values) >= 5:
            print(f"✅ True values: {true_values[:5].tolist()}")
        
        return {
            'dataset': dataset_name,
            'inference_time': inference_time,
            'predictions': mean[0].tolist(),
            'quantiles': quantiles[0].tolist(),
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            'dataset': dataset_name,
            'error': str(e),
            'success': False
        }

def create_deployment_guide():
    """Create a simple deployment guide"""
    guide = """
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
"""
    
    with open('m1_ultra_experiments/deployment_guide.md', 'w') as f:
        f.write(guide)
    
    print("✅ Created deployment_guide.md")

def main():
    """Run custom dataset tests"""
    print("Chronos-Bolt Custom Dataset Testing")
    print("=" * 40)
    
    # Create sample datasets
    datasets = create_sample_datasets()
    
    results = []
    
    # Test each dataset
    for filename, df in datasets.items():
        dataset_name = filename.replace('.csv', '')
        
        if 'stock' in filename:
            result = test_custom_dataset(dataset_name, df, 'price', 30)
        elif 'temperature' in filename:
            result = test_custom_dataset(dataset_name, df, 'temperature', 24)
        elif 'sales' in filename:
            result = test_custom_dataset(dataset_name, df, 'sales', 8)
        
        results.append(result)
    
    # Create deployment guide
    create_deployment_guide()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Custom Dataset Testing Summary:")
    successful = sum(1 for r in results if r['success'])
    print(f"✅ {successful}/{len(results)} datasets tested successfully")
    
    if successful > 0:
        print("\n🎉 Your M1 Ultra is ready for custom time series forecasting!")
        print("📝 Check m1_ultra_experiments/deployment_guide.md for usage patterns")
        print("📁 Sample datasets saved in m1_ultra_experiments/custom_datasets/ directory")

if __name__ == "__main__":
    main()