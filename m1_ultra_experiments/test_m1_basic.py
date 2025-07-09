#!/usr/bin/env python3
"""
Basic test script for Chronos-Bolt on M1 Ultra Mac
Tests local CPU inference with the AirPassengers dataset
"""

import torch
import pandas as pd
import numpy as np
from chronos import BaseChronosPipeline
import time
import sys

def test_system_info():
    """Print system information for debugging"""
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Apple Silicon MPS available: {torch.backends.mps.is_available()}")
    print(f"Number of CPU cores: {torch.get_num_threads()}")
    print()

def test_air_passengers_example():
    """Test with the AirPassengers dataset from the README"""
    print("=== AirPassengers Dataset Test ===")
    
    # Load the AirPassengers dataset
    url = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    try:
        df = pd.read_csv(url)
        print(f"Dataset loaded successfully: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few values: {df['#Passengers'].head().tolist()}")
        print()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Load Chronos-Bolt tiny model (smallest for testing)
    print("Loading Chronos-Bolt tiny model...")
    start_time = time.time()
    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",
            device_map="cpu",  # Force CPU usage
            torch_dtype=torch.float32,
        )
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        print(f"Model type: {type(pipeline)}")
        print()
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Prepare data
    context = torch.tensor(df["#Passengers"].values, dtype=torch.float32)
    print(f"Context shape: {context.shape}")
    print(f"Context data type: {context.dtype}")
    print()
    
    # Test quantile prediction
    print("Testing quantile prediction...")
    start_time = time.time()
    try:
        quantiles, mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=12,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")
        print(f"Quantiles shape: {quantiles.shape}")
        print(f"Mean shape: {mean.shape}")
        print(f"Mean prediction (first 6 months): {mean[0, :6].tolist()}")
        print()
        return True
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

def test_simple_synthetic_data():
    """Test with simple synthetic data"""
    print("=== Synthetic Data Test ===")
    
    # Create simple synthetic time series
    np.random.seed(42)
    t = np.arange(100)
    synthetic_data = 10 + 2 * np.sin(0.3 * t) + np.random.normal(0, 0.5, 100)
    
    context = torch.tensor(synthetic_data, dtype=torch.float32)
    print(f"Synthetic data shape: {context.shape}")
    
    # Load model
    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Test prediction
    try:
        quantiles, mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=24,
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
        print(f"Prediction successful")
        print(f"Quantiles shape: {quantiles.shape}")
        print(f"Mean shape: {mean.shape}")
        print(f"Sample predictions: {mean[0, :5].tolist()}")
        return True
    except Exception as e:
        print(f"Error during prediction: {e}")
        return False

def main():
    """Run all tests"""
    print("Chronos-Bolt M1 Ultra Test Suite")
    print("=" * 40)
    
    test_system_info()
    
    success_count = 0
    total_tests = 2
    
    if test_air_passengers_example():
        success_count += 1
    
    if test_simple_synthetic_data():
        success_count += 1
    
    print("=" * 40)
    print(f"Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All tests passed! Chronos-Bolt is working on M1 Ultra")
    else:
        print("❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()