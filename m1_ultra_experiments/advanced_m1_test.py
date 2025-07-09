#!/usr/bin/env python3
"""
Advanced test script for Chronos-Bolt on M1 Ultra Mac
Tests different model sizes and provides visualization capabilities
"""

import torch
import pandas as pd
import numpy as np
from chronos import BaseChronosPipeline
import time
import sys
from pathlib import Path

def test_model_sizes():
    """Test different Chronos-Bolt model sizes"""
    print("=== Model Size Comparison ===")
    
    # Load AirPassengers data
    url = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    df = pd.read_csv(url)
    context = torch.tensor(df["#Passengers"].values, dtype=torch.float32)
    
    models = [
        ("amazon/chronos-bolt-tiny", "9M parameters"),
        ("amazon/chronos-bolt-mini", "21M parameters"),
        ("amazon/chronos-bolt-small", "48M parameters"),
        # ("amazon/chronos-bolt-base", "205M parameters")  # Uncomment if you want to test
    ]
    
    results = []
    
    for model_name, description in models:
        print(f"\nTesting {model_name} ({description})...")
        
        try:
            # Load model
            start_time = time.time()
            pipeline = BaseChronosPipeline.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            load_time = time.time() - start_time
            
            # Run inference
            start_time = time.time()
            quantiles, mean = pipeline.predict_quantiles(
                context=context,
                prediction_length=12,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            inference_time = time.time() - start_time
            
            # Store results
            result = {
                'model': model_name,
                'description': description,
                'load_time': load_time,
                'inference_time': inference_time,
                'predictions': mean[0, :6].tolist(),
                'quantile_shape': quantiles.shape,
                'mean_shape': mean.shape
            }
            results.append(result)
            
            print(f"  ✅ Load time: {load_time:.2f}s")
            print(f"  ✅ Inference time: {inference_time:.2f}s")
            print(f"  ✅ First 6 predictions: {mean[0, :6].tolist()}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Print summary
    print("\n=== Model Comparison Summary ===")
    print(f"{'Model':<25} {'Load Time':<12} {'Inference Time':<15} {'First Prediction':<15}")
    print("-" * 70)
    for result in results:
        model_short = result['model'].split('/')[-1]
        print(f"{model_short:<25} {result['load_time']:<12.2f} {result['inference_time']:<15.2f} {result['predictions'][0]:<15.2f}")
    
    return results

def test_longer_sequences():
    """Test with longer input sequences"""
    print("\n=== Longer Sequence Test ===")
    
    # Create longer synthetic time series
    np.random.seed(42)
    t = np.arange(500)  # 500 timesteps
    synthetic_data = (
        10 + 
        2 * np.sin(0.1 * t) + 
        0.5 * np.sin(0.5 * t) + 
        np.random.normal(0, 0.3, 500)
    )
    
    context = torch.tensor(synthetic_data, dtype=torch.float32)
    print(f"Input sequence length: {len(context)}")
    
    # Test with tiny model
    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        
        start_time = time.time()
        quantiles, mean = pipeline.predict_quantiles(
            context=context,
            prediction_length=48,  # Predict 48 steps ahead
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
        )
        inference_time = time.time() - start_time
        
        print(f"✅ Prediction successful")
        print(f"✅ Inference time: {inference_time:.2f}s")
        print(f"✅ Output shape: {quantiles.shape}")
        print(f"✅ Sample predictions: {mean[0, :10].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_processing():
    """Test batch processing with multiple time series"""
    print("\n=== Batch Processing Test ===")
    
    # Create batch of synthetic time series
    np.random.seed(42)
    batch_size = 4
    sequence_length = 100
    
    batch_data = []
    for i in range(batch_size):
        t = np.arange(sequence_length)
        series = (
            10 + i * 2 +  # Different base levels
            2 * np.sin(0.2 * t + i * 0.5) +  # Different phases
            np.random.normal(0, 0.3, sequence_length)
        )
        batch_data.append(torch.tensor(series, dtype=torch.float32))
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {sequence_length}")
    
    try:
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        
        start_time = time.time()
        quantiles, mean = pipeline.predict_quantiles(
            context=batch_data,  # Pass as list
            prediction_length=24,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        inference_time = time.time() - start_time
        
        print(f"✅ Batch prediction successful")
        print(f"✅ Inference time: {inference_time:.2f}s")
        print(f"✅ Output shape: {quantiles.shape}")
        print(f"✅ Mean predictions for first series: {mean[0, :6].tolist()}")
        print(f"✅ Mean predictions for last series: {mean[-1, :6].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def save_results_to_file(results, filename="chronos_m1_results.txt"):
    """Save test results to a file"""
    with open(filename, "w") as f:
        f.write("Chronos-Bolt M1 Ultra Test Results\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("System Information:\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"MPS available: {torch.backends.mps.is_available()}\n")
        f.write(f"CPU cores: {torch.get_num_threads()}\n\n")
        
        f.write("Model Performance:\n")
        f.write(f"{'Model':<25} {'Load Time':<12} {'Inference Time':<15}\n")
        f.write("-" * 55 + "\n")
        for result in results:
            model_short = result['model'].split('/')[-1]
            f.write(f"{model_short:<25} {result['load_time']:<12.2f} {result['inference_time']:<15.2f}\n")
    
    print(f"Results saved to {filename}")

def main():
    """Run all advanced tests"""
    print("Chronos-Bolt M1 Ultra Advanced Test Suite")
    print("=" * 50)
    
    # Test different model sizes
    results = test_model_sizes()
    
    # Test longer sequences
    test_longer_sequences()
    
    # Test batch processing
    test_batch_processing()
    
    # Save results
    if results:
        save_results_to_file(results)
    
    print("\n" + "=" * 50)
    print("🎉 Advanced testing complete!")
    print("💡 Next steps: Try with your own dataset or explore fine-tuning")

if __name__ == "__main__":
    main()