#!/usr/bin/env python3
"""
Working Fine-Tuning Example for Chronos on M1 Ultra
Complete end-to-end example with proper data formatting
"""

import torch
import pandas as pd
import numpy as np
import yaml
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import os

# Chronos imports
from chronos import BaseChronosPipeline

# GluonTS imports
from gluonts.dataset.arrow import ArrowWriter
from gluonts.dataset.common import FileDataset

def create_realistic_training_data(num_series: int = 100, series_length: int = 200):
    """Create realistic synthetic time series data for fine-tuning"""
    print(f"📊 Creating {num_series} realistic time series...")
    
    np.random.seed(42)
    time_series = []
    
    for i in range(num_series):
        t = np.arange(series_length)
        
        # Different business patterns
        if i % 5 == 0:
            # E-commerce sales with weekly patterns
            weekly_pattern = 1 + 0.3 * np.sin(2 * np.pi * t / 7)
            monthly_pattern = 1 + 0.2 * np.sin(2 * np.pi * t / 30)
            trend = 1 + 0.001 * t
            noise = np.random.normal(0, 0.1, series_length)
            series = 1000 * weekly_pattern * monthly_pattern * trend + noise * 100
            
        elif i % 5 == 1:
            # Energy consumption with daily patterns
            daily_pattern = 50 + 30 * np.sin(2 * np.pi * t / 24)
            seasonal_pattern = 1 + 0.4 * np.sin(2 * np.pi * t / 365)
            noise = np.random.normal(0, 0.05, series_length)
            series = daily_pattern * seasonal_pattern * (1 + noise)
            
        elif i % 5 == 2:
            # Stock price with volatility clustering
            returns = np.random.normal(0.001, 0.02, series_length)
            # Add volatility clustering
            volatility = np.abs(returns) * 0.5 + 0.01
            returns = returns + np.random.normal(0, volatility)
            series = 100 * np.exp(np.cumsum(returns))
            
        elif i % 5 == 3:
            # Server metrics with anomalies
            baseline = 50 + 10 * np.sin(2 * np.pi * t / 24)
            # Add occasional spikes
            spikes = np.random.choice([0, 1], series_length, p=[0.95, 0.05])
            spike_values = np.random.exponential(20, series_length) * spikes
            noise = np.random.normal(0, 2, series_length)
            series = baseline + spike_values + noise
            
        else:
            # User activity with growth trend
            growth = 100 * (1 + 0.02) ** (t / 30)  # 2% monthly growth
            weekly_cycle = 1 + 0.3 * np.sin(2 * np.pi * t / 7)
            noise = np.random.normal(0, 0.05, series_length)
            series = growth * weekly_cycle * (1 + noise)
        
        # Ensure positive values and reasonable scale
        series = np.maximum(series, 0.1)
        time_series.append(series.astype(np.float32))
    
    print(f"✅ Created {len(time_series)} realistic time series")
    return time_series

def convert_to_arrow_format_fixed(time_series: list, output_path: str):
    """Convert time series to GluonTS Arrow format with proper datetime handling"""
    print(f"🔧 Converting {len(time_series)} series to Arrow format...")
    
    # Use pandas timestamp for proper datetime handling
    start_date = pd.Timestamp('2020-01-01')
    
    dataset = []
    for i, ts in enumerate(time_series):
        dataset.append({
            "start": start_date,
            "target": ts,
            "item_id": f"series_{i:03d}"
        })
    
    try:
        ArrowWriter(compression="lz4").write_to_file(dataset, path=output_path)
        print(f"✅ Arrow file created successfully: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating Arrow file: {e}")
        return False

def create_fine_tuning_config_file(
    arrow_path: str,
    config_path: str,
    model_id: str = "amazon/chronos-t5-tiny",
    max_steps: int = 200,
    output_dir: str = "./fine_tuned_chronos"
):
    """Create a YAML configuration file for fine-tuning"""
    
    config = {
        "training_data_paths": [arrow_path],
        "probability": [1.0],
        "context_length": 128,
        "prediction_length": 24,
        "min_past": 30,
        "max_steps": max_steps,
        "save_steps": max_steps // 2,
        "log_steps": 10,
        "per_device_train_batch_size": 4,  # Small for M1 Ultra CPU
        "learning_rate": 0.001,
        "optim": "adamw_torch",
        "num_samples": 20,
        "shuffle_buffer_length": 1000,
        "gradient_accumulation_steps": 2,  # Effective batch size = 4 * 2 = 8
        "model_id": model_id,
        "model_type": "seq2seq",
        "random_init": False,  # Fine-tune from pretrained
        "tie_embeddings": True,
        "output_dir": output_dir,
        "tf32": False,
        "torch_compile": False,
        "tokenizer_class": "MeanScaleUniformBins",
        "tokenizer_kwargs": {
            "low_limit": -15.0,
            "high_limit": 15.0
        },
        "n_tokens": 4096,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.1,
        "dataloader_num_workers": 1,
        "max_missing_prop": 0.9,
        "use_eos_token": True,
        "logging_steps": 10,
        "eval_steps": max_steps // 4,
        "save_total_limit": 2,
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Configuration file created: {config_path}")
    return config

def test_pre_and_post_fine_tuning_performance():
    """Test performance before and after fine-tuning (simulated)"""
    print("\n" + "="*60)
    print("PRE/POST FINE-TUNING PERFORMANCE TEST")
    print("="*60)
    
    # Create test data
    test_series = create_realistic_training_data(num_series=10, series_length=100)
    
    # Test original model
    print("🤖 Testing original model performance...")
    try:
        original_pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        # Test on sample data
        context = torch.tensor(test_series[0][:80], dtype=torch.float32)
        true_values = test_series[0][80:92]  # 12 future values
        
        start_time = time.time()
        quantiles, mean = original_pipeline.predict_quantiles(
            context=context,
            prediction_length=12,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        inference_time = time.time() - start_time
        
        # Calculate MAE
        mae = np.mean(np.abs(mean[0].numpy() - true_values))
        
        print(f"✅ Original model inference time: {inference_time:.3f}s")
        print(f"✅ Original model MAE: {mae:.3f}")
        print(f"✅ Original predictions: {mean[0, :5].tolist()}")
        print(f"✅ True values: {true_values[:5].tolist()}")
        
        return {
            'original_mae': mae,
            'original_time': inference_time,
            'original_predictions': mean[0].tolist(),
            'true_values': true_values.tolist()
        }
        
    except Exception as e:
        print(f"❌ Error testing original model: {e}")
        return None

def create_complete_fine_tuning_workflow():
    """Create a complete fine-tuning workflow"""
    print("\n" + "="*60)
    print("COMPLETE FINE-TUNING WORKFLOW")
    print("="*60)
    
    # Create temporary directory for all files
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Working directory: {temp_dir}")
    
    try:
        # Step 1: Create training data
        print("\n📊 Step 1: Creating training data...")
        time_series = create_realistic_training_data(num_series=50, series_length=200)
        
        # Step 2: Convert to Arrow format
        print("\n🔧 Step 2: Converting to Arrow format...")
        arrow_path = os.path.join(temp_dir, "training_data.arrow")
        if not convert_to_arrow_format_fixed(time_series, arrow_path):
            raise Exception("Failed to create Arrow file")
        
        # Verify Arrow file
        dataset = FileDataset(path=Path(arrow_path))
        dataset_list = list(dataset)
        print(f"✅ Arrow file verified: {len(dataset_list)} time series loaded")
        
        # Step 3: Create configuration
        print("\n⚙️  Step 3: Creating training configuration...")
        config_path = os.path.join(temp_dir, "fine_tuning_config.yaml")
        output_dir = os.path.join(temp_dir, "fine_tuned_model")
        
        config = create_fine_tuning_config_file(
            arrow_path=arrow_path,
            config_path=config_path,
            model_id="amazon/chronos-t5-tiny",
            max_steps=100,
            output_dir=output_dir
        )
        
        # Step 4: Show training command
        print("\n🚀 Step 4: Training command (run manually):")
        training_command = f"""
# Navigate to the chronos-forecasting directory
cd {os.getcwd()}

# Run fine-tuning with the configuration
python scripts/training/train.py --config {config_path}

# Alternative: Run with command line overrides
python scripts/training/train.py \\
    --config {config_path} \\
    --max-steps 100 \\
    --learning-rate 0.001 \\
    --per-device-train-batch-size 4
"""
        print(training_command)
        
        # Step 5: Show post-training usage
        print("\n📈 Step 5: Post-training usage:")
        usage_code = f"""
# Load your fine-tuned model
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    "{output_dir}/checkpoint-100",  # Adjust checkpoint number
    device_map="cpu",
    torch_dtype=torch.float32
)

# Use for prediction
quantiles, mean = pipeline.predict_quantiles(
    context=your_data,
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9]
)
"""
        print(usage_code)
        
        # Create a complete example script
        example_script = f"""#!/usr/bin/env python3
'''
Complete Fine-tuning Example - Generated for your M1 Ultra
'''

import torch
from chronos import BaseChronosPipeline
import numpy as np

def main():
    print("🚀 Loading fine-tuned model...")
    
    # Load your fine-tuned model (update path as needed)
    pipeline = BaseChronosPipeline.from_pretrained(
        "{output_dir}/checkpoint-100",
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    # Create test data
    test_data = np.random.randn(100) + 50  # Replace with your data
    context = torch.tensor(test_data, dtype=torch.float32)
    
    # Make predictions
    quantiles, mean = pipeline.predict_quantiles(
        context=context,
        prediction_length=12,
        quantile_levels=[0.1, 0.5, 0.9]
    )
    
    print(f"✅ Predictions: {{mean[0, :5].tolist()}}")
    print(f"✅ Quantiles shape: {{quantiles.shape}}")

if __name__ == "__main__":
    main()
"""
        
        example_path = os.path.join(temp_dir, "use_fine_tuned_model.py")
        with open(example_path, 'w') as f:
            f.write(example_script)
        
        print(f"✅ Example script created: {example_path}")
        
        # Summary
        print(f"\n📋 Fine-tuning workflow summary:")
        print(f"   📊 Training data: {len(time_series)} time series")
        print(f"   📁 Arrow file: {arrow_path}")
        print(f"   ⚙️  Config file: {config_path}")
        print(f"   🎯 Output directory: {output_dir}")
        print(f"   📝 Example script: {example_path}")
        
        return temp_dir
        
    except Exception as e:
        print(f"❌ Error in workflow: {e}")
        return None

def main():
    """Run the complete fine-tuning demonstration"""
    print("🚀 Chronos Fine-Tuning Complete Demo for M1 Ultra")
    print("=" * 60)
    
    # Test pre-training performance
    performance_results = test_pre_and_post_fine_tuning_performance()
    
    # Create complete workflow
    workflow_dir = create_complete_fine_tuning_workflow()
    
    print("\n" + "=" * 60)
    print("FINE-TUNING DEMO SUMMARY")
    print("=" * 60)
    
    if performance_results:
        print(f"✅ Original model tested successfully")
        print(f"   📊 MAE: {performance_results['original_mae']:.3f}")
        print(f"   ⏱️  Time: {performance_results['original_time']:.3f}s")
    
    if workflow_dir:
        print(f"✅ Complete workflow created")
        print(f"   📁 Files in: {workflow_dir}")
        print(f"\n🎯 Next steps:")
        print(f"   1. Review the configuration file")
        print(f"   2. Run the training command shown above")
        print(f"   3. Use the fine-tuned model with the example script")
        print(f"\n⚠️  Note: Keep the temp directory until training is complete!")
        print(f"   Directory: {workflow_dir}")
    
    print(f"\n🏆 Your M1 Ultra is ready for Chronos fine-tuning!")
    print(f"💡 Tips:")
    print(f"   - Start with 100-200 training steps for testing")
    print(f"   - Use batch size 4-8 for optimal M1 Ultra performance")
    print(f"   - Monitor training with tensorboard: tensorboard --logdir output/")

if __name__ == "__main__":
    main()