#!/usr/bin/env python3
"""
Clean POS Router Training Script

A truly modular, focused training script that demonstrates proper separation of concerns.
This is what a modular training script should look like - clean, readable, and focused.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from cli.args import parse_args, print_args_summary
from training.trainer import POSTrainer
from utils.model_utils import create_model_directory, print_model_summary


def main():
    """Main training entry point - clean and focused."""
    print("🚀 POS Router Training")
    print("=" * 60)
    
    try:
        # Parse arguments
        args = parse_args()
        print_args_summary(args)
        
        # Create model directory
        model_dir = create_model_directory(args.model_dir)
        print(f"📁 Model directory: {model_dir}")
        
        # Initialize trainer
        trainer = POSTrainer(args)
        
        # Setup data and model
        dataset_info = trainer.setup_data()
        trainer.setup_model()
        
        # Train the model
        final_results = trainer.train()
        
        # Save model artifacts
        artifacts = trainer.save_model(final_results, model_dir)
        
        # Print summary
        print_model_summary(trainer.model_name, artifacts, final_results, args)
        
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 