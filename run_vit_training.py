#!/usr/bin/env python3
"""
Example script to run ViT-Small training with sparse updates
"""

import os
import sys
import argparse

# Add algorithm directory to path
sys.path.append('./algorithm')

def main():
    parser = argparse.ArgumentParser(description='Run ViT-Small training with sparse updates')
    parser.add_argument('--config', type=str, default='configs/vit_small_100kb.yaml', 
                       help='Configuration file')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--run_dir', type=str, default='./runs/vit_small_sparse',
                       help='Output directory for experiments')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--memory_budget', type=str, default='100kb',
                       choices=['100kb', '150kb', '200kb'],
                       help='Memory budget for sparse training')
    
    args = parser.parse_args()
    
    # Construct command for train_cls.py
    cmd_parts = [
        'python', 'algorithm/train_cls.py',
        args.config,
        '--run_dir', args.run_dir,
        '--data_provider.root', args.data_root,
        '--data_provider.base_batch_size', str(args.batch_size),
        '--run_config.n_epochs', str(args.epochs),
        '--run_config.base_lr', str(args.lr),
    ]
    
    # Add memory budget specific config
    if args.memory_budget == '100kb':
        cmd_parts.extend([
            '--backward_config.n_bias_update', '8',
            '--backward_config.weight_update_ratio', '0.25'
        ])
    elif args.memory_budget == '150kb':
        cmd_parts.extend([
            '--backward_config.n_bias_update', '12',
            '--backward_config.weight_update_ratio', '0.5'
        ])
    elif args.memory_budget == '200kb':
        cmd_parts.extend([
            '--backward_config.n_bias_update', '18',
            '--backward_config.weight_update_ratio', '0.75'
        ])
    
    # Create output directory
    os.makedirs(args.run_dir, exist_ok=True)
    
    print("Starting ViT-Small training with sparse updates...")
    print(f"Configuration: {args.config}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.run_dir}")
    print(f"Memory budget: {args.memory_budget}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Print the command that would be executed
    print("Command to execute:")
    print(' '.join(cmd_parts))
    print()
    
    # Execute the training
    import subprocess
    try:
        result = subprocess.run(cmd_parts, check=True, cwd='.')
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()