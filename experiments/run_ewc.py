"""
Run EWC (Elastic Weight Consolidation) baseline on task-incremental continual learning benchmarks.

Usage:
    python experiments/run_ewc.py --dataset split_mnist --n_tasks 5 --ewc_lambda 1000
    python experiments/run_ewc.py --dataset split_cifar10 --n_tasks 5 --ewc_lambda 5000
    python experiments/run_ewc.py --dataset split_cifar100 --n_tasks 10 --ewc_lambda 10000
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.baselines import EWCTrainer
from src.utils import get_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EWC baseline for continual learning')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='split_mnist',
                       choices=['split_mnist', 'split_cifar10', 'split_cifar100'],
                       help='Dataset to use')
    parser.add_argument('--n_tasks', type=int, default=5,
                       help='Number of tasks')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')

    # Model arguments
    parser.add_argument('--architecture', type=str, default='auto',
                       help='Model architecture (auto, mlp, convnet)')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden layer size for MLP')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout probability')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam'],
                       help='Optimizer')

    # EWC-specific arguments
    parser.add_argument('--ewc_lambda', type=float, default=1000.0,
                       help='EWC regularization strength (λ)')
    parser.add_argument('--ewc_mode', type=str, default='online',
                       choices=['online', 'separate'],
                       help='EWC mode: online or separate')
    parser.add_argument('--ewc_gamma', type=float, default=1.0,
                       help='Decay factor for online EWC')

    # Logging arguments
    parser.add_argument('--use_tensorboard', action='store_true',
                       help='Use TensorBoard logging')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--log_dir', type=str, default='./results/tensorboard',
                       help='TensorBoard log directory')
    parser.add_argument('--save_checkpoints', action='store_true',
                       help='Save model checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/checkpoints',
                       help='Checkpoint directory')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_architecture(dataset, architecture):
    """Get appropriate architecture for dataset."""
    if architecture == 'auto':
        if dataset == 'split_mnist':
            return 'mlp'
        else:  # CIFAR datasets
            return 'convnet'
    return architecture


def main():
    """Main training loop."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*70)
    print("EWC BASELINE - TASK-INCREMENTAL CONTINUAL LEARNING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Number of tasks: {args.n_tasks}")
    print(f"  Device: {device}")
    print(f"  Random seed: {args.seed}")
    print(f"  Epochs per task: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"\n  EWC lambda (λ): {args.ewc_lambda}")
    print(f"  EWC mode: {args.ewc_mode}")
    if args.ewc_mode == 'online':
        print(f"  EWC gamma (γ): {args.ewc_gamma}")
    print()

    # Load dataset
    print("Loading dataset...")
    train_loaders, val_loaders, test_loaders = get_dataset(
        dataset_name=args.dataset,
        n_tasks=args.n_tasks,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=0.0
    )

    # Determine number of classes per task
    classes_per_task = {
        'split_mnist': 10 // args.n_tasks,
        'split_cifar10': 10 // args.n_tasks,
        'split_cifar100': 100 // args.n_tasks,
    }[args.dataset]

    print(f"Dataset loaded: {args.n_tasks} tasks, {classes_per_task} classes per task")

    # Get architecture
    architecture = get_architecture(args.dataset, args.architecture)
    print(f"Using architecture: {architecture}")

    # Create model
    model_kwargs = {}
    if architecture == 'mlp':
        model_kwargs['hidden_size'] = args.hidden_size

    model = get_model(
        architecture=architecture,
        num_classes_per_task=classes_per_task,
        num_tasks=args.n_tasks,
        dropout=args.dropout,
        **model_kwargs
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # Create loss criterion
    criterion = nn.CrossEntropyLoss()

    # Create configuration
    config = {
        'use_tensorboard': args.use_tensorboard,
        'use_wandb': args.use_wandb,
        'log_dir': args.log_dir,
        'save_checkpoints': args.save_checkpoints,
        'checkpoint_dir': args.checkpoint_dir,
    }

    # Create EWC trainer
    trainer = EWCTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_tasks=args.n_tasks,
        num_classes_per_task=classes_per_task,
        config=config,
        ewc_lambda=args.ewc_lambda,
        mode=args.ewc_mode,
        gamma=args.ewc_gamma
    )

    print("\n" + "="*70)
    print("STARTING CONTINUAL LEARNING TRAINING")
    print("="*70)

    # Train on each task sequentially
    for task_id in range(args.n_tasks):
        # Train on current task
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],
            val_loader=val_loaders[task_id],
            epochs=args.epochs
        )

        # Evaluate on all tasks seen so far
        accuracies = trainer.evaluate_all_tasks(
            task_dataloaders=test_loaders,
            current_task=task_id
        )

        # Save checkpoint
        if args.save_checkpoints:
            trainer.save_checkpoint(task_id)

    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    final_metrics = trainer.metrics.get_all_metrics()
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-"*40)
    print(f"{'Average Accuracy':<25} {final_metrics['average_accuracy']:.4f}")
    print(f"{'Forgetting':<25} {final_metrics['forgetting']:.4f}")
    print(f"{'Backward Transfer':<25} {final_metrics['backward_transfer']:.4f}")
    print()

    # Print final accuracy matrix
    trainer.metrics.print_summary()

    # Save final results
    if args.save_checkpoints:
        results_file = os.path.join(args.checkpoint_dir, f'ewc_lambda{args.ewc_lambda}_results.txt')
        with open(results_file, 'w') as f:
            f.write("EWC Baseline Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Number of tasks: {args.n_tasks}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"EWC Lambda: {args.ewc_lambda}\n")
            f.write(f"EWC Mode: {args.ewc_mode}\n\n")
            f.write(f"Average Accuracy: {final_metrics['average_accuracy']:.4f}\n")
            f.write(f"Forgetting: {final_metrics['forgetting']:.4f}\n")
            f.write(f"Backward Transfer: {final_metrics['backward_transfer']:.4f}\n\n")
            f.write("Accuracy Matrix:\n")
            f.write(str(trainer.metrics.get_accuracy_matrix()))
        print(f"Results saved to {results_file}")

    # Clean up
    trainer.close()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED")
    print("="*70)


if __name__ == '__main__':
    main()
