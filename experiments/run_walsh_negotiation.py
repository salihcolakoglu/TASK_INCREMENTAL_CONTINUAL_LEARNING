"""
Run Walsh Negotiation method on task-incremental continual learning benchmarks.

The Walsh Negotiation method combines:
1. Walsh-Hadamard code-based representations
2. Negotiated targets (mix of true labels and initial predictions)
3. Task-specific classifier heads (2 neurons per task)

Key differences from standard Negotiation:
- Uses Walsh codes for stable representation learning
- Trains on BCE loss for Walsh layer + CE loss for classifier
- Initial negotiation rate = 0.3 (vs 0.5 for standard negotiation)
- Uses plasticity formula to update negotiation rate

Usage:
    # Run on Split MNIST (default)
    python experiments/run_walsh_negotiation.py --dataset split_mnist --n_tasks 5

    # Run with custom parameters
    python experiments/run_walsh_negotiation.py --dataset split_mnist --alpha 0.3 --code_dim 16
"""

import argparse
import sys
import os
import json
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.baselines import WalshNegotiationTrainer, WalshMLP, WalshConvNet, WalshConvNetLite
from src.utils import get_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Walsh Negotiation method for continual learning')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='split_mnist',
                       choices=['split_mnist', 'split_cifar10', 'split_cifar100'],
                       help='Dataset to use')
    parser.add_argument('--n_tasks', type=int, default=5,
                       help='Number of tasks')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')

    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden layer size for MLP')
    parser.add_argument('--code_dim', type=int, default=16,
                       help='Walsh code dimension')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout probability')
    parser.add_argument('--lite', action='store_true',
                       help='Use WalshConvNetLite (2 conv blocks instead of 3, fewer params)')

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

    # Walsh Negotiation-specific arguments
    parser.add_argument('--alpha', '--initial_negotiation_rate', type=float, default=0.3,
                       dest='alpha',
                       help='Initial negotiation rate α₀ (default: 0.3)')

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
    parser.add_argument('--save_dir', type=str, default='./results/walsh_experiments',
                       help='Directory to save JSON results')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    print("WALSH NEGOTIATION METHOD - TASK-INCREMENTAL CONTINUAL LEARNING")
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
    print(f"\nWalsh Negotiation Parameters:")
    print(f"  Walsh code dimension: {args.code_dim}")
    print(f"  Initial α₀: {args.alpha:.3f}")
    print(f"  α will increase after each task using plasticity formula")
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

    # Create model based on dataset
    if args.dataset == 'split_mnist':
        model = WalshMLP(
            input_size=784,
            hidden_size=args.hidden_size,
            code_dim=args.code_dim,
            num_classes_per_task=classes_per_task,
            num_tasks=args.n_tasks,
            dropout=args.dropout
        )
        model_name = "WalshMLP"
    else:  # CIFAR-10 or CIFAR-100
        if args.lite:
            model = WalshConvNetLite(
                code_dim=args.code_dim,
                num_classes_per_task=classes_per_task,
                num_tasks=args.n_tasks,
                dropout=args.dropout,
                input_channels=3
            )
            model_name = "WalshConvNetLite"
        else:
            model = WalshConvNet(
                code_dim=args.code_dim,
                num_classes_per_task=classes_per_task,
                num_tasks=args.n_tasks,
                dropout=args.dropout,
                input_channels=3
            )
            model_name = "WalshConvNet"

    print(f"{model_name} model created with {sum(p.numel() for p in model.parameters())} parameters")

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

    # Create Walsh Negotiation trainer
    trainer = WalshNegotiationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_tasks=args.n_tasks,
        num_classes_per_task=classes_per_task,
        config=config,
        initial_negotiation_rate=args.alpha,
        code_dim=args.code_dim,
    )

    print("\n" + "="*70)
    print("STARTING CONTINUAL LEARNING WITH WALSH NEGOTIATION")
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

    # Print negotiation rate progression
    print("Negotiation Rate Progression:")
    print("-"*40)
    for i, alpha in enumerate(trainer.negotiation_rate_history):
        print(f"  After Task {i-1 if i > 0 else 'Init'}: α = {alpha:.4f}")
    print()

    # Print Walsh code assignments
    print("Walsh Code Assignments:")
    print("-"*40)
    for class_label, code_idx in sorted(trainer.tracker.class_to_code_idx.items()):
        print(f"  Class {class_label} -> Walsh code {code_idx}")
    print()

    # Print final accuracy matrix
    trainer.metrics.print_summary()

    # Save final results to JSON (always)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_type = "lite" if args.lite else "full"
    results_file = save_dir / f"{args.dataset}_walsh_{model_type}_alpha{args.alpha}_epochs{args.epochs}_seed{args.seed}_{timestamp}.json"

    # Get accuracy matrix as list
    accuracy_matrix = trainer.metrics.get_accuracy_matrix()
    if hasattr(accuracy_matrix, 'tolist'):
        accuracy_matrix = accuracy_matrix.tolist()

    results_data = {
        'method': f'Walsh-Negotiation-{model_type}',
        'args': vars(args),
        'results': {
            'avg_accuracy': final_metrics['average_accuracy'],
            'forgetting': final_metrics['forgetting'],
            'backward_transfer': final_metrics['backward_transfer'],
            'task_accuracies': final_metrics.get('task_accuracies', []),
            'accuracy_matrix': accuracy_matrix,
        },
        'negotiation_rate_history': trainer.negotiation_rate_history,
        'walsh_code_assignments': {str(k): v for k, v in trainer.tracker.class_to_code_idx.items()},
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Save checkpoint if requested
    if args.save_checkpoints:
        checkpoint_file = os.path.join(args.checkpoint_dir, 'walsh_negotiation_results.txt')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            f.write("Walsh Negotiation Method Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Number of tasks: {args.n_tasks}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Walsh code dimension: {args.code_dim}\n")
            f.write(f"Initial α₀: {args.alpha:.3f}\n\n")
            f.write(f"Average Accuracy: {final_metrics['average_accuracy']:.4f}\n")
            f.write(f"Forgetting: {final_metrics['forgetting']:.4f}\n")
            f.write(f"Backward Transfer: {final_metrics['backward_transfer']:.4f}\n\n")
            f.write("Negotiation Rate Progression:\n")
            for i, alpha in enumerate(trainer.negotiation_rate_history):
                f.write(f"  After Task {i-1 if i > 0 else 'Init'}: α = {alpha:.4f}\n")
            f.write("\nWalsh Code Assignments:\n")
            for class_label, code_idx in sorted(trainer.tracker.class_to_code_idx.items()):
                f.write(f"  Class {class_label} -> Walsh code {code_idx}\n")
            f.write("\nAccuracy Matrix:\n")
            f.write(str(trainer.metrics.get_accuracy_matrix()))
        print(f"Checkpoint saved to {checkpoint_file}")

    # Clean up
    trainer.close()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED")
    print("="*70)

    return final_metrics


if __name__ == '__main__':
    main()
