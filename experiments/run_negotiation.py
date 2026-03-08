"""
Run Negotiation method on task-incremental continual learning benchmarks.

The Negotiation method prevents catastrophic forgetting by training on "negotiated targets"
that are a weighted combination of true labels and the model's INITIAL predictions.

IMPORTANT: Negotiated labels are computed ONCE using the untrained initial model,
BEFORE any training begins. This makes the method:
- Independent of training epochs
- Independent of training dynamics
- Reproducible and stable

Algorithm:
1. Initialize model with random weights
2. Get initial model predictions on ALL tasks' training data
3. Create negotiated labels: y_negotiated = (1-α) * y_true + α * y_pred_initial
4. Store negotiated labels
5. Train sequentially on each task using its stored negotiated labels

Usage:
    # Run on Split MNIST (default α₀=0.5)
    python experiments/run_negotiation.py --dataset split_mnist --n_tasks 5

    # Run on Split CIFAR-10 with custom initial negotiation rate
    python experiments/run_negotiation.py --dataset split_cifar10 --n_tasks 5 --alpha 0.3

    # Run on Split CIFAR-100 with alpha=0.7 (70% initial model, 30% labels)
    python experiments/run_negotiation.py --dataset split_cifar100 --n_tasks 10 --alpha 0.7
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
from src.baselines import NegotiationTrainer
from src.utils import get_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Negotiation method for continual learning')

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

    # Negotiation-specific arguments
    parser.add_argument('--alpha', '--initial_negotiation_rate', type=float, default=None,
                       dest='alpha',
                       help='Initial negotiation rate α₀ (0 < α < 1). '
                            'Default: 0.5 (equal weighting between labels and initial predictions). '
                            'Lower values trust labels more, higher values trust initial model more.')
    parser.add_argument('--no_update_alpha', action='store_true',
                       help='Keep alpha constant (recommended, since labels are pre-computed)')

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
    # Make CuDNN deterministic
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


def get_default_alpha(dataset):
    """Get default initial negotiation rate for dataset."""
    # Using 0.5 as default for all datasets (equal weighting)
    # This can be tuned via hyperparameter search for each dataset
    defaults = {
        'split_mnist': 0.5,       # Equal weighting
        'split_cifar10': 0.5,     # Equal weighting
        'split_cifar100': 0.5,    # Equal weighting
    }
    return defaults.get(dataset, 0.5)


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

    # Get default alpha if not provided
    if args.alpha is None:
        args.alpha = get_default_alpha(args.dataset)

    print("="*70)
    print("NEGOTIATION METHOD - TASK-INCREMENTAL CONTINUAL LEARNING")
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
    print(f"\nNegotiation Parameters:")
    print(f"  Initial α₀: {args.alpha:.3f}")
    if args.no_update_alpha:
        print(f"  α will remain CONSTANT at {args.alpha:.3f} throughout training")
    else:
        print(f"  α will increase after each task using: α_new = α * (1/(2α - α²))")
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

    # Create Negotiation trainer
    trainer = NegotiationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_tasks=args.n_tasks,
        num_classes_per_task=classes_per_task,
        config=config,
        initial_negotiation_rate=args.alpha,
        update_negotiation_rate=not args.no_update_alpha
    )

    print("\n" + "="*70)
    print("STARTING CONTINUAL LEARNING WITH NEGOTIATION")
    print("="*70)

    # CRITICAL STEP: Pre-compute negotiated labels for ALL tasks
    # This must be done BEFORE any training to ensure labels are based on
    # the INITIAL model state and are independent of training dynamics
    print("\n" + "="*70)
    print("STEP 1: PRE-COMPUTING NEGOTIATED LABELS")
    print("="*70)
    trainer.prepare_negotiated_labels(
        train_loaders=train_loaders,
        batch_size=args.batch_size
    )

    # Now train on each task sequentially
    print("\n" + "="*70)
    print("STEP 2: TRAINING ON TASKS WITH PRE-COMPUTED LABELS")
    print("="*70)

    for task_id in range(args.n_tasks):
        # Train on current task using pre-computed negotiated labels
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],  # Not used, kept for interface consistency
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

    # Print final accuracy matrix
    trainer.metrics.print_summary()

    # Save final results
    if args.save_checkpoints:
        results_file = os.path.join(args.checkpoint_dir, 'negotiation_results.txt')
        with open(results_file, 'w') as f:
            f.write("Negotiation Method Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Number of tasks: {args.n_tasks}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Initial α₀: {args.alpha:.3f}\n\n")
            f.write(f"Average Accuracy: {final_metrics['average_accuracy']:.4f}\n")
            f.write(f"Forgetting: {final_metrics['forgetting']:.4f}\n")
            f.write(f"Backward Transfer: {final_metrics['backward_transfer']:.4f}\n\n")
            f.write("Negotiation Rate Progression:\n")
            for i, alpha in enumerate(trainer.negotiation_rate_history):
                f.write(f"  After Task {i-1 if i > 0 else 'Init'}: α = {alpha:.4f}\n")
            f.write("\nAccuracy Matrix:\n")
            f.write(str(trainer.metrics.get_accuracy_matrix()))
        print(f"Results saved to {results_file}")

    # Clean up
    trainer.close()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED")
    print("="*70)


if __name__ == '__main__':
    main()
