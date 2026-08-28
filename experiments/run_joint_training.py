"""
Run Joint Training (upper bound) on task-incremental continual learning benchmarks.

Joint Training trains a single model on ALL tasks simultaneously (data from all
tasks is visited in every epoch, with per-task head routing). It is the standard
upper bound in continual learning: tasks are not learned sequentially, so there
is no forgetting by construction.

Usage:
    python experiments/run_joint_training.py --dataset split_mnist --epochs 50 --seed 42
    python experiments/run_joint_training.py --dataset split_cifar10 --epochs 50 --seed 42
    python experiments/run_joint_training.py --dataset split_cifar100 --n_tasks 10 --epochs 50 --seed 42

Note: 'epochs' counts passes over EACH task's training data, matching the
per-task epoch budget of the sequential experiments (equal-compute upper bound).

Author: prepared for the ASOC submission (FAZ 2.1), follows the repository's
existing script pattern (run_finetune.py / run_all_experiments.py).
"""

import argparse
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.baselines import FineTuningTrainer
from src.utils import get_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Joint Training upper bound (task-incremental)')

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
    parser.add_argument('--epochs', type=int, default=50,
                       help='Passes over each task per joint epoch (default 50: equal-compute budget)')
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

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./results/joint_training',
                       help='Directory for result JSON files')

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
    """Main joint-training loop."""
    args = parse_args()

    set_seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("JOINT TRAINING (UPPER BOUND) - TASK-INCREMENTAL CONTINUAL LEARNING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Number of tasks: {args.n_tasks}")
    print(f"  Device: {device}")
    print(f"  Random seed: {args.seed}")
    print(f"  Joint epochs (passes per task): {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
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

    classes_per_task = {
        'split_mnist': 10 // args.n_tasks,
        'split_cifar10': 10 // args.n_tasks,
        'split_cifar100': 100 // args.n_tasks,
    }[args.dataset]

    print(f"Dataset loaded: {args.n_tasks} tasks, {classes_per_task} classes per task")

    architecture = get_architecture(args.dataset, args.architecture)
    print(f"Using architecture: {architecture}")

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

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    # No tensorboard/wandb/checkpoints: keep the run minimal and fast
    config = {
        'use_tensorboard': False,
        'use_wandb': False,
        'save_checkpoints': False,
    }

    trainer = FineTuningTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_tasks=args.n_tasks,
        num_classes_per_task=classes_per_task,
        config=config
    )

    print("\n" + "=" * 70)
    print("STARTING JOINT TRAINING (all tasks simultaneously)")
    print("=" * 70)

    start_time = time.time()

    # Joint epoch = one full pass over EACH task's training data.
    # This matches the per-task epoch budget of the sequential experiments.
    for epoch in range(args.epochs):
        epoch_losses = []
        for task_id in range(args.n_tasks):
            stats = trainer._train_epoch(task_id, train_loaders[task_id], epoch)
            epoch_losses.append(stats['loss'])
        print(f"Joint epoch {epoch + 1}/{args.epochs} - mean task loss: {np.mean(epoch_losses):.4f}")

    train_time = time.time() - start_time
    print(f"\nJoint training completed in {train_time:.1f}s")

    # Final evaluation on all tasks
    accuracies = trainer.evaluate_all_tasks(
        task_dataloaders=test_loaders,
        current_task=args.n_tasks - 1
    )

    task_accuracies = [accuracies[t] for t in range(args.n_tasks)]
    avg_acc = float(np.mean(task_accuracies))

    print("\n" + "=" * 70)
    print("FINAL RESULTS (JOINT TRAINING UPPER BOUND)")
    print("=" * 70)
    print(f"Average Accuracy: {avg_acc:.4f}")
    print("Note: forgetting and backward transfer are 0 by construction")
    print("(tasks are learned simultaneously, not sequentially).")

    # Save result JSON (same schema as run_all_experiments.py)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'method': 'joint',
        'dataset': args.dataset,
        'seed': args.seed,
        'n_tasks': args.n_tasks,
        'epochs': args.epochs,
        'architecture': architecture,
        'upper_bound': True,
        'final_metrics': {
            'average_accuracy': avg_acc,
            'forgetting': 0.0,          # zero by construction (no sequential learning)
            'backward_transfer': 0.0    # zero by construction
        },
        'accuracy_matrix': [task_accuracies],
        'task_accuracies': task_accuracies,
        'train_time_sec': train_time,
        'hyperparameters': {
            'lr': args.lr,
            'batch_size': args.batch_size,
            'optimizer': args.optimizer,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
        },
        'note': 'Joint training upper bound: single model trained on all tasks '
                'simultaneously with per-task head routing. Forgetting/BWT are '
                'zero by construction, not measured.',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    result_file = output_dir / f"joint_{args.dataset}_seed{args.seed}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {result_file}")

    trainer.close()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == '__main__':
    main()
