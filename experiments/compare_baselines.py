"""
Compare all baseline methods on task-incremental continual learning benchmarks.

This script runs Fine-tuning, EWC, and SI with the same hyperparameters
and collects results for comparison.

Usage:
    python experiments/compare_baselines.py --dataset split_mnist --n_tasks 5
    python experiments/compare_baselines.py --dataset split_cifar10 --n_tasks 5 --epochs 20
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.baselines import FineTuningTrainer, EWCTrainer, SynapticIntelligenceTrainer
from src.utils import get_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare baseline methods')

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

    # Method-specific hyperparameters
    parser.add_argument('--ewc_lambda', type=float, default=1000.0,
                       help='EWC regularization strength')
    parser.add_argument('--ewc_mode', type=str, default='online',
                       choices=['online', 'separate'],
                       help='EWC mode')
    parser.add_argument('--si_lambda', type=float, default=1.0,
                       help='SI regularization strength')
    parser.add_argument('--si_epsilon', type=float, default=0.001,
                       help='SI numerical stability')
    parser.add_argument('--damping', type=float, default=0.1,
                       help='SI damping parameter')

    # Experiment control
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['finetune', 'ewc', 'si'],
                       choices=['finetune', 'ewc', 'si'],
                       help='Methods to compare')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='Random seeds for multiple runs')

    # Logging
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--results_dir', type=str, default='./results/comparisons',
                       help='Directory to save results')

    # Other arguments
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


def run_method(method_name, args, seed, train_loaders, val_loaders, test_loaders,
               classes_per_task, architecture, device):
    """Run a single method and return metrics."""
    print(f"\n{'='*70}")
    print(f"Running {method_name.upper()} (seed={seed})")
    print(f"{'='*70}")

    # Set seed
    set_seed(seed)

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

    # Create trainer based on method
    config = {
        'use_tensorboard': False,
        'use_wandb': False,
    }

    if method_name == 'finetune':
        trainer = FineTuningTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=args.n_tasks,
            num_classes_per_task=classes_per_task,
            config=config
        )
    elif method_name == 'ewc':
        trainer = EWCTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=args.n_tasks,
            num_classes_per_task=classes_per_task,
            config=config,
            ewc_lambda=args.ewc_lambda,
            mode=args.ewc_mode
        )
    elif method_name == 'si':
        trainer = SynapticIntelligenceTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=args.n_tasks,
            num_classes_per_task=classes_per_task,
            config=config,
            si_lambda=args.si_lambda,
            si_epsilon=args.si_epsilon,
            damping=args.damping
        )

    # Train on all tasks
    for task_id in range(args.n_tasks):
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],
            val_loader=val_loaders[task_id],
            epochs=args.epochs
        )

        # Evaluate
        trainer.evaluate_all_tasks(
            task_dataloaders=test_loaders,
            current_task=task_id
        )

    # Get final metrics
    final_metrics = trainer.metrics.get_all_metrics()

    # Clean up
    trainer.close()

    return final_metrics


def main():
    """Main comparison loop."""
    args = parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*70)
    print("BASELINE COMPARISON - TASK-INCREMENTAL CONTINUAL LEARNING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Number of tasks: {args.n_tasks}")
    print(f"  Device: {device}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Methods: {args.methods}")
    print(f"  Epochs per task: {args.epochs}")
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

    # Determine number of classes per task
    classes_per_task = {
        'split_mnist': 10 // args.n_tasks,
        'split_cifar10': 10 // args.n_tasks,
        'split_cifar100': 100 // args.n_tasks,
    }[args.dataset]

    # Get architecture
    architecture = get_architecture(args.dataset, args.architecture)
    print(f"Using architecture: {architecture}\n")

    # Store results
    all_results = {method: [] for method in args.methods}

    # Run experiments
    for seed in args.seeds:
        for method in args.methods:
            metrics = run_method(
                method_name=method,
                args=args,
                seed=seed,
                train_loaders=train_loaders,
                val_loaders=val_loaders,
                test_loaders=test_loaders,
                classes_per_task=classes_per_task,
                architecture=architecture,
                device=device
            )
            all_results[method].append(metrics)

    # Aggregate results (mean and std across seeds)
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    table_data = []
    for method in args.methods:
        results = all_results[method]

        avg_acc = [r['average_accuracy'] for r in results]
        forgetting = [r['forgetting'] for r in results]
        bwt = [r['backward_transfer'] for r in results]

        if len(results) > 1:
            # Multiple seeds: show mean ± std
            table_data.append([
                method.upper(),
                f"{np.mean(avg_acc):.4f} ± {np.std(avg_acc):.4f}",
                f"{np.mean(forgetting):.4f} ± {np.std(forgetting):.4f}",
                f"{np.mean(bwt):.4f} ± {np.std(bwt):.4f}"
            ])
        else:
            # Single seed: show value only
            table_data.append([
                method.upper(),
                f"{avg_acc[0]:.4f}",
                f"{forgetting[0]:.4f}",
                f"{bwt[0]:.4f}"
            ])

    headers = ["Method", "Avg Accuracy", "Forgetting", "Backward Transfer"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    # Save results if requested
    if args.save_results:
        os.makedirs(args.results_dir, exist_ok=True)
        results_file = os.path.join(
            args.results_dir,
            f"{args.dataset}_n{args.n_tasks}_comparison.txt"
        )

        with open(results_file, 'w') as f:
            f.write("Baseline Comparison Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Number of tasks: {args.n_tasks}\n")
            f.write(f"Seeds: {args.seeds}\n")
            f.write(f"Methods: {args.methods}\n")
            f.write(f"Epochs per task: {args.epochs}\n\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
            f.write("\n\nDetailed Results:\n")
            for method in args.methods:
                f.write(f"\n{method.upper()}:\n")
                for i, metrics in enumerate(all_results[method]):
                    f.write(f"  Seed {args.seeds[i]}:\n")
                    f.write(f"    Average Accuracy: {metrics['average_accuracy']:.4f}\n")
                    f.write(f"    Forgetting: {metrics['forgetting']:.4f}\n")
                    f.write(f"    Backward Transfer: {metrics['backward_transfer']:.4f}\n")

        print(f"\nResults saved to {results_file}")

    print("\n" + "="*70)
    print("COMPARISON COMPLETED")
    print("="*70)


if __name__ == '__main__':
    main()
