"""
EWC Hyperparameter Grid Search for Task-Incremental Continual Learning.

This script performs a grid search over EWC hyperparameters (lambda, mode) to find
optimal settings for CIFAR-10 and CIFAR-100 datasets. Results are saved with unique
filenames to prevent overwriting.

Usage:
    # Search on CIFAR-10
    python experiments/ewc_hyperparam_search.py --dataset split_cifar10 --n_tasks 5

    # Search on CIFAR-100
    python experiments/ewc_hyperparam_search.py --dataset split_cifar100 --n_tasks 10

    # Custom lambda values
    python experiments/ewc_hyperparam_search.py --dataset split_cifar10 --lambda_values 1 10 100 500

    # Include MNIST in search
    python experiments/ewc_hyperparam_search.py --dataset split_mnist --n_tasks 5
"""

import argparse
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from tabulate import tabulate
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.baselines import EWCTrainer
from src.utils import get_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EWC hyperparameter grid search')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='split_cifar10',
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

    # EWC hyperparameter grid
    parser.add_argument('--lambda_values', type=float, nargs='+',
                       default=[1, 10, 50, 100, 500, 1000, 5000],
                       help='EWC lambda values to search over')
    parser.add_argument('--ewc_modes', type=str, nargs='+',
                       default=['online'],
                       choices=['online', 'separate'],
                       help='EWC modes to search over')
    parser.add_argument('--ewc_gamma', type=float, default=1.0,
                       help='Decay factor for online EWC')

    # Experiment arguments
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='Random seeds to use')
    parser.add_argument('--results_dir', type=str, default='./results/hyperparam_search',
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


def run_single_experiment(args, ewc_lambda, ewc_mode, seed, device, architecture, classes_per_task):
    """Run a single experiment with given hyperparameters."""

    # Set random seed
    set_seed(seed)

    # Load dataset
    train_loaders, val_loaders, test_loaders = get_dataset(
        dataset_name=args.dataset,
        n_tasks=args.n_tasks,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=0.0
    )

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

    # Create configuration (no logging to speed up)
    config = {
        'use_tensorboard': False,
        'use_wandb': False,
        'save_checkpoints': False,
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
        ewc_lambda=ewc_lambda,
        mode=ewc_mode,
        gamma=args.ewc_gamma
    )

    # Train on each task sequentially
    for task_id in range(args.n_tasks):
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],
            val_loader=val_loaders[task_id],
            epochs=args.epochs
        )

        trainer.evaluate_all_tasks(
            task_dataloaders=test_loaders,
            current_task=task_id
        )

    # Get final metrics
    final_metrics = trainer.metrics.get_all_metrics()

    # Clean up
    trainer.close()

    return {
        'avg_accuracy': final_metrics['average_accuracy'],
        'forgetting': final_metrics['forgetting'],
        'backward_transfer': final_metrics['backward_transfer'],
        'accuracy_matrix': trainer.metrics.get_accuracy_matrix().tolist()
    }


def main():
    """Main hyperparameter search loop."""
    args = parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Get architecture
    architecture = get_architecture(args.dataset, args.architecture)

    # Determine number of classes per task
    classes_per_task = {
        'split_mnist': 10 // args.n_tasks,
        'split_cifar10': 10 // args.n_tasks,
        'split_cifar100': 100 // args.n_tasks,
    }[args.dataset]

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / f"ewc_{args.dataset}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EWC HYPERPARAMETER GRID SEARCH")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Number of tasks: {args.n_tasks}")
    print(f"  Architecture: {architecture}")
    print(f"  Device: {device}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Epochs per task: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"\n  Lambda values: {args.lambda_values}")
    print(f"  EWC modes: {args.ewc_modes}")
    print(f"\n  Results directory: {results_dir}")
    print()

    # Create grid of hyperparameters
    total_experiments = len(args.lambda_values) * len(args.ewc_modes) * len(args.seeds)
    print(f"Total experiments to run: {total_experiments}")
    print("="*80)
    print()

    # Store all results
    all_results = []
    experiment_count = 0

    # Grid search
    for ewc_lambda in args.lambda_values:
        for ewc_mode in args.ewc_modes:
            for seed in args.seeds:
                experiment_count += 1

                print(f"\n[{experiment_count}/{total_experiments}] Running experiment:")
                print(f"  Lambda: {ewc_lambda}, Mode: {ewc_mode}, Seed: {seed}")
                print("-" * 80)

                try:
                    # Run experiment
                    result = run_single_experiment(
                        args=args,
                        ewc_lambda=ewc_lambda,
                        ewc_mode=ewc_mode,
                        seed=seed,
                        device=device,
                        architecture=architecture,
                        classes_per_task=classes_per_task
                    )

                    # Store result with hyperparameters
                    result_entry = {
                        'dataset': args.dataset,
                        'n_tasks': args.n_tasks,
                        'ewc_lambda': ewc_lambda,
                        'ewc_mode': ewc_mode,
                        'seed': seed,
                        'avg_accuracy': result['avg_accuracy'],
                        'forgetting': result['forgetting'],
                        'backward_transfer': result['backward_transfer'],
                        'accuracy_matrix': result['accuracy_matrix']
                    }

                    all_results.append(result_entry)

                    # Save individual result with unique filename
                    result_filename = f"ewc_lambda{ewc_lambda}_mode{ewc_mode}_seed{seed}.json"
                    result_filepath = results_dir / result_filename

                    with open(result_filepath, 'w') as f:
                        json.dump(result_entry, f, indent=2)

                    print(f"  ✓ Avg Accuracy: {result['avg_accuracy']:.4f}")
                    print(f"  ✓ Forgetting: {result['forgetting']:.4f}")
                    print(f"  ✓ Saved to: {result_filename}")

                except Exception as e:
                    print(f"  ✗ FAILED: {str(e)}")
                    continue

    # Compute aggregated statistics per hyperparameter setting
    print("\n" + "="*80)
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print("="*80)

    aggregated_results = []

    for ewc_lambda in args.lambda_values:
        for ewc_mode in args.ewc_modes:
            # Filter results for this hyperparameter setting
            filtered_results = [
                r for r in all_results
                if r['ewc_lambda'] == ewc_lambda and r['ewc_mode'] == ewc_mode
            ]

            if not filtered_results:
                continue

            # Compute mean and std
            avg_accuracies = [r['avg_accuracy'] for r in filtered_results]
            forgettings = [r['forgetting'] for r in filtered_results]
            bwts = [r['backward_transfer'] for r in filtered_results]

            aggregated_results.append({
                'ewc_lambda': ewc_lambda,
                'ewc_mode': ewc_mode,
                'n_seeds': len(filtered_results),
                'avg_accuracy_mean': np.mean(avg_accuracies),
                'avg_accuracy_std': np.std(avg_accuracies),
                'forgetting_mean': np.mean(forgettings),
                'forgetting_std': np.std(forgettings),
                'bwt_mean': np.mean(bwts),
                'bwt_std': np.std(bwts)
            })

    # Sort by average accuracy (descending)
    aggregated_results.sort(key=lambda x: x['avg_accuracy_mean'], reverse=True)

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE (Sorted by Average Accuracy)")
    print("="*80)

    table_data = []
    for r in aggregated_results:
        table_data.append([
            f"{r['ewc_lambda']:.0f}",
            r['ewc_mode'],
            f"{r['avg_accuracy_mean']:.4f} ± {r['avg_accuracy_std']:.4f}",
            f"{r['forgetting_mean']:.4f} ± {r['forgetting_std']:.4f}",
            f"{r['bwt_mean']:.4f} ± {r['bwt_std']:.4f}",
            r['n_seeds']
        ])

    headers = ['Lambda', 'Mode', 'Avg Accuracy', 'Forgetting', 'BWT', 'Seeds']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # Find best hyperparameters
    best_result = aggregated_results[0]
    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS")
    print("="*80)
    print(f"  Lambda: {best_result['ewc_lambda']}")
    print(f"  Mode: {best_result['ewc_mode']}")
    print(f"  Average Accuracy: {best_result['avg_accuracy_mean']:.4f} ± {best_result['avg_accuracy_std']:.4f}")
    print(f"  Forgetting: {best_result['forgetting_mean']:.4f} ± {best_result['forgetting_std']:.4f}")
    print()

    # Save aggregated results
    aggregated_filepath = results_dir / "aggregated_results.json"
    with open(aggregated_filepath, 'w') as f:
        json.dump(aggregated_results, f, indent=2)

    # Save summary report
    summary_filepath = results_dir / "SUMMARY.txt"
    with open(summary_filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EWC HYPERPARAMETER SEARCH - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Number of tasks: {args.n_tasks}\n")
        f.write(f"Architecture: {architecture}\n")
        f.write(f"Epochs per task: {args.epochs}\n")
        f.write(f"Seeds: {args.seeds}\n\n")
        f.write(f"Lambda values tested: {args.lambda_values}\n")
        f.write(f"Modes tested: {args.ewc_modes}\n")
        f.write(f"Total experiments: {len(all_results)}\n\n")
        f.write("="*80 + "\n")
        f.write("RESULTS TABLE\n")
        f.write("="*80 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("BEST HYPERPARAMETERS\n")
        f.write("="*80 + "\n")
        f.write(f"  Lambda: {best_result['ewc_lambda']}\n")
        f.write(f"  Mode: {best_result['ewc_mode']}\n")
        f.write(f"  Average Accuracy: {best_result['avg_accuracy_mean']:.4f} ± {best_result['avg_accuracy_std']:.4f}\n")
        f.write(f"  Forgetting: {best_result['forgetting_mean']:.4f} ± {best_result['forgetting_std']:.4f}\n")

    print(f"Summary saved to: {summary_filepath}")
    print(f"All results saved to: {results_dir}")

    print("\n" + "="*80)
    print("GRID SEARCH COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()
