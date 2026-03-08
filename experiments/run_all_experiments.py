"""
Comprehensive experiment runner for all baselines on all datasets.

This script runs a complete experimental suite:
- All methods: Fine-tuning, EWC, SI, MAS, LwF, Negotiation
- All datasets: Split MNIST, Split CIFAR-10, Split CIFAR-100
- Multiple seeds for statistical significance
- Organized result saving and logging

Usage:
    # Run all experiments with default settings
    python experiments/run_all_experiments.py

    # Run specific datasets
    python experiments/run_all_experiments.py --datasets split_mnist split_cifar10

    # Run specific methods
    python experiments/run_all_experiments.py --methods finetune ewc negotiation

    # Multiple seeds for statistical significance
    python experiments/run_all_experiments.py --seeds 42 43 44

    # Quick test run (fewer epochs)
    python experiments/run_all_experiments.py --quick
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.baselines import (FineTuningTrainer, EWCTrainer, SynapticIntelligenceTrainer,
                            MASTrainer, LwFTrainer, NegotiationTrainer)
from src.utils import get_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive continual learning experiments'
    )

    # Experiment selection
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['split_mnist', 'split_cifar10', 'split_cifar100'],
                       choices=['split_mnist', 'split_cifar10', 'split_cifar100'],
                       help='Datasets to run experiments on')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['finetune', 'ewc', 'si', 'mas', 'lwf', 'negotiation'],
                       choices=['finetune', 'ewc', 'si', 'mas', 'lwf', 'negotiation'],
                       help='Methods to evaluate')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 43, 44],
                       help='Random seeds for multiple runs')

    # Dataset configuration
    parser.add_argument('--n_tasks', type=int, default=None,
                       help='Number of tasks (default: 5 for all datasets)')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')

    # Training configuration
    parser.add_argument('--epochs_mnist', type=int, default=10,
                       help='Epochs per task for MNIST')
    parser.add_argument('--epochs_cifar', type=int, default=20,
                       help='Epochs per task for CIFAR datasets')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam'],
                       help='Optimizer')

    # Method hyperparameters
    parser.add_argument('--ewc_lambda', type=float, default=1000.0,
                       help='EWC regularization strength')
    parser.add_argument('--ewc_mode', type=str, default='online',
                       choices=['online', 'separate'],
                       help='EWC mode')
    parser.add_argument('--si_lambda', type=float, default=1.0,
                       help='SI regularization strength')
    parser.add_argument('--si_epsilon', type=float, default=0.001,
                       help='SI epsilon')
    parser.add_argument('--damping', type=float, default=0.1,
                       help='SI damping')
    parser.add_argument('--mas_lambda', type=float, default=1.0,
                       help='MAS regularization strength')
    parser.add_argument('--mas_n_samples', type=int, default=200,
                       help='MAS number of samples for importance estimation')
    parser.add_argument('--lwf_lambda', type=float, default=1.0,
                       help='LwF distillation strength')
    parser.add_argument('--lwf_temperature', type=float, default=2.0,
                       help='LwF distillation temperature')
    parser.add_argument('--negotiation_alpha', type=float, default=0.5,
                       help='Negotiation rate (constant at 0.5)')
    parser.add_argument('--update_negotiation_alpha', action='store_true',
                       help='Use adaptive negotiation rate (default: constant)')

    # Output configuration
    parser.add_argument('--results_dir', type=str,
                       default='./results/experiments',
                       help='Directory to save all results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment run (default: timestamp)')

    # Convenience flags
    parser.add_argument('--quick', action='store_true',
                       help='Quick test run (1 seed, 5 epochs)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset_config(dataset_name):
    """Get default configuration for each dataset."""
    configs = {
        'split_mnist': {
            'n_tasks': 5,
            'architecture': 'mlp',
            'hidden_size': 256,
            'dropout': 0.0,
        },
        'split_cifar10': {
            'n_tasks': 5,
            'architecture': 'convnet',
            'hidden_size': None,
            'dropout': 0.0,
        },
        'split_cifar100': {
            'n_tasks': 10,
            'architecture': 'convnet',
            'hidden_size': None,
            'dropout': 0.0,
        }
    }
    return configs[dataset_name]


def create_trainer(method_name, model, optimizer, criterion, device,
                   num_tasks, num_classes_per_task, args):
    """Create a trainer for the specified method."""
    config = {
        'use_tensorboard': False,
        'use_wandb': False,
    }

    if method_name == 'finetune':
        return FineTuningTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=num_classes_per_task,
            config=config
        )
    elif method_name == 'ewc':
        return EWCTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=num_classes_per_task,
            config=config,
            ewc_lambda=args.ewc_lambda,
            mode=args.ewc_mode
        )
    elif method_name == 'si':
        return SynapticIntelligenceTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=num_classes_per_task,
            config=config,
            si_lambda=args.si_lambda,
            si_epsilon=args.si_epsilon,
            damping=args.damping
        )
    elif method_name == 'mas':
        return MASTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=num_classes_per_task,
            config=config,
            mas_lambda=args.mas_lambda,
            num_samples=args.mas_n_samples
        )
    elif method_name == 'lwf':
        return LwFTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=num_classes_per_task,
            config=config,
            lwf_lambda=args.lwf_lambda,
            temperature=args.lwf_temperature
        )
    elif method_name == 'negotiation':
        return NegotiationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=num_classes_per_task,
            config=config,
            initial_negotiation_rate=args.negotiation_alpha,
            update_negotiation_rate=args.update_negotiation_alpha
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")


def run_single_experiment(method_name, dataset_name, seed, args, device, exp_dir):
    """Run a single experiment configuration."""

    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    n_tasks = args.n_tasks if args.n_tasks is not None else dataset_config['n_tasks']
    architecture = dataset_config['architecture']

    # Determine epochs
    if 'mnist' in dataset_name:
        epochs = args.epochs_mnist
    else:
        epochs = args.epochs_cifar

    if args.verbose:
        print(f"\n{'='*70}")
        print(f"Running: {method_name.upper()} on {dataset_name} (seed={seed})")
        print(f"{'='*70}")

    # Set seed
    set_seed(seed)

    # Load dataset
    train_loaders, val_loaders, test_loaders = get_dataset(
        dataset_name=dataset_name,
        n_tasks=n_tasks,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=0.0
    )

    # Get number of classes per task
    total_classes = {
        'split_mnist': 10,
        'split_cifar10': 10,
        'split_cifar100': 100
    }[dataset_name]
    classes_per_task = total_classes // n_tasks

    # Create model
    model_kwargs = {}
    if architecture == 'mlp':
        model_kwargs['hidden_size'] = dataset_config['hidden_size']

    model = get_model(
        architecture=architecture,
        num_classes_per_task=classes_per_task,
        num_tasks=n_tasks,
        dropout=dataset_config['dropout'],
        **model_kwargs
    )

    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.0
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=0.0
        )

    # Create loss criterion
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = create_trainer(
        method_name=method_name,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_tasks=n_tasks,
        num_classes_per_task=classes_per_task,
        args=args
    )

    # Track per-task accuracies
    task_accuracies = []

    # Train on all tasks
    for task_id in range(n_tasks):
        if args.verbose:
            print(f"\n--- Training on Task {task_id} ---")

        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],
            val_loader=val_loaders[task_id],
            epochs=epochs
        )

        # Evaluate on all tasks seen so far
        accuracies = trainer.evaluate_all_tasks(
            task_dataloaders=test_loaders,
            current_task=task_id
        )
        task_accuracies.append(accuracies)

    # Get final metrics
    final_metrics = trainer.metrics.get_all_metrics()
    accuracy_matrix = trainer.metrics.get_accuracy_matrix()

    # Clean up
    trainer.close()

    # Save results for this run
    result = {
        'method': method_name,
        'dataset': dataset_name,
        'seed': seed,
        'n_tasks': n_tasks,
        'epochs': epochs,
        'architecture': architecture,
        'final_metrics': {
            'average_accuracy': float(final_metrics['average_accuracy']),
            'forgetting': float(final_metrics['forgetting']),
            'backward_transfer': float(final_metrics['backward_transfer'])
        },
        'accuracy_matrix': accuracy_matrix.tolist(),
        'task_accuracies': task_accuracies,
        'hyperparameters': {
            'lr': args.lr,
            'batch_size': args.batch_size,
            'optimizer': args.optimizer,
        }
    }

    # Add method-specific hyperparameters
    if method_name == 'ewc':
        result['hyperparameters']['ewc_lambda'] = args.ewc_lambda
        result['hyperparameters']['ewc_mode'] = args.ewc_mode
    elif method_name == 'si':
        result['hyperparameters']['si_lambda'] = args.si_lambda
        result['hyperparameters']['si_epsilon'] = args.si_epsilon
        result['hyperparameters']['damping'] = args.damping
    elif method_name == 'mas':
        result['hyperparameters']['mas_lambda'] = args.mas_lambda
        result['hyperparameters']['mas_n_samples'] = args.mas_n_samples
    elif method_name == 'lwf':
        result['hyperparameters']['lwf_lambda'] = args.lwf_lambda
        result['hyperparameters']['lwf_temperature'] = args.lwf_temperature
    elif method_name == 'negotiation':
        result['hyperparameters']['negotiation_alpha'] = args.negotiation_alpha
        result['hyperparameters']['update_negotiation_alpha'] = args.update_negotiation_alpha

    # Save individual result
    result_file = exp_dir / f"{method_name}_{dataset_name}_seed{seed}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    if args.verbose:
        print(f"\nResults saved to {result_file}")
        print(f"Average Accuracy: {final_metrics['average_accuracy']:.4f}")
        print(f"Forgetting: {final_metrics['forgetting']:.4f}")

    return result


def aggregate_results(all_results, output_dir):
    """Aggregate results across seeds and create summary reports."""

    # Group by method and dataset
    grouped = {}
    for result in all_results:
        key = (result['method'], result['dataset'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    # Compute statistics
    summary = []
    for (method, dataset), results in grouped.items():
        avg_accs = [r['final_metrics']['average_accuracy'] for r in results]
        forgettings = [r['final_metrics']['forgetting'] for r in results]
        bwts = [r['final_metrics']['backward_transfer'] for r in results]

        summary.append({
            'method': method,
            'dataset': dataset,
            'n_seeds': len(results),
            'avg_accuracy_mean': np.mean(avg_accs),
            'avg_accuracy_std': np.std(avg_accs),
            'forgetting_mean': np.mean(forgettings),
            'forgetting_std': np.std(forgettings),
            'bwt_mean': np.mean(bwts),
            'bwt_std': np.std(bwts),
        })

    # Save aggregated summary
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def create_summary_report(summary, output_dir, args):
    """Create human-readable summary report."""

    report_file = output_dir / 'SUMMARY_REPORT.txt'

    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CONTINUAL LEARNING EXPERIMENTS - SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Datasets: {', '.join(args.datasets)}\n")
        f.write(f"Methods: {', '.join(args.methods)}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Number of runs per configuration: {len(args.seeds)}\n\n")

        # Group by dataset
        datasets = sorted(set(s['dataset'] for s in summary))

        for dataset in datasets:
            f.write("\n" + "="*70 + "\n")
            f.write(f"DATASET: {dataset.upper()}\n")
            f.write("="*70 + "\n\n")

            dataset_results = [s for s in summary if s['dataset'] == dataset]

            # Create table
            table_data = []
            for s in dataset_results:
                if s['n_seeds'] > 1:
                    table_data.append([
                        s['method'].upper(),
                        f"{s['avg_accuracy_mean']:.4f} ± {s['avg_accuracy_std']:.4f}",
                        f"{s['forgetting_mean']:.4f} ± {s['forgetting_std']:.4f}",
                        f"{s['bwt_mean']:.4f} ± {s['bwt_std']:.4f}",
                    ])
                else:
                    table_data.append([
                        s['method'].upper(),
                        f"{s['avg_accuracy_mean']:.4f}",
                        f"{s['forgetting_mean']:.4f}",
                        f"{s['bwt_mean']:.4f}",
                    ])

            headers = ["Method", "Avg Accuracy", "Forgetting", "Backward Transfer"]
            table_str = tabulate(table_data, headers=headers, tablefmt="grid")
            f.write(table_str + "\n")

        # Overall summary
        f.write("\n" + "="*70 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("="*70 + "\n\n")

        # Find best methods per dataset
        for dataset in datasets:
            dataset_results = [s for s in summary if s['dataset'] == dataset]
            best_acc = max(dataset_results, key=lambda x: x['avg_accuracy_mean'])
            best_forget = min(dataset_results, key=lambda x: x['forgetting_mean'])

            f.write(f"\n{dataset}:\n")
            f.write(f"  Best Average Accuracy: {best_acc['method'].upper()} "
                   f"({best_acc['avg_accuracy_mean']:.4f})\n")
            f.write(f"  Lowest Forgetting: {best_forget['method'].upper()} "
                   f"({best_forget['forgetting_mean']:.4f})\n")

        f.write("\n" + "="*70 + "\n")
        f.write("Hyperparameters:\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Optimizer: {args.optimizer}\n")
        if 'ewc' in args.methods:
            f.write(f"  EWC lambda: {args.ewc_lambda}\n")
            f.write(f"  EWC mode: {args.ewc_mode}\n")
        if 'si' in args.methods:
            f.write(f"  SI lambda: {args.si_lambda}\n")
            f.write(f"  SI epsilon: {args.si_epsilon}\n")
            f.write(f"  SI damping: {args.damping}\n")
        if 'mas' in args.methods:
            f.write(f"  MAS lambda: {args.mas_lambda}\n")
            f.write(f"  MAS n_samples: {args.mas_n_samples}\n")
        if 'lwf' in args.methods:
            f.write(f"  LwF lambda: {args.lwf_lambda}\n")
            f.write(f"  LwF temperature: {args.lwf_temperature}\n")
        if 'negotiation' in args.methods:
            f.write(f"  Negotiation alpha: {args.negotiation_alpha}\n")
            f.write(f"  Update negotiation alpha: {args.update_negotiation_alpha}\n")

        f.write("\n" + "="*70 + "\n")
        f.write(f"All individual results saved in: {output_dir}\n")
        f.write("="*70 + "\n")

    return report_file


def main():
    """Main experiment runner."""
    args = parse_args()

    # Quick mode settings
    if args.quick:
        args.seeds = [42]
        args.epochs_mnist = 5
        args.epochs_cifar = 5
        print("\n⚡ QUICK MODE: Running with 1 seed and 5 epochs per task\n")

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Create experiment directory
    if args.experiment_name is None:
        experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        experiment_name = args.experiment_name

    output_dir = Path(args.results_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration
    config_file = output_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("="*70)
    print("COMPREHENSIVE CONTINUAL LEARNING EXPERIMENTS")
    print("="*70)
    print(f"\nExperiment ID: {experiment_name}")
    print(f"Results directory: {output_dir}")
    print(f"Device: {device}")
    print(f"\nConfiguration:")
    print(f"  Datasets: {args.datasets}")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Total experiments: {len(args.datasets) * len(args.methods) * len(args.seeds)}")
    print("="*70)

    # Run all experiments
    all_results = []
    total = len(args.datasets) * len(args.methods) * len(args.seeds)
    current = 0

    for dataset in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                current += 1
                print(f"\n[{current}/{total}] Running: {method} on {dataset} (seed={seed})")

                result = run_single_experiment(
                    method_name=method,
                    dataset_name=dataset,
                    seed=seed,
                    args=args,
                    device=device,
                    exp_dir=output_dir
                )
                all_results.append(result)

                print(f"  ✓ Avg Acc: {result['final_metrics']['average_accuracy']:.4f}, "
                      f"Forgetting: {result['final_metrics']['forgetting']:.4f}")

    # Aggregate results
    print("\n" + "="*70)
    print("Aggregating results...")
    summary = aggregate_results(all_results, output_dir)

    # Create summary report
    report_file = create_summary_report(summary, output_dir, args)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary report: {report_file}")
    print(f"Individual results: {len(all_results)} JSON files")
    print("\n" + "="*70)

    # Print quick summary to console
    print("\nQUICK SUMMARY:\n")

    # Group by dataset for console output
    datasets = sorted(set(s['dataset'] for s in summary))
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        dataset_results = [s for s in summary if s['dataset'] == dataset]

        table_data = []
        for s in dataset_results:
            if s['n_seeds'] > 1:
                table_data.append([
                    s['method'].upper(),
                    f"{s['avg_accuracy_mean']:.4f} ± {s['avg_accuracy_std']:.4f}",
                    f"{s['forgetting_mean']:.4f} ± {s['forgetting_std']:.4f}",
                ])
            else:
                table_data.append([
                    s['method'].upper(),
                    f"{s['avg_accuracy_mean']:.4f}",
                    f"{s['forgetting_mean']:.4f}",
                ])

        headers = ["Method", "Avg Accuracy", "Forgetting"]
        print(tabulate(table_data, headers=headers, tablefmt="simple"))

    print(f"\nFor detailed results, see: {report_file}")
    print("="*70)


if __name__ == '__main__':
    main()
