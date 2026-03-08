"""
Run experiments for MAS and LwF baselines that haven't been tested yet.

This script runs comprehensive experiments for:
- MAS (Memory Aware Synapses) - estimated λ=1.0
- LwF (Learning without Forgetting) - estimated λ=1.0, T=2.0

On all datasets:
- Split MNIST
- Split CIFAR-10
- Split CIFAR-100

Usage:
    # Run all untested baselines on all datasets
    python experiments/run_untested_baselines.py

    # Run specific method
    python experiments/run_untested_baselines.py --methods mas

    # Run on specific datasets
    python experiments/run_untested_baselines.py --datasets split_cifar10 split_cifar100

    # With multiple seeds for statistical significance
    python experiments/run_untested_baselines.py --seeds 42 43 44
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
from pathlib import Path
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.baselines import MASTrainer, LwFTrainer
from src.utils import get_dataset
from configs.config_loader import load_config, get_method_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run untested baselines (MAS, LwF) on all datasets'
    )

    # Experiment selection
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['mas', 'lwf'],
                       choices=['mas', 'lwf'],
                       help='Methods to test')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['split_mnist', 'split_cifar10', 'split_cifar100'],
                       choices=['split_mnist', 'split_cifar10', 'split_cifar100'],
                       help='Datasets to test on')

    # Experimental settings
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='Random seeds')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Epochs per task')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')

    # Output
    parser.add_argument('--results_dir', type=str, default='./results/untested_baselines',
                       help='Directory to save results')
    parser.add_argument('--save_detailed', action='store_true',
                       help='Save detailed per-seed results')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_experiment(method_name, dataset_name, seed, device, epochs):
    """
    Run a single experiment.

    Args:
        method_name: 'mas' or 'lwf'
        dataset_name: Dataset name
        seed: Random seed
        device: Device
        epochs: Epochs per task

    Returns:
        Dictionary with results
    """
    # Load configs
    config = load_config(dataset_name)
    dataset_config = config['dataset']
    model_config = config['model']
    training_config = config['training']
    optimizer_config = config['optimizer']
    method_config = config[method_name]

    # Set seed
    set_seed(seed)

    # Load dataset
    train_loaders, val_loaders, test_loaders = get_dataset(
        dataset_name=dataset_config['name'],
        n_tasks=dataset_config['n_tasks'],
        data_root=dataset_config['data_root'],
        batch_size=training_config['batch_size'],
        num_workers=training_config['num_workers'],
        validation_split=0.0
    )

    # Create model
    model_kwargs = {}
    if model_config['architecture'] == 'mlp':
        model_kwargs['hidden_size'] = model_config.get('hidden_size', 256)

    model = get_model(
        architecture=model_config['architecture'],
        num_classes_per_task=dataset_config['classes_per_task'],
        num_tasks=dataset_config['n_tasks'],
        dropout=model_config.get('dropout', 0.0),
        **model_kwargs
    )

    # Create optimizer
    if optimizer_config['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
    elif optimizer_config['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer_config = {
        'use_tensorboard': False,
        'use_wandb': False,
        'save_checkpoints': False,
    }

    if method_name == 'mas':
        trainer = MASTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=dataset_config['n_tasks'],
            num_classes_per_task=dataset_config['classes_per_task'],
            config=trainer_config,
            mas_lambda=method_config['lambda'],
            num_samples=method_config.get('num_samples', 200)
        )
    elif method_name == 'lwf':
        trainer = LwFTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=dataset_config['n_tasks'],
            num_classes_per_task=dataset_config['classes_per_task'],
            config=trainer_config,
            lwf_lambda=method_config['lambda'],
            temperature=method_config.get('temperature', 2.0)
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Train on all tasks
    for task_id in range(dataset_config['n_tasks']):
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],
            val_loader=val_loaders[task_id],
            epochs=epochs
        )

        trainer.evaluate_all_tasks(
            task_dataloaders=test_loaders,
            current_task=task_id
        )

    # Get final metrics
    final_metrics = trainer.metrics.get_all_metrics()
    trainer.close()

    return {
        'method': method_name,
        'dataset': dataset_name,
        'seed': seed,
        'avg_accuracy': final_metrics['average_accuracy'],
        'forgetting': final_metrics['forgetting'],
        'backward_transfer': final_metrics['backward_transfer'],
        'accuracy_matrix': trainer.metrics.get_accuracy_matrix().tolist()
    }


def main():
    """Main entry point."""
    args = parse_args()

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TESTING UNTESTED BASELINES: MAS & LwF")
    print("="*80)
    print(f"\nMethods: {args.methods}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {device}")
    print(f"Epochs per task: {args.epochs}")
    print(f"Results directory: {results_dir}")
    print()

    # Calculate total experiments
    total_experiments = len(args.methods) * len(args.datasets) * len(args.seeds)
    print(f"Total experiments to run: {total_experiments}")
    print("="*80)
    print()

    # Run all experiments
    all_results = []
    experiment_count = 0

    for method in args.methods:
        for dataset in args.datasets:
            for seed in args.seeds:
                experiment_count += 1

                print(f"\n[{experiment_count}/{total_experiments}] Running:")
                print(f"  Method: {method.upper()}")
                print(f"  Dataset: {dataset}")
                print(f"  Seed: {seed}")
                print("-" * 80)

                try:
                    result = run_single_experiment(
                        method_name=method,
                        dataset_name=dataset,
                        seed=seed,
                        device=device,
                        epochs=args.epochs
                    )

                    all_results.append(result)

                    print(f"  ✓ Avg Accuracy: {result['avg_accuracy']:.4f}")
                    print(f"  ✓ Forgetting: {result['forgetting']:.4f}")
                    print(f"  ✓ Backward Transfer: {result['backward_transfer']:.4f}")

                    # Save individual result
                    if args.save_detailed:
                        result_file = results_dir / f"{method}_{dataset}_seed{seed}.json"
                        with open(result_file, 'w') as f:
                            json.dump(result, f, indent=2)

                except Exception as e:
                    print(f"  ✗ FAILED: {str(e)}")
                    continue

    # Aggregate results
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)

    aggregated_results = []

    for method in args.methods:
        for dataset in args.datasets:
            # Filter results
            filtered = [r for r in all_results
                       if r['method'] == method and r['dataset'] == dataset]

            if not filtered:
                continue

            # Compute statistics
            avg_accs = [r['avg_accuracy'] for r in filtered]
            forgettings = [r['forgetting'] for r in filtered]
            bwts = [r['backward_transfer'] for r in filtered]

            aggregated_results.append({
                'method': method,
                'dataset': dataset,
                'n_seeds': len(filtered),
                'avg_accuracy_mean': np.mean(avg_accs),
                'avg_accuracy_std': np.std(avg_accs),
                'forgetting_mean': np.mean(forgettings),
                'forgetting_std': np.std(forgettings),
                'bwt_mean': np.mean(bwts),
                'bwt_std': np.std(bwts)
            })

    # Save aggregated results
    aggregated_file = results_dir / "aggregated_results.json"
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)

    # Print results by dataset
    print("\n")
    for dataset in args.datasets:
        dataset_results = [r for r in aggregated_results if r['dataset'] == dataset]

        if not dataset_results:
            continue

        print("="*80)
        print(f"RESULTS: {dataset.upper()}")
        print("="*80)

        table_data = []
        for r in dataset_results:
            table_data.append([
                r['method'].upper(),
                f"{r['avg_accuracy_mean']:.4f} ± {r['avg_accuracy_std']:.4f}",
                f"{r['forgetting_mean']:.4f} ± {r['forgetting_std']:.4f}",
                f"{r['bwt_mean']:.4f} ± {r['bwt_std']:.4f}",
                r['n_seeds']
            ])

        headers = ['Method', 'Avg Accuracy', 'Forgetting', 'BWT', 'Seeds']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()

    # Create summary report
    summary_file = results_dir / "SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("UNTESTED BASELINES RESULTS - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Methods tested: {args.methods}\n")
        f.write(f"Datasets: {args.datasets}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Epochs per task: {args.epochs}\n")
        f.write(f"Total experiments: {len(all_results)}\n\n")

        for dataset in args.datasets:
            dataset_results = [r for r in aggregated_results if r['dataset'] == dataset]

            if not dataset_results:
                continue

            f.write("="*80 + "\n")
            f.write(f"DATASET: {dataset.upper()}\n")
            f.write("="*80 + "\n\n")

            table_data = []
            for r in dataset_results:
                table_data.append([
                    r['method'].upper(),
                    f"{r['avg_accuracy_mean']:.4f} ± {r['avg_accuracy_std']:.4f}",
                    f"{r['forgetting_mean']:.4f} ± {r['forgetting_std']:.4f}",
                    f"{r['bwt_mean']:.4f} ± {r['bwt_std']:.4f}",
                    r['n_seeds']
                ])

            headers = ['Method', 'Avg Accuracy', 'Forgetting', 'BWT', 'Seeds']
            f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
            f.write("\n\n")

        # Add comparison with existing baselines
        f.write("="*80 + "\n")
        f.write("COMPARISON WITH EXISTING BASELINES\n")
        f.write("="*80 + "\n\n")
        f.write("For reference, here are the existing baseline results:\n\n")

        f.write("Split MNIST:\n")
        f.write("  Fine-tuning: 98.16% accuracy, 1.72% forgetting\n")
        f.write("  EWC (λ=1000): 99.25% accuracy, 0.47% forgetting\n")
        f.write("  SI (λ=1.0): 99.17% accuracy, 0.36% forgetting\n\n")

        f.write("Split CIFAR-10:\n")
        f.write("  Fine-tuning: 73.89% accuracy, 13.31% forgetting\n")
        f.write("  EWC (λ=50): 79.79% accuracy, 10.11% forgetting\n")
        f.write("  SI (λ=1.0): 74.02% accuracy, 4.32% forgetting\n\n")

        f.write("Split CIFAR-100:\n")
        f.write("  Fine-tuning: 42.39% accuracy, 21.96% forgetting\n")
        f.write("  EWC (λ=10): 55.33% accuracy, 15.01% forgetting\n")
        f.write("  SI (λ=1.0): 47.75% accuracy, 3.71% forgetting\n\n")

    print("="*80)
    print("EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print(f"  - aggregated_results.json")
    print(f"  - SUMMARY.txt")
    if args.save_detailed:
        print(f"  - Individual result files for each experiment")
    print()


if __name__ == '__main__':
    main()
