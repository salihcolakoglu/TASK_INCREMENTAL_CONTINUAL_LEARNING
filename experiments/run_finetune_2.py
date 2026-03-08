"""
Hyperparameter search for MAS and LwF baselines.

This script performs grid search to find optimal hyperparameters for:
- MAS (Memory Aware Synapses): λ and num_samples
- LwF (Learning without Forgetting): λ and temperature

Based on initial results showing:
- LwF: 90.11% on CIFAR-10, 67.23% on CIFAR-100 (BEST so far!)
- MAS: 82.34% on CIFAR-10, 63.17% on CIFAR-100

Usage:
    # Search MAS hyperparameters on CIFAR-10
    python experiments/run_finetune_2.py --method mas --dataset split_cifar10

    # Search LwF hyperparameters on CIFAR-100
    python experiments/run_finetune_2.py --method lwf --dataset split_cifar100

    # Search both methods on all datasets
    python experiments/run_finetune_2.py --method mas lwf --dataset split_mnist split_cifar10 split_cifar100

    # Custom grid for MAS
    python experiments/run_finetune_2.py --method mas --mas_lambda 0.1 1.0 10.0 --mas_samples 100 200

    # Custom grid for LwF
    python experiments/run_finetune_2.py --method lwf --lwf_lambda 0.5 1.0 5.0 --lwf_temp 2.0 3.0
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
from itertools import product

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.baselines import MASTrainer, LwFTrainer
from src.utils import get_dataset
from configs.config_loader import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hyperparameter search for MAS and LwF'
    )

    # Method and dataset selection
    parser.add_argument('--method', type=str, nargs='+',
                       default=['mas', 'lwf'],
                       choices=['mas', 'lwf'],
                       help='Methods to search')
    parser.add_argument('--dataset', type=str, nargs='+',
                       default=['split_cifar10'],
                       choices=['split_mnist', 'split_cifar10', 'split_cifar100'],
                       help='Datasets to test on')

    # MAS hyperparameter grid
    parser.add_argument('--mas_lambda', type=float, nargs='+',
                       default=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                       help='MAS lambda values to search')
    parser.add_argument('--mas_samples', type=int, nargs='+',
                       default=[100, 200, 500],
                       help='MAS num_samples values to search')

    # LwF hyperparameter grid
    parser.add_argument('--lwf_lambda', type=float, nargs='+',
                       default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
                       help='LwF lambda values to search')
    parser.add_argument('--lwf_temp', type=float, nargs='+',
                       default=[1.0, 2.0, 3.0, 4.0, 5.0],
                       help='LwF temperature values to search')

    # Experimental settings
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='Random seeds')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Epochs per task')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')

    # Output
    parser.add_argument('--results_dir', type=str, default='./results/hyperparam_search_mas_lwf',
                       help='Directory to save results')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_mas_experiment(dataset_name, mas_lambda, num_samples, seed, device, epochs):
    """
    Run a single MAS experiment.

    Args:
        dataset_name: Dataset name
        mas_lambda: MAS regularization strength
        num_samples: Number of samples for importance estimation
        seed: Random seed
        device: Device
        epochs: Epochs per task

    Returns:
        Dictionary with results
    """
    # Load config
    config = load_config(dataset_name)
    dataset_config = config['dataset']
    model_config = config['model']
    training_config = config['training']
    optimizer_config = config['optimizer']

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

    trainer = MASTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_tasks=dataset_config['n_tasks'],
        num_classes_per_task=dataset_config['classes_per_task'],
        config=trainer_config,
        mas_lambda=mas_lambda,
        num_samples=num_samples
    )

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
        'method': 'mas',
        'dataset': dataset_name,
        'seed': seed,
        'mas_lambda': mas_lambda,
        'num_samples': num_samples,
        'avg_accuracy': final_metrics['average_accuracy'],
        'forgetting': final_metrics['forgetting'],
        'backward_transfer': final_metrics['backward_transfer'],
        'accuracy_matrix': trainer.metrics.get_accuracy_matrix().tolist()
    }


def run_lwf_experiment(dataset_name, lwf_lambda, temperature, seed, device, epochs):
    """
    Run a single LwF experiment.

    Args:
        dataset_name: Dataset name
        lwf_lambda: LwF distillation loss weight
        temperature: LwF temperature
        seed: Random seed
        device: Device
        epochs: Epochs per task

    Returns:
        Dictionary with results
    """
    # Load config
    config = load_config(dataset_name)
    dataset_config = config['dataset']
    model_config = config['model']
    training_config = config['training']
    optimizer_config = config['optimizer']

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

    trainer = LwFTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_tasks=dataset_config['n_tasks'],
        num_classes_per_task=dataset_config['classes_per_task'],
        config=trainer_config,
        lwf_lambda=lwf_lambda,
        temperature=temperature
    )

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
        'method': 'lwf',
        'dataset': dataset_name,
        'seed': seed,
        'lwf_lambda': lwf_lambda,
        'temperature': temperature,
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
    print("HYPERPARAMETER SEARCH: MAS & LwF")
    print("="*80)
    print(f"\nMethods: {args.method}")
    print(f"Datasets: {args.dataset}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {device}")
    print(f"Epochs per task: {args.epochs}")
    print(f"Results directory: {results_dir}")

    if 'mas' in args.method:
        print(f"\nMAS grid:")
        print(f"  λ values: {args.mas_lambda}")
        print(f"  num_samples values: {args.mas_samples}")
        print(f"  Total MAS configs: {len(args.mas_lambda) * len(args.mas_samples)}")

    if 'lwf' in args.method:
        print(f"\nLwF grid:")
        print(f"  λ values: {args.lwf_lambda}")
        print(f"  Temperature values: {args.lwf_temp}")
        print(f"  Total LwF configs: {len(args.lwf_lambda) * len(args.lwf_temp)}")

    # Calculate total experiments
    total_experiments = 0
    if 'mas' in args.method:
        total_experiments += len(args.dataset) * len(args.seeds) * len(args.mas_lambda) * len(args.mas_samples)
    if 'lwf' in args.method:
        total_experiments += len(args.dataset) * len(args.seeds) * len(args.lwf_lambda) * len(args.lwf_temp)

    print(f"\nTotal experiments: {total_experiments}")
    print("="*80)
    print()

    # Run all experiments
    all_results = []
    experiment_count = 0

    # MAS experiments
    if 'mas' in args.method:
        for dataset in args.dataset:
            for seed in args.seeds:
                for mas_lambda in args.mas_lambda:
                    for num_samples in args.mas_samples:
                        experiment_count += 1

                        print(f"\n[{experiment_count}/{total_experiments}] MAS:")
                        print(f"  Dataset: {dataset}")
                        print(f"  λ={mas_lambda}, samples={num_samples}, seed={seed}")
                        print("-" * 80)

                        try:
                            result = run_mas_experiment(
                                dataset_name=dataset,
                                mas_lambda=mas_lambda,
                                num_samples=num_samples,
                                seed=seed,
                                device=device,
                                epochs=args.epochs
                            )

                            all_results.append(result)

                            print(f"  ✓ Avg Accuracy: {result['avg_accuracy']:.4f}")
                            print(f"  ✓ Forgetting: {result['forgetting']:.4f}")

                            # Save individual result
                            result_file = results_dir / f"mas_{dataset}_lambda{mas_lambda}_samples{num_samples}_seed{seed}.json"
                            with open(result_file, 'w') as f:
                                json.dump(result, f, indent=2)

                        except Exception as e:
                            print(f"  ✗ FAILED: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            continue

    # LwF experiments
    if 'lwf' in args.method:
        for dataset in args.dataset:
            for seed in args.seeds:
                for lwf_lambda in args.lwf_lambda:
                    for temperature in args.lwf_temp:
                        experiment_count += 1

                        print(f"\n[{experiment_count}/{total_experiments}] LwF:")
                        print(f"  Dataset: {dataset}")
                        print(f"  λ={lwf_lambda}, T={temperature}, seed={seed}")
                        print("-" * 80)

                        try:
                            result = run_lwf_experiment(
                                dataset_name=dataset,
                                lwf_lambda=lwf_lambda,
                                temperature=temperature,
                                seed=seed,
                                device=device,
                                epochs=args.epochs
                            )

                            all_results.append(result)

                            print(f"  ✓ Avg Accuracy: {result['avg_accuracy']:.4f}")
                            print(f"  ✓ Forgetting: {result['forgetting']:.4f}")

                            # Save individual result
                            result_file = results_dir / f"lwf_{dataset}_lambda{lwf_lambda}_temp{temperature}_seed{seed}.json"
                            with open(result_file, 'w') as f:
                                json.dump(result, f, indent=2)

                        except Exception as e:
                            print(f"  ✗ FAILED: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            continue

    # Aggregate and analyze results
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)

    # Save all results
    all_results_file = results_dir / "all_results.json"
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Analyze by method and dataset
    for method in args.method:
        for dataset in args.dataset:
            # Filter results
            filtered = [r for r in all_results
                       if r['method'] == method and r['dataset'] == dataset]

            if not filtered:
                continue

            print(f"\n{'='*80}")
            print(f"{method.upper()} - {dataset.upper()}")
            print(f"{'='*80}\n")

            # Group by hyperparameters
            if method == 'mas':
                # Group by lambda and num_samples
                grouped = {}
                for r in filtered:
                    key = (r['mas_lambda'], r['num_samples'])
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(r)

                # Compute statistics
                table_data = []
                for (mas_lambda, num_samples), results in sorted(grouped.items()):
                    avg_accs = [r['avg_accuracy'] for r in results]
                    forgettings = [r['forgetting'] for r in results]

                    table_data.append([
                        f"λ={mas_lambda}",
                        f"N={num_samples}",
                        f"{np.mean(avg_accs):.4f} ± {np.std(avg_accs):.4f}",
                        f"{np.mean(forgettings):.4f} ± {np.std(forgettings):.4f}",
                        len(results)
                    ])

                headers = ['Lambda', 'Samples', 'Avg Accuracy', 'Forgetting', 'Seeds']
                print(tabulate(table_data, headers=headers, tablefmt='grid'))

                # Find best configuration
                best_acc_idx = np.argmax([float(row[2].split('±')[0]) for row in table_data])
                best_forget_idx = np.argmin([float(row[3].split('±')[0]) for row in table_data])

                print(f"\n⭐ Best accuracy: {table_data[best_acc_idx][0]}, {table_data[best_acc_idx][1]} → {table_data[best_acc_idx][2]}")
                print(f"⭐ Best (lowest) forgetting: {table_data[best_forget_idx][0]}, {table_data[best_forget_idx][1]} → {table_data[best_forget_idx][3]}")

            elif method == 'lwf':
                # Group by lambda and temperature
                grouped = {}
                for r in filtered:
                    key = (r['lwf_lambda'], r['temperature'])
                    if key not in grouped:
                        grouped[key] = []
                    grouped[key].append(r)

                # Compute statistics
                table_data = []
                for (lwf_lambda, temperature), results in sorted(grouped.items()):
                    avg_accs = [r['avg_accuracy'] for r in results]
                    forgettings = [r['forgetting'] for r in results]

                    table_data.append([
                        f"λ={lwf_lambda}",
                        f"T={temperature}",
                        f"{np.mean(avg_accs):.4f} ± {np.std(avg_accs):.4f}",
                        f"{np.mean(forgettings):.4f} ± {np.std(forgettings):.4f}",
                        len(results)
                    ])

                headers = ['Lambda', 'Temperature', 'Avg Accuracy', 'Forgetting', 'Seeds']
                print(tabulate(table_data, headers=headers, tablefmt='grid'))

                # Find best configuration
                best_acc_idx = np.argmax([float(row[2].split('±')[0]) for row in table_data])
                best_forget_idx = np.argmin([float(row[3].split('±')[0]) for row in table_data])

                print(f"\n⭐ Best accuracy: {table_data[best_acc_idx][0]}, {table_data[best_acc_idx][1]} → {table_data[best_acc_idx][2]}")
                print(f"⭐ Best (lowest) forgetting: {table_data[best_forget_idx][0]}, {table_data[best_forget_idx][1]} → {table_data[best_forget_idx][3]}")

    # Create summary report
    summary_file = results_dir / "SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MAS & LwF HYPERPARAMETER SEARCH - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Methods: {args.method}\n")
        f.write(f"Datasets: {args.dataset}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Total experiments: {len(all_results)}\n\n")

        if 'mas' in args.method:
            f.write(f"MAS grid: λ={args.mas_lambda}, samples={args.mas_samples}\n")
        if 'lwf' in args.method:
            f.write(f"LwF grid: λ={args.lwf_lambda}, T={args.lwf_temp}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("BEST CONFIGURATIONS\n")
        f.write("="*80 + "\n\n")

        for method in args.method:
            for dataset in args.dataset:
                filtered = [r for r in all_results
                           if r['method'] == method and r['dataset'] == dataset]

                if not filtered:
                    continue

                f.write(f"\n{method.upper()} - {dataset.upper()}:\n")

                # Find best
                best_acc = max(filtered, key=lambda x: x['avg_accuracy'])
                best_forget = min(filtered, key=lambda x: x['forgetting'])

                if method == 'mas':
                    f.write(f"  Best accuracy: λ={best_acc['mas_lambda']}, samples={best_acc['num_samples']} → {best_acc['avg_accuracy']:.4f}\n")
                    f.write(f"  Best forgetting: λ={best_forget['mas_lambda']}, samples={best_forget['num_samples']} → {best_forget['forgetting']:.4f}\n")
                elif method == 'lwf':
                    f.write(f"  Best accuracy: λ={best_acc['lwf_lambda']}, T={best_acc['temperature']} → {best_acc['avg_accuracy']:.4f}\n")
                    f.write(f"  Best forgetting: λ={best_forget['lwf_lambda']}, T={best_forget['temperature']} → {best_forget['forgetting']:.4f}\n")

    print("\n" + "="*80)
    print("SEARCH COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    print(f"  - all_results.json")
    print(f"  - SUMMARY.txt")
    print(f"  - Individual result files for each configuration")
    print()


if __name__ == '__main__':
    main()
