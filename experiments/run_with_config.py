"""
Run experiments using optimal hyperparameters from config files.

This script automatically loads the optimal hyperparameters for each method
and dataset from the configuration files, eliminating the need to manually
specify hyperparameters.

Usage:
    # Run EWC on CIFAR-10 with optimal hyperparameters
    python experiments/run_with_config.py --dataset split_cifar10 --method ewc

    # Run SI on CIFAR-100 with optimal hyperparameters
    python experiments/run_with_config.py --dataset split_cifar100 --method si

    # Run with multiple seeds
    python experiments/run_with_config.py --dataset split_cifar10 --method ewc --seeds 42 43 44
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
from src.baselines import (
    FineTuningTrainer,
    EWCTrainer,
    SynapticIntelligenceTrainer,
    MASTrainer,
    LwFTrainer
)
from src.utils import get_dataset
from configs.config_loader import (
    load_config,
    get_method_config,
    get_training_config,
    get_dataset_config,
    get_model_config
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run CL experiments with optimal hyperparameters from configs'
    )

    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['split_mnist', 'split_cifar10', 'split_cifar100'],
                       help='Dataset to use')
    parser.add_argument('--method', type=str, required=True,
                       choices=['finetune', 'ewc', 'si', 'mas', 'lwf'],
                       help='Method to run')

    # Optional overrides
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Random seeds (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Epochs per task (overrides config)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')

    # Logging
    parser.add_argument('--save_checkpoints', action='store_true',
                       help='Save model checkpoints')
    parser.add_argument('--use_tensorboard', action='store_true',
                       help='Use TensorBoard logging')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_trainer(method, model, optimizer, criterion, device,
                   dataset_config, method_config, experiment_config):
    """
    Create trainer instance based on method.

    Args:
        method: Method name
        model: Neural network model
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device
        dataset_config: Dataset configuration
        method_config: Method-specific configuration
        experiment_config: Experiment configuration

    Returns:
        Trainer instance
    """
    num_tasks = dataset_config['n_tasks']
    classes_per_task = dataset_config['classes_per_task']

    config = {
        'use_tensorboard': experiment_config.get('use_tensorboard', False),
        'use_wandb': experiment_config.get('use_wandb', False),
        'save_checkpoints': experiment_config.get('save_checkpoints', False),
        'log_dir': experiment_config.get('log_dir', './results/tensorboard'),
        'checkpoint_dir': experiment_config.get('checkpoint_dir', './results/checkpoints'),
    }

    if method == 'finetune':
        return FineTuningTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=classes_per_task,
            config=config
        )

    elif method == 'ewc':
        return EWCTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=classes_per_task,
            config=config,
            ewc_lambda=method_config['lambda'],
            mode=method_config['mode'],
            gamma=method_config.get('gamma', 1.0)
        )

    elif method == 'si':
        return SynapticIntelligenceTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=classes_per_task,
            config=config,
            si_lambda=method_config['lambda'],
            si_epsilon=method_config.get('epsilon', 0.001),
            damping=method_config.get('damping', 0.1)
        )

    elif method == 'mas':
        return MASTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=classes_per_task,
            config=config,
            mas_lambda=method_config['lambda'],
            num_samples=method_config.get('num_samples', 200)
        )

    elif method == 'lwf':
        return LwFTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=classes_per_task,
            config=config,
            lwf_lambda=method_config['lambda'],
            temperature=method_config.get('temperature', 2.0)
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def run_experiment(args, seed):
    """Run a single experiment with given seed."""

    # Load configurations
    full_config = load_config(args.dataset)
    dataset_config = full_config['dataset']
    model_config = full_config['model']
    training_config = full_config['training']
    optimizer_config = full_config['optimizer']
    method_config = full_config[args.method]

    # Override with command-line arguments
    if args.epochs is not None:
        training_config['epochs_per_task'] = args.epochs

    # Set random seed
    set_seed(seed)

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*70)
    print(f"{args.method.upper()} - {args.dataset.upper()}")
    print("="*70)
    print(f"\nConfiguration (loaded from configs/{args.dataset}.yaml):")
    print(f"  Dataset: {dataset_config['name']}")
    print(f"  Tasks: {dataset_config['n_tasks']}")
    print(f"  Classes per task: {dataset_config['classes_per_task']}")
    print(f"  Device: {device}")
    print(f"  Seed: {seed}")
    print(f"  Epochs per task: {training_config['epochs_per_task']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {optimizer_config['lr']}")
    print(f"\n  Method hyperparameters:")
    for key, value in method_config.items():
        print(f"    {key}: {value}")
    print()

    # Load dataset
    print("Loading dataset...")
    train_loaders, val_loaders, test_loaders = get_dataset(
        dataset_name=dataset_config['name'],
        n_tasks=dataset_config['n_tasks'],
        data_root=dataset_config['data_root'],
        batch_size=training_config['batch_size'],
        num_workers=training_config['num_workers'],
        validation_split=training_config.get('validation_split', 0.0)
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

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

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

    # Create experiment config
    experiment_config = {
        'use_tensorboard': args.use_tensorboard,
        'use_wandb': args.use_wandb,
        'save_checkpoints': args.save_checkpoints,
        'log_dir': full_config['experiment'].get('log_dir', './results/tensorboard'),
        'checkpoint_dir': full_config['experiment'].get('checkpoint_dir', './results/checkpoints'),
    }

    # Create trainer
    trainer = create_trainer(
        method=args.method,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        dataset_config=dataset_config,
        method_config=method_config,
        experiment_config=experiment_config
    )

    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    # Train on each task
    for task_id in range(dataset_config['n_tasks']):
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],
            val_loader=val_loaders[task_id],
            epochs=training_config['epochs_per_task']
        )

        trainer.evaluate_all_tasks(
            task_dataloaders=test_loaders,
            current_task=task_id
        )

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

    trainer.metrics.print_summary()
    trainer.close()

    return final_metrics


def main():
    """Main entry point."""
    args = parse_args()

    # Load config to get default seeds
    full_config = load_config(args.dataset)

    # Use seeds from config if not provided
    if args.seeds is None:
        seeds = full_config['experiment'].get('seeds', [42])
    else:
        seeds = args.seeds

    print(f"\nRunning {len(seeds)} experiment(s) with seeds: {seeds}\n")

    # Run experiments
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i+1}/{len(seeds)} - Seed {seed}")
        print(f"{'='*70}\n")

        metrics = run_experiment(args, seed)
        all_results.append(metrics)

    # Aggregate results if multiple seeds
    if len(seeds) > 1:
        print("\n" + "="*70)
        print("AGGREGATED RESULTS (Mean ± Std)")
        print("="*70)

        avg_accs = [r['average_accuracy'] for r in all_results]
        forgettings = [r['forgetting'] for r in all_results]
        bwts = [r['backward_transfer'] for r in all_results]

        print(f"\nAverage Accuracy:  {np.mean(avg_accs):.4f} ± {np.std(avg_accs):.4f}")
        print(f"Forgetting:        {np.mean(forgettings):.4f} ± {np.std(forgettings):.4f}")
        print(f"Backward Transfer: {np.mean(bwts):.4f} ± {np.std(bwts):.4f}")

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED")
    print("="*70)


if __name__ == '__main__':
    main()
