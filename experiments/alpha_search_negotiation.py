"""
Alpha Hyperparameter Search for Negotiation Methods.

This script systematically searches for the optimal alpha (negotiation rate)
for both Softmax-Negotiation and Sigmoid-Negotiation across all datasets.

Alpha represents the mixing ratio:
  negotiated_label = (1 - alpha) * true_label + alpha * initial_prediction

Alpha range: 0.0 (pure supervised) to 1.0 (pure initial predictions)
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, './src')

from models.networks import SimpleMLP, SimpleConvNet
from utils.data_utils import get_split_mnist, get_split_cifar10, get_split_cifar100
from utils.metrics import ContinualLearningMetrics
from baselines.negotiation import NegotiationTrainer
from baselines.sigmoid_negotiation import SigmoidNegotiationTrainer


def create_model(dataset_name, num_classes_per_task, num_tasks, device):
    """Create model based on dataset."""
    if dataset_name == 'split_mnist':
        model = SimpleMLP(
            num_classes_per_task=num_classes_per_task,
            num_tasks=num_tasks,
            hidden_size=256
        )
    else:  # CIFAR-10 or CIFAR-100
        model = SimpleConvNet(
            num_classes_per_task=num_classes_per_task,
            num_tasks=num_tasks
        )
    return model.to(device)


def get_data_loaders(dataset_name, num_tasks, batch_size):
    """Get data loaders for specified dataset."""
    if dataset_name == 'split_mnist':
        return get_split_mnist(n_tasks=num_tasks, batch_size=batch_size)
    elif dataset_name == 'split_cifar10':
        return get_split_cifar10(n_tasks=num_tasks, batch_size=batch_size)
    elif dataset_name == 'split_cifar100':
        return get_split_cifar100(n_tasks=num_tasks, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_single_experiment(
    method_name,
    trainer_class,
    alpha,
    model,
    train_loaders,
    test_loaders,
    num_classes_per_task,
    num_tasks,
    epochs,
    device,
    lr=0.01
):
    """Run a single negotiation experiment with specified alpha."""

    print(f"\n{'='*70}")
    print(f"Running: {method_name} with α={alpha}")
    print(f"{'='*70}")

    # Create fresh model
    model_copy = type(model)(
        num_classes_per_task=model.num_classes_per_task,
        num_tasks=model.num_tasks
    ).to(device)

    # Create optimizer
    optimizer = optim.SGD(model_copy.parameters(), lr=lr, momentum=0.9)

    # Create trainer
    if trainer_class == NegotiationTrainer:
        trainer = trainer_class(
            model=model_copy,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=num_classes_per_task,
            initial_negotiation_rate=alpha
        )
    else:  # SigmoidNegotiationTrainer
        trainer = trainer_class(
            model=model_copy,
            optimizer=optimizer,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=num_classes_per_task,
            alpha=alpha
        )

    # Prepare negotiated labels BEFORE training
    print(f"Preparing negotiated labels with α={alpha}...")
    trainer.prepare_negotiated_labels(train_loaders)

    # Initialize metrics tracker
    metrics = ContinualLearningMetrics(num_tasks=num_tasks)

    # Train on all tasks
    for task_id in range(num_tasks):
        print(f"\n{'─'*70}")
        print(f"Task {task_id}/{num_tasks-1}")
        print(f"{'─'*70}")

        # Train
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],
            epochs=epochs
        )

        # Evaluate on all tasks seen so far
        print(f"\nEvaluating after Task {task_id}:")
        task_accuracies = {}
        for t in range(task_id + 1):
            # Use evaluate() if available, otherwise use _eval_single_task()
            if hasattr(trainer, 'evaluate'):
                acc = trainer.evaluate(t, test_loaders[t])
            else:
                result = trainer._eval_single_task(t, test_loaders[t])
                acc = result['accuracy']
            task_accuracies[t] = acc
            print(f"  Task {t}: {acc:.4f}")
        metrics.update(task_id, task_accuracies)

    # Get final metrics
    final_metrics = metrics.get_all_metrics()

    result = {
        'method': method_name,
        'alpha': alpha,
        'avg_accuracy': final_metrics['average_accuracy'],
        'forgetting': final_metrics['forgetting'],
        'backward_transfer': final_metrics['backward_transfer'],
        'task_accuracies': final_metrics['task_accuracies'],
        'accuracy_matrix': [
            {str(j): metrics.accuracy_matrix[task_id][j]
             for j in range(task_id + 1)}
            for task_id in range(num_tasks)
        ]
    }

    print(f"\n{'='*70}")
    print(f"Results for {method_name} (α={alpha}):")
    print(f"  Average Accuracy: {result['avg_accuracy']:.4f}")
    print(f"  Forgetting:       {result['forgetting']:.4f}")
    print(f"  Backward Transfer: {result['backward_transfer']:.4f}")
    print(f"{'='*70}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Alpha Hyperparameter Search for Negotiation Methods'
    )
    parser.add_argument('--dataset', type=str, default='split_cifar100',
                       choices=['split_mnist', 'split_cifar10', 'split_cifar100'],
                       help='Dataset to use')
    parser.add_argument('--n_tasks', type=int, default=None,
                       help='Number of tasks (auto-set if not specified)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Epochs per task')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--alpha_values', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                       help='Comma-separated alpha values to test')
    parser.add_argument('--variants', type=str, default='softmax,sigmoid',
                       choices=['softmax', 'sigmoid', 'both'],
                       help='Which variants to test')
    parser.add_argument('--save_dir', type=str, default='./results/alpha_search',
                       help='Directory to save results')
    args = parser.parse_args()

    # Parse alpha values
    alpha_values = [float(a) for a in args.alpha_values.split(',')]

    # Setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"ALPHA HYPERPARAMETER SEARCH FOR NEGOTIATION METHODS")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Alpha values: {alpha_values}")
    print(f"Variants: {args.variants}")
    print(f"Epochs per task: {args.epochs}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")

    # Determine dataset configuration
    if args.dataset == 'split_mnist':
        num_tasks = args.n_tasks or 5
        num_classes_per_task = 2
    elif args.dataset == 'split_cifar10':
        num_tasks = args.n_tasks or 5
        num_classes_per_task = 2
    else:  # split_cifar100
        num_tasks = args.n_tasks or 10
        num_classes_per_task = 10

    # Load data
    print(f"Loading {args.dataset} dataset...")
    train_loaders, val_loaders, test_loaders = get_data_loaders(
        args.dataset, num_tasks, args.batch_size
    )
    print(f"✓ Loaded {num_tasks} tasks with {num_classes_per_task} classes per task\n")

    # Create base model (for structure reference)
    base_model = create_model(
        args.dataset, num_classes_per_task, num_tasks, device
    )

    # Determine which variants to test
    variants_to_test = []
    if args.variants in ['softmax', 'both']:
        variants_to_test.append(('Softmax-Negotiation', NegotiationTrainer))
    if args.variants in ['sigmoid', 'both']:
        variants_to_test.append(('Sigmoid-Negotiation', SigmoidNegotiationTrainer))

    # Run experiments for all alpha values and variants
    all_results = []

    for variant_name, trainer_class in variants_to_test:
        print(f"\n{'#'*70}")
        print(f"Testing {variant_name}")
        print(f"{'#'*70}\n")

        for alpha in alpha_values:
            try:
                result = run_single_experiment(
                    method_name=variant_name,
                    trainer_class=trainer_class,
                    alpha=alpha,
                    model=base_model,
                    train_loaders=train_loaders,
                    test_loaders=test_loaders,
                    num_classes_per_task=num_classes_per_task,
                    num_tasks=num_tasks,
                    epochs=args.epochs,
                    device=device,
                    lr=args.lr
                )
                all_results.append(result)

            except Exception as e:
                print(f"\n⚠️  ERROR with {variant_name} α={alpha}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Print summary table
    print(f"\n\n{'='*70}")
    print(f"ALPHA SEARCH SUMMARY - {args.dataset.upper()}")
    print(f"{'='*70}")

    for variant_name, _ in variants_to_test:
        variant_results = [r for r in all_results if r['method'] == variant_name]
        if not variant_results:
            continue

        print(f"\n{variant_name}:")
        print(f"{'Alpha':>8} {'Accuracy':>12} {'Forgetting':>12} {'BWT':>12}")
        print(f"{'-'*50}")

        for r in sorted(variant_results, key=lambda x: x['alpha']):
            print(f"{r['alpha']:>8.1f} {r['avg_accuracy']:>12.4f} "
                  f"{r['forgetting']:>12.4f} {r['backward_transfer']:>12.4f}")

        # Find best alpha
        best_acc = max(variant_results, key=lambda x: x['avg_accuracy'])
        best_forg = min(variant_results, key=lambda x: x['forgetting'])

        print(f"\n  Best accuracy:  α={best_acc['alpha']:.1f} → {best_acc['avg_accuracy']:.4f}")
        print(f"  Best forgetting: α={best_forg['alpha']:.1f} → {best_forg['forgetting']:.4f}")

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = save_dir / f"alpha_search_{args.dataset}_seed{args.seed}_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'alpha_values': alpha_values,
            'num_tasks': num_tasks,
            'num_classes_per_task': num_classes_per_task,
            'results': all_results
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
