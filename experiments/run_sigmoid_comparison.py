"""
Main comparison script for Sigmoid vs Softmax experiments.
Compares softmax and sigmoid variants of baselines and negotiation.
"""

import argparse
import torch
import torch.optim as optim
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.networks import SimpleMLP, SimpleConvNet
from utils.data_utils import get_split_mnist, get_split_cifar10, get_split_cifar100
from utils.metrics import ContinualLearningMetrics

# Import trainers
from baselines.finetune import FineTuningTrainer
from baselines.ewc import EWCTrainer
from baselines.si import SynapticIntelligenceTrainer
from baselines.negotiation import NegotiationTrainer
from baselines.sigmoid_finetune import SigmoidFineTuneTrainer
from baselines.sigmoid_ewc import SigmoidEWCTrainer
from baselines.sigmoid_si import SigmoidSITrainer
from baselines.sigmoid_negotiation import SigmoidNegotiationTrainer, HybridNegotiationTrainer


def get_dataset(dataset_name, n_tasks, batch_size):
    """Load dataset."""
    if dataset_name == 'split_mnist':
        return get_split_mnist(n_tasks=n_tasks, batch_size=batch_size)
    elif dataset_name == 'split_cifar10':
        return get_split_cifar10(n_tasks=n_tasks, batch_size=batch_size)
    elif dataset_name == 'split_cifar100':
        return get_split_cifar100(n_tasks=n_tasks, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_model(dataset, num_classes_per_task, n_tasks, device):
    """Create fresh model for each experiment."""
    if dataset == 'split_mnist':
        model = SimpleMLP(
            input_size=28*28,
            hidden_size=256,
            num_classes_per_task=num_classes_per_task,
            num_tasks=n_tasks
        )
    else:  # CIFAR
        model = SimpleConvNet(
            num_classes_per_task=num_classes_per_task,
            num_tasks=n_tasks
        )
    return model.to(device)


def run_experiment(
    method_name,
    trainer_class,
    trainer_kwargs,
    dataset,
    num_classes_per_task,
    n_tasks,
    train_loaders,
    test_loaders,
    epochs,
    lr,
    device,
    is_negotiation=False
):
    """Run a single experiment and return results."""
    print(f"\n{'#'*60}")
    print(f"Running: {method_name}")
    print(f"{'#'*60}")

    # Create fresh model
    model = create_model(dataset, num_classes_per_task, n_tasks, device)

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Create trainer
    trainer = trainer_class(
        model=model,
        optimizer=optimizer,
        device=device,
        **trainer_kwargs
    )

    # Special handling for negotiation methods
    if is_negotiation:
        trainer.prepare_negotiated_labels(train_loaders)

    # Train on all tasks
    metrics = ContinualLearningMetrics(n_tasks)
    accuracy_matrix = []

    for task_id in range(n_tasks):
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],
            epochs=epochs
        )

        # Evaluate
        accuracies = trainer.evaluate_all_tasks(test_loaders, task_id)
        accuracy_matrix.append(accuracies)
        metrics.update(task_id, accuracies)

    # Get final metrics
    final_metrics = metrics.get_metrics()

    result = {
        'method': method_name,
        'avg_accuracy': final_metrics['average_accuracy'],
        'forgetting': final_metrics['forgetting'],
        'backward_transfer': final_metrics['backward_transfer'],
        'task_accuracies': final_metrics['task_accuracies'],
        'accuracy_matrix': accuracy_matrix,
    }

    print(f"\n{method_name} Results:")
    print(f"  Avg Accuracy: {result['avg_accuracy']:.4f}")
    print(f"  Forgetting:   {result['forgetting']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Sigmoid vs Softmax Comparison')
    parser.add_argument('--dataset', type=str, default='split_mnist',
                       choices=['split_mnist', 'split_cifar10', 'split_cifar100'])
    parser.add_argument('--n_tasks', type=int, default=None,
                       help='Number of tasks (default: 5 for MNIST/CIFAR10, 10 for CIFAR100)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['finetune', 'ewc', 'si', 'negotiation'],
                       choices=['finetune', 'ewc', 'si', 'negotiation', 'all'])
    parser.add_argument('--save_dir', type=str, default='./results/sigmoid_experiments')
    args = parser.parse_args()

    # Set default n_tasks based on dataset
    if args.n_tasks is None:
        args.n_tasks = 10 if args.dataset == 'split_cifar100' else 5

    # Handle 'all' methods
    if 'all' in args.methods:
        args.methods = ['finetune', 'ewc', 'si', 'negotiation']

    # Setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Tasks: {args.n_tasks}")

    # Load data
    print(f"\nLoading data...")
    train_loaders, val_loaders, test_loaders = get_dataset(
        dataset_name=args.dataset,
        n_tasks=args.n_tasks,
        batch_size=args.batch_size
    )

    # Determine classes per task
    if args.dataset == 'split_mnist':
        num_classes_per_task = 2
    elif args.dataset == 'split_cifar10':
        num_classes_per_task = 2
    else:  # cifar100
        num_classes_per_task = 10

    print(f"Classes per task: {num_classes_per_task}")

    # Define hyperparameters based on dataset (from your optimal configs)
    if args.dataset == 'split_mnist':
        ewc_lambda = 1000
        si_lambda = 1.0
    elif args.dataset == 'split_cifar10':
        ewc_lambda = 50
        si_lambda = 1.0
    else:  # cifar100
        ewc_lambda = 10
        si_lambda = 1.0

    # Common kwargs
    base_kwargs = {
        'num_tasks': args.n_tasks,
        'num_classes_per_task': num_classes_per_task,
        'criterion': torch.nn.CrossEntropyLoss(),
    }

    # Define all experiments to run
    experiments = []

    if 'finetune' in args.methods:
        experiments.extend([
            ('Softmax-FineTune', FineTuningTrainer, {**base_kwargs}, False),
            ('Sigmoid-FineTune', SigmoidFineTuneTrainer, {**base_kwargs}, False),
        ])

    if 'ewc' in args.methods:
        experiments.extend([
            ('Softmax-EWC', EWCTrainer, {**base_kwargs, 'ewc_lambda': ewc_lambda}, False),
            ('Sigmoid-EWC', SigmoidEWCTrainer, {**base_kwargs, 'ewc_lambda': ewc_lambda}, False),
        ])

    if 'si' in args.methods:
        experiments.extend([
            ('Softmax-SI', SynapticIntelligenceTrainer, {**base_kwargs, 'si_lambda': si_lambda}, False),
            ('Sigmoid-SI', SigmoidSITrainer, {**base_kwargs, 'si_lambda': si_lambda}, False),
        ])

    if 'negotiation' in args.methods:
        experiments.extend([
            ('Softmax-Negotiation', NegotiationTrainer, {**base_kwargs, 'initial_negotiation_rate': 0.5}, True),
            ('Sigmoid-Negotiation', SigmoidNegotiationTrainer, {**base_kwargs, 'alpha': 0.5}, True),
            ('Hybrid-Negotiation', HybridNegotiationTrainer, {**base_kwargs, 'alpha': 0.5}, True),
        ])

    # Run all experiments
    results = []
    for method_name, trainer_class, trainer_kwargs, is_negotiation in experiments:
        try:
            result = run_experiment(
                method_name=method_name,
                trainer_class=trainer_class,
                trainer_kwargs=trainer_kwargs,
                dataset=args.dataset,
                num_classes_per_task=num_classes_per_task,
                n_tasks=args.n_tasks,
                train_loaders=train_loaders,
                test_loaders=test_loaders,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                is_negotiation=is_negotiation
            )
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  ERROR running {method_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'Accuracy':>12} {'Forgetting':>12} {'BWT':>12}")
    print(f"{'-'*70}")

    for r in sorted(results, key=lambda x: x['avg_accuracy'], reverse=True):
        print(f"{r['method']:<30} {r['avg_accuracy']:>12.4f} "
              f"{r['forgetting']:>12.4f} {r['backward_transfer']:>12.4f}")

    # Highlight improvements
    print(f"\n{'='*70}")
    print("SIGMOID vs SOFTMAX IMPROVEMENTS")
    print(f"{'='*70}")

    softmax_results = {r['method'].replace('Softmax-', ''): r for r in results if 'Softmax-' in r['method']}
    sigmoid_results = {r['method'].replace('Sigmoid-', ''): r for r in results if 'Sigmoid-' in r['method']}

    for method in softmax_results:
        if method in sigmoid_results:
            soft = softmax_results[method]
            sig = sigmoid_results[method]

            acc_diff = sig['avg_accuracy'] - soft['avg_accuracy']
            forg_diff = sig['forgetting'] - soft['forgetting']

            print(f"\n{method}:")
            print(f"  Accuracy:  {soft['avg_accuracy']:.4f} → {sig['avg_accuracy']:.4f} "
                  f"({acc_diff:+.4f}, {acc_diff/soft['avg_accuracy']*100:+.1f}%)")
            print(f"  Forgetting: {soft['forgetting']:.4f} → {sig['forgetting']:.4f} "
                  f"({forg_diff:+.4f}, {forg_diff/soft['forgetting']*100 if soft['forgetting'] > 0 else 0:+.1f}%)")

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = save_dir / f"{args.dataset}_comparison_seed{args.seed}_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': results
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")


if __name__ == '__main__':
    main()
