"""
Run sigmoid-based negotiation experiments.
Tests both pure sigmoid negotiation and hybrid (sigmoid-neg + softmax-train).
"""

import argparse
import torch
import torch.optim as optim
import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.networks import SimpleMLP, SimpleConvNet
from utils.data_utils import get_split_mnist, get_split_cifar10, get_split_cifar100
from utils.metrics import ContinualLearningMetrics
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


def main():
    parser = argparse.ArgumentParser(description='Sigmoid Negotiation Experiments')
    parser.add_argument('--dataset', type=str, default='split_mnist',
                       choices=['split_mnist', 'split_cifar10', 'split_cifar100'])
    parser.add_argument('--n_tasks', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Negotiation rate (0.0=pure labels, 1.0=pure model)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variant', type=str, default='sigmoid',
                       choices=['sigmoid', 'hybrid'],
                       help='sigmoid: full sigmoid, hybrid: sigmoid-neg + softmax-train')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to JSON file')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading {args.dataset}...")
    train_loaders, val_loaders, test_loaders = get_dataset(
        dataset_name=args.dataset,
        n_tasks=args.n_tasks,
        batch_size=args.batch_size
    )

    # Model configuration
    if args.dataset == 'split_mnist':
        num_classes_per_task = 2
        model = SimpleMLP(
            input_size=28*28,
            hidden_size=256,
            num_classes_per_task=num_classes_per_task,
            num_tasks=args.n_tasks
        )
    else:  # CIFAR
        num_classes_per_task = 2 if args.dataset == 'split_cifar10' else 10
        model = SimpleConvNet(
            num_classes_per_task=num_classes_per_task,
            num_tasks=args.n_tasks
        )

    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Classes per task: {num_classes_per_task}")

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Create trainer
    TrainerClass = SigmoidNegotiationTrainer if args.variant == 'sigmoid' else HybridNegotiationTrainer

    trainer = TrainerClass(
        model=model,
        optimizer=optimizer,
        device=device,
        num_tasks=args.n_tasks,
        num_classes_per_task=num_classes_per_task,
        alpha=args.alpha,
    )

    print(f"\n{'='*60}")
    print(f"METHOD: {trainer.method_name}")
    print(f"Negotiation rate α = {args.alpha}")
    print(f"{'='*60}")

    # CRITICAL: Prepare negotiated labels BEFORE training
    trainer.prepare_negotiated_labels(train_loaders, batch_size=args.batch_size)

    # Train on all tasks
    metrics = ContinualLearningMetrics(args.n_tasks)
    accuracy_matrix = []

    for task_id in range(args.n_tasks):
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],  # Not used but kept for API
            epochs=args.epochs
        )

        # Evaluate on all tasks seen so far
        print(f"\nEvaluation after Task {task_id}:")
        accuracies = trainer.evaluate_all_tasks(test_loaders, task_id)
        accuracy_matrix.append(accuracies)

        # Update metrics
        metrics.update(task_id, accuracies)

    # Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    final_metrics = metrics.get_metrics()
    print(f"Average Accuracy:  {final_metrics['average_accuracy']:.4f}")
    print(f"Forgetting:        {final_metrics['forgetting']:.4f}")
    print(f"Backward Transfer: {final_metrics['backward_transfer']:.4f}")

    print(f"\nPer-task accuracies:")
    for t, acc in enumerate(final_metrics['task_accuracies']):
        print(f"  Task {t}: {acc:.4f}")

    # Save results
    if args.save_results:
        results = {
            'method': trainer.method_name,
            'dataset': args.dataset,
            'n_tasks': args.n_tasks,
            'alpha': args.alpha,
            'seed': args.seed,
            'epochs': args.epochs,
            'variant': args.variant,
            'final_metrics': {
                'average_accuracy': final_metrics['average_accuracy'],
                'forgetting': final_metrics['forgetting'],
                'backward_transfer': final_metrics['backward_transfer'],
            },
            'task_accuracies': final_metrics['task_accuracies'],
            'accuracy_matrix': accuracy_matrix,
        }

        # Create results directory
        results_dir = os.path.join('results', 'sigmoid_experiments')
        os.makedirs(results_dir, exist_ok=True)

        # Save file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{args.variant}_negotiation_{args.dataset}_alpha{args.alpha}_seed{args.seed}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {filepath}")


if __name__ == '__main__':
    main()
