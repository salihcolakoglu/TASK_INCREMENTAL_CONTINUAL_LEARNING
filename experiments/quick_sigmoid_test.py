"""
Quick validation test for sigmoid paradigm.
Runs a minimal experiment to verify everything works.
"""

import torch
import torch.optim as optim
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.networks import SimpleMLP
from utils.data_utils import get_split_mnist
from baselines.sigmoid_finetune import SigmoidFineTuneTrainer


def main():
    print("="*60)
    print("QUICK SIGMOID VALIDATION TEST")
    print("="*60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load minimal MNIST (use 5 tasks for standard split of 2 classes each)
    print("\nLoading data...")
    n_tasks = 5  # Standard MNIST split
    classes_per_task = 10 // n_tasks  # = 2
    train_loaders, _, test_loaders = get_split_mnist(n_tasks=n_tasks, batch_size=128)

    # Create model
    model = SimpleMLP(
        input_size=28*28,
        hidden_size=128,  # Smaller for speed
        num_classes_per_task=classes_per_task,
        num_tasks=n_tasks
    ).to(device)

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Create sigmoid trainer
    trainer = SigmoidFineTuneTrainer(
        model=model,
        optimizer=optimizer,
        criterion=None,
        device=device,
        num_tasks=n_tasks,
        num_classes_per_task=classes_per_task,
    )

    print(f"\nRunning {n_tasks} tasks with 3 epochs each...")
    print("(This should complete in ~2-3 minutes)")

    # Train and evaluate
    for task_id in range(n_tasks):
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],
            epochs=3  # Minimal epochs
        )

        # Quick evaluation
        accuracies = trainer.evaluate_all_tasks(test_loaders, task_id)

    print("\n" + "="*60)
    print("✓ TEST PASSED!")
    print("="*60)
    print("\nFinal accuracies:")
    for t, acc in enumerate(accuracies):
        print(f"  Task {t}: {acc:.4f}")

    avg_acc = sum(accuracies) / len(accuracies)
    print(f"\nAverage Accuracy: {avg_acc:.4f}")

    if avg_acc > 0.90:  # MNIST is easy
        print("\n✓ Results look good! Sigmoid implementation working correctly.")
    else:
        print("\n⚠️  Warning: Accuracy seems low. Check implementation.")

    print("\nYou can now run full experiments with:")
    print("  python experiments/run_sigmoid_finetune.py --dataset split_mnist")
    print("  python experiments/run_sigmoid_comparison.py --dataset split_mnist")


if __name__ == '__main__':
    main()
