"""
Test script to verify the base trainer class works correctly.
This is a quick sanity check before implementing baselines.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.insert(0, './src')

from models import get_model
from utils import BaseTrainer, get_dataset


def test_base_trainer():
    """Test the base trainer on Split MNIST with 2 tasks."""

    print("="*60)
    print("Testing Base Trainer")
    print("="*60)

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_tasks = 2  # Just 2 tasks for quick test
    num_classes_per_task = 5  # 5 classes per task (10 classes total: MNIST digits 0-9)

    print(f"\nDevice: {device}")
    print(f"Number of tasks: {n_tasks}")
    print(f"Classes per task: {num_classes_per_task}")

    # Create model
    model = get_model(
        architecture="mlp",
        num_classes_per_task=num_classes_per_task,
        num_tasks=n_tasks,
        hidden_size=128,
        dropout=0.0
    )

    # Create optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Load dataset
    print("\nLoading Split MNIST dataset...")
    train_loaders, val_loaders, test_loaders = get_dataset(
        dataset_name="split_mnist",
        n_tasks=n_tasks,
        batch_size=32,
        num_workers=0,  # 0 for testing to avoid multiprocessing issues
        validation_split=0.0
    )

    print(f"Train loader 0 batches: {len(train_loaders[0])}")
    print(f"Test loader 0 batches: {len(test_loaders[0])}")

    # Create trainer
    config = {
        "use_tensorboard": False,  # Disable for quick test
        "use_wandb": False,
        "save_checkpoints": False,
    }

    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_tasks=n_tasks,
        num_classes_per_task=num_classes_per_task,
        config=config
    )

    print("\n" + "="*60)
    print("Starting Continual Learning Training")
    print("="*60)

    # Train on each task sequentially
    for task_id in range(n_tasks):
        # Train on current task
        trainer.train_task(
            task_id=task_id,
            train_loader=train_loaders[task_id],
            epochs=2  # Just 2 epochs for quick test
        )

        # Evaluate on all tasks seen so far
        accuracies = trainer.evaluate_all_tasks(
            task_dataloaders=test_loaders,
            current_task=task_id
        )

    # Get final metrics
    print("\n" + "="*60)
    print("Final Results")
    print("="*60)

    final_metrics = trainer.metrics.get_all_metrics()
    print(f"\nAverage Accuracy: {final_metrics['average_accuracy']:.4f}")
    print(f"Forgetting:       {final_metrics['forgetting']:.4f}")
    print(f"Backward Transfer: {final_metrics['backward_transfer']:.4f}")

    # Print accuracy matrix
    print("\nAccuracy Matrix:")
    trainer.metrics.print_summary()

    trainer.close()

    print("\n" + "="*60)
    print("Base Trainer Test Completed Successfully!")
    print("="*60)

    return True


if __name__ == "__main__":
    try:
        test_base_trainer()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
