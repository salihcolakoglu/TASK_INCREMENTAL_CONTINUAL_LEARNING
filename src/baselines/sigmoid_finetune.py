"""
Sigmoid-based Fine-tuning baseline for continual learning.
Uses binary cross-entropy loss instead of softmax cross-entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.base_trainer import BaseTrainer
from utils.activations import ActivationType, compute_loss, get_predictions


class SigmoidFineTuneTrainer(BaseTrainer):
    """
    Fine-tuning trainer with sigmoid activation.

    Key differences from standard fine-tuning:
    - Uses sigmoid activation instead of softmax
    - Uses binary cross-entropy loss instead of cross-entropy
    - Each class output is independent (no competition)
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,  # Ignored, we use our own loss
        device,
        num_tasks,
        num_classes_per_task,
        config=None,
    ):
        # Initialize parent with sigmoid activation
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_tasks=num_tasks,
            num_classes_per_task=num_classes_per_task,
            config=config,
            activation_type=ActivationType.SIGMOID,
        )

        self.method_name = "Sigmoid-FineTune"

    def _compute_task_loss(self, logits, labels, task_id):
        """
        Compute sigmoid-based loss for current task.

        Args:
            logits: Model outputs [batch_size, num_classes_per_task]
            labels: Class labels [batch_size] (0-indexed within task)
            task_id: Current task ID

        Returns:
            Scalar loss value
        """
        return compute_loss(
            logits=logits,
            labels=labels,
            num_classes=self.num_classes_per_task,
            activation_type=ActivationType.SIGMOID,
        )

    def _train_epoch(self, task_id, train_loader, epoch):
        """Train one epoch with sigmoid loss."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # Handle both (data, target) and (data, target, task_id) formats
            if len(batch) == 3:
                data, target, _ = batch  # Ignore task_id from loader
            else:
                data, target = batch
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(data, task_id=task_id)

            # Compute sigmoid loss
            loss = self._compute_task_loss(logits, target, task_id)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = get_predictions(logits, ActivationType.SIGMOID)
            correct += (predictions == target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _evaluate_task(self, task_id, test_loader):
        """Evaluate on a specific task."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                # Handle both (data, target) and (data, target, task_id) formats
                if len(batch) == 3:
                    data, target, _ = batch
                else:
                    data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                logits = self.model(data, task_id=task_id)
                predictions = get_predictions(logits, ActivationType.SIGMOID)
                correct += (predictions == target).sum().item()
                total += target.size(0)

        return correct / total

    def train_task(self, task_id, train_loader, val_loader=None, epochs=10):
        """
        Train on a single task using sigmoid loss.

        Args:
            task_id: Task identifier
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
        """
        print(f"\n{'='*60}")
        print(f"Training Task {task_id} with {self.method_name}")
        print(f"Activation: SIGMOID (independent class outputs)")
        print(f"{'='*60}")

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(task_id, train_loader, epoch)

            # Validation (optional)
            val_acc = None
            if val_loader is not None:
                val_acc = self._evaluate_task(task_id, val_loader)

            # Print progress
            if (epoch + 1) % 2 == 0 or epoch == 0:
                val_str = f", Val Acc: {val_acc:.4f}" if val_acc else ""
                print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, "
                      f"Train Acc={train_acc:.4f}{val_str}")

        # After-task hooks (for subclasses)
        self._after_task(task_id, train_loader)

        print(f"✓ Completed Task {task_id}")

    def _after_task(self, task_id, train_loader):
        """Hook for subclasses to add post-task processing."""
        pass

    def evaluate(self, task_id, test_loader):
        """Evaluate on a specific task."""
        return self._evaluate_task(task_id, test_loader)

    def evaluate_all_tasks(self, test_loaders, current_task):
        """Evaluate on all tasks seen so far."""
        accuracies = {}
        for t in range(current_task + 1):
            acc = self.evaluate(t, test_loaders[t])
            accuracies[t] = acc
            print(f"  Task {t} Accuracy: {acc:.4f}")
        return accuracies


# For convenience, also create a factory function
def create_sigmoid_finetune_trainer(
    model,
    device,
    num_tasks,
    num_classes_per_task,
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0,
    config=None,
):
    """Factory function to create SigmoidFineTuneTrainer with optimizer."""
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    return SigmoidFineTuneTrainer(
        model=model,
        optimizer=optimizer,
        criterion=None,  # Not used
        device=device,
        num_tasks=num_tasks,
        num_classes_per_task=num_classes_per_task,
        config=config,
    )
