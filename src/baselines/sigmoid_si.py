"""
Sigmoid-based SI (Synaptic Intelligence) for continual learning.
Combines sigmoid activation with online importance estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.activations import ActivationType, compute_loss, get_predictions


class SigmoidSITrainer:
    """
    SI trainer with sigmoid activation.

    Key features:
    - Sigmoid activation with BCE loss
    - Online importance tracking via path integral
    - Parameter-wise importance based on contribution to loss
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        num_tasks,
        num_classes_per_task,
        si_lambda=1.0,
        epsilon=0.001,
        damping=0.1,
        config=None,
        criterion=None,  # Ignored, we use BCE loss
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task

        # SI hyperparameters
        self.si_lambda = si_lambda
        self.epsilon = epsilon  # Small constant for numerical stability
        self.damping = damping

        # Importance tracking
        self.omega = {}  # Accumulated importance
        self.prev_params = {}  # Parameters at start of task
        self.W = {}  # Running sum of (gradient * delta)

        # Initialize
        self._init_si_buffers()

        self.method_name = "Sigmoid-SI"
        self.activation_type = ActivationType.SIGMOID

    def _init_si_buffers(self):
        """Initialize SI tracking buffers."""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.omega[n] = torch.zeros_like(p)
                self.prev_params[n] = p.clone().detach()
                self.W[n] = torch.zeros_like(p)

    def _si_loss(self):
        """Compute SI regularization loss."""
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.omega and p.requires_grad:
                # Normalized importance
                importance = self.omega[n] / (self.omega[n] + self.damping)
                loss += (importance * (p - self.prev_params[n]).pow(2)).sum()
        return loss

    def _update_W(self, task_id, data, target):
        """Update running importance estimate."""
        self.model.zero_grad()
        logits = self.model(data, task_id=task_id)
        loss = compute_loss(logits, target, self.num_classes_per_task, ActivationType.SIGMOID)
        loss.backward()

        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                # W accumulates gradient * parameter change
                delta = p.detach() - self.prev_params[n]
                self.W[n] += (-p.grad.detach() * delta)

    def _consolidate_importance(self):
        """After task: consolidate importance into omega."""
        for n, p in self.model.named_parameters():
            if n in self.W and p.requires_grad:
                delta = (p.detach() - self.prev_params[n]).pow(2) + self.epsilon
                self.omega[n] += self.W[n] / delta

                # Reset for next task
                self.W[n] = torch.zeros_like(p)
                self.prev_params[n] = p.clone().detach()

    def _train_epoch(self, task_id, train_loader, epoch):
        """Train one epoch with sigmoid + SI."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            # Handle both (data, target) and (data, target, task_id) formats
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch
            data, target = data.to(self.device), target.to(self.device)

            # Update importance BEFORE optimizer step
            self._update_W(task_id, data, target)

            self.optimizer.zero_grad()
            logits = self.model(data, task_id=task_id)

            # Task loss
            task_loss = compute_loss(
                logits, target, self.num_classes_per_task,
                ActivationType.SIGMOID
            )

            # SI regularization
            si_loss = self._si_loss()

            # Total loss
            loss = task_loss + (self.si_lambda / 2.0) * si_loss

            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += task_loss.item()
            predictions = get_predictions(logits, ActivationType.SIGMOID)
            correct += (predictions == target).sum().item()
            total += target.size(0)

        return total_loss / len(train_loader), correct / total

    def train_task(self, task_id, train_loader, val_loader=None, epochs=10):
        """Train on a single task."""
        print(f"\n{'='*60}")
        print(f"Training Task {task_id} with {self.method_name}")
        print(f"Activation: SIGMOID | SI λ={self.si_lambda}")
        print(f"{'='*60}")

        # Store params at start of task
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.prev_params[n] = p.clone().detach()
                self.W[n] = torch.zeros_like(p)

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(task_id, train_loader, epoch)

            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

        # Consolidate importance after task
        print(f"  Consolidating importance...")
        self._consolidate_importance()

        print(f"✓ Completed Task {task_id}")

    def evaluate(self, task_id, test_loader):
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

    def evaluate_all_tasks(self, test_loaders, current_task):
        """Evaluate on all tasks seen so far."""
        accuracies = {}
        for t in range(current_task + 1):
            acc = self.evaluate(t, test_loaders[t])
            accuracies[t] = acc
            print(f"  Task {t} Accuracy: {acc:.4f}")
        return accuracies
