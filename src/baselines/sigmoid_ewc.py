"""
Sigmoid-based EWC (Elastic Weight Consolidation) for continual learning.
Combines sigmoid activation with Fisher Information regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils.activations import ActivationType, compute_loss, get_predictions


class SigmoidEWCTrainer:
    """
    EWC trainer with sigmoid activation.

    Combines:
    - Sigmoid activation (independent class outputs)
    - Binary cross-entropy loss
    - Fisher Information Matrix regularization
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        num_tasks,
        num_classes_per_task,
        ewc_lambda=100.0,
        fisher_sample_size=200,
        online=True,
        gamma=1.0,
        config=None,
        criterion=None,  # Ignored, we use BCE loss
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task

        # EWC hyperparameters
        self.ewc_lambda = ewc_lambda
        self.fisher_sample_size = fisher_sample_size
        self.online = online
        self.gamma = gamma  # Decay factor for online EWC

        # Storage for Fisher and parameters
        self.fisher_dict = {}
        self.params_dict = {}

        self.method_name = "Sigmoid-EWC"
        self.activation_type = ActivationType.SIGMOID

    def _compute_fisher(self, task_id, train_loader):
        """
        Compute Fisher Information Matrix for current task.

        NOTE: With sigmoid, we compute Fisher using BCE loss gradients.
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        # Sample subset of data
        n_samples = 0
        for batch in train_loader:
            if n_samples >= self.fisher_sample_size:
                break

            # Handle both (data, target) and (data, target, task_id) formats
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)

            self.model.zero_grad()
            logits = self.model(data, task_id=task_id)

            # Compute loss with sigmoid
            loss = compute_loss(
                logits, target, self.num_classes_per_task,
                ActivationType.SIGMOID
            )
            loss.backward()

            # Accumulate squared gradients
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * batch_size

            n_samples += batch_size

        # Normalize
        for n in fisher:
            fisher[n] /= n_samples

        return fisher

    def _store_params(self):
        """Store current parameters as reference."""
        return {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}

    def _ewc_loss(self):
        """Compute EWC regularization loss."""
        if not self.fisher_dict:
            return 0.0

        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher_dict and p.requires_grad:
                loss += (self.fisher_dict[n] * (p - self.params_dict[n]).pow(2)).sum()

        return loss

    def _train_epoch(self, task_id, train_loader, epoch):
        """Train one epoch with sigmoid loss + EWC regularization."""
        self.model.train()
        total_loss = 0.0
        total_ewc_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            # Handle both (data, target) and (data, target, task_id) formats
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(data, task_id=task_id)

            # Task loss (sigmoid BCE)
            task_loss = compute_loss(
                logits, target, self.num_classes_per_task,
                ActivationType.SIGMOID
            )

            # EWC regularization loss
            ewc_loss = self._ewc_loss()

            # Total loss
            loss = task_loss + (self.ewc_lambda / 2.0) * ewc_loss

            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += task_loss.item()
            if isinstance(ewc_loss, float):
                total_ewc_loss += ewc_loss
            else:
                total_ewc_loss += ewc_loss.item()

            predictions = get_predictions(logits, ActivationType.SIGMOID)
            correct += (predictions == target).sum().item()
            total += target.size(0)

        return total_loss / len(train_loader), correct / total

    def train_task(self, task_id, train_loader, val_loader=None, epochs=10):
        """Train on a single task."""
        print(f"\n{'='*60}")
        print(f"Training Task {task_id} with {self.method_name}")
        print(f"Activation: SIGMOID | EWC λ={self.ewc_lambda}")
        print(f"{'='*60}")

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(task_id, train_loader, epoch)

            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

        # After training: compute Fisher and store parameters
        print(f"  Computing Fisher Information...")
        new_fisher = self._compute_fisher(task_id, train_loader)

        if self.online and self.fisher_dict:
            # Online EWC: accumulate Fisher with decay
            for n in new_fisher:
                self.fisher_dict[n] = self.gamma * self.fisher_dict[n] + new_fisher[n]
        else:
            self.fisher_dict = new_fisher

        self.params_dict = self._store_params()

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
