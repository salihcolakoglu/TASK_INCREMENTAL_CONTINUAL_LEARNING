"""
Sigmoid-based Negotiation method for continual learning.

Key differences from softmax negotiation:
- Uses sigmoid for initial model predictions
- Creates softer, more uniform initial targets
- Uses BCE loss for training with soft targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.activations import ActivationType, get_probabilities, compute_soft_target_loss


class NegotiatedDataset(Dataset):
    """Dataset wrapper that stores negotiated soft labels."""

    def __init__(self, x_data, y_negotiated, y_true, task_id):
        self.x = x_data
        self.y_negotiated = y_negotiated  # Soft targets
        self.y_true = y_true  # Hard labels for accuracy
        self.task_id = task_id

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_negotiated[idx], self.y_true[idx]


class SigmoidNegotiationTrainer:
    """
    Negotiation trainer with sigmoid activation.

    The negotiation creates soft targets by mixing:
    - One-hot true labels
    - Sigmoid predictions from initial (untrained) model

    Benefits of sigmoid for negotiation:
    - Initial predictions are more uniform (~0.5 for random model)
    - No artificial competition between classes
    - Softer gradients during training
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
        num_tasks,
        num_classes_per_task,
        alpha=0.5,  # Negotiation rate
        config=None,
        criterion=None,  # Ignored, we use BCE loss
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.alpha = alpha

        # Storage for pre-computed negotiated loaders
        self.negotiated_loaders = {}
        self.labels_prepared = False

        self.method_name = "Sigmoid-Negotiation"
        self.activation_type = ActivationType.SIGMOID

    def prepare_negotiated_labels(self, train_loaders, batch_size=128):
        """
        Pre-compute negotiated labels for ALL tasks using INITIAL model.

        Must be called ONCE before any training begins.

        Args:
            train_loaders: List of training data loaders
            batch_size: Batch size for negotiated loaders
        """
        print(f"\n{'='*60}")
        print("PRE-COMPUTING NEGOTIATED LABELS (SIGMOID)")
        print(f"Negotiation rate α = {self.alpha}")
        print(f"  → {(1-self.alpha)*100:.0f}% true labels + {self.alpha*100:.0f}% initial predictions")
        print(f"{'='*60}")

        self.model.eval()

        for task_id, train_loader in enumerate(train_loaders):
            print(f"\nTask {task_id}: Computing negotiated labels...")

            # Collect all data for this task
            all_x = []
            all_y = []

            for batch in train_loader:
                # Handle both (data, target) and (data, target, task_id) formats
                if len(batch) == 3:
                    data, target, _ = batch
                else:
                    data, target = batch
                all_x.append(data)
                all_y.append(target)

            all_x = torch.cat(all_x, dim=0)
            all_y = torch.cat(all_y, dim=0)

            print(f"  Collected {len(all_x)} samples")

            # IMPORTANT: Ensure labels are in range [0, num_classes_per_task)
            # They should already be remapped by TaskIncrementalDataset,
            # but we double-check for safety (multiprocessing can cause issues)
            min_label = all_y.min().item()
            max_label = all_y.max().item()
            print(f"  Label range: [{min_label}, {max_label}]")

            if max_label >= self.num_classes_per_task:
                print(f"  ⚠️  WARNING: Labels not properly remapped! Remapping now...")
                # Assume labels are sequential starting from task_id * num_classes_per_task
                class_offset = task_id * self.num_classes_per_task
                all_y = all_y - class_offset
                print(f"  Applied offset: {class_offset}, new range: [{all_y.min().item()}, {all_y.max().item()}]")

            # Get initial model predictions using SIGMOID
            with torch.no_grad():
                all_x_device = all_x.to(self.device)
                logits = self.model(all_x_device, task_id=task_id)

                # SIGMOID predictions (independent, no competition)
                y_pred = torch.sigmoid(logits).cpu()

            # Create one-hot encoding of true labels
            y_onehot = F.one_hot(all_y, self.num_classes_per_task).float()

            # NEGOTIATE: Mix true labels with initial predictions
            y_negotiated = (1 - self.alpha) * y_onehot + self.alpha * y_pred

            # Create dataset and loader
            negotiated_dataset = NegotiatedDataset(
                x_data=all_x,
                y_negotiated=y_negotiated,
                y_true=all_y,
                task_id=task_id
            )

            self.negotiated_loaders[task_id] = DataLoader(
                negotiated_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )

            # Print statistics
            pred_entropy = -(y_pred * torch.log(y_pred + 1e-8)).sum(dim=1).mean()
            print(f"  Initial prediction entropy: {pred_entropy:.4f}")
            print(f"  ✓ Negotiated labels stored for Task {task_id}")

        self.labels_prepared = True
        print(f"\n✓ All negotiated labels prepared!")

    def _train_epoch(self, task_id, epoch):
        """Train one epoch using pre-computed negotiated labels."""
        if task_id not in self.negotiated_loaders:
            raise RuntimeError(f"Negotiated labels for task {task_id} not found! "
                             "Call prepare_negotiated_labels() first.")

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        negotiated_loader = self.negotiated_loaders[task_id]

        for x, y_neg, y_true in negotiated_loader:
            x = x.to(self.device)
            y_neg = y_neg.to(self.device)
            y_true = y_true.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x, task_id=task_id)

            # BCE loss with soft targets
            # y_neg is already in [0, 1] range from sigmoid
            loss = F.binary_cross_entropy_with_logits(logits, y_neg)

            loss.backward()
            self.optimizer.step()

            # Track metrics (accuracy uses hard labels)
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == y_true).sum().item()
            total += y_true.size(0)

        return total_loss / len(negotiated_loader), correct / total

    def train_task(self, task_id, train_loader, val_loader=None, epochs=10):
        """
        Train on a single task using pre-computed negotiated labels.

        Args:
            task_id: Task identifier
            train_loader: Original loader (NOT USED - kept for API compatibility)
            val_loader: Validation loader (optional)
            epochs: Number of training epochs
        """
        if not self.labels_prepared:
            raise RuntimeError("Must call prepare_negotiated_labels() before training!")

        print(f"\n{'='*60}")
        print(f"Training Task {task_id} with {self.method_name}")
        print(f"Using PRE-COMPUTED negotiated labels (α={self.alpha})")
        print(f"{'='*60}")

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(task_id, epoch)

            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

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
                predictions = logits.argmax(dim=1)
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


# Also create a HYBRID version: sigmoid negotiation + softmax training
class HybridNegotiationTrainer(SigmoidNegotiationTrainer):
    """
    Uses sigmoid for negotiation, but softmax for training.

    This tests whether the benefit comes from:
    - Sigmoid during negotiation (softer initial predictions)
    - Sigmoid during training (softer gradients)
    - Or both
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_name = "Hybrid-Negotiation (Sigmoid-neg, Softmax-train)"

    def _train_epoch(self, task_id, epoch):
        """Train with softmax loss but sigmoid-negotiated targets."""
        if task_id not in self.negotiated_loaders:
            raise RuntimeError("Call prepare_negotiated_labels() first!")

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        negotiated_loader = self.negotiated_loaders[task_id]

        for x, y_neg, y_true in negotiated_loader:
            x = x.to(self.device)
            y_neg = y_neg.to(self.device)
            y_true = y_true.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x, task_id=task_id)

            # KL divergence loss (softmax-style)
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(y_neg * log_probs).sum(dim=1).mean()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == y_true).sum().item()
            total += y_true.size(0)

        return total_loss / len(negotiated_loader), correct / total
