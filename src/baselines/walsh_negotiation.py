"""
Walsh Negotiation method for task-incremental continual learning.

This method combines:
1. Walsh-Hadamard code-based representations (from class-incremental setting)
2. Negotiated targets for reducing forgetting
3. Task-specific classifier heads (2 neurons per task for task-incremental)

The key innovation:
- Use Walsh codes to create stable representations in the feature space
- Train on negotiated targets (mix of true labels and initial predictions)
- Use separate classifier heads per task (like fine-tuning)

Algorithm:
1. Initialize Walsh codebook and representation tracker
2. For each task:
   a. Assign Walsh codes to classes based on initial model predictions
   b. Create negotiated targets mixing Walsh codes with initial predictions
   c. Train on negotiated targets using BCE loss
   d. Update negotiation rate using plasticity formula
3. At inference: Use task-specific heads for classification

Reference:
    Adapted from class-incremental Walsh negotiation to task-incremental setting.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.base_trainer import BaseTrainer


def create_walsh_codebook(size: int) -> torch.Tensor:
    """
    Create Walsh-Hadamard codebook with binary {0, 1} values.

    Args:
        size: Dimension of the Walsh codes

    Returns:
        Walsh codebook tensor of shape [size, size]
    """
    n = int(np.log2(size)) + 1
    W = np.array([[1]], dtype=np.float32)
    for _ in range(n):
        W = np.block([[W, W], [W, 1 - W]])
    return torch.from_numpy(W[:size, :size])


def optimal_plasticity_rate(neg_rate: float) -> float:
    """
    Calculate optimal plasticity multiplier for equal capacity allocation.

    Args:
        neg_rate: Current negotiation rate

    Returns:
        Plasticity multiplier
    """
    if neg_rate <= 0 or neg_rate >= 1:
        return 1.0
    return 1.0 / (2.0 * neg_rate - neg_rate ** 2)


def bce_distance(prediction: torch.Tensor, walsh_code: torch.Tensor, eps: float = 1e-7) -> float:
    """
    Compute BCE distance between prediction and Walsh code.

    Args:
        prediction: Model predictions (sigmoid outputs)
        walsh_code: Target Walsh code
        eps: Small constant for numerical stability

    Returns:
        BCE distance (scalar)
    """
    p = prediction.clamp(eps, 1.0 - eps)
    bce = -(walsh_code * torch.log(p) + (1 - walsh_code) * torch.log(1 - p))
    return bce.mean().item()


class RepresentationTracker:
    """
    Tracks Walsh code assignments to classes.

    This maintains the mapping from class labels to Walsh codes and handles
    the assignment of codes to new classes based on model predictions.
    """

    def __init__(self, codebook: torch.Tensor):
        """
        Args:
            codebook: Walsh codebook tensor
        """
        self.codebook = codebook.clone()
        self.num_codes = codebook.size(0)
        self.code_dim = codebook.size(1)
        self.class_to_code_idx: Dict[int, int] = {}
        self.available: List[bool] = [True] * self.num_codes
        self.assigned_codes: List[torch.Tensor] = []

    @torch.no_grad()
    def assign_code(self, class_label: int, model: nn.Module,
                    class_samples: torch.Tensor, device: torch.device,
                    verbose: bool = True) -> int:
        """
        Assign closest available Walsh code to a class.

        Args:
            class_label: The class to assign a code to
            model: Neural network model
            class_samples: Samples from the class
            device: Torch device
            verbose: Whether to print assignment info

        Returns:
            Index of assigned Walsh code
        """
        if class_label in self.class_to_code_idx:
            return self.class_to_code_idx[class_label]

        model.eval()
        logits = model.get_walsh_features(class_samples.to(device))
        predictions = torch.sigmoid(logits)
        mean_pred = predictions.mean(dim=0).cpu()

        best_idx, best_dist = None, float('inf')
        for idx in range(self.num_codes):
            if not self.available[idx]:
                continue
            dist = bce_distance(mean_pred, self.codebook[idx])
            if dist < best_dist:
                best_dist, best_idx = dist, idx

        if best_idx is None:
            raise RuntimeError(f"No available codes for class {class_label}")

        self.class_to_code_idx[class_label] = best_idx
        self.available[best_idx] = False
        self.assigned_codes.append(self.codebook[best_idx].clone())

        if verbose:
            print(f"    Class {class_label} -> Walsh code {best_idx} (BCE dist: {best_dist:.4f})")
        return best_idx

    def build_targets(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Build Walsh code target matrix from class labels.

        Args:
            labels: Class labels

        Returns:
            Walsh code targets tensor
        """
        targets = torch.zeros(len(labels), self.code_dim)
        for i, label in enumerate(labels.tolist()):
            code_idx = self.class_to_code_idx[int(label)]
            targets[i] = self.codebook[code_idx]
        return targets


class WalshNegotiatedDataset(Dataset):
    """
    Dataset wrapper that stores Walsh-negotiated labels alongside original data.
    """

    def __init__(self, x_data: torch.Tensor, walsh_targets: torch.Tensor,
                 true_labels: torch.Tensor, task_id: int):
        """
        Args:
            x_data: Input data
            walsh_targets: Negotiated Walsh code targets
            true_labels: Original class labels (0-indexed within task)
            task_id: Task identifier
        """
        self.x = x_data
        self.walsh_targets = walsh_targets
        self.y_true = true_labels
        self.task_id = task_id

        assert len(self.x) == len(self.walsh_targets) == len(self.y_true), \
            "Sample and label counts must match!"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.walsh_targets[idx], self.y_true[idx]


class WalshMLP(nn.Module):
    """
    MLP with Walsh code output layer and task-specific classifier heads.

    Architecture:
    - Shared feature extractor (MLP layers)
    - Walsh code output layer (for representation learning)
    - Task-specific classifier heads (for final classification)
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        code_dim: int = 16,
        num_classes_per_task: int = 2,
        num_tasks: int = 5,
        dropout: float = 0.0
    ):
        super(WalshMLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.code_dim = code_dim
        self.num_classes_per_task = num_classes_per_task
        self.num_tasks = num_tasks

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Walsh code output layer (shared across all tasks)
        self.walsh_layer = nn.Linear(hidden_size, code_dim)

        # Task-specific classifier heads (one head per task, 2 neurons each)
        self.heads = nn.ModuleList([
            nn.Linear(code_dim, num_classes_per_task)
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input tensor
            task_id: Task identifier (required for task-incremental)

        Returns:
            Logits for the specified task
        """
        features = self.features(x)
        walsh_features = self.walsh_layer(features)

        # Apply sigmoid to get Walsh-like representation
        walsh_repr = torch.sigmoid(walsh_features)

        if task_id is None:
            raise ValueError("task_id must be provided for task-incremental learning")

        # Pass through task-specific head
        return self.heads[task_id](walsh_repr)

    def get_walsh_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Walsh code features (before sigmoid).

        Args:
            x: Input tensor

        Returns:
            Walsh code logits
        """
        features = self.features(x)
        return self.walsh_layer(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features."""
        return self.features(x)


class WalshConvNet(nn.Module):
    """
    ConvNet with Walsh code output layer and task-specific classifier heads.

    Architecture (based on SimpleConvNet):
    - Shared convolutional feature extractor (3 conv blocks)
    - Walsh code output layer (for representation learning)
    - Task-specific classifier heads (for final classification)
    """

    def __init__(
        self,
        code_dim: int = 16,
        num_classes_per_task: int = 2,
        num_tasks: int = 5,
        dropout: float = 0.0,
        input_channels: int = 3
    ):
        super(WalshConvNet, self).__init__()

        self.code_dim = code_dim
        self.num_classes_per_task = num_classes_per_task
        self.num_tasks = num_tasks

        # Feature extractor (same as SimpleConvNet)
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),

            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),

            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),

            nn.Flatten(),
        )

        # Calculate feature size after convolutions
        # For CIFAR (32x32): after 3 pooling layers -> 4x4
        self.feature_size = 256 * 4 * 4

        # Walsh code output layer (shared across all tasks)
        self.walsh_layer = nn.Linear(self.feature_size, code_dim)

        # Task-specific classifier heads (one head per task)
        self.heads = nn.ModuleList([
            nn.Linear(code_dim, num_classes_per_task)
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input tensor
            task_id: Task identifier (required for task-incremental)

        Returns:
            Logits for the specified task
        """
        features = self.features(x)
        walsh_features = self.walsh_layer(features)

        # Apply sigmoid to get Walsh-like representation
        walsh_repr = torch.sigmoid(walsh_features)

        if task_id is None:
            raise ValueError("task_id must be provided for task-incremental learning")

        # Pass through task-specific head
        return self.heads[task_id](walsh_repr)

    def get_walsh_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Walsh code features (before sigmoid).

        Args:
            x: Input tensor

        Returns:
            Walsh code logits
        """
        features = self.features(x)
        return self.walsh_layer(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features."""
        return self.features(x)


class WalshConvNetLite(nn.Module):
    """
    Lighter ConvNet with Walsh code output - removes last conv block.

    Architecture:
    - 2 conv blocks (64 -> 128 channels) instead of 3
    - Walsh layer directly from flattened features (128×8×8 = 8192 -> code_dim)
    - Task-specific classifier heads

    This results in FEWER parameters than SimpleConvNet when code_dim <= 256.
    """

    def __init__(
        self,
        code_dim: int = 256,
        num_classes_per_task: int = 10,
        num_tasks: int = 10,
        dropout: float = 0.0,
        input_channels: int = 3
    ):
        super(WalshConvNetLite, self).__init__()

        self.code_dim = code_dim
        self.num_classes_per_task = num_classes_per_task
        self.num_tasks = num_tasks

        # Feature extractor (2 conv blocks only - no third block)
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),

            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),

            nn.Flatten(),
        )

        # Calculate feature size after convolutions
        # For CIFAR (32x32): after 2 pooling layers -> 8x8
        self.feature_size = 128 * 8 * 8  # = 8192

        # Walsh code output layer (shared across all tasks)
        # This replaces the third conv block
        self.walsh_layer = nn.Linear(self.feature_size, code_dim)

        # Task-specific classifier heads (one head per task)
        self.heads = nn.ModuleList([
            nn.Linear(code_dim, num_classes_per_task)
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass for classification."""
        features = self.features(x)
        walsh_features = self.walsh_layer(features)

        # Apply sigmoid to get Walsh-like representation
        walsh_repr = torch.sigmoid(walsh_features)

        if task_id is None:
            raise ValueError("task_id must be provided for task-incremental learning")

        return self.heads[task_id](walsh_repr)

    def get_walsh_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get Walsh code features (before sigmoid)."""
        features = self.features(x)
        return self.walsh_layer(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features."""
        return self.features(x)


class WalshNegotiationTrainer(BaseTrainer):
    """
    Walsh Negotiation trainer for task-incremental continual learning.

    This method combines Walsh code representations with negotiated targets
    and task-specific classifier heads.

    Algorithm:
    1. Initialize Walsh codebook and representation tracker
    2. For each task:
       a. Collect all samples for the task
       b. Assign Walsh codes to classes based on initial predictions
       c. Create negotiated targets: y_neg = (1-α) * walsh_code + α * pred_initial
       d. Train on negotiated targets using BCE loss for Walsh layer
       e. Train classifier head using cross-entropy
       f. Update negotiation rate using plasticity formula
    """

    def __init__(
        self,
        model: WalshMLP,
        optimizer,
        criterion,
        device,
        num_tasks,
        num_classes_per_task,
        config=None,
        initial_negotiation_rate=0.3,
        code_dim=16,
    ):
        """
        Initialize Walsh Negotiation trainer.

        Args:
            model: WalshMLP model
            optimizer: Optimizer
            criterion: Loss criterion
            device: Device to train on
            num_tasks: Total number of tasks
            num_classes_per_task: Number of classes per task
            config: Configuration dictionary
            initial_negotiation_rate: Initial α value (default: 0.3)
            code_dim: Dimension of Walsh codes
        """
        super().__init__(
            model, optimizer, criterion, device,
            num_tasks, num_classes_per_task, config
        )

        self.initial_negotiation_rate = initial_negotiation_rate
        self.negotiation_rate = initial_negotiation_rate
        self.code_dim = code_dim

        # Create Walsh codebook and tracker
        self.codebook = create_walsh_codebook(code_dim)
        self.tracker = RepresentationTracker(self.codebook)

        # Storage for negotiated datasets
        self.negotiated_loaders = {}

        # Track negotiation rate history
        self.negotiation_rate_history = [initial_negotiation_rate]

        # Track global class labels (original labels before remapping)
        self.global_class_offset = 0

        print(f"\n{'='*70}")
        print(f"Walsh Negotiation Method Initialized")
        print(f"{'='*70}")
        print(f"Walsh code dimension: {code_dim}")
        print(f"Initial negotiation rate α₀ = {initial_negotiation_rate:.3f}")
        print(f"  - Walsh code weight: {1-initial_negotiation_rate:.1%}")
        print(f"  - Initial model weight: {initial_negotiation_rate:.1%}")
        print(f"Negotiation rate will increase after each task using plasticity formula")
        print(f"{'='*70}\n")

    def prepare_task_data(self, task_id: int, train_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect all training data for a task.

        Args:
            task_id: Task identifier
            train_loader: Training data loader

        Returns:
            Tuple of (x_data, labels)
        """
        all_x = []
        all_y = []

        for batch in train_loader:
            if len(batch) == 3:
                x, y, t = batch
            else:
                x, y = batch

            all_x.append(x)
            all_y.append(y)

        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)

        return all_x, all_y

    def assign_walsh_codes(self, task_id: int, x_data: torch.Tensor,
                          labels: torch.Tensor, verbose: bool = True):
        """
        Assign Walsh codes to classes in the current task.

        Args:
            task_id: Task identifier
            x_data: Input data
            labels: Class labels (0-indexed within task)
            verbose: Whether to print assignment info
        """
        # Get unique classes in this task (should be 0 and 1 for 2-class tasks)
        unique_classes = torch.unique(labels).tolist()

        # Calculate global class labels
        class_offset = task_id * self.num_classes_per_task

        for local_class in unique_classes:
            global_class = int(local_class) + class_offset

            # Get samples for this class
            mask = labels == local_class
            class_samples = x_data[mask]

            # Assign Walsh code
            self.tracker.assign_code(
                global_class, self.model, class_samples,
                self.device, verbose=verbose
            )

    def create_negotiated_targets(self, task_id: int, x_data: torch.Tensor,
                                  labels: torch.Tensor) -> torch.Tensor:
        """
        Create negotiated Walsh code targets.

        Args:
            task_id: Task identifier
            x_data: Input data
            labels: Class labels (0-indexed within task)

        Returns:
            Negotiated Walsh code targets
        """
        # Calculate global class labels
        class_offset = task_id * self.num_classes_per_task
        global_labels = labels + class_offset

        # Build true Walsh targets
        y_walsh = self.tracker.build_targets(global_labels)

        # Get initial model predictions
        self.model.eval()
        with torch.no_grad():
            logits = self.model.get_walsh_features(x_data.to(self.device))
            y_pred = torch.sigmoid(logits).cpu()

        # Create negotiated targets
        y_negotiated = (1 - self.negotiation_rate) * y_walsh + self.negotiation_rate * y_pred

        return y_negotiated

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Train on a single task with Walsh negotiation.

        Args:
            task_id: Current task ID
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train

        Returns:
            Training statistics
        """
        self.current_task = task_id

        print(f"\n{'='*60}")
        print(f"Training on Task {task_id} with Walsh Negotiation")
        print(f"Current negotiation rate α = {self.negotiation_rate:.4f}")
        print(f"{'='*60}")

        # Hook: before task training
        self.before_task(task_id)

        # Step 1: Collect all training data
        print(f"Collecting training data for Task {task_id}...")
        x_data, labels = self.prepare_task_data(task_id, train_loader)
        print(f"  Collected {len(x_data)} samples")

        # Step 2: Assign Walsh codes to classes
        print(f"Assigning Walsh codes to classes...")
        self.assign_walsh_codes(task_id, x_data, labels, verbose=True)

        # Step 3: Create negotiated targets
        print(f"Creating negotiated targets...")
        y_negotiated = self.create_negotiated_targets(task_id, x_data, labels)
        print(f"  Negotiated: {100*(1-self.negotiation_rate):.1f}% Walsh codes + "
              f"{100*self.negotiation_rate:.1f}% initial predictions")

        # Step 4: Create negotiated dataset and loader
        negotiated_dataset = WalshNegotiatedDataset(
            x_data=x_data,
            walsh_targets=y_negotiated,
            true_labels=labels,
            task_id=task_id
        )

        batch_size = train_loader.batch_size if hasattr(train_loader, 'batch_size') else 128
        negotiated_loader = DataLoader(
            negotiated_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        # Step 5: Train
        print(f"Training for {epochs} epochs...")
        self.model.train()

        for epoch in range(epochs):
            train_stats = self._train_epoch_walsh(task_id, negotiated_loader, epoch)

            # Validation
            if val_loader is not None:
                val_stats = self._eval_single_task(task_id, val_loader)
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_stats['loss']:.4f}, "
                      f"Train Acc: {train_stats['accuracy']:.4f}, "
                      f"Val Acc: {val_stats['accuracy']:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_stats['loss']:.4f}, "
                      f"Train Acc: {train_stats['accuracy']:.4f}")

            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar(f"Task{task_id}/train_loss",
                                     train_stats['loss'], epoch)
                self.writer.add_scalar(f"Task{task_id}/train_acc",
                                     train_stats['accuracy'], epoch)
                self.writer.add_scalar(f"WalshNegotiation/alpha",
                                     self.negotiation_rate, task_id)

        # Hook: after task training
        self.after_task(task_id)

        print(f"Task {task_id} completed\n")

        return train_stats

    def _train_epoch_walsh(
        self,
        task_id: int,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch with Walsh negotiated targets.

        Training approach:
        - Train using soft-target cross-entropy on the classifier output
        - The soft targets are derived from the Walsh negotiated representation
        - This is similar to how standard negotiation works but with Walsh codes

        Args:
            task_id: Current task ID
            train_loader: DataLoader with (x, walsh_targets, y_true) tuples
            epoch: Current epoch number

        Returns:
            Training statistics
        """
        from tqdm import tqdm

        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (x, y_walsh, y_true) in enumerate(pbar):
            x = x.to(self.device)
            y_walsh = y_walsh.to(self.device)
            y_true = y_true.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Phase 1: Train Walsh layer with BCE on negotiated targets
            # This learns the Walsh representation
            walsh_logits = self.model.get_walsh_features(x)
            walsh_loss = F.binary_cross_entropy_with_logits(walsh_logits, y_walsh)

            # Phase 2: Train classifier head with CE on true labels
            # Use detached Walsh representation to prevent gradient interference
            # from BCE affecting classifier, and vice versa
            with torch.no_grad():
                walsh_repr = torch.sigmoid(walsh_logits)

            # Manually compute classifier output using detached Walsh repr
            classifier_logits = self.model.heads[task_id](walsh_repr)
            classifier_loss = F.cross_entropy(classifier_logits, y_true)

            # Total loss - separate updates for Walsh and classifier
            loss = walsh_loss + classifier_loss

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()

            # Get predictions using the full forward pass
            with torch.no_grad():
                full_logits = self.model(x, task_id=task_id)
                pred = full_logits.argmax(dim=1)
            correct += (pred == y_true).sum().item()
            total += y_true.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

            self.global_step += 1

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }

    def after_task(self, task_id: int):
        """
        Called after training on a task.
        Updates the negotiation rate using the plasticity formula.
        """
        old_rate = self.negotiation_rate

        # Plasticity formula: increases α over time
        plasticity_rate = optimal_plasticity_rate(self.negotiation_rate)
        self.negotiation_rate = self.negotiation_rate * plasticity_rate

        # Ensure it doesn't exceed 1.0
        self.negotiation_rate = min(self.negotiation_rate, 0.999)

        # Store history
        self.negotiation_rate_history.append(self.negotiation_rate)

        print(f"\n[INFO] Negotiation rate updated: {old_rate:.4f} → {self.negotiation_rate:.4f}")

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar("WalshNegotiation/alpha_after_task",
                                 self.negotiation_rate, task_id)

    def get_method_state(self) -> Optional[Dict[str, Any]]:
        """Get method-specific state for checkpointing."""
        return {
            'negotiation_rate': self.negotiation_rate,
            'initial_negotiation_rate': self.initial_negotiation_rate,
            'negotiation_rate_history': self.negotiation_rate_history,
            'code_dim': self.code_dim,
            'tracker_class_to_code': self.tracker.class_to_code_idx,
            'tracker_available': self.tracker.available,
        }

    def load_method_state(self, state: Dict[str, Any]):
        """Load method-specific state from checkpoint."""
        self.negotiation_rate = state['negotiation_rate']
        self.initial_negotiation_rate = state['initial_negotiation_rate']
        self.negotiation_rate_history = state['negotiation_rate_history']
        self.tracker.class_to_code_idx = state['tracker_class_to_code']
        self.tracker.available = state['tracker_available']

        print(f"Loaded Walsh negotiation state: α={self.negotiation_rate:.4f}")
