"""
Negotiation method for task-incremental continual learning.

The Negotiation method prevents catastrophic forgetting through adaptive label negotiation.
Instead of training only on true labels, the model trains on "negotiated targets" that are
a weighted combination of true labels and the model's INITIAL predictions (before any training).

The key idea:
1. BEFORE any training, get model's initial predictions on ALL tasks' training data
2. Create negotiated targets ONCE: ŷ = (1 - α) * y_true + α * y_pred_initial
3. Store these negotiated targets permanently
4. Train on stored negotiated targets throughout all tasks
5. This makes the method independent of training epochs and dynamics

The negotiation rate α can optionally increase after each task using:
    plasticity_rate = 1 / (2α - α²)
    α_new = α * plasticity_rate

This adaptive mechanism balances:
- Plasticity: Learning new information (when α is low, trust labels more)
- Stability: Preserving old knowledge (when α is high, trust model more)

IMPORTANT: Negotiated labels are computed ONCE using the initial untrained model,
NOT recalculated at each task. This prevents vulnerability to training dynamics.

Reference:
    Based on negotiated representations approach for continual learning.
    Adapted from class-incremental to task-incremental setting.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Dict, Any, Optional, List
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.base_trainer import BaseTrainer


class NegotiatedDataset(Dataset):
    """
    Dataset wrapper that stores negotiated labels alongside original data.

    This ensures:
    1. Label-sample correspondence is maintained
    2. Original dataset is not modified
    3. Negotiated labels are stored efficiently
    """

    def __init__(self, original_data: List[torch.Tensor], negotiated_labels: torch.Tensor,
                 true_labels: torch.Tensor, task_id: int):
        """
        Args:
            original_data: List of tensors (x, y, t) from original dataset
            negotiated_labels: Pre-computed negotiated soft targets [N, num_classes]
            true_labels: Original hard labels [N]
            task_id: Task identifier
        """
        self.x = original_data[0]
        self.y_negotiated = negotiated_labels
        self.y_true = true_labels
        self.task_id = task_id

        assert len(self.x) == len(self.y_negotiated) == len(self.y_true), \
            "Sample and label counts must match!"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_negotiated[idx], self.y_true[idx]


class NegotiationTrainer(BaseTrainer):
    """
    Negotiation trainer for task-incremental continual learning.

    The Negotiation method creates soft targets by mixing true labels with
    the model's INITIAL predictions (before any training). The mixing ratio
    (negotiation_rate) can optionally increase over tasks.

    Algorithm:
    1. Initialize negotiation_rate = α₀ (e.g., 0.5)
    2. BEFORE training: Pre-compute negotiated labels for ALL tasks
       a. Get INITIAL model predictions: y_pred = model_initial(x_train)
       b. Negotiate targets ONCE: y_neg = (1-α) * y_true + α * y_pred_initial
       c. Store negotiated targets for each task
    3. For each task t:
       a. Load pre-computed negotiated targets
       b. Train on negotiated targets: minimize KL(model(x), y_neg)
       c. Optionally update α: α_new = α * (1 / (2α - α²))
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        num_tasks,
        num_classes_per_task,
        config=None,
        initial_negotiation_rate=0.5,
        update_negotiation_rate=False,
    ):
        """
        Initialize Negotiation trainer.

        Args:
            model: Neural network model
            optimizer: Optimizer
            criterion: Loss criterion (not used directly; KL divergence for soft targets)
            device: Device to train on
            num_tasks: Total number of tasks
            num_classes_per_task: Number of classes per task
            config: Configuration dictionary
            initial_negotiation_rate: Initial α value (0 < α < 1)
                - Lower values (e.g., 0.3): Trust labels more (90% label, 10% model)
                - Higher values (e.g., 0.7): Trust model more (30% label, 70% model)
                - Recommended: 0.5 (equal weighting)
            update_negotiation_rate: Whether to update α after each task (default: False)
                - True: α increases after each task
                - False: α stays constant throughout training (RECOMMENDED)
        """
        super().__init__(
            model, optimizer, criterion, device,
            num_tasks, num_classes_per_task, config
        )

        self.initial_negotiation_rate = initial_negotiation_rate
        self.negotiation_rate = initial_negotiation_rate
        self.update_negotiation_rate = update_negotiation_rate

        # Storage for pre-computed negotiated labels
        # Dict mapping task_id -> negotiated DataLoader
        self.negotiated_loaders = {}

        # Track negotiation rate history for analysis
        self.negotiation_rate_history = [initial_negotiation_rate]

        print(f"\n{'='*70}")
        print(f"Negotiation Method Initialized")
        print(f"{'='*70}")
        print(f"Initial negotiation rate α₀ = {initial_negotiation_rate:.3f}")
        print(f"  - Label weight: {1-initial_negotiation_rate:.1%}")
        print(f"  - Initial model weight: {initial_negotiation_rate:.1%}")
        if update_negotiation_rate:
            print(f"Negotiation rate will increase after each task")
        else:
            print(f"Negotiation rate will remain constant")
        print(f"{'='*70}\n")

    def prepare_negotiated_labels(self, train_loaders: List[DataLoader], batch_size: int = 128):
        """
        Pre-compute negotiated labels for ALL tasks using the INITIAL model.

        This is called ONCE before any training begins. It:
        1. Uses the initial untrained model to get predictions on all tasks
        2. Creates negotiated targets for each task
        3. Stores them for later use during training

        This ensures negotiation is independent of training dynamics and number of epochs.

        Args:
            train_loaders: List of training DataLoaders for all tasks
            batch_size: Batch size for creating negotiated DataLoaders
        """
        print(f"\n{'='*70}")
        print(f"PRE-COMPUTING NEGOTIATED LABELS FOR ALL TASKS")
        print(f"Using INITIAL model (before any training)")
        print(f"{'='*70}\n")

        self.model.eval()

        for task_id, train_loader in enumerate(train_loaders):
            print(f"Task {task_id}: Pre-computing negotiated labels...")

            # Step 1: Collect all training data for this task
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

            print(f"  Collected {len(all_x)} training samples")

            # IMPORTANT: Ensure labels are in range [0, num_classes_per_task)
            # They should already be remapped by TaskIncrementalDataset,
            # but we double-check for safety (multiprocessing can cause issues)
            min_label = all_y.min().item()
            max_label = all_y.max().item()

            if max_label >= self.num_classes_per_task:
                print(f"  ⚠️  WARNING: Labels range [{min_label}, {max_label}] exceeds num_classes_per_task={self.num_classes_per_task}")
                print(f"  Remapping labels now...")
                # Assume labels are sequential starting from task_id * num_classes_per_task
                class_offset = task_id * self.num_classes_per_task
                all_y = all_y - class_offset
                print(f"  Applied offset: {class_offset}, new range: [{all_y.min().item()}, {all_y.max().item()}]")

            # Step 2: Get INITIAL model's predictions (CRITICAL: this is done BEFORE training)
            with torch.no_grad():
                all_x_device = all_x.to(self.device)
                logits = self.model(all_x_device, task_id=task_id)

                # Get soft predictions (probabilities)
                y_pred = F.softmax(logits, dim=1).cpu()

            # Step 3: Create negotiated targets
            # Convert true labels to one-hot
            y_true_onehot = F.one_hot(all_y, num_classes=self.num_classes_per_task).float()

            # Negotiate: mix true labels with INITIAL model predictions
            y_negotiated = (1 - self.negotiation_rate) * y_true_onehot + \
                           self.negotiation_rate * y_pred

            print(f"  Negotiated: {100*(1-self.negotiation_rate):.1f}% labels + "
                  f"{100*self.negotiation_rate:.1f}% initial predictions")

            # Step 4: Create dataset and dataloader with negotiated targets
            # Use NegotiatedDataset to maintain sample-label correspondence
            negotiated_dataset = NegotiatedDataset(
                original_data=[all_x, all_y, None],
                negotiated_labels=y_negotiated,
                true_labels=all_y,
                task_id=task_id
            )

            negotiated_loader = DataLoader(
                negotiated_dataset,
                batch_size=batch_size,
                shuffle=True,  # Shuffle for training, but correspondence is maintained
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=True if self.device.type == 'cuda' else False
            )

            # Step 5: Store for later use
            self.negotiated_loaders[task_id] = negotiated_loader

            print(f"  ✓ Negotiated labels stored for Task {task_id}\n")

        print(f"{'='*70}")
        print(f"NEGOTIATED LABELS PREPARED FOR ALL {len(train_loaders)} TASKS")
        print(f"These labels are FIXED and will NOT change during training.")
        print(f"{'='*70}\n")

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Train on a single task with PRE-COMPUTED negotiated labels.

        This method:
        1. Retrieves pre-computed negotiated labels for the task
        2. Trains on these fixed negotiated targets
        3. Does NOT recalculate labels (making it epoch-independent)

        Args:
            task_id: Current task ID
            train_loader: Training data loader (NOT USED - we use pre-computed)
            val_loader: Optional validation data loader
            epochs: Number of epochs to train

        Returns:
            Training statistics
        """
        self.current_task = task_id

        # Check if negotiated labels were pre-computed
        if task_id not in self.negotiated_loaders:
            raise RuntimeError(
                f"Negotiated labels for task {task_id} not found! "
                f"You must call prepare_negotiated_labels() before training."
            )

        print(f"\n{'='*60}")
        print(f"Training on Task {task_id} with Negotiation")
        print(f"Using PRE-COMPUTED negotiated labels")
        print(f"Current negotiation rate α = {self.negotiation_rate:.4f}")
        print(f"{'='*60}")

        # Hook: before task training
        self.before_task(task_id)

        # Get pre-computed negotiated loader
        negotiated_loader = self.negotiated_loaders[task_id]

        # Train using pre-computed negotiated targets
        print(f"Training with PRE-COMPUTED negotiated targets for {epochs} epochs...")
        self.model.train()

        for epoch in range(epochs):
            train_stats = self._train_epoch_negotiated(
                task_id, negotiated_loader, epoch
            )

            # Validation (use original validation loader with hard labels)
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
                self.writer.add_scalar(f"Negotiation/alpha",
                                     self.negotiation_rate, task_id)
                if val_loader:
                    self.writer.add_scalar(f"Task{task_id}/val_acc",
                                         val_stats['accuracy'], epoch)

        # Hook: after task training
        self.after_task(task_id)

        print(f"Task {task_id} completed\n")

        return train_stats

    def _train_epoch_negotiated(
        self,
        task_id: int,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch with negotiated targets.

        This is a custom training loop that handles soft targets.

        Args:
            task_id: Current task ID
            train_loader: DataLoader with (x, y_negotiated, y_true) tuples
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
        for batch_idx, (x, y_negotiated, y_true) in enumerate(pbar):
            x = x.to(self.device)
            y_negotiated = y_negotiated.to(self.device)
            y_true = y_true.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x, task_id=task_id)

            # Compute loss with soft targets
            # We use KL divergence or cross-entropy with soft targets
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(y_negotiated * log_probs).sum(dim=1).mean()

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics (use hard labels for accuracy)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
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

        IMPORTANT NOTE: Since negotiated labels are PRE-COMPUTED for all tasks,
        updating negotiation_rate here has NO EFFECT on future tasks' labels.
        All negotiated labels were fixed when prepare_negotiated_labels() was called.

        This method is kept for consistency with the base trainer interface.
        """
        if not self.update_negotiation_rate:
            # Keep negotiation rate constant
            print(f"\n[INFO] Negotiation rate α={self.negotiation_rate:.4f} (constant)")
            print(f"       All tasks use pre-computed labels: "
                  f"{100*(1-self.negotiation_rate):.1f}% original + "
                  f"{100*self.negotiation_rate:.1f}% initial predictions")
            return

        # NOTE: This update is for tracking purposes only.
        # It does NOT affect the pre-computed negotiated labels.
        old_rate = self.negotiation_rate

        # Plasticity formula: increases α over time
        plasticity_rate = 1.0 / (2 * self.negotiation_rate - self.negotiation_rate ** 2)
        self.negotiation_rate = self.negotiation_rate * plasticity_rate

        # Ensure it doesn't exceed 1.0 (shouldn't happen mathematically, but safety check)
        self.negotiation_rate = min(self.negotiation_rate, 0.999)

        # Store history
        self.negotiation_rate_history.append(self.negotiation_rate)

        print(f"\n[INFO] Negotiation rate tracking: {old_rate:.4f} → {self.negotiation_rate:.4f}")
        print(f"       (This does NOT affect pre-computed labels for future tasks)")

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar("Negotiation/alpha_after_task",
                                 self.negotiation_rate, task_id)

    def get_method_state(self) -> Optional[Dict[str, Any]]:
        """
        Get method-specific state for checkpointing.
        """
        return {
            'negotiation_rate': self.negotiation_rate,
            'initial_negotiation_rate': self.initial_negotiation_rate,
            'negotiation_rate_history': self.negotiation_rate_history,
            'update_negotiation_rate': self.update_negotiation_rate,
        }

    def load_method_state(self, state: Dict[str, Any]):
        """
        Load method-specific state from checkpoint.
        """
        self.negotiation_rate = state['negotiation_rate']
        self.initial_negotiation_rate = state['initial_negotiation_rate']
        self.negotiation_rate_history = state['negotiation_rate_history']
        self.update_negotiation_rate = state.get('update_negotiation_rate', True)

        print(f"Loaded negotiation state: α={self.negotiation_rate:.4f}")
