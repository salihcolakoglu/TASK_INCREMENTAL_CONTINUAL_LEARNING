"""
Base trainer class for task-incremental continual learning.
All baseline methods and your custom method will inherit from this class.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, List, Tuple, Any
from tqdm import tqdm
import numpy as np

from .metrics import ContinualLearningMetrics


class BaseTrainer:
    """
    Base trainer for task-incremental continual learning.

    This class provides:
    - Training loop with progress bars
    - Evaluation on multiple tasks
    - Metric tracking (accuracy, forgetting, etc.)
    - Checkpointing
    - TensorBoard logging
    - Optional W&B logging

    Methods can override:
    - before_task(): Called before training on a new task
    - after_task(): Called after training on a task
    - compute_loss(): Custom loss computation
    - before_backward(): Called before backward pass (for custom regularization)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_tasks: int,
        num_classes_per_task: int,
        config: Optional[Dict[str, Any]] = None,
        activation_type: str = "softmax",  # NEW: activation type for experiments
    ):
        """
        Initialize base trainer.

        Args:
            model: Neural network model (with multi-head output)
            optimizer: Optimizer
            criterion: Loss criterion (e.g., CrossEntropyLoss)
            device: Device to train on
            num_tasks: Total number of tasks
            num_classes_per_task: Number of classes per task
            config: Additional configuration (logging, checkpointing, etc.)
            activation_type: Activation type ("softmax" or "sigmoid")
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.activation_type = activation_type

        # Import activation utilities
        from .activations import (
            compute_loss,
            get_predictions,
            get_probabilities,
            ActivationType
        )
        self._compute_loss_fn = lambda logits, labels: compute_loss(
            logits, labels, num_classes_per_task, activation_type
        )
        self._get_predictions = lambda logits: get_predictions(logits, activation_type)
        self._get_probabilities = lambda logits: get_probabilities(logits, activation_type)

        # Configuration
        self.config = config or {}
        self.use_tensorboard = self.config.get("use_tensorboard", True)
        self.use_wandb = self.config.get("use_wandb", False)
        self.save_checkpoints = self.config.get("save_checkpoints", True)
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./results/checkpoints")
        self.log_dir = self.config.get("log_dir", "./results/tensorboard")

        # Metrics
        self.metrics = ContinualLearningMetrics(num_tasks)
        self.current_task = 0

        # Logging
        self.writer = None
        if self.use_tensorboard:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)

        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not available, disabling W&B logging")
                self.use_wandb = False

        # Checkpointing
        if self.save_checkpoints:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training statistics
        self.global_step = 0
        self.task_start_time = None

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> Dict[str, float]:
        """
        Train on a single task.

        Args:
            task_id: Current task ID
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train

        Returns:
            Training statistics
        """
        self.current_task = task_id
        self.task_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Training on Task {task_id}")
        print(f"{'='*60}")

        # Hook: before task training
        self.before_task(task_id)

        # Training loop
        for epoch in range(epochs):
            train_stats = self._train_epoch(task_id, train_loader, epoch)

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
                if val_loader:
                    self.writer.add_scalar(f"Task{task_id}/val_acc",
                                         val_stats['accuracy'], epoch)

        # Hook: after task training
        self.after_task(task_id)

        task_time = time.time() - self.task_start_time
        print(f"Task {task_id} completed in {task_time:.2f}s")

        return train_stats

    def _train_epoch(
        self,
        task_id: int,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch (x, y, task_id)
            if len(batch) == 3:
                x, y, t = batch
            else:
                x, y = batch
                t = torch.full((x.size(0),), task_id, dtype=torch.long)

            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x, task_id=task_id)

            # Compute loss (can be overridden for custom losses)
            loss = self.compute_loss(logits, y, task_id, batch_idx)

            # Hook: before backward (for regularization)
            loss = self.before_backward(loss, task_id, batch_idx)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

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

    def evaluate_all_tasks(
        self,
        task_dataloaders: List[DataLoader],
        current_task: Optional[int] = None
    ) -> Dict[int, float]:
        """
        Evaluate on all tasks seen so far.

        Args:
            task_dataloaders: List of test dataloaders (one per task)
            current_task: Current task (evaluate up to this task)

        Returns:
            Dictionary mapping task_id -> accuracy
        """
        if current_task is None:
            current_task = self.current_task

        accuracies = {}

        print(f"\n{'='*60}")
        print(f"Evaluating on all tasks (0-{current_task})")
        print(f"{'='*60}")

        for task_id in range(current_task + 1):
            stats = self._eval_single_task(task_id, task_dataloaders[task_id])
            accuracies[task_id] = stats['accuracy']
            print(f"Task {task_id}: Accuracy = {stats['accuracy']:.4f}")

        # Update metrics
        self.metrics.update(current_task, accuracies)

        # Print continual learning metrics
        self.metrics.print_summary(current_task)

        # Log to tensorboard
        if self.writer:
            for task_id, acc in accuracies.items():
                self.writer.add_scalar(
                    f"Eval/task_{task_id}_accuracy",
                    acc,
                    current_task
                )

            # Log continual learning metrics
            cl_metrics = self.metrics.get_all_metrics(current_task)
            for metric_name, value in cl_metrics.items():
                # Skip task_accuracies (it's a list, not a scalar)
                if metric_name == 'task_accuracies':
                    continue
                self.writer.add_scalar(f"Metrics/{metric_name}", value, current_task)

        return accuracies

    def _eval_single_task(
        self,
        task_id: int,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate on a single task."""
        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch
                if len(batch) == 3:
                    x, y, t = batch
                else:
                    x, y = batch

                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                logits = self.model(x, task_id=task_id)
                loss = self.criterion(logits, y)

                # Statistics
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return {
            'loss': total_loss / len(test_loader),
            'accuracy': correct / total
        }

    def save_checkpoint(self, task_id: int, filename: Optional[str] = None):
        """Save model checkpoint."""
        if not self.save_checkpoints:
            return

        if filename is None:
            filename = f"task_{task_id}_checkpoint.pth"

        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'task_id': task_id,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics.get_accuracy_matrix(),
            'global_step': self.global_step,
        }

        # Add method-specific state
        method_state = self.get_method_state()
        if method_state:
            checkpoint['method_state'] = method_state

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_task = checkpoint['task_id']
        self.global_step = checkpoint['global_step']

        # Load method-specific state
        if 'method_state' in checkpoint:
            self.load_method_state(checkpoint['method_state'])

        print(f"Checkpoint loaded from {filepath}")

    # ============================================================
    # Methods to override in subclasses
    # ============================================================

    def before_task(self, task_id: int):
        """
        Called before training on a new task.
        Override this for task-specific initialization.
        """
        pass

    def after_task(self, task_id: int):
        """
        Called after training on a task.
        Override this for post-task processing (e.g., storing importance weights).
        """
        pass

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        task_id: int,
        batch_idx: int
    ) -> torch.Tensor:
        """
        Compute loss for the current batch.
        Override this for custom loss computation.

        Args:
            logits: Model predictions
            targets: Ground truth labels
            task_id: Current task ID
            batch_idx: Current batch index

        Returns:
            Loss tensor
        """
        return self.criterion(logits, targets)

    def before_backward(
        self,
        loss: torch.Tensor,
        task_id: int,
        batch_idx: int
    ) -> torch.Tensor:
        """
        Called before backward pass.
        Override this for custom regularization.

        Args:
            loss: Current loss
            task_id: Current task ID
            batch_idx: Current batch index

        Returns:
            Modified loss (with regularization added)
        """
        return loss

    def get_method_state(self) -> Optional[Dict[str, Any]]:
        """
        Get method-specific state for checkpointing.
        Override this to save additional state.
        """
        return None

    def load_method_state(self, state: Dict[str, Any]):
        """
        Load method-specific state from checkpoint.
        Override this to load additional state.
        """
        pass

    def close(self):
        """Clean up resources."""
        if self.writer:
            self.writer.close()
