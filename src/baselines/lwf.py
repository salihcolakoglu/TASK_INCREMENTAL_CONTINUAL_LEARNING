"""
Learning without Forgetting (LwF) for task-incremental continual learning.

LwF prevents catastrophic forgetting using knowledge distillation. When training on
a new task, it uses the predictions from the previous model as soft targets to
preserve knowledge about old tasks.

Reference:
    Li & Hoiem. "Learning without Forgetting."
    ECCV 2016. https://arxiv.org/abs/1606.09282
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.base_trainer import BaseTrainer


class LwFTrainer(BaseTrainer):
    """
    Learning without Forgetting (LwF) trainer.

    LwF uses knowledge distillation to prevent forgetting. When learning a new task,
    it preserves the network's responses on old tasks by using the old model's
    predictions as soft targets.

    The key idea:
    1. After training task t, save a copy of the model
    2. When training task t+1:
       - Standard classification loss on new task (hard labels)
       - Distillation loss on old task heads (soft targets from old model)

    The total loss is:
        L_total = L_new + λ * L_distill

    where:
        L_new = CrossEntropy(predictions, hard_labels) for new task
        L_distill = Σ_{old tasks} KL(old_model_output || new_model_output)

    Temperature scaling is used to create softer probability distributions:
        σ(z, T) = exp(z_i/T) / Σ_j exp(z_j/T)

    The distillation loss is scaled by T² to maintain gradient magnitude.
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
        lwf_lambda=1.0,
        temperature=2.0,
    ):
        """
        Initialize LwF trainer.

        Args:
            model: Neural network model
            optimizer: Optimizer
            criterion: Loss criterion (for new task)
            device: Device to train on
            num_tasks: Total number of tasks
            num_classes_per_task: Number of classes per task
            config: Configuration dictionary
            lwf_lambda: Distillation loss weight (λ)
            temperature: Temperature for softening distributions (T)
        """
        super().__init__(
            model, optimizer, criterion, device,
            num_tasks, num_classes_per_task, config
        )

        self.lwf_lambda = lwf_lambda
        self.temperature = temperature

        # Storage for old model (previous task's model)
        self.old_model = None

        print(f"LwF initialized with λ={lwf_lambda}, temperature={temperature}")

    def after_task(self, task_id: int):
        """
        Called after training on a task.
        Save a copy of the current model to use for distillation.
        """
        print(f"\nSaving model snapshot for task {task_id}...")

        # Create a deep copy of the model
        self.old_model = deepcopy(self.model)

        # Set old model to eval mode and freeze it
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False

        print(f"Model snapshot saved.")

    def _compute_distillation_loss(self, x, current_task_id):
        """
        Compute knowledge distillation loss for old tasks.

        The distillation loss preserves the model's predictions on old task heads
        by matching the current model's outputs to the old model's outputs.

        Args:
            x: Input batch
            current_task_id: ID of current task being trained

        Returns:
            Distillation loss (scalar)
        """
        if self.old_model is None or current_task_id == 0:
            # No old tasks to distill from
            return 0.0

        distill_loss = 0.0

        # Compute distillation loss for each old task
        for old_task_id in range(current_task_id):
            # Get old model's predictions (frozen, no gradients)
            with torch.no_grad():
                old_logits = self.old_model(x, task_id=old_task_id)

            # Get current model's predictions (with gradients)
            new_logits = self.model(x, task_id=old_task_id)

            # Apply temperature scaling and compute soft targets
            # Temperature makes the distribution softer (more uncertain)
            old_soft_targets = F.softmax(old_logits / self.temperature, dim=1)
            new_log_probs = F.log_softmax(new_logits / self.temperature, dim=1)

            # Compute KL divergence between old and new distributions
            # Using cross-entropy for numerical stability
            # KL(P||Q) = Σ P(x) log(P(x)/Q(x)) = -Σ P(x) log(Q(x)) + Σ P(x) log(P(x))
            # Since P is fixed (old model), we only need the first term (cross-entropy)
            task_distill_loss = F.kl_div(
                new_log_probs,
                old_soft_targets,
                reduction='batchmean'
            )

            # Scale by T² to maintain gradient magnitude
            # (derivative of softmax with temperature introduces 1/T factor)
            task_distill_loss *= (self.temperature ** 2)

            distill_loss += task_distill_loss

        # Average over old tasks
        if current_task_id > 0:
            distill_loss /= current_task_id

        return distill_loss

    def before_backward(self, loss, task_id, batch_idx):
        """
        Add LwF distillation loss before backward pass.

        Args:
            loss: Current task loss (classification on new task)
            task_id: Current task ID
            batch_idx: Current batch index

        Returns:
            Modified loss with distillation term
        """
        # Don't add distillation for first task
        if task_id == 0 or self.old_model is None:
            return loss

        # Note: We need access to the current batch input to compute distillation
        # This is stored in self._current_batch_x during training
        if not hasattr(self, '_current_batch_x'):
            return loss

        # Compute distillation loss on old task heads
        distill_loss = self._compute_distillation_loss(
            self._current_batch_x,
            task_id
        )

        # Combine losses
        total_loss = loss + self.lwf_lambda * distill_loss

        return total_loss

    def _train_epoch(self, task_id, train_loader, epoch):
        """
        Override training epoch to store batch inputs for distillation.
        """
        self.model.train()

        # Keep old model in eval mode
        if self.old_model is not None:
            self.old_model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            if len(batch) == 3:
                x, y, t = batch
            else:
                x, y = batch

            x, y = x.to(self.device), y.to(self.device)

            # Store input for distillation loss computation
            self._current_batch_x = x

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x, task_id=task_id)

            # Compute classification loss on new task
            loss = self.compute_loss(logits, y, task_id, batch_idx)

            # Hook: before backward (adds distillation loss)
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

        # Clean up
        if hasattr(self, '_current_batch_x'):
            delattr(self, '_current_batch_x')

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }

    def get_method_state(self) -> Optional[Dict[str, Any]]:
        """
        Get LwF-specific state for checkpointing.
        """
        state = {
            'lwf_lambda': self.lwf_lambda,
            'temperature': self.temperature,
        }

        if self.old_model is not None:
            state['old_model'] = self.old_model.state_dict()

        return state

    def load_method_state(self, state: Dict[str, Any]):
        """
        Load LwF-specific state from checkpoint.
        """
        self.lwf_lambda = state['lwf_lambda']
        self.temperature = state['temperature']

        if 'old_model' in state:
            self.old_model = deepcopy(self.model)
            self.old_model.load_state_dict(state['old_model'])
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False
