"""
Synaptic Intelligence (SI) for task-incremental continual learning.

SI prevents catastrophic forgetting by computing parameter importance online
during training. It tracks how much each parameter contributes to reducing
the loss along the optimization trajectory.

Reference:
    Zenke et al. "Continual Learning Through Synaptic Intelligence."
    ICML 2017. https://arxiv.org/abs/1703.04200
"""

import os
import sys
import torch
from copy import deepcopy
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.base_trainer import BaseTrainer


class SynapticIntelligenceTrainer(BaseTrainer):
    """
    Synaptic Intelligence (SI) trainer.

    SI computes parameter importance online by tracking the contribution of
    each parameter to the loss reduction during training. The importance is
    measured using a path integral of gradients along the optimization trajectory.

    The importance for parameter i is:
        ω_i = Σ_t |g_t * Δθ_t| / (Δθ_i² + ε)

    where:
        - g_t is the gradient at step t
        - Δθ_t is the parameter update at step t
        - Δθ_i is the total parameter change for task i
        - ε is a small constant for numerical stability

    The regularization loss is:
        L_SI = λ/2 * Σ_i ω_i / (c + ω_i) * (θ_i - θ*_i)²

    where:
        - λ is the regularization strength
        - c is a damping parameter
        - θ*_i is the parameter value after previous task
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
        si_lambda=1.0,
        si_epsilon=0.001,
        damping=0.1,
    ):
        """
        Initialize SI trainer.

        Args:
            model: Neural network model
            optimizer: Optimizer
            criterion: Loss criterion
            device: Device to train on
            num_tasks: Total number of tasks
            num_classes_per_task: Number of classes per task
            config: Configuration dictionary
            si_lambda: Regularization strength (λ)
            si_epsilon: Small constant for numerical stability (ε)
            damping: Damping parameter (c) for normalizing importance
        """
        super().__init__(
            model, optimizer, criterion, device,
            num_tasks, num_classes_per_task, config
        )

        self.si_lambda = si_lambda
        self.si_epsilon = si_epsilon
        self.damping = damping

        # Storage for consolidated importance and old parameters
        self.omega = {}  # Consolidated importance after each task
        self.old_params = {}  # Parameters after previous task

        # Online tracking during current task
        self.W = {}  # Running sum of gradient * parameter_update
        self.prev_params = {}  # Parameters at previous step

        # Initialize tracking dictionaries
        self._init_tracking()

        print(f"SI initialized with λ={si_lambda}, ε={si_epsilon}, damping={damping}")

    def _init_tracking(self):
        """Initialize tracking variables for online importance computation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.W[name] = torch.zeros_like(param)
                self.prev_params[name] = param.clone().detach()

    def before_task(self, task_id: int):
        """
        Called before training on a new task.
        Reset online tracking variables.
        """
        # Reset W for new task
        for name in self.W.keys():
            self.W[name].zero_()

        # Store current parameters as previous
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prev_params[name] = param.clone().detach()

    def after_task(self, task_id: int):
        """
        Called after training on a task.
        Consolidate importance and update old parameters.
        """
        print(f"\nConsolidating importance for Task {task_id}...")

        # Get current parameters
        current_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Compute importance for this task
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Compute total parameter change for this task
                if task_id == 0:
                    # First task: initialize from initial parameters
                    prev_task_param = self.prev_params[name]
                else:
                    # Subsequent tasks: use params from start of this task
                    # (which were the params at end of previous task)
                    prev_task_param = self.old_params[name]

                delta = current_params[name] - prev_task_param

                # Compute importance: |W| / (delta² + epsilon)
                # W accumulated the gradient * parameter_update during training
                importance = self.W[name].abs() / (delta.pow(2) + self.si_epsilon)

                # Consolidate: add to running importance
                if task_id == 0:
                    self.omega[name] = importance
                else:
                    self.omega[name] += importance

        # Update old parameters
        self.old_params = current_params

        print(f"Importance consolidated.")

    def _update_importance_online(self):
        """
        Update importance tracking during training.
        This should be called after each optimizer step.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Compute parameter update
                    delta_param = param - self.prev_params[name]

                    # Accumulate: W += gradient * delta_param
                    # This approximates the path integral
                    self.W[name] += -param.grad * delta_param

                    # Update previous parameters for next step
                    self.prev_params[name] = param.clone().detach()

    def _train_epoch(self, task_id, train_loader, epoch):
        """
        Override training epoch to update importance online.
        """
        self.model.train()

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

            # SI-specific: Update importance online
            self._update_importance_online()

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

    def before_backward(self, loss, task_id, batch_idx):
        """
        Add SI regularization loss before backward pass.

        Args:
            loss: Current task loss
            task_id: Current task ID
            batch_idx: Current batch index

        Returns:
            Modified loss with SI regularization
        """
        # Don't add regularization for the first task
        if task_id == 0:
            return loss

        # Compute SI penalty
        si_loss = 0.0

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.omega:
                omega = self.omega[name]
                old_param = self.old_params[name]

                # SI regularization: omega / (c + omega) * (theta - theta*)²
                # The normalization prevents unbounded growth of importance
                si_loss += (
                    omega / (self.damping + omega) * (param - old_param).pow(2)
                ).sum()

        # Add regularization to loss
        total_loss = loss + (self.si_lambda / 2.0) * si_loss

        return total_loss

    def get_method_state(self) -> Optional[Dict[str, Any]]:
        """
        Get SI-specific state for checkpointing.
        """
        state = {
            'si_lambda': self.si_lambda,
            'si_epsilon': self.si_epsilon,
            'damping': self.damping,
            'omega': self.omega,
            'old_params': self.old_params,
            'W': self.W,
            'prev_params': self.prev_params,
        }
        return state

    def load_method_state(self, state: Dict[str, Any]):
        """
        Load SI-specific state from checkpoint.
        """
        self.si_lambda = state['si_lambda']
        self.si_epsilon = state['si_epsilon']
        self.damping = state['damping']
        self.omega = state['omega']
        self.old_params = state['old_params']
        self.W = state['W']
        self.prev_params = state['prev_params']
