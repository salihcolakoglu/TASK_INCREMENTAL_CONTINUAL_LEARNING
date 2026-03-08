"""
Memory Aware Synapses (MAS) for task-incremental continual learning.

MAS prevents catastrophic forgetting by computing parameter importance based on
how sensitive the learned function is to parameter changes. Unlike EWC which uses
Fisher Information, MAS uses the gradient magnitude of the output with respect
to parameters.

Reference:
    Aljundi et al. "Memory Aware Synapses: Learning what (not) to forget."
    ECCV 2018. https://arxiv.org/abs/1711.09601
"""

import os
import sys
import torch
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.base_trainer import BaseTrainer


class MASTrainer(BaseTrainer):
    """
    Memory Aware Synapses (MAS) trainer.

    MAS computes parameter importance by measuring the sensitivity of the learned
    function to parameter changes. The importance Ω for parameter θ_i is computed as:

        Ω_i = (1/N) Σ_x ||∂F(x)/∂θ_i||

    where F(x) is the network output and N is the number of samples.

    Key differences from EWC:
    1. Uses gradient of output (not log-likelihood) - measures function sensitivity
    2. Doesn't require labels - can work on unlabeled data
    3. More stable importance estimation across different task types

    The regularization loss is:
        L_MAS = λ/2 * Σ_i Ω_i * (θ_i - θ*_i)²

    where:
        - λ is the regularization strength
        - Ω_i is the accumulated importance for parameter i
        - θ_i is the current parameter value
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
        mas_lambda=1.0,
        num_samples=200,
    ):
        """
        Initialize MAS trainer.

        Args:
            model: Neural network model
            optimizer: Optimizer
            criterion: Loss criterion
            device: Device to train on
            num_tasks: Total number of tasks
            num_classes_per_task: Number of classes per task
            config: Configuration dictionary
            mas_lambda: Regularization strength (λ)
            num_samples: Number of samples to use for importance estimation
        """
        super().__init__(
            model, optimizer, criterion, device,
            num_tasks, num_classes_per_task, config
        )

        self.mas_lambda = mas_lambda
        self.num_samples = num_samples

        # Storage for accumulated importance and old parameters
        self.omega = {}  # Accumulated importance across all tasks
        self.old_params = {}  # Parameters after previous task

        # Initialize omega to zeros
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.omega[name] = torch.zeros_like(param)

        print(f"MAS initialized with λ={mas_lambda}, num_samples={num_samples}")

    def after_task(self, task_id: int):
        """
        Called after training on a task.
        Compute parameter importance and store old parameters.

        The importance is computed as the average gradient magnitude of the
        network output with respect to each parameter, evaluated on the task data.
        """
        print(f"\nComputing MAS importance for Task {task_id}...")

        # Compute importance for this task
        task_importance = self._compute_importance()

        # Accumulate importance across tasks
        for name in task_importance.keys():
            self.omega[name] += task_importance[name]

        # Store current parameters as old parameters
        self.old_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        print(f"MAS importance computed and accumulated.")

    def _compute_importance(self):
        """
        Compute parameter importance using MAS.

        The importance is computed as:
            Ω_i = (1/N) Σ_x ||∂F(x)/∂θ_i||

        where F(x) is the network output (logits before softmax).

        This measures how sensitive the learned function is to changes in each parameter.
        Unlike EWC's Fisher Information, this doesn't require labels and measures
        output sensitivity rather than likelihood sensitivity.

        Returns:
            Dictionary mapping parameter names to importance values
        """
        importance = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Set model to eval mode
        self.model.eval()

        # Get current task ID
        task_id = self.current_task

        # Access the training data loader
        if not hasattr(self, '_current_train_loader'):
            print("Warning: No train loader available for importance computation")
            return importance

        data_loader = self._current_train_loader

        # Collect samples and compute importance
        samples_collected = 0

        for batch_idx, batch in enumerate(data_loader):
            if samples_collected >= self.num_samples:
                break

            # Unpack batch
            if len(batch) == 3:
                x, y, t = batch
            else:
                x, y = batch

            x = x.to(self.device)

            # Process each sample individually
            for i in range(x.size(0)):
                if samples_collected >= self.num_samples:
                    break

                # Zero gradients
                self.model.zero_grad()

                # Forward pass for single sample
                sample = x[i:i+1]  # Keep batch dimension
                output = self.model(sample, task_id=task_id)

                # MAS uses the L2 norm of the output as the objective
                # This measures how much the network "cares" about this input
                # We want to preserve parameters that strongly affect the output

                # Compute L2 norm of output (before softmax)
                # ||F(x)||² = Σ_j F_j(x)²
                output_norm = output.norm(dim=1, p=2)

                # Backward pass to get gradients
                output_norm.backward()

                # Accumulate absolute gradients (importance)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Accumulate the absolute value of gradients
                        importance[name] += param.grad.abs()

                samples_collected += 1

        # Normalize by number of samples
        for name in importance.keys():
            if samples_collected > 0:
                importance[name] /= samples_collected

        # Set model back to train mode
        self.model.train()

        return importance

    def before_backward(self, loss, task_id, batch_idx):
        """
        Add MAS regularization loss before backward pass.

        Args:
            loss: Current task loss
            task_id: Current task ID
            batch_idx: Current batch index

        Returns:
            Modified loss with MAS regularization
        """
        # Don't add regularization for the first task
        if task_id == 0:
            return loss

        # Compute MAS penalty
        mas_loss = 0.0

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.omega:
                omega = self.omega[name]
                old_param = self.old_params[name]

                # MAS regularization: Ω_i * (θ_i - θ*_i)²
                mas_loss += (omega * (param - old_param).pow(2)).sum()

        # Add regularization to loss
        total_loss = loss + (self.mas_lambda / 2.0) * mas_loss

        return total_loss

    def train_task(self, task_id, train_loader, val_loader=None, epochs=10):
        """
        Override train_task to store the data loader for importance computation.
        """
        # Store data loader for importance computation
        self._current_train_loader = train_loader

        # Call parent train_task
        return super().train_task(task_id, train_loader, val_loader, epochs)

    def get_method_state(self) -> Optional[Dict[str, Any]]:
        """
        Get MAS-specific state for checkpointing.
        """
        state = {
            'mas_lambda': self.mas_lambda,
            'num_samples': self.num_samples,
            'omega': self.omega,
            'old_params': self.old_params,
        }
        return state

    def load_method_state(self, state: Dict[str, Any]):
        """
        Load MAS-specific state from checkpoint.
        """
        self.mas_lambda = state['mas_lambda']
        self.num_samples = state['num_samples']
        self.omega = state['omega']
        self.old_params = state['old_params']
