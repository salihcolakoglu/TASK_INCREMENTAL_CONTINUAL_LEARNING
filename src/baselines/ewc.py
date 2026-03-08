"""
Elastic Weight Consolidation (EWC) for task-incremental continual learning.

EWC prevents catastrophic forgetting by adding a regularization term that penalizes
changes to parameters that are important for previous tasks. The importance is
measured using the Fisher Information Matrix.

Reference:
    Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks."
    PNAS 2017. https://arxiv.org/abs/1612.00796
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


class EWCTrainer(BaseTrainer):
    """
    Elastic Weight Consolidation (EWC) trainer.

    EWC computes the Fisher Information Matrix (FIM) after training on each task
    to identify important parameters. When training on new tasks, it adds a
    regularization term to prevent large changes to important parameters.

    The regularization loss is:
        L_EWC = λ/2 * Σ_i F_i * (θ_i - θ*_i)^2

    where:
        - λ is the regularization strength
        - F_i is the Fisher information for parameter i
        - θ_i is the current parameter value
        - θ*_i is the parameter value after previous task

    Modes:
        - online: Accumulate Fisher information across tasks (more memory efficient)
        - separate: Keep separate Fisher matrices per task (more accurate)
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
        ewc_lambda=1000.0,
        mode='online',
        gamma=1.0,
    ):
        """
        Initialize EWC trainer.

        Args:
            model: Neural network model
            optimizer: Optimizer
            criterion: Loss criterion
            device: Device to train on
            num_tasks: Total number of tasks
            num_classes_per_task: Number of classes per task
            config: Configuration dictionary
            ewc_lambda: Regularization strength (λ)
            mode: 'online' or 'separate'
            gamma: Decay factor for online EWC (only used if mode='online')
        """
        super().__init__(
            model, optimizer, criterion, device,
            num_tasks, num_classes_per_task, config
        )

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.gamma = gamma

        # Storage for old parameters and Fisher information
        if mode == 'online':
            # Online EWC: single Fisher matrix and params dict
            self.fisher = {}
            self.old_params = {}
        elif mode == 'separate':
            # Separate EWC: list of Fisher matrices and params per task
            self.fisher_list = []
            self.old_params_list = []
        else:
            raise ValueError(f"Unknown EWC mode: {mode}. Use 'online' or 'separate'")

        print(f"EWC initialized with λ={ewc_lambda}, mode={mode}")

    def after_task(self, task_id: int):
        """
        Called after training on a task.
        Compute Fisher Information Matrix and store old parameters.
        """
        print(f"\nComputing Fisher Information Matrix for Task {task_id}...")

        # Compute Fisher information
        fisher = self._compute_fisher()

        # Store parameters
        old_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        if self.mode == 'online':
            # Online EWC: accumulate Fisher information
            if task_id == 0:
                # First task: initialize
                self.fisher = fisher
                self.old_params = old_params
            else:
                # Subsequent tasks: weighted average
                for name in fisher.keys():
                    self.fisher[name] = (
                        self.gamma * self.fisher[name] +
                        fisher[name]
                    )
                self.old_params = old_params

        elif self.mode == 'separate':
            # Separate EWC: store each task's Fisher and params
            self.fisher_list.append(fisher)
            self.old_params_list.append(old_params)

        print(f"Fisher Information Matrix computed and stored.")

    def _compute_fisher(self, num_samples: int = 200):
        """
        Compute Fisher Information Matrix using the diagonal approximation.

        The diagonal FIM is computed as:
            F_i = E[(∂log p(y|x,θ)/∂θ_i)^2]

        We approximate this by sampling data and computing gradients.

        Args:
            num_samples: Number of samples to use for estimation

        Returns:
            Dictionary mapping parameter names to Fisher information
        """
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Set model to eval mode
        self.model.eval()

        # Get data from current task
        # Note: We use the training data from the task we just finished
        task_id = self.current_task

        # We'll sample batches until we get enough samples
        samples_collected = 0

        # Create a temporary data loader (we need to access it somehow)
        # For now, we'll compute Fisher on the full dataset
        # In practice, you might want to pass the data loader to this method

        # Since we don't have direct access to the data loader here,
        # we'll store it during training
        if not hasattr(self, '_current_train_loader'):
            print("Warning: No train loader available for Fisher computation")
            return fisher

        data_loader = self._current_train_loader

        with torch.no_grad():
            # Zero out gradients
            self.model.zero_grad()

        for batch_idx, batch in enumerate(data_loader):
            if samples_collected >= num_samples:
                break

            # Unpack batch
            if len(batch) == 3:
                x, y, t = batch
            else:
                x, y = batch

            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            self.model.zero_grad()
            logits = self.model(x, task_id=task_id)

            # Compute log likelihood
            # We use the predicted labels (not true labels) for Fisher computation
            # This is the standard approach in EWC
            log_probs = F.log_softmax(logits, dim=1)

            # For each sample, compute gradient of log probability
            for i in range(x.size(0)):
                if samples_collected >= num_samples:
                    break

                self.model.zero_grad()

                # Compute loss for this sample
                loss = -log_probs[i, y[i]]
                loss.backward(retain_graph=True)

                # Accumulate squared gradients (Fisher)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.pow(2)

                samples_collected += 1

        # Normalize by number of samples
        for name in fisher.keys():
            fisher[name] /= samples_collected

        return fisher

    def before_backward(self, loss, task_id, batch_idx):
        """
        Add EWC regularization loss before backward pass.

        Args:
            loss: Current task loss
            task_id: Current task ID
            batch_idx: Current batch index

        Returns:
            Modified loss with EWC regularization
        """
        # Don't add regularization for the first task
        if task_id == 0:
            return loss

        # Compute EWC penalty
        ewc_loss = 0.0

        if self.mode == 'online':
            # Online EWC: single regularization term
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.fisher:
                    fisher = self.fisher[name]
                    old_param = self.old_params[name]
                    ewc_loss += (fisher * (param - old_param).pow(2)).sum()

        elif self.mode == 'separate':
            # Separate EWC: sum over all previous tasks
            for task_fisher, task_params in zip(self.fisher_list, self.old_params_list):
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in task_fisher:
                        fisher = task_fisher[name]
                        old_param = task_params[name]
                        ewc_loss += (fisher * (param - old_param).pow(2)).sum()

        # Add regularization to loss
        total_loss = loss + (self.ewc_lambda / 2.0) * ewc_loss

        return total_loss

    def train_task(self, task_id, train_loader, val_loader=None, epochs=10):
        """
        Override train_task to store the data loader for Fisher computation.
        """
        # Store data loader for Fisher computation
        self._current_train_loader = train_loader

        # Call parent train_task
        return super().train_task(task_id, train_loader, val_loader, epochs)

    def get_method_state(self) -> Optional[Dict[str, Any]]:
        """
        Get EWC-specific state for checkpointing.
        """
        state = {
            'ewc_lambda': self.ewc_lambda,
            'mode': self.mode,
            'gamma': self.gamma,
        }

        if self.mode == 'online':
            state['fisher'] = self.fisher
            state['old_params'] = self.old_params
        elif self.mode == 'separate':
            state['fisher_list'] = self.fisher_list
            state['old_params_list'] = self.old_params_list

        return state

    def load_method_state(self, state: Dict[str, Any]):
        """
        Load EWC-specific state from checkpoint.
        """
        self.ewc_lambda = state['ewc_lambda']
        self.mode = state['mode']
        self.gamma = state.get('gamma', 1.0)

        if self.mode == 'online':
            self.fisher = state['fisher']
            self.old_params = state['old_params']
        elif self.mode == 'separate':
            self.fisher_list = state['fisher_list']
            self.old_params_list = state['old_params_list']
