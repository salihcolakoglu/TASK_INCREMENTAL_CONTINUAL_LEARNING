"""
Neural network architectures for continual learning experiments.
All models support multi-head output for task-incremental learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for MNIST experiments.
    Supports multi-head output for task-incremental learning.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        num_classes_per_task: int = 10,
        num_tasks: int = 5,
        dropout: float = 0.0
    ):
        super(SimpleMLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
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

        # Multi-head classifier (one head per task)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, num_classes_per_task)
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            task_id: Task identifier (required for task-incremental)

        Returns:
            Logits for the specified task
        """
        features = self.features(x)

        if task_id is None:
            raise ValueError("task_id must be provided for task-incremental learning")

        return self.heads[task_id](features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.features(x)


class SimpleConvNet(nn.Module):
    """
    Simple ConvNet for CIFAR experiments.
    Supports multi-head output for task-incremental learning.
    """

    def __init__(
        self,
        num_classes_per_task: int = 10,
        num_tasks: int = 5,
        dropout: float = 0.0,
        input_channels: int = 3
    ):
        super(SimpleConvNet, self).__init__()

        self.num_classes_per_task = num_classes_per_task
        self.num_tasks = num_tasks

        # Feature extractor
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

        # Multi-head classifier
        self.heads = nn.ModuleList([
            nn.Linear(self.feature_size, num_classes_per_task)
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            task_id: Task identifier (required for task-incremental)

        Returns:
            Logits for the specified task
        """
        features = self.features(x)

        if task_id is None:
            raise ValueError("task_id must be provided for task-incremental learning")

        return self.heads[task_id](features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.features(x)


def get_model(
    architecture: str,
    num_classes_per_task: int,
    num_tasks: int,
    dropout: float = 0.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        architecture: Model architecture name ('mlp', 'convnet', 'resnet18', etc.)
        num_classes_per_task: Number of classes per task
        num_tasks: Total number of tasks
        dropout: Dropout probability
        **kwargs: Additional architecture-specific arguments

    Returns:
        Model instance
    """
    if architecture == "mlp":
        return SimpleMLP(
            num_classes_per_task=num_classes_per_task,
            num_tasks=num_tasks,
            dropout=dropout,
            **kwargs
        )
    elif architecture == "convnet":
        return SimpleConvNet(
            num_classes_per_task=num_classes_per_task,
            num_tasks=num_tasks,
            dropout=dropout,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
