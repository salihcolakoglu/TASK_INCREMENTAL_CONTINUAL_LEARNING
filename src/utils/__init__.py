"""Utilities for continual learning experiments."""

from .base_trainer import BaseTrainer
from .metrics import ContinualLearningMetrics
from .data_utils import (
    TaskIncrementalDataset,
    get_split_mnist,
    get_split_cifar10,
    get_split_cifar100,
    get_dataset
)

__all__ = [
    'BaseTrainer',
    'ContinualLearningMetrics',
    'TaskIncrementalDataset',
    'get_split_mnist',
    'get_split_cifar10',
    'get_split_cifar100',
    'get_dataset',
]
