"""
Data utilities for task-incremental continual learning.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import List, Tuple, Optional
import numpy as np


class TaskIncrementalDataset(Dataset):
    """
    Wrapper for task-incremental datasets.
    Returns (x, y, task_id) tuples with remapped labels.

    Labels are remapped to be 0-indexed within each task.
    For example, if task contains classes [4, 5, 6, 7, 8, 9],
    they will be remapped to [0, 1, 2, 3, 4, 5].
    """

    def __init__(self, dataset: Dataset, task_id: int, class_offset: int = 0):
        """
        Args:
            dataset: Base dataset
            task_id: Task identifier
            class_offset: Offset to subtract from labels (e.g., first class in task)
        """
        self.dataset = dataset
        self.task_id = task_id
        self.class_offset = class_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # Remap label to be 0-indexed for this task
        y_remapped = y - self.class_offset
        return x, y_remapped, torch.tensor(self.task_id, dtype=torch.long)


def get_split_mnist(
    n_tasks: int = 5,
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    validation_split: float = 0.0
) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
    """
    Create Split MNIST benchmark.

    Args:
        n_tasks: Number of tasks (5 or 10)
        data_root: Root directory for data
        batch_size: Batch size
        num_workers: Number of data loading workers
        validation_split: Fraction of training data for validation

    Returns:
        (train_loaders, val_loaders, test_loaders)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full MNIST dataset
    train_dataset = datasets.MNIST(
        data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_root, train=False, download=True, transform=transform
    )

    # Split into tasks
    classes_per_task = 10 // n_tasks
    train_loaders = []
    val_loaders = []
    test_loaders = []

    for task_id in range(n_tasks):
        # Define class range for this task
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task

        # Filter train data
        train_indices = [
            i for i, (_, label) in enumerate(train_dataset)
            if start_class <= label < end_class
        ]

        # Validation split
        if validation_split > 0:
            n_val = int(len(train_indices) * validation_split)
            np.random.shuffle(train_indices)
            val_indices = train_indices[:n_val]
            train_indices = train_indices[n_val:]

            val_subset = torch.utils.data.Subset(train_dataset, val_indices)
            val_task_dataset = TaskIncrementalDataset(val_subset, task_id, class_offset=start_class)
            val_loader = DataLoader(
                val_task_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            val_loaders.append(val_loader)
        else:
            val_loaders.append(None)

        # Create train loader
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        train_task_dataset = TaskIncrementalDataset(train_subset, task_id, class_offset=start_class)
        train_loader = DataLoader(
            train_task_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        train_loaders.append(train_loader)

        # Filter test data
        test_indices = [
            i for i, (_, label) in enumerate(test_dataset)
            if start_class <= label < end_class
        ]

        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        test_task_dataset = TaskIncrementalDataset(test_subset, task_id, class_offset=start_class)
        test_loader = DataLoader(
            test_task_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders


def get_split_cifar10(
    n_tasks: int = 5,
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    validation_split: float = 0.0,
    augment: bool = True
) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
    """
    Create Split CIFAR-10 benchmark.

    Args:
        n_tasks: Number of tasks (typically 5)
        data_root: Root directory for data
        batch_size: Batch size
        num_workers: Number of data loading workers
        validation_split: Fraction of training data for validation
        augment: Whether to use data augmentation

    Returns:
        (train_loaders, val_loaders, test_loaders)
    """
    # Define transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2023, 0.1994, 0.2010))
    ])

    # Load full CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        data_root, train=False, download=True, transform=test_transform
    )

    # Split into tasks
    classes_per_task = 10 // n_tasks
    train_loaders = []
    val_loaders = []
    test_loaders = []

    for task_id in range(n_tasks):
        # Define class range for this task
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task

        # Filter train data
        train_indices = [
            i for i, (_, label) in enumerate(train_dataset)
            if start_class <= label < end_class
        ]

        # Validation split
        if validation_split > 0:
            n_val = int(len(train_indices) * validation_split)
            np.random.shuffle(train_indices)
            val_indices = train_indices[:n_val]
            train_indices = train_indices[n_val:]

            val_subset = torch.utils.data.Subset(train_dataset, val_indices)
            val_task_dataset = TaskIncrementalDataset(val_subset, task_id, class_offset=start_class)
            val_loader = DataLoader(
                val_task_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            val_loaders.append(val_loader)
        else:
            val_loaders.append(None)

        # Create train loader
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        train_task_dataset = TaskIncrementalDataset(train_subset, task_id, class_offset=start_class)
        train_loader = DataLoader(
            train_task_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        train_loaders.append(train_loader)

        # Filter test data
        test_indices = [
            i for i, (_, label) in enumerate(test_dataset)
            if start_class <= label < end_class
        ]

        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        test_task_dataset = TaskIncrementalDataset(test_subset, task_id, class_offset=start_class)
        test_loader = DataLoader(
            test_task_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders


def get_split_cifar100(
    n_tasks: int = 10,
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    validation_split: float = 0.0,
    augment: bool = True
) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
    """
    Create Split CIFAR-100 benchmark.

    Args:
        n_tasks: Number of tasks (typically 10 or 20)
        data_root: Root directory for data
        batch_size: Batch size
        num_workers: Number of data loading workers
        validation_split: Fraction of training data for validation
        augment: Whether to use data augmentation

    Returns:
        (train_loaders, val_loaders, test_loaders)
    """
    # Define transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                               (0.2675, 0.2565, 0.2761))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                               (0.2675, 0.2565, 0.2761))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761))
    ])

    # Load full CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(
        data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        data_root, train=False, download=True, transform=test_transform
    )

    # Split into tasks
    classes_per_task = 100 // n_tasks
    train_loaders = []
    val_loaders = []
    test_loaders = []

    for task_id in range(n_tasks):
        # Define class range for this task
        start_class = task_id * classes_per_task
        end_class = (task_id + 1) * classes_per_task

        # Filter train data
        train_indices = [
            i for i, (_, label) in enumerate(train_dataset)
            if start_class <= label < end_class
        ]

        # Validation split
        if validation_split > 0:
            n_val = int(len(train_indices) * validation_split)
            np.random.shuffle(train_indices)
            val_indices = train_indices[:n_val]
            train_indices = train_indices[n_val:]

            val_subset = torch.utils.data.Subset(train_dataset, val_indices)
            val_task_dataset = TaskIncrementalDataset(val_subset, task_id, class_offset=start_class)
            val_loader = DataLoader(
                val_task_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            val_loaders.append(val_loader)
        else:
            val_loaders.append(None)

        # Create train loader
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        train_task_dataset = TaskIncrementalDataset(train_subset, task_id, class_offset=start_class)
        train_loader = DataLoader(
            train_task_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        train_loaders.append(train_loader)

        # Filter test data
        test_indices = [
            i for i, (_, label) in enumerate(test_dataset)
            if start_class <= label < end_class
        ]

        test_subset = torch.utils.data.Subset(test_dataset, test_indices)
        test_task_dataset = TaskIncrementalDataset(test_subset, task_id, class_offset=start_class)
        test_loader = DataLoader(
            test_task_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        test_loaders.append(test_loader)

    return train_loaders, val_loaders, test_loaders


def get_dataset(
    dataset_name: str,
    n_tasks: int,
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    validation_split: float = 0.0,
    **kwargs
) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
    """
    Factory function to get dataset loaders.

    Args:
        dataset_name: Name of dataset ('split_mnist', 'split_cifar10', 'split_cifar100')
        n_tasks: Number of tasks
        data_root: Root directory for data
        batch_size: Batch size
        num_workers: Number of data loading workers
        validation_split: Fraction of training data for validation
        **kwargs: Additional dataset-specific arguments

    Returns:
        (train_loaders, val_loaders, test_loaders)
    """
    if dataset_name == "split_mnist":
        return get_split_mnist(
            n_tasks=n_tasks,
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_split=validation_split
        )
    elif dataset_name == "split_cifar10":
        return get_split_cifar10(
            n_tasks=n_tasks,
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_split=validation_split,
            **kwargs
        )
    elif dataset_name == "split_cifar100":
        return get_split_cifar100(
            n_tasks=n_tasks,
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_split=validation_split,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
