"""
Configuration loader for continual learning experiments.

This module provides utilities to load optimal hyperparameters for each
dataset and method based on experimental results.

Usage:
    from configs.config_loader import load_config, get_method_config

    # Load full config for a dataset
    config = load_config('split_cifar10')

    # Get method-specific hyperparameters
    ewc_params = get_method_config('split_cifar10', 'ewc')
    si_params = get_method_config('split_cifar100', 'si')
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(dataset_name: str) -> Dict[str, Any]:
    """
    Load configuration for a specific dataset.

    Args:
        dataset_name: Name of dataset ('split_mnist', 'split_cifar10', 'split_cifar100')

    Returns:
        Dictionary containing all configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If dataset name is invalid
    """
    valid_datasets = ['split_mnist', 'split_cifar10', 'split_cifar100']

    if dataset_name not in valid_datasets:
        raise ValueError(
            f"Invalid dataset '{dataset_name}'. "
            f"Must be one of {valid_datasets}"
        )

    # Get config directory
    config_dir = Path(__file__).parent
    config_file = config_dir / f"{dataset_name}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}"
        )

    # Load YAML
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_method_config(dataset_name: str, method_name: str) -> Dict[str, Any]:
    """
    Get hyperparameters for a specific method on a dataset.

    Args:
        dataset_name: Name of dataset
        method_name: Method name ('finetune', 'ewc', 'si', 'mas', 'lwf')

    Returns:
        Dictionary of method-specific hyperparameters

    Example:
        >>> ewc_params = get_method_config('split_cifar10', 'ewc')
        >>> print(ewc_params['lambda'])
        50.0
    """
    valid_methods = ['finetune', 'ewc', 'si', 'mas', 'lwf']

    if method_name not in valid_methods:
        raise ValueError(
            f"Invalid method '{method_name}'. "
            f"Must be one of {valid_methods}"
        )

    config = load_config(dataset_name)

    if method_name not in config:
        raise KeyError(
            f"Method '{method_name}' not found in config for {dataset_name}"
        )

    return config[method_name]


def get_training_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get common training hyperparameters for a dataset.

    Args:
        dataset_name: Name of dataset

    Returns:
        Dictionary of training hyperparameters (epochs, batch_size, lr, etc.)
    """
    config = load_config(dataset_name)

    training_config = {
        **config.get('training', {}),
        **config.get('optimizer', {}),
    }

    return training_config


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset-specific configuration.

    Args:
        dataset_name: Name of dataset

    Returns:
        Dictionary of dataset parameters (n_tasks, classes_per_task, etc.)
    """
    config = load_config(dataset_name)
    return config.get('dataset', {})


def get_model_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get model architecture configuration.

    Args:
        dataset_name: Name of dataset

    Returns:
        Dictionary of model parameters (architecture, hidden_size, etc.)
    """
    config = load_config(dataset_name)
    return config.get('model', {})


def print_config_summary(dataset_name: str):
    """
    Print a summary of the configuration for a dataset.

    Args:
        dataset_name: Name of dataset
    """
    config = load_config(dataset_name)

    print("="*80)
    print(f"Configuration Summary: {dataset_name.upper()}")
    print("="*80)

    # Dataset info
    print("\nDataset:")
    for key, value in config.get('dataset', {}).items():
        print(f"  {key}: {value}")

    # Model info
    print("\nModel:")
    for key, value in config.get('model', {}).items():
        print(f"  {key}: {value}")

    # Training info
    print("\nTraining:")
    for key, value in config.get('training', {}).items():
        print(f"  {key}: {value}")

    print("\nOptimizer:")
    for key, value in config.get('optimizer', {}).items():
        print(f"  {key}: {value}")

    # Method-specific
    print("\nMethod Hyperparameters:")
    print("-"*80)

    for method in ['ewc', 'si', 'mas', 'lwf']:
        if method in config:
            print(f"\n{method.upper()}:")
            for key, value in config[method].items():
                print(f"  {key}: {value}")

    # Benchmarks
    if 'benchmarks' in config:
        print("\n" + "="*80)
        print("Performance Benchmarks:")
        print("="*80)
        for method, metrics in config['benchmarks'].items():
            print(f"\n{method}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    print("\n" + "="*80)


def get_all_methods_config(dataset_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Get hyperparameters for all methods on a dataset.

    Args:
        dataset_name: Name of dataset

    Returns:
        Dictionary mapping method names to their hyperparameters
    """
    config = load_config(dataset_name)

    methods_config = {}
    for method in ['finetune', 'ewc', 'si', 'mas', 'lwf']:
        if method in config:
            methods_config[method] = config[method]

    return methods_config


def compare_method_configs(method_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Compare hyperparameters for a method across all datasets.

    Args:
        method_name: Method name ('ewc', 'si', 'mas', 'lwf')

    Returns:
        Dictionary mapping dataset names to method hyperparameters
    """
    datasets = ['split_mnist', 'split_cifar10', 'split_cifar100']

    comparison = {}
    for dataset in datasets:
        try:
            comparison[dataset] = get_method_config(dataset, method_name)
        except (FileNotFoundError, KeyError) as e:
            comparison[dataset] = {"error": str(e)}

    return comparison


if __name__ == '__main__':
    """
    Example usage and testing.
    """
    import sys

    # Test all datasets
    for dataset in ['split_mnist', 'split_cifar10', 'split_cifar100']:
        print_config_summary(dataset)
        print("\n\n")

    # Compare EWC hyperparameters across datasets
    print("="*80)
    print("EWC Lambda Comparison Across Datasets")
    print("="*80)
    ewc_comparison = compare_method_configs('ewc')
    for dataset, params in ewc_comparison.items():
        if 'error' not in params:
            print(f"{dataset}: λ = {params['lambda']}")
        else:
            print(f"{dataset}: {params['error']}")

    print("\n")
    print("="*80)
    print("SI Lambda Comparison Across Datasets (Should be consistent)")
    print("="*80)
    si_comparison = compare_method_configs('si')
    for dataset, params in si_comparison.items():
        if 'error' not in params:
            print(f"{dataset}: λ = {params['lambda']}")
        else:
            print(f"{dataset}: {params['error']}")
