"""
Activation function utilities for continual learning experiments.
Provides both softmax and sigmoid-based classification approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationType:
    """Enum-like class for activation types."""
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"


def get_output_activation(activation_type, dim=-1):
    """
    Get the appropriate output activation function.

    Args:
        activation_type: "softmax" or "sigmoid"
        dim: dimension for softmax (ignored for sigmoid)

    Returns:
        Callable activation function
    """
    if activation_type == ActivationType.SOFTMAX:
        return lambda x: F.softmax(x, dim=dim)
    elif activation_type == ActivationType.SIGMOID:
        return torch.sigmoid
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


def compute_loss(logits, labels, num_classes, activation_type, label_smoothing=0.0):
    """
    Compute loss based on activation type.

    Args:
        logits: Raw model outputs [batch_size, num_classes]
        labels: Integer class labels [batch_size]
        num_classes: Number of classes
        activation_type: "softmax" or "sigmoid"
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        Scalar loss value
    """
    if activation_type == ActivationType.SOFTMAX:
        # Standard cross-entropy loss
        if label_smoothing > 0:
            # Manual label smoothing
            y_onehot = F.one_hot(labels, num_classes).float()
            y_smooth = y_onehot * (1 - label_smoothing) + label_smoothing / num_classes
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(y_smooth * log_probs).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(logits, labels)

    elif activation_type == ActivationType.SIGMOID:
        # Binary cross-entropy per class
        y_onehot = F.one_hot(labels, num_classes).float()
        if label_smoothing > 0:
            y_onehot = y_onehot * (1 - label_smoothing) + label_smoothing / num_classes
        loss = F.binary_cross_entropy_with_logits(logits, y_onehot)

    else:
        raise ValueError(f"Unknown activation type: {activation_type}")

    return loss


def compute_soft_target_loss(logits, soft_targets, activation_type, temperature=1.0):
    """
    Compute loss with soft targets (for negotiation and distillation).

    Args:
        logits: Raw model outputs [batch_size, num_classes]
        soft_targets: Soft probability targets [batch_size, num_classes]
        activation_type: "softmax" or "sigmoid"
        temperature: Temperature for scaling (used in softmax)

    Returns:
        Scalar loss value
    """
    if activation_type == ActivationType.SOFTMAX:
        # KL divergence with temperature
        log_probs = F.log_softmax(logits / temperature, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        # Scale by T^2 for gradient magnitude consistency
        loss = loss * (temperature ** 2)

    elif activation_type == ActivationType.SIGMOID:
        # Binary cross-entropy with soft targets
        # Soft targets should already be in [0, 1] range
        loss = F.binary_cross_entropy_with_logits(logits, soft_targets)

    else:
        raise ValueError(f"Unknown activation type: {activation_type}")

    return loss


def get_predictions(logits, activation_type):
    """
    Get class predictions from logits.

    Args:
        logits: Raw model outputs [batch_size, num_classes]
        activation_type: "softmax" or "sigmoid"

    Returns:
        Predicted class indices [batch_size]
    """
    # For both softmax and sigmoid, argmax of logits gives the prediction
    # (sigmoid is monotonic, so argmax(sigmoid(x)) == argmax(x))
    return logits.argmax(dim=1)


def get_probabilities(logits, activation_type):
    """
    Get probability outputs from logits.

    Args:
        logits: Raw model outputs [batch_size, num_classes]
        activation_type: "softmax" or "sigmoid"

    Returns:
        Probabilities [batch_size, num_classes]
    """
    if activation_type == ActivationType.SOFTMAX:
        return F.softmax(logits, dim=1)
    elif activation_type == ActivationType.SIGMOID:
        return torch.sigmoid(logits)
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


# Convenience functions for common operations

def softmax_loss(logits, labels, num_classes, label_smoothing=0.0):
    """Shorthand for softmax-based loss."""
    return compute_loss(logits, labels, num_classes, ActivationType.SOFTMAX, label_smoothing)


def sigmoid_loss(logits, labels, num_classes, label_smoothing=0.0):
    """Shorthand for sigmoid-based loss."""
    return compute_loss(logits, labels, num_classes, ActivationType.SIGMOID, label_smoothing)
