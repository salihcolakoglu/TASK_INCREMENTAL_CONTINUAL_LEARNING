"""Neural network models for continual learning."""

from .networks import SimpleMLP, SimpleConvNet, get_model

__all__ = ['SimpleMLP', 'SimpleConvNet', 'get_model']
