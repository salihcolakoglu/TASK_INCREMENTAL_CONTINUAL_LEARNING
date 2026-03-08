"""Baseline methods for task-incremental continual learning."""

from .finetune import FineTuningTrainer
from .ewc import EWCTrainer
from .si import SynapticIntelligenceTrainer
from .mas import MASTrainer
from .lwf import LwFTrainer
from .negotiation import NegotiationTrainer

# Sigmoid variants
from .sigmoid_finetune import SigmoidFineTuneTrainer
from .sigmoid_ewc import SigmoidEWCTrainer
from .sigmoid_si import SigmoidSITrainer
from .sigmoid_negotiation import SigmoidNegotiationTrainer, HybridNegotiationTrainer

# Walsh Negotiation
from .walsh_negotiation import WalshNegotiationTrainer, WalshMLP, WalshConvNet, WalshConvNetLite

__all__ = [
    'FineTuningTrainer',
    'EWCTrainer',
    'SynapticIntelligenceTrainer',
    'MASTrainer',
    'LwFTrainer',
    'NegotiationTrainer',
    # Sigmoid variants
    'SigmoidFineTuneTrainer',
    'SigmoidEWCTrainer',
    'SigmoidSITrainer',
    'SigmoidNegotiationTrainer',
    'HybridNegotiationTrainer',
    # Walsh Negotiation
    'WalshNegotiationTrainer',
    'WalshMLP',
    'WalshConvNet',
    'WalshConvNetLite',
]
