"""
Fine-tuning (FT) baseline for task-incremental continual learning.

This is the simplest baseline that trains sequentially on tasks without
any mechanism to prevent catastrophic forgetting. It serves as a reference
point to measure how much forgetting occurs.

Reference:
- This is the "naive" sequential learning approach
- Expected to show significant catastrophic forgetting
"""

import os
import sys

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.base_trainer import BaseTrainer


class FineTuningTrainer(BaseTrainer):
    """
    Fine-tuning baseline trainer.

    This simply trains on each task sequentially without any special
    mechanisms to prevent catastrophic forgetting. It inherits all
    functionality from BaseTrainer without modifications.

    This baseline is useful for:
    1. Measuring the extent of catastrophic forgetting
    2. Serving as a lower bound for continual learning methods
    3. Validating that the experimental setup works correctly
    """

    def __init__(self, *args, **kwargs):
        """Initialize Fine-tuning trainer (same as BaseTrainer)."""
        super().__init__(*args, **kwargs)

    # No need to override any methods - fine-tuning is just
    # sequential training without any forgetting mitigation
