"""
Default configuration for task-incremental continual learning experiments.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = "mlp"  # Options: mlp, resnet18, resnet32
    hidden_size: int = 256
    num_classes_per_task: int = 10
    dropout: float = 0.0


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset: str = "split_mnist"  # Options: split_mnist, split_cifar10, split_cifar100
    n_tasks: int = 5
    data_root: str = "./data"
    train_transform: Optional[str] = None
    test_transform: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimizer
    optimizer: str = "sgd"  # Options: sgd, adam, adamw
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0

    # Training
    epochs_per_task: int = 10
    batch_size: int = 128
    num_workers: int = 4

    # Scheduler
    use_scheduler: bool = False
    scheduler_type: str = "step"  # Options: step, cosine, plateau
    lr_decay: float = 0.1
    lr_decay_steps: List[int] = None

    # Other
    seed: int = 42
    device: str = "cuda"


@dataclass
class MethodConfig:
    """Method-specific configuration."""
    method: str = "finetune"  # Options: finetune, ewc, si, mas, lwf, joint, mtl

    # EWC parameters
    ewc_lambda: float = 1000.0
    ewc_mode: str = "online"  # Options: online, separate

    # SI parameters
    si_lambda: float = 1.0
    si_eps: float = 0.001

    # MAS parameters
    mas_lambda: float = 1.0

    # LwF parameters
    lwf_alpha: float = 1.0
    lwf_temperature: float = 2.0


@dataclass
class LoggingConfig:
    """Logging and experiment tracking configuration."""
    use_wandb: bool = False
    wandb_project: str = "task-incremental-cl"
    wandb_entity: Optional[str] = None

    use_tensorboard: bool = True
    tensorboard_dir: str = "./results/tensorboard"

    save_checkpoints: bool = True
    checkpoint_dir: str = "./results/checkpoints"

    log_interval: int = 10
    eval_interval: int = 1


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    method: MethodConfig = MethodConfig()
    logging: LoggingConfig = LoggingConfig()

    # Experiment metadata
    experiment_name: str = "default_experiment"
    description: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.training.lr_decay_steps is None:
            self.training.lr_decay_steps = [30, 60, 80]


# Preset configurations for common experiments

def get_mnist_config(method: str = "finetune") -> ExperimentConfig:
    """Get configuration for Split MNIST experiments."""
    config = ExperimentConfig()
    config.data.dataset = "split_mnist"
    config.data.n_tasks = 5
    config.model.architecture = "mlp"
    config.model.hidden_size = 256
    config.training.epochs_per_task = 10
    config.method.method = method
    config.experiment_name = f"split_mnist_{method}"
    return config


def get_cifar10_config(method: str = "finetune") -> ExperimentConfig:
    """Get configuration for Split CIFAR-10 experiments."""
    config = ExperimentConfig()
    config.data.dataset = "split_cifar10"
    config.data.n_tasks = 5
    config.model.architecture = "resnet18"
    config.training.epochs_per_task = 50
    config.training.lr = 0.1
    config.training.use_scheduler = True
    config.method.method = method
    config.experiment_name = f"split_cifar10_{method}"
    return config


def get_cifar100_config(method: str = "finetune") -> ExperimentConfig:
    """Get configuration for Split CIFAR-100 experiments."""
    config = ExperimentConfig()
    config.data.dataset = "split_cifar100"
    config.data.n_tasks = 10
    config.model.architecture = "resnet18"
    config.training.epochs_per_task = 50
    config.training.lr = 0.1
    config.training.use_scheduler = True
    config.method.method = method
    config.experiment_name = f"split_cifar100_{method}"
    return config
