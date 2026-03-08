# Configuration Files - Optimal Hyperparameters

This directory contains optimal hyperparameters for all baseline methods on each dataset, based on comprehensive experimental results (December 7, 2025).

## 📁 Files

- **`split_mnist.yaml`** - Configuration for Split MNIST (5 tasks, 2 classes each)
- **`split_cifar10.yaml`** - Configuration for Split CIFAR-10 (5 tasks, 2 classes each)
- **`split_cifar100.yaml`** - Configuration for Split CIFAR-100 (10 tasks, 10 classes each)
- **`config_loader.py`** - Python utilities to load configurations
- **`README.md`** - This file

## 🎯 Key Findings

### EWC Lambda is Dataset-Dependent (CRITICAL!)

```yaml
Split MNIST:    λ = 1000  ✓ Works well
Split CIFAR-10: λ = 50    ✓ Optimal (λ=1000 → FAILS)
Split CIFAR-100: λ = 10   ✓ Optimal (λ=1000 → FAILS)
```

**WARNING**: Using λ=1000 (MNIST optimal) on CIFAR datasets causes catastrophic failure (random guessing)!

### SI is Robust Across All Datasets

```yaml
All datasets: λ = 1.0  ✓ Consistent performance
```

SI requires no hyperparameter tuning - same λ works everywhere!

## 📖 Usage

### Using Python Config Loader

```python
from configs.config_loader import load_config, get_method_config

# Load full configuration
config = load_config('split_cifar10')

# Get method-specific hyperparameters
ewc_params = get_method_config('split_cifar10', 'ewc')
print(ewc_params['lambda'])  # Output: 50.0

si_params = get_method_config('split_cifar100', 'si')
print(si_params['lambda'])   # Output: 1.0

# Get training hyperparameters
training_config = get_training_config('split_cifar10')
print(training_config['lr'])  # Output: 0.01

# Print summary
from configs.config_loader import print_config_summary
print_config_summary('split_cifar10')
```

### Using Configs in Experiment Scripts

```python
import yaml
from configs.config_loader import load_config

# Load config
config = load_config('split_cifar10')

# Extract hyperparameters
dataset_config = config['dataset']
ewc_config = config['ewc']
training_config = config['training']
optimizer_config = config['optimizer']

# Use in your experiments
trainer = EWCTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_tasks=dataset_config['n_tasks'],
    num_classes_per_task=dataset_config['classes_per_task'],
    ewc_lambda=ewc_config['lambda'],      # Optimal value!
    mode=ewc_config['mode'],
    gamma=ewc_config['gamma']
)
```

### Direct YAML Loading

```python
import yaml

with open('configs/split_cifar10.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access parameters
ewc_lambda = config['ewc']['lambda']
si_lambda = config['si']['lambda']
```

## 📊 Configuration Structure

Each YAML file contains:

```yaml
dataset:          # Dataset-specific settings
  name: "split_cifar10"
  n_tasks: 5
  classes_per_task: 2

model:            # Model architecture settings
  architecture: "convnet"
  dropout: 0.0

training:         # Common training hyperparameters
  epochs_per_task: 10
  batch_size: 128

optimizer:        # Optimizer settings
  type: "sgd"
  lr: 0.01
  momentum: 0.9

ewc:             # EWC-specific hyperparameters
  lambda: 50.0   # OPTIMAL for CIFAR-10
  mode: "online"

si:              # SI-specific hyperparameters
  lambda: 1.0
  epsilon: 0.001

mas:             # MAS-specific hyperparameters
  lambda: 1.0
  num_samples: 200

lwf:             # LwF-specific hyperparameters
  lambda: 1.0
  temperature: 2.0

benchmarks:      # Performance benchmarks
  ewc_optimal:
    avg_accuracy: 0.7979
    forgetting: 0.1011
```

## 🎯 Method Selection Guide

### When to Use Each Method

| Scenario | Recommended Method | Config |
|----------|-------------------|--------|
| **Quick baseline** | SI | `si: {lambda: 1.0}` |
| **Best accuracy** | EWC with optimal λ | See dataset-specific config |
| **Lowest forgetting** | SI | `si: {lambda: 1.0}` |
| **Unknown dataset** | SI | Robust default |
| **Production** | SI | No tuning needed |
| **Research** | EWC | Higher potential if tuned |

### Performance Summary

#### Split MNIST (Easy)
- **EWC (λ=1000)**: 99.25% accuracy ⭐ Best
- **SI (λ=1.0)**: 99.17% accuracy, 0.36% forgetting ⭐ Lowest forgetting

#### Split CIFAR-10 (Medium)
- **EWC (λ=50)**: 79.79% accuracy ⭐ Best (+5.9% vs SI)
- **SI (λ=1.0)**: 74.02% accuracy, 4.32% forgetting ⭐ Lowest forgetting

#### Split CIFAR-100 (Hard)
- **EWC (λ=10)**: 55.33% accuracy ⭐ Best (+7.6% vs SI)
- **SI (λ=1.0)**: 47.75% accuracy, 3.71% forgetting ⭐ Lowest forgetting

## ⚠️ Critical Warnings

### EWC Hyperparameter Sensitivity

**DO NOT use λ=1000 on CIFAR datasets!**

```python
# ❌ WRONG - Causes catastrophic failure
ewc_lambda_cifar10 = 1000  # → 50% accuracy (random guessing)
ewc_lambda_cifar100 = 1000 # → 10% accuracy (random guessing)

# ✅ CORRECT - Use optimal values from configs
from configs.config_loader import get_method_config
ewc_params = get_method_config('split_cifar10', 'ewc')
ewc_lambda = ewc_params['lambda']  # → 50
```

### Dataset-Specific Tuning

Always load dataset-specific configs:

```python
# ✅ CORRECT - Load appropriate config
config = load_config('split_cifar10')  # Gets λ=50 for EWC

# ❌ WRONG - Using MNIST config for CIFAR
mnist_config = load_config('split_mnist')
ewc_lambda_for_cifar = mnist_config['ewc']['lambda']  # → 1000 (FAILS!)
```

## 🔬 Experimental Validation

All hyperparameters in these configs are based on:

- ✅ Grid search over λ ∈ [1, 10, 50, 100, 500, 1000, 5000]
- ✅ Multiple seeds (3) for CIFAR-100 statistical significance
- ✅ Consistent training protocol (10 epochs, lr=0.01, SGD)
- ✅ Tested to failure to find upper bounds

See `EXPERIMENTAL_RESULTS_REPORT.md` for full analysis.

## 📈 Future Work

Methods awaiting testing:
- **MAS**: Implemented, estimated λ=1.0 (similar to SI expected)
- **LwF**: Implemented, estimated λ=1.0, T=2.0

Once tested, configs will be updated with experimental results.

## 🚀 Quick Start Examples

### Run EWC with Optimal Hyperparameters

```bash
# CIFAR-10 (uses λ=50 from config)
python experiments/run_ewc.py \
  --dataset split_cifar10 \
  --n_tasks 5 \
  --ewc_lambda 50 \
  --epochs 10

# Or load from config in your script
python -c "
from configs.config_loader import get_method_config
params = get_method_config('split_cifar10', 'ewc')
print(f'Run with: --ewc_lambda {params[\"lambda\"]}')
"
```

### Run All Baselines with Optimal Settings

```bash
# Create a runner script that loads configs automatically
python experiments/run_with_optimal_config.py \
  --dataset split_cifar10 \
  --methods ewc si mas lwf
```

## 📝 Notes

1. **SI is the safest default** - no tuning required
2. **EWC has highest potential** - but needs careful tuning
3. **CIFAR-100 is the best benchmark** - clear method differentiation
4. **Focus development on CIFAR-100** - most room for improvement

For detailed analysis, see:
- `EXPERIMENTAL_RESULTS_REPORT.md` - Full experimental analysis
- `BASELINE_ANALYSIS.md` - Method comparison and architecture analysis
- `PROGRESS.md` - Project progress tracking
