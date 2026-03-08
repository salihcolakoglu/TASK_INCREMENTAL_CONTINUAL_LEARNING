# Reproducibility Package Manifest

**Package**: `reproducibility_package.tar.gz`
**Size**: 146 KB
**Created**: 2026-02-09
**Total Files**: 100

## Contents Overview

This archive contains all essential code, configurations, and experimental results needed to reproduce the Walsh Negotiation experiments for task-incremental continual learning.

## Directory Structure

```
reproducibility_package/
├── src/                           # Source code (19 Python files)
│   ├── baselines/                 # Baseline method implementations
│   │   ├── __init__.py
│   │   ├── ewc.py                 # Elastic Weight Consolidation
│   │   ├── finetune.py            # Naive fine-tuning baseline
│   │   ├── lwf.py                 # Learning without Forgetting
│   │   ├── mas.py                 # Memory Aware Synapses
│   │   ├── negotiation.py         # Standard negotiation method
│   │   ├── si.py                  # Synaptic Intelligence
│   │   ├── sigmoid_ewc.py         # Sigmoid-based EWC variant
│   │   ├── sigmoid_finetune.py    # Sigmoid-based fine-tuning
│   │   ├── sigmoid_negotiation.py # Sigmoid-based negotiation
│   │   ├── sigmoid_si.py          # Sigmoid-based SI
│   │   └── walsh_negotiation.py   # Walsh Negotiation (main method)
│   ├── models/                    # Neural network architectures
│   │   ├── __init__.py
│   │   ├── networks.py            # ConvNet, MLP, Walsh-based models
│   │   └── networks (copy).py     # Backup copy
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── activations.py         # Custom activation functions
│       ├── base_trainer.py        # Base trainer class
│       ├── data_utils.py          # Dataset loading utilities
│       └── metrics.py             # Evaluation metrics
│
├── experiments/                   # Experiment scripts (23 Python files)
│   ├── run_walsh_negotiation.py   # Main script for Walsh experiments
│   ├── run_ewc.py                 # Run EWC baseline
│   ├── run_finetune.py            # Run fine-tuning baseline
│   ├── run_si.py                  # Run SI baseline
│   ├── run_mas.py                 # Run MAS baseline
│   ├── run_lwf.py                 # Run LwF baseline
│   ├── run_negotiation.py         # Run standard negotiation
│   ├── run_sigmoid_*.py           # Sigmoid-based experiments
│   ├── alpha_search_negotiation.py # Hyperparameter search
│   ├── analyze_multiseed_results.py # Result analysis
│   ├── visualize_multiseed_results.py # Result visualization
│   ├── visualize_walsh_learning_curves.py # Learning curve plots
│   ├── visualize_cross_dataset_summary.py # Cross-dataset comparison
│   ├── compare_baselines.py       # Baseline comparison
│   ├── ewc_hyperparam_search.py   # EWC hyperparameter search
│   ├── run_all_experiments.py     # Batch experiment runner
│   ├── run_with_config.py         # Config-based experiments
│   ├── run_untested_baselines.py  # Additional baselines
│   └── quick_sigmoid_test.py      # Quick test script
│
├── configs/                       # Configuration files
│   ├── README.md                  # Configuration documentation
│   ├── config_loader.py           # Config loading utilities
│   ├── default_config.py          # Default hyperparameters
│   ├── split_mnist.yaml           # MNIST experiment config
│   ├── split_cifar10.yaml         # CIFAR-10 experiment config
│   └── split_cifar100.yaml        # CIFAR-100 experiment config
│
├── results/walsh_experiments/     # Experimental results (17 JSON files)
│   ├── Split MNIST (5 seeds):
│   │   ├── split_mnist_walsh_full_alpha0.5_epochs50_seed42_*.json
│   │   ├── split_mnist_walsh_full_alpha0.5_epochs50_seed43_*.json
│   │   ├── split_mnist_walsh_full_alpha0.5_epochs50_seed44_*.json
│   │   ├── split_mnist_walsh_full_alpha0.5_epochs50_seed45_*.json
│   │   └── split_mnist_walsh_full_alpha0.5_epochs50_seed46_*.json
│   ├── Split CIFAR-10 (5 seeds):
│   │   ├── split_cifar10_walsh_full_alpha0.5_epochs50_seed42_*.json
│   │   ├── split_cifar10_walsh_full_alpha0.5_epochs50_seed43_*.json
│   │   ├── split_cifar10_walsh_full_alpha0.5_epochs50_seed44_*.json
│   │   ├── split_cifar10_walsh_full_alpha0.5_epochs50_seed45_*.json
│   │   └── split_cifar10_walsh_full_alpha0.5_epochs50_seed46_*.json
│   └── Split CIFAR-100 (7 files, including test runs):
│       ├── split_cifar100_walsh_full_alpha0.5_epochs1_seed42_*.json (test)
│       ├── split_cifar100_walsh_full_alpha0.5_epochs50_seed42_*.json
│       ├── split_cifar100_walsh_full_alpha0.5_epochs50_seed43_*.json (2 versions)
│       ├── split_cifar100_walsh_full_alpha0.5_epochs50_seed44_*.json
│       ├── split_cifar100_walsh_full_alpha0.5_epochs50_seed45_*.json
│       └── split_cifar100_walsh_full_alpha0.5_epochs50_seed46_*.json
│
├── README.md                      # Project overview
├── REPRODUCE.md                   # Detailed reproduction guide
└── requirements.txt               # Python dependencies

```

## File Count by Category

| Category | Count | Description |
|----------|-------|-------------|
| Source Code (src/) | 19 | Core implementation files |
| Experiments | 23 | Experiment runner scripts |
| Configs | 6 | Configuration files |
| Results | 17 | Walsh experiment JSON results |
| Documentation | 3 | README, REPRODUCE, requirements |
| **Total** | **68** | **Source files (excluding __pycache__)** |

## Key Files

### Essential for Reproduction

1. **`REPRODUCE.md`** - Step-by-step reproduction instructions
2. **`requirements.txt`** - All Python dependencies
3. **`experiments/run_walsh_negotiation.py`** - Main experiment script
4. **`src/baselines/walsh_negotiation.py`** - Walsh Negotiation implementation
5. **`src/models/networks.py`** - Neural network architectures

### Baseline Implementations

The package includes 12 baseline methods:
- Fine-tuning (FT)
- Elastic Weight Consolidation (EWC)
- Synaptic Intelligence (SI)
- Memory Aware Synapses (MAS)
- Learning without Forgetting (LwF)
- Standard Negotiation
- Sigmoid-based variants (EWC, SI, Finetune, Negotiation)
- **Walsh Negotiation** (primary contribution)

### Configuration Files

Pre-configured YAML files for each benchmark:
- `split_mnist.yaml`: 5 tasks, 2 classes per task
- `split_cifar10.yaml`: 5 tasks, 2 classes per task
- `split_cifar100.yaml`: 10 tasks, 10 classes per task

### Experimental Results

Complete results for Walsh Negotiation method:
- **15 successful experiments** (3 datasets × 5 seeds)
- **2 additional test runs** (CIFAR-100 with 1 epoch, CIFAR-100 re-run)
- Seeds: 42, 43, 44, 45, 46
- Alpha: 0.5 (initial negotiation rate)
- Epochs: 50 per task

Each JSON file contains:
- Full accuracy matrix
- Task-wise accuracies
- Forgetting metrics
- Backward transfer
- Negotiation rate history
- Walsh code assignments
- All hyperparameters

## Dataset Requirements (Not Included)

The following datasets will be automatically downloaded on first run:
- MNIST: ~50 MB
- CIFAR-10: ~170 MB
- CIFAR-100: ~170 MB

Total dataset size: ~390 MB

## Usage

### Extract the Archive

```bash
tar -xzf reproducibility_package.tar.gz
cd reproducibility_package/
```

### Set Up Environment

```bash
# Create conda environment
conda create -n continual_learning python=3.10 -y
conda activate continual_learning

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Single experiment
python experiments/run_walsh_negotiation.py --dataset split_mnist --seed 42

# All experiments (multi-seed)
# See REPRODUCE.md for detailed instructions
```

### Verify Results

Compare your results with the provided JSON files in `results/walsh_experiments/`.

## Notes

- **Python cache files** (`__pycache__`) are included but can be safely deleted
- **Backup files** (e.g., `networks (copy).py`) are included for reference
- **Multiple result files** for CIFAR-100 seed 43 indicate re-runs (use latest timestamp)
- **Test run** (epochs=1) demonstrates script functionality with quick execution

## Checksums

To verify package integrity:

```bash
# MD5 checksum
md5sum reproducibility_package.tar.gz

# SHA256 checksum
sha256sum reproducibility_package.tar.gz
```

## Version Information

- **PyTorch**: 2.5.1
- **CUDA**: 12.1
- **Avalanche**: 0.6.0
- **Python**: 3.10.19

See `requirements.txt` for complete dependency list.

## License

See project README.md for license information.

## Support

For reproduction issues:
1. Check `REPRODUCE.md` troubleshooting section
2. Verify all dependencies are installed correctly
3. Compare results with provided JSON files
4. Review paper for algorithmic details

## Archive Creation

```bash
# Command used to create this archive
tar -czvf reproducibility_package.tar.gz \
  src/ \
  experiments/ \
  configs/ \
  requirements.txt \
  README.md \
  REPRODUCE.md \
  results/walsh_experiments/*.json
```

**Date**: 2026-02-09
**Working Directory**: `/home/nuri/miniconda3/envs/ContinualML/projects/TASK_INCREMENTAL_CONTINUAL_LEARNING`
