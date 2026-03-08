# Task-Incremental Continual Learning

Research project implementing an exemplar-free, single-model, non-pretrained method for task-incremental continual learning.

## Project Structure

```
├── data/                  # Datasets (MNIST, CIFAR-10, CIFAR-100)
├── src/                   # Source code
│   ├── baselines/        # Baseline method implementations
│   ├── models/           # Neural network architectures
│   ├── utils/            # Utility functions
│   └── My_method/        # Your proposed method
├── experiments/          # Experiment configurations and scripts
├── results/              # Experimental results, logs, checkpoints
├── paper/                # Paper drafts and figures
└── configs/              # Configuration files

## Environment Setup

This project uses the shared ContinualML conda environment with:
- Python 3.10.19
- PyTorch 2.5.1 (CUDA 12.1)
- Avalanche 0.6.0
- GPU: NVIDIA RTX 3090 (24GB)

Dependencies are listed in `requirements.txt`.

## Quick Start

1. Test your setup:
```bash
python test_setup.py
```

2. Run a simple baseline (Fine-tuning on Split MNIST):
```bash
python experiments/run_baseline.py --method finetune --dataset mnist
```

## Datasets

Pre-downloaded datasets in `data/`:
- Split MNIST (5 or 10 tasks)
- Split CIFAR-10 (5 tasks)
- Split CIFAR-100 (10 or 20 tasks)

## Baseline Methods

Planned comparisons (all exemplar-free):
1. Fine-tuning (FT)
2. Joint Training (upper bound)
3. Elastic Weight Consolidation (EWC)
4. Synaptic Intelligence (SI)
5. Memory Aware Synapses (MAS)
6. Learning without Forgetting (LwF)
7. Multi-Task Learning (MTL)
8. Walsh Negotiation (SOTA - current best performing method)

## Walsh Negotiation Experiments

Walsh Negotiation is our current SOTA method, achieving excellent performance across all benchmarks with minimal forgetting.

### Running Experiments

```bash
# Run Walsh Negotiation on Split MNIST
python experiments/run_walsh_negotiation.py --dataset split_mnist --n_tasks 5 --epochs 50 --alpha 0.5 --code_dim 128

# Run Walsh Negotiation on Split CIFAR-10
python experiments/run_walsh_negotiation.py --dataset split_cifar10 --n_tasks 5 --epochs 50 --alpha 0.5 --code_dim 128

# Run Walsh Negotiation on Split CIFAR-100
python experiments/run_walsh_negotiation.py --dataset split_cifar100 --n_tasks 10 --epochs 50 --alpha 0.5 --code_dim 128

# Multi-seed validation (seeds 42-46)
for seed in 42 43 44 45 46; do
  python experiments/run_walsh_negotiation.py --dataset split_cifar100 --epochs 50 --seed $seed
done
```

### Key Hyperparameters

- **alpha (α₀)**: 0.5 - Negotiation rate controlling weight update balance
- **code_dim**: 128 - Walsh code dimension for task representation
- **epochs**: 50 - Training epochs per task
- **architecture**: ResNet18 for CIFAR-10/100, simple CNN for MNIST

### Results Summary

Walsh Negotiation achieves state-of-the-art performance with minimal forgetting:

| Dataset | Average Accuracy | Forgetting |
|---------|------------------|------------|
| Split MNIST | 98.75% ± 0.07% | 0.10% ± 0.07% |
| Split CIFAR-10 | 90.11% ± 0.78% | 1.71% ± 0.44% |
| Split CIFAR-100 | 66.71% ± 0.84% | 2.94% ± 1.05% |

Results are averaged over 5 random seeds (42-46). Walsh Negotiation demonstrates:
- Near-perfect performance on MNIST with negligible forgetting
- Strong performance on CIFAR-10 with minimal catastrophic forgetting
- Best-in-class results on challenging CIFAR-100 benchmark

## Current Status

Walsh Negotiation achieves state-of-the-art results on all benchmarks, demonstrating the effectiveness of negotiation-based continual learning. See `Project Plan.md` for detailed implementation roadmap.

## Experiment Tracking

Using Weights & Biases and TensorBoard for experiment tracking.
- W&B Project: [TBD]
- TensorBoard logs: `results/tensorboard/`

## License

[TBD]
