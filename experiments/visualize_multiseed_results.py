"""
Multi-Seed Results Visualization Script

Generates publication-ready figures:
1. Accuracy comparison across datasets
2. Forgetting comparison across datasets
3. Accuracy-Forgetting trade-off plots
4. Box plots showing variance across seeds

Date: December 19, 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Color palette
COLORS = {
    'Softmax-SI': '#2E7D32',       # Dark green
    'Softmax-EWC': '#1976D2',      # Dark blue
    'Softmax-FineTune': '#F57C00', # Dark orange
    'Sigmoid-FineTune': '#C2185B', # Dark pink
    'Sigmoid-EWC': '#7B1FA2',      # Dark purple
    'Sigmoid-SI': '#D32F2F',       # Dark red
    'Softmax-Negotiation': '#388E3C', # Medium green
    'Sigmoid-Negotiation': '#E64A19', # Medium red
    'Hybrid-Negotiation': '#FBC02D',  # Yellow
}


def load_multiseed_data(results_dir, dataset, pattern):
    """Load and aggregate multi-seed results."""
    results_path = Path(results_dir)
    # Construct pattern correctly
    full_pattern = f"{dataset}_comparison_{pattern}"
    files = list(results_path.glob(full_pattern))

    aggregated = defaultdict(lambda: {
        'accuracy': [],
        'forgetting': [],
        'seeds': []
    })

    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)

            # Extract seed from filename
            seed_str = file.name.split('seed')[1].split('_')[0]
            seed = int(seed_str)

            # Aggregate results
            for result in data.get('results', []):
                method = result['method']
                aggregated[method]['accuracy'].append(result['avg_accuracy'])
                aggregated[method]['forgetting'].append(result['forgetting'])
                aggregated[method]['seeds'].append(seed)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")

    return dict(aggregated)


def plot_accuracy_comparison(datasets_data, save_path):
    """Plot accuracy comparison across datasets."""
    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = list(datasets_data.keys())
    methods = set()
    for data in datasets_data.values():
        methods.update(data.keys())

    # Filter out broken methods
    methods = [m for m in methods if m != 'Sigmoid-SI']
    methods = sorted(methods)

    x = np.arange(len(datasets))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        means = []
        stds = []

        for dataset in datasets:
            if method in datasets_data[dataset]:
                acc = datasets_data[dataset][method]['accuracy']
                means.append(np.mean(acc) * 100)
                stds.append(np.std(acc, ddof=1) * 100 if len(acc) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)

        offset = (i - len(methods)/2) * width + width/2
        ax.bar(x + offset, means, width, yerr=stds,
               label=method, color=COLORS.get(method, 'gray'),
               capsize=3, alpha=0.8)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Multi-Seed Accuracy Comparison (Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved accuracy comparison to {save_path}")
    plt.close()


def plot_forgetting_comparison(datasets_data, save_path):
    """Plot forgetting comparison across datasets."""
    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = list(datasets_data.keys())
    methods = set()
    for data in datasets_data.values():
        methods.update(data.keys())

    # Filter out broken methods
    methods = [m for m in methods if m != 'Sigmoid-SI']
    methods = sorted(methods)

    x = np.arange(len(datasets))
    width = 0.8 / len(methods)

    for i, method in enumerate(methods):
        means = []
        stds = []

        for dataset in datasets:
            if method in datasets_data[dataset]:
                forg = datasets_data[dataset][method]['forgetting']
                means.append(np.mean(forg) * 100)
                stds.append(np.std(forg, ddof=1) * 100 if len(forg) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)

        offset = (i - len(methods)/2) * width + width/2
        ax.bar(x + offset, means, width, yerr=stds,
               label=method, color=COLORS.get(method, 'gray'),
               capsize=3, alpha=0.8)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Forgetting (%)')
    ax.set_title('Multi-Seed Forgetting Comparison (Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved forgetting comparison to {save_path}")
    plt.close()


def plot_accuracy_forgetting_tradeoff(dataset_data, dataset_name, save_path):
    """Plot accuracy-forgetting trade-off for a single dataset."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for method, data in dataset_data.items():
        if method == 'Sigmoid-SI':  # Skip broken method
            continue

        acc = np.array(data['accuracy']) * 100
        forg = np.array(data['forgetting']) * 100

        mean_acc = np.mean(acc)
        mean_forg = np.mean(forg)
        std_acc = np.std(acc, ddof=1) if len(acc) > 1 else 0
        std_forg = np.std(forg, ddof=1) if len(forg) > 1 else 0

        # Plot mean point
        ax.scatter(mean_forg, mean_acc, s=200,
                  color=COLORS.get(method, 'gray'),
                  alpha=0.7, edgecolors='black', linewidth=1.5,
                  label=method, zorder=3)

        # Plot error bars (95% CI)
        ax.errorbar(mean_forg, mean_acc,
                   xerr=1.96*std_forg/np.sqrt(len(forg)),
                   yerr=1.96*std_acc/np.sqrt(len(acc)),
                   color=COLORS.get(method, 'gray'),
                   alpha=0.5, linewidth=2, zorder=2)

    ax.set_xlabel('Forgetting (%)')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title(f'Accuracy-Forgetting Trade-off: {dataset_name}\n(Error bars: 95% CI)')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    # Add "ideal" region annotation
    ax.axhline(y=ax.get_ylim()[1]*0.9, color='green', linestyle='--', alpha=0.3)
    ax.axvline(x=ax.get_xlim()[1]*0.1, color='green', linestyle='--', alpha=0.3)
    ax.text(0.02, 0.98, 'Ideal Region\n(High Acc, Low Forg)',
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round',
           facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved trade-off plot to {save_path}")
    plt.close()


def plot_variance_comparison(datasets_data, save_path):
    """Plot variance (std) comparison across methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    methods = set()
    for data in datasets_data.values():
        methods.update(data.keys())
    methods = [m for m in sorted(methods) if m != 'Sigmoid-SI']

    acc_stds = {method: [] for method in methods}
    forg_stds = {method: [] for method in methods}
    dataset_names = []

    for dataset, data in datasets_data.items():
        dataset_names.append(dataset)
        for method in methods:
            if method in data:
                acc = data[method]['accuracy']
                forg = data[method]['forgetting']
                acc_stds[method].append(np.std(acc, ddof=1) * 100 if len(acc) > 1 else 0)
                forg_stds[method].append(np.std(forg, ddof=1) * 100 if len(forg) > 1 else 0)
            else:
                acc_stds[method].append(0)
                forg_stds[method].append(0)

    x = np.arange(len(dataset_names))
    width = 0.8 / len(methods)

    # Accuracy variance
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2) * width + width/2
        ax1.bar(x + offset, acc_stds[method], width,
               label=method, color=COLORS.get(method, 'gray'), alpha=0.8)

    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Standard Deviation (%)')
    ax1.set_title('Accuracy Variance Across Seeds')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Forgetting variance
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2) * width + width/2
        ax2.bar(x + offset, forg_stds[method], width,
               label=method, color=COLORS.get(method, 'gray'), alpha=0.8)

    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Standard Deviation (%)')
    ax2.set_title('Forgetting Variance Across Seeds')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_names)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved variance comparison to {save_path}")
    plt.close()


def plot_method_ranking(datasets_data, save_path):
    """Plot method ranking across datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    datasets = list(datasets_data.keys())
    methods = set()
    for data in datasets_data.values():
        methods.update(data.keys())
    methods = [m for m in sorted(methods) if m != 'Sigmoid-SI']

    # Accuracy ranking
    acc_ranks = {method: [] for method in methods}
    for dataset, data in datasets_data.items():
        method_accs = []
        for method in methods:
            if method in data:
                acc = np.mean(data[method]['accuracy'])
                method_accs.append((method, acc))
            else:
                method_accs.append((method, 0))

        # Rank by accuracy (higher is better)
        method_accs.sort(key=lambda x: x[1], reverse=True)
        for rank, (method, _) in enumerate(method_accs, 1):
            acc_ranks[method].append(rank)

    # Forgetting ranking
    forg_ranks = {method: [] for method in methods}
    for dataset, data in datasets_data.items():
        method_forgs = []
        for method in methods:
            if method in data:
                forg = np.mean(data[method]['forgetting'])
                method_forgs.append((method, forg))
            else:
                method_forgs.append((method, 1.0))

        # Rank by forgetting (lower is better)
        method_forgs.sort(key=lambda x: x[1])
        for rank, (method, _) in enumerate(method_forgs, 1):
            forg_ranks[method].append(rank)

    x = np.arange(len(datasets))
    width = 0.8 / len(methods)

    # Plot accuracy ranks
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2) * width + width/2
        ax1.bar(x + offset, acc_ranks[method], width,
               label=method, color=COLORS.get(method, 'gray'), alpha=0.8)

    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Rank (1 = Best)')
    ax1.set_title('Accuracy Ranking Across Datasets')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylim(len(methods) + 0.5, 0.5)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Plot forgetting ranks
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2) * width + width/2
        ax2.bar(x + offset, forg_ranks[method], width,
               label=method, color=COLORS.get(method, 'gray'), alpha=0.8)

    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Rank (1 = Best)')
    ax2.set_title('Forgetting Ranking Across Datasets')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.set_ylim(len(methods) + 0.5, 0.5)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved ranking plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready visualizations from multi-seed results'
    )
    parser.add_argument('--results_dir', type=str,
                       default='./results/sigmoid_experiments',
                       help='Directory containing result JSON files')
    parser.add_argument('--pattern', type=str, default='*seed[4-4][2-6]_2025*.json',
                       help='File pattern for multi-seed results')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Directory to save figures')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"MULTI-SEED VISUALIZATION GENERATOR")
    print(f"{'='*80}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Pattern: {args.pattern}\n")

    # Load data for all datasets
    datasets = {
        'MNIST': load_multiseed_data(args.results_dir, 'split_mnist', args.pattern),
        'CIFAR-10': load_multiseed_data(args.results_dir, 'split_cifar10', args.pattern),
        'CIFAR-100': load_multiseed_data(args.results_dir, 'split_cifar100', args.pattern),
    }

    print("Generating figures...\n")

    # 1. Accuracy comparison
    plot_accuracy_comparison(datasets, output_dir / 'accuracy_comparison.png')

    # 2. Forgetting comparison
    plot_forgetting_comparison(datasets, output_dir / 'forgetting_comparison.png')

    # 3. Trade-off plots for each dataset
    for dataset_name, dataset_data in datasets.items():
        safe_name = dataset_name.replace('-', '_').lower()
        plot_accuracy_forgetting_tradeoff(
            dataset_data, dataset_name,
            output_dir / f'tradeoff_{safe_name}.png'
        )

    # 4. Variance comparison
    plot_variance_comparison(datasets, output_dir / 'variance_comparison.png')

    # 5. Method ranking
    plot_method_ranking(datasets, output_dir / 'method_ranking.png')

    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"All figures saved to: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"  - accuracy_comparison.png")
    print(f"  - forgetting_comparison.png")
    print(f"  - tradeoff_mnist.png")
    print(f"  - tradeoff_cifar_10.png")
    print(f"  - tradeoff_cifar_100.png")
    print(f"  - variance_comparison.png")
    print(f"  - method_ranking.png")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
