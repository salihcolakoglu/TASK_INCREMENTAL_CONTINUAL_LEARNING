"""
Cross-Dataset Summary Visualization with Walsh Negotiation

This script creates a comprehensive summary figure comparing Walsh Negotiation
with baseline methods across multiple datasets (MNIST, CIFAR-10, CIFAR-100).

The figure includes:
1. Grouped bar chart: Accuracy comparison across datasets
2. Grouped bar chart: Forgetting comparison across datasets
3. Scatter plot: Accuracy-forgetting trade-off with Walsh highlighted

Date: February 7, 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

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

# Color palette - Walsh Negotiation gets a distinctive color
COLORS = {
    'Walsh Negotiation': '#FF1744',    # Bright red - SOTA method
    'Softmax-EWC': '#1976D2',           # Dark blue
    'Softmax-SI': '#2E7D32',            # Dark green
    'Softmax-FineTune': '#F57C00',      # Dark orange
}

# Marker styles
MARKERS = {
    'Walsh Negotiation': 's',           # Square for Walsh
    'Softmax-EWC': 'o',                 # Circle
    'Softmax-SI': '^',                  # Triangle
    'Softmax-FineTune': 'D',            # Diamond
}


def load_walsh_results(walsh_dir, dataset):
    """Load Walsh Negotiation results for a specific dataset."""
    walsh_path = Path(walsh_dir)

    # Map dataset names to file patterns
    dataset_map = {
        'MNIST': 'split_mnist',
        'CIFAR-10': 'split_cifar10',
        'CIFAR-100': 'split_cifar100'
    }

    pattern = f"{dataset_map[dataset]}_walsh_full_alpha0.5_epochs50_seed4[2-6]_*.json"
    files = list(walsh_path.glob(pattern))

    accuracies = []
    forgettings = []
    bwts = []

    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)

            results = data.get('results', {})
            accuracies.append(results.get('avg_accuracy', 0))
            forgettings.append(results.get('forgetting', 0))
            bwts.append(results.get('backward_transfer', 0))
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")

    return {
        'accuracy': accuracies,
        'forgetting': forgettings,
        'backward_transfer': bwts
    }


def load_baseline_results(baseline_dir, dataset):
    """Load baseline method results for a specific dataset."""
    baseline_path = Path(baseline_dir)

    # Map dataset names to file patterns
    dataset_map = {
        'MNIST': 'split_mnist',
        'CIFAR-10': 'split_cifar10',
        'CIFAR-100': 'split_cifar100'
    }

    pattern = f"{dataset_map[dataset]}_comparison_seed4[2-6]_*.json"
    files = list(baseline_path.glob(pattern))

    aggregated = defaultdict(lambda: {
        'accuracy': [],
        'forgetting': [],
        'backward_transfer': []
    })

    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)

            for result in data.get('results', []):
                method = result['method']
                aggregated[method]['accuracy'].append(result['avg_accuracy'])
                aggregated[method]['forgetting'].append(result['forgetting'])
                aggregated[method]['backward_transfer'].append(result['backward_transfer'])
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")

    return dict(aggregated)


def create_cross_dataset_summary(walsh_dir, baseline_dir, output_path):
    """Create comprehensive cross-dataset summary figure."""

    # Load data for all datasets
    datasets = ['MNIST', 'CIFAR-10', 'CIFAR-100']

    # Combine Walsh and baseline data
    all_data = {}
    for dataset in datasets:
        all_data[dataset] = {}

        # Load Walsh results
        walsh_results = load_walsh_results(walsh_dir, dataset)
        if walsh_results['accuracy']:  # Only add if data exists
            all_data[dataset]['Walsh Negotiation'] = walsh_results

        # Load baseline results
        baseline_results = load_baseline_results(baseline_dir, dataset)
        all_data[dataset].update(baseline_results)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])  # Accuracy comparison
    ax2 = fig.add_subplot(gs[0, 1])  # Forgetting comparison
    ax3 = fig.add_subplot(gs[0, 2])  # Trade-off scatter

    # Subplot 1: Accuracy Comparison
    plot_accuracy_comparison(all_data, ax1)

    # Subplot 2: Forgetting Comparison
    plot_forgetting_comparison(all_data, ax2)

    # Subplot 3: Accuracy-Forgetting Trade-off
    plot_tradeoff_scatter(all_data, ax3)

    # Add panel labels
    for ax, label in zip([ax1, ax2, ax3], ['(a)', '(b)', '(c)']):
        ax.text(-0.1, 1.05, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved cross-dataset summary to: {output_path}")
    plt.close()


def plot_accuracy_comparison(all_data, ax):
    """Plot accuracy comparison across datasets."""
    datasets = list(all_data.keys())

    # Select key methods to show (including Walsh)
    key_methods = ['Walsh Negotiation', 'Softmax-EWC', 'Softmax-SI', 'Softmax-FineTune']

    x = np.arange(len(datasets))
    width = 0.8 / len(key_methods)

    for i, method in enumerate(key_methods):
        means = []
        stds = []

        for dataset in datasets:
            if method in all_data[dataset]:
                acc = all_data[dataset][method]['accuracy']
                means.append(np.mean(acc) * 100)
                stds.append(np.std(acc, ddof=1) * 100 if len(acc) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)

        offset = (i - len(key_methods)/2) * width + width/2
        bars = ax.bar(x + offset, means, width, yerr=stds,
                     label=method, color=COLORS.get(method, 'gray'),
                     capsize=3, alpha=0.9, edgecolor='black', linewidth=0.5)

        # Highlight Walsh Negotiation with pattern
        if method == 'Walsh Negotiation':
            for bar in bars:
                bar.set_linewidth(2.0)

    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy Comparison Across Datasets', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)


def plot_forgetting_comparison(all_data, ax):
    """Plot forgetting comparison across datasets."""
    datasets = list(all_data.keys())

    # Select key methods to show (including Walsh)
    key_methods = ['Walsh Negotiation', 'Softmax-EWC', 'Softmax-SI', 'Softmax-FineTune']

    x = np.arange(len(datasets))
    width = 0.8 / len(key_methods)

    for i, method in enumerate(key_methods):
        means = []
        stds = []

        for dataset in datasets:
            if method in all_data[dataset]:
                forg = all_data[dataset][method]['forgetting']
                means.append(np.mean(forg) * 100)
                stds.append(np.std(forg, ddof=1) * 100 if len(forg) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)

        offset = (i - len(key_methods)/2) * width + width/2
        bars = ax.bar(x + offset, means, width, yerr=stds,
                     label=method, color=COLORS.get(method, 'gray'),
                     capsize=3, alpha=0.9, edgecolor='black', linewidth=0.5)

        # Highlight Walsh Negotiation with pattern
        if method == 'Walsh Negotiation':
            for bar in bars:
                bar.set_linewidth(2.0)

    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Forgetting (%)', fontweight='bold')
    ax.set_title('Forgetting Comparison Across Datasets', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def plot_tradeoff_scatter(all_data, ax):
    """Plot accuracy-forgetting trade-off scatter plot."""
    datasets = list(all_data.keys())
    key_methods = ['Walsh Negotiation', 'Softmax-EWC', 'Softmax-SI', 'Softmax-FineTune']

    # Dataset markers
    dataset_markers = {'MNIST': 'o', 'CIFAR-10': 's', 'CIFAR-100': '^'}
    dataset_colors_shade = {'MNIST': 0.3, 'CIFAR-10': 0.6, 'CIFAR-100': 1.0}

    for method in key_methods:
        for dataset in datasets:
            if method not in all_data[dataset]:
                continue

            acc = np.array(all_data[dataset][method]['accuracy']) * 100
            forg = np.array(all_data[dataset][method]['forgetting']) * 100

            mean_acc = np.mean(acc)
            mean_forg = np.mean(forg)
            std_acc = np.std(acc, ddof=1) if len(acc) > 1 else 0
            std_forg = np.std(forg, ddof=1) if len(forg) > 1 else 0

            # Get base color and adjust brightness for dataset
            base_color = COLORS.get(method, 'gray')

            # Size: Walsh is larger
            size = 300 if method == 'Walsh Negotiation' else 150

            # Plot mean point
            marker = dataset_markers[dataset]
            alpha = 0.9 if method == 'Walsh Negotiation' else 0.7

            ax.scatter(mean_forg, mean_acc, s=size,
                      color=base_color, marker=marker,
                      alpha=alpha, edgecolors='black', linewidth=1.5,
                      label=f'{method} - {dataset}', zorder=3)

            # Plot error bars (std dev)
            ax.errorbar(mean_forg, mean_acc,
                       xerr=std_forg, yerr=std_acc,
                       color=base_color, alpha=0.4, linewidth=1.5,
                       zorder=2)

    ax.set_xlabel('Forgetting (%) - Lower is Better', fontweight='bold')
    ax.set_ylabel('Average Accuracy (%) - Higher is Better', fontweight='bold')
    ax.set_title('Accuracy-Forgetting Trade-off', fontweight='bold')

    # Create custom legend
    # Method legend
    method_handles = []
    for method in key_methods:
        handle = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=COLORS.get(method, 'gray'),
                           markersize=10, label=method,
                           markeredgecolor='black', markeredgewidth=1)
        method_handles.append(handle)

    # Dataset legend
    dataset_handles = []
    for dataset, marker in dataset_markers.items():
        handle = plt.Line2D([0], [0], marker=marker, color='w',
                           markerfacecolor='gray', markersize=8,
                           label=dataset, markeredgecolor='black',
                           markeredgewidth=1)
        dataset_handles.append(handle)

    # Combine legends
    legend1 = ax.legend(handles=method_handles, loc='upper right',
                       title='Method', framealpha=0.9, fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=dataset_handles, loc='lower left',
             title='Dataset', framealpha=0.9, fontsize=8)

    ax.grid(alpha=0.3, linestyle='--')

    # Add "ideal" region annotation (low forgetting, high accuracy)
    ax.axhline(y=85, color='green', linestyle=':', alpha=0.3, linewidth=1.5)
    ax.axvline(x=5, color='green', linestyle=':', alpha=0.3, linewidth=1.5)
    ax.text(0.02, 0.98, 'Ideal Region',
           transform=ax.transAxes, fontsize=8, style='italic',
           verticalalignment='top', bbox=dict(boxstyle='round',
           facecolor='lightgreen', alpha=0.2, edgecolor='green'))


def print_summary_statistics(walsh_dir, baseline_dir):
    """Print summary statistics for all methods and datasets."""
    datasets = ['MNIST', 'CIFAR-10', 'CIFAR-100']

    print("\n" + "="*80)
    print("WALSH NEGOTIATION vs BASELINES - SUMMARY STATISTICS")
    print("="*80)

    for dataset in datasets:
        print(f"\n{dataset}:")
        print("-" * 80)

        # Walsh results
        walsh_results = load_walsh_results(walsh_dir, dataset)
        if walsh_results['accuracy']:
            acc_mean = np.mean(walsh_results['accuracy']) * 100
            acc_std = np.std(walsh_results['accuracy'], ddof=1) * 100
            forg_mean = np.mean(walsh_results['forgetting']) * 100
            forg_std = np.std(walsh_results['forgetting'], ddof=1) * 100
            print(f"  Walsh Negotiation:")
            print(f"    Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
            print(f"    Forgetting: {forg_mean:.2f}% ± {forg_std:.2f}%")

        # Baseline results
        baseline_results = load_baseline_results(baseline_dir, dataset)
        key_methods = ['Softmax-EWC', 'Softmax-SI', 'Softmax-FineTune']

        for method in key_methods:
            if method in baseline_results:
                data = baseline_results[method]
                if data['accuracy']:
                    acc_mean = np.mean(data['accuracy']) * 100
                    acc_std = np.std(data['accuracy'], ddof=1) * 100
                    forg_mean = np.mean(data['forgetting']) * 100
                    forg_std = np.std(data['forgetting'], ddof=1) * 100
                    print(f"  {method}:")
                    print(f"    Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
                    print(f"    Forgetting: {forg_mean:.2f}% ± {forg_std:.2f}%")

    print("\n" + "="*80)


def main():
    """Main function."""
    # Define paths
    project_root = Path(__file__).parent.parent
    walsh_dir = project_root / 'results' / 'walsh_experiments'
    baseline_dir = project_root / 'results' / 'sigmoid_experiments'
    output_dir = project_root / 'figures'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("CROSS-DATASET SUMMARY VISUALIZATION WITH WALSH NEGOTIATION")
    print("="*80)
    print(f"Walsh results directory: {walsh_dir}")
    print(f"Baseline results directory: {baseline_dir}")
    print(f"Output directory: {output_dir}")

    # Print summary statistics
    print_summary_statistics(walsh_dir, baseline_dir)

    # Create visualization
    output_path = output_dir / 'cross_dataset_summary_walsh.png'
    print(f"\nGenerating cross-dataset summary figure...")
    create_cross_dataset_summary(walsh_dir, baseline_dir, output_path)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Output file: {output_path}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
