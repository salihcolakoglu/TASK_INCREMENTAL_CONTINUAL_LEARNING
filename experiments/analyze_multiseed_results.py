"""
Multi-Seed Results Analysis Script

This script aggregates results from multiple seeds and computes:
- Mean accuracy ± standard deviation
- Mean forgetting ± standard deviation
- Statistical significance tests
- Publication-ready tables

Date: December 18, 2025
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from scipy import stats


def load_results(results_dir, pattern):
    """Load all result files matching pattern."""
    results_path = Path(results_dir)
    files = list(results_path.glob(pattern))

    all_results = []
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                all_results.append((file.name, data))
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")

    return all_results


def aggregate_by_method(results, dataset_filter=None):
    """
    Aggregate results by method across multiple seeds.

    Returns:
        dict: {method_name: {'accuracy': [list], 'forgetting': [list], ...}}
    """
    aggregated = defaultdict(lambda: {
        'accuracy': [],
        'forgetting': [],
        'backward_transfer': [],
        'seeds': []
    })

    for filename, data in results:
        # Extract seed from filename
        if 'seed' in filename:
            seed_str = filename.split('seed')[1].split('_')[0]
            try:
                seed = int(seed_str)
            except:
                seed = None
        else:
            seed = None

        # Filter by dataset if specified
        if dataset_filter:
            if dataset_filter not in filename:
                continue

        # Aggregate results
        for result in data.get('results', []):
            method = result['method']
            aggregated[method]['accuracy'].append(result['avg_accuracy'])
            aggregated[method]['forgetting'].append(result['forgetting'])
            aggregated[method]['backward_transfer'].append(result['backward_transfer'])
            if seed is not None:
                aggregated[method]['seeds'].append(seed)

    return dict(aggregated)


def compute_statistics(values):
    """Compute mean, std, and confidence interval."""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0

    # 95% confidence interval
    if len(values) > 1:
        ci = stats.t.interval(0.95, len(values)-1,
                             loc=mean,
                             scale=stats.sem(values))
        ci_range = (ci[1] - ci[0]) / 2
    else:
        ci_range = 0.0

    return {
        'mean': mean,
        'std': std,
        'n': len(values),
        'ci_95': ci_range,
        'min': np.min(values),
        'max': np.max(values)
    }


def print_summary_table(aggregated, metric='accuracy', sort_by='mean'):
    """Print a publication-ready summary table."""
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE - {metric.upper()}")
    print(f"{'='*80}")
    print(f"{'Method':<30s} {'Mean':>10s} {'Std':>10s} {'95% CI':>10s} {'N':>5s} {'Range':>15s}")
    print(f"{'-'*80}")

    # Compute statistics for each method
    stats_data = []
    for method, data in aggregated.items():
        if metric not in data or len(data[metric]) == 0:
            continue

        stats_dict = compute_statistics(data[metric])
        stats_dict['method'] = method
        stats_data.append(stats_dict)

    # Sort by specified metric
    stats_data.sort(key=lambda x: x[sort_by], reverse=(metric == 'accuracy'))

    # Print rows
    for stats_dict in stats_data:
        method = stats_dict['method']
        mean = stats_dict['mean']
        std = stats_dict['std']
        ci = stats_dict['ci_95']
        n = stats_dict['n']
        min_val = stats_dict['min']
        max_val = stats_dict['max']

        range_str = f"[{min_val:.4f}, {max_val:.4f}]"

        print(f"{method:<30s} {mean:>10.4f} {std:>10.4f} {ci:>10.4f} {n:>5d} {range_str:>15s}")

    print(f"{'-'*80}\n")


def print_latex_table(aggregated, metrics=['accuracy', 'forgetting']):
    """Generate LaTeX table code for paper."""
    print(f"\n{'='*80}")
    print(f"LATEX TABLE CODE (for paper)")
    print(f"{'='*80}\n")

    # Header
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Multi-seed validation results (mean $\pm$ std over 5 seeds)}")
    print(r"\label{tab:multiseed}")

    # Column spec
    n_metrics = len(metrics)
    col_spec = "l" + "c" * n_metrics
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")

    # Header row
    header = "Method"
    for metric in metrics:
        header += f" & {metric.capitalize()}"
    header += r" \\"
    print(header)
    print(r"\midrule")

    # Compute statistics for all methods
    method_stats = {}
    for method, data in aggregated.items():
        method_stats[method] = {}
        for metric in metrics:
            if metric in data and len(data[metric]) > 0:
                stats_dict = compute_statistics(data[metric])
                method_stats[method][metric] = stats_dict

    # Sort by first metric (typically accuracy)
    sorted_methods = sorted(method_stats.items(),
                           key=lambda x: x[1].get(metrics[0], {}).get('mean', 0),
                           reverse=True)

    # Print rows
    for method, stats in sorted_methods:
        row = method.replace('_', '\\_')
        for metric in metrics:
            if metric in stats:
                mean = stats[metric]['mean']
                std = stats[metric]['std']
                row += f" & {mean:.2f} $\\pm$ {std:.2f}"
            else:
                row += " & -"
        row += r" \\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def compare_methods(aggregated, method1, method2, metric='accuracy'):
    """Statistical comparison between two methods using t-test."""
    if method1 not in aggregated or method2 not in aggregated:
        print(f"Error: One or both methods not found")
        return

    data1 = aggregated[method1].get(metric, [])
    data2 = aggregated[method2].get(metric, [])

    if len(data1) == 0 or len(data2) == 0:
        print(f"Error: No data for comparison")
        return

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(data1, data2)

    stats1 = compute_statistics(data1)
    stats2 = compute_statistics(data2)

    print(f"\n{'='*80}")
    print(f"STATISTICAL COMPARISON - {metric.upper()}")
    print(f"{'='*80}")
    print(f"{method1}: {stats1['mean']:.4f} ± {stats1['std']:.4f} (n={stats1['n']})")
    print(f"{method2}: {stats2['mean']:.4f} ± {stats2['std']:.4f} (n={stats2['n']})")
    print(f"\nT-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.001:
        print(f"  Significance: *** (p < 0.001) - HIGHLY SIGNIFICANT")
    elif p_value < 0.01:
        print(f"  Significance: ** (p < 0.01) - SIGNIFICANT")
    elif p_value < 0.05:
        print(f"  Significance: * (p < 0.05) - SIGNIFICANT")
    else:
        print(f"  Significance: ns (p >= 0.05) - NOT SIGNIFICANT")

    diff = stats1['mean'] - stats2['mean']
    print(f"\nDifference: {diff:.4f} ({diff/stats2['mean']*100:+.2f}%)")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze multi-seed experiment results'
    )
    parser.add_argument('--results_dir', type=str,
                       default='./results/sigmoid_experiments',
                       help='Directory containing result JSON files')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['split_mnist', 'split_cifar10', 'split_cifar100'],
                       help='Filter by dataset')
    parser.add_argument('--compare', type=str, nargs=2, default=None,
                       metavar=('METHOD1', 'METHOD2'),
                       help='Compare two methods statistically')
    parser.add_argument('--latex', action='store_true',
                       help='Generate LaTeX table code')
    parser.add_argument('--pattern', type=str, default='*.json',
                       help='File pattern to match')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"MULTI-SEED RESULTS ANALYSIS")
    print(f"{'='*80}")
    print(f"Results directory: {args.results_dir}")
    print(f"Dataset filter: {args.dataset or 'None (all datasets)'}")
    print(f"Pattern: {args.pattern}")

    # Load results
    results = load_results(args.results_dir, args.pattern)
    print(f"Loaded {len(results)} result files")

    if len(results) == 0:
        print("Error: No result files found!")
        return

    # Aggregate by method
    aggregated = aggregate_by_method(results, args.dataset)
    print(f"Found {len(aggregated)} unique methods\n")

    # Print summary tables
    print_summary_table(aggregated, metric='accuracy', sort_by='mean')
    print_summary_table(aggregated, metric='forgetting', sort_by='mean')

    # LaTeX table
    if args.latex:
        print_latex_table(aggregated)

    # Statistical comparison
    if args.compare:
        method1, method2 = args.compare
        compare_methods(aggregated, method1, method2, metric='accuracy')
        compare_methods(aggregated, method1, method2, metric='forgetting')

    # Save aggregated results
    output_file = Path(args.results_dir) / 'multiseed_aggregated.json'

    # Convert to serializable format
    serializable_data = {}
    for method, data in aggregated.items():
        serializable_data[method] = {
            'accuracy': compute_statistics(data['accuracy']),
            'forgetting': compute_statistics(data['forgetting']),
            'backward_transfer': compute_statistics(data['backward_transfer']),
            'seeds': data['seeds']
        }

    with open(output_file, 'w') as f:
        json.dump(serializable_data, f, indent=2)

    print(f"Aggregated statistics saved to: {output_file}")

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR PAPER:")
    print(f"{'='*80}")
    print("1. Report results as: mean ± std (n=5)")
    print("2. Use 95% CI for error bars in plots")
    print("3. Report p-values for key comparisons")
    print("4. Highlight statistically significant differences")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
