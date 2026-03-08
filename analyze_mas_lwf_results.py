"""
Analyze MAS and LwF hyperparameter search results.

This script reads all the individual result JSON files and finds the best
configurations for each method and dataset.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from tabulate import tabulate


def analyze_results(results_dir):
    """Analyze all results and find best configurations."""

    # Load all result files
    all_results = []
    results_path = Path(results_dir)

    for json_file in results_path.glob("*.json"):
        with open(json_file, 'r') as f:
            result = json.load(f)
            all_results.append(result)

    print(f"Loaded {len(all_results)} results\n")

    # Group by method and dataset
    grouped = defaultdict(list)
    for result in all_results:
        key = (result['method'], result['dataset'])
        grouped[key].append(result)

    # Analyze each method-dataset combination
    best_configs = {}

    for (method, dataset), results in sorted(grouped.items()):
        print("="*80)
        print(f"{method.upper()} - {dataset.upper()}")
        print("="*80)
        print(f"Total configurations tested: {len(results)}\n")

        # Sort by accuracy (descending)
        results_sorted_acc = sorted(results, key=lambda x: x['avg_accuracy'], reverse=True)

        # Sort by forgetting (ascending - lower is better)
        results_sorted_forget = sorted(results, key=lambda x: x['forgetting'])

        # Display top 10 by accuracy
        print("Top 10 configurations by ACCURACY:")
        print("-"*80)

        table_data = []
        for i, r in enumerate(results_sorted_acc[:10], 1):
            if method == 'mas':
                config = f"λ={r['mas_lambda']}, N={r['num_samples']}"
            else:  # lwf
                config = f"λ={r['lwf_lambda']}, T={r['temperature']}"

            table_data.append([
                i,
                config,
                f"{r['avg_accuracy']:.4f}",
                f"{r['forgetting']:.4f}",
                f"{r['backward_transfer']:.4f}"
            ])

        headers = ['Rank', 'Configuration', 'Accuracy', 'Forgetting', 'BWT']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()

        # Display top 10 by forgetting (lowest)
        print("Top 10 configurations by FORGETTING (lowest):")
        print("-"*80)

        table_data = []
        for i, r in enumerate(results_sorted_forget[:10], 1):
            if method == 'mas':
                config = f"λ={r['mas_lambda']}, N={r['num_samples']}"
            else:  # lwf
                config = f"λ={r['lwf_lambda']}, T={r['temperature']}"

            table_data.append([
                i,
                config,
                f"{r['avg_accuracy']:.4f}",
                f"{r['forgetting']:.4f}",
                f"{r['backward_transfer']:.4f}"
            ])

        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()

        # Best overall
        best_acc = results_sorted_acc[0]
        best_forget = results_sorted_forget[0]

        print("⭐ BEST BY ACCURACY:")
        if method == 'mas':
            print(f"   λ={best_acc['mas_lambda']}, num_samples={best_acc['num_samples']}")
        else:
            print(f"   λ={best_acc['lwf_lambda']}, temperature={best_acc['temperature']}")
        print(f"   Accuracy: {best_acc['avg_accuracy']:.4f}")
        print(f"   Forgetting: {best_acc['forgetting']:.4f}")
        print()

        print("⭐ BEST BY FORGETTING (lowest):")
        if method == 'mas':
            print(f"   λ={best_forget['mas_lambda']}, num_samples={best_forget['num_samples']}")
        else:
            print(f"   λ={best_forget['lwf_lambda']}, temperature={best_forget['temperature']}")
        print(f"   Accuracy: {best_forget['avg_accuracy']:.4f}")
        print(f"   Forgetting: {best_forget['forgetting']:.4f}")
        print()

        # Store best configs
        best_configs[(method, dataset)] = {
            'best_accuracy': best_acc,
            'best_forgetting': best_forget
        }

    # Summary recommendations
    print("\n" + "="*80)
    print("RECOMMENDED CONFIGURATIONS FOR CONFIG FILES")
    print("="*80)
    print()

    for method in ['mas', 'lwf']:
        print(f"\n{method.upper()} Recommendations:")
        print("-"*80)

        for dataset in ['split_mnist', 'split_cifar10', 'split_cifar100']:
            if (method, dataset) not in best_configs:
                continue

            best = best_configs[(method, dataset)]['best_accuracy']

            print(f"\n{dataset}:")
            if method == 'mas':
                print(f"  lambda: {best['mas_lambda']}")
                print(f"  num_samples: {best['num_samples']}")
            else:
                print(f"  lambda: {best['lwf_lambda']}")
                print(f"  temperature: {best['temperature']}")
            print(f"  → Accuracy: {best['avg_accuracy']:.4f}, Forgetting: {best['forgetting']:.4f}")

    return best_configs


if __name__ == '__main__':
    results_dir = "./results/hyperparam_search_mas_lwf/20251208_033358"
    best_configs = analyze_results(results_dir)

    # Save recommendations to JSON
    recommendations = {}
    for (method, dataset), configs in best_configs.items():
        if dataset not in recommendations:
            recommendations[dataset] = {}

        best = configs['best_accuracy']

        if method == 'mas':
            recommendations[dataset]['mas'] = {
                'lambda': best['mas_lambda'],
                'num_samples': best['num_samples'],
                'avg_accuracy': best['avg_accuracy'],
                'forgetting': best['forgetting']
            }
        else:
            recommendations[dataset]['lwf'] = {
                'lambda': best['lwf_lambda'],
                'temperature': best['temperature'],
                'avg_accuracy': best['avg_accuracy'],
                'forgetting': best['forgetting']
            }

    output_file = Path(results_dir) / "optimal_configs.json"
    with open(output_file, 'w') as f:
        json.dump(recommendations, f, indent=2)

    print(f"\n\nOptimal configurations saved to: {output_file}")
