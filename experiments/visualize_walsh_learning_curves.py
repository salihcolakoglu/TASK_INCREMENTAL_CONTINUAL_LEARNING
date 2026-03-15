"""
Walsh Negotiation Learning Curves and Forgetting Trajectory Visualization

Generates publication-ready figures showing:
1. Per-task accuracy curves for each dataset (MNIST, CIFAR-10, CIFAR-100)
2. Forgetting trajectories showing how Task 1 accuracy degrades across datasets

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

# Color palette for tasks (extended for CIFAR-100's 10 tasks)
TASK_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


def load_walsh_results(results_dir, dataset, expected_n_tasks=None):
    """Load Walsh experiment results for a specific dataset."""
    results_path = Path(results_dir)
    pattern = f"{dataset}_walsh_full_alpha0.5_epochs50_seed*_202602*.json"
    files = list(results_path.glob(pattern))

    if not files:
        print(f"Warning: No files found for pattern {pattern}")
        return None

    results = []
    task_counts = defaultdict(list)

    for file in sorted(files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)

            # Extract seed from filename
            seed_str = file.name.split('seed')[1].split('_')[0]
            seed = int(seed_str)

            accuracy_matrix = np.array(data['results']['accuracy_matrix'])
            n_tasks = accuracy_matrix.shape[0]

            task_counts[n_tasks].append(file)

            results.append({
                'seed': seed,
                'accuracy_matrix': accuracy_matrix,
                'method': data.get('method', 'Walsh-Negotiation'),
                'args': data.get('args', {}),
                'n_tasks': n_tasks,
                'filename': file.name
            })
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")

    # Find the most common task count
    if task_counts:
        most_common_n_tasks = max(task_counts.keys(), key=lambda k: len(task_counts[k]))

        if len(task_counts) > 1:
            print(f"Warning: {dataset} has inconsistent task counts:")
            for n, files in sorted(task_counts.items()):
                print(f"  {n} tasks: {len(files)} files")
            print(f"  Using {most_common_n_tasks} tasks for consistency")

        # Filter to keep only results with the most common task count
        results = [r for r in results if r['n_tasks'] == most_common_n_tasks]

    print(f"Loaded {len(results)} results for {dataset} (n_tasks={results[0]['n_tasks'] if results else 'N/A'})")
    return results if results else None


def extract_learning_curves(results):
    """
    Extract learning curves from accuracy matrices.

    For each task, the learning curve shows the accuracy on that task
    as training progresses through subsequent tasks.

    Returns:
        dict: {task_id: list of (mean_accuracies, std_accuracies)}
    """
    if not results:
        return None

    n_tasks = results[0]['accuracy_matrix'].shape[1]

    # Collect accuracies for each task across all seeds
    task_curves = defaultdict(list)

    for result in results:
        acc_matrix = result['accuracy_matrix']

        # For each task, extract its accuracy trajectory
        for task_id in range(n_tasks):
            curve = []
            for train_step in range(task_id, n_tasks):
                accuracy = acc_matrix[train_step, task_id]
                if accuracy > 0:  # Only include non-zero values
                    curve.append(accuracy)
            task_curves[task_id].append(curve)

    # Compute mean and std for each task
    task_stats = {}
    for task_id, curves in task_curves.items():
        # Convert to numpy array (all curves should have same length)
        curves_array = np.array(curves)
        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_curve)
        task_stats[task_id] = (mean_curve, std_curve)

    return task_stats


def plot_learning_curves(results, dataset_name, save_path):
    """Plot learning curves for all tasks in a dataset."""
    if not results:
        print(f"Skipping {dataset_name} - no results")
        return

    task_stats = extract_learning_curves(results)
    n_tasks = len(task_stats)

    fig, ax = plt.subplots(figsize=(10, 6))

    for task_id in range(n_tasks):
        mean_curve, std_curve = task_stats[task_id]

        # X-axis represents training progression (which task we're currently training)
        x = np.arange(task_id, task_id + len(mean_curve))

        # Plot mean curve
        ax.plot(x, mean_curve * 100,
                label=f'Task {task_id + 1}',
                color=TASK_COLORS[task_id % len(TASK_COLORS)],
                linewidth=2, marker='o', markersize=6)

        # Add error band (mean ± std)
        ax.fill_between(x,
                        (mean_curve - std_curve) * 100,
                        (mean_curve + std_curve) * 100,
                        color=TASK_COLORS[task_id % len(TASK_COLORS)],
                        alpha=0.2)

    ax.set_xlabel('Training Phase (Task Being Learned)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Walsh Negotiation Learning Curves: {dataset_name}\n(Mean ± Std across {len(results)} seeds)')
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([f'Task {i+1}' for i in range(n_tasks)])
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved learning curves to {save_path}")
    plt.close()


def extract_task1_forgetting_trajectory(results):
    """
    Extract Task 1 accuracy trajectory (forgetting).

    Returns:
        tuple: (mean_trajectory, std_trajectory)
    """
    if not results:
        return None, None

    trajectories = []
    for result in results:
        acc_matrix = result['accuracy_matrix']
        # Task 0 (first task) accuracy across all training phases
        task1_trajectory = acc_matrix[:, 0]
        trajectories.append(task1_trajectory)

    # Convert to array and compute statistics
    trajectories_array = np.array(trajectories)
    mean_traj = np.mean(trajectories_array, axis=0)
    std_traj = np.std(trajectories_array, axis=0, ddof=1) if len(trajectories) > 1 else np.zeros_like(mean_traj)

    # Filter out zero values for plotting (but keep arrays aligned)
    # We'll handle zeros in the plotting function instead
    return mean_traj, std_traj


def plot_forgetting_trajectory(datasets_results, save_path):
    """Plot Task 1 forgetting trajectory across all datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))

    dataset_colors = {'MNIST': '#2E7D32', 'CIFAR-10': '#1976D2', 'CIFAR-100': '#D32F2F'}

    max_tasks = 0
    for dataset_name, results in datasets_results.items():
        if not results:
            continue

        mean_traj, std_traj = extract_task1_forgetting_trajectory(results)
        n_tasks = len(mean_traj)
        max_tasks = max(max_tasks, n_tasks)
        x = np.arange(n_tasks)

        # Plot mean trajectory
        ax.plot(x, mean_traj * 100,
                label=f'{dataset_name} ({n_tasks} tasks)',
                color=dataset_colors.get(dataset_name, 'gray'),
                linewidth=2, marker='o', markersize=6)

        # Add error band
        ax.fill_between(x,
                        (mean_traj - std_traj) * 100,
                        (mean_traj + std_traj) * 100,
                        color=dataset_colors.get(dataset_name, 'gray'),
                        alpha=0.2)

    ax.set_xlabel('Training Phase (Task Being Learned)')
    ax.set_ylabel('Task 1 Accuracy (%)')
    ax.set_title('Walsh Negotiation: Task 1 Forgetting Trajectory\n(Mean ± Std across seeds)')
    ax.set_xticks(range(max_tasks))
    ax.set_xticklabels([f'T{i+1}' for i in range(max_tasks)])
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim([50, 105])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved forgetting trajectory to {save_path}")
    plt.close()


def print_statistics(datasets_results):
    """Print summary statistics for all datasets."""
    print("\n" + "="*80)
    print("WALSH NEGOTIATION SUMMARY STATISTICS")
    print("="*80)

    for dataset_name, results in datasets_results.items():
        if not results:
            continue

        print(f"\n{dataset_name}:")
        print(f"  Number of seeds: {len(results)}")

        # Average accuracy
        avg_accs = [r['accuracy_matrix'][-1, :].mean() for r in results]
        print(f"  Average Accuracy: {np.mean(avg_accs)*100:.2f}% ± {np.std(avg_accs, ddof=1)*100:.2f}%")

        # Task 1 forgetting
        task1_initial = [r['accuracy_matrix'][0, 0] for r in results]
        task1_final = [r['accuracy_matrix'][-1, 0] for r in results]
        forgetting = [(init - final) for init, final in zip(task1_initial, task1_final)]
        print(f"  Task 1 Forgetting: {np.mean(forgetting)*100:.2f}% ± {np.std(forgetting, ddof=1)*100:.2f}%")

        # Per-task final accuracies
        n_tasks = results[0]['accuracy_matrix'].shape[1]
        print(f"  Per-task final accuracies:")
        for task_id in range(n_tasks):
            task_accs = [r['accuracy_matrix'][-1, task_id] for r in results]
            print(f"    Task {task_id+1}: {np.mean(task_accs)*100:.2f}% ± {np.std(task_accs, ddof=1)*100:.2f}%")

    print("\n" + "="*80 + "\n")


def main():
    # Paths
    results_dir = Path('./results/walsh_experiments')
    output_dir = Path('./figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("WALSH NEGOTIATION LEARNING CURVES GENERATOR")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}\n")

    # Load results for all datasets
    datasets = {
        'MNIST': load_walsh_results(results_dir, 'split_mnist'),
        'CIFAR-10': load_walsh_results(results_dir, 'split_cifar10'),
        'CIFAR-100': load_walsh_results(results_dir, 'split_cifar100'),
    }

    print("\nGenerating visualizations...\n")

    # Plot learning curves for each dataset
    for dataset_name, results in datasets.items():
        safe_name = dataset_name.replace('-', '').lower()
        plot_learning_curves(
            results,
            dataset_name,
            output_dir / f'walsh_learning_curves_{safe_name}.png'
        )

    # Plot forgetting trajectory across all datasets
    plot_forgetting_trajectory(
        datasets,
        output_dir / 'walsh_forgetting_trajectory.png'
    )

    # Print summary statistics
    print_statistics(datasets)

    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"All figures saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - walsh_learning_curves_mnist.png")
    print(f"  - walsh_learning_curves_cifar10.png")
    print(f"  - walsh_learning_curves_cifar100.png")
    print(f"  - walsh_forgetting_trajectory.png")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
