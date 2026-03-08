"""
Metrics for evaluating continual learning performance.
"""

import numpy as np
from typing import Dict, List, Optional


class ContinualLearningMetrics:
    """
    Tracks and computes continual learning metrics.

    Key metrics:
    - Average Accuracy: Mean accuracy across all seen tasks
    - Forgetting: How much performance dropped on old tasks
    - Forward Transfer: Performance on new tasks compared to joint training
    - Backward Transfer: Impact of learning new tasks on old tasks
    """

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        # accuracy_matrix[i][j] = accuracy on task j after training on task i
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        self.current_task = 0

    def update(self, task_id: int, accuracies: Dict[int, float]):
        """
        Update metrics after evaluating on all tasks.

        Args:
            task_id: Current task being trained
            accuracies: Dict mapping task_id -> accuracy
        """
        for eval_task_id, acc in accuracies.items():
            self.accuracy_matrix[task_id][eval_task_id] = acc

        self.current_task = max(self.current_task, task_id)

    def get_average_accuracy(self, task_id: Optional[int] = None) -> float:
        """
        Compute average accuracy on all tasks seen so far.

        Args:
            task_id: Evaluate up to this task (default: current task)

        Returns:
            Average accuracy
        """
        if task_id is None:
            task_id = self.current_task

        if task_id == 0:
            return self.accuracy_matrix[0][0]

        # Average accuracy on all tasks seen up to task_id
        accs = [self.accuracy_matrix[task_id][j] for j in range(task_id + 1)]
        return np.mean(accs)

    def get_forgetting(self, task_id: Optional[int] = None) -> float:
        """
        Compute average forgetting.

        Forgetting for task j = max accuracy on j - current accuracy on j

        Args:
            task_id: Evaluate up to this task (default: current task)

        Returns:
            Average forgetting
        """
        if task_id is None:
            task_id = self.current_task

        if task_id == 0:
            return 0.0

        forgetting_values = []
        for j in range(task_id):  # Don't include current task
            max_acc = np.max(self.accuracy_matrix[:task_id+1, j])
            current_acc = self.accuracy_matrix[task_id][j]
            forgetting_values.append(max_acc - current_acc)

        return np.mean(forgetting_values) if forgetting_values else 0.0

    def get_forward_transfer(
        self,
        task_id: int,
        random_baseline: Optional[Dict[int, float]] = None
    ) -> float:
        """
        Compute forward transfer for task_id.

        Forward transfer = acc on task_id (when first trained) - random baseline

        Args:
            task_id: Task to evaluate
            random_baseline: Random guess accuracy for each task

        Returns:
            Forward transfer
        """
        if random_baseline is None:
            # Assume 10-class classification
            random_baseline = {i: 0.1 for i in range(self.num_tasks)}

        # Accuracy on task_id right after training on it
        acc = self.accuracy_matrix[task_id][task_id]
        baseline = random_baseline.get(task_id, 0.1)

        return acc - baseline

    def get_backward_transfer(self, task_id: Optional[int] = None) -> float:
        """
        Compute backward transfer (positive = learning new tasks helps old tasks).

        Args:
            task_id: Evaluate up to this task (default: current task)

        Returns:
            Average backward transfer
        """
        if task_id is None:
            task_id = self.current_task

        if task_id == 0:
            return 0.0

        bt_values = []
        for j in range(task_id):
            # Accuracy on task j after training all tasks
            final_acc = self.accuracy_matrix[task_id][j]
            # Accuracy on task j right after training task j
            initial_acc = self.accuracy_matrix[j][j]
            bt_values.append(final_acc - initial_acc)

        return np.mean(bt_values) if bt_values else 0.0

    def get_all_metrics(self, task_id: Optional[int] = None) -> Dict[str, float]:
        """
        Get all metrics at once.

        Args:
            task_id: Evaluate up to this task (default: current task)

        Returns:
            Dictionary of all metrics
        """
        if task_id is None:
            task_id = self.current_task

        # Get per-task accuracies (final evaluation on each task)
        task_accuracies = [
            self.accuracy_matrix[task_id][j] for j in range(task_id + 1)
        ]

        return {
            "average_accuracy": self.get_average_accuracy(task_id),
            "forgetting": self.get_forgetting(task_id),
            "backward_transfer": self.get_backward_transfer(task_id),
            "task_accuracies": task_accuracies,
        }

    # Alias for backwards compatibility
    def get_metrics(self, task_id: Optional[int] = None) -> Dict[str, float]:
        """Alias for get_all_metrics() for backwards compatibility."""
        return self.get_all_metrics(task_id)

    def get_accuracy_matrix(self) -> np.ndarray:
        """Get the full accuracy matrix."""
        return self.accuracy_matrix.copy()

    def print_summary(self, task_id: Optional[int] = None):
        """Print a summary of metrics."""
        if task_id is None:
            task_id = self.current_task

        metrics = self.get_all_metrics(task_id)

        print(f"\n{'='*60}")
        print(f"Continual Learning Metrics (after Task {task_id})")
        print(f"{'='*60}")
        print(f"Average Accuracy:    {metrics['average_accuracy']:.4f}")
        print(f"Forgetting:          {metrics['forgetting']:.4f}")
        print(f"Backward Transfer:   {metrics['backward_transfer']:.4f}")
        print(f"{'='*60}\n")

        # Print accuracy matrix
        print("Accuracy Matrix:")
        print("Rows: Training up to task i")
        print("Cols: Evaluation on task j")
        print("\n    ", end="")
        for j in range(task_id + 1):
            print(f"T{j:2d}   ", end="")
        print()

        for i in range(task_id + 1):
            print(f"T{i:2d} ", end="")
            for j in range(task_id + 1):
                if j <= i:
                    print(f"{self.accuracy_matrix[i][j]:.3f} ", end="")
                else:
                    print("  -   ", end="")
            print()
        print()
