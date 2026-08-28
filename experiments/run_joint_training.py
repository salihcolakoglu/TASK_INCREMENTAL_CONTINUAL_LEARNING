"""
Run Joint Training reference (upper reference point) on task-incremental
continual learning benchmarks.

v1.1 (2026-08-28) — audit round-1 fixes (Codex 6.1 / Claude KB-12):
  1. Batch-level interleaving: every joint epoch cycles through tasks at BATCH
     granularity with a per-epoch shuffled task order (seeded), instead of
     processing tasks as full sequential blocks. This removes the fixed
     "last task is always last" recency bias of v1.0.
  2. Unmeasured metrics are reported as null (None) with an explicit
     metric_status field instead of 0.0 — no metric misstatement.
  3. Unique run_id in the output filename + no-overwrite refusal: re-running
     the same (dataset, seed) never silently destroys earlier evidence.
  4. config_hash (sha256 of the canonical config) and the repo's git commit
     (when available) are stored with the result.
  5. Naming: method is 'joint_reference' — this is a reference point trained
     with the baseline architecture/loss, NOT a universal mathematical upper
     bound for Walsh Negotiation (different output structure).

Budget definition (equal to the sequential protocol): 'epochs' counts passes
over EACH task's training data (50 by default), matching the per-task epoch
budget of the sequential experiments.

Usage:
    python experiments/run_joint_training.py --dataset split_mnist --epochs 50 --seed 42
    python experiments/run_joint_training.py --dataset split_cifar100 --n_tasks 10 --epochs 50 --seed 42
"""

import argparse
import sys
import os
import json
import time
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import get_model
from src.baselines import FineTuningTrainer
from src.utils import get_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Joint training reference (task-incremental, batch-interleaved)')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='split_mnist',
                       choices=['split_mnist', 'split_cifar10', 'split_cifar100'],
                       help='Dataset to use')
    parser.add_argument('--n_tasks', type=int, default=5,
                       help='Number of tasks')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')

    # Model arguments
    parser.add_argument('--architecture', type=str, default='auto',
                       help='Model architecture (auto, mlp, convnet)')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden layer size for MLP')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout probability')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Passes over each task per joint epoch (default 50: equal-budget)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam'],
                       help='Optimizer')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (init, data order, task interleave order)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./results/joint_training',
                       help='Directory for result JSON files')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_architecture(dataset, architecture):
    """Get appropriate architecture for dataset."""
    if architecture == 'auto':
        if dataset == 'split_mnist':
            return 'mlp'
        else:  # CIFAR datasets
            return 'convnet'
    return architecture


def get_git_commit():
    """Return the repo's current commit hash, or None if unavailable."""
    try:
        repo_root = os.path.join(os.path.dirname(__file__), '..')
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=repo_root, capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def joint_epoch_batch_interleaved(trainer, train_loaders, epoch, rng):
    """
    One joint epoch: iterate over all tasks at BATCH granularity.

    Each task contributes its batches via a persistent iterator; within an
    epoch, the task visit order is shuffled (seeded rng). When a task's
    iterator is exhausted mid-epoch it is skipped for the rest of that epoch
    (all tasks have equal batch counts in these benchmarks, so this is a
    no-op in practice).

    Uses the trainer's own compute_loss/before_backward hooks so the training
    mechanics stay identical to the sequential FineTuning baseline.
    """
    n_tasks = len(train_loaders)
    iters = [iter(loader) for loader in train_loaders]
    n_batches = max(len(loader) for loader in train_loaders)

    total_loss, correct, total = 0.0, 0, 0
    trainer.model.train()

    for batch_round in range(n_batches):
        order = list(range(n_tasks))
        rng.shuffle(order)  # per-round shuffled task order (kills recency bias)
        for task_id in order:
            try:
                batch = next(iters[task_id])
            except StopIteration:
                continue
            x, y = batch[0], batch[1]
            x, y = x.to(trainer.device), y.to(trainer.device)

            trainer.optimizer.zero_grad()
            logits = trainer.model(x, task_id=task_id)
            loss = trainer.compute_loss(logits, y, task_id, batch_round)
            loss = trainer.before_backward(loss, task_id, batch_round)
            loss.backward()
            trainer.optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            trainer.global_step += 1

    return {'loss': total_loss / max(total, 1), 'accuracy': correct / max(total, 1)}


def main():
    """Main joint-training loop (batch-interleaved)."""
    args = parse_args()
    set_seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    classes_per_task = {
        'split_mnist': 10 // args.n_tasks,
        'split_cifar10': 10 // args.n_tasks,
        'split_cifar100': 100 // args.n_tasks,
    }[args.dataset]

    architecture = get_architecture(args.dataset, args.architecture)

    # Canonical config + hash (before any training)
    config = {
        'method': 'joint_reference',
        'dataset': args.dataset,
        'n_tasks': args.n_tasks,
        'classes_per_task': classes_per_task,
        'architecture': architecture,
        'hidden_size': args.hidden_size if architecture == 'mlp' else None,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'seed': args.seed,
        'schedule': 'batch-interleaved, per-round shuffled task order',
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode('utf-8')
    ).hexdigest()[:16]

    print("=" * 70)
    print("JOINT TRAINING REFERENCE - TASK-INCREMENTAL (batch-interleaved)")
    print("=" * 70)
    print(f"  Config hash: {config_hash}")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # Load dataset
    print("Loading dataset...")
    train_loaders, val_loaders, test_loaders = get_dataset(
        dataset_name=args.dataset,
        n_tasks=args.n_tasks,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=0.0
    )
    print(f"Dataset loaded: {args.n_tasks} tasks, {classes_per_task} classes per task")

    model_kwargs = {}
    if architecture == 'mlp':
        model_kwargs['hidden_size'] = args.hidden_size

    model = get_model(
        architecture=architecture,
        num_classes_per_task=classes_per_task,
        num_tasks=args.n_tasks,
        dropout=args.dropout,
        **model_kwargs
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    trainer = FineTuningTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_tasks=args.n_tasks,
        num_classes_per_task=classes_per_task,
        config={'use_tensorboard': False, 'use_wandb': False, 'save_checkpoints': False}
    )

    print("\n" + "=" * 70)
    print("STARTING JOINT TRAINING (batch-interleaved, all tasks simultaneously)")
    print("=" * 70)

    start_time = time.time()
    rng = np.random.RandomState(args.seed)

    for epoch in range(args.epochs):
        stats = joint_epoch_batch_interleaved(trainer, train_loaders, epoch, rng)
        print(f"Joint epoch {epoch + 1}/{args.epochs} - "
              f"loss: {stats['loss']:.4f}, acc: {100 * stats['accuracy']:.2f}%")

    train_time = time.time() - start_time
    print(f"\nJoint training completed in {train_time:.1f}s")

    # Final evaluation on all tasks
    accuracies = trainer.evaluate_all_tasks(
        task_dataloaders=test_loaders,
        current_task=args.n_tasks - 1
    )
    task_accuracies = [accuracies[t] for t in range(args.n_tasks)]
    avg_acc = float(np.mean(task_accuracies))

    print("\n" + "=" * 70)
    print("FINAL RESULTS (JOINT REFERENCE)")
    print("=" * 70)
    print(f"Average Accuracy: {avg_acc:.4f}")
    print("Forgetting/BWT: not measured (tasks learned simultaneously)")

    # Save result JSON — unique run_id, no-overwrite refusal
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = output_dir / f"joint_{args.dataset}_seed{args.seed}_{run_id}.json"
    if result_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {result_file}"
        )

    result = {
        'method': 'joint_reference',
        'dataset': args.dataset,
        'seed': args.seed,
        'run_id': run_id,
        'n_tasks': args.n_tasks,
        'epochs': args.epochs,
        'architecture': architecture,
        'reference_point': True,
        'final_metrics': {
            'average_accuracy': avg_acc,
            'forgetting': None,           # not measured: tasks learned simultaneously
            'backward_transfer': None,    # not measured: tasks learned simultaneously
        },
        'metric_status': {
            'forgetting': 'not_applicable_joint_training',
            'backward_transfer': 'not_applicable_joint_training',
        },
        'task_accuracies': task_accuracies,
        'accuracy_matrix': [task_accuracies],  # single post-training evaluation row
        'train_time_sec': train_time,
        'config': config,
        'config_hash': config_hash,
        'git_commit': get_git_commit(),
        'note': 'Joint reference: single model trained on all tasks with '
                'batch-level interleaving (per-round shuffled task order). '
                'Same trunk/heads/loss as the FineTuning baseline; NOT a '
                'universal upper bound for Walsh Negotiation (different '
                'output structure). Forgetting/BWT intentionally null.',
        'script_version': 'run_joint_training.py v1.1',
    }

    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {result_file}")
    trainer.close()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == '__main__':
    main()
