"""
Test script to verify the continual learning setup is working correctly.
Run this before starting experiments to ensure all components are functional.
"""

import sys
import torch
import avalanche as avl
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

def test_environment():
    """Test basic environment setup."""
    print("=" * 60)
    print("TESTING ENVIRONMENT SETUP")
    print("=" * 60)

    # Python version
    print(f"✓ Python version: {sys.version.split()[0]}")

    # PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Avalanche
    print(f"✓ Avalanche version: {avl.__version__}")

    print("\n✓ Environment setup is correct!")
    return True


def test_data_loading():
    """Test data loading with Split MNIST."""
    print("\n" + "=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)

    try:
        # Create Split MNIST benchmark (5 tasks)
        benchmark = SplitMNIST(
            n_experiences=5,
            return_task_id=True,  # Task-incremental setting
            seed=42,
            dataset_root='./data'
        )

        print(f"✓ Created Split MNIST benchmark with {len(benchmark.train_stream)} tasks")
        print(f"✓ Task 0 training samples: {len(benchmark.train_stream[0].dataset)}")
        print(f"✓ Task 0 test samples: {len(benchmark.test_stream[0].dataset)}")

        # Test data loader
        train_loader = torch.utils.data.DataLoader(
            benchmark.train_stream[0].dataset,
            batch_size=32,
            shuffle=True
        )

        # Get one batch
        x, y, t = next(iter(train_loader))
        print(f"✓ Batch shape: {x.shape}")
        print(f"✓ Labels shape: {y.shape}")
        print(f"✓ Task IDs shape: {t.shape}")
        print(f"✓ Task ID: {t[0].item()}")

        print("\n✓ Data loading is working correctly!")
        return True

    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_model_training():
    """Test basic model training."""
    print("\n" + "=" * 60)
    print("TESTING MODEL TRAINING")
    print("=" * 60)

    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Using device: {device}")

        # Create benchmark
        benchmark = SplitMNIST(
            n_experiences=2,  # Just 2 tasks for quick test
            return_task_id=True,
            seed=42,
            dataset_root='./data'
        )

        # Create model (multi-head for task-incremental)
        model = SimpleMLP(num_classes=10, hidden_size=128)

        # Define optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        # Create evaluation plugin
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
            loggers=[InteractiveLogger()]
        )

        # Create strategy (Naive = simple fine-tuning)
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=32,
            train_epochs=1,  # Just 1 epoch for quick test
            eval_mb_size=32,
            device=device,
            evaluator=eval_plugin
        )

        print("✓ Model and strategy created")

        # Train on first task
        print("\n✓ Training on Task 0 (1 epoch, for testing only)...")
        strategy.train(benchmark.train_stream[0])

        # Evaluate
        print("\n✓ Evaluating on Task 0...")
        results = strategy.eval(benchmark.test_stream[0])

        print("\n✓ Model training is working correctly!")
        return True

    except Exception as e:
        print(f"✗ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TASK-INCREMENTAL CONTINUAL LEARNING - SETUP TEST")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("Environment", test_environment()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Model Training", test_model_training()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready for experiments.")
        print("\nNext steps:")
        print("1. Review the Project Plan.md")
        print("2. Start implementing baselines in src/baselines/")
        print("3. Configure experiment tracking (W&B or TensorBoard)")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
