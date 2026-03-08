# Sigmoid Paradigm Experiments - Usage Guide

## Quick Start

### 1. **Quick Validation Test** (Recommended First Step)
Run this first to verify everything is working:

```bash
python experiments/quick_sigmoid_test.py
```

This runs a minimal 3-task MNIST experiment with sigmoid and should complete in ~1-2 minutes.

**Expected output:**
- Final accuracies > 90% on all tasks
- No errors or warnings
- Message: "✓ Results look good!"

---

## Experiment Runners

### 2. **Test Individual Sigmoid Methods**

#### Sigmoid Fine-tuning
```bash
# Quick test on MNIST (5 tasks, 5 epochs)
python experiments/run_sigmoid_finetune.py \
    --dataset split_mnist \
    --n_tasks 5 \
    --epochs 5

# Full experiment on CIFAR-100
python experiments/run_sigmoid_finetune.py \
    --dataset split_cifar100 \
    --n_tasks 10 \
    --epochs 20 \
    --seed 42
```

**Options:**
- `--dataset`: split_mnist | split_cifar10 | split_cifar100
- `--n_tasks`: Number of tasks (default: 5 for MNIST/CIFAR10, 10 for CIFAR100)
- `--epochs`: Epochs per task
- `--lr`: Learning rate (default: 0.01)
- `--seed`: Random seed

---

### 3. **Test Sigmoid Negotiation**

```bash
# Sigmoid negotiation with α=0.5
python experiments/run_sigmoid_negotiation.py \
    --dataset split_mnist \
    --alpha 0.5 \
    --variant sigmoid

# Hybrid (sigmoid negotiation + softmax training)
python experiments/run_sigmoid_negotiation.py \
    --dataset split_cifar100 \
    --alpha 0.5 \
    --variant hybrid \
    --save_results

# Test different alpha values
for alpha in 0.1 0.3 0.5 0.7 0.9; do
    python experiments/run_sigmoid_negotiation.py \
        --dataset split_cifar100 \
        --alpha $alpha \
        --variant sigmoid \
        --save_results
done
```

**Options:**
- `--alpha`: Negotiation rate (0.0=pure labels, 1.0=pure initial model predictions)
- `--variant`: sigmoid | hybrid
  - `sigmoid`: Full sigmoid (negotiation + training)
  - `hybrid`: Sigmoid negotiation + softmax training
- `--save_results`: Save results to JSON

---

### 4. **Full Softmax vs Sigmoid Comparison**

This is the main experiment script that compares all methods:

```bash
# Compare all methods on MNIST (quick test)
python experiments/run_sigmoid_comparison.py \
    --dataset split_mnist \
    --methods all \
    --epochs 5

# Full comparison on CIFAR-10
python experiments/run_sigmoid_comparison.py \
    --dataset split_cifar10 \
    --methods finetune ewc si negotiation \
    --epochs 20 \
    --seed 42

# CIFAR-100 with multiple seeds
for seed in 42 43 44; do
    python experiments/run_sigmoid_comparison.py \
        --dataset split_cifar100 \
        --methods all \
        --epochs 20 \
        --seed $seed
done
```

**Options:**
- `--methods`: Choose methods to compare
  - Individual: `finetune`, `ewc`, `si`, `negotiation`
  - All: `all` (runs all methods)
- Other options: same as above

**What it compares:**
For each selected method, it runs:
- **Softmax version** (your existing baselines)
- **Sigmoid version** (new sigmoid implementation)
- **Hybrid** (for negotiation only)

---

## Expected Results

### Phase 1: MNIST Validation
Run on MNIST to verify everything works:

```bash
python experiments/run_sigmoid_comparison.py \
    --dataset split_mnist \
    --methods all \
    --epochs 5
```

**Expected:**
- All methods should achieve >95% accuracy
- Sigmoid methods may have slightly lower accuracy but lower forgetting
- No errors or crashes

### Phase 2: CIFAR-10 Experiments
Main experiments on medium-difficulty dataset:

```bash
python experiments/run_sigmoid_comparison.py \
    --dataset split_cifar10 \
    --methods all \
    --epochs 20 \
    --seed 42
```

**Hypothesis to test:**
- Sigmoid should reduce forgetting by 10-30%
- Accuracy may drop slightly (0-5%)
- Sigmoid-Negotiation should outperform Softmax-Negotiation

### Phase 3: CIFAR-100 (Hardest Test)
Most challenging benchmark:

```bash
python experiments/run_sigmoid_comparison.py \
    --dataset split_cifar100 \
    --methods all \
    --epochs 20 \
    --seed 42
```

**This is where you should see the clearest differences!**

---

## Understanding the Results

### Output Format

The comparison script prints:

1. **Progress for each method** as it trains
2. **Comparison table** sorted by accuracy:
   ```
   Method                         Accuracy    Forgetting          BWT
   ----------------------------------------------------------------------
   Sigmoid-EWC                      0.7523        0.0432      -0.0432
   Softmax-EWC                      0.7421        0.0821      -0.0821
   ...
   ```

3. **Sigmoid vs Softmax improvements**:
   ```
   EWC:
     Accuracy:  0.7421 → 0.7523 (+0.0102, +1.4%)
     Forgetting: 0.0821 → 0.0432 (-0.0389, -47.4%)
   ```

### Key Metrics

- **Average Accuracy**: Mean accuracy across all tasks (higher is better)
- **Forgetting**: How much performance drops on old tasks (lower is better)
- **BWT (Backward Transfer)**: Negative forgetting (closer to 0 or positive is better)

### What to Look For

✅ **Success indicators:**
- Sigmoid has lower forgetting than softmax (main hypothesis)
- Sigmoid-Negotiation > Softmax-Negotiation
- Improvements most visible on CIFAR-100

❌ **Warning signs:**
- Sigmoid accuracy is >10% lower than softmax
- Sigmoid forgetting is higher than softmax
- NaN or infinity in results

---

## Results Storage

Results are automatically saved to:
```
results/sigmoid_experiments/
├── split_mnist_comparison_seed42_20251217_143052.json
├── sigmoid_negotiation_split_cifar100_alpha0.5_seed42_20251217_144123.json
└── ...
```

Each JSON file contains:
- Full configuration (args)
- Per-method results
- Accuracy matrices
- Final metrics

---

## Troubleshooting

### Issue: Import errors
**Solution:** Make sure you're running from the project root:
```bash
cd /path/to/TASK_INCREMENTAL_CONTINUAL_LEARNING
python experiments/run_sigmoid_comparison.py ...
```

### Issue: CUDA out of memory
**Solution:** Reduce batch size:
```bash
python experiments/run_sigmoid_comparison.py \
    --dataset split_cifar100 \
    --batch_size 64  # Instead of 128
```

### Issue: Sigmoid accuracy is very low
**Possible causes:**
1. Learning rate too low → try `--lr 0.02`
2. Not enough epochs → increase `--epochs`
3. Implementation bug → run quick_sigmoid_test.py to verify

### Issue: Results don't match hypothesis
**This is actually interesting!**
- Document what you observe
- Try different datasets
- Check if the pattern is consistent across seeds

---

## Recommended Experimental Pipeline

### Step 1: Validation (5 minutes)
```bash
python experiments/quick_sigmoid_test.py
```

### Step 2: MNIST Quick Test (10 minutes)
```bash
python experiments/run_sigmoid_comparison.py \
    --dataset split_mnist \
    --methods finetune negotiation \
    --epochs 5
```

### Step 3: CIFAR-10 Full Test (2-3 hours)
```bash
python experiments/run_sigmoid_comparison.py \
    --dataset split_cifar10 \
    --methods all \
    --epochs 20 \
    --seed 42
```

### Step 4: CIFAR-100 Main Experiments (6-8 hours)
```bash
for seed in 42 43 44; do
    python experiments/run_sigmoid_comparison.py \
        --dataset split_cifar100 \
        --methods all \
        --epochs 20 \
        --seed $seed
done
```

### Step 5: Negotiation Alpha Search (4-5 hours)
```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python experiments/run_sigmoid_negotiation.py \
        --dataset split_cifar100 \
        --alpha $alpha \
        --variant sigmoid \
        --save_results \
        --seed 42
done
```

---

## Next Steps

After running experiments:

1. **Analyze results** using the comparison tables
2. **Compare with existing baselines** in PROGRESS.md:
   - Current SOTA: LwF (69.18% on CIFAR-100)
   - Can sigmoid improve on this?
3. **Document findings** in a new markdown file
4. **Create visualizations** (optional - we can create plotting scripts)

---

## Questions?

If you encounter issues:
1. Check the error message carefully
2. Verify you're in the correct directory
3. Make sure all dependencies are installed
4. Run `quick_sigmoid_test.py` to verify basic functionality

Good luck with your experiments! 🚀
