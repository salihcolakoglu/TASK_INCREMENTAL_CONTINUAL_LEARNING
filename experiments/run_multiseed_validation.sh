#!/bin/bash
################################################################################
# Multi-Seed Validation Runner
# Purpose: Run all key experiments with multiple seeds for statistical validation
# Date: December 18, 2025
################################################################################

set -e  # Exit on error

# Configuration
SEEDS=(42 43 44 45 46)
DATASETS=("split_mnist" "split_cifar10" "split_cifar100")
LOG_DIR="./logs/multiseed"
RESULTS_DIR="./results"

# Create log directory
mkdir -p "$LOG_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to run sigmoid comparison experiments
run_sigmoid_comparison() {
    local dataset=$1
    local methods=$2
    local seed=$3
    local epochs=$4

    log_info "Running sigmoid comparison: dataset=$dataset, methods=$methods, seed=$seed"

    local log_file="$LOG_DIR/sigmoid_${dataset}_${methods}_seed${seed}.log"

    python experiments/run_sigmoid_comparison.py \
        --dataset "$dataset" \
        --methods "$methods" \
        --epochs "$epochs" \
        --seed "$seed" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        log_success "Completed: $dataset / $methods / seed=$seed"
    else
        log_error "Failed: $dataset / $methods / seed=$seed (see $log_file)"
        return 1
    fi
}

# Function to run alpha search experiments
run_alpha_search() {
    local dataset=$1
    local seed=$2
    local alpha_values=$3
    local variants=$4
    local epochs=$5

    log_info "Running alpha search: dataset=$dataset, seed=$seed, variants=$variants"

    local log_file="$LOG_DIR/alpha_${dataset}_${variants}_seed${seed}.log"

    python experiments/alpha_search_negotiation.py \
        --dataset "$dataset" \
        --seed "$seed" \
        --alpha_values "$alpha_values" \
        --variants "$variants" \
        --epochs "$epochs" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        log_success "Completed: Alpha search on $dataset / seed=$seed"
    else
        log_error "Failed: Alpha search on $dataset / seed=$seed (see $log_file)"
        return 1
    fi
}

# Function to estimate total time
estimate_time() {
    local total_experiments=$1
    local avg_time_per_exp=$2

    local total_hours=$(echo "scale=1; $total_experiments * $avg_time_per_exp" | bc)
    echo "$total_hours hours"
}

################################################################################
# Main Experiment Configurations
################################################################################

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           MULTI-SEED VALIDATION EXPERIMENT RUNNER                  ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Ask user which experiments to run
echo "Select experiment mode:"
echo "  1) Quick validation (top 3 methods, CIFAR-100 only, 5 seeds)"
echo "  2) Comprehensive validation (all methods, all datasets, 5 seeds)"
echo "  3) Alpha search validation (optimal alphas, all datasets, 5 seeds)"
echo "  4) Full validation (everything, 5 seeds)"
echo "  5) Custom (specify manually)"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        #######################################################################
        # MODE 1: Quick Validation
        # Top 3 methods on CIFAR-100 with 5 seeds
        # Estimated time: ~60 GPU hours (12 hours × 5 seeds)
        #######################################################################

        log_info "MODE 1: Quick Validation"
        log_info "Methods: SI, EWC, Negotiation (sigmoid)"
        log_info "Dataset: CIFAR-100"
        log_info "Seeds: ${SEEDS[*]}"
        log_info "Estimated time: 60 GPU hours"
        echo ""

        read -p "Proceed? [y/N]: " confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            log_warning "Aborted by user"
            exit 0
        fi

        TOTAL=0
        FAILED=0

        for seed in "${SEEDS[@]}"; do
            # Softmax-SI (best forgetting)
            run_sigmoid_comparison "split_cifar100" "si" "$seed" 20 || ((FAILED++))
            ((TOTAL++))

            # Softmax-EWC (best accuracy among regularization)
            run_sigmoid_comparison "split_cifar100" "ewc" "$seed" 20 || ((FAILED++))
            ((TOTAL++))

            # Sigmoid-Negotiation (best sigmoid method)
            run_sigmoid_comparison "split_cifar100" "negotiation" "$seed" 20 || ((FAILED++))
            ((TOTAL++))
        done

        log_info "Quick validation complete: $((TOTAL - FAILED))/$TOTAL succeeded"
        ;;

    2)
        #######################################################################
        # MODE 2: Comprehensive Validation
        # All methods, all datasets, 5 seeds
        # Estimated time: ~825 GPU hours (165 hours × 5 seeds)
        #######################################################################

        log_info "MODE 2: Comprehensive Validation"
        log_info "Methods: ALL (finetune, ewc, si, negotiation)"
        log_info "Datasets: MNIST, CIFAR-10, CIFAR-100"
        log_info "Seeds: ${SEEDS[*]}"
        log_info "Estimated time: 825 GPU hours"
        echo ""

        log_warning "This will take a VERY long time!"
        read -p "Proceed? [y/N]: " confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            log_warning "Aborted by user"
            exit 0
        fi

        TOTAL=0
        FAILED=0

        for seed in "${SEEDS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                # Determine epochs based on dataset
                case $dataset in
                    split_mnist)
                        epochs=10
                        ;;
                    split_cifar10)
                        epochs=20
                        ;;
                    split_cifar100)
                        epochs=20
                        ;;
                esac

                # Run all methods (excluding SI for sigmoid due to NaN issue)
                run_sigmoid_comparison "$dataset" "all" "$seed" "$epochs" || ((FAILED++))
                ((TOTAL++))
            done
        done

        log_info "Comprehensive validation complete: $((TOTAL - FAILED))/$TOTAL succeeded"
        ;;

    3)
        #######################################################################
        # MODE 3: Alpha Search Validation
        # Optimal alphas with 5 seeds for all datasets
        # Estimated time: ~90 GPU hours
        #######################################################################

        log_info "MODE 3: Alpha Search Validation"
        log_info "Testing optimal alphas: 0.2 (softmax), 0.7 (sigmoid)"
        log_info "Datasets: MNIST, CIFAR-10, CIFAR-100"
        log_info "Seeds: ${SEEDS[*]}"
        log_info "Estimated time: 90 GPU hours"
        echo ""

        read -p "Proceed? [y/N]: " confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            log_warning "Aborted by user"
            exit 0
        fi

        TOTAL=0
        FAILED=0

        for seed in "${SEEDS[@]}"; do
            # MNIST (10 epochs)
            run_alpha_search "split_mnist" "$seed" "0.2,0.7" "both" 10 || ((FAILED++))
            ((TOTAL++))

            # CIFAR-10 (20 epochs)
            run_alpha_search "split_cifar10" "$seed" "0.2,0.7" "both" 20 || ((FAILED++))
            ((TOTAL++))

            # CIFAR-100 (20 epochs)
            run_alpha_search "split_cifar100" "$seed" "0.2,0.7" "both" 20 || ((FAILED++))
            ((TOTAL++))
        done

        log_info "Alpha search validation complete: $((TOTAL - FAILED))/$TOTAL succeeded"
        ;;

    4)
        #######################################################################
        # MODE 4: Full Validation
        # Everything with 5 seeds
        # Estimated time: ~900+ GPU hours
        #######################################################################

        log_info "MODE 4: Full Validation (Comprehensive + Alpha Search)"
        log_error "WARNING: This will take 900+ GPU hours!"
        echo ""

        read -p "Are you SURE? Type 'YES' to confirm: " confirm
        if [[ $confirm != "YES" ]]; then
            log_warning "Aborted by user"
            exit 0
        fi

        # Run Mode 2 (Comprehensive)
        log_info "Starting comprehensive validation..."
        TOTAL=0
        FAILED=0

        for seed in "${SEEDS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                case $dataset in
                    split_mnist)
                        epochs=10
                        ;;
                    split_cifar10)
                        epochs=20
                        ;;
                    split_cifar100)
                        epochs=20
                        ;;
                esac

                run_sigmoid_comparison "$dataset" "all" "$seed" "$epochs" || ((FAILED++))
                ((TOTAL++))
            done
        done

        # Run Mode 3 (Alpha Search)
        log_info "Starting alpha search validation..."

        for seed in "${SEEDS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                case $dataset in
                    split_mnist)
                        epochs=10
                        alpha_vals="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
                        ;;
                    split_cifar10)
                        epochs=20
                        alpha_vals="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
                        ;;
                    split_cifar100)
                        epochs=20
                        alpha_vals="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
                        ;;
                esac

                run_alpha_search "$dataset" "$seed" "$alpha_vals" "both" "$epochs" || ((FAILED++))
                ((TOTAL++))
            done
        done

        log_info "Full validation complete: $((TOTAL - FAILED))/$TOTAL succeeded"
        ;;

    5)
        #######################################################################
        # MODE 5: Custom
        #######################################################################

        log_info "MODE 5: Custom Configuration"
        echo ""

        read -p "Enter dataset (split_mnist/split_cifar10/split_cifar100): " dataset
        read -p "Enter methods (all/finetune/ewc/si/negotiation): " methods
        read -p "Enter epochs (10/20): " epochs
        read -p "Enter seeds (comma-separated, e.g., 42,43,44): " seed_input

        IFS=',' read -ra CUSTOM_SEEDS <<< "$seed_input"

        log_info "Custom config: dataset=$dataset, methods=$methods, epochs=$epochs"
        log_info "Seeds: ${CUSTOM_SEEDS[*]}"
        echo ""

        read -p "Proceed? [y/N]: " confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            log_warning "Aborted by user"
            exit 0
        fi

        TOTAL=0
        FAILED=0

        for seed in "${CUSTOM_SEEDS[@]}"; do
            run_sigmoid_comparison "$dataset" "$methods" "$seed" "$epochs" || ((FAILED++))
            ((TOTAL++))
        done

        log_info "Custom validation complete: $((TOTAL - FAILED))/$TOTAL succeeded"
        ;;

    *)
        log_error "Invalid choice: $choice"
        exit 1
        ;;
esac

################################################################################
# Summary
################################################################################

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                      VALIDATION COMPLETE                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
log_success "Total experiments: $TOTAL"
log_success "Successful: $((TOTAL - FAILED))"
if [ $FAILED -gt 0 ]; then
    log_error "Failed: $FAILED"
    log_info "Check logs in $LOG_DIR for details"
else
    log_success "All experiments completed successfully!"
fi
echo ""
log_info "Results saved to: $RESULTS_DIR/sigmoid_experiments/"
log_info "Logs saved to: $LOG_DIR/"
echo ""
log_info "Next steps:"
echo "  1. Run: python experiments/analyze_multiseed_results.py"
echo "  2. Generate visualizations"
echo "  3. Write paper with mean ± std results"
echo ""
