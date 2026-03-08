# Reproducibility Package Available

This directory contains a complete reproducibility package for the Walsh Negotiation experiments.

## Package Files

### Main Archive (146 KB, 100 files)
- `reproducibility_package.tar.gz` - Complete code, configurations, and results

### Verification
- `reproducibility_package.tar.gz.md5` - MD5 checksum
- `reproducibility_package.tar.gz.sha256` - SHA256 checksum  
- `verify_package.sh` - Automated verification script

### Documentation (5 guides, ~42 KB)
1. **QUICKSTART.md** - Get started in 10 minutes
2. **PACKAGE_README.md** - Complete package guide
3. **REPRODUCE.md** - Detailed reproduction instructions
4. **MANIFEST.md** - Complete file listing
5. **REPRODUCIBILITY_INDEX.md** - Master index and maintenance guide

### Summary
- **DELIVERABLES_SUMMARY.txt** - Complete package overview

## Quick Start

```bash
# Extract
tar -xzf reproducibility_package.tar.gz
cd reproducibility_package/

# Verify (optional)
./verify_package.sh

# Setup environment
conda create -n continual_learning python=3.10 -y
conda activate continual_learning
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt

# Run quick test
python experiments/run_walsh_negotiation.py --dataset split_mnist --epochs 1 --seed 42
```

## Which Document Should I Read?

- **First-time user?** → Start with `QUICKSTART.md`
- **Reproducing results?** → Read `REPRODUCE.md`
- **Need complete guide?** → See `PACKAGE_README.md`
- **Looking for specific file?** → Check `MANIFEST.md`
- **Package maintainer?** → Review `REPRODUCIBILITY_INDEX.md`

## Verification

```bash
# Verify checksums
md5sum -c reproducibility_package.tar.gz.md5
sha256sum -c reproducibility_package.tar.gz.sha256

# Verify contents
./verify_package.sh
```

Expected checksums:
- MD5: `49aad4ab9b5058d7f42a3298096cce47`
- SHA256: `7b9f404405ea6d56069ddb92b1200fb433f3187a3f2b3a2bb60a00e1305678e0`

## What's Inside the Archive?

- **Source code**: 19 Python files (Walsh Negotiation + 11 baselines)
- **Experiments**: 23 Python scripts
- **Configurations**: 6 config files (MNIST, CIFAR-10, CIFAR-100)
- **Results**: 17 JSON files (15 complete experiments)
- **Documentation**: README, requirements, reproduction guide

## Time to Reproduce

- **Quick test**: 1-2 minutes (1 epoch)
- **Single experiment**: 5-15 minutes (50 epochs)
- **Full reproduction**: 2-3 hours (15 experiments, 5 seeds × 3 datasets)

## System Requirements

**Minimum**: 8GB GPU, 16GB RAM, Python 3.10+, CUDA 11.8+
**Recommended**: RTX 3090, 32GB RAM, Python 3.10.19, CUDA 12.1

## Support

See documentation files for:
- Environment setup → `REPRODUCE.md`
- Troubleshooting → `PACKAGE_README.md`
- File locations → `MANIFEST.md`
- Quick commands → `QUICKSTART.md`

---

**Package Version**: 1.0
**Date**: February 9, 2026
**Status**: Ready for distribution
