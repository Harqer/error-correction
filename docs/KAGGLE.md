
# Kaggle Usage for ALPHAQUBIT

This document explains how to set up Kaggle datasets and kernels to run smoke tests or short training runs.

## Setup
- Export env vars:

```
export KAGGLE_USERNAME=YOUR_USERNAME
export KAGGLE_KEY=YOUR_KEY
```

## Create a Kaggle dataset (small subset recommended)
```bash
mkdir -p ~/alphaqubit-kaggle/dataset
cp path/to/your/pretrain_data/some_small_files/* ~/alphaqubit-kaggle/dataset
cat > ~/alphaqubit-kaggle/dataset/dataset-metadata.json <<'JSON'
{ "title":"alphaqubit-pretrain-small", "id":"$KAGGLE_USERNAME/alphaqubit-pretrain-small", "licenses":[{"name":"CC0-1.0"}] }
JSON
kaggle datasets create -p ~/alphaqubit-kaggle/dataset
```

## Push and run kernel
```bash
cd ~/ALPHAQUBIT/kaggle_kernel
kaggle kernels push -p .
```

Note: This kernel is intended for short smoke runs. For full production-scale pretraining, consider using a cloud compute provider with larger disk and runtime quotas.
