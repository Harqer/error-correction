
# AlphaQubit Kaggle Kernel

This kernel runs a small smoke test for the AlphaQubit repo. It:
- Installs requirements
- Clones the repo
- Runs a small data generation and a short training pass

To use:
- Set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables or upload `kaggle.json` in the environment.
- Upload a small dataset (e.g., `alphaqubit-pretrain-small`) to Kaggle and add as dataset source.
- Push kernel using `kaggle kernels push -p ./kaggle_kernel`.
