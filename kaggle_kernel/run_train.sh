
#!/bin/bash
set -euo pipefail

# Install dependencies and repo code if needed
pip install -r requirements.txt || true

# Show basic system info
python - <<'PY'
import platform, shutil, torch
print(platform.platform())
print('Torch CUDA:', torch.cuda.is_available())
print('Disk free (GB):', shutil.disk_usage('/').free / 1024**3)
PY

# Attempt to clone user's repo if provided
if [ -n "${KAGGLE_USERNAME:-}" ]; then
  git clone --depth=1 https://github.com/${KAGGLE_USERNAME}/ALPHAQUBIT.git alphaqubit || true
fi
# Fallback to origin
if [ ! -d alphaqubit ]; then
  git clone --depth=1 https://github.com/xuda1979/ALPHAQUBIT.git alphaqubit || true
fi

cd alphaqubit || exit 1

# Run a small smoke sample generation + training
python generate_data.py --model pauli_plus --basis z --samples 1000 || true
python ai_models/train.py --config configs/paper_aligned.yaml --samples 1000 --epochs 1 --batch-size 8 --model-path alphaqubit_paper_aligned_smoke.pth || true

# Save outputs to /kaggle/working
mkdir -p /kaggle/working/output || true
cp -v output/* /kaggle/working/output || true
cp -v alphaqubit_paper_aligned_smoke.pth /kaggle/working/ || true

echo "Kernel finished."
