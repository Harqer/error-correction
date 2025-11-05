
import os
import argparse
import math
import gc
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from model_mla import AlphaQubitDecoder  # make sure this is on PYTHONPATH



# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Fine‑tune AlphaQubit decoder on one experiment folder")
    p.add_argument("--dataset", required=True, help="Path to a *single* experiment folder (contains detection_events.b8 etc.)")
    p.add_argument("--batch-size", "-b", type=int, default=1024, help="Mini‑batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--weight-decay", "-w", type=float, default=1e-3, help="L2 weight decay")
    p.add_argument("--epochs", "-e", type=int, default=30, help="Number of training epochs")
    p.add_argument("--train-samples", "-t", type=int, default=19880, help="Max # training shots")
    p.add_argument("--valid-samples", "-v", type=int, default=5120, help="Max # validation shots")
    p.add_argument("--patience", "-p", type=int, default=5, help="Early‑stopping patience (epochs)")
    p.add_argument(
        "--model_path",
        "-m",
        type=Path,
        required=True,
        help="Specify the path of the model to load and save",
    )
    p.add_argument("--npu", action="store_true", help="Use available NPUs for training")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Dataset for one folder
# -----------------------------------------------------------------------------

class SingleFolderDataset(Dataset):
    """Loads *one* experiment run produced by Google's surface‑code pipeline."""

    def __init__(self, folder: str):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        self.folder = folder

        # 1) labels – 1 bit per shot (whether logical flip occurred)
        lbl_path = os.path.join(folder, "obs_flips_actual.01")
        if not os.path.exists(lbl_path):
            raise FileNotFoundError(f"Label file missing: {lbl_path}")
        lbls = np.loadtxt(lbl_path, dtype=np.int8)
        self.labels = torch.from_numpy(lbls).float()
        self.n_shots = len(self.labels)

        # 2) detection events
        det_path = os.path.join(folder, "detection_events.b8")
        if not os.path.exists(det_path):
            raise FileNotFoundError(f"Detection events missing: {det_path}")
        self.raw = np.memmap(det_path, dtype=np.uint8, mode="r")

        # 3) work out stabiliser count
        total_bits = self.raw.size * 8
        if total_bits % self.n_shots:
            raise ValueError("Byte length not divisible by #shots – corrupt file?")
        self.bits_per_shot = total_bits // self.n_shots
        if self.bits_per_shot % 8:
            raise ValueError("bits_per_shot must be divisible by 8")
        self.bytes_per_shot = self.bits_per_shot // 8

        # 4) grid padding so that S+1 is perfect square (model expects this)
        d = math.isqrt(self.bits_per_shot + 1)
        if d * d != self.bits_per_shot + 1:
            d += 1
        self.S_pad = d * d - 1  # padded stabiliser count
        self.grid_size = d - 1  # data qubit grid dimension

        # 5) basis id from folder name
        base = os.path.basename(folder).lower()
        self.basis_id = 0 if "_bx_" in base else 1

        # 6) final_mask pattern (checkerboard – matches model_mla logic)
        self.final_mask = torch.tensor(
            [1 if (r + c) % 2 == 0 else 2 for r in range(d) for c in range(d)][1:],  # skip dummy stabiliser 0
            dtype=torch.long,
        )

    def __len__(self):
        return self.n_shots

    def __getitem__(self, idx):
        # detection events -> bit tensor
        start = idx * self.bytes_per_shot
        bits = np.unpackbits(self.raw[start : start + self.bytes_per_shot])
        x = torch.from_numpy(bits).float().view(1, -1, 1)  # (R=1, S_raw, F=1)
        if x.size(1) < self.S_pad:
            pad = torch.zeros(1, self.S_pad - x.size(1), 1)
            x = torch.cat([x, pad], dim=1)
        # add basis feature so F=2
        basis_feat = torch.full((1, self.S_pad, 1), float(self.basis_id))
        x = torch.cat([x, basis_feat], dim=-1)
        y = self.labels[idx]
        return (x, torch.tensor(self.basis_id), self.final_mask.clone()), y

# -----------------------------------------------------------------------------
# Training on a single folder
# -----------------------------------------------------------------------------

def train_on_folder(folder: str, args, device):
    print(f"\n=== Fine‑tuning on {folder} ===")
    ds = SingleFolderDataset(folder)

    model_path: Path = args.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # limit samples if requested
    train_n = min(args.train_samples, len(ds))
    valid_n = min(args.valid_samples, len(ds) - train_n)
    test_n = len(ds) - train_n - valid_n
    lengths = [train_n, valid_n, test_n]
    while sum(lengths) < len(ds):
        lengths[0] += 1  # add remainder to train split
    train_ds, valid_ds, test_ds = random_split(ds, lengths)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, pin_memory=True)

    # model dimensions from dataset
    model = AlphaQubitDecoder(
        num_features=2,
        hidden_dim=256,
        num_stabilizers=ds.S_pad,
        grid_size=ds.grid_size,
        num_heads=8,
        num_layers=12,
    ).to(device)
    if args.npu and hasattr(torch, "npu"):
        npu_count = getattr(torch.npu, "device_count", lambda: 1)()
        if npu_count > 1:
            print(f"Using {npu_count} NPUs!")
            model = nn.DataParallel(model)
    elif device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
            print(f"Loaded base weights from {model_path}")
        except Exception as e:
            print(f"[WARN] Could not load {model_path}: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    patience_cnt = 0
  

    for ep in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for (xb, basis, mask), yb in tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs} [train]"):
            xb, basis, mask, yb = xb.to(device), basis.to(device), mask.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb, basis, mask), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); running += loss.item()

        # validation
        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for (xb, basis, mask), yb in valid_loader:
                xb, basis, mask, yb = xb.to(device), basis.to(device), mask.to(device), yb.to(device)
                v_loss += criterion(model(xb, basis, mask), yb).item()
        v_loss /= max(1, len(valid_loader))
        print(f"Epoch {ep}: train {running/len(train_loader):.4f} | val {v_loss:.4f}")
        if v_loss < best_val:
            best_val = v_loss; patience_cnt = 0
            torch.save(model.state_dict(), model_path)
            print(f"    ↳ saved best model for this dataset to {model_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print("Early stopping triggered")
                break

    # test accuracy
    best_path = model_path
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval(); correct = total = 0
    with torch.no_grad():
        for (xb, basis, mask), yb in test_loader:
            xb, basis, mask, yb = xb.to(device), basis.to(device), mask.to(device), yb.to(device)
            preds = (torch.sigmoid(model(xb, basis, mask)) > 0.5).float()
            correct += (preds == yb).sum().item(); total += yb.numel()
    print(f"Test accuracy: {correct/total:.4f}\n")
    del model, opt, train_loader, valid_loader, test_loader; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# Entry‑point
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.npu:
        try:
            import torch_npu  # patches torch to add NPU support
        except ImportError:
            print("Warning: torch_npu package is not installed; NPU support may be unavailable.")

    if args.npu:
        if hasattr(torch, "npu") and torch.npu.is_available():
            device = torch.device("npu")
        else:
            print("Warning: --npu specified but NPU support is unavailable.")
            print("This could be due to:")
            print("1. Ascend-specific PyTorch not installed")
            print("2. No Ascend NPU devices detected")
            print("3. NPU drivers not properly configured")
            print("Falling back to CUDA/CPU...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_on_folder(args.dataset, args, device)

if __name__ == "__main__":
    main()
