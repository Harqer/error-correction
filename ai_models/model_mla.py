import os
import math
import sys
import argparse
import hashlib
from pathlib import Path
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler
from tqdm import tqdm
import torch
import time
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# All checkpoints trained via this script are stored in ai_models/models/ so
# that helper utilities such as run_decode_all.py can discover them reliably
# no matter which working directory was active when training was launched.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
SIMULATED_DATA_DIR = PROJECT_ROOT / "simulated_data"

from ai_models.pauli_plus_dataset import PauliPlusDataset

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


from mla.core import DeepSeekMLA
  
class StabilizerEmbedder(nn.Module):
    def __init__(self, num_features, hidden_dim, num_stabilizers):
        super().__init__()
        self.feature_projs = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])
        self.index_embed = nn.Embedding(num_stabilizers, hidden_dim)
        self.final_on = nn.Embedding(1, hidden_dim)
        self.final_off = nn.Embedding(1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Initialize properly
        for proj in self.feature_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0)
        nn.init.normal_(self.index_embed.weight, mean=0, std=0.02)

    def forward(self, x, final_mask):
        B, S, _ = x.shape
        h = torch.zeros(B, S, self.index_embed.embedding_dim, device=x.device)
        
        for i, proj in enumerate(self.feature_projs):
            h += proj(x[..., i:i+1])
            
        h += self.index_embed(torch.arange(S, device=x.device))
        
        # Final mask embeddings
        h += (final_mask == 1).unsqueeze(-1) * self.final_on(torch.tensor(0, device=x.device))
        h += (final_mask == 2).unsqueeze(-1) * self.final_off(torch.tensor(0, device=x.device))
        
        return self.norm(h)

class SyndromeTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_stabilizers, grid_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = DeepSeekMLA(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def forward(self, x, events, prev_events):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
class SyndromeTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, num_stabilizers, grid_size):
        super().__init__()
        self.layers = nn.ModuleList([
            SyndromeTransformerLayer(hidden_dim, num_heads, num_stabilizers, grid_size)
            for _ in range(num_layers)
        ])

    def forward(self, x, events, prev_events):
        for layer in self.layers:
            x = layer(x, events, prev_events)
        return x

class ReadoutNetwork(nn.Module):
    def __init__(self, hidden_dim, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, basis):
        B, S, D = x.shape
        d = self.grid_size
        
        # Add dummy stabilizer and reshape
        x = torch.cat([x.new_zeros(B, 1, D), x], dim=1)
        x = x.transpose(1, 2).view(B, D, d+1, d+1)
        
        # Convolution
        x = self.conv(x).permute(0, 2, 3, 1)  # [B, d, d, D]
        
        # Line-wise readout
        outputs = []
        for i in range(B):
            if basis[i] == 0:  # X basis
                lines = x[i].mean(dim=1)  # [d, D]
            else:  # Z basis
                lines = x[i].mean(dim=0)  # [d, D]
                
            logits = self.mlp(lines).squeeze()  # [d]
            outputs.append(logits.mean())
            
        return torch.stack(outputs)

class AlphaQubitDecoder(nn.Module):
    def __init__(self, num_features, hidden_dim, num_stabilizers, grid_size, num_heads=8, num_layers=12):
        super().__init__()
        self.embedder = StabilizerEmbedder(num_features, hidden_dim, num_stabilizers)
        self.transformer = SyndromeTransformer(hidden_dim, num_heads, num_layers, num_stabilizers, grid_size)
        self.readout = ReadoutNetwork(hidden_dim, grid_size)
        
        # Initialize output layer
        for layer in self.readout.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, inputs, basis, final_mask):
        B, R, S, F = inputs.shape
        state = torch.zeros(B, S, self.embedder.index_embed.embedding_dim, device=inputs.device)
        prev_events = torch.zeros(B, S, device=inputs.device)
        
        # Process each round
        for r in range(R):
            x = inputs[:, r]
            emb = self.embedder(x, final_mask if r == R-1 else torch.zeros_like(final_mask))
            state = (state + emb) / math.sqrt(2.0)
            state = self.transformer(state, x[..., 0], prev_events)
            prev_events = x[..., 0]
            
        return self.readout(state, basis)

def get_basis_from_filename(filename):
    """Extract basis from filename (0 for X, 1 for Z, -1 if unknown)"""
    name = filename.lower()
    if "_bx_" in name:
        return 0
    elif "_bz_" in name:
        return 1
    return -1

def _strip_samples_prefix(name: str) -> str:
    return name[len("samples_") :] if name.startswith("samples_") else name


def model_stem_from_npz(npz_path: str | os.PathLike[str]) -> str:
    """Return a deterministic, collision-resistant model stem for ``npz_path``."""

    path = Path(npz_path)
    resolved = path.resolve()

    try:
        relative = resolved.relative_to(SIMULATED_DATA_DIR)
    except ValueError:
        relative = Path(path.name)

    stem_path = relative.with_suffix("")
    parts = list(stem_path.parts)
    if parts:
        parts[-1] = _strip_samples_prefix(parts[-1])

    raw = "/".join(parts) if parts else stem_path.as_posix()
    if not raw:
        raw = path.with_suffix("").name or "model"

    safe = raw.replace("/", "_")
    if "/" in raw:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe}__{digest}"

    return safe


def get_model_name_from_path(npz_path: str | os.PathLike[str]) -> Path:
    """Derive a checkpoint path inside ``ai_models/models`` for the dataset."""

    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = model_stem_from_npz(npz_path)
    return DEFAULT_MODEL_DIR / f"{stem}.pth"


def train(model, tr_loader, va_loader, epochs, lr, device, model_save_path: Path):
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    if epochs <= 0:
        print("Skipping training because epochs <= 0")
        return

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(tr_loader),
        epochs=epochs
    )

    best_val = float('inf')
    last_save_time = datetime.now()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0

        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}/{epochs}")
        for (xb, basis, mask), yb in pbar:
            xb, basis, mask, yb = xb.to(device), basis.to(device), mask.to(device), yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb, basis, mask)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            current_time = datetime.now()
            if current_time - last_save_time >= timedelta(minutes=10):
                torch.save(model.state_dict(), model_save_path)
                print(f"\nCheckpoint saved to {model_save_path} at {current_time}")
                last_save_time = current_time
        
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for (xb, basis, mask), yb in va_loader:
                xb, basis, mask, yb = xb.to(device), basis.to(device), mask.to(device), yb.to(device)
                outputs = model(xb, basis, mask)
                val_loss += criterion(outputs, yb).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == yb).sum().item()
        
        avg_loss = total_loss / len(tr_loader)
        val_loss = val_loss / len(va_loader)
        val_acc = correct / len(va_loader.dataset)
        
        print(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_file", required=True, help="Path to the NPZ file containing training data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--npu", action="store_true", help="Use available NPUs for training")
    args = parser.parse_args()

    # ---------- Device and optional DDP initialization ----------
    if args.npu and hasattr(torch, "npu") and torch.npu.is_available():
        # Setup DDP on NPUs when launched via torchrun/torch.distributed
        rank_env = os.environ.get("RANK")
        local_rank_env = os.environ.get("LOCAL_RANK")
        if rank_env is not None and local_rank_env is not None:
            dist.init_process_group(backend="hccl", init_method="env://")
            local_rank = int(local_rank_env)
        else:
            local_rank = 0
        # Pin this process to the given NPU
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    else:
        # Fallback to GPU or CPU
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    # Get basis from filename
    basis = get_basis_from_filename(args.npz_file)
    if basis == -1:
        print("Warning: Could not determine basis from filename, defaulting to X basis (0)")
        basis = 0
    
    # Create dataset from single NPZ file
    dataset = PauliPlusDataset(args.npz_file, basis)
    
    n = len(dataset)
    idx = torch.randperm(n)
    split = int(0.9 * n)
    
    tr_ds = torch.utils.data.Subset(dataset, idx[:split])
    va_ds = torch.utils.data.Subset(dataset, idx[split:])
    
    sampler = DistributedSampler(tr_ds) if dist.is_initialized() else None
    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True
    )
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)

    (x0, _, _), _ = dataset[0]
    R, S, F = x0.shape
    d = int(math.sqrt(S + 1))
    grid_size = d - 1
    
    print(f"Rounds={R} Stabilisers={S} Features={F} grid={d}×{d}")
    
    model = AlphaQubitDecoder(F, 256, S, grid_size, num_heads=8, num_layers=12).to(device)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )

    # Generate model save path from input file name
    model_save_path = get_model_name_from_path(args.npz_file)
    
    train(model, tr_loader, va_loader, args.epochs, args.lr, device, model_save_path)
    print(f"Training complete. Model saved to {model_save_path}")
