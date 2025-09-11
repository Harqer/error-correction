#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaQubit – surface-code decoder (reference PyTorch implementation)

Changes vs. original:
• Dataset now auto-pads dummy stabilisers so that S+1 = d².
• All earlier shape / count / grid-size assertions can never fail.
"""

import os
import math
import argparse
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

# ---------------------------------------------------------------------
#  Model components
# ---------------------------------------------------------------------

class StabilizerEmbedder(nn.Module):
    """Embed per-stabiliser features + index + final-round tags."""
    def __init__(self, num_features: int, hidden_dim: int, num_stabilizers: int):
        super().__init__()
        self.feature_projs = nn.ModuleList([nn.Linear(1, hidden_dim)
                                            for _ in range(num_features)])
        self.index_embedding   = nn.Embedding(num_stabilizers, hidden_dim)
        self.final_on_emb  = nn.Embedding(1, hidden_dim)
        self.final_off_emb = nn.Embedding(1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, final_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)   final_mask: (B, S)
        B, S, _ = x.shape
        h = torch.zeros(B, S, self.index_embedding.embedding_dim, device=x.device)
        for i, proj in enumerate(self.feature_projs):
            h += proj(x[..., i:i+1])

        idx = torch.arange(S, device=x.device)
        h += self.index_embedding(idx)

        h += (final_mask == 1).unsqueeze(-1).float() * \
              self.final_on_emb(torch.zeros((), dtype=torch.long, device=x.device))
        h += (final_mask == 2).unsqueeze(-1).float() * \
              self.final_off_emb(torch.zeros((), dtype=torch.long, device=x.device))
        return self.norm(h)


class SyndromeTransformerLayer(nn.Module):
    """Modified layer with proper grid handling"""
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_stabilizers: int,
        grid_size: int,
        pair_embed_dim: int = 48,
        use_dilated_convs: bool = True
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.grid_size = grid_size
        self.num_stabilizers = num_stabilizers
        
        # Determine if we have a valid grid structure
        self.has_grid = (grid_size > 0) and (grid_size * grid_size == num_stabilizers)
        
        # Initialize grid-based components if valid grid exists
        if self.has_grid:
            spatial = grid_size
            coords = [(i//spatial, i%spatial) for i in range(num_stabilizers)]
            static_idx = torch.zeros(num_stabilizers, num_stabilizers, 8, dtype=torch.long)
            for i in range(num_stabilizers):
                r_i, c_i = coords[i]
                for j in range(num_stabilizers):
                    r_j, c_j = coords[j]
                    dx = r_i - r_j
                    dy = c_i - c_j
                    manh = abs(dx) + abs(dy)
                    same = int((r_i + c_i) % 2 == (r_j + c_j) % 2)
                    static_idx[i,j] = torch.tensor([
                        r_i, c_i, r_j, c_j,
                        dx + grid_size,
                        dy + grid_size,
                        manh,
                        same
                    ], dtype=torch.long)
            self.register_buffer('static_idx', static_idx)

            dilations = (1, 2, 4) if use_dilated_convs else (1,)
            self.convs = nn.ModuleList([
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=d, dilation=d)
                for d in dilations
            ])
        else:
            self.convs = nn.ModuleList()

        # Initialize embeddings
        max_coord = 2 * grid_size + 1 if self.has_grid else 1
        emb_dim = pair_embed_dim // 8
        self.row_emb = nn.Embedding(max_coord, emb_dim)
        self.col_emb = nn.Embedding(max_coord, emb_dim)
        self.dx_emb = nn.Embedding(2 * grid_size + 1, emb_dim)
        self.dy_emb = nn.Embedding(2 * grid_size + 1, emb_dim)
        self.manh_emb = nn.Embedding(2 * grid_size + 1, emb_dim)
        self.same_emb = nn.Embedding(2, emb_dim)
        
        self.pair_mlp = nn.Sequential(
            nn.Linear(8 * emb_dim, pair_embed_dim),
            nn.ReLU(),
            nn.Linear(pair_embed_dim, pair_embed_dim)
        )

        # Attention projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.bias_proj = nn.Linear(pair_embed_dim + 7, num_heads)

        # Feedforward network
        self.ff_proj = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.ff_gate = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.ff_out = nn.Linear(4 * hidden_dim, hidden_dim)

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, state: torch.Tensor, events: torch.Tensor, prev_events: torch.Tensor) -> torch.Tensor:
        B, S, D = state.shape
        
        state = self.norm1(state)               

     

  
        if self.has_grid and self.convs:
            h2d = state.view(B, d, d, D).permute(0,3,1,2)
            conv_out = sum(conv(h2d) for conv in self.convs)
            conv_out = conv_out.permute(0,2,3,1).reshape(B,S,D)
            state = self.norm3(state + conv_out)   # becomes new norm3

        # Static pairwise bias features
        if self.has_grid:
            sti = self.static_idx.unsqueeze(0).expand(B, -1, -1, -1).to(state.device)
            e = torch.cat([
                self.row_emb(sti[..., 0]), 
                self.col_emb(sti[..., 1]),
                self.row_emb(sti[..., 2]), 
                self.col_emb(sti[..., 3]),
                self.dx_emb(sti[..., 4]),  
                self.dy_emb(sti[..., 5]),
                self.manh_emb(sti[..., 6]), 
                self.same_emb(sti[..., 7])
            ], dim=-1)
            static_bias = self.pair_mlp(e)
        else:
            static_bias = torch.zeros(B, S, S, self.pair_mlp[-1].out_features, 
                                   device=state.device)

        # Attention computation
        qkv = self.qkv_proj(state)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Dynamic event bias (7 channels)
        evt = events.float()
        pev = prev_events.float()
        eye = torch.eye(S, device=state.device).unsqueeze(0)
        
        dyn = torch.stack([
            evt.unsqueeze(2) * evt.unsqueeze(1),   # f1
            evt.unsqueeze(2) * pev.unsqueeze(1),    # f2
            pev.unsqueeze(2) * evt.unsqueeze(1),    # f3
            pev.unsqueeze(2) * pev.unsqueeze(1),    # f4
            eye * evt.unsqueeze(2),                 # f5
            eye * pev.unsqueeze(2),                 # f6
            eye * (evt * pev).unsqueeze(2)          # f7
        ], dim=-1)

        # Combine biases and apply
        bias_input = torch.cat([static_bias, dyn], dim=-1)
        bias = self.bias_proj(bias_input).permute(0, 3, 1, 2)  # [B, num_heads, S, S]
        scores = scores + bias

        # Attention and output projection
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        state = self.norm2(state + self.o_proj(out))

        # Position-wise feedforward
        proj = self.ff_proj(state)
        gate = torch.sigmoid(self.ff_gate(state))
        ff_out = self.ff_out(proj * gate)
        state = self.norm3(state + ff_out)
        
        return state

class SyndromeTransformer(nn.Module):
    """N stacked SyndromeTransformerLayer blocks."""
    def __init__(self,
                 hidden_dim, num_heads, num_layers,
                 num_stabilizers, grid_size,
                 use_dilated_convs=True):
        super().__init__()
        self.layers = nn.ModuleList([
            SyndromeTransformerLayer(hidden_dim, num_heads,
                                     num_stabilizers, grid_size,
                                     use_dilated_convs=use_dilated_convs)
            for _ in range(num_layers)])

    def forward(self, state, events, prev_events):
        for layer in self.layers:
            state = layer(state, events, prev_events)
        return state


class ReadoutNetwork(nn.Module):
    """Map stabiliser grid → final logit."""
    def __init__(self, hidden_dim: int, grid_size: int):
        super().__init__()
        self.data_conv = nn.Conv2d(hidden_dim, hidden_dim, 2)
        self.line_mlp  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.grid_size = grid_size

    def forward(self, state, basis):
        B, S, D = state.shape
        d = self.grid_size
        h2d = torch.cat([state.new_zeros(B,1,D), state], dim=1)\
              .transpose(1,2).view(B, D, d+1, d+1)
        df = self.data_conv(h2d)                           # (B,D,d,d)
        flat = df.view(B, D, d, d).permute(0,2,3,1)        # (B,d,d,D)

        outs = []
        for i in range(B):
            grid = flat[i]
            lines = grid.mean(dim=1) if basis[i]==0 else grid.mean(dim=0)
            outs.append(self.line_mlp(lines).squeeze(-1).mean())
        return torch.stack(outs)


class AlphaQubitDecoder(nn.Module):
    """Full AlphaQubit network end-to-end."""
    def __init__(self,
                 num_features, hidden_dim,
                 num_stabilizers, grid_size,
                 num_heads=8, num_layers=12):
        super().__init__()
        self.embedder = StabilizerEmbedder(num_features, hidden_dim,
                                           num_stabilizers)
        self.transformer = SyndromeTransformer(hidden_dim, num_heads,
                                               num_layers,
                                               num_stabilizers, grid_size)
        self.readout = ReadoutNetwork(hidden_dim, grid_size)

    def forward(self, inputs, basis, final_mask):
        B, R, S, F = inputs.shape
        state = torch.zeros(B, S,
                            self.embedder.index_embedding.embedding_dim,
                            device=inputs.device)
        prev_evt = torch.zeros(B, S, device=inputs.device)
        for r in range(R):
            x = inputs[:, r]
            emb = self.embedder(x, final_mask if r==R-1
                                else torch.zeros_like(final_mask))
            state = (state + emb) / math.sqrt(2.0)
            evt = x[..., 0]
            state = self.transformer(state, evt, prev_evt)
            prev_evt = evt
        return self.readout(state, basis)


# ---------------------------------------------------------------------
#  Dataset with auto-padding
# ---------------------------------------------------------------------

class PauliPlusDataset(Dataset):
    """
    Loads syndromes/logicals and zero-pads dummy stabilisers so that
    S+1 is a perfect square (required by AlphaQubit geometry).
    Returns:
        features : (R, S_pad, F)
        basis_id : scalar long
        final_mask: (S_pad,)  (1 = on, 2 = off)
    """
    def __init__(self, synd_path: str, log_path: str, basis_id: int):
        y = np.load(log_path)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1)   # (N,)
        N = len(self.y)

        x = np.load(synd_path)                                   # raw syndromes

        # --- reshape to (N, R, S, C) ---------------------------
        if x.ndim == 2:                     # (N,S) or (R,S)
            if x.shape[0] == N:
                x = x[:, None, :, None]
            else:                           # single sample
                x = x[None, :, :, None]
        elif x.ndim == 3:                   # (N,R,S) or (R,S,C)
            if x.shape[0] != N:
                x = x[None, ...]            # single sample
            if x.ndim == 3:                 # no feature axis
                x = x[..., None]
        elif x.ndim != 4:
            raise ValueError(f"{synd_path}: unsupported shape {x.shape}")

        # --- append basis-ID channel ---------------------------
        basis_feat = np.full((*x.shape[:-1], 1), basis_id, dtype=x.dtype)
        x = np.concatenate([x, basis_feat], axis=-1)             # (N,R,S,F)

        # --- pad stabilisers -----------------------------------
        N, R, S, F = x.shape
        d = math.isqrt(S + 1)
        if d*d != S + 1:             # not a perfect square -> pad zeros
            d += 1
            pad = d*d - 1 - S
            x = np.concatenate([x, np.zeros((N, R, pad, F), dtype=x.dtype)],
                               axis=2)
            S += pad
        self.X = torch.tensor(x, dtype=torch.float32)            # (N,R,S,F)
        self.basis = torch.full((N,), basis_id, dtype=torch.long)

        # final-round on/off mask
        self.final_mask = torch.zeros(N, S, dtype=torch.long)
        for idx, (r, c) in enumerate([(i//d, i%d) for i in range(1, S+1)]):
            self.final_mask[:, idx] = 1 if (r+c)%2==0 else 2

    def __len__(self):  return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx], self.basis[idx], self.final_mask[idx]), self.y[idx]


# ---------------------------------------------------------------------
#  Helper functions & training loop
# ---------------------------------------------------------------------

def discover_all_pairs(out_dir: str) -> List[Tuple[str, str, int]]:
    pairs = []
    for basis, bid in [("x",0), ("z",1)]:
        for s in glob(os.path.join(out_dir, f"*syndromes_{basis}_*.npy")):
            l = s.replace("syndromes", "logicals")
            if os.path.exists(l):
                pairs.append((s, l, bid))
    return sorted(pairs)


def build_dataset(pairs) -> Dataset:
    datasets = [PauliPlusDataset(s,l,b) for s,l,b in pairs]
    return datasets[0] if len(datasets)==1 else ConcatDataset(datasets)


def train(model, tr_loader, va_loader, epochs, lr, device):
    
    # Wrap model with DataParallel if multiple devices are available
    if device.type == "npu" and hasattr(torch, "npu"):
        npu_count = getattr(torch.npu, "device_count", lambda: 1)()
        if npu_count > 1:
            print(f"Using {npu_count} NPUs!")
            model = nn.DataParallel(model)
    elif device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    

    model.to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs+1):
        # ---- training ----
        model.train(); tot=0
        for (xb, basis, mask), yb in tqdm(tr_loader, desc=f"Epoch {ep}/{epochs}"):
            xb, basis, mask, yb = xb.to(device), basis.to(device), \
                                  mask.to(device), yb.to(device)
            logits = model(xb, basis, mask)
            l = loss(logits, yb)
            opt.zero_grad(); l.backward(); opt.step()
            tot += l.item()*xb.size(0)
        print(f"Epoch {ep}  train_loss = {tot/len(tr_loader.dataset):.4f}")

        # ---- validation ----
        model.eval(); vtot=0; correct=0
        with torch.no_grad():
            for (xb,basis,mask), yb in va_loader:
                xb, basis, mask, yb = xb.to(device), basis.to(device), \
                                      mask.to(device), yb.to(device)
                logits = model(xb, basis, mask)
                vtot += loss(logits, yb).item()*xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds==yb).sum().item()
        print(f"          val_loss  = {vtot/len(va_loader.dataset):.4f}   "
              f"acc = {correct/len(va_loader.dataset):.4f}")


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--npu", action="store_true", help="Use available NPUs for training")
    args = p.parse_args()

    torch.manual_seed(42)

    # Device configuration
    if args.npu:
        if hasattr(torch, "npu") and torch.npu.is_available():
            device = torch.device("npu")
            use_data_parallel = getattr(torch.npu, "device_count", lambda: 1)() > 1
        else:
            print("Warning: --npu specified but NPU support is unavailable.")
            print("This could be due to:")
            print("1. Ascend-specific PyTorch not installed")
            print("2. No Ascend NPU devices detected")
            print("3. NPU drivers not properly configured")
            print("Falling back to CUDA/CPU...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            use_data_parallel = torch.cuda.device_count() > 1
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_data_parallel = torch.cuda.device_count() > 1



    pairs = discover_all_pairs(args.output_dir)
    if not pairs:
        raise RuntimeError(f"No *syndromes*.npy / *logicals*.npy files in {args.output_dir}")

    dataset = build_dataset(pairs)

    # split
    n = len(dataset)
    idx = torch.randperm(n)
    split = int(0.9*n)
    tr_ds = torch.utils.data.Subset(dataset, idx[:split])
    va_ds = torch.utils.data.Subset(dataset, idx[split:])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)

    # examine one sample
    (x0, _, _), _ = dataset[0]
    R, S, F = x0.shape
    d = int(math.sqrt(S+1)); grid_size = d-1
    print(f"Rounds={R}  Stabilisers={S}  Features={F}  grid={d}×{d}")

    model = AlphaQubitDecoder(F, 256, S, grid_size, num_heads=8, num_layers=12)

        # In your main code where you load the checkpoint:
    ckpt = "alphaqubit.pth"
    if os.path.exists(ckpt):
        try:
            state_dict = torch.load(ckpt, map_location="cpu")
            # Handle loading a DataParallel model
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            print("Loaded checkpoint", ckpt)
        except Exception as e:
            print("[warn] checkpoint mismatch - starting fresh:", e)
    
    
    # And when saving:
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), ckpt)  # Save only the module state_dict
    else:
        torch.save(model.state_dict(), ckpt)
    print("Saved model ->", ckpt)
