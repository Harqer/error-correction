import re
import math
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_models.pauli_plus_dataset import PauliPlusDataset
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

        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, basis):
        B, S, D = x.shape
        d = self.grid_size

        x = torch.cat([x.new_zeros(B, 1, D), x], dim=1)
        x = x.transpose(1, 2).view(B, D, d+1, d+1)

        x = self.conv(x).permute(0, 2, 3, 1)

        outputs = []
        for i in range(B):
            if basis[i] == 0:
                lines = x[i].mean(dim=1)
            else:
                lines = x[i].mean(dim=0)

            logits = self.mlp(lines).squeeze()
            outputs.append(logits.mean())

        return torch.stack(outputs)


class AlphaQubitDecoder(nn.Module):
    def __init__(self, num_features, hidden_dim, num_stabilizers, grid_size, num_heads=8, num_layers=12):
        super().__init__()
        self.embedder = StabilizerEmbedder(num_features, hidden_dim, num_stabilizers)
        self.transformer = SyndromeTransformer(hidden_dim, num_heads, num_layers, num_stabilizers, grid_size)
        self.readout = ReadoutNetwork(hidden_dim, grid_size)

        for layer in self.readout.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, inputs, basis, final_mask):
        B, R, S, F = inputs.shape
        state = torch.zeros(B, S, self.embedder.index_embed.embedding_dim, device=inputs.device)
        prev_events = torch.zeros(B, S, device=inputs.device)

        for r in range(R):
            x = inputs[:, r]
            emb = self.embedder(x, final_mask if r == R-1 else torch.zeros_like(final_mask))
            state = (state + emb) / math.sqrt(2.0)
            state = self.transformer(state, x[..., 0], prev_events)
            prev_events = x[..., 0]

        return self.readout(state, basis)


def get_basis_from_filename(filename):
    name = filename.lower()
    if "_bx_" in name:
        return 0
    if "_bz_" in name:
        return 1
    return -1


def model_stem_from_npz(npz_path) -> str:
    p = Path(npz_path)
    stem = p.stem
    if stem.startswith("samples_"):
        stem = stem[len("samples_"):]
    for anchor in ("pretrain_data", "simulated_data"):
        try:
            rel = p.parent.resolve().relative_to(Path(anchor).resolve())
            if str(rel) != ".":
                stem = "_".join(list(rel.parts) + [stem])
                break
        except Exception:
            continue
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem


def train(
    model,
    tr_loader,
    va_loader,
    epochs,
    lr,
    device,
    model_save_path,
    tqdm_position: int = 0,
    *,
    device_label: str = "",
    tqdm_kwargs=None,
):
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(tr_loader), epochs=epochs
    )
    criterion = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    last_save_time = datetime.now()

    base_tqdm_kwargs = dict(tqdm_kwargs or {})
    if "position" not in base_tqdm_kwargs:
        base_tqdm_kwargs["position"] = max(int(tqdm_position), 0)
    base_tqdm_kwargs.setdefault("dynamic_ncols", True)

    desc_prefix = f"[{device_label}] " if device_label else ""

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        train_pbar = tqdm(
            tr_loader,
            total=len(tr_loader),
            desc=f"{desc_prefix}Epoch {epoch}/{epochs}",
            **dict(base_tqdm_kwargs),
        )
        for (xb, basis, mask), yb in train_pbar:
            xb, basis, mask, yb = xb.to(device), basis.to(device), mask.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb, basis, mask)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            train_pbar.set_postfix_str(f"loss={loss.item():.3f}")
            current_time = datetime.now()
            if current_time - last_save_time >= timedelta(minutes=10):
                torch.save(model.state_dict(), model_save_path)
                last_save_time = current_time
        train_pbar.close()

        model.eval()
        val_loss = 0
        correct = 0
        val_pbar = tqdm(
            va_loader,
            total=len(va_loader),
            desc=f"{desc_prefix}Val {epoch}/{epochs}",
            **dict(base_tqdm_kwargs),
        )
        with torch.no_grad():
            for (xb, basis, mask), yb in val_pbar:
                xb, basis, mask, yb = xb.to(device), basis.to(device), mask.to(device), yb.to(device)
                outputs = model(xb, basis, mask)
                val_loss += criterion(outputs, yb).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == yb).sum().item()
        val_pbar.close()

        avg_loss = total_loss / len(tr_loader)
        val_loss = val_loss / len(va_loader)
        val_acc = correct / len(va_loader.dataset)
        print(
            f"{desc_prefix}Epoch {epoch}: Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_file", required=True, help="Path to the NPZ file containing training data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--npu", action="store_true", help="Use available NPUs for training")
    parser.add_argument(
        "--device_index",
        type=int,
        default=0,
        help="Device index for the selected accelerator backend (e.g., 0, 1, 2, …)",
    )
    parser.add_argument("--tqdm_position", type=int, default=0, help="Row position for tqdm progress (multi-run)")
    parser.add_argument(
        "--model-save-path",
        type=str,
        default=None,
        help="Explicit checkpoint path (.pth). Defaults to ai_models/models/<stem>.pth",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optionally cap the number of samples loaded from the dataset (useful for smoke tests)",
    )
    args = parser.parse_args()

    use_npu = bool(
        args.npu
        and hasattr(torch, "npu")
        and getattr(torch.npu, "is_available", lambda: False)()
    )
    if use_npu:
        idx = int(args.device_index)
        torch.npu.set_device(idx)
        device = torch.device(f"npu:{idx}")
        device_str = f"npu:{idx}"
    elif torch.cuda.is_available():
        idx = int(args.device_index)
        torch.cuda.set_device(idx)
        device = torch.device(f"cuda:{idx}")
        device_str = f"cuda:{idx}"
    else:
        device = torch.device("cpu")
        device_str = "cpu"
    print(f"Using device: {device_str}")

    tqdm_kw = dict(position=max(int(args.tqdm_position), 0), dynamic_ncols=True)

    basis = get_basis_from_filename(args.npz_file)
    if basis == -1:
        print("Warning: Could not determine basis from filename, defaulting to X basis (0)")
        basis = 0

    dataset = PauliPlusDataset(args.npz_file, basis)
    total_samples = len(dataset)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive when provided")
        total_samples = min(total_samples, args.max_samples)

    if total_samples < 2:
        raise RuntimeError("Dataset must contain at least two samples for train/val split")

    perm = torch.randperm(len(dataset))[:total_samples]
    split = max(1, int(0.9 * total_samples))

    train_indices = perm[:split].tolist()
    val_indices = perm[split:].tolist()

    tr_ds = torch.utils.data.Subset(dataset, train_indices)
    va_ds = torch.utils.data.Subset(dataset, val_indices)

    pin_memory = device.type in {"cuda", "npu"}
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)

    (x0, _, _), _ = dataset[0]
    R, S, F = x0.shape
    d = int(math.sqrt(S + 1))
    grid_size = d - 1
    print(f"Rounds={R} Stabilisers={S} Features={F} grid={d}×{d}")

    model = AlphaQubitDecoder(F, 256, S, grid_size, num_heads=8, num_layers=12).to(device)

    if args.model_save_path:
        model_save_path = Path(args.model_save_path)
    else:
        model_dir = Path(__file__).resolve().parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = model_dir / f"{model_stem_from_npz(args.npz_file)}.pth"

    train(
        model,
        tr_loader,
        va_loader,
        args.epochs,
        args.lr,
        device,
        str(model_save_path),
        tqdm_position=args.tqdm_position,
        device_label=device_str,
        tqdm_kwargs=tqdm_kw,
    )
    print(f"Training complete.\nModel saved to {model_save_path}")
