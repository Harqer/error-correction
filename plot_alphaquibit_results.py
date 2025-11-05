# File: plot_alphaqubit_results.py

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from ai_models.model import AlphaQubitDecoder
from ai_models.fine_tune import SingleFolderDataset
import argparse

# —— 解析命令行参数 —— 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="Path to experiment subfolders")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(os.getcwd(), "finetuned_models"),
        help="Directory where alphaqubit_*.pth models live",
    )

    return parser.parse_args()
    

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# —— 模型加载 —— 
def load_model(model_path, folder_path):
    # …
    # infer basis from folder name
    folder_name = os.path.basename(folder_path)
    basis = 'Z' if '_bZ_' in folder_name else 'X'

    S = ds.bits_per_shot
    model = AlphaQubitDecoder(
        num_features=1,
        hidden_dim=128,
        num_stabilizers=S,
        grid_size=0,
        num_heads=4,
        num_layers=3,
        dilation=1,
        basis=basis
    )
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model, ds


# —— 计算 LER —— 
import re
import numpy as np
import torch
from torch.utils.data import DataLoader

def compute_epsilon(model, ds, rounds):
    """
    Compute the Logical Error per Round ε by inverting:
      E = 1 - accuracy
      E = ½[1 - (1 - 2ε)^n]  ⇒  ε = ½[1 - (1 - 2E)^(1/n)]
    """
    loader = DataLoader(ds, batch_size=512, pin_memory=True)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(model.device), y.to(model.device)
            logits = model(x).view(-1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    E = 1 - (correct / total)
    # avoid numerical issues
    one_minus_2E = max(0.0, min(1.0, 1 - 2 * E))
    ε = 0.5 * (1 - one_minus_2E ** (1 / rounds))
    return ε


# —— 主流程 —— 
if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root

    folders = [os.path.join(data_root, d) for d in os.listdir(data_root)
               if os.path.isdir(os.path.join(data_root, d))]

    folder_names = []
    ler_values = []

    for folder in sorted(folders):
        folder_name = os.path.basename(folder)
        # e.g. "..._r25_..."
        m = re.search(r"_r(\d+)_", folder_name)
        if not m:
            print(f"[SKIP] can't parse rounds from {folder_name}")
            continue
        rounds = int(m.group(1))
        model_path = os.path.join(args.model_dir, f"alphaqubit_{folder_name}.pth")

        model, ds = load_model(model_path, folder)
        ε = compute_epsilon(model, ds, rounds)
        folder_names.append(f"{folder_name} (ε)")
        ler_values.append(ε)


    # —— 绘图 —— 
    x = np.arange(len(folder_names))
    plt.figure(figsize=(12, 6))
    plt.bar(x, ler_values, color='tab:blue')
    plt.xticks(x, folder_names, rotation=75, ha='right')
    plt.ylabel("Logical Error Rate (LER)")
    plt.title("Logical Error Rate per Experiment Folder (Fine-Tuned Models)")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()