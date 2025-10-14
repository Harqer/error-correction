# AlphaQubit：量子纠错模拟与机器学习解码工具包

## 概述

AlphaQubit 提供从量子纠错仿真数据生成、模型训练到解码评估的完整流程。仓库包含可配置的噪声模型、基于 PyTorch 的训练脚本以及数据可视化工具，帮助研究者快速开展量子误码校正实验。

## 功能特性

- **数据生成**：支持检测误差模型（DEM）、电路去极化噪声（SI1000）以及泄漏、串扰与软读出（Pauli+）等多种噪声类型。
- **配置实验**：`configs/` 目录提供 YAML 配置文件，可灵活设置噪声与训练参数。
- **表面码仿真**：`simulator/` 内含表面码仿真器，实现论文级别的物理噪声模型。
- **模型管理**：训练脚本统一把权重保存至 `ai_models/models/`，解码脚本会在该目录及仓库根目录、`ai_models/checkpoints/` 等位置自动搜寻 `.pth` 文件。
- **可视化工具**：提供 `npy_viewer.py`（.npy 查看）与 `plot_alphaqubit_results.py`（性能曲线绘制）。

## 安装

```bash
git clone https://github.com/xuda1979/ALPHAQUBIT.git
cd ALPHAQUBIT
# 可选：创建并激活虚拟环境
python3.8 -m venv venv
source venv/bin/activate
# 安装依赖
pip install --upgrade pip
pip install numpy scipy stim pyyaml torch leakysim>=0.4.0

# 如需在华为 Ascend NPU 上训练，请安装带有 torch.npu 的 PyTorch 发行版并根据官方文档完成驱动配置。
```

## 仓库结构

```plaintext
├── ai_models/                   # 模型训练与解码脚本
├── configs/                     # 实验配置 YAML 文件
├── google_experiment_data/      # Google Sycamore 实验数据
├── simulator/                   # 量子纠错仿真器
├── generate_data.py             # 训练数据生成脚本
├── npy_viewer.py                # .npy 数据查看工具
├── plot_alphaqubit_results.py   # 解码性能绘制脚本
├── models/                      # 可选：手动放置 .pth 权重的目录
└── README.md                    # 项目说明
```

## 工作流程

### 1. 生成噪声与综合数据

使用 `generate_data.py` 可按噪声模型生成样本：

```bash
# 检测误差模型（DEM）
python generate_data.py --model dem --samples 10000

# 电路去极化噪声（SI1000）
python generate_data.py --model si1000 --samples 10000

# 泄漏、串扰与软读出（Pauli+）
python generate_data.py --model pauli_plus --samples 10000

# 与论文对齐的物理噪声模型（支持 X/Z 基）
python generate_data.py --model paper_aligned --basis z --samples 10000
python generate_data.py --model paper_aligned --basis x --samples 10000
```

- 默认输出目录为 `output/`，文件名含时间戳，如 `output/dem_syndromes_z_YYYYMMDD_HHMMSS.npy`。
- `google_qec_simulator/main.py` 可直接生成 `.npz` 噪声文件，并通过 `--device {cpu,cuda,npu}` 指定计算硬件：

```bash
python google_qec_simulator/main.py path/to/exp --shots 10000 --device npu
```

#### 预生成大规模预训练数据

仓库提供一键脚本，用于生成论文同等规模的预训练噪声：

```bash
python make_all_pretraining_noise.py \
  --dem-samples 2500000 \
  --si1000-samples 1500000 \
  --si1000-p-grid 0.004,0.008,0.012,0.016 \
  --soft-shots 4000000 \
  --soft-device auto \
  --out-dir pretrain_data
```

该配置对应约 $8.5\times10^6$ 条离散综合样本与 4.0M 次 soft shots，需要约 15 GB（布尔综合）+1.1 GB（soft shots）存储。如资源受限，可按比例缩放各 `--*-samples`，保持不同噪声类型的相对比重。

#### 批量生成实验数据

若 `experiment_data/` 中含有多个实验子目录，可批量调用模拟器生成 `samples_<experiment>.npz`：

```bash
python run_create_all_samples.py --shots 2000
python run_create_all_samples.py --skip-existing --device npu  # 支持跳过已生成文件与 NPU 加速
```

脚本会递归查找含 `.stim` 的实验目录，将输出写入 `simulated_data/`，并按相对路径命名，例如 `simulated_data/samples_folder_subfolder.npz`。

### 2. 检查与浏览数据

列出 `output/` 下的 `.npy` 文件并调用 `npy_viewer.py`：

```bash
ls output/*.npy
python npy_viewer.py output/dem_syndromes_z_20240229_101530.npy
```

### 3. 训练模型

使用单个配置文件训练：

```bash
python ai_models/train.py --config configs/dem.yaml
```

- 可直接在 YAML 中调整超参数与噪声设置。
- `ai_models/train.py` 默认将 `dem/si1000/pauli_plus/paper_aligned` 对应模型保存为仓库根目录下的 `alphaqubit_<模型类型>.pth`，可通过 `--model-path` 修改输出位置。

批量训练 `simulated_data/` 中的所有实验：

```bash
python run_training_all.py
python run_training_all.py --npu  # 自动检测 Ascend NPU 并行调度
```

`run_training_all.py` 会遍历 `simulated_data/*.npz`，为每个数据集调用 `ai_models/model_mla.py` 并把权重写入 `ai_models/models/NAME.pth`。

### 4. 解码与评估

#### 单个数据集

```bash
python ai_models/decode.py \
  --model alphaqubit_model.pth \
  --data output/dem_syndromes_z_20240229_101530.npy
```

结果写入 `results/`。如需自定义目录，可使用 `--results-dir`。

#### 批量解码

```bash
python run_decode_all.py --model ai_models/models/surface_code_bX_d5_r01_center_5_5.pth
python run_decode_all.py --model ai_models/models/
```

- `--model` 支持传入单个文件、目录或留空（若仅检测到一个模型则自动使用）。
- 当仅提供文件名时，脚本会依次在仓库根目录、`ai_models/checkpoints/`、`ai_models/models/`、`checkpoints/` 与 `models/` 中搜索。
- 默认遍历 `output/`，可通过 `--data-root` 或通配符（如 `"simulated_data/*.npz"`）指定其它数据源。
- `--predictions-dir` 保存逐次测量概率，`--skip-existing` 跳过已生成指标，`--dry-run` 仅打印执行计划。
- 若未生成综合数据，脚本会提示未找到解码目标。请使用前述数据生成脚本准备独立采样的测试集，以可靠评估泛化性能。

批量脚本同样适用于微调后模型：`ai_models/fine_tune.py` 与 `run_fine_tune_all.py` 会在仓库根目录生成 `alphaqubit_<folder>.pth`，也会被自动发现。

所有批量解码结果默认写入 `results/`，若传入目录则为每个模型创建子目录（例如 `results/alphaqubit_dem/`），同时可通过 `--predictions-dir` 输出 `*_probs.npy`。

### 5. 可视化

```bash
python plot_alphaqubit_results.py --input results/metrics.json
```

## Paper-aligned 噪声模型参数

`configs/paper_aligned.yaml` 提供了与 Google 论文中物理机制一一对应的参数：

| 参数 | 物理机制 |
| --- | --- |
| `T1_us`, `Tphi_us` | 振幅/相位弛豫，经 GPT 处理后注入到所有 1Q 门 |
| `p_cz_crosstalk_ZZ` | 并行 CZ 的 ZZ 串扰，作为相关错误注入 |
| `p_cz_swap_like` | CZ 期间的 swap-like 误差，建模为 (XX+YY)/2 |
| `p_cz_leak_11_to_02` | 相位诱导的泄漏，近似为相关 ZZ 误差 |
| `p_leak_transport_12_to_30` | 泄漏传输，引入附加的单量子比特 Pauli 噪声 |
| `p_readout`, `p_reset` | 测量与复位的经典翻转误差 |

## 参考建议

- 建议使用独立采样的数据集进行评估，避免与训练数据重合。
- 在支持的硬件上使用 `--device npu` 或 `--device cuda` 可显著加速 soft 通道采样与模型训练。

