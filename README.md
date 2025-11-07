# AlphaQubit：量子纠错模拟与机器学习解码工具包

## 概述

AlphaQubit 提供从量子纠错仿真数据生成、模型训练到解码评估的完整流程。仓库包含可配置的噪声模型、基于 PyTorch 的训练脚本以及数据可视化工具，帮助研究者快速开展量子误码校正实验。

## 功能特性

- **数据生成**：支持检测误差模型（DEM）、电路去极化噪声（SI1000）以及泄漏、串扰与软读出（Pauli+）等多种噪声类型。
- **配置实验**：`configs/` 目录提供 YAML 配置文件，可灵活设置噪声与训练参数。
- **表面码仿真**：`simulator/` 内含表面码仿真器，实现论文级别的物理噪声模型。
- **模型管理**：训练脚本统一把权重保存至 `ai_models/models/`，解码脚本会在该目录及仓库根目录、`ai_models/checkpoints/` 等位置自动搜寻 `.pth` 文件。
- **可视化工具**：提供 `npy_viewer.py`（.npy 查看）与 `plot_alphaqubit_results.py`（性能曲线绘制）。

## 模型架构与数据说明

- **基础设施**：解码器基于 PyTorch 实现，核心由三部分组成：稳定子特征嵌入器、堆叠的改良 Transformer 层（内部使用 DeepSeek MLA 多头注意力）、以及读取逻辑误差对数似然的卷积+MLP 头部。【F:ai_models/model_mla.py†L16-L143】
- **输入特征**：每个样本的张量形状为 `(N, R, S, 3)`，其中第 0 通道是离散检测事件比特，第 1、2 通道分别为软读出后计算态与泄漏态的后验概率；这些通道由仿真脚本在采样时将检测比特与两路 soft channel 拼接而成。【F:google_qec_simulator/data_manager.py†L5-L28】【F:google_qec_simulator/experiment_simulator.py†L24-L33】
- **输出含义**：模型输出单个标量对数几率（logit），与 `BCEWithLogitsLoss` 搭配训练，表示对应逻辑比特发生错误的概率；推理时可通过 `torch.sigmoid` 转换为 [0,1] 概率。【F:ai_models/model_mla.py†L79-L143】【F:ai_models/model_mla.py†L188-L194】
- **参数规模**：针对随仓库提供的 `d=5` 软读出样例（稳定子数 24、特征通道 3），默认隐藏维度 256、12 层、8 头的 AlphaQubit 解码器共有 8,242,177 个可训练参数。【354d2a†L1-L27】
- **训练样本**：示例数据集 `simulated_data/samples_surface_code_bX_d5_r01_center_5_5.npz` 含 20,000 条样本，形状 `(20000, 1, 24, 3)`；批量预训练脚本 `make_all_pretraining_noise.py` 的默认配置会合成约 $8.5\times10^6$ 条离散综合样本与 4.0M 次 soft shots。【96f910†L1-L8】【F:README.md†L102-L103】

## 安装

```bash
git clone https://github.com/xuda1979/ALPHAQUBIT.git
cd ALPHAQUBIT
# 可选：创建并激活虚拟环境
python3.8 -m venv venv
source venv/bin/activate
# 安装依赖（任选其一）
pip install --upgrade pip
pip install numpy scipy stim pyyaml torch leakysim>=0.4.0
# 或使用项目罗列的依赖清单
pip install -r requirements.txt

# 如需在华为 Ascend NPU 上训练，请安装带有 torch.npu 的 PyTorch 发行版并根据官方文档完成驱动配置。
```

## 仓库结构

```plaintext
├── ai_models/                   # 模型训练与解码脚本
├── configs/                     # 实验配置 YAML 文件
├── simulator/                   # 量子纠错仿真器
├── simulated_data/              # google_qec_simulator 生成的 .npz（脚本可自动建立）
├── generate_data.py             # 训练数据生成脚本
├── npy_viewer.py                # .npy 数据查看工具
├── plot_alphaqubit_results.py   # 解码性能绘制脚本
├── models/                      # 可选：手动放置 .pth 权重的目录
└── README.md                    # 项目说明
```

> **提示**：真实实验 Stim 电路存放在外部目录
> ``~/work/google_qec3v5_experiment_data``，仓库内不再附带 `experiment_data/` 副本。

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
- `google_qec_simulator` 可直接生成 `.npz` 噪声文件，并通过 `--device {cpu,cuda,npu}` 指定计算硬件：

```bash
python -m google_qec_simulator.main path/to/exp --shots 10000 --device npu
# 默认输出写入 pretrain_data/samples_<实验文件夹>.npz，可用 --out 自定义位置
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
# 若实验 Stim 位于额外目录（例如 Google QEC 数据集），可重复传入 --experiment-root：
# python make_all_pretraining_noise.py --out-dir pretrain_data --experiment-root ~/work/google_qec3v5_experiment_data
```

该脚本会依次调用 `generate_data.py`（DEM/SI1000）与 `run_create_all_samples.py`（soft/IQ），并把生成的 `.npy`/`.npz` 文件收集到 `--out-dir` 指定的目录，同时写出 `MANIFEST.json`（记录时间戳与所有产物）。soft/IQ 噪声会按照实验名称自动分目录存放，例如 `pretrain_data/<experiment>/samples_*.npz`，便于按实验拆分训练数据。上述配置对应约 $8.5\times10^6$ 条离散综合样本与 4.0M 次 soft shots，需要约 15 GB（布尔综合）+1.1 GB（soft shots）存储。如资源受限，可按比例缩放各 `--*-samples`，保持不同噪声类型的相对比重。

#### 批量生成实验数据

若 `~/work/google_qec3v5_experiment_data/` 中含有多个实验子目录，可批量调用模拟器生成 `samples_<experiment>.npz`：

```bash
python run_create_all_samples.py --shots 2000
python run_create_all_samples.py --skip-existing --device npu  # 支持跳过已生成文件与 NPU 加速
# 支持多目录：python run_create_all_samples.py ~/work/google_qec3v5_experiment_data /path/to/others
```

脚本会递归查找含 `.stim` 的实验目录，将输出写入 `pretrain_data/`，并按相对路径命名，例如 `pretrain_data/samples_folder_subfolder.npz`。
该流程直接调用 `google_qec_simulator.main.simulate_folder` 完成采样，因此运行期间会实时输出 `[scan]`/`[sample]` 等进度信息，便于观察当前实验与累计耗时。
因此上述单行命令会自动遍历 `~/work/google_qec3v5_experiment_data/` 中的全部实验并依次生成噪声。

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

批量训练 `pretrain_data/`（或其它目录）中的所有实验：

```bash
python run_training_all.py
python run_training_all.py --npu  # 自动检测 Ascend NPU 并行调度
python run_training_all.py --epochs 1 --batch-size 32 --max-samples 1024  # 本地快速冒烟
# 自动生成缺失数据时可指定 Stim 根目录：
# python run_training_all.py --experiment-root ~/work/google_qec3v5_experiment_data
```

在运行批量训练前，建议先执行上一节的一键脚本：

```bash
python make_all_pretraining_noise.py --out-dir pretrain_data
```

它会为每个实验生成并复制 soft/IQ `.npz` 至 `pretrain_data/<experiment>/`，随后 `run_training_all.py` 会优先遍历该目录为每个数据集调用 `ai_models/model_mla.py` 并把权重写入 `ai_models/models/NAME.pth`；若 `pretrain_data/` 为空，则自动回退到 `simulated_data/**/*.npz`。通过 `--epochs`、`--batch-size` 与 `--max-samples` 可以快速调整训练时长；其中 `--max-samples` 会在加载数据集后裁剪样本数，便于在 CPU 上进行冒烟测试。如需自定义数据目录，可重复传入 `--data-root <path>`，该选项会覆盖默认搜索路径（因此若要保留默认目录，请一并显式给出）。脚本会自动串行调度任务；在支持 Ascend NPU 并传入 `--npu` 时，会检测可用设备并并行分配训练进程，因此无需单独的串行脚本。

### 4. 解码与评估

#### 单个数据集

```bash
python ai_models/decode.py \
  --model finetuned_models/alphaqubit_sycamore_runs.pth \
  --data output/dem_syndromes_z_20240229_101530.npy
```

结果写入 `results/`。如需自定义目录，可使用 `--results-dir`。示例命令默认假设权重来自仓库根目录下的 `finetuned_models/` 文件夹——`ai_models/fine_tune.py` 与批量脚本 `run_fine_tune_all.py` 会在该目录生成面向具体实验微调过的 `alphaqubit_<folder>.pth` 权重，解码阶段推荐直接使用这些模型。

#### 批量解码

```bash
python run_decode_all.py --model finetuned_models/
python run_decode_all.py --model finetuned_models/alphaqubit_sycamore_runs.pth
```

- `--model` 支持传入单个文件、目录或留空（若仅检测到一个模型则自动使用）。
- 当仅提供文件名时，脚本会优先在仓库根目录下的 `finetuned_models/` 目录查找微调权重，随后再依次检查 `ai_models/checkpoints/`、`ai_models/models/`、`checkpoints/` 与 `models/` 等位置。
- 默认遍历 `output/`，可通过 `--data-root` 或通配符（如 `"simulated_data/*.npz"`）指定其它数据源。
- `--predictions-dir` 保存逐次测量概率，`--skip-existing` 跳过已生成指标，`--dry-run` 仅打印执行计划。
- 若未生成综合数据，脚本会提示未找到解码目标。请使用前述数据生成脚本准备独立采样的测试集，以可靠评估泛化性能。

批量脚本同样适用于微调后模型：`ai_models/fine_tune.py` 与 `run_fine_tune_all.py` 会在 `finetuned_models/` 目录生成 `alphaqubit_<folder>.pth`，也会被自动发现。

所有批量解码结果默认写入 `results/`，若传入目录则为每个模型创建子目录（例如 `results/alphaqubit_dem/`），同时可通过 `--predictions-dir` 输出 `*_probs.npy`。

### 5. 可视化

`plot_alphaquibit_results.py` 可以对经过微调的实验文件夹计算 LER 并绘制柱状图：

```bash
python plot_alphaquibit_results.py --data-root ~/work/google_qec3v5_experiment_data/sycamore_runs --model-dir finetuned_models
```

- `--data-root`：包含多个实验子目录（每个子目录需含 `detection_events.b8`、`obs_flips_actual.01` 等文件）。
- `--model-dir`：对应的 `alphaqubit_<folder>.pth` 模型所在目录，默认为 `finetuned_models/`。

脚本会读取各子目录、按文件名解析轮数 `rXX`，再载入同名权重计算逻辑错误率，并绘制汇总图。

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

