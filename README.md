# AlphaQubit：量子纠错模拟与机器学习解码工具包

## 概述

AlphaQubit 提供从量子纠错仿真数据生成、模型训练到解码评估的完整流程。支持多种噪声模型，基于 PyTorch 实现，并附带可视化工具，帮助研究者快速开展量子误码校正实验。

## 功能特性

- **数据生成**：支持检测误差模型（DEM）、电路去极化噪声（SI1000）以及泄漏、串扰与软读出（Pauli+）等多种噪声类型
- **配置实验**：通过 `configs/` 目录下的 YAML 文件灵活调整实验参数
- **表面码仿真**：`simulator/` 中实现量子表面码仿真器
- **模型权重**：仓库不再内置 `.pth` 文件；训练脚本会把新模型统一保存在 `ai_models/models/`，解码脚本会自动在该目录与仓库根目录、`ai_models/checkpoints/` 等位置查找
- **可视化工具**：包括 `npy_viewer.py`（.npy 数据查看）和 `plot_alphaqubit_results.py`（绘制性能曲线）
- **PyTorch 集成**：在 `ai_models/` 中提供训练与推理脚本

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

# 如需在华为 Ascend NPU 上训练，请安装带有 `torch.npu` 的 PyTorch 发行版并根据官方文档完成驱动配置。
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
├── models/                      # 可选：将训练或下载得到的 .pth 权重放在此处，脚本会自动搜索
└── README.md                    # 项目说明
```

## 使用方法

### 1. 生成数据

```bash
# 检测误差模型（DEM）
python generate_data.py --model dem --samples 10000

# 电路去极化噪声（SI1000）
python generate_data.py --model si1000 --samples 10000

# 泄漏、串扰与软读出（Pauli+）
python generate_data.py --model pauli_plus --samples 10000

# 与论文对齐的物理噪声模型（直接调用本仓库内 paper 的实现）
python generate_data.py --model paper_aligned --basis z --samples 10000
# 或 X 基
python generate_data.py --model paper_aligned --basis x --samples 10000
# 使用 --basis 参数在 X 与 Z 基间切换
```

### 一键生成所有预训练噪声数据

```bash
# From within the repo root
python make_all_pretraining_noise.py \
  --dem-samples 2500000 \
  --si1000-samples 1500000 \
  --si1000-p-grid 0.004,0.008,0.012,0.016 \
  --soft-shots 4000000 \
  --soft-device auto \
  --out-dir pretrain_data
```

上述命令基于一次性预备约 $8.5\times10^6$ 条离散综合样本（DEM $2.5$M + SI1000 四个 $p$ 点各 $1.5$M）以及每个 soft 电路 $4.0$M 次 IQ shots 的估算，足以覆盖默认 20 epoch 训练时约 $1.7\times10^8$ 个样本步，使 600 万参数的解码器平均获得 $\approx28$ 次梯度观测。该规模需要约 15 GB（布尔综合数据）+ 1.1 GB（soft shots）的存储空间；若资源受限，可按比例缩放所有 `--*-samples` 并保持不同噪声类型之间的相对比重。

生成的数据默认保存在 `output/` 目录。

使用 `google_qec_simulator` 直接生成 `.npz` 噪声文件时，可通过 `--device` 指定
计算设备，例如：

```bash
python google_qec_simulator/main.py path/to/exp --shots 10000 --device npu
```

在支持的硬件上选用 `--device npu`（或 `cuda`）可显著加速 soft 通道的采样过程。

### 批量为所有实验生成噪声数据

若 `experiment_data/` 下包含多个实验子目录，可使用批处理脚本依次调用模拟器，
一次性写出所有 `samples_<experiment>.npz`：

```bash
python run_create_all_samples.py --shots 2000
# 可选：跳过已存在的 .npz 并在 Ascend NPU 上运行
python run_create_all_samples.py --skip-existing --device npu
```

脚本会递归发现 `experiment_data/` 中含有 `.stim` 的子目录，并将输出保存至
`simulated_data/`。输出文件名基于相对路径生成，默认格式为
`simulated_data/samples_<子目录层级以_连接>.npz`。

### 2. 查看数据

生成的数据文件名会包含时间戳，例如 `output/dem_syndromes_z_20240229_101530.npy`。查看时可以先列出
`output/` 目录下的 `.npy` 文件，再将其中一个路径传给 `npy_viewer.py`：

```bash
ls output/*.npy
python npy_viewer.py output/dem_syndromes_z_20240229_101530.npy
```

### 3. 训练模型

```bash
python ai_models/train.py --config configs/dem.yaml
```

可在相应的 YAML 文件中调整超参数与噪声设置。

若已按实验子目录在 `simulated_data/` 下生成多个 `samples_<experiment>.npz` 文件，可使用批量入口一次性训练所有实验对应的模型：

```bash
python run_training_all.py
# Ascend NPU 并行调度示例
python run_training_all.py --npu
```

上述脚本会遍历 `simulated_data/*.npz`，逐个调用 `ai_models/model_mla.py`，并在提供 `--npu` 时自动检测可用 Ascend NPU 数量以并行调度任务。

训练完成后，所有新生成的权重都会出现在 `ai_models/models/` 目录中：

```bash
ls ai_models/models
python run_decode_all.py --model ai_models/models/
```

这样即可在完成一次批量训练后，直接用单条命令批量解码与评估。`run_decode_all.py` 会对 `output/` 及其备选目录中的全部综合数据执行推理。

> **输出位置**：若未显式提供 `--results-dir`，脚本会把每个数据集对应的 `*_metrics.json` 写入仓库根目录下的 `results/`，并在同时解码多个模型时自动按模型文件名创建子目录（例如 `results/alphaqubit_dem/`）。若指定了 `--results-dir`，则始终写入该目录；同理，如需保存逐次测量的概率向量，可使用 `--predictions-dir` 指向某个文件夹，脚本会输出 `*_probs.npy`。运行时终端会提示实际的写入路径。

### 4. 解码与评估

#### 单个文件

```bash
python ai_models/decode.py \
  --model alphaqubit_model.pth \
  --data output/dem_syndromes_z_20240229_101530.npy
```

解码结果（如逻辑错误率）将存于 `results/` 目录。

#### 一键批量解码所有数据集

批处理脚本 **不会** 替你下载模型，它只接受已经存在的 `.pth` 权重。请先确认模型文件放在什么位置：

- `ai_models/train.py` 默认把 `dem/si1000/pauli_plus/paper_aligned` 四种配置分别保存为仓库根目录下的 `alphaqubit_<模型类型>.pth`，可以用 `--model-path` 改写输出路径。
- `ai_models/model_mla.py`（`run_training_all.py` 的底层入口）会把 `simulated_data/samples_NAME.npz` 训练出的模型统一写入 `ai_models/models/NAME.pth`。
- `ai_models/fine_tune.py` 与 `run_fine_tune_all.py` 会针对数据集文件夹 `folder/` 生成 `alphaqubit_<folder>.pth`，同样直接落在仓库根目录。

获得权重后即可批量解码：

```bash
# `--model` 支持单个文件、目录或留空（自动发现唯一的模型）
python run_decode_all.py --model ai_models/models/surface_code_bX_d5_r01_center_5_5.pth
python run_decode_all.py --model ai_models/models/
```

如果只提供文件名，脚本会按顺序在以下目录里搜索：仓库根目录、`ai_models/checkpoints/`、`ai_models/models/`、`checkpoints/`、`models/`。把 `.pth` 复制到这些目录之一，就能用 `python run_decode_all.py --model alphaqubit_model.pth` 调用；若这些目录里只存在一个 `.pth` 文件，也可以直接运行 `python run_decode_all.py`，脚本会自动选中它。若提供目录路径，则会遍历其中的所有 `.pth` 文件并为每个模型分别生成结果。

- 如需快速测试，可先执行前文的 `generate_data.py` 或 `run_create_all_samples.py` 生成 `output/` 下的综合数据；否则脚本会提示未找到解码目标。
- 默认遍历 `output/`，可通过 `--data-root` 指定其它目录，或传入通配符 `python run_decode_all.py --model ... "simulated_data/*.npz"`。
- `--results-dir` 控制指标输出目录，`--predictions-dir` 额外保存逐次测量的预测概率。
- `--skip-existing` 会跳过已经生成指标文件的数据集，支持 `--dry-run` 仅打印将执行的命令。

> **提示**：用于测试与评估的数据应来自模型训练过程中未出现过的独立采样，以便可靠衡量泛化性能。

### 5. 绘制性能

```bash
python plot_alphaqubit_results.py --input results/metrics.json
```

### Paper-aligned 噪声模型参数

`configs/paper_aligned.yaml` 提供了与 Google 论文中物理机制一一对应的参数，主要包括：

| 参数 | 物理机制 |
| --- | --- |
| `T1_us`, `Tphi_us` | 振幅/相位弛豫，经 GPT 处理后注入到所有 1Q 门 |
| `p_cz_crosstalk_ZZ` | 并行 CZ 的 ZZ 串扰，作为相关错误注入 |
| `p_cz_swap_like` | CZ 期间的 swap‑like 误差，建模为 (XX+YY)/2 |
| `p_cz_leak_11_to_02` | 相位诱导的泄漏，近似为相关 ZZ 误差 |
| `p_leak_transport_12_to_30` | 泄漏传输，引入附加的单量子比特 Pauli 噪声 |
| `p_readout`, `p_reset` | 测量与复位的经典翻转误差 |
| `p_heat_01`, `p_heat_12`, `dqlr_matrix` | 多级被动加热与 DQLR 不完美，折算为额外的 Pauli 噪声 |
| `p_1q_excess`, `p_cz_excess`, `p_idle_excess` | 门前后及空闲期间的残余 Pauli 噪声 |

该实现还显式模拟了四能级泄漏传输（`|12⟩→|30⟩`, `|21⟩→|03⟩`）、顺序被动加热（`p_heat_01`, `p_heat_12`）以及三结果读出概率 `[p0, p1, pl]`，完整复现论文描述的噪声过程。

这些参数在 `simulator.PauliPlusSimulator.apply_paper_aligned_noise` 中被消费，确保仿真噪声与论文方法保持一致。此 `paper_aligned` 模式**直接委托**到本仓库随附的 `my_noise_model` 实现，保证与论文方法 100% 一致。

### 默认参数取值（来自论文 Table S4）

| YAML 键 | 数值 | 来源 |
| --- | --- | --- |
| `cycle_ns` | `1076.0` | 循环时间 |
| `T1_us` | `73.0` | 平均 $T_1$ |
| `Tphi_us` | `720.0` | 调整以复现 $0.9\times10^{-2}$ 空闲误差 |
| `p_heat_01` | `0.0` | $|0\rangle \to |1\rangle$ 加热概率 |
| `p_heat_12` | `2.5e-4` | $|1\rangle \to |2\rangle$ 加热概率 |
| `p_readout` | `8.0e-3` | 读出误差 |
| `p_reset` | `1.5e-3` | 复位误差 |
| `p_cz_crosstalk_ZZ` | `5.5e-4` | CZ 串扰 |
| `p_cz_leak_11_to_02` | `2.0e-4` | CZ 泄漏概率 |
| `p_cz_excess` | `2.75e-3` | 其余 CZ 误差以匹配 $3.5\times10^{-3}$ 总误差 |
| `p_1q_excess` | `6.2e-4` | 单量子比特门误差 |

## `my_noise_model` 物理方法详解

为便于复现与二次开发，这里补充 `my_noise_model/` 目录中主要模块的物理含义与
相互关系：

- **`circuit_builder.py`**：在 Stim 生成的表面码理想电路上逐层“穿衣”，把 T₁/T₂
  弛豫、去极化、CZ 串扰、泄漏与读出复位误差全部转化为 Pauli 通道，并借助
  `leakysim` 实现广义 Pauli 托恩 (GPT)。同时暴露软测量模型，便于在生成数据时
  联合采样离散/连续噪声。【F:my_noise_model/circuit_builder.py†L1-L210】
- **`channels.py` 与 `kraus_utils.py`**：提供超导器件各类噪声过程的 Kraus 表示，
  如振幅阻尼、纯退相干、去极化、CZ 诱导泄漏、泄漏运输、被动加热与 DQLR
  复位。所有函数都遵循论文中的参数化方式，确保可组合成合法的 CPTP 通道。
  【F:my_noise_model/channels.py†L1-L176】【F:my_noise_model/kraus_utils.py†L1-L274】
- **`gpt.py` 与 `gpta.py`**：实现 GPT 的线性代数细节，包括 Kraus→Choi→Pauli
  概率的转换、Pauli 传输矩阵对角化、泄漏概率估计等，使任意噪声通道都能以
  Pauli 概率表表示。【F:my_noise_model/gpt.py†L1-L196】【F:my_noise_model/gpta.py†L1-L210】
- **`pauli_twirl.py`**：给出标准/广义 Pauli 托恩的实现，既支持纯量子比特的 PTM
  方法，也支持调用 `leakysim` 获取含泄漏的广义通道概率。【F:my_noise_model/pauli_twirl.py†L1-L109】
- **`pauli_plus.py`**：综合前述工具，根据 YAML 配置产出 Pauli+ 噪声表（Pauli
  概率 + 泄漏率），用于论文中的采样器与模拟器。【F:my_noise_model/pauli_plus.py†L1-L106】
- **`paper_aligned.py`**：封装论文 SI 的完整噪声流程，自动将配置映射到
  :class:`PauliPlusSimulator` 并预计算空闲/复位等通道的 GPT 结果。
  【F:my_noise_model/paper_aligned.py†L1-L109】
- **`iq_readout.py` 与 `softxor.py`**：构建软 I/Q 读出模型与软检测器组合规则，
  支持在泄漏存在时输出三结果后验概率并生成连续测量数据。
  【F:my_noise_model/iq_readout.py†L1-L102】【F:my_noise_model/softxor.py†L1-L33】
- **`si1000.py`**：提供 SI1000 去极化模型的精确权重，可与 Pauli+ 噪声互补使用。
  【F:my_noise_model/si1000.py†L1-L30】

这些模块共同构成了仓库的物理噪声基线，可直接套用或按需替换某一环节，以
研究不同噪声源对表面码性能的影响。

## 全流程示例：从噪声文件生成到 NPU 上的全规模训练

1. **生成噪声样本**  
   使用 `run_create_all_samples.py` 调用 `google_qec_simulator` 为 `simulated_data/` 下的每个电路生成噪声 `.npz` 文件：
   ```bash
   python run_create_all_samples.py
   ```

2. **在 NPU 上训练所有样本**  
   在安装了 Ascend PyTorch（支持 `torch.npu`）的环境中运行：
   ```bash
   python run_training_all.py --npu
   ```
   该脚本会遍历 `simulated_data/*.npz`，并对每个文件执行 `ai_models/model_mla.py` 训练；当检测到多块 NPU 时，会自动将任务分配到所有设备，实现全规模并行训练。

3. **解码与评估**  
   训练完成后，可继续使用前述的 `ai_models/decode.py` 及可视化脚本对模型性能进行评估与展示。

## 配置

编辑 `configs/` 下的 YAML 文件，可自定义电路布局、误差率、样本数量及训练参数。

## 预训练模型

使用内置的预训练模型，无需重新训练即可快速进行解码：

```bash
python ai_models/decode.py --model alphaqubit_model.pth --data output/si1000_samples.npy
```

## 许可证

本项目基于 MIT 协议开源。

## 引用

若在研究中使用本工具包，请引用：

```bibtex
@article{Xu2024AlphaQubit,
  title={Learning high-accuracy error decoding for quantum processors},
  author={Xu, David and Google Quantum AI},
  journal={Nature},
  year={2025},
}
```

