# 动作识别数据集分析工具集

围绕“原始数据 → 标准化 → 可视化/统计分析”的完整流程，提供可扩展的工具与脚手架：读取与转换标注、抽样导出、统计分析（类别分布、BBox 尺寸、IoU 重合、时空管长度）等。

## 特性

- 标注读取与抽样：基于 LabelMe JSON 的目录遍历、按类别随机抽样、导出上下文帧
- 统计分析：
  - 单目录与多目录聚合的类别分布、归一化 BBox 尺寸与异常统计
  - 检测框重合（IoU）分布与阈值计数、每框重合度直方图、动作/动作对重合排名
  - 时空管长度分布（LabelMe/部分 PKL 数据集）
- 数据格式脚手架：标准 `StandardDataset`/CSV 结构与占位转换器，便于后续接入自定义数据源
- CLI 一键化：导出样例、可视化、统计与合并平铺

## 环境与安装

前置：Python ≥ 3.10（见 `pyproject.toml`）。依赖：`numpy`、`pandas`、`pillow`、`opencv-python`（可选，用于图像绘制/可视化）。

使用 Conda（推荐，与本项目常用环境一致）：

```bash
conda create -n ada python=3.10 -y
conda activate ada
pip install -e .
```

或使用纯 pip：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

若不安装为包，仅在 datasets 脚本目录下运行，请先设置：

```bash
export PYTHONPATH="src:${PYTHONPATH}"
```

## 目录结构

```text
src/action_data_analysis/
├─ datasets/                 # 标准数据结构
│  ├─ __init__.py
│  └─ schema.py              # StandardRecord/StandardDataset/Metadata
├─ io/                       # I/O 与工具
│  ├─ json.py                # LabelMe 读取、shape 解析（含 bbox 与动作）
│  └─ pkl.py                 # PKL 汇总（占位）
├─ convert/                  # 转换管道（占位实现）
│  ├─ from_pkl.py            # PKL → 标准 JSON
│  ├─ from_raw.py            # 原始目录 → 标准 JSON
│  └─ to_csv.py              # 标准 JSON → CSV
├─ analyze/                  # 分析与导出
│  ├─ visual.py              # 上下文帧可视化（OpenCV）
│  ├─ stats.py               # 类别/BBox/时空管聚合统计
│  ├─ overlap.py             # IoU 重合统计
│  ├─ tube_lengths.py        # 时空管长度分布（LabelMe/FineSports/MultiSports）
│  └─ export.py              # 抽样导出与平铺合并
└─ cli/main.py               # 命令行入口
```

更多：
- 文档：`docs/datasets/standard_format.md`（标准 JSON/CSV 结构）
- 示例：`examples/minimal.py`、`examples/inspect_finesports_pkl.py`
- 数据脚本：`datasets/sportshhi/scripts/README.md`

## 快速开始（CLI）

CLI 通过模块方式运行（尚未注册 console_scripts）：

```bash
python -m action_data_analysis.cli.main -h | cat
```

- 导出每类样例与上下文帧（复制图像与 JSON）：

```bash
python -m action_data_analysis.cli.main export-samples \
  /path/to/LabelMe_dir_or_root ... \
  --out /path/to/output/json \
  --dataset-name MultiSports \
  --per-class 3 --context 6
```

- 平铺合并多个导出目录，统一重写 JSON 的 `imagePath`：

```bash
python -m action_data_analysis.cli.main merge-flatten \
  /path/to/output/json/MultiSports \
  /path/to/output/json/SportsHHI \
  --out /path/to/output/flat
```

- 可视化按类别抽样的上下文帧（在图像上绘制 bbox 与动作）：

```bash
python -m action_data_analysis.cli.main visualize-samples \
  /path/to/LabelMe_dir_or_root ... \
  --out /path/to/output/vis \
  --per-class 3 --context 25
```

- 统计单目录（类别分布、BBox 尺寸、异常、时空管）：

```bash
python -m action_data_analysis.cli.main analyze-folder \
  /path/to/LabelMe_dir \
  --out /path/to/output/dir
```

- 聚合统计多目录：

```bash
python -m action_data_analysis.cli.main analyze-dirs \
  /path/to/LabelMe_root_or_dir ... \
  --out /path/to/output/dir
```

- IoU 重合统计（含每框重合度直方图、动作/动作对重合 Top）：

```bash
python -m action_data_analysis.cli.main analyze-overlaps \
  /path/to/LabelMe_root_or_dir ... \
  --out /path/to/output/dir \
  --thresholds 0.1 0.5 0.7
```

- 时空管长度分布：

```bash
# LabelMe 目录（自动 stride 推断：MultiSports/FineSports=1，SportsHHI=5）
python -m action_data_analysis.cli.main analyze-tube-lengths \
  labelme /path/to/LabelMe_root_or_dir ... \
  --out /path/to/output/tubes

# FineSports / MultiSports（基于 PKL）
python -m action_data_analysis.cli.main analyze-tube-lengths \
  finesports /path/to/FineSports-GT.pkl --out /path/to/output/tubes
python -m action_data_analysis.cli.main analyze-tube-lengths \
  multisports /path/to/MultiSports-GT.pkl --out /path/to/output/tubes
```

## Python API 示例

```python
from action_data_analysis.analyze.stats import compute_labelme_folder_stats, render_stats_markdown

stats = compute_labelme_folder_stats("/path/to/LabelMe_dir")
md = render_stats_markdown(stats)
print(md)
```

## 数据格式

- LabelMe（读取字段最小集合）：`imagePath`, `imageWidth`, `imageHeight`, `shapes[*].points/attributes/flags/group_id`
- 标准结构（骨架，便于异构数据统一）：见 `docs/datasets/standard_format.md`

```text
StandardDataset:
- version: string
- records: StandardRecord[]
- metadata: Metadata

StandardRecord:
- video_path: string
- action_label: string
- start_frame?: int
- end_frame?: int
- metadata?: dict
```

转换模块（占位，按需扩展）：
- `convert/from_pkl.py`、`convert/from_raw.py`、`convert/to_csv.py`

## 标准数据集格式（std）

面向文件系统的“标准数据集”结构，便于统一浏览、抽样与统计分析。由源目录（如 `MultiSports_json`、`FineSports_json`）转换得到一个同级的 `_std` 目录。

- **顶层命名**：`<源目录名>_std`，例如：
  - `MultiSports_json` → `MultiSports_json_std`
  - `FineSports_json` → `FineSports_json_std`
- **子目录**：
  - `videos/`：按“样例文件夹”为单位组织；每个样例文件夹包含抽帧图片与对应的 LabelMe JSON 标注。
  - `stats/`：统计结果（预留，可为空）。后续分析会在此生成聚合统计（如 `class_distribution.csv`、`stats.json`）。
- **样例定义**：一个样例是一个目录，目录中包含该视频片段的抽帧图像与对应的标注 JSON。
- **去重策略**：目标中若已存在同名样例目录，自动追加 `_1`, `_2`, ...

### MultiSports → std（样例组织）
- 在源目录中选取“最深层且包含图像”的目录作为一个样例，复制/移动整个样例目录到 `videos/` 下。
- 样例目录名默认保持与源目录一致（若重名则按去重策略追加后缀）。

示例目录结构：
```text
MultiSports_json_std/
├─ videos/
│  ├─ 000123_000456/            # 一个样例（目录名示例，实际以源目录名为准）
│  │  ├─ 000001.jpg
│  │  ├─ 000002.jpg
│  │  └─ 000001.json            # 对应 LabelMe 标注（可能每帧或关键帧）
│  └─ 000789_000999/
│     ├─ ...
│     └─ ...
└─ stats/                        # 预留统计目录，可为空
```

### FineSports → std（扁平命名：动作_编号）
- 源结构通常为两层：`动作/编号/样例内容`。转换后在 `videos/` 下合并为一层，目录名为：`动作_编号/`。
- 若 `动作/` 下直接是样例文件（无编号子目录），则约定为 `动作_0/`。
- 命名冲突时按去重策略追加 `_1`, `_2`, ...

示例目录结构：
```text
FineSports_json_std/
├─ videos/
│  ├─ DribbleJumper_07159_1/
│  │  ├─ 000001.jpg
│  │  ├─ 000002.jpg
│  │  └─ 000001.json
│  └─ Basket_00003/
│     ├─ ...
│     └─ ...
└─ stats/
```

### 转换脚本与用法
- **默认行为**：复制（不改动源目录）。如需移动，追加 `--move`。
- **运行前**（若未安装为包）：确保环境包含源代码路径：
  ```bash
  export PYTHONPATH="src:${PYTHONPATH}"
  ```
- **MultiSports**：
  ```bash
  datasets/multisports/scripts/multisports_to_std.py /path/to/MultiSports_json
  # 或移动：
  datasets/multisports/scripts/multisports_to_std.py /path/to/MultiSports_json --move
  ```
- **FineSports**：
  ```bash
  datasets/finesports/scripts/finesports_to_std.py /path/to/FineSports_json
  # 或移动：
  datasets/finesports/scripts/finesports_to_std.py /path/to/FineSports_json --move
  ```

## 约束与路线图

- 已实现：
  - LabelMe 读取/抽样/可视化、类别/BBox/IoU/时空管统计、样例导出与平铺合并
- 占位与待实现：
  - `io.pkl.summarize_pkl_file`、`convert/*` 的标准化转换与校验
  - `analyze.stats.compute_dataset_stats`（兼容函数名，当前留空）
- 兼容性：
  - IoU/统计在纯 LabelMe JSON 即可运行；OpenCV 仅用于绘制/可视化

## 开发

安装开发依赖并运行质量工具：

```bash
pip install -e .[dev]
ruff check src tests
black --check src tests
pytest -q
```

## 相关脚本（SportsHHI）

更完整的对齐、复制与 IoU 分析流程，请参考：`datasets/sportshhi/scripts/README.md`。

## License

本项目采用 MIT License，详见根目录 `LICENSE`。

## 致谢

感谢 MultiSports、FineSports、SportsHHI 等数据集的公开与社区工具生态。
