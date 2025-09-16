# src 目录说明（action_data_analysis）

本目录包含动作识别数据集分析与转换的核心实现，围绕“标准数据集（std）”组织，支持标注读取、样例导出、统计与可视化、以及标签映射转换到 `converted`。

## 结构与职责

```text
src/action_data_analysis/
├─ analyze/          # 统计与可视化
│  ├─ export.py      # 样例导出与平铺合并；std 目录发现辅助
│  ├─ stats.py       # 类别分布、bbox 归一化尺寸、分辨率与时空管统计
│  ├─ overlap.py     # IoU 重合统计：全局与按动作/动作对
│  ├─ tube_lengths.py# tube 长度/时间分箱统计与绘图
│  └─ visual.py      # 上下文帧可视化（OpenCV 可选）
├─ cli/              # 命令行入口与文档
│  ├─ main.py        # CLI 子命令（export、analyze-*、convert-std 等）
│  └─ README.md      # CLI 用法说明与示例
├─ convert/          # 转换管道（可复用的转换逻辑）
│  ├─ label_map_converter.py # LabelMe/COCO 的标签映射、清理与（可选）图片复制
│  ├─ std_converter.py       # 基于 std 的样例级转换（std → converted）
│  ├─ from_pkl.py            # PKL → 标准 JSON（占位）
│  ├─ from_raw.py            # 原始目录 → 标准 JSON（占位）
│  └─ to_csv.py              # 标准 JSON → CSV（占位）
├─ datasets/         # 各数据源相关脚本（示例、工具）
├─ io/               # I/O 层：LabelMe/PKL 解析等
│  ├─ json.py        # LabelMe 遍历与 shape 解析，抽取 bbox/action
│  └─ pkl.py         # PKL 汇总（占位）
└─ core.py           # 核心基类/类型（预留）
```

## 标准数据集（std）

- 约定结构：
  ```text
  <Dataset>_std/
  ├─ videos/
  │  └─ <sample>/  # 抽帧图片与对应的 LabelMe JSON
  └─ stats/         # 统计输出，初期可为空
  ```
- 发现规则：`analyze/export.py` 提供 `_discover_std_sample_folders`，从 `<root>/videos/*` 收集样例目录。
- 转换：`convert/std_converter.py` 将 std 转换为 `<root>_converted`，支持按 CSV 标签映射、样例级删除（无 JSON）与整样例图片复制。

## 关键能力与入口

- 标注读取与解析：`io/json.py` 的 `iter_labelme_dir`、`extract_bbox_and_action`
- 样例发现（std）：`analyze/export.py` 的 `_discover_std_sample_folders`
- 统计：
  - 单样例目录：`analyze/stats.py: compute_labelme_folder_stats`
  - 多目录聚合：`analyze/stats.py: compute_aggregate_stats`
  - IoU 重合：`analyze/overlap.py: compute_aggregate_overlaps`
  - Tube 长度/时间分箱：`analyze/tube_lengths.py`
- 转换：
  - 标签映射（通用）：`convert/label_map_converter.py: convert_dataset_dir`
  - std → converted：`convert/std_converter.py: convert_std_dataset`

## CLI（推荐）

- 入口：`python -m action_data_analysis.cli.main -h`
- 文档：`src/action_data_analysis/cli/README.md`

示例：
```bash
# 统计 std 数据集
python -m action_data_analysis.cli.main analyze-dirs /path/to/<Dataset>_std --out output/<Dataset>_analysis

# IoU 重合（std）
python -m action_data_analysis.cli.main analyze-overlaps /path/to/<Dataset>_std --out output/<Dataset>_analysis

# std → converted（按映射，样例级清理、整样例图片复制）
python -m action_data_analysis.cli.main convert-std /path/to/<Dataset>_std /path/to/mapping.csv
```

## 模块详解与示例

### io/json.py
- **iter_labelme_dir(folder)**: 遍历目录内的 LabelMe JSON，产出 `(json_path, obj)`。
- **extract_bbox_and_action(shape)**: 从 `shape` 中抽取 `((x1,y1,x2,y2), action_name)`，优先 `attributes.action`，其次 `label`。

示例：
```python
from action_data_analysis.io.json import iter_labelme_dir, extract_bbox_and_action

for json_path, rec in iter_labelme_dir("/path/to/std/videos/sample_001"):
  for sh in rec.get("shapes", []) or []:
    parsed = extract_bbox_and_action(sh)
    if parsed is None:
      continue
    (x1, y1, x2, y2), action = parsed
    # 自定义处理...
```

### analyze/export.py
- **_discover_std_sample_folders(paths)**: 收集 `<root>/videos/*` 样例目录；入参可为 std 根目录、`videos` 目录或具体样例目录。
- **export_samples_with_context(...)**: 从若干 LabelMe 目录按类别采样，导出上下文帧到结构化目录。
- **merge_flatten_datasets(...)**: 平铺合并结构化导出目录，重写 JSON 的 `imagePath` 指向新文件名。

示例（发现 std 样例目录）：
```python
from action_data_analysis.analyze.export import _discover_std_sample_folders

samples = _discover_std_sample_folders(["/path/to/FineSports_json_std"])  # 返回 videos/*
```

示例（合并平铺）：
```python
from action_data_analysis.analyze.export import merge_flatten_datasets

merge_flatten_datasets([
  "/path/to/output/json/FineSports",
  "/path/to/output/json/MultiSports",
], "/path/to/output/flat")
```

### analyze/stats.py
- **compute_labelme_folder_stats(folder)**: 统计单个样例目录（帧目录）的类别分布、bbox 归一化尺寸、异常计数、分辨率与时空管信息。
- **compute_aggregate_stats(folders)**: 聚合多个样例目录的上述统计。
- **render_stats_markdown(stats)**: 渲染统计结果为 Markdown 文本（用于 README_annotations）。

示例：
```python
from action_data_analysis.analyze.stats import compute_labelme_folder_stats, compute_aggregate_stats, render_stats_markdown

single = compute_labelme_folder_stats("/path/to/std/videos/sample_001")
md = render_stats_markdown(single)

multi = compute_aggregate_stats([
  "/path/to/std/videos/sample_001",
  "/path/to/std/videos/sample_002",
])
```

### analyze/overlap.py
- **compute_folder_overlaps(folder, thresholds=[0.1,0.3,0.5])**: 单目录内框间 IoU 分布、阈值计数、每框度分布、按动作/动作对统计。
- **compute_aggregate_overlaps(folders, thresholds=...)**: 多目录聚合的 IoU 统计。
- **render_overlaps_markdown(stats)**: 渲染 IoU 统计为 Markdown。

示例：
```python
from action_data_analysis.analyze.overlap import compute_folder_overlaps, compute_aggregate_overlaps

s = compute_folder_overlaps("/path/to/std/videos/sample_001", thresholds=[0.1, 0.5, 0.7])
agg = compute_aggregate_overlaps([
  "/path/to/std/videos/sample_001",
  "/path/to/std/videos/sample_002",
], thresholds=[0.5])
```

### analyze/tube_lengths.py
- **compute_tube_lengths_for_labelme_dirs(folders)**: 以 `(track_id, action)` 连续帧段统计 tube 长度（帧数），返回按动作聚合与 overall 概览。
- **collect_overall_tube_lengths_labelme_dirs(folders)**: 汇总多个目录的所有 tube 长度列表。
- **bin_tube_lengths(lengths)** 与 **plot_length_bins(...)**: 分箱与绘图（无显示环境 `Agg`）。
- **compute_time_bins_for_labelme_dirs(folders, seconds_per_frame)**: 将帧长度换算为秒的三段计数（<=0.5, 0.5-1, 1+）。
- 亦提供 **compute_tube_lengths_for_finesports/multisports**（基于 PKL）与 **compute_tube_lengths_for_sportshhi**（基于 CSV）。

示例：
```python
from action_data_analysis.analyze.tube_lengths import (
  compute_tube_lengths_for_labelme_dirs,
  collect_overall_tube_lengths_labelme_dirs,
  bin_tube_lengths,
)

folders = [
  "/path/to/std/videos/sample_001",
  "/path/to/std/videos/sample_002",
]
res = compute_tube_lengths_for_labelme_dirs(folders)
lengths = collect_overall_tube_lengths_labelme_dirs(folders)
counts = bin_tube_lengths(lengths)
```

### analyze/visual.py
- **visualize_samples_with_context(...)**: 从若干 LabelMe 目录按类别采样，绘制上下文帧（需要 OpenCV）；通常通过 CLI 使用。

示例（CLI）：
```bash
python -m action_data_analysis.cli.main visualize-samples /path/to/std/videos --out output/vis --per-class 3 --context 25
```

### convert/label_map_converter.py
- **load_label_mapping(csv_path)**: 读取包含列 `label,label2` 的映射 CSV；空目标标签表示删除该类标注。
- **convert_single_json(json_path, mapping)**: 自动检测 LabelMe/COCO 并转换 JSON 对象（内存中）。
- **convert_dataset_dir(dataset_dir, mapping_csv, output_dir=None, overwrite=False, copy_images=False)**:
  - 遍历目录树内所有 JSON 并转换；默认输出到同级 `*_converted`；
  - LabelMe 转换后若某 JSON 无任何 `shapes`，则删除该 JSON（或不写出）；
  - `copy_images=True` 时，会就近复制与 JSON 配对的图片文件（按 `imagePath` 或同名推断）。

示例：
```python
from action_data_analysis.convert.label_map_converter import convert_dataset_dir

stats = convert_dataset_dir(
  dataset_dir="/path/to/std/videos/sample_001",
  mapping_csv="/path/to/mapping.csv",
  output_dir="/tmp/sample_001_converted",
  overwrite=False,
  copy_images=True,
)
print(stats)
```

### convert/std_converter.py
- **convert_std_dataset(std_root, mapping_csv, out_root=None, copy_images=True)**：针对 std 根目录（含 `videos/`）：
  - 逐“样例目录”转换标注，若样例转换后完全没有 JSON，则删除整个样例目录；
  - 若样例保留，则复制该样例目录下“所有图片文件”（不仅限于与 JSON 配对的）；
  - 输出到 `<std_root>_converted`（或 `--out` 指定），保留 `videos/` 与 `stats/` 结构，并生成 `conversion_summary.json`。

示例（CLI）：
```bash
python -m action_data_analysis.cli.main convert-std \
  /path/to/FineSports_json_std \
  /path/to/mapping.csv \
  --out /path/to/FineSports_json_std_converted
```

## 复用与扩展建议

- 新数据源优先转为 std 结构，随后使用通用统计与转换。
- 自定义标签归一化：复用 `label_map_converter.convert_dataset_dir`，或在 std 上用 `convert-std`。
- 所有新增模块请补充到本 README 与 CLI README 中，以保持入口一致性与可发现性。
