# 动作识别数据集分析模板（骨架）

目标：围绕“原始数据 → 标准 JSON → 简化 CSV → 可视/统计分析”的流程，提供可扩展的最小骨架。

## 目录映射

```text
src/action_data_analysis/
├─ datasets/                 # 数据集结构与类型
│  ├─ __init__.py            # 导出 StandardRecord / StandardDataset / Metadata
│  └─ schema.py              # 标准 JSON 的 TypedDict 架构
├─ io/                       # I/O：读取/写出/元数据提取
│  ├─ __init__.py
│  ├─ pkl.py                 # PKL 元数据汇总（占位）
│  └─ json.py                # 标准 JSON 读写（占位）
├─ convert/                  # 转换：raw/pkl → standard.json → csv
│  ├─ __init__.py
│  ├─ from_pkl.py            # PKL → 标准 JSON（占位）
│  ├─ from_raw.py            # 原始目录 → 标准 JSON（占位）
│  └─ to_csv.py              # 标准 JSON → CSV（占位）
├─ analyze/                  # 分析：视觉/统计
│  ├─ __init__.py
│  ├─ visual.py              # 视觉分析（占位）
│  └─ stats.py               # 统计分析（占位）
└─ cli/
   ├─ __init__.py
   └─ main.py                # CLI 骨架（argparse，占位）
```

相关文档与示例：
- `docs/datasets/standard_format.md`：标准 JSON/CSV 结构说明（骨架）
- `examples/minimal.py`：保持模板示例，附带新模块的引用注释

## 标准格式（摘要）

- JSON 顶层结构：`StandardDataset { version, records[], metadata }`
- `records[i]`：`{ video_path, action_label, start_frame?, end_frame?, metadata? }`
- CSV 列：`video_path, action_label`

## 预期工作流（占位 API）

1) 原始数据分析/PKL 元数据记录：
- `action_data_analysis.io.pkl.summarize_pkl_file(pkl_path)`

2) 转换至标准格式：
- `action_data_analysis.convert.convert_pkl_to_standard_json(pkl_path, output_json_path)`
- `action_data_analysis.convert.convert_raw_to_standard_json(raw_dir, output_json_path)`
- `action_data_analysis.convert.standard_json_to_csv(json_path, output_csv_path)`

3) 数据集分析：
- 视觉分析：`action_data_analysis.analyze.visual.visualize_dataset(json_or_csv_path, output_dir)`
- 统计分析：`action_data_analysis.analyze.stats.compute_dataset_stats(json_or_csv_path)`

> 以上函数目前均为占位实现，便于根据具体数据源定制。

## CLI（骨架）

- 入口：`src/action_data_analysis/cli/main.py`（`argparse`）
- 子命令（规划）：
  - `convert`：格式转换（`pkl/raw -> standard.json -> csv`）
  - `analyze`：视觉/统计分析（输入 `json/csv`）

> 后续可将 CLI 入口挂接为可执行脚本（在 `pyproject.toml` 中添加 `project.scripts`）。
