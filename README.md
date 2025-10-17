# 动作识别数据集分析工具集

围绕“原始数据 → 标准化（std） → 可视化/统计/导出”的完整流程，提供可扩展的脚手架与 CLI 工具。

## 安装与环境

```bash
conda create -n ada python=3.10 -y
conda activate ada
pip install -e .
# 或
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 未安装为包时（直接运行源码）：
export PYTHONPATH="src:${PYTHONPATH}"
```

## 用法总览（CLI 子命令一览）

详细参数与完整示例请查看 CLI 说明文档：`src/action_data_analysis/cli/README.md`。
也可使用 `-h` 查看任意子命令帮助，例如：

```bash
python -m action_data_analysis.cli.main -h
python -m action_data_analysis.cli.main <subcommand> -h
```

- 标准化/清洗与转换
  - `clean-json`: 清洗 LabelMe JSON（3σ 异常框、过长 tube 剔除）
  - `std-to-pkl`: 将 std（帧级 JSON）导出为标准 PKL
  - `pkl-to-std`: 从标准 PKL 重建/回写 std
  - `convert-std`: 基于标签映射将 `<root>_std → <root>_std_converted`
  - `subset-std`: 从 std 抽取每类至少 N 个样例的代表性子集
  - `filter-std`: 依据样例名前缀 include/exclude，生成新的 std
  - `remove-tubes`: 按 `video_id,track_id` 列表移除对应 tube 标注

- 样例导出与视频生成
  - `export-samples`: 按类别导出样例与上下文帧（复制图像与 JSON），该方法主要是为了查看各类动作具体长什么样，与接下来的merge-flatten可以配合使用
  - `merge-flatten`: 合并多个导出目录并重写 JSON 中的 `imagePath`，即文件结构扁平化处理，便于一次性查看所有样例
  - `export-tubes`: 将 tube 片段导出为 MP4（square/letterbox，fps/尺寸可调）

- 统计与分析（std）
  - `analyze-folder`: 单目录统计（类别分布、BBox 尺寸/异常、时空管）
  - `analyze-dirs`: 多目录聚合统计（同上指标的汇总）
  - `analyze-overlaps`: IoU 重合统计（阈值计数/直方图/动作与动作对重合 Top），该命令是为了统计SportsHHI和MultiSports的重合情况
  - `analyze-tube-lengths`: 按类别统计 tube 长度分布（含异常值定位）
  - `plot-tube-lengths`: 生成 tube 长度分箱图
  - `time-bins`: 按时长（秒）出分箱统计
  - `tube-stats`: 对 tube 数据集（train/val/test + labels_dict）做类别分布与缺失路径统计

- 数据拆分与管道化
  - `split-std`: 从 `<std_root>/videos/*` 自动发现样例并按比例写出 `{train,val,test}.csv`
  - `split-by-tube`: 将每条 tube 拆分为独立样例（仅保留该 tube 的帧与 JSON）
  - `split-by-tube-fpswin`: 按 fps 为窗口切分 tube（支持单数据集输出与 mask.csv）
  - `split-gt-csv`: 将 `gt.csv` 按比例切分为 `{train,val,test}.csv`
  - `merge_tube_datasets`: 合并多个 tube视频 数据集，统一标签空间并重写标注 CSV

## 结构总览

- `clean`: 标注清洗工具（面向标准数据集 `..._std` 的 `videos/*` JSON），支持 3σ 异常框与过长 tube 剔除等；具体用法见 `clean/README.md`。
- `datasets`: 各数据集的适配/转换脚本与说明（如 MultiSports、FineSports、SportsHHI 等）；具体用法见各自子目录的文档或 `scripts/README.md`。
- `scripts`: 用于数据集样例导出查看分析使用，即最早期的查看各个样例，此外还有一键统计的脚本；具体说明见 `scripts/readme.md`。
- `src`: 源码目录（核心在 `src/action_data_analysis/`）：`cli` 为命令入口，`analyze` 负责统计/可视化/导出，`convert` 提供标准化与拆分，`clean` 为清洗实现，`io` 提供 I/O 与工具。具体用法详见`src/action_data_analysis/cli/README.md`
- `resnet_tactical_recognition`: 用于与NBA_TPCE项目的输出配合，进行相关的后处理操作，具体说明写在`resnet_tactical_recognition/find_missing_non_tactical.py`，`resnet_tactical_recognition/readme.md`这里记录相关的命令

