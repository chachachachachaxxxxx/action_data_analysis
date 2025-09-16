### scripts 目录说明（SportsHHI）

本目录包含与 SportsHHI 数据相关的解析、转换、对齐与合并的脚本。多数脚本依赖项目源码路径，请在运行前设置：

```bash
export PYTHONPATH="src:${PYTHONPATH}"
```

默认数据与输出路径多为绝对路径，建议按需在命令行参数中覆盖。

---

#### 1) 对齐与导出对齐 CSV（以 SportsHHI 第一帧为基准）
- 文件：`align_sh_first_to_ms_export_csv.py`
- 作用：
  - 对每个同时存在于 MultiSports 与 SportsHHI 的视频，固定 SportsHHI 的第一帧（两侧共同最小帧号），在 MultiSports 中向后搜索最相似帧（均值绝对像素差〈threshold 即认为命中），导出对齐结果 CSV。
- 主要参数：`--multisports_root`、`--sportshhi_root`、`--out_csv`、`--max-k`、`--threshold`
- 产物：对齐结果 CSV，列含 `video_id, sh_index, sh_file, ms_index, ms_file, k, mean, status`
- 示例：
```bash
datasets/sportshhi/scripts/run_align_sh_first_to_ms_export_csv.sh \
  --multisports-root /path/to/MultiSports_json \
  --sportshhi-root   /path/to/SportsHHI_json \
  --out-csv          /path/to/alignment_search.csv \
  --max-k 20 --threshold 3.0
```

辅助脚本：`run_align_sh_first_to_ms_export_csv.sh`（包装器，带参数解析和默认路径）。

---

#### 2) 逐视频第一帧差异度（RGB/GRAY）
- 文件：`compute_first_frame_gray_diff.py`
- 作用：
  - 选定视频在两数据集中共同的第一帧，计算 RGB 或灰度空间下的像素均值绝对差，用于粗评对齐难度。
- 主要参数：`--multisports_root`、`--sportshhi_root`、`--video_id`、`--space {rgb,gray}`
- 输出：打印单行结果，包括 `mean_abs_diff`。
- 示例：
```bash
python -m datasets.sportshhi.scripts.compute_first_frame_gray_diff \
  --multisports_root /path/to/MultiSports_json \
  --sportshhi_root   /path/to/SportsHHI_json \
  --video_id v_xxx --space rgb
```

---

#### 3) 基于对齐 CSV 的复制与重合度统计（IoU）
- 文件：`copy_and_compare_from_alignment.py`
- 作用：
  - 读取对齐 CSV（如第 1 步产物），从两侧复制对齐帧及对应 LabelMe JSON 到目标目录，并在公共标注帧上统计检测框的 IoU 重合度（包括阈值占比、均值、逐帧明细）。
- 主要参数：
  - `--alignment_csv`、`--multisports_root`、`--sportshhi_root`、`--out_root`
  - `--allow_status`（如：`ok,no_below_threshold`）
  - `--thresholds`（如：`"0.5 0.75 0.9"`）
  - `--limit`、`--max_frames`
- 产物：
  - 复制后的帧与 JSON：`out_root/<video_id>/...`
  - 统计：`out_root/__stats__/overlap_summary.csv` 与 `overlap_by_frame.csv`
- 示例（包装器）：
```bash
datasets/sportshhi/scripts/run_overlap_analysis.sh
```

辅助脚本：`run_overlap_analysis.sh`（封装常用参数、阈值等）。

---

#### 4) 基于对齐 CSV 的 JSON 合并（以 SportsHHI 图像为基）
- 文件：`merge_json_from_alignment.py`
- 作用：
  - 读取对齐 CSV，使用 SportsHHI 的图像尺寸为基准，将 MultiSports 与 SportsHHI 的 LabelMe shapes 按需缩放/对齐后合并到同一 JSON，便于后续统一使用。
- 主要参数：
  - `--alignment_csv`、`--multisports_root`、`--sportshhi_root`、`--out_root`
  - `--allow_status`（如：`ok,no_below_threshold`）
  - `--limit`、`--max_frames`
- 产物：`out_root/<video_id>/*.json`（合并后的 LabelMe JSON），以及复制的相关图像。
- 示例（包装器）：
```bash
datasets/sportshhi/scripts/run_merge_json.sh
```

---

#### 5) 转换 SportsHHI 篮球标注 到 LabelMe JSON 并复制帧
- 文件：`convert_to_labelme_json.py`
- 作用：
  - 读取 `basketball_annotations` 下的 CSV（train/val）与 `sports_action_list.pbtxt`，将有标注的帧复制到输出目录，并为每帧生成 LabelMe JSON。可选地：均衡抽样、限制视频/帧数、包含副框（team=1）。
- 主要参数：`--rawframes_src`、`--ann_dir`、`--out_root`、`--limit_videos`、`--limit_frames`、`--balanced_per_action`、`--include_secondary`、`--filter_video`
- 产物：`out_root/<video_id>/{*.jpg,*.json}`
- 示例（包装器）：
```bash
datasets/sportshhi/scripts/convert_to_labelme.sh \
  --rawframes /path/to/rawframes \
  --ann       /path/to/basketball_annotations \
  --out       /path/to/SportsHHI_json \
  --limit_videos 3 --limit_frames 50 --include_secondary
```

辅助脚本：`convert_to_labelme.sh`（带日志与健壮的参数拼接）。

---

#### 6) 解析 SportsHHI 原始标注并生成统计报告
- 文件：`parse_annotations.py`
- 作用：
  - 解析 `annotations/` 下 CSV/PKL/PBTXT，合并 train/val，附加类别元数据，计算数据规模、类别/运动分布、框尺寸与质量检查、视频级 TopN 等，输出 JSON 与 Markdown 报告。
- 产物（默认写入 `datasets/sportshhi/results/`）：
  - `annotations_summary.json`、`README_annotations.md` 等（实际文件名以脚本实现为准）。
- 示例：
```bash
python -m datasets.sportshhi.scripts.parse_annotations
```

---

#### 7) 提取“篮球”子集标注
- 文件：`extract_basketball.py`
- 作用：
  - 基于 `sports_action_list.pbtxt` 中以 `basketball` 开头的类别，过滤 `annotations/` 下的 CSV/PKL，写出仅含篮球的 CSV/PKL 与相应的 PBTXT 到 `basketball_annotations/`。
- 产物：`basketball_annotations/{sports_train_v1.csv, sports_val_v1.csv, sports_action_list.pbtxt, *.pkl}`
- 示例：
```bash
python -m datasets.sportshhi.scripts.extract_basketball
```

---

#### 8) 合并与统计一体化（封装脚本）
- 文件：`align_merge_multisports_sportshhi.sh`
- 作用：
  - 一键对齐 MultiSports 与 SportsHHI 的公共帧，合并两侧 LabelMe JSON，并在公共标注帧上统计 IoU≥阈值 的匹配数量。
- 说明：
  - 该脚本调用同目录下的 Python 实现（`align_merge_and_compare.py`）。如缺失请先补齐或改为直接使用第 1、3、4 步的单步脚本。
- 产物：
  - 合并 JSON：`--out/<video_id>/*.json`
  - 统计：`--out/__stats__/iou_counts.{json,csv}`
- 示例：
```bash
datasets/sportshhi/scripts/align_merge_multisports_sportshhi.sh \
  --multispor ts-root /path/to/MultiSports_json \
  --sportshhi-root   /path/to/SportsHHI_json \
  --out             /path/to/MultiSports_SportsHHI_overlap_json \
  --iou 0.99
```

---

### 备注
- 绝对路径和默认值以代码内实现为准，必要时用命令行参数覆盖。
- 若需在子环境执行，请确保依赖（numpy、pandas、Pillow 或 OpenCV 等）已安装。


