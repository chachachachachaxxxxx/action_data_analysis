# CLI 使用说明（action_data_analysis.cli.main）

## 安装与环境

- 推荐 Conda 环境（项目默认环境名为 `ada`）
- 未安装为包时，需要提前设置：
  ```bash
  export PYTHONPATH="src:${PYTHONPATH}"
  ```

## 标准数据集（std）

- 结构：
  ```text
  <Dataset>_std/
  ├─ videos/
  │  ├─ <sample1>/  # 带图片与 LabelMe JSON
  │  └─ <sample2>/
  └─ stats/         # 可为空
  ```
- 发现规则：CLI 的统计命令会自动从 `<root>/videos/*` 收集样例目录。

## std ↔ pkl 转换

将 std（LabelMe 帧标注目录）与标准 PKL 之间互转。

### std → pkl

```bash
python -m action_data_analysis.cli.main std-to-pkl \
  /path/to/<Dataset>_std \
  --out /path/to/<Dataset>_std \
  --name gt.pkl

# 绝对路径示例
python -m action_data_analysis.cli.main std-to-pkl \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset \
  --name gt.pkl
```

- 输出：`<out>/annotations/gt.pkl`（附带 `labels.json` 方便查看）

### pkl → std

```bash
python -m action_data_analysis.cli.main pkl-to-std \
  /path/to/<Dataset>_std/annotations/gt.pkl \
  /path/to/<Dataset>_std/videos \
  --out /path/to/<Dataset>_std_rebuild \
  --copy-images

# 就地写 JSON（不复制图片）
python -m action_data_analysis.cli.main pkl-to-std \
  /path/to/<Dataset>_std/annotations/gt.pkl \
  /path/to/<Dataset>_std/videos \
  --inplace

# 绝对路径示例
python -m action_data_analysis.cli.main pkl-to-std \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset/annotations/gt.pkl \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset/videos \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset_rebuild \
  --copy-images
```

- 输出：`<out>/videos/<video_id>/{frames+json}` 与 `<out>/annotations/labels.json`

### 标准 PKL 结构

- 顶层（Dict）：`version, labels, videos, nframes, resolution, gttubes`
- `labels: List[str]`：类别名列表，索引即 `class_id`
- `videos: List[str]`：视频/样例 ID 列表（通常与 `gttubes` 的键一致）
- `nframes: Dict[video_id -> int]`：每个样例的帧数（可选，若无法解析可为 0）
- `resolution: Dict[video_id -> (H, W)]`：每个样例的分辨率（像素，高、宽），若未知可为 `(0,0)`
- `gttubes: Dict[video_id -> Dict[class_id -> List[np.ndarray]]]`
  - 每条 tube 是形如 `[T, C]` 的二维数组：
    - 第 1 列：`frame_index`（1-based）
    - 最后 4 列：`x1, y1, x2, y2`（像素坐标，左上-右下）
    - 若无中间附加字段，通常 `C=5`

## 转换：std → converted

将 std 数据集按标签映射转换为 `<root>_converted`，并清理无 JSON 的样例目录。

```bash
python -m action_data_analysis.cli.main convert-std \
  /path/to/FineSports_json_std \
  /path/to/mapping.csv \
  --out /path/to/FineSports_json_std_converted # 可选；默认：同级目录追加 _converted
  # --no-copy-images  # 可选，默认会复制图片到镜像目录

python -m action_data_analysis.cli.main convert-std \
  /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std \
  /storage/wangxinxing/code/action_data_analysis/data/finesports.csv \
  --out /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std_converted

python -m action_data_analysis.cli.main convert-std \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std \
  /storage/wangxinxing/code/action_data_analysis/data/multisports.csv \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted
  
```

- 输出：
  - `<out>/videos/<sample>/...`（转换后的 JSON 与复制的图片，若开启）
  - `<out>/stats/`（预留）
  - `<out>/conversion_summary.json`（转换汇总）
- 清理：若某个样例目录在转换后没有任何 JSON 文件，将被自动删除。

## 抽取代表性子集（std）

从原始 std 数据集中抽取一个“小而全”的代表性子集，保证每个动作至少 N 个样例（样例单位为 `videos/*` 子目录）。

```bash
# 基本用法：每类至少 3 个样例（默认）
python -m action_data_analysis.cli.main subset-std \
  /path/to/<Dataset>_std \
  --out /path/to/<Dataset>_std_subset \
  --per-class 3

# 也可直接传 videos/ 或若干 videos/* 目录（自动发现样例）
python -m action_data_analysis.cli.main subset-std \
  /path/to/<Dataset>_std/videos \
  --per-class 5

# 绝对路径示例
python -m action_data_analysis.cli.main subset-std \
  /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std_converted \
  --out /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std_converted_subset \
  --per-class 10

python -m action_data_analysis.cli.main subset-std \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset \
  --per-class 10

python -m action_data_analysis.cli.main subset-std \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_std_by_tube_fpswin \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_std_by_tube_fpswin_subset \
  --per-class 10
```

- 输出：
  - `<out>/videos/<sample>/...`（被选中的样例目录，包含图片与 JSON）
  - `<out>/stats/`（预留）
  - `<out>/subset_summary.json`（子集摘要：来源、动作列表、每类计数、是否有类不足 N）

注意：若某些动作在原数据中本就不足 N，则 `subset_summary.json` 的 `missing_actions` 会列出仍然不足的类别，以便后续数据补充或调参。

## 导出 tube 视频（std → MP4）

将标准格式（std）的帧与 LabelMe JSON 按 tube（轨迹）导出为 MP4，固定 25fps、224x224。支持两种策略：
- letterbox：裁出 bbox 后按比例缩放并居中贴到 224x224（保比例，黑边填充）
- square：将 bbox 按中心扩为正方形后裁剪，再缩放至 224x224

```bash
# letterbox 策略（推荐保持比例）
python -m action_data_analysis.cli.main export-tubes \
  /path/to/<Dataset>_std \
  --out /path/to/output/tubes_224_letterbox \
  --strategy letterbox --fps 25 --size 224

# square 策略（统一正方形裁剪）
python -m action_data_analysis.cli.main export-tubes \
  /path/to/<Dataset>_std \
  --out /path/to/output/tubes_224_square \
  --strategy square --fps 25 --size 224

# 可选：最小 tube 帧数过滤（小于该长度的 tube 不导出）
python -m action_data_analysis.cli.main export-tubes \
  /path/to/<Dataset>_std \
  --out /path/to/output/tubes_224_letterbox \
  --strategy letterbox --fps 25 --size 224 --min-len 4

python -m action_data_analysis.cli.main export-tubes \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_tube_std_converted_subset/videos \
  --strategy square --fps 25 --size 224

python -m action_data_analysis.cli.main export-tubes \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_tube_std_converted_subset \
  --strategy square --fps 25 --size 224 \
  --labels /storage/wangxinxing/code/action_data_analysis/datasets/multisports/results/labels_dict.json

python -m action_data_analysis.cli.main export-tubes \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_std_by_tube_fpswin \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_tube_std_converted_std_by_tube_fpswin \
  --strategy square --fps 25 --size 224 \
  --labels /storage/wangxinxing/code/action_data_analysis/datasets/multisports/results/labels_dict.json

python -m action_data_analysis.cli.main export-tubes \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_std_by_tube_fpswin_subset_renamed \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_tube_std_converted_std_by_tube_fpswin_subset_renamed \
  --strategy square --fps 25 --size 224 \
  --labels /storage/wangxinxing/code/action_data_analysis/datasets/multisports/results/labels_dict.json

```

- 输入路径可为 std 根目录、`videos` 目录或若干 `videos/*` 样例目录（自动发现样例）。
- 输出结构：`<out>/<action>/<sample>__tid<track>__seg<k>.mp4`。

## 统计命令（仅 std）

- 目录聚合统计：
```bash
  python -m action_data_analysis.cli.main analyze-dirs /path/to/<Dataset>_std --out output/<Dataset>_analysis

python -m action_data_analysis.cli.main analyze-dirs /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std_converted --out /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std_converted/stats

python -m action_data_analysis.cli.main analyze-dirs /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted/stats

python -m action_data_analysis.cli.main analyze-dirs /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset/stats

python -m action_data_analysis.cli.main analyze-dirs /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_std_by_tube_fpswin_subset --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_std_by_tube_fpswin_subset/stats
```

- IoU 重合统计：
```bash
  python -m action_data_analysis.cli.main analyze-overlaps /path/to/<Dataset>_std --out output/<Dataset>_analysis
  ```
- Tube 长度分箱图：
```bash
  python -m action_data_analysis.cli.main plot-tube-lengths /path/to/<Dataset>_std --out output/plot.png

python -m action_data_analysis.cli.main plot-tube-lengths /path/to/<Dataset>_std --out output/plot.png
  ```
- 时长分箱（秒）：
```bash
  python -m action_data_analysis.cli.main time-bins /path/to/<Dataset>_std --spf 0.04 --out output/time
  ```

### 按类别的时空管长度统计

- 功能：统计每个类别的时空管长度分布（帧数），输出 JSON 与 CSV（含 min/p25/mean/p75/max）。

```bash
# LabelMe 目录（std 根、videos 目录或若干 videos/* 样例目录；自动识别样例）
python -m action_data_analysis.cli.main analyze-tube-lengths \
  labelme /path/to/<Dataset>_std \
  --out /path/to/<Dataset>_tube_stats

# 绝对路径示例（LabelMe/std）
python -m action_data_analysis.cli.main analyze-tube-lengths \
  labelme /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset/tube_stats

python -m action_data_analysis.cli.main analyze-tube-lengths \
  labelme /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted/tube_stats

# 基于标准 PKL（FineSports / MultiSports）
python -m action_data_analysis.cli.main analyze-tube-lengths \
  finesports /path/to/FineSports_gt.pkl --out /path/to/<Dataset>_tube_stats
python -m action_data_analysis.cli.main analyze-tube-lengths \
  multisports /path/to/MultiSports_gt.pkl --out /path/to/<Dataset>_tube_stats
```

- 输出：
  - `<out>/labelme_tube_lengths.json` 或 `<out>/{finesports|multisports}_tube_lengths.json`
  - `<out>/labelme_tube_lengths_per_action.csv`（列：action,count,min,p25,mean,p75,max）

### 时空管长度异常值（按类别）与定位（输出中提供）

- 定义：对每个类别计算 Q1、Q3、IQR=Q3-Q1，长度 < Q1-1.5·IQR 或 > Q3+1.5·IQR 的 tube 视为异常值。
- 运行 `analyze-tube-lengths` 后，保存的 JSON 中会包含 `outliers` 字段，提供每类的阈值与异常样例定位信息（不需要额外脚本）。

- 输出结构（节选）：

```json
{
  "per_action": {
    "ActionA": { "count": 123, "summary": {"min":1, "p25": 3, "mean": 5.4, "p75": 7, "max": 40} }
  },
  "overall": { "count": 999, "summary": {"min": 1, "p25": 2, "mean": 4.8, "p75": 8, "max": 60} },
  "outliers": {
    "ActionA": {
      "thresholds": {"q1": 3.0, "q3": 7.0, "iqr": 4.0, "lower": -3.0, "upper": 13.0},
      "low": [
        {"folder": "/path/to/videos/sample_0001", "tid": "12", "start": 5, "end": 6, "length": 2}
      ],
      "high": [
        {"folder": "/path/to/videos/sample_0099", "tid": "3", "start": 100, "end": 150, "length": 51}
      ]
    }
  }
}
```

- 说明：
  - `folder` 为样例目录绝对路径，`tid` 为轨迹/实例 id；`start`、`end` 为该异常 tube 的帧索引范围，`length` 为帧数。
  - 对无异常的类别，`low` 或 `high` 可能为空数组。

## 其他

- 导出样例、平铺合并等见 `README.md` 主文档与 `src/action_data_analysis/analyze/*`。

## 按时空管拆分为新的 std 数据集（std → std_by_tube）

将一个样例目录中包含的多条时空管，拆分为多个新样例；每个新样例只保留该 tube 覆盖到的帧图片以及对应 JSON，且 JSON 中仅保留该 tube 的 shape（同帧其他 tube 标注全部移除）。

```bash
# 基本用法：自动从 <root>/videos/* 发现样例
python -m action_data_analysis.cli.main split-by-tube \
  /path/to/<Dataset>_std \
  --out /path/to/<Dataset>_std_by_tube \
  --min-len 1

python -m action_data_analysis.cli.main split-by-tube \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset_std_by_tube \
  --min-len 1

# 也可直接传 videos 目录或若干样例目录
python -m action_data_analysis.cli.main split-by-tube \
  /path/to/<Dataset>_std/videos/sample_0001 /path/to/<Dataset>_std/videos/sample_0002 \
  --out /path/to/<Dataset>_std_by_tube


```

- 输出结构：
  - `<out>/videos/<orig_sample>__tid<ID>__seg<K>/...`（仅该 tube 的帧与 JSON）
  - `<out>/stats/`（预留）
  - `<out>/split_summary.json`（处理汇总：输入样例数、发现 tube 数、输出新样例数）

- 过滤：`--min-len` 可用来丢弃过短的 tube 片段。

## 按 fps 窗口切分为新的 std 数据集（std → std_by_tube_fpswin）

将 tube 按窗口长度=fps、步长=fps//2 切分：
- 若 tube 段长度 < 窗口长度：以该段中心为窗口中心，向两侧均匀拓展到窗口长度；窗口内缺少该 tube 原始标注的帧，用“最近的有标注帧”的框补全；并且若该窗口超出了有效范围，会滑动至有效范围内
- 若 tube 段长度 ≥ 窗口长度：沿时间轴滑窗（stride=fps//2），同样对缺标帧用最近标注补全。
- 生成 `annotations/mask.csv` 记录每个窗口内哪些帧缺少原始标注（missing=1），以及它们参考的源帧号。



```bash
python -m action_data_analysis.cli.main split-by-tube-fpswin \
  /path/to/<Dataset>_std \
  --out /path/to/<Dataset>_std_by_tube_fpswin \
  --fps 25 \
  --min-len 1

python -m action_data_analysis.cli.main split-by-tube-fpswin \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset_std_by_tube_fpswin \
  --fps 25

python -m action_data_analysis.cli.main split-by-tube-fpswin \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_std_by_tube_fpswin \
  --fps 25
```

- 输出结构：
  - `<out>/videos/<orig_sample>__tid<ID>__seg<K>__win<W>/...`（窗口内所有帧均有 JSON；缺标用最近标注补全）
  - `<out>/annotations/mask.csv`（四列：window,frame_index,missing,source_frame）
  - `<out>/split_fpswin_summary.json`（处理汇总）

## 切分 gt.csv 为 train/val/test

- 功能：
  - 按 train:val:test 比例随机切分
  - 去掉表头、删除 `frames` 列
  - 为视频路径列添加前缀（默认：`videos/`）

```bash
# 基本用法（默认比例 8:1:1，前缀 videos/）
python -m action_data_analysis.cli.main split-gt-csv \
  /path/to/<Dataset>_std_by_tube_fpswin/annotations/gt.csv

# 指定比例/前缀/视频列/输出目录/随机种子
python -m action_data_analysis.cli.main split-gt-csv \
  /path/to/<Dataset>_std_by_tube_fpswin/annotations/gt.csv \
  --ratios 8 1 1 \
  --prefix videos/ \
  --video-col tube \
  --out /path/to/<Dataset>_std_by_tube_fpswin/annotations \
  --seed 42

# 绝对路径示例
python -m action_data_analysis.cli.main split-gt-csv \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_tube_std_converted_std_by_tube_fpswin/annotations/gt.csv \
  --ratios 8 1 1
```

- 输出：在指定（或默认）目录生成 `train.csv`、`val.csv`、`test.csv`
  - 无表头
  - 已删除 `frames` 列
  - 视频路径列已添加前缀（默认 `videos/`）
