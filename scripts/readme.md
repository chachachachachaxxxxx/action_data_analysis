### 前置条件
- **环境**: 已安装 Python 3.9+，并可运行 `python`；已安装 `opencv-python` 仅在使用可视化脚本时需要。
- **项目路径**: `/storage/wangxinxing/code/action_data_analysis`
- **权限**: 首次使用前为脚本添加执行权限：
```bash
chmod +x scripts/*.sh
```

### 一键导出并合并（推荐）
- **目标**: 从 FineSports 与 SportsHHI 各数据集中，按类别随机采样每类 3 例，并为每例导出中心帧及前后各 6 帧（共最多 13 帧），复制原始 `json` 与 `img`；随后将两者平铺合并到单一目录并重命名。
```bash
zsh scripts/export_merged_samples.sh \
  --multisports-root /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json \
  --fine-root /storage/wangxinxing/code/action_data_analysis/data/FineSports_json \
  --sportshhi-root /storage/wangxinxing/code/action_data_analysis/data/SportsHHI_main_json \
  --out-json /storage/wangxinxing/code/action_data_analysis/output/json2 \
  --out-merged /storage/wangxinxing/code/action_data_analysis/output/merged2 \
  --per-class 3 --context 6
```
- **输出**:
  - 按数据集分开的样例：`output/json/{FineSports,SportsHHI}/<action>/<sample_id>/*.{jpg,json}`
  - 平铺合并后的样例：`output/merged/{dataset}__{action}__{sample_id}__{basename}.{ext}`

### 单数据集一键导出并合并
- MultiSports：
```bash
zsh scripts/export_merged_samples_multisports.sh \
  --multisports-root /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json \
  --out-json /storage/wangxinxing/code/action_data_analysis/output/json2 \
  --out-merged /storage/wangxinxing/code/action_data_analysis/output/merged_MultiSports \
  --per-class 3 --context 6
```
- FineSports：
```bash

```
- SportsHHI：
```bash
zsh scripts/export_merged_samples_sportshhi.sh \
  --sportshhi-root /storage/wangxinxing/code/action_data_analysis/data/SportsHHI_main_json \
  --out-json /storage/wangxinxing/code/action_data_analysis/output/json2 \
  --out-merged /storage/wangxinxing/code/action_data_analysis/output/merged_SportsHHI \
  --per-class 3 --context 6
```

### 分步执行（分别导出两个数据集，再合并）
1) 导出 FineSports：
```bash
zsh scripts/export_samples.sh \
  --out /storage/wangxinxing/code/action_data_analysis/output/json \
  --dataset-name FineSports \
  --per-class 3 --context 6 \
  /storage/wangxinxing/code/action_data_analysis/data/FineSports_json
```
2) 导出 SportsHHI：
```bash
zsh scripts/export_samples.sh \
  --out /storage/wangxinxing/code/action_data_analysis/output/json \
  --dataset-name SportsHHI \
  --per-class 3 --context 6 \
  /storage/wangxinxing/code/action_data_analysis/data/SportsHHI_json
```
3) 合并平铺：
```bash
PYTHONPATH=src python -m action_data_analysis.cli.main merge-flatten \
  /storage/wangxinxing/code/action_data_analysis/output/json/FineSports \
  /storage/wangxinxing/code/action_data_analysis/output/json/SportsHHI \
  --out /storage/wangxinxing/code/action_data_analysis/output/merged
```

### 仅统计与可视化（可选）
- 一键统计（std 数据集）：
```bash
bash scripts/stats.sh <STD_ROOT> [--out <OUT_DIR>] [--spf <SECONDS_PER_FRAME>] [--skip-dirs] [--skip-plot] [--skip-time]
```
- 说明：聚合执行 3 项统计并写入 `<STD_ROOT>/stats`（默认）：
  - 目录聚合统计
  - Tube 长度分箱图（`tube_lengths.png`）
  - 时长分箱（秒）（`time_bins/`）
- 示例：
```bash
zsh scripts/stats.sh /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std_converted \
--spf 0.166666 \
--out /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std_converted/stats

zsh scripts/stats.sh /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std \
--spf 0.166666 \
--out /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std/stats

zsh scripts/stats.sh \
/storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted \
--spf 0.04 \
--out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted/stats

zsh scripts/stats.sh \
/storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std \
--spf 0.04 \
--out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std/stats
```
- 统计多个目录并输出聚合结果：
```bash
zsh scripts/analyze_dirs.sh --out /storage/wangxinxing/code/action_data_analysis/output/SportsHHI_main_analysis \
  /storage/wangxinxing/code/action_data_analysis/data/SportsHHI_main_json/*(/)

zsh scripts/analyze_dirs.sh --out /storage/wangxinxing/code/action_data_analysis/output/FineSports_analysis \
  /storage/wangxinxing/code/action_data_analysis/data/FineSports_json/*/*(/)

zsh scripts/analyze_dirs.sh --out /storage/wangxinxing/code/action_data_analysis/output/MultiSports_analysis \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json/*(/)
```
- 检测框重合统计（IoU）：
```bash
zsh scripts/analyze_overlaps_sportshhi.sh \
  --sportshhi-root /storage/wangxinxing/code/action_data_analysis/data/SportsHHI_main_json \
  --out /storage/wangxinxing/code/action_data_analysis/output/SportsHHI_main_analysis \
  --thresholds "0.99"
```
- 可视化样例（需要 OpenCV；仅绘制，不复制原图）：
```bash
zsh scripts/visualize_samples.sh --out /storage/wangxinxing/code/action_data_analysis/output/vis --per-class 3 --context 12 \
  /storage/wangxinxing/code/action_data_analysis/data/FineSports_json/*/*(/)

zsh scripts/visualize_samples.sh --out /storage/wangxinxing/code/action_data_analysis/output/vis_sportshhi --per-class 3 --context 12 \
  /storage/wangxinxing/code/action_data_analysis/data/SportsHHI_json/*/
```

### 参数说明（核心）
- **--per-class**: 每个动作类别采样的样例数量（默认 3）。
- **--context**: 采样帧两侧各扩展的帧数（默认 6），总计 `2*context+1` 帧；若不足则使用全部可用帧。
- **--out-json**: 分数据集导出的根目录，默认 `output/json`。
- **--out-merged**: 平铺合并后的输出目录，默认 `output/merged`。


