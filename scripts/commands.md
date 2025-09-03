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
  --fine-root /storage/wangxinxing/code/action_data_analysis/data/FineSports_json \
  --sportshhi-root /storage/wangxinxing/code/action_data_analysis/data/SportsHHI_json \
  --out-json /storage/wangxinxing/code/action_data_analysis/output/json \
  --out-merged /storage/wangxinxing/code/action_data_analysis/output/merged \
  --per-class 3 --context 6
```
- **输出**:
  - 按数据集分开的样例：`output/json/{FineSports,SportsHHI}/<action>/<sample_id>/*.{jpg,json}`
  - 平铺合并后的样例：`output/merged/{dataset}__{action}__{sample_id}__{basename}.{ext}`

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
- 统计多个目录并输出聚合结果：
```bash
zsh scripts/analyze_dirs.sh --out /storage/wangxinxing/code/action_data_analysis/output/all_agg \
  /storage/wangxinxing/code/action_data_analysis/data/SportsHHI_json/*(/)

zsh scripts/analyze_dirs.sh --out /storage/wangxinxing/code/action_data_analysis/output/all_agg \
  /storage/wangxinxing/code/action_data_analysis/data/FineSports_json/*/*(/)
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