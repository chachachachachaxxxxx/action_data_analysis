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

## 统计命令（仅 std）

- 目录聚合统计：
  ```bash
  python -m action_data_analysis.cli.main analyze-dirs /path/to/<Dataset>_std --out output/<Dataset>_analysis

python -m action_data_analysis.cli.main analyze-dirs /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std_converted --out /storage/wangxinxing/code/action_data_analysis/data/FineSports_json_std_converted/stats

python -m action_data_analysis.cli.main analyze-dirs /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted/stats
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

## 其他

- 导出样例、平铺合并等见 `README.md` 主文档与 `src/action_data_analysis/analyze/*`。
