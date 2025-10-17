### scripts 目录说明（FineSports）

本目录包含 FineSports 数据的解析、导出与目录规范化脚本。部分脚本使用固定路径（如 `annotations/FineSports-GT.pkl`）或绝对路径，请按需通过命令行参数覆盖，或在运行前设置合适的 `PYTHONPATH` 以便模块导入：

```bash
export PYTHONPATH="src:${PYTHONPATH}"
```

---

#### 1) 解析 FineSports 标注并生成统计
- 文件：`parse_annotations.py`
- 作用：
  - 读取 `annotations/FineSports-GT.pkl`，推断 `gttubes` 结构，统计类别、分割、主框尺寸分布及异常值，生成 JSON 与 Markdown 报告。
- 产物（默认写入 `datasets/finesports/scripts/results/`）：
  - `annotations_summary.json`、`README_annotations.md`
- 示例：
```bash
python -m datasets.finesports.scripts.parse_annotations
```

---

#### 2) 将 FineSports tubes 导出为 LabelMe JSON（并复制帧）
- 文件：`convert_finesports_to_labelme.py`
- 作用：
  - 读取 `annotations/FineSports-GT.pkl`，遍历 `rawframes_src`（形如 `action/clip` 的目录结构），按 tube 的逐帧 bbox 为每帧生成 LabelMe JSON，并将对应图片复制到 `out_root`。
- 主要参数：`--rawframes_src`、`--out_root`、`--limit_actions`、`--limit_clips`、`--limit_frames`
- 产物：`out_root/<action>/<clip>/{*.jpg,*.json}`
- 备注：
  - 当 tube 坐标为归一化值（0..1）且脚本从 `resolution` 推断出图像尺寸时，会自动换算为像素坐标。
- 示例：
```bash
python -m datasets.finesports.scripts.convert_finesports_to_labelme \
  --rawframes_src /path/to/FineSports_NewFrames \
  --out_root      /path/to/FineSports_json \
  --limit_actions 3 --limit_clips 10 --limit_frames 60
```

---

#### 3) 将 FineSports_json 目录转为 std 布局
- 文件：`finesports_to_std.py`
- 作用：
  - 将以 `action/clip` 组织的样本目录转换为标准 `std` 布局：`<out>/videos/<sample_dir>` 与 `<out>/stats/`，便于后续统一处理与类别映射。
- 主要参数：`src_dir`、`out_dir?`、`--move`
- 产物：`<out_dir>/videos/` 与 `<out_dir>/stats/`
- 示例：
```bash
python -m datasets.finesports.scripts.finesports_to_std \
  /path/to/FineSports_json \
  /path/to/FineSports_json_std
```

---

#### 4) 推荐流程与脚本关系
- 基本流程：
  1. 运行 `parse_annotations.py` 获取数据规模与质量概览；
  2. 使用 `convert_finesports_to_labelme.py` 将标注导出为每帧 LabelMe JSON（需要已抽帧的 `rawframes_src`）；
  3. 使用 `finesports_to_std.py` 将导出结果转为 `std` 布局；
  4. 随后可在项目根 CLI 上进行类别映射与清理：
     - `python -m action_data_analysis.cli.main convert-std <std_root> <mapping.csv>`。

---

### 备注
- 绝对路径与默认值以脚本实现为准，必要时通过命令行参数覆盖。
- 依赖：`numpy`、`pandas`（若在统计中使用）、`Pillow` 或 `opencv-python`（用于读取图像尺寸）、`shutil` 等标准库。


