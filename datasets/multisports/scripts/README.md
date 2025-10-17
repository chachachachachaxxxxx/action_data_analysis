### scripts 目录说明（MultiSports）

本目录包含 MultiSports 数据的解析、导出、核验与目录规范化脚本。部分脚本使用固定绝对路径，请按需通过参数覆盖，或设置 `PYTHONPATH`：

```bash
export PYTHONPATH="src:${PYTHONPATH}"
```

---

#### 1) MultiSports→LabelMe（篮球子集导出）
- 文件：`export_basketball_to_labelme.py`
- 作用：
  - 读取 `multisports_GT.pkl`（或篮球子集 pkl），对每个篮球视频将逐帧 tube 合成为 LabelMe JSON，并复制图像至 `out_root/<video>` 目录。
- 主要参数：`--ann_pkl`、`--rawframes_root`、`--out_root`、`--limit_videos`、`--limit_frames`、`--filter_video`
- 产物：`out_root/basketball/XXXX/{*.jpg,*.json}`
- 示例：
```bash
python -m datasets.multisports.scripts.export_basketball_to_labelme \
  --ann_pkl /path/to/multisports_basketball.pkl \
  --rawframes_root /path/to/trainval/basketball_frames \
  --out_root /path/to/MultiSports_json
```

---

#### 2) 导出数据一致性检查（tube 连续性与跳帧）
该脚本是由于当时group_id导出错误，会出现两个tube group_id相同的情况检查使用
- 文件：`check_exported_tubes.py`
- 作用：
  - 统计 `MultiSports_json` 下各视频的 `(track_id, action)` 对应的帧序列，按局部 stride 检测分段与跳帧情况，输出汇总与示例。
- 产物：`results/exported_tubes_check.json`、`results/README_exported_tubes_check.md`
- 示例：
```bash
python -m datasets.multisports.scripts.check_exported_tubes
```

---

#### 3) 仅统计篮球 pkl 结构并抽样展示
- 文件：`inspect_basketball_pkl.py`
- 作用：
  - 读取 `annotations_basketball/multisports_basketball.pkl`，检查顶层键、示例视频的 tube 形状与列范围，输出 schema 与示例文件。
- 产物：`results/basketball_pkl_schema.json`、`results/basketball_pkl_example.json`、`results/README_basketball_pkl.md`
- 示例：
```bash
python -m datasets.multisports.scripts.inspect_basketball_pkl
```

---

#### 4) 检查原始 PKL（全量 MultiSports）
- 文件：`inspect_multisports_pkl.py`
- 作用：
  - 读取全量 `multisports_GT.pkl`，打印分割规模、示例 `gttubes` 结构与列统计，辅助理解源数据结构。
- 输出：标准输出日志与若干 json（见脚本打印）。
- 示例：
```bash
python -m datasets.multisports.scripts.inspect_multisports_pkl
```

---

#### 5) tube 数量核验（按跳帧拆段对比）
- 文件：`verify_tube_counts.py`
- 作用：
  - 对篮球子集 pkl 中每条 tube 按 `gap_tolerance` 拆分为连续段，统计原始 tube 总数与拆分后段数差异，并给出按类别增量。
- 产物：`results/verify_tube_counts.json`、`results/README_verify_tube_counts.md`
- 示例：
```bash
python -m datasets.multisports.scripts.verify_tube_counts
```

---

#### 6) 统计视频数量（示例脚本）
- 文件：`count_games.py`
- 作用：
  - 对一个目录下的 mp4 文件，以“去除扩展名并去掉末 3 位占位符”的方式聚合并计数，用于估算不同对局数量。
- 示例：
```bash
python -m datasets.multisports.scripts.count_games
```

---

#### 7) MultiSports_json 转 std 布局
其实就是直接copypaste创建对应的文件夹即可，该脚本并无实际用处

- 文件：`multisports_to_std.py`
- 作用：
  - 自 `MultiSports_json` 中选择最深层包含图像的目录作为样本，转换为标准 `std` 布局：`<out>/videos/<sample_dir>` 与 `<out>/stats/`。
- 主要参数：`src_dir`、`out_dir?`、`--move`
- 示例：
```bash
python -m datasets.multisports.scripts.multisports_to_std \
  /path/to/MultiSports_json \
  /path/to/MultiSports_json_std
```

---

#### 8) 推荐流程与脚本关系
- 基本流程：
  1. 若需篮球子集，先生成 `annotations_basketball/multisports_basketball.pkl`（参见 datasets 根目录相关脚本与 README）；
  2. 使用 `export_basketball_to_labelme.py` 将篮球视频导出为 LabelMe JSON；
  3. 用 `check_exported_tubes.py` 进行导出质量检查；
  4. 需要统一目录结构时，执行 `multisports_to_std.py` 转换为 `std`；
  5. 如需核验分段统计，对篮球子集运行 `verify_tube_counts.py`；
  6. `inspect_*_pkl.py` 辅助理解 PKL 结构与样例。

---

### 备注
- 部分脚本使用绝对路径常量，请按需通过参数覆盖或修改路径。
- 依赖：`numpy`、`Pillow` 或 `opencv-python`（读取图像尺寸）、以及项目内的 `action_data_analysis.io.json` 迭代工具。


