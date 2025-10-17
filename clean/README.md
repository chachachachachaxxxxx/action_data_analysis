### clean 模块使用说明（MultiSports 数据清洗）

本目录提供基于 MultiSports 标注/时空管的“战术视角”清洗流水线：
- 从标准数据构建 step1 数据集；
- 抽取每个 tube 的首帧生成待筛图像；
- 使用 Qwen2.5-VL 模型对“战术视角”进行二分类；
- 解析模型输出，得到需剔除的 `video_id,track_id`；
- 在 JSON 时空管层面删除对应 tubes，生成 step2 清洗结果；
- 附带变更统计 `step3_info.json`。

---

### 目录结构

- `step1.sh`: 运行标准数据过滤，生成 `MultiSports_step1`。
- `step2/`
  - `multisports_clean_extractor.py`: 从 `MultiSports_step1` 抽取每个 tube 的首帧，生成图像用于模型判别。
  - `tactical_multisports.py`: 单张推理版战术视角分类（Qwen2.5-VL）。
  - `tactical_batch_multisports.py`: 批量推理版战术视角分类（推荐）。
  - `batch_clean.sh`: 批量分类脚本示例，输出 `tactical2.txt`（格式：`file_id,label`）。
  - `extract_zero_pairs.py`: 解析 `label=0`（非战术视角），输出 `tactical_zero.txt`（格式：`video_id,track_id`）。
  - `remove_tubes.sh`: 按 `video_id,track_id` 删除 tubes，输出 `MultiSports_step2_json`。
  - `readme.md`: 精简版分步说明。
  - `test/`: 与战术视角分类相关的测试与样例。
- `utils/`
  - `extract_zero_labels.py`: 从 `file_id,label` 文本中过滤出 `label=0` 行（便捷工具）。
  - `vis.py`: 将 `label=0` 的样本首帧复制到可视化目录（便于抽检）。
- `step3_info.json`: 前后id不一致但为同一个人的统计信息。

---

### 环境依赖

- Python 3.10+
- PyTorch（建议启用 GPU）
- Transformers（Qwen2.5-VL 模型）
- 其他：`tqdm`, `Pillow`（PIL），以及可从代码导入的 `qwen_vl_utils`

说明：`tactical_*` 脚本默认使用本地模型路径 `Qwen/Qwen2.5-VL-7B-Instruct`，并启用了 `flash_attention_2` 与合适的 `device_map`。若环境不同，请按需修改脚本内默认参数或在命令行覆盖。

---

### 数据路径约定（默认示例）

- 标准数据（已转换）：`/storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted`
- Step1 输出：`/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1`
- 首帧输出：`/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame`
- Step2 输出（JSON）：`/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step2_json`

如需自定义，请在对应脚本或 shell 中调整路径参数。

---

### 快速开始（建议流程）

1) 生成 Step1 标准数据子集

```bash
zsh /storage/wangxinxing/code/action_data_analysis/clean/step1.sh
```

脚本等价于：

```bash
python -m action_data_analysis.cli.main filter-std \
  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted \
  --out /storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1 \
  --exclude-prefix v_2BhBRkkAqbQ_c v_-9kabh1K8UA_c
```
过滤了v_2BhBRkkAqbQ_c v_-9kabh1K8UA_c这两个前缀开头的视频样例

2) 抽取每个 tube 的首帧

```bash
python /storage/wangxinxing/code/action_data_analysis/clean/step2/multisports_clean_extractor.py
```

默认将首帧（可选择等比例缩放或原图复制）写入：

- `/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/images`

生成的文件名格式：`{video_id}_{track_id}_{frame_index:06d}.jpg`

3) 使用 Qwen2.5-VL 进行“战术视角”二分类

推荐批量脚本（可根据环境修改 `--model`、`--device`、`--batch-size`）：

```bash
zsh /storage/wangxinxing/code/action_data_analysis/clean/step2/batch_clean.sh
```

其核心命令为：

```bash
python /storage/wangxinxing/code/action_data_analysis/clean/step2/tactical_batch_multisports.py \
  --input  /storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame_vis \
  --output /storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/annotations/tactical2.txt \
  --model  /storage/wangxinxing/model/Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda \
  --max-new-tokens 16 \
  --batch-size 6
```

输出文本格式：`file_id,label`，其中 `file_id` 与首帧图片名（去扩展名）一致。

4) 解析 `label=0` 并生成需剔除的 `video_id,track_id`

方法 A：直接运行解析脚本（建议先在脚本内将 `input_txt` 指向上一步的 `tactical2.txt`）

```bash
python /storage/wangxinxing/code/action_data_analysis/clean/step2/extract_zero_pairs.py
```

脚本会输出：

- `/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/annotations/tactical_zero.txt`
- 文本格式：`video_id,track_id`

方法 B：先用便捷工具提取 `label=0` 行，再转成 `video_id,track_id`（需自行改造或参考脚本注释）

```bash
python /storage/wangxinxing/code/action_data_analysis/clean/utils/extract_zero_labels.py
```

5) 删除对应 tubes，生成 Step2 JSON

```bash
zsh /storage/wangxinxing/code/action_data_analysis/clean/step2/remove_tubes.sh
```

等价于：

```bash
STD=/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1
TXT=/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/annotations/tactical_zero.txt
OUT=/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step2_json

python -m action_data_analysis.cli.main remove-tubes \
  "$STD" \
  "$TXT" \
  --out "$OUT"
```

6) 查看变更统计（可选）

`/storage/wangxinxing/code/action_data_analysis/clean/step3_info.json` 给出 step2 结果的汇总与若干示例字段：

- `root`: step2 输出根目录
- `samples`: 样本数量
- `changes`: 发生动作/目标变化的跨帧对数量
- `changes_detail`: 若干样例条目（含帧号、动作标签、IoU、bbox 等）

---

### 可视化与抽检（可选）

若需将 `label=0` 的样本首帧集中到一个目录便于人工抽检，可参考：

```bash
python /storage/wangxinxing/code/action_data_analysis/clean/utils/vis.py
```

脚本会读取 `annotations/tactical.txt`（或按需修改），把对应首帧从 `MultiSports_step1` 复制到 `MultiSports_step1_first_frame_vis`。

---

### 常见问题（FAQ）

- GPU/显存不足：
  - 降低 `--batch-size` 或 `--max-new-tokens`；
  - 将 `attn_implementation` 调整或关闭 `flash_attention_2`；
  - 使用 `tactical_multisports.py` 单张模式做回退。
- 模型路径错误：确认 `--model` 指向本地可用的 Qwen2.5-VL 模型目录。
- `qwen_vl_utils` 导入失败：确保该模块在 Python 路径中（可放至项目根或与脚本同级）。
- 文本路径不一致：`extract_zero_pairs.py` 内部默认路径为示例，运行前请根据实际输出修改。

---

### 小抄（Cheatsheet）

按下列顺序执行最稳妥：

```bash
# 1) 生成 Step1
zsh clean/step1.sh

# 2) 抽首帧
python clean/step2/multisports_clean_extractor.py

# 3) 批量战术视角分类
zsh clean/step2/batch_clean.sh

# 4) 提取 label=0 并转 video_id,track_id
python clean/step2/extract_zero_pairs.py

# 5) tube 级删除并生成 Step2 JSON
zsh clean/step2/remove_tubes.sh
```

如需自定义路径或模型，请直接修改对应脚本参数或 shell 变量。


