本目录用于战术视角识别的辅助脚本与示例数据。

### 脚本作用
- **find_missing_non_tactical.py**: 基于分类预测结果(`predictions.csv`)与 STD 格式数据集，找出“没有任何『非战术视角』帧”的样例，并导出清单；同时可统计每个样例的总帧数、非战术帧数与占比并写入 CSV。

### 输入与前提
- **STD 数据集根目录**: 目录结构需为 `<STD_ROOT>/videos/<sample>/xxx.{jpg,jpeg,png}`。
- **预测 CSV(predictions.csv)**: 至少包含列 `id`,`label_name`。
  - `id` 形如 `sample/000001.jpg`（脚本用其首段作为样例名）。
  - `label_name` 为预测到的类别名，脚本默认匹配 "非战术视角"，可用 `--label` 改变。

### 输出
- `--out`: 一个 txt 文件，每行一个“未包含『非战术视角』帧”的样例名。
- `--stats-csv`: 每个样例的 `total_frames, non_tactical_frames, non_tactical_ratio` 统计表。

### 常用参数
- `--pred-csv`: 预测结果 CSV 路径（默认指向 temp2/predictions.csv）。
- `--label`: 匹配的“非战术视角”类别名（默认: 非战术视角）。
- `--img-exts`: 统计总帧时计入的图片扩展名，默认 `jpg,jpeg,png`。

### 使用示例
```shell
python /storage/wangxinxing/code/action_data_analysis/resnet_tactical_recognition/find_missing_non_tactical.py <STD_ROOT> \
  --pred-csv /abs/path/to/predictions.csv \
  --label 非战术视角 \
  --out /abs/path/to/samples_without_non_tactical.txt

python /storage/wangxinxing/code/action_data_analysis/resnet_tactical_recognition/find_missing_non_tactical.py /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset \
  --pred-csv /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset/predictions/predictions.csv \
  --label 非战术视角 \
  --out samples_without_non_tactical.txt

python /storage/wangxinxing/code/action_data_analysis/resnet_tactical_recognition/find_missing_non_tactical.py /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted \
  --pred-csv /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted/predictions/predictions.csv \
  --label 非战术视角 \
  --out temp2/samples_without_non_tactical2.txt
```

```shell
python /storage/wangxinxing/code/action_data_analysis/resnet_tactical_recognition/find_missing_non_tactical.py <STD_ROOT> \
  --pred-csv /storage/wangxinxing/code/action_data_analysis/temp2/predictions.csv \
  --label 非战术视角 \
  --out /storage/wangxinxing/code/action_data_analysis/temp2/samples_without_non_tactical.txt \
  --stats-csv /storage/wangxinxing/code/action_data_analysis/temp2/non_tactical_stats.csv

python /storage/wangxinxing/code/action_data_analysis/resnet_tactical_recognition/find_missing_non_tactical.py /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted \
  --pred-csv /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted/predictions/predictions.csv \
  --label 非战术视角 \
  --out /storage/wangxinxing/code/action_data_analysis/temp2/samples_without_non_tactical.txt \
  --stats-csv /storage/wangxinxing/code/action_data_analysis/temp2/non_tactical_stats.csv
```

```shell
python /storage/wangxinxing/code/action_data_analysis/resnet_tactical_recognition/find_missing_non_tactical.py /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted \
  --pred-csv /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted/predictions/predictions.csv \
  --stats-csv /storage/wangxinxing/code/action_data_analysis/temp2/non_tactical_stats.csv
```

### 目录中其他文件
- `labels_dict.json`: 类别 id 到类别名的简单映射，供参考；与上述脚本无直接依赖。
- `example.csv`: 示例数据行，供参考；与上述脚本无直接依赖。