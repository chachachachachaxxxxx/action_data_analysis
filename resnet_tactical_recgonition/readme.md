只有不含战术视角的视频
```shell
python /storage/wangxinxing/code/action_data_analysis/temp2/find_missing_non_tactical.py <STD_ROOT> \
  --pred-csv /abs/path/to/predictions.csv \
  --label 非战术视角 \
  --out /abs/path/to/samples_without_non_tactical.txt

python /storage/wangxinxing/code/action_data_analysis/temp2/find_missing_non_tactical.py /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset \
  --pred-csv /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted_subset/predictions/predictions.csv \
  --label 非战术视角 \
  --out samples_without_non_tactical.txt

python /storage/wangxinxing/code/action_data_analysis/temp2/find_missing_non_tactical.py /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted \
  --pred-csv /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted/predictions/predictions.csv \
  --label 非战术视角 \
  --out temp2/samples_without_non_tactical2.txt
```


```shell
python /storage/wangxinxing/code/action_data_analysis/temp2/find_missing_non_tactical.py <STD_ROOT> \
  --pred-csv /storage/wangxinxing/code/action_data_analysis/temp2/predictions.csv \
  --label 非战术视角 \
  --out /storage/wangxinxing/code/action_data_analysis/temp2/samples_without_non_tactical.txt \
  --stats-csv /storage/wangxinxing/code/action_data_analysis/temp2/non_tactical_stats.csv

 python /storage/wangxinxing/code/action_data_analysis/temp2/find_missing_non_tactical.py /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted \
  --pred-csv /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted/predictions/predictions.csv \
  --label 非战术视角 \
  --out /storage/wangxinxing/code/action_data_analysis/temp2/samples_without_non_tactical.txt \
  --stats-csv /storage/wangxinxing/code/action_data_analysis/temp2/non_tactical_stats.csv
```

```shell
python /storage/wangxinxing/code/action_data_analysis/temp2/find_missing_non_tactical.py /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted \
  --pred-csv /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json_std_converted/predictions/predictions.csv \
  --stats-csv /storage/wangxinxing/code/action_data_analysis/temp2/non_tactical_stats.csv
```