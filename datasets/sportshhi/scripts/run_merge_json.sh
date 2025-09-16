PYTHONPATH="src" python -m datasets.sportshhi.scripts.merge_json_from_alignment \
  --alignment_csv /storage/wangxinxing/code/action_data_analysis/datasets/sportshhi/results/alignment_search.csv \
  --multisports_root /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json \
  --sportshhi_root /storage/wangxinxing/code/action_data_analysis/data/SportsHHI_main_json \
  --out_root /storage/wangxinxing/code/action_data_analysis/data/MultiSports_SportsHHI_merged \
  --allow_status ok,no_below_threshold