STD=/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1
TXT=/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/annotations/tactical_zero.txt        # 每行: video_id,track_id
OUT=/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step2_json

python -m action_data_analysis.cli.main remove-tubes \
  "$STD" \
  "$TXT" \
  --out "$OUT"