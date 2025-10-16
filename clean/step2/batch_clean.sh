python /storage/wangxinxing/code/action_data_analysis/clean/step2/tactical_batch_multisports.py \
    --input /storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame_vis \
    --output /storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/annotations/tactical2.txt \
    --model /storage/wangxinxing/model/Qwen/Qwen2.5-VL-7B-Instruct \
    --device cuda \
    --max-new-tokens 16 \
    --batch-size 6