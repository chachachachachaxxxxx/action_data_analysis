# 读取tactical_batch.py的输出文件作为输入
# 然后解析在标准数据里面找到相应的帧
# 解析过程可以参考extract_zero_pairs.py
# 复制粘贴即可

from tqdm import tqdm
import os
import json
import shutil
from typing import List, Tuple
import sys

def parse_input_line(line: str) -> Tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        print(f"[WARN] 非法行（缺少逗号分隔）: {line}", file=sys.stderr)
        return None
    file_id = parts[0]
    label = parts[1]
    return file_id, label

def extract_video_track(file_id: str) -> Tuple[str, str] | None:
    # file_id expected: {video_id}_{track_id}_{frame_id}
    parts = file_id.rsplit("_", 2)
    if len(parts) != 3:
        print(f"[WARN] file_id 不是 video_track_frame 三段命名，跳过: {file_id}", file=sys.stderr)
        return None
    video_id, track_id, frame_id = parts
    return video_id, track_id, frame_id

def extract_video_track2(file_id: str) -> Tuple[str, str] | None:
    # file_id expected: {video_id}_{track_id}_{frame_id}
    parts = file_id.rsplit("_", 1)
    if len(parts) != 2:
        print(f"[WARN] file_id 不是 video_track_frame 三段命名，跳过: {file_id}", file=sys.stderr)
        return None
    video_id, frame_id = parts
    return video_id, frame_id



def main():
    input_txt = "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/annotations/tactical.txt"
    std_root = "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1"
    out_root = "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame_vis"
    os.makedirs(out_root, exist_ok=True)

    pairs: List[Tuple[str, str, str]] = []

    with open(input_txt, "r", encoding="utf-8") as fin:
        for raw in fin:
            parsed = parse_input_line(raw)
            if not parsed:
                continue
            file_id, label = parsed
            if label.strip() != "0":
                continue
            video_id, track_id, frame_id = extract_video_track(file_id)
            # video_id, frame_id = extract_video_track2(file_id)
            # 此处要添加占位符，frame_id要前置零填充到6位
            frame_id = str(frame_id).zfill(6)
            pairs.append((video_id, track_id, frame_id))
    
    for video_id, track_id, frame_id in tqdm(pairs, desc="复制图片", unit="张"):
        video_path = os.path.join(std_root, "videos", video_id)
        frame_path = os.path.join(video_path, f"{frame_id}.jpg")
        shutil.copy(frame_path, os.path.join(out_root, f"{video_id}_{track_id}_{frame_id}.jpg"))


if __name__ == "__main__":
    main()