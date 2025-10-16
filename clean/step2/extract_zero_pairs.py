import argparse
import os
import sys
from typing import List, Tuple, Set


def parse_input_line(line: str) -> Tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Expect format: file_id,label
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


def main():
    # parser = argparse.ArgumentParser(description="从 file_id,label 文本中提取 label=0 的 video_id,track_id")
    # parser.add_argument("--input", required=True, help="tactical_batch.py 输出的txt路径，格式: file_id,label")
    # parser.add_argument("--output", required=True, help="输出txt文件路径，格式: video_id,track_id")
    # args = parser.parse_args()

    # input_txt: str = args.input
    # output_txt: str = args.output

    input_txt = "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/annotations/tactical_label0_my.txt"
    output_txt = "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame/annotations/tactical_zero.txt"

    if not os.path.isfile(input_txt):
        print(f"输入文件不存在: {input_txt}", file=sys.stderr)
        sys.exit(1)

    pairs: List[Tuple[str, str, str]] = []

    with open(input_txt, "r", encoding="utf-8") as fin:
        for raw in fin:
            parsed = parse_input_line(raw)
            if not parsed:
                continue
            file_id, label = parsed
            if label.strip() != "0":
                continue
            vt = extract_video_track(file_id)
            if not vt:
                continue
            pairs.append(vt)

    if not pairs:
        print("未发现 label=0 的条目，输出将为空。", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(output_txt)), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as fout:
        for video_id, track_id, frame_id in pairs:
            # fout.write(f"{video_id},{track_id},{frame_id}\n")
            fout.write(f"{video_id},{track_id}\n")

    print(f"完成。写出 {len(pairs)} 条记录到: {output_txt}")

if __name__ == "__main__":
    main()


