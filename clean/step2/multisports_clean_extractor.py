# 第一步先找出标准数据集的所有时空管
# 然后提取每一个时空管的第一帧图片
# 等比例压缩到320x180
# 重命名为{{sample_id}}_{{frame_index}}.jpg
# 保存到指定的文件夹

import os
import sys
import shutil
from typing import Tuple

from PIL import Image
from tqdm import tqdm

def _ensure_sys_path() -> None:
  """将仓库 `src` 加入 sys.path，便于从脚本运行时导入包。"""
  # 本脚本位于 <repo>/temp/，src 在 <repo>/src
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  src_path = os.path.join(repo_root, "src")
  if src_path not in sys.path:
    sys.path.insert(0, src_path)


_ensure_sys_path()

from action_data_analysis.analyze.export_tube_videos import (
    _build_tubes_from_folder,
)

def _resize_letterbox(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """将图片等比例缩放至目标尺寸，并使用黑边信箱填充至精确大小。

    保持纵横比不变，缩放后置于居中位置，背景为黑色。
    """
    target_w, target_h = target_size
    src_w, src_h = image.size

    if src_w == 0 or src_h == 0:
        return Image.new("RGB", (target_w, target_h), color=(0, 0, 0))

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def extract_first_frame(dataset_path: str, output_path: str, target_size: Tuple[int, int] = (320, 180), resize: bool = True) -> None:
    samples_path = os.path.join(dataset_path, "videos")

    samples = sorted(os.listdir(samples_path)) if os.path.isdir(samples_path) else []
    for sample in tqdm(samples, desc="Samples", unit="sample"):
        sample_path = os.path.join(samples_path, sample)
        if not os.path.isdir(sample_path):
            continue

        sample_id = os.path.splitext(sample)[0]
        tubes = _build_tubes_from_folder(sample_path)

        # 对每一个时空管提取第一帧
        for tube_key, tube_value in tqdm(tubes.items(), desc=f"{sample_id} tubes", unit="tube", leave=False):
            # tube_value: List[Tuple[frame_index, frame_path, (x1,y1,x2,y2)]]
            track_id, _ = tube_key
            first_frame_index, first_frame_path, _ = tube_value[0]

            out_name = f"{sample_id}_{track_id}_{int(first_frame_index):06d}.jpg"
            out_path = os.path.join(output_path, "images", out_name)
            
            if not resize:
                # 不缩放：优先直接拷贝 JPEG；若不是 JPEG，则转存为 JPEG
                lower = first_frame_path.lower()
                if lower.endswith((".jpg", ".jpeg")):
                    try:
                        shutil.copy(first_frame_path, out_path)
                        continue
                    except Exception:
                        # 拷贝失败则回退到转存
                        pass
                try:
                    with Image.open(first_frame_path) as img:
                        img = img.convert("RGB")
                        img.save(out_path, quality=95)
                except Exception:
                    continue
                continue

            # 缩放：等比例缩放至目标尺寸后保存
            try:
                with Image.open(first_frame_path) as img:
                    img = img.convert("RGB")
                    out_img = _resize_letterbox(img, target_size)
                    out_img.save(out_path, quality=95)
            except Exception:
                continue


def main():
    pass    


if __name__ == "__main__":
    dataset_path  = "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1"
    output_path = "/storage/wangxinxing/code/action_data_analysis/data/MultiSports_step1_first_frame"
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    extract_first_frame(dataset_path, output_path, resize=False)
    main()