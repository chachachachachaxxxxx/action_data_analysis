#!/usr/bin/env python3
import os
import argparse
from typing import Tuple, List, Dict, Optional
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png")


def list_sorted_images(video_dir: str) -> List[str]:
    try:
        files = [f for f in os.listdir(video_dir) if f.lower().endswith(IMG_EXTS)]
    except FileNotFoundError:
        return []
    def key_of(fn: str) -> Tuple[int, str]:
        name, _ = os.path.splitext(fn)
        digits = ''.join([c for c in name if c.isdigit()])
        num = int(digits) if digits else 0
        return (num, fn)
    files.sort(key=key_of)
    return files


def build_index_map(files: List[str]) -> Dict[int, str]:
    idx_map: Dict[int, str] = {}
    for fn in files:
        name, _ = os.path.splitext(fn)
        digits = ''.join([c for c in name if c.isdigit()])
        if not digits:
            continue
        try:
            idx = int(digits)
        except Exception:
            continue
        if idx not in idx_map:
            idx_map[idx] = fn
    return idx_map


def load_gray(path: str) -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray]]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            if im.mode != "L":
                im = im.convert("L")
            w, h = im.size
            arr = np.frombuffer(im.tobytes(), dtype=np.uint8).reshape((h, w))
            return (int(w), int(h)), arr
    except Exception:
        try:
            import cv2  # type: ignore
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None, None
            h, w = img.shape[:2]
            return (int(w), int(h)), img.astype(np.uint8)
        except Exception:
            return None, None


def load_rgb(path: str) -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray]]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            arr = np.frombuffer(im.tobytes(), dtype=np.uint8).reshape((h, w, 3))
            return (int(w), int(h)), arr
    except Exception:
        try:
            import cv2  # type: ignore
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                return None, None
            h, w = img.shape[:2]
            # BGR -> RGB
            img = img[:, :, ::-1]
            return (int(w), int(h)), img.astype(np.uint8)
        except Exception:
            return None, None


def main():
    parser = argparse.ArgumentParser(description="计算两个数据集同视频第一帧的平均像素差异（rgb或gray）")
    parser.add_argument("--multisports_root", type=str, required=True)
    parser.add_argument("--sportshhi_root", type=str, required=True)
    parser.add_argument("--video_id", type=str, default="v_2j7kLB-vEEk_c010")
    parser.add_argument("--space", type=str, default="rgb", choices=["rgb", "gray"], help="使用 rgb（三通道均值）或 gray（单通道）")
    args = parser.parse_args()

    ms_dir = os.path.join(args.multisports_root, args.video_id)
    sh_dir = os.path.join(args.sportshhi_root, args.video_id)
    ms_files = list_sorted_images(ms_dir)
    sh_files = list_sorted_images(sh_dir)
    if not ms_files or not sh_files:
        print("missing frames")
        return

    ms_idx_map = build_index_map(ms_files)
    sh_idx_map = build_index_map(sh_files)
    common = sorted(set(ms_idx_map.keys()) & set(sh_idx_map.keys()))
    if not common:
        print("no common frame indices")
        return

    i0 = common[0]  # 第一帧（共同最小帧号）
    ms_path = os.path.join(ms_dir, ms_idx_map[i0])
    sh_path = os.path.join(sh_dir, sh_idx_map[i0])

    if args.space == "gray":
        (mw, mh), ma = load_gray(ms_path)
        (sw, sh), sb = load_gray(sh_path)
        if ma is None or sb is None or (mw != sw or mh != sh):
            print("size mismatch or read fail")
            return
        diff = np.abs(ma.astype(np.int16) - sb.astype(np.int16))
        mean_val = float(diff.mean())
        print(f"video_id={args.video_id} space=gray ms_file={os.path.basename(ms_path)} sh_file={os.path.basename(sh_path)} mean_abs_diff={mean_val:.6f}")
    else:
        (mw, mh), ma = load_rgb(ms_path)
        (sw, sh), sb = load_rgb(sh_path)
        if ma is None or sb is None or (mw != sw or mh != sh):
            print("size mismatch or read fail")
            return
        diff = np.abs(ma.astype(np.int16) - sb.astype(np.int16)).mean(axis=2)
        mean_val = float(diff.mean())
        print(f"video_id={args.video_id} space=rgb ms_file={os.path.basename(ms_path)} sh_file={os.path.basename(sh_path)} mean_abs_diff={mean_val:.6f}")


if __name__ == "__main__":
    main()


