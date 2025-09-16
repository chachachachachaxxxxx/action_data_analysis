#!/usr/bin/env python3
import os
import csv
import argparse
from typing import Tuple, List, Dict, Optional
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png")


def list_videos(root: str) -> List[str]:
    try:
        return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    except FileNotFoundError:
        return []


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


def load_rgb(path: str):
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            return im.size, im.tobytes()
    except Exception:
        try:
            import cv2  # type: ignore
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                return None, None
            h, w = img.shape[:2]
            img = img[:, :, ::-1]
            return (w, h), img.tobytes()
        except Exception:
            return None, None


def mean_abs_diff(a_bytes: bytes, b_bytes: bytes, size: Tuple[int, int]) -> Optional[float]:
    if a_bytes is None or b_bytes is None:
        return None
    w, h = int(size[0]), int(size[1])
    try:
        a = np.frombuffer(a_bytes, dtype=np.uint8).reshape((h, w, 3))
        b = np.frombuffer(b_bytes, dtype=np.uint8).reshape((h, w, 3))
    except Exception:
        return None
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    return float(diff.mean())


def align_one_video(video_id: str, ms_root: str, sh_root: str, max_k: int | None, threshold: float) -> Dict[str, object]:
    ms_dir = os.path.join(ms_root, video_id)
    sh_dir = os.path.join(sh_root, video_id)
    ms_files = list_sorted_images(ms_dir)
    sh_files = list_sorted_images(sh_dir)
    rec: Dict[str, object] = {
        "video_id": video_id,
        "status": "",
        "sh_index": None,
        "sh_file": "",
        "ms_index": None,
        "ms_file": "",
        "k": None,
        "mean": None,
    }
    if not ms_files or not sh_files:
        rec["status"] = "missing_frames"
        return rec

    ms_idx_map = build_index_map(ms_files)
    sh_idx_map = build_index_map(sh_files)
    common = sorted(set(ms_idx_map.keys()) & set(sh_idx_map.keys()))
    if not common:
        rec["status"] = "no_common_index"
        return rec

    i0 = common[0]  # SportsHHI 固定第一帧（共同最小帧号）
    rec["sh_index"] = int(i0)
    rec["sh_file"] = sh_idx_map[i0]

    sh_path = os.path.join(sh_dir, sh_idx_map[i0])
    (sw, shh), sb = load_rgb(sh_path)
    if not (sb and sw and shh):
        rec["status"] = "sh_read_fail"
        return rec

    # 搜索 MS 的 i0+k；max_k(None或<=0) 表示尝试全部后续帧；
    # 提前停止：遇到第一个 mean < threshold 即返回。
    if max_k is not None and int(max_k) > 0:
        cand_ms = [i0 + k for k in range(0, int(max_k) + 1) if (i0 + k) in ms_idx_map]
    else:
        cand_ms = [idx for idx in sorted(ms_idx_map.keys()) if idx >= i0]

    best = (None, None, None)  # (ms_idx, k, mean) 仅用于未命中阈值时的信息记录
    for ms_idx in cand_ms:
        ms_path = os.path.join(ms_dir, ms_idx_map[ms_idx])
        (mw, mh), mb = load_rgb(ms_path)
        if not (mb and mw == sw and mh == shh):
            continue
        mean = mean_abs_diff(mb, sb, (mw, mh))
        if mean is None:
            continue
        k = int(ms_idx - i0)
        # 提前停止条件
        if float(mean) < float(threshold):
            rec["status"] = "ok"
            rec["ms_index"] = int(ms_idx)
            rec["ms_file"] = ms_idx_map[ms_idx]
            rec["k"] = int(k)
            rec["mean"] = float(mean)
            return rec
        # 记录当前最佳（用于未命中阈值时的参考）
        if best[2] is None or float(mean) < float(best[2]):
            best = (int(ms_idx), int(k), float(mean))

    # 未找到低于阈值的候选
    if best[0] is None:
        rec["status"] = "no_match"
    else:
        rec["status"] = "no_below_threshold"
        rec["ms_index"] = best[0]
        rec["ms_file"] = ms_idx_map[best[0]]
        rec["k"] = best[1]
        rec["mean"] = best[2]
    return rec


def main():
    parser = argparse.ArgumentParser(description="以 SportsHHI 第一帧为基准，为每个视频在 MultiSports 中搜索起始对齐帧并导出CSV")
    parser.add_argument("--multisports_root", type=str, required=True)
    parser.add_argument("--sportshhi_root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="/storage/wangxinxing/code/action_data_analysis/output/start_refine/alignment_search.csv")
    parser.add_argument("--max-k", type=int, default=20, help="最多前进 K 帧搜索；<=0 表示尝试所有后续帧")
    parser.add_argument("--threshold", type=float, default=3.0, help="提前停止阈值（均值绝对差 < threshold 即命中）")
    args = parser.parse_args()

    ms_videos = set(list_videos(args.multisports_root))
    sh_videos = set(list_videos(args.sportshhi_root))
    videos = sorted(ms_videos & sh_videos)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "sh_index", "sh_file", "ms_index", "ms_file", "k", "mean", "status"])
        for i, vid in enumerate(videos):
            if i % 10 == 0:
                print(f"Align {i+1}/{len(videos)}: {vid}")
            rec = align_one_video(vid, args.multisports_root, args.sportshhi_root, args.max_k, args.threshold)
            writer.writerow([
                rec.get("video_id", ""),
                rec.get("sh_index", ""),
                rec.get("sh_file", ""),
                rec.get("ms_index", ""),
                rec.get("ms_file", ""),
                rec.get("k", ""),
                rec.get("mean", ""),
                rec.get("status", ""),
            ])

    print(f"Wrote alignment CSV: {args.out_csv}")


if __name__ == "__main__":
    main()


