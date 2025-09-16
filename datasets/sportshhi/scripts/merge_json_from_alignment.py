#!/usr/bin/env python3
import os
import csv
import json
import shutil
import argparse
from typing import Any, Dict, Iterable, List, Optional, Tuple

from action_data_analysis.io.json import read_labelme_json


IMG_EXTS = (".jpg", ".jpeg", ".png")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_copy(src: str, dst: str) -> bool:
    try:
        if os.path.isfile(src):
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            return True
    except Exception:
        return False
    return False


def read_alignment_rows(csv_path: str, status_allow: Optional[Iterable[str]] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if status_allow is not None:
                if (r.get("status") or "").strip() not in status_allow:
                    continue
            rows.append(r)
    return rows


def _digits_from_name(name: str) -> Optional[int]:
    digits = ''.join([c for c in name if c.isdigit()])
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def build_index_map_from_dir(video_dir: str) -> Dict[int, str]:
    try:
        files = [f for f in os.listdir(video_dir) if any(f.lower().endswith(ext) for ext in IMG_EXTS)]
    except FileNotFoundError:
        return {}
    idx_map: Dict[int, str] = {}
    for fn in files:
        name, _ = os.path.splitext(fn)
        idx = _digits_from_name(name)
        if idx is None:
            continue
        if idx not in idx_map:
            idx_map[idx] = fn
    return idx_map


def scale_shape_points(shape: Dict[str, Any], sx: float, sy: float) -> Dict[str, Any]:
    out = dict(shape)
    pts = out.get("points") or []
    new_pts: List[List[float]] = []
    for p in pts:
        if isinstance(p, list) and len(p) == 2:
            new_pts.append([float(p[0]) * sx, float(p[1]) * sy])
        elif isinstance(p, list) and len(p) == 4:
            new_pts.append([float(p[0]) * sx, float(p[1]) * sy, float(p[2]) * sx, float(p[3]) * sy])
        else:
            new_pts.append(p)
    out["points"] = new_pts
    return out


def write_labelme_json(image_path: str, width: int, height: int, shapes: List[Dict[str, Any]], out_json_path: str) -> None:
    data = {
        "version": "2.5.4",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_image_size(img_path: str) -> Tuple[int, int]:
    """返回 (width, height)。失败返回 (0,0)。"""
    try:
        from PIL import Image  # type: ignore
        with Image.open(img_path) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        try:
            import cv2  # type: ignore
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return (0, 0)
            h, w = img.shape[:2]
            return int(w), int(h)
        except Exception:
            return (0, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 alignment CSV 合并 MultiSports 与 SportsHHI 的 LabelMe JSON，采用 SportsHHI 图片")
    parser.add_argument("--alignment_csv", type=str,
                        default="/storage/wangxinxing/code/action_data_analysis/datasets/sportshhi/results/alignment_search.csv")
    parser.add_argument("--multisports_root", type=str,
                        default="/storage/wangxinxing/code/action_data_analysis/data/MultiSports_json")
    parser.add_argument("--sportshhi_root", type=str,
                        default="/storage/wangxinxing/code/action_data_analysis/data/SportsHHI_json")
    parser.add_argument("--out_root", type=str,
                        default="/storage/wangxinxing/code/action_data_analysis/data/MultiSports_SportsHHI_merged")
    parser.add_argument("--allow_status", type=str, default="ok",
                        help="逗号分隔，允许的对齐状态；默认: ok。可选含 no_below_threshold")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条记录；<=0 表示全部")
    parser.add_argument("--max_frames", type=int, default=0, help="逐帧向后最多处理多少帧；<=0 表示直到一侧结束")
    args = parser.parse_args()

    allow_status = {s.strip() for s in (args.allow_status or "").split(",") if s.strip()}
    rows = read_alignment_rows(args.alignment_csv, status_allow=allow_status if allow_status else None)
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    ensure_dir(args.out_root)

    merged_count = 0
    copied_images = 0

    for r in rows:
        video_id = (r.get("video_id") or "").strip()
        ms_file = (r.get("ms_file") or "").strip()
        sh_file = (r.get("sh_file") or "").strip()
        if not video_id or not ms_file or not sh_file:
            continue

        sh_dir = os.path.join(args.sportshhi_root, video_id)
        ms_dir = os.path.join(args.multisports_root, video_id)
        if not os.path.isdir(sh_dir) or not os.path.isdir(ms_dir):
            continue

        try:
            sh_start_idx = int(r.get("sh_index") or 0)
        except Exception:
            sh_start_idx = _digits_from_name(os.path.splitext(sh_file)[0]) or 0
        try:
            ms_start_idx = int(r.get("ms_index") or 0)
        except Exception:
            ms_start_idx = _digits_from_name(os.path.splitext(ms_file)[0]) or 0
        if sh_start_idx <= 0 or ms_start_idx <= 0:
            continue

        sh_idx_map = build_index_map_from_dir(sh_dir)
        ms_idx_map = build_index_map_from_dir(ms_dir)
        dst_dir = os.path.join(args.out_root, video_id)
        ensure_dir(dst_dir)

        t = 0
        processed_frames = 0
        while True:
            sh_idx = sh_start_idx + t
            ms_idx = ms_start_idx + t
            if sh_idx not in sh_idx_map or ms_idx not in ms_idx_map:
                break

            sh_fn = sh_idx_map[sh_idx]
            ms_fn = ms_idx_map[ms_idx]
            sh_img = os.path.join(sh_dir, sh_fn)
            ms_img = os.path.join(ms_dir, ms_fn)
            sh_json_path = os.path.splitext(sh_img)[0] + ".json"
            ms_json_path = os.path.splitext(ms_img)[0] + ".json"

            sh_shapes: List[Dict[str, Any]] = []
            ms_shapes: List[Dict[str, Any]] = []
            sh_w, sh_h = 0, 0

            if os.path.isfile(sh_json_path):
                sh_rec = read_labelme_json(sh_json_path)
                sh_shapes = list(sh_rec.get("shapes", []) or [])
                sh_w = int(sh_rec.get("imageWidth") or 0)
                sh_h = int(sh_rec.get("imageHeight") or 0)
            if sh_w <= 0 or sh_h <= 0:
                # 回退到读取图片尺寸
                sh_w, sh_h = read_image_size(sh_img)

            ms_w, ms_h = 0, 0
            if os.path.isfile(ms_json_path):
                ms_rec = read_labelme_json(ms_json_path)
                ms_shapes = list(ms_rec.get("shapes", []) or [])
                ms_w = int(ms_rec.get("imageWidth") or 0)
                ms_h = int(ms_rec.get("imageHeight") or 0)
            if (ms_w <= 0 or ms_h <= 0) and os.path.isfile(ms_img):
                # 回退读取 MS 图片尺寸（若存在）
                mw, mh = read_image_size(ms_img)
                ms_w, ms_h = int(mw), int(mh)

            if sh_w <= 0 or sh_h <= 0:
                # 仍无法得到基准尺寸，则跳过该帧
                t += 1
                processed_frames += 1
                if args.max_frames and processed_frames >= int(args.max_frames):
                    break
                continue

            merged_shapes: List[Dict[str, Any]] = []
            for s in sh_shapes:
                ss = dict(s)
                flags = dict(ss.get("flags") or {})
                flags["source"] = "SH"
                ss["flags"] = flags
                merged_shapes.append(ss)

            if ms_shapes:
                sx = float(sh_w) / float(ms_w or 1)
                sy = float(sh_h) / float(ms_h or 1)
                for s in ms_shapes:
                    ss = scale_shape_points(s, sx, sy)
                    flags = dict(ss.get("flags") or {})
                    flags["source"] = "MS"
                    ss["flags"] = flags
                    merged_shapes.append(ss)

            out_img = os.path.join(dst_dir, sh_fn)
            out_json = os.path.join(dst_dir, os.path.splitext(sh_fn)[0] + ".json")

            if safe_copy(sh_img, out_img):
                copied_images += 1
            write_labelme_json(out_img, sh_w, sh_h, merged_shapes, out_json)
            merged_count += 1

            t += 1
            processed_frames += 1
            if args.max_frames and processed_frames >= int(args.max_frames):
                break

    print(f"Done. Merged frames: {merged_count}, images copied: {copied_images}, out_root: {args.out_root}")


if __name__ == "__main__":
    main()


