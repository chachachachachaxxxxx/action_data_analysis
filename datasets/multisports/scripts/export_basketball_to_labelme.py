#!/usr/bin/env python3
import os
import sys
import json
import math
import argparse
import pickle
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


REPO_ROOT = "/storage/wangxinxing/code/action_data_analysis"
MULTISPORTS_ANN_PATH = os.path.join(REPO_ROOT, "datasets", "multisports", "annotations", "multisports_GT.pkl")


def load_annotations(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def is_basketball_video(vid: str) -> bool:
    return isinstance(vid, str) and vid.startswith("basketball/")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    if Image is not None:
        try:
            with Image.open(image_path) as im:
                w, h = im.size
                return int(w), int(h)
        except Exception:
            pass
    if cv2 is not None:
        try:
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                return int(w), int(h)
        except Exception:
            pass
    return None


def write_labelme_json(image_path: str, width: int, height: int, shapes: List[dict], out_json_path: str) -> None:
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


def infer_bbox_columns(arr) -> Tuple[int, int, int, int]:
    # Heuristic: last 4 columns are x1,y1,x2,y2 in pixel coords
    if getattr(arr, "ndim", None) != 2 or arr.shape[1] < 4:
        return (-1, -1, -1, -1)
    return (arr.shape[1] - 4, arr.shape[1] - 3, arr.shape[1] - 2, arr.shape[1] - 1)


def to_rectangle_points(x1: float, y1: float, x2: float, y2: float) -> List[List[float]]:
    return [
        [round(float(x1), 3), round(float(y1), 3)],
        [round(float(x2), 3), round(float(y1), 3)],
        [round(float(x2), 3), round(float(y2), 3)],
        [round(float(x1), 3), round(float(y2), 3)],
    ]


def export_video(
    video_id: str,
    tubes_by_class: Dict[int, List[Any]],
    resolution: Tuple[int, int],
    idx_to_label: Dict[int, str],
    raw_video_frames_root: str,
    out_root: str,
    limit_frames: Optional[int] = None,
) -> Tuple[int, int]:
    """Export frames and labelme json for a single video.

    Returns (num_json, num_missing_frames)
    """
    # expected source dir of frames
    # The user said original videos are under trainval/basketball; typically frames are extracted already as jpgs.
    # We assume frames are already images under raw_video_frames_root/<video_subdir> or <video_basename>.
    # MultiSports video id looks like "basketball/xxxx". We'll try both structures.
    vid_rel = video_id.split("/", 1)[1] if "/" in video_id else video_id

    cand_src_dirs = [
        os.path.join(raw_video_frames_root, vid_rel),
        os.path.join(raw_video_frames_root, os.path.basename(vid_rel)),
    ]
    src_dir = None
    for d in cand_src_dirs:
        if os.path.isdir(d):
            src_dir = d
            break
    if src_dir is None:
        # last resort: scan subdirs to find one that contains jpgs and matches vid_rel prefix
        for root, dirs, files in os.walk(raw_video_frames_root):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                if vid_rel in root or os.path.basename(vid_rel) == os.path.basename(root):
                    src_dir = root
                    break
    if src_dir is None:
        return (0, 0)

    dst_dir = os.path.join(out_root, vid_rel)
    ensure_dir(dst_dir)

    # Index frames in source dir
    frame_files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not frame_files:
        return (0, 0)
    # sort with numeric awareness
    def key_of(fn: str) -> Tuple[int, str]:
        name, _ = os.path.splitext(fn)
        num = 0
        for part in name.split("_"):
            if part.isdigit():
                try:
                    num = int(part)
                    break
                except Exception:
                    pass
        return (num, fn)

    frame_files.sort(key=key_of)

    # Build frame index -> filename
    idx_to_file: Dict[int, str] = {}
    for fn in frame_files:
        name, _ = os.path.splitext(fn)
        num = None
        # try digits at end
        digits = ''.join(ch for ch in name if ch.isdigit())
        if digits:
            try:
                num = int(digits)
            except Exception:
                num = None
        if num is None:
            # fallback to ordering index (1-based)
            num = len(idx_to_file) + 1
        if num not in idx_to_file:
            idx_to_file[num] = fn

    # Prepare shapes by frame index: gather across classes and tubes
    width, height = 0, 0
    if resolution and len(resolution) >= 2:
        # MultiSports stores (H,W) or (W,H); we ensure width>=height mapping
        r0, r1 = resolution[0], resolution[1]
        if r0 >= r1:
            height, width = int(r1), int(r0)
        else:
            height, width = int(r0), int(r1)

    # 直接以 tube 的索引作为 group_id，保证同一视频内每条 tube 的 id 稳定且简单
    shapes_per_frame: Dict[int, List[dict]] = defaultdict(list)
    # 采用全局递增的 group_id：第一个 tube 为 0，第二个为 1，依次递增
    next_group_id = 0
    # tubes_by_class: label_idx -> list of ndarray tubes; each tube: rows [frame, ..., x1,y1,x2,y2]
    for label_idx, tubes in tubes_by_class.items():
        if not isinstance(tubes, list):
            continue
        for tube_idx, tube in enumerate(tubes):
            if getattr(tube, "ndim", None) != 2 or tube.shape[1] < 5:
                continue
            fcol = 0
            x1c, y1c, x2c, y2c = infer_bbox_columns(tube)
            if x1c < 0:
                continue
            gid = int(next_group_id)
            for row in tube:
                try:
                    frame_index = int(row[fcol])
                except Exception:
                    continue
                x1, y1, x2, y2 = float(row[x1c]), float(row[y1c]), float(row[x2c]), float(row[y2c])
                pts = to_rectangle_points(x1, y1, x2, y2)
                # 使用原始标签文本，不做规范化改写
                raw_label = idx_to_label.get(int(label_idx), "")
                action_text = raw_label
                shape = {
                    "label": "player",
                    "description": None,
                    "points": pts,
                    "group_id": gid,
                    "difficult": False,
                    "shape_type": "rectangle",
                    "flags": {},
                    "attributes": {"team": "0", "action": action_text},
                }
                shapes_per_frame[frame_index].append(shape)
            next_group_id += 1

    # Iterate frames with annotations, copy image and write json
    processed_json, missing_frames = 0, 0
    # Determine ordered frame indices
    frame_indices = sorted(shapes_per_frame.keys())
    if limit_frames is not None:
        frame_indices = frame_indices[: max(0, int(limit_frames))]

    for fid in frame_indices:
        # map fid to filename
        fname = idx_to_file.get(fid)
        if fname is None:
            # fallback: try zero-padded formats
            for cand in [f"{fid:06d}", f"img_{fid:05d}", f"{fid}"]:
                for ext in [".jpg", ".jpeg", ".png"]:
                    alt = cand + ext
                    if os.path.exists(os.path.join(src_dir, alt)):
                        fname = alt
                        break
                if fname:
                    break
        if fname is None:
            missing_frames += 1
            continue

        src_img = os.path.join(src_dir, fname)
        dst_img = os.path.join(dst_dir, fname)
        if not os.path.exists(dst_img):
            try:
                shutil.copy2(src_img, dst_img)
            except Exception:
                continue

        # determine size if unknown
        w, h = width, height
        if w <= 0 or h <= 0:
            size = get_image_size(src_img)
            if size:
                w, h = size

        out_json_path = os.path.join(dst_dir, os.path.splitext(fname)[0] + ".json")
        write_labelme_json(dst_img, w, h, shapes_per_frame.get(fid, []), out_json_path)
        processed_json += 1

    # Also mirror all images for this video to dst (optional but handy)
    # This ensures "每个视频一个文件夹，视频导出成图片放在该文件夹下" 完整满足
    copied_all = 0
    for fn in frame_files:
        src_img = os.path.join(src_dir, fn)
        dst_img = os.path.join(dst_dir, fn)
        if not os.path.exists(dst_img):
            try:
                shutil.copy2(src_img, dst_img)
                copied_all += 1
            except Exception:
                pass

    return (processed_json, missing_frames)


def main():
    """
    python /storage/wangxinxing/code/action_data_analysis/datasets/multisports/scripts/export_basketball_to_labelme.py \
  --ann_pkl /storage/wangxinxing/code/action_data_analysis/datasets/multisports/annotations_basketball/multisports_basketball.pkl \
  --rawframes_root /storage/wangxinxing/code/action_data_analysis/data/MultiSports/data/trainval/basketball_frames \
  --out_root /storage/wangxinxing/code/action_data_analysis/data/MultiSports_json
    """
    parser = argparse.ArgumentParser(description="Export MultiSports basketball annotations to LabelMe JSON with per-video folders")
    parser.add_argument("--ann_pkl", type=str, default=MULTISPORTS_ANN_PATH, help="Path to multisports_GT.pkl")
    parser.add_argument("--rawframes_root", type=str, required=True, help="Root directory containing extracted frames for basketball videos")
    parser.add_argument("--out_root", type=str, required=True, help="Output directory to write per-video folders with images and JSONs")
    parser.add_argument("--limit_videos", type=int, default=None, help="Optionally limit number of videos to export")
    parser.add_argument("--limit_frames", type=int, default=None, help="Optionally limit frames per video for which to write JSON")
    parser.add_argument("--filter_video", type=str, default=None, help="Comma-separated video ids to include (e.g., basketball/xxxx)")
    args = parser.parse_args()

    ann = load_annotations(args.ann_pkl)
    gttubes: Dict[str, Dict[int, List[Any]]] = ann.get("gttubes", {})
    labels: List[str] = ann.get("labels", [])
    idx_to_label: Dict[int, str] = {i: l for i, l in enumerate(labels)}
    resolution: Dict[str, Tuple[int, int]] = ann.get("resolution", {})
    videos = [v for v in gttubes.keys() if is_basketball_video(v)]

    if args.filter_video:
        allow = {v.strip() for v in args.filter_video.split(",") if v.strip()}
        videos = [v for v in videos if v in allow]

    videos.sort()
    if args.limit_videos is not None:
        videos = videos[: max(0, int(args.limit_videos))]

    ensure_dir(args.out_root)

    total_json, total_missing = 0, 0
    for i, vid in enumerate(videos):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(videos)}: {vid}")
        tubes = gttubes.get(vid, {})
        res = resolution.get(vid, (0, 0))
        pj, mf = export_video(vid, tubes, res, idx_to_label, args.rawframes_root, args.out_root, args.limit_frames)
        total_json += pj
        total_missing += mf

    print(f"Done. Videos: {len(videos)}, json_written: {total_json}, missing_frames: {total_missing}")


if __name__ == "__main__":
    main()


