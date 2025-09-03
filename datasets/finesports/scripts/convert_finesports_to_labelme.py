import os
import json
import shutil
from typing import Dict, List, Tuple

import argparse
import pickle


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PKL_PATH = os.path.join(BASE_DIR, "annotations", "FineSports-GT.pkl")


def to_labelme_json(image_path: str, width: int, height: int, shapes: List[dict]) -> dict:
    return {
        "version": "2.5.4",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert FineSports tubes to LabelMe JSON and copy frames")
    parser.add_argument("--rawframes_src", type=str, required=True, help="Source NewFrames root (per-action/per-clip folders)")
    parser.add_argument("--out_root", type=str, required=True, help="Output root for copied images and JSON")
    parser.add_argument("--limit_actions", type=int, default=None, help="Limit number of action folders")
    parser.add_argument("--limit_clips", type=int, default=None, help="Limit clips per action")
    parser.add_argument("--limit_frames", type=int, default=None, help="Limit frames per clip")
    args = parser.parse_args()

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    labels: List[str] = data.get("labels", [])
    gttubes: Dict[str, Dict[int, list]] = data.get("gttubes", {})
    resolution: Dict[str, Tuple[int, int]] = data.get("resolution", {})

    label_to_id: Dict[str, int] = {name: i for i, name in enumerate(labels)}

    os.makedirs(args.out_root, exist_ok=True)

    # Iterate action folders
    action_names = sorted([d for d in os.listdir(args.rawframes_src) if os.path.isdir(os.path.join(args.rawframes_src, d))])
    if args.limit_actions is not None:
        action_names = action_names[: args.limit_actions]

    processed_json = 0
    copied_images = 0
    missing_tube = 0

    for ai, action in enumerate(action_names):
        action_dir = os.path.join(args.rawframes_src, action)
        if action not in label_to_id:
            # Skip actions not in labels
            continue
        cls_id = label_to_id[action]
        clips = sorted([d for d in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, d))])
        if args.limit_clips is not None:
            clips = clips[: args.limit_clips]

        for ci, clip in enumerate(clips):
            clip_id = f"{action}/{clip}"
            # Expect frames named 00001.jpg, 00002.jpg, ...
            src_clip_dir = os.path.join(action_dir, clip)
            frame_files = [f for f in os.listdir(src_clip_dir) if f.lower().endswith(".jpg")]
            frame_files.sort()
            if args.limit_frames is not None:
                frame_files = frame_files[: args.limit_frames]

            # Find tubes for this clip and class
            per_vid = gttubes.get(clip_id)
            if not isinstance(per_vid, dict) or cls_id not in per_vid:
                missing_tube += 1
                continue
            tubes = per_vid[cls_id]
            if not tubes:
                missing_tube += 1
                continue
            tube = tubes[0]
            # Columns: assume [frame_idx, x1, y1, x2, y2]
            # frame index appears 1-based from probing
            # get resolution to recover pixel boxes if needed
            H, W = None, None
            if clip_id in resolution and isinstance(resolution[clip_id], (list, tuple)) and len(resolution[clip_id]) >= 2:
                r0, r1 = resolution[clip_id][0], resolution[clip_id][1]
                if r0 >= r1:
                    H, W = int(r1), int(r0)
                else:
                    H, W = int(r0), int(r1)

            # Build mapping frame_idx -> bbox
            frame_to_bbox: Dict[int, Tuple[float, float, float, float]] = {}
            for row in tube:
                f = int(row[0])  # 1-based
                x1, y1, x2, y2 = float(row[-4]), float(row[-3]), float(row[-2]), float(row[-1])
                frame_to_bbox[f] = (x1, y1, x2, y2)

            # Prepare output clip dir
            dst_clip_dir = os.path.join(args.out_root, action, clip)
            os.makedirs(dst_clip_dir, exist_ok=True)

            # Copy frames and write per-frame JSON
            for fname in frame_files:
                src_img = os.path.join(src_clip_dir, fname)
                dst_img = os.path.join(dst_clip_dir, fname)
                if not os.path.exists(dst_img):
                    try:
                        shutil.copy2(src_img, dst_img)
                        copied_images += 1
                    except Exception:
                        continue

                try:
                    fid = int(os.path.splitext(fname)[0])  # 00001 -> 1
                except Exception:
                    continue
                # Images are 1-based; JSON bbox expected in pixel units for LabelMe
                if fid not in frame_to_bbox:
                    continue
                x1, y1, x2, y2 = frame_to_bbox[fid]

                # If boxes are normalized (0..1), convert using resolution; else assume pixel already
                if (x2 <= 2.0 and y2 <= 2.0) and H and W:
                    px1, py1 = x1 * W, y1 * H
                    px2, py2 = x2 * W, y2 * H
                else:
                    px1, py1, px2, py2 = x1, y1, x2, y2

                rect = [
                    [round(px1, 3), round(py1, 3)],
                    [round(px2, 3), round(py1, 3)],
                    [round(px2, 3), round(py2, 3)],
                    [round(px1, 3), round(py2, 3)],
                ]
                shapes = [{
                    "label": "player",
                    "description": None,
                    "points": rect,
                    "group_id": 0,
                    "difficult": False,
                    "direction": 0,
                    "shape_type": "rectangle",
                    "flags": {"visibility": 1.0, "class_id": 1},
                    "attributes": {"team": "0", "action": action},
                }]
                out_json = to_labelme_json(dst_img, int(W or 0), int(H or 0), shapes)
                json_name = os.path.splitext(fname)[0] + ".json"
                with open(os.path.join(dst_clip_dir, json_name), "w", encoding="utf-8") as f:
                    json.dump(out_json, f, ensure_ascii=False, indent=2)
                processed_json += 1

    print(json.dumps({
        "out_root": args.out_root,
        "processed_json": processed_json,
        "copied_images": copied_images,
        "missing_tube": missing_tube,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

