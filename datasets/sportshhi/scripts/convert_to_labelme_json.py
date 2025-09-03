import os
import json
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import argparse
import pandas as pd

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_ANN_DIR = os.path.join(BASE_DIR, "basketball_annotations")


CSV_COLUMNS = [
    "video_id",
    "frame_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "x1_2",
    "y1_2",
    "x2_2",
    "y2_2",
    "action_id",
    "person_id",
    "instance_id",
]


def parse_action_list(pbtxt_path: str) -> Dict[int, str]:
    id_to_name: Dict[int, str] = {}
    if not os.path.exists(pbtxt_path):
        return id_to_name
    current = {}
    with open(pbtxt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("name:"):
                parts = line.split('"')
                current["name"] = parts[1] if len(parts) > 1 else line.split(":", 1)[1].strip()
            elif line.startswith("label_id:"):
                try:
                    current["id"] = int(line.split(":", 1)[1].strip())
                except Exception:
                    continue
            elif line.startswith("}"):
                if "id" in current and "name" in current:
                    id_to_name[current["id"]] = current["name"]
                current = {}
    if current and "id" in current and "name" in current and current["id"] not in id_to_name:
        id_to_name[current["id"]] = current["name"]
    return id_to_name


def simplify_action_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    # 保持原始动作名称，不做修改
    return name


def load_basketball_csvs(ann_dir: str) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for fname in ["sports_train_v1.csv", "sports_val_v1.csv"]:
        path = os.path.join(ann_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, header=None, names=CSV_COLUMNS, dtype={"frame_id": str})
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=CSV_COLUMNS)


def find_image_filename(video_dir: str, frame_id: str) -> str:
    """Find an image filename for the given frame id under a video directory.

    Supports .jpg/.jpeg and case-insensitive extensions. Tries common naming
    patterns first, then falls back to directory scan with prefix matching.
    """
    # Normalize frame id to int for tolerance of leading zeros in CSV
    try:
        fid_int = int(str(frame_id))
    except Exception:
        fid_int = None

    exts = [".jpg", ".jpeg"]
    base_candidates = []
    if fid_int is not None:
        base_candidates.extend([
            f"img_{fid_int:05d}",
            f"{fid_int:06d}",
            f"{fid_int}",
            f"img_{fid_int}",
        ])
    # Also consider literal frame_id as provided
    base_candidates.extend([
        f"img_{frame_id}",
        f"{frame_id}",
    ])

    # Try exact constructions first
    for base in base_candidates:
        for ext in exts:
            cand = base + ext
            full = os.path.join(video_dir, cand)
            if os.path.exists(full):
                return cand
            # Also try uppercase extension
            cand_up = base + ext.upper()
            full_up = os.path.join(video_dir, cand_up)
            if os.path.exists(full_up):
                return cand_up

    # Fallback: scan directory, match by zero-padded 6-digit prefix or suffix
    prefix6 = None
    if fid_int is not None:
        prefix6 = str(fid_int).zfill(6)
    for fname in os.listdir(video_dir):
        lower = fname.lower()
        if not (lower.endswith('.jpg') or lower.endswith('.jpeg')):
            continue
        if prefix6 and (fname.startswith(prefix6) or fname.endswith(prefix6 + os.path.splitext(fname)[1])):
            return fname
        # Also accept any candidate base as substring match
        for base in base_candidates:
            if base in fname:
                return fname
    return ""


def get_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    """Return (width, height) of the image, trying PIL then OpenCV.

    Returns None if cannot determine size.
    """
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
    parser = argparse.ArgumentParser(description="Convert basketball annotations to LabelMe JSON and copy frames")
    parser.add_argument("--rawframes_src", type=str, required=True, help="Source rawframes root")
    parser.add_argument("--ann_dir", type=str, default=DEFAULT_ANN_DIR, help="Basketball annotations directory")
    parser.add_argument("--out_root", type=str, required=True, help="Output root for copied images and JSON")
    parser.add_argument("--limit_videos", type=int, default=None, help="Optionally limit number of videos for conversion")
    parser.add_argument("--limit_frames", type=int, default=None, help="Optionally limit number of frames per video")
    parser.add_argument("--balanced_per_action", type=int, default=None, help="If set, sample up to N frames for each action label")
    parser.add_argument("--include_secondary", action="store_true", help="Include secondary bbox (x1_2..y2_2) as additional shapes; team=1")
    parser.add_argument("--filter_video", type=str, default=None, help="Comma-separated video_ids to include only")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # Load annotations and action names
    df = load_basketball_csvs(args.ann_dir)
    print(f"Loaded rows: {len(df)} from {args.ann_dir}")
    id_to_name = parse_action_list(os.path.join(args.ann_dir, "sports_action_list.pbtxt"))

    # Group by (video_id, frame_id)
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        video_id = row["video_id"]
        frame_id = str(row["frame_id"]).zfill(4)  # keep as string
        action_id = int(row["action_id"]) if pd.notna(row["action_id"]) else None
        person_id = int(row["person_id"]) if pd.notna(row["person_id"]) else None

        action_name = simplify_action_name(id_to_name.get(action_id, ""))

        # Primary bbox
        try:
            x1 = float(row["x1"]); y1 = float(row["y1"]); x2 = float(row["x2"]); y2 = float(row["y2"])
        except Exception:
            continue

        shape = {
            "label": "player",
            "description": None,
            "points": [[x1, y1, x2, y2]],  # placeholder, will scale & expand to 4 corners later
            "group_id": person_id,
            "difficult": False,
            "direction": 0,
            "shape_type": "rectangle",
            "flags": {
                "visibility": 1.0,
                "class_id": 1,
            },
            "attributes": {
                "team": "0",
                "action": action_name,
            },
        }
        grouped[(video_id, frame_id)].append(shape)

        # Secondary bbox if present and requested
        if args.include_secondary and pd.notna(row.get("x1_2")) and pd.notna(row.get("y1_2")) and pd.notna(row.get("x2_2")) and pd.notna(row.get("y2_2")):
            try:
                x1b = float(row["x1_2"]); y1b = float(row["y1_2"]); x2b = float(row["x2_2"]); y2b = float(row["y2_2"])
                shape_b = json.loads(json.dumps(shape))  # shallow copy via serialize
                shape_b["points"] = [[x1b, y1b, x2b, y2b]]
                # Set team for secondary bbox to "1"
                attrs_b = shape_b.get("attributes") or {}
                attrs_b["team"] = "1"
                shape_b["attributes"] = attrs_b
                grouped[(video_id, frame_id)].append(shape_b)
            except Exception:
                pass

    # Build pairs, optionally filter by video ids
    if args.filter_video:
        allow = {v.strip() for v in args.filter_video.split(",") if v.strip()}
        all_pairs = [((vid, fid), shapes) for (vid, fid), shapes in grouped.items() if vid in allow]
    else:
        all_pairs = list(grouped.items())
    print(f"Grouped (video,frame) pairs: {len(all_pairs)}")
    # Optional limits
    if args.limit_videos is not None or args.limit_frames is not None:
        by_video: Dict[str, List[Tuple[str, List[dict]]]] = defaultdict(list)
        for (vid, fid), shapes in all_pairs:
            by_video[vid].append((fid, shapes))
        limited_items: List[Tuple[Tuple[str, str], List[dict]]] = []
        # Make iteration order deterministic by sorting video ids
        for i, vid in enumerate(sorted(by_video.keys())):
            items = by_video[vid]
            if args.limit_videos is not None and i >= args.limit_videos:
                break
            items.sort(key=lambda x: x[0])
            if args.limit_frames is not None:
                items = items[: args.limit_frames]
            for fid, shapes in items:
                limited_items.append(((vid, fid), shapes))
        all_pairs = limited_items
        print(f"After limits: pairs -> {len(all_pairs)}")

    # Balanced per action sampling (overrides limit_* if provided together)
    selected_video_set = None
    if args.balanced_per_action is not None:
        # Build action -> list of ((vid,fid), shapes)
        action_to_items: Dict[str, List[Tuple[Tuple[str, str], List[dict]]]] = defaultdict(list)
        for key, shapes in all_pairs:
            # determine action label for this frame: first non-empty action among shapes
            action_name = ""
            for sh in shapes:
                attrs = sh.get("attributes") or {}
                if isinstance(attrs, dict):
                    val = attrs.get("action")
                    if isinstance(val, str) and val:
                        action_name = val
                        break
            action_to_items[action_name].append((key, shapes))
        # Deterministic ordering inside each action and across actions
        for items in action_to_items.values():
            items.sort(key=lambda it: (it[0][0], it[0][1]))
        selected: List[Tuple[Tuple[str, str], List[dict]]] = []
        dist: Dict[str, int] = {}
        for action_name in sorted(action_to_items.keys()):
            chosen = action_to_items[action_name][: max(0, args.balanced_per_action)]
            selected.extend(chosen)
            dist[action_name] = len(chosen)
        all_pairs = selected
        try:
            print("Balanced sampling enabled:", json.dumps({"per_action": args.balanced_per_action, "total_pairs": len(all_pairs), "picked": dist}, ensure_ascii=False))
        except Exception:
            print(f"Balanced sampling enabled: per_action={args.balanced_per_action}, total_pairs={len(all_pairs)}")
        selected_video_set = {vid for (vid, _fid), _shapes in all_pairs}

    # Collect videos for copying
    videos_with_ann = sorted({vid for (vid, _fid) in grouped.keys()})
    print(f"Videos with basketball annotations: {len(videos_with_ann)}")

    # If filters/limits/balanced applied, restrict copying strictly to selected videos
    if args.filter_video:
        allow = {v.strip() for v in args.filter_video.split(",") if v.strip()}
        copy_video_list = sorted([vid for vid in videos_with_ann if vid in allow])
    elif args.balanced_per_action is not None and selected_video_set is not None:
        copy_video_list = sorted(selected_video_set)
    elif args.limit_videos is not None or args.limit_frames is not None:
        copy_video_list = sorted({vid for (vid, _fid), _shapes in all_pairs})
    else:
        copy_video_list = videos_with_ann

    # Mirror-copy all frames (jpg/jpeg) for selected videos
    total_copied = 0
    print(f"Starting to copy frames for {len(copy_video_list)} videos...")
    for i, vid in enumerate(copy_video_list):
        if i % 10 == 0:
            print(f"Processing video {i+1}/{len(copy_video_list)}: {vid}")
        src_video_dir = os.path.join(args.rawframes_src, vid)
        if not os.path.isdir(src_video_dir):
            print(f"Warning: {src_video_dir} not found")
            continue
        dst_video_dir = os.path.join(args.out_root, vid)
        os.makedirs(dst_video_dir, exist_ok=True)
        jpg_count = 0
        for fname in os.listdir(src_video_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg')):
                continue
            src_img = os.path.join(src_video_dir, fname)
            dst_img = os.path.join(dst_video_dir, fname)
            if not os.path.exists(dst_img):
                try:
                    shutil.copy2(src_img, dst_img)
                    total_copied += 1
                    jpg_count += 1
                except Exception as e:
                    print(f"Error copying {src_img}: {e}")
        if i % 10 == 0:
            print(f"  Copied {jpg_count} images for {vid}")
    print(f"Copied full frames for {len(copy_video_list)} videos, images copied: {total_copied}")

    # Iterate groups, copy image and write JSON
    processed = 0
    missing = 0
    for idx, ((video_id, frame_id), shapes) in enumerate(all_pairs):
        src_video_dir = os.path.join(args.rawframes_src, video_id)
        if not os.path.exists(src_video_dir):
            missing += 1
            continue
        fname = find_image_filename(src_video_dir, frame_id)
        if idx < 5:
            print(f"check[{idx}] video_dir={src_video_dir} frame_id={frame_id} -> fname={fname}")
        if not fname:
            missing += 1
            continue
        src_img = os.path.join(src_video_dir, fname)

        # Prepare output dir
        dst_video_dir = os.path.join(args.out_root, video_id)
        os.makedirs(dst_video_dir, exist_ok=True)
        dst_img = os.path.join(dst_video_dir, fname)

        # Read image size with fallbacks
        size = get_image_size(src_img)
        if size is None:
            print(f"Warning: cannot determine size for {src_img}. Using width=0,height=0; JSON will contain zero-sized rectangles.")
            width, height = 0, 0
        else:
            width, height = size

        # Scale normalized boxes to pixel and expand to 4-corner rectangles
        final_shapes: List[dict] = []
        for sh in shapes:
            nx1, ny1, nx2, ny2 = sh["points"][0]
            px1 = float(nx1) * width; py1 = float(ny1) * height
            px2 = float(nx2) * width; py2 = float(ny2) * height
            rect = [
                [round(px1, 3), round(py1, 3)],
                [round(px2, 3), round(py1, 3)],
                [round(px2, 3), round(py2, 3)],
                [round(px1, 3), round(py2, 3)],
            ]
            sh2 = dict(sh)
            sh2["points"] = rect
            final_shapes.append(sh2)

        out_json = to_labelme_json(dst_img, width, height, final_shapes)
        json_name = os.path.splitext(fname)[0] + ".json"
        with open(os.path.join(dst_video_dir, json_name), "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        processed += 1

    print(f"Done: images and JSON written to {args.out_root}. processed_json={processed}, missing_pairs={missing}")


if __name__ == "__main__":
    main()

