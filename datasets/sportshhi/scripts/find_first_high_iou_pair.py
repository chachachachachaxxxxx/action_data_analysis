#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from typing import List, Optional, Tuple

from action_data_analysis.io.json import read_labelme_json, extract_bbox_and_action


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def extract_bboxes_with_action(json_path: str) -> List[Tuple[Tuple[float, float, float, float], str]]:
    if not os.path.isfile(json_path):
        return []
    try:
        rec = read_labelme_json(json_path)
        out: List[Tuple[Tuple[float, float, float, float], str]] = []
        for sh in rec.get("shapes", []) or []:
            parsed = extract_bbox_and_action(sh)
            if not parsed:
                continue
            (x1, y1, x2, y2), act = parsed
            if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(((float(x1), float(y1), float(x2), float(y2)), act or ""))
        return out
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description="找到 alignment CSV 中首个视频，从其起始帧向后，首个 IoU≥阈值 的框对")
    ap.add_argument("--alignment_csv", type=str, required=True)
    ap.add_argument("--multisports_root", type=str, required=True)
    ap.add_argument("--sportshhi_root", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    import csv
    with open(args.alignment_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        first = None
        for row in r:
            first = row
            break
    if not first:
        raise SystemExit("alignment_csv is empty")

    video_id = (first.get("video_id") or "").strip()
    if not video_id:
        raise SystemExit("missing video_id in first row")
    sh_file = (first.get("sh_file") or "").strip()
    ms_file = (first.get("ms_file") or "").strip()

    try:
        sh_start = int(first.get("sh_index") or 0)
    except Exception:
        sh_start = int(os.path.splitext(sh_file)[0]) if sh_file.isdigit() else 1
    try:
        ms_start = int(first.get("ms_index") or 0)
    except Exception:
        ms_start = int(os.path.splitext(ms_file)[0]) if ms_file.isdigit() else 1
    if not sh_start or not ms_start:
        raise SystemExit("invalid start indices")

    def build_index_map(root: str) -> dict:
        try:
            files = sorted([f for f in os.listdir(root) if f.lower().endswith((".jpg",".jpeg",".png"))])
        except FileNotFoundError:
            return {}
        m = {}
        for fn in files:
            name, _ = os.path.splitext(fn)
            try:
                idx = int(name)
            except Exception:
                continue
            if idx not in m:
                m[idx] = fn
        return m

    sh_dir = os.path.join(args.sportshhi_root, video_id)
    ms_dir = os.path.join(args.multisports_root, video_id)
    sh_map = build_index_map(sh_dir)
    ms_map = build_index_map(ms_dir)

    t = 0
    processed = 0
    while True:
        sh_idx = sh_start + t
        ms_idx = ms_start + t
        if sh_idx not in sh_map or ms_idx not in ms_map:
            break
        sh_fn = sh_map[sh_idx]
        ms_fn = ms_map[ms_idx]
        sh_json = os.path.join(sh_dir, os.path.splitext(sh_fn)[0] + ".json")
        ms_json = os.path.join(ms_dir, os.path.splitext(ms_fn)[0] + ".json")
        sh_items = extract_bboxes_with_action(sh_json)
        ms_items = extract_bboxes_with_action(ms_json)
        if sh_items and ms_items:
            pairs = []
            for i, (a_box, a_act) in enumerate(ms_items):
                for j, (b_box, b_act) in enumerate(sh_items):
                    v = iou_xyxy(a_box, b_box)
                    if v >= args.threshold:
                        pairs.append((i, j, v))
            if pairs:
                pairs.sort(key=lambda x: -x[2])
                print(json.dumps({
                    "video_id": video_id,
                    "t": t,
                    "ms_index": ms_idx,
                    "sh_index": sh_idx,
                    "ms_file": ms_fn,
                    "sh_file": sh_fn,
                    "threshold": args.threshold,
                    "pairs": [
                        {
                            "ms_box": [*map(float, ms_items[i][0])],
                            "ms_action": ms_items[i][1],
                            "sh_box": [*map(float, sh_items[j][0])],
                            "sh_action": sh_items[j][1],
                            "iou": float(v)
                        } for i, j, v in pairs
                    ]
                }, ensure_ascii=False))
                return
        t += 1
        processed += 1
        if args.max_frames and processed >= int(args.max_frames):
            break

    print(json.dumps({"video_id": video_id, "message": "no pairs >= threshold found"}, ensure_ascii=False))


if __name__ == "__main__":
    main()


