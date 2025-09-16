#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

from action_data_analysis.io.json import read_labelme_json, extract_bbox_and_action


def _digits_from_name(name: str) -> Optional[int]:
    digits = ''.join([c for c in name if c.isdigit()])
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _build_index_map_from_dir(video_dir: str) -> Dict[int, str]:
    try:
        files = [f for f in os.listdir(video_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
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


def _read_bboxes_with_action(json_path: str) -> List[Tuple[Tuple[float, float, float, float], str]]:
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


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="输出第一个视频中所有 IoU>阈值 的跨域配对框（含动作），按时序打印")
    ap.add_argument("--alignment_csv", type=str, required=True)
    ap.add_argument("--multisports_root", type=str, required=True)
    ap.add_argument("--sportshhi_root", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--allow_status", type=str, default="ok,no_below_threshold")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--out_root", type=str, default="", help="若提供，则写入 <out_root>/__stats__/pairs_over_threshold[.per_pair].<video>.jsonl")
    ap.add_argument("--per_pair", action="store_true", help="按每个配对一行输出（stdout 与 JSONL）")
    args = ap.parse_args()

    import csv
    allow = {s.strip() for s in (args.allow_status or "").split(",") if s.strip()}
    first_row: Optional[Dict[str, str]] = None
    with open(args.alignment_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if allow and (row.get("status") or "").strip() not in allow:
                continue
            first_row = row
            break

    if not first_row:
        raise SystemExit("No row found in alignment_csv after filtering allow_status")

    video_id = (first_row.get("video_id") or "").strip()
    ms_file = (first_row.get("ms_file") or "").strip()
    sh_file = (first_row.get("sh_file") or "").strip()
    if not video_id or not ms_file or not sh_file:
        raise SystemExit("Invalid first row: missing video_id/ms_file/sh_file")

    try:
        sh_start = int(first_row.get("sh_index") or 0)
    except Exception:
        sh_start = _digits_from_name(os.path.splitext(sh_file)[0]) or 0
    try:
        ms_start = int(first_row.get("ms_index") or 0)
    except Exception:
        ms_start = _digits_from_name(os.path.splitext(ms_file)[0]) or 0
    if not sh_start or not ms_start:
        raise SystemExit("Invalid start indices")

    sh_dir = os.path.join(args.sportshhi_root, video_id)
    ms_dir = os.path.join(args.multisports_root, video_id)
    sh_map = _build_index_map_from_dir(sh_dir)
    ms_map = _build_index_map_from_dir(ms_dir)

    # 若指定输出目录，准备 JSONL 输出文件
    def _sanitize(name: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)[:80]

    jsonl_fp = None
    out_path = None
    if args.out_root:
        stats_dir = os.path.join(args.out_root, "__stats__")
        os.makedirs(stats_dir, exist_ok=True)
        fname = (
            f"pairs_over_threshold.per_pair.{_sanitize(video_id)}.gt{args.threshold:.2f}.jsonl"
            if args.per_pair
            else f"pairs_over_threshold.{_sanitize(video_id)}.gt{args.threshold:.2f}.jsonl"
        )
        out_path = os.path.join(stats_dir, fname)
        jsonl_fp = open(out_path, "w", encoding="utf-8")

    t = 0
    processed = 0
    while True:
        sh_idx = sh_start + t
        ms_idx = ms_start + t
        if sh_idx not in sh_map or ms_idx not in ms_map:
            break
        sh_fn = sh_map[sh_idx]
        ms_fn = ms_map[ms_idx]
        sh_items = _read_bboxes_with_action(os.path.join(sh_dir, os.path.splitext(sh_fn)[0] + ".json"))
        ms_items = _read_bboxes_with_action(os.path.join(ms_dir, os.path.splitext(ms_fn)[0] + ".json"))
        if sh_items and ms_items:
            pairs = []
            for i, (ms_box, ms_act) in enumerate(ms_items):
                for j, (sh_box, sh_act) in enumerate(sh_items):
                    v = _iou_xyxy(ms_box, sh_box)
                    if v > args.threshold:
                        pairs.append({
                            "ms_box": [*map(float, ms_box)],
                            "ms_action": ms_act,
                            "sh_box": [*map(float, sh_box)],
                            "sh_action": sh_act,
                            "iou": float(v),
                            "ms_idx": int(ms_idx),
                            "sh_idx": int(sh_idx),
                        })
            if pairs:
                if args.per_pair:
                    for p in pairs:
                        one = {
                            "video_id": video_id,
                            "t": int(t),
                            "ms_file": ms_fn,
                            "sh_file": sh_fn,
                            "threshold": float(args.threshold),
                            "ms_box": p["ms_box"],
                            "ms_action": p["ms_action"],
                            "sh_box": p["sh_box"],
                            "sh_action": p["sh_action"],
                            "iou": p["iou"],
                            "ms_idx": p["ms_idx"],
                            "sh_idx": p["sh_idx"],
                        }
                        print(json.dumps(one, ensure_ascii=False))
                        if jsonl_fp is not None:
                            jsonl_fp.write(json.dumps(one, ensure_ascii=False) + "\n")
                else:
                    record = {
                        "video_id": video_id,
                        "t": int(t),
                        "ms_file": ms_fn,
                        "sh_file": sh_fn,
                        "threshold": float(args.threshold),
                        "pairs": pairs,
                    }
                    # 控制台输出
                    print(json.dumps(record, ensure_ascii=False))
                    # 文件输出（JSONL）
                    if jsonl_fp is not None:
                        jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        t += 1
        processed += 1
        if args.max_frames and processed >= int(args.max_frames):
            break

    if jsonl_fp is not None:
        jsonl_fp.close()
        print(json.dumps({
            "video_id": video_id,
            "written": out_path,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()


