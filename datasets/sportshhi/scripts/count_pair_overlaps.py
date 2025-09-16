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


def _extract_bboxes(json_path: str) -> List[Tuple[float, float, float, float]]:
    if not os.path.isfile(json_path):
        return []
    try:
        rec = read_labelme_json(json_path)
        out: List[Tuple[float, float, float, float]] = []
        for sh in rec.get("shapes", []) or []:
            parsed = extract_bbox_and_action(sh)
            if not parsed:
                continue
            (x1, y1, x2, y2), _ = parsed
            if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            out.append((float(x1), float(y1), float(x2), float(y2)))
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
    ap = argparse.ArgumentParser(description="统计 NxM 框对 IoU 分布直方图（逐帧累加）")
    ap.add_argument("--alignment_csv", type=str, required=True)
    ap.add_argument("--multisports_root", type=str, required=True)
    ap.add_argument("--sportshhi_root", type=str, required=True)
    ap.add_argument("--video-id", type=str, default="", help="仅统计指定 video_id；为空则处理 CSV 中全部允许行")
    ap.add_argument("--allow_status", type=str, default="ok,no_below_threshold")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--out_root", type=str, default="", help="若提供，则把结果写入 <out_root>/__stats__/pair_iou_hist.{json,csv}")
    args = ap.parse_args()

    import csv
    allow = {s.strip() for s in (args.allow_status or "").split(",") if s.strip()}

    rows: List[Dict[str, str]] = []
    with open(args.alignment_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if allow and (row.get("status") or "").strip() not in allow:
                continue
            if args.video_id and (row.get("video_id") or "").strip() != args.video_id:
                continue
            rows.append(row)

    total_pairs = 0
    # 10 桶： [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
    hist_counts = [0 for _ in range(10)]
    frames_considered = 0
    videos = set()

    for r in rows:
        video_id = (r.get("video_id") or "").strip()
        ms_file = (r.get("ms_file") or "").strip()
        sh_file = (r.get("sh_file") or "").strip()
        videos.add(video_id)
        if not video_id or not ms_file or not sh_file:
            continue

        ms_dir = os.path.join(args.multisports_root, video_id)
        sh_dir = os.path.join(args.sportshhi_root, video_id)

        try:
            sh_start = int(r.get("sh_index") or 0)
        except Exception:
            sh_start = _digits_from_name(os.path.splitext(sh_file)[0]) or 0
        try:
            ms_start = int(r.get("ms_index") or 0)
        except Exception:
            ms_start = _digits_from_name(os.path.splitext(ms_file)[0]) or 0
        if not sh_start or not ms_start:
            continue

        sh_map = _build_index_map_from_dir(sh_dir)
        ms_map = _build_index_map_from_dir(ms_dir)

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
            sh_b = _extract_bboxes(sh_json)
            ms_b = _extract_bboxes(ms_json)
            if sh_b and ms_b:
                frames_considered += 1
                n, m = len(ms_b), len(sh_b)
                total_pairs += n * m
                for a in ms_b:
                    for b in sh_b:
                        v = _iou_xyxy(a, b)
                        # 将 IoU 映射到桶索引：0..9
                        idx = int(v * 10)
                        if idx < 0:
                            idx = 0
                        if idx > 9:
                            idx = 9
                        hist_counts[idx] += 1
            t += 1
            processed += 1
            if args.max_frames and processed >= int(args.max_frames):
                break

    bins = [f"{i/10:.1f}-{(i+1)/10:.1f}" if i < 9 else "0.9-1.0" for i in range(10)]
    result = {
        "videos": sorted(videos),
        "frames_considered": frames_considered,
        "pairs_total": total_pairs,
        "bins": bins,
        "histogram": hist_counts,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 可选写出到 __stats__
    if args.out_root:
        stats_dir = os.path.join(args.out_root, "__stats__")
        os.makedirs(stats_dir, exist_ok=True)
        # 文件名后缀：若限定了 video-id，则在文件名中加入安全后缀
        def _sanitize(name: str) -> str:
            return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)[:80]

        suffix = f".{_sanitize(args.video_id)}" if args.video_id else ""
        json_path = os.path.join(stats_dir, f"pair_iou_hist{suffix}.json")
        csv_path = os.path.join(stats_dir, f"pair_iou_hist{suffix}.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 写 CSV：两列 bin,count
        import csv as _csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["bin", "count"])
            for b, cnt in zip(bins, hist_counts):
                w.writerow([b, int(cnt)])


if __name__ == "__main__":
    main()


