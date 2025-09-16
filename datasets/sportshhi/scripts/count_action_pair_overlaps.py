#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
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
            out.append(((float(x1), float(y1), float(x2), float(y2)), (act or "").strip()))
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


def _parse_pbtxt_name_to_id(pbtxt_path: str) -> Dict[str, int]:
    """解析 pbtxt，返回 name->id 的映射。若文件缺失返回空映射。"""
    name_to_id: Dict[str, int] = {}
    if not pbtxt_path or not os.path.isfile(pbtxt_path):
        return name_to_id
    current_name: Optional[str] = None
    current_id: Optional[int] = None
    try:
        with open(pbtxt_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("name:"):
                    parts = line.split('"')
                    current_name = parts[1] if len(parts) > 1 else line.split(":", 1)[1].strip()
                elif line.startswith("label_id:"):
                    try:
                        current_id = int(line.split(":", 1)[1].strip())
                    except Exception:
                        current_id = None
                elif line.startswith("}"):
                    if current_name is not None and current_id is not None:
                        name_to_id[current_name] = current_id
                    current_name, current_id = None, None
        if current_name is not None and current_id is not None and current_name not in name_to_id:
            name_to_id[current_name] = current_id
    except Exception:
        return {}
    return name_to_id


def main() -> None:
    ap = argparse.ArgumentParser(description="统计 SportsHHI vs MultiSports 动作配对计数（IoU > 阈值）")
    ap.add_argument("--alignment_csv", type=str, required=True)
    ap.add_argument("--multisports_root", type=str, required=True)
    ap.add_argument("--sportshhi_root", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--allow_status", type=str, default="ok,no_below_threshold")
    ap.add_argument("--video-id", type=str, default="", help="仅统计指定 video_id；为空则处理全部允许行")
    ap.add_argument("--max_frames", type=int, default=0, help="逐帧向后最多处理多少帧；<=0 表示直到一侧结束")
    ap.add_argument("--out_root", type=str, default="", help="若提供，则把结果写入 <out_root>/__stats__/action_pair_counts.{json,csv}")
    ap.add_argument("--ms_pbtxt", type=str, default="", help="可选：MultiSports 的动作 pbtxt（用于 name->id）")
    ap.add_argument("--sh_pbtxt", type=str, default="", help="可选：SportsHHI 的动作 pbtxt（用于 name->id）")
    args = ap.parse_args()

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

    ms_name_to_id = _parse_pbtxt_name_to_id(args.ms_pbtxt) if args.ms_pbtxt else {}
    sh_name_to_id = _parse_pbtxt_name_to_id(args.sh_pbtxt) if args.sh_pbtxt else {}

    pair_counts_by_name: Counter[str] = Counter()
    pair_counts_by_id: Counter[str] = Counter()
    frames_considered = 0
    videos = set()
    total_pairs_over_th = 0

    for r in rows:
        video_id = (r.get("video_id") or "").strip()
        ms_file = (r.get("ms_file") or "").strip()
        sh_file = (r.get("sh_file") or "").strip()
        if not video_id or not ms_file or not sh_file:
            continue
        videos.add(video_id)

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
            sh_items = _read_bboxes_with_action(os.path.join(sh_dir, os.path.splitext(sh_fn)[0] + ".json"))
            ms_items = _read_bboxes_with_action(os.path.join(ms_dir, os.path.splitext(ms_fn)[0] + ".json"))
            if sh_items and ms_items:
                frames_considered += 1
                for ms_box, ms_act in ms_items:
                    for sh_box, sh_act in sh_items:
                        v = _iou_xyxy(ms_box, sh_box)
                        if v > args.threshold:
                            total_pairs_over_th += 1
                            key_name = f"{sh_act}-{ms_act}"
                            pair_counts_by_name[key_name] += 1

                            # 尝试映射到 ID：优先用 pbtxt；否则若名称是纯数字则直接解析
                            sh_id: Optional[int] = None
                            ms_id: Optional[int] = None
                            if sh_name_to_id:
                                sh_id = sh_name_to_id.get(sh_act)
                            if ms_name_to_id:
                                ms_id = ms_name_to_id.get(ms_act)
                            if sh_id is None:
                                try:
                                    sh_id = int(sh_act) if sh_act.strip().isdigit() else None
                                except Exception:
                                    sh_id = None
                            if ms_id is None:
                                try:
                                    ms_id = int(ms_act) if ms_act.strip().isdigit() else None
                                except Exception:
                                    ms_id = None

                            if sh_id is not None and ms_id is not None:
                                key_id = f"{sh_id}-{ms_id}"
                                pair_counts_by_id[key_id] += 1

            t += 1
            processed += 1
            if args.max_frames and processed >= int(args.max_frames):
                break

    # 排序输出（按计数降序）
    top_by_name = sorted(pair_counts_by_name.items(), key=lambda x: x[1], reverse=True)
    top_by_id = sorted(pair_counts_by_id.items(), key=lambda x: x[1], reverse=True)

    result = {
        "videos": sorted(videos),
        "frames_considered": int(frames_considered),
        "threshold": float(args.threshold),
        "pairs_over_threshold_total": int(total_pairs_over_th),
        "counts_by_name": [{"pair": k, "count": int(v)} for k, v in top_by_name],
        "counts_by_id": [{"pair": k, "count": int(v)} for k, v in top_by_id],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 可选写出到 __stats__
    if args.out_root:
        stats_dir = os.path.join(args.out_root, "__stats__")
        os.makedirs(stats_dir, exist_ok=True)

        def _sanitize(name: str) -> str:
            return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)[:80]

        suffix = f".{_sanitize(args.video_id)}" if args.video_id else ""
        json_path = os.path.join(stats_dir, f"action_pair_counts{suffix}.json")
        csv_path = os.path.join(stats_dir, f"action_pair_counts{suffix}.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 写 CSV：sh_action,ms_action,sh_id,ms_id,count
        # 需要将 by_name 聚合为列，同时尝试解析 id
        rows_out: List[List[str]] = []
        for k, cnt in top_by_name:
            sh_act, ms_act = k.split("-", 1)
            sh_id = ""
            ms_id = ""
            if sh_name_to_id and sh_act in sh_name_to_id:
                sh_id = str(sh_name_to_id[sh_act])
            elif sh_act.strip().isdigit():
                sh_id = sh_act.strip()
            if ms_name_to_id and ms_act in ms_name_to_id:
                ms_id = str(ms_name_to_id[ms_act])
            elif ms_act.strip().isdigit():
                ms_id = ms_act.strip()
            rows_out.append([sh_act, ms_act, sh_id, ms_id, str(int(cnt))])

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sh_action", "ms_action", "sh_id", "ms_id", "count"])
            for row in rows_out:
                w.writerow(row)


if __name__ == "__main__":
    main()


