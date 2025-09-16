#!/usr/bin/env python3
import os
import csv
import json
import shutil
import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import math

# 复用项目内的 LabelMe JSON 解析
from action_data_analysis.io.json import read_labelme_json, extract_bbox_and_action


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


def list_videos(root: str) -> List[str]:
    try:
        return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    except FileNotFoundError:
        return []


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


def best_iou_each(source: List[Tuple[float, float, float, float]], target: List[Tuple[float, float, float, float]]) -> List[float]:
    if not source:
        return []
    if not target:
        return [0.0 for _ in source]
    out: List[float] = []
    for a in source:
        best = 0.0
        for b in target:
            v = iou_xyxy(a, b)
            if v > best:
                best = v
        out.append(best)
    return out


def extract_bboxes_from_labelme(json_path: str) -> List[Tuple[float, float, float, float]]:
    if not os.path.isfile(json_path):
        return []
    try:
        rec = read_labelme_json(json_path)
        bboxes: List[Tuple[float, float, float, float]] = []
        for shape in rec.get("shapes", []) or []:
            parsed = extract_bbox_and_action(shape)
            if not parsed:
                continue
            (x1, y1, x2, y2), _ = parsed
            # 过滤非法框
            if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            bboxes.append((float(x1), float(y1), float(x2), float(y2)))
        return bboxes
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 alignment CSV 复制两侧帧与 JSON 到目标目录，并统计检测框 IoU 重合度")
    parser.add_argument("--alignment_csv", type=str,
                        default="/storage/wangxinxing/code/action_data_analysis/datasets/sportshhi/results/alignment_search.csv")
    parser.add_argument("--multisports_root", type=str,
                        default="/storage/wangxinxing/code/action_data_analysis/data/MultiSports_json")
    parser.add_argument("--sportshhi_root", type=str,
                        default="/storage/wangxinxing/code/action_data_analysis/data/SportsHHI_json")
    parser.add_argument("--out_root", type=str,
                        default="/storage/wangxinxing/code/action_data_analysis/data/MultiSports_SportsHHI_overlap_json")
    parser.add_argument("--allow_status", type=str, default="ok",
                        help="逗号分隔，允许的对齐状态；默认: ok。可选含 no_below_threshold")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条记录；<=0 表示全部")
    parser.add_argument("--thresholds", type=str, default="0.5 0.75 0.9",
                        help="统计阈值（空格分隔）用于 >=IoU 的占比统计")
    parser.add_argument("--max_frames", type=int, default=0, help="逐帧向后最多处理多少帧；<=0 表示直到一侧结束")
    args = parser.parse_args()

    allow_status = {s.strip() for s in (args.allow_status or "").split(",") if s.strip()}
    rows = read_alignment_rows(args.alignment_csv, status_allow=allow_status if allow_status else None)
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    ensure_dir(args.out_root)
    stats_dir = os.path.join(args.out_root, "__stats__")
    ensure_dir(stats_dir)

    thresholds: List[float] = []
    for tok in (args.thresholds or "").split():
        try:
            thresholds.append(float(tok))
        except Exception:
            continue
    thresholds = sorted(set(thresholds))

    summary_csv = os.path.join(stats_dir, "overlap_summary.csv")
    per_frame_csv = os.path.join(stats_dir, "overlap_by_frame.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8") as fcsv, \
         open(per_frame_csv, "w", newline="", encoding="utf-8") as fframe:
        writer = csv.writer(fcsv)
        header = [
            "video_id", "ms_file", "sh_file", "k", "mean", "status",
            "common_frames",  # 从起始帧向后，双方均有标注的帧数
            "num_ms_boxes", "num_sh_boxes",  # 汇总公共标注帧的框总数
            "mean_best_iou_ms_to_sh", "mean_best_iou_sh_to_ms",  # 在公共标注帧上的均值
        ] + [f"ms_to_sh@{t}" for t in thresholds] + [f"sh_to_ms@{t}" for t in thresholds]
        writer.writerow(header)

        frame_writer = csv.writer(fframe)
        frame_header = [
            "video_id", "t", "sh_index", "ms_index", "sh_file", "ms_file",
            "num_ms_boxes", "num_sh_boxes",
            "mean_best_iou_ms_to_sh", "mean_best_iou_sh_to_ms",
        ] + [f"ms_to_sh@{t}" for t in thresholds] + [f"sh_to_ms@{t}" for t in thresholds]
        frame_writer.writerow(frame_header)

        copied_images = 0
        copied_jsons = 0
        for i, r in enumerate(rows):
            video_id = (r.get("video_id") or "").strip()
            ms_file = (r.get("ms_file") or "").strip()
            sh_file = (r.get("sh_file") or "").strip()
            k = r.get("k") or ""
            mean = r.get("mean") or ""
            status = (r.get("status") or "").strip()

            if not video_id or not ms_file or not sh_file:
                continue

            # 源路径
            ms_dir = os.path.join(args.multisports_root, video_id)
            sh_dir = os.path.join(args.sportshhi_root, video_id)
            src_ms_img = os.path.join(ms_dir, ms_file)
            src_sh_img = os.path.join(sh_dir, sh_file)
            src_ms_json = os.path.splitext(src_ms_img)[0] + ".json"
            src_sh_json = os.path.splitext(src_sh_img)[0] + ".json"

            # 目标路径：<out>/<video_id>/
            dst_dir = os.path.join(args.out_root, video_id)
            ensure_dir(dst_dir)

            # 拷贝图片，带前缀避免重名
            dst_ms_img = os.path.join(dst_dir, f"ms_{ms_file}")
            dst_sh_img = os.path.join(dst_dir, f"sh_{sh_file}")
            if safe_copy(src_ms_img, dst_ms_img):
                copied_images += 1
            if safe_copy(src_sh_img, dst_sh_img):
                copied_images += 1

            # 拷贝 JSON（如果存在）
            dst_ms_json = os.path.join(dst_dir, f"ms_{os.path.splitext(ms_file)[0]}.json")
            dst_sh_json = os.path.join(dst_dir, f"sh_{os.path.splitext(sh_file)[0]}.json")
            if safe_copy(src_ms_json, dst_ms_json):
                copied_jsons += 1
            if safe_copy(src_sh_json, dst_sh_json):
                copied_jsons += 1

            # 解析起始索引
            sh_start_idx: Optional[int] = None
            ms_start_idx: Optional[int] = None
            try:
                sh_start_idx = int(r.get("sh_index") or 0)
            except Exception:
                sh_start_idx = _digits_from_name(os.path.splitext(sh_file)[0])
            try:
                ms_start_idx = int(r.get("ms_index") or 0)
            except Exception:
                ms_start_idx = _digits_from_name(os.path.splitext(ms_file)[0])
            if not sh_start_idx or not ms_start_idx:
                continue

            sh_idx_map = build_index_map_from_dir(sh_dir)
            ms_idx_map = build_index_map_from_dir(ms_dir)

            # 聚合：仅统计双方均有标注（均有框）的帧
            agg_ms2sh: List[float] = []
            agg_sh2ms: List[float] = []
            agg_ms_boxes = 0
            agg_sh_boxes = 0
            agg_common_frames = 0

            # 从 t=0 开始复制；t 逐增，直到一侧无帧或达到 max_frames
            t = 0
            processed_frames = 0
            while True:
                sh_idx = sh_start_idx + t
                ms_idx = ms_start_idx + t
                if sh_idx not in sh_idx_map or ms_idx not in ms_idx_map:
                    break

                sh_fn = sh_idx_map[sh_idx]
                ms_fn = ms_idx_map[ms_idx]
                # 复制图片
                if safe_copy(os.path.join(sh_dir, sh_fn), os.path.join(dst_dir, f"sh_{sh_fn}")):
                    copied_images += 1
                if safe_copy(os.path.join(ms_dir, ms_fn), os.path.join(dst_dir, f"ms_{ms_fn}")):
                    copied_images += 1
                # 复制 JSON
                if safe_copy(os.path.join(sh_dir, os.path.splitext(sh_fn)[0] + ".json"),
                             os.path.join(dst_dir, f"sh_{os.path.splitext(sh_fn)[0]}.json")):
                    copied_jsons += 1
                if safe_copy(os.path.join(ms_dir, os.path.splitext(ms_fn)[0] + ".json"),
                             os.path.join(dst_dir, f"ms_{os.path.splitext(ms_fn)[0]}.json")):
                    copied_jsons += 1

                # 逐帧统计：仅在两侧均有标注时计入 per_frame 与聚合
                ms_b = extract_bboxes_from_labelme(os.path.join(ms_dir, os.path.splitext(ms_fn)[0] + ".json"))
                sh_b = extract_bboxes_from_labelme(os.path.join(sh_dir, os.path.splitext(sh_fn)[0] + ".json"))
                if ms_b and sh_b:
                    ms2sh = best_iou_each(ms_b, sh_b)
                    sh2ms = best_iou_each(sh_b, ms_b)
                    agg_ms2sh.extend(ms2sh)
                    agg_sh2ms.extend(sh2ms)
                    agg_ms_boxes += len(ms_b)
                    agg_sh_boxes += len(sh_b)
                    agg_common_frames += 1

                    mean_ms2sh = float(sum(ms2sh) / len(ms2sh)) if ms2sh else 0.0
                    mean_sh2ms = float(sum(sh2ms) / len(sh2ms)) if sh2ms else 0.0

                    frame_vals = [
                        video_id, t, sh_idx, ms_idx, sh_fn, ms_fn,
                        len(ms_b), len(sh_b), f"{mean_ms2sh:.6f}", f"{mean_sh2ms:.6f}",
                    ]
                    for th in thresholds:
                        frac = float(sum(1 for v in ms2sh if v >= th) / len(ms2sh)) if ms2sh else 0.0
                        frame_vals.append(f"{frac:.6f}")
                    for th in thresholds:
                        frac = float(sum(1 for v in sh2ms if v >= th) / len(sh2ms)) if sh2ms else 0.0
                        frame_vals.append(f"{frac:.6f}")
                    frame_writer.writerow(frame_vals)

                t += 1
                processed_frames += 1
                if args.max_frames and processed_frames >= int(args.max_frames):
                    break

            # 写入汇总（仅基于公共标注帧）
            mean_ms_to_sh = float(sum(agg_ms2sh) / len(agg_ms2sh)) if agg_ms2sh else 0.0
            mean_sh_to_ms = float(sum(agg_sh2ms) / len(agg_sh2ms)) if agg_sh2ms else 0.0

            row_vals = [
                video_id, ms_file, sh_file, k, mean, status,
                agg_common_frames,
                agg_ms_boxes, agg_sh_boxes,
                f"{mean_ms_to_sh:.6f}", f"{mean_sh_to_ms:.6f}",
            ]
            for t in thresholds:
                frac = float(sum(1 for v in agg_ms2sh if v >= t) / len(agg_ms2sh)) if agg_ms2sh else 0.0
                row_vals.append(f"{frac:.6f}")
            for t in thresholds:
                frac = float(sum(1 for v in agg_sh2ms if v >= t) / len(agg_sh2ms)) if agg_sh2ms else 0.0
                row_vals.append(f"{frac:.6f}")
            writer.writerow(row_vals)

    # 同步写一个简单的 README 说明
    readme_path = os.path.join(stats_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as fr:
        fr.write(
            "本目录由 copy_and_compare_from_alignment.py 生成：\n"
            "- overlap_summary.csv: 每条对齐记录起始帧的 IoU 概览。\n"
            "- overlap_by_frame.csv: 自起始帧起，逐帧向后直到一侧无帧为止的 IoU 统计。\n"
            "- 上级目录下每个视频子目录包含复制的两侧图片与 JSON（前缀 ms_/sh_）。\n"
        )

    print(f"Done. Wrote files: summary={summary_csv}, per_frame={per_frame_csv}")


if __name__ == "__main__":
    main()


