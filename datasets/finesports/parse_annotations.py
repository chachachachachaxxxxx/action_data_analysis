#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse FineSports annotations (FineSports-GT.pkl) and generate:
- results/annotations_summary.json
- results/README_annotations.md

This script infers the structure of gttubes and computes basic stats.
"""

import os
import json
import math
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple


DATA_ROOT = os.path.abspath(os.path.dirname(__file__))
ANNOTATIONS_PKL = os.path.join(DATA_ROOT, "annotations", "FineSports-GT.pkl")
RESULTS_DIR = os.path.join(DATA_ROOT, "results")
DOCS_DESC = os.path.join(DATA_ROOT, "docs", "description.md")


def safe_load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "mean": 0.0, "p75": 0.0, "max": 0.0}
    s = sorted(values)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return float(s[0])
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return float(s[lo])
        frac = idx - lo
        return float(s[lo] * (1 - frac) + s[hi] * frac)

    return {
        "min": float(s[0]),
        "p25": q(0.25),
        "mean": float(sum(s) / n),
        "p75": q(0.75),
        "max": float(s[-1]),
    }


def infer_bbox_columns(arr) -> Tuple[int, int, int, int]:
    """Heuristic: assume last 4 columns are x1,y1,x2,y2 when width>=5."""
    if arr.ndim != 2 or arr.shape[1] < 4:
        return (-1, -1, -1, -1)
    # Prefer last 4
    x1_col, y1_col, x2_col, y2_col = arr.shape[1] - 4, arr.shape[1] - 3, arr.shape[1] - 2, arr.shape[1] - 1
    return (x1_col, y1_col, x2_col, y2_col)


def compute_stats(d: Dict[str, Any]) -> Dict[str, Any]:
    labels: List[str] = d.get("labels", [])
    gttubes: Dict[str, Any] = d.get("gttubes", {})
    nframes: Dict[str, int] = d.get("nframes", {})
    resolution: Dict[str, Any] = d.get("resolution", {})

    # Split info if available
    split_keys = ["train_videos", "val_videos", "test_videos", "videos"]
    splits = {}
    for k in split_keys:
        if k in d and isinstance(d[k], list):
            splits[k] = list(d[k])

    # Count actions per class label id
    action_counter: Counter = Counter()
    per_video_counts: Counter = Counter()
    # store normalized stats (0..1) when resolution is available; otherwise best-effort
    bbox_w_primary: List[float] = []
    bbox_h_primary: List[float] = []
    bbox_area_primary: List[float] = []

    coords_out_of_range = 0
    x1_ge_x2 = 0
    y1_ge_y2 = 0
    missing_values = 0

    videos = list(gttubes.keys())
    for vid in videos:
        per_video_sum = 0
        class_to_tubes = gttubes.get(vid, {})
        if not isinstance(class_to_tubes, dict):
            continue
        for cls_id, tubes in class_to_tubes.items():
            if not isinstance(tubes, list):
                continue
            action_counter[cls_id] += len(tubes)
            per_video_sum += len(tubes)
            for a in tubes:
                if getattr(a, "ndim", None) != 2 or a.shape[1] < 4:
                    missing_values += 1
                    continue
                x1c, y1c, x2c, y2c = infer_bbox_columns(a)
                if x1c < 0:
                    missing_values += 1
                    continue
                x1, y1, x2, y2 = a[:, x1c].astype(float), a[:, y1c].astype(float), a[:, x2c].astype(float), a[:, y2c].astype(float)
                # try to get (H, W) from resolution
                H, W = None, None
                if isinstance(resolution.get(vid), (list, tuple)) and len(resolution.get(vid)) >= 2:
                    r0, r1 = resolution[vid][0], resolution[vid][1]
                    # heuristic: larger one is width
                    if r0 >= r1:
                        H, W = float(r1), float(r0)
                    else:
                        H, W = float(r0), float(r1)

                # compute widths/heights and anomalies in pixel domain first
                w_px = (x2 - x1)
                h_px = (y2 - y1)
                x1_ge_x2 += int((w_px <= 0).sum())
                y1_ge_y2 += int((h_px <= 0).sum())

                # normalize if resolution available; otherwise pass-through but clamp-based check skipped
                if H and W and H > 0 and W > 0:
                    x1n = x1 / W
                    y1n = y1 / H
                    x2n = x2 / W
                    yn2 = y2 / H
                    wn = (x2n - x1n)
                    hn = (yn2 - y1n)
                    bbox_w_primary.extend([float(v) for v in wn.tolist()])
                    bbox_h_primary.extend([float(v) for v in hn.tolist()])
                    bbox_area_primary.extend([(float(wi) * float(hi)) for wi, hi in zip(wn.tolist(), hn.tolist())])
                    # range check after normalization with tolerance
                    tol = 1e-6
                    coords_out_of_range += int(((x1n < -tol) | (x1n > 1 + tol) | (y1n < -tol) | (y1n > 1 + tol) | (x2n < -tol) | (x2n > 1 + tol) | (yn2 < -tol) | (yn2 > 1 + tol)).sum())
                else:
                    # fall back to pixel stats (these won't be directly comparable across videos)
                    bbox_w_primary.extend([float(v) for v in w_px.tolist()])
                    bbox_h_primary.extend([float(v) for v in h_px.tolist()])
                    bbox_area_primary.extend([(float(wi) * float(hi)) for wi, hi in zip(w_px.tolist(), h_px.tolist())])
                    # skip coords_out_of_range because we don't know bounds without resolution

        per_video_counts[vid] = per_video_sum

    # Build summary
    basic = {
        "num_videos": len(videos),
        "num_labels": len(labels),
        "num_action_instances": int(sum(action_counter.values())),
    }

    # Map class id to name if possible (labels may be list of names; align index->name)
    class_map_sample = {}
    if isinstance(labels, list) and labels:
        # try treat cls_id as string/int indexing into labels if numeric
        for i, name in enumerate(labels[:12]):
            class_map_sample[str(i)] = name

    top_actions = []
    # cls_id may be int or str; convert to str keys
    for cls_id, cnt in action_counter.most_common(20):
        key = str(cls_id)
        label_name = None
        if isinstance(cls_id, int) and 0 <= cls_id < len(labels):
            label_name = labels[cls_id]
        top_actions.append([label_name if label_name else key, int(cnt)])

    summary: Dict[str, Any] = {
        "paths": {
            "annotations_dir": os.path.join(DATA_ROOT, "annotations"),
            "results_dir": RESULTS_DIR,
        },
        "schema": {
            "notes": "gttubes[video_id][class_id] -> list of ndarray tubes; last 4 columns are (x1,y1,x2,y2) in pixel coordinates. We normalize by per-video resolution (H,W).",
            "coordinates_normalized": True,
            "coordinate_range": "[0, 1] after normalization by resolution",
        },
        "basic": basic,
        "splits": {k: len(v) for k, v in splits.items()},
        "labels_sample": labels[:12] if isinstance(labels, list) else None,
        "top_actions": top_actions,
        "bbox_primary": {
            "width": quantiles(bbox_w_primary),
            "height": quantiles(bbox_h_primary),
            "area": quantiles(bbox_area_primary),
            "anomalies": {
                "coords_out_of_range": int(coords_out_of_range),
                "x1_ge_x2": int(x1_ge_x2),
                "y1_ge_y2": int(y1_ge_y2),
                "missing_values": int(missing_values),
            },
            "num_boxes": int(len(bbox_w_primary)),
        },
        "per_video_top": per_video_counts.most_common(30),
    }

    return summary


def read_desc_md(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_readme(summary: Dict[str, Any], desc_md: str) -> str:
    lines: List[str] = []
    # 数据概览
    lines.append("## 数据概览")
    lines.append(f"- **标注目录**: `{summary['paths']['annotations_dir']}`")
    lines.append(f"- **结果目录**: `{summary['paths']['results_dir']}`")
    lines.append(f"- **视频数**: {summary['basic']['num_videos']}")
    lines.append(f"- **动作实例数**: {summary['basic']['num_action_instances']}")
    lines.append(f"- **标签数（细粒度）**: {summary['basic']['num_labels']}")
    lines.append("")

    # 类别与分布
    lines.append("## 类别与分布")
    if summary.get("labels_sample"):
        lines.append(f"- **标签样例**: {json.dumps(summary['labels_sample'], ensure_ascii=False)}")
    if summary.get("top_actions"):
        lines.append("- **Top-20 类别计数**:")
        for name, cnt in summary["top_actions"]:
            lines.append(f"  - {name}: {cnt}")
    lines.append("")

    # 主框尺寸统计（tube 中逐帧 bbox）
    bp = summary.get("bbox_primary", {})
    if bp:
        lines.append("## 主框尺寸统计")
        for key in ["width", "height", "area"]:
            if key in bp:
                q = bp[key]
                lines.append(f"- **{key}**: min={q['min']}, p25={q['p25']}, mean={q['mean']}, p75={q['p75']}, max={q['max']}")
        if "anomalies" in bp:
            an = bp["anomalies"]
            lines.append("- **异常**:")
            lines.append(f"  - 坐标越界: {an['coords_out_of_range']}")
            lines.append(f"  - x1>=x2: {an['x1_ge_x2']}")
            lines.append(f"  - y1>=y2: {an['y1_ge_y2']}")
            lines.append(f"  - 缺失值: {an['missing_values']}")
        lines.append("")

    # 每视频标注 Top-20
    pvt = summary.get("per_video_top", [])
    if pvt:
        lines.append("## 每视频标注 Top-20")
        for vid, cnt in pvt[:20]:
            lines.append(f"- {vid}: {cnt}")
        lines.append("")

    # 结合描述文档
    if desc_md.strip():
        lines.append("## 数据集说明（摘录）")
        lines.append("")
        lines.append(desc_md.strip())

    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dir(RESULTS_DIR)
    data = safe_load_pickle(ANNOTATIONS_PKL)
    summary = compute_stats(data)

    # write json summary
    json_path = os.path.join(RESULTS_DIR, "annotations_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # write README
    desc = read_desc_md(DOCS_DESC)
    readme_path = os.path.join(RESULTS_DIR, "README_annotations.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(render_readme(summary, desc))

    print(f"Wrote {json_path}")
    print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()

