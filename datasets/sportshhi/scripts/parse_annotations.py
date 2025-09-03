import os
import json
import pickle
from collections import Counter
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


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


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_action_list(pbtxt_path: str) -> dict:
    """Parse sports_action_list.pbtxt -> {id: {name, label_type}}"""
    actions = {}
    if not os.path.exists(pbtxt_path):
        return actions

    current = {}
    with open(pbtxt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("name:"):
                # name: "basketball - pass"
                name = line.split('"')
                current["name"] = name[1] if len(name) > 1 else line.split(":", 1)[1].strip()
            elif line.startswith("label_id:"):
                try:
                    current["id"] = int(line.split(":", 1)[1].strip())
                except Exception:
                    continue
            elif line.startswith("label_type:"):
                current["label_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("}"):
                if "id" in current and "name" in current:
                    actions[current["id"]] = {
                        "name": current.get("name"),
                        "label_type": current.get("label_type"),
                    }
                current = {}

    # Some pbtxt files do not end with a closing brace on the last block
    if current and "id" in current and "name" in current and current["id"] not in actions:
        actions[current["id"]] = {
            "name": current.get("name"),
            "label_type": current.get("label_type"),
        }

    return actions


def load_split_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(csv_path, header=None, names=CSV_COLUMNS)
    # Normalize dtypes
    df["frame_id"] = df["frame_id"].astype(str)
    # Coordinates are floats in [0,1] per sample; keep as float
    for col in ["x1", "y1", "x2", "y2", "x1_2", "y1_2", "x2_2", "y2_2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["action_id", "person_id", "instance_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def attach_action_names(df: pd.DataFrame, id_to_action: dict) -> pd.DataFrame:
    if df.empty:
        return df
    id_to_name = {i: meta.get("name") for i, meta in id_to_action.items()}
    id_to_type = {i: meta.get("label_type") for i, meta in id_to_action.items()}
    df = df.copy()
    df["action_name"] = df["action_id"].map(id_to_name)
    df["action_type"] = df["action_id"].map(id_to_type)
    # sport group from action_name prefix (e.g., "basketball - ...")
    def _sport_from_name(name: str) -> str:
        if not isinstance(name, str):
            return None
        return name.split(" - ")[0].strip()

    df["sport"] = df["action_name"].map(_sport_from_name)
    return df


def compute_bbox_stats(df: pd.DataFrame, prefix: str = "") -> dict:
    keys = [f"{prefix}x1", f"{prefix}y1", f"{prefix}x2", f"{prefix}y2"]
    if any(k not in df.columns for k in keys) or df.empty:
        return {}

    w = (df[keys[2]] - df[keys[0]]).astype(float)
    h = (df[keys[3]] - df[keys[1]]).astype(float)
    area = (w * h).astype(float)

    def _series_stats(s: pd.Series) -> dict:
        s = s.dropna()
        if s.empty:
            return {}
        return {
            "min": float(np.nanmin(s)),
            "p25": float(np.nanpercentile(s, 25)),
            "mean": float(np.nanmean(s)),
            "p75": float(np.nanpercentile(s, 75)),
            "max": float(np.nanmax(s)),
        }

    anomalies = {
        "coords_out_of_range": int(
            (~df[keys[0]].between(0, 1) |
             ~df[keys[1]].between(0, 1) |
             ~df[keys[2]].between(0, 1) |
             ~df[keys[3]].between(0, 1)).sum()
        ),
        "x1_ge_x2": int((df[keys[0]] >= df[keys[2]]).sum()),
        "y1_ge_y2": int((df[keys[1]] >= df[keys[3]]).sum()),
        "missing_values": int(df[keys].isna().any(axis=1).sum()),
    }

    return {
        "width": _series_stats(w),
        "height": _series_stats(h),
        "area": _series_stats(area),
        "anomalies": anomalies,
        "num_boxes": int(len(df) - int(df[keys].isna().any(axis=1).sum())),
    }


def top_k(counter_like: dict, k: int = 20) -> list:
    items = sorted(counter_like.items(), key=lambda x: x[1], reverse=True)
    return items[:k]


def analyze_annotations() -> dict:
    action_pbtxt = os.path.join(ANNOTATIONS_DIR, "sports_action_list.pbtxt")
    id_to_action = parse_action_list(action_pbtxt)

    train_csv = os.path.join(ANNOTATIONS_DIR, "sports_train_v1.csv")
    val_csv = os.path.join(ANNOTATIONS_DIR, "sports_val_v1.csv")

    df_train = load_split_csv(train_csv)
    df_val = load_split_csv(val_csv)
    df_train["split"] = "train"
    df_val["split"] = "val"

    df_all = pd.concat([df_train, df_val], ignore_index=True)
    df_all = attach_action_names(df_all, id_to_action)

    # Basic stats
    basic = {
        "num_rows_total": int(len(df_all)),
        "num_rows_train": int(len(df_train)),
        "num_rows_val": int(len(df_val)),
        "num_videos": int(df_all["video_id"].nunique()),
        "num_actions": int(len(id_to_action)),
        "num_action_ids_present": int(df_all["action_id"].nunique()),
        "num_person_ids": int(df_all["person_id"].nunique()),
        "num_instances": int(df_all["instance_id"].nunique()),
    }

    # Per-split counts
    per_split = (
        df_all.groupby(["split"]).size().rename("count").reset_index().to_dict("records")
    )

    # Class distribution
    action_counts = (
        df_all["action_name"].fillna("<unknown>").value_counts().to_dict()
    )
    action_counts_by_split = (
        df_all.groupby(["split", "action_name"]).size().unstack(fill_value=0)
    )

    # Sport distribution (by prefix from action_name)
    sport_counts = df_all["sport"].fillna("<unknown>").value_counts().to_dict()

    # Per video counts
    per_video_counts = df_all.groupby("video_id").size().sort_values(ascending=False)

    # BBox stats for primary and secondary boxes
    bbox_stats_primary = compute_bbox_stats(df_all, prefix="")
    bbox_stats_secondary = compute_bbox_stats(df_all, prefix="x1_2"[:-2])  # same keys computed explicitly below
    # Actually pass explicit prefix for secondary
    bbox_stats_secondary = compute_bbox_stats(df_all.rename(columns={
        "x1_2": "s_x1", "y1_2": "s_y1", "x2_2": "s_x2", "y2_2": "s_y2"
    }), prefix="s_")

    # Anomalies: action ids not in mapping
    unknown_action_id_rows = int((~df_all["action_id"].isin(list(id_to_action.keys()))).sum())

    # Duplicates: identical key tuple rows
    dup_subset = [
        "video_id",
        "frame_id",
        "action_id",
        "person_id",
        "instance_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "x1_2",
        "y1_2",
        "x2_2",
        "y2_2",
    ]
    duplicates = int(df_all.duplicated(subset=dup_subset).sum())

    # Per-sport action breakdown (top 10 per sport)
    per_sport_top = {}
    for sport, sub in df_all.groupby("sport"):
        cnt = sub["action_name"].value_counts().to_dict()
        per_sport_top[sport if pd.notna(sport) else "<unknown>"] = top_k(cnt, k=10)

    # Build summary
    summary = {
        "paths": {
            "annotations_dir": ANNOTATIONS_DIR,
            "results_dir": RESULTS_DIR,
        },
        "schema": {
            "columns": CSV_COLUMNS,
            "coordinates_normalized": True,
            "coordinate_range": "[0, 1] expected",
            "notes": "(x1,y1)-(x2,y2) are primary bbox; (x1_2,y1_2)-(x2_2,y2_2) optional secondary bbox",
        },
        "basic": basic,
        "per_split_counts": per_split,
        "action_map_size": len(id_to_action),
        "action_map_sample": dict(list({i: v["name"] for i, v in id_to_action.items()}.items())[:10]),
        "top_actions": top_k(action_counts, 30),
        "sport_distribution": sport_counts,
        "per_video_top": list(per_video_counts.head(30).items()),
        "bbox_primary": bbox_stats_primary,
        "bbox_secondary": bbox_stats_secondary,
        "anomalies": {
            "unknown_action_id_rows": unknown_action_id_rows,
            "duplicate_rows": duplicates,
        },
    }

    return summary, df_all, id_to_action


def analyze_proposals() -> dict:
    results = {}
    for fname in [
        "sports_dense_proposals_train.pkl",
        "sports_dense_proposals_val.pkl",
        "gt_dense_proposals_train.pkl",
        "gt_dense_proposals_val.pkl",
    ]:
        fpath = os.path.join(ANNOTATIONS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            entry = {
                "path": fpath,
                "type": str(type(data)),
            }
            if isinstance(data, dict):
                entry["num_keys"] = len(data)
                keys = list(data.keys())
                entry["sample_keys"] = keys[:5]
                if keys:
                    sample = data[keys[0]]
                    entry["sample_value_type"] = str(type(sample))
                    entry["sample_value_shape"] = getattr(sample, "shape", None)
                    entry["sample_value_len"] = len(sample) if hasattr(sample, "__len__") else None
            results[fname] = entry
        except Exception as e:
            results[fname] = {"path": fpath, "error": str(e)}
    return results


def to_markdown(summary: dict, id_to_action: dict) -> str:
    lines = []
    lines.append("## 数据概览")
    lines.append(f"- **标注目录**: `{summary['paths']['annotations_dir']}`")
    lines.append(f"- **结果目录**: `{summary['paths']['results_dir']}`")
    lines.append(f"- **样本总数**: {summary['basic']['num_rows_total']}")
    lines.append(f"- **视频数**: {summary['basic']['num_videos']}")
    lines.append("")

    lines.append("## CSV 架构与字段")
    cols = ", ".join(f"`{c}`" for c in summary["schema"]["columns"])
    lines.append(f"- **列**: {cols}")
    lines.append("- **坐标**: 归一化到 [0,1]，一号框 `(x1,y1)-(x2,y2)`，二号框可选 `(_2)`")
    lines.append("")

    lines.append("## 类别与分布")
    lines.append(f"- **类别映射数量**: {summary['action_map_size']}")
    lines.append("- **映射样例**:")
    sample_map = summary.get("action_map_sample", {})
    for k, v in sample_map.items():
        lines.append(f"  - `{k}`: {v}")
    lines.append("")
    lines.append("- **Top-20 类别计数**:")
    for name, cnt in summary.get("top_actions", [])[:20]:
        lines.append(f"  - {name}: {cnt}")
    lines.append("")

    lines.append("## 运动项目分布")
    for sport, cnt in summary.get("sport_distribution", {}).items():
        lines.append(f"- {sport}: {cnt}")
    lines.append("")

    def _bbox_section(title: str, stats: dict):
        lines.append(f"## {title}")
        if not stats:
            lines.append("- 无统计数据")
            return
        for k in ["width", "height", "area"]:
            s = stats.get(k, {})
            if s:
                lines.append(f"- **{k}**: min={s.get('min')}, p25={s.get('p25')}, mean={s.get('mean')}, p75={s.get('p75')}, max={s.get('max')}")
        an = stats.get("anomalies", {})
        if an:
            lines.append("- **异常**:")
            lines.append(f"  - 坐标越界: {an.get('coords_out_of_range', 0)}")
            lines.append(f"  - x1>=x2: {an.get('x1_ge_x2', 0)}")
            lines.append(f"  - y1>=y2: {an.get('y1_ge_y2', 0)}")
            lines.append(f"  - 缺失值: {an.get('missing_values', 0)}")

    _bbox_section("主框尺寸统计", summary.get("bbox_primary", {}))
    lines.append("")
    _bbox_section("副框尺寸统计", summary.get("bbox_secondary", {}))
    lines.append("")

    an = summary.get("anomalies", {})
    lines.append("## 数据质量")
    lines.append(f"- 未在类别映射中的 `action_id` 行数: {an.get('unknown_action_id_rows', 0)}")
    lines.append(f"- 重复标注行数: {an.get('duplicate_rows', 0)}")
    lines.append("")

    lines.append("## 每视频标注 Top-20")
    for vid, cnt in summary.get("per_video_top", [])[:20]:
        lines.append(f"- {vid}: {cnt}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze SportsHHI annotations and generate reports")
    parser.add_argument("--input_dir", type=str, default=None, help="Annotations input directory (overrides default)")
    parser.add_argument("--output_dir", type=str, default=None, help="Results output directory (overrides default)")
    args = parser.parse_args()

    # Override globals if args provided
    global ANNOTATIONS_DIR, RESULTS_DIR
    if args.input_dir:
        ANNOTATIONS_DIR = os.path.abspath(args.input_dir)
    if args.output_dir:
        RESULTS_DIR = os.path.abspath(args.output_dir)

    ensure_results_dir()

    # 解析标注
    summary, df_all, id_to_action = analyze_annotations()

    # 解析 proposals/gt pkl 概览
    proposals = analyze_proposals()
    summary["proposals_overview"] = proposals

    # 写 JSON 总结
    json_out = os.path.join(RESULTS_DIR, "annotations_summary.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 写 Markdown 文档
    md = to_markdown(summary, id_to_action)
    md_out = os.path.join(RESULTS_DIR, "README_annotations.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"JSON 总结: {json_out}")
    print(f"说明文档: {md_out}")


if __name__ == "__main__":
    main()

