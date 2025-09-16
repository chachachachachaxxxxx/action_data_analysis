#!/usr/bin/env python3
import os
import json
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore


REPO_ROOT = "/storage/wangxinxing/code/action_data_analysis"
PKL_PATH = os.path.join(
    REPO_ROOT,
    "datasets",
    "multisports",
    "annotations_basketball",
    "multisports_basketball.pkl",
)
RESULTS_DIR = os.path.join(
    REPO_ROOT,
    "datasets",
    "multisports",
    "results",
)


def safe_len(obj: Any) -> int:
    try:
        return int(len(obj))  # type: ignore
    except Exception:
        return -1


def summarize_container(obj: Any, max_items: int = 5) -> Dict[str, Any]:
    if isinstance(obj, dict):
        keys = list(obj.keys())[:max_items]
        return {
            "type": "dict",
            "size": safe_len(obj),
            "sample_keys": keys,
            "sample_value_types": {str(k): type(obj[k]).__name__ for k in keys},
        }
    if isinstance(obj, (list, tuple)):
        return {
            "type": type(obj).__name__,
            "size": safe_len(obj),
            "sample_types": [type(x).__name__ for x in obj[:max_items]],
        }
    return {"type": type(obj).__name__}


def pick_example_basketball_video(gttubes: Dict[str, Any]) -> str:
    # Prefer a basketball video id
    for k in gttubes.keys():
        if isinstance(k, str) and k.startswith("basketball/"):
            return k
    # Fallback: first key
    return next(iter(gttubes.keys())) if gttubes else ""


def extract_example_for_video(
    vid: str,
    gttubes: Dict[str, Dict[int, List[Any]]],
    idx_to_label: Dict[int, str],
    max_rows: int = 3,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"video": vid, "exists": False}
    if not vid or vid not in gttubes:
        return out
    entry = gttubes[vid]
    out["exists"] = True
    out["num_label_indices"] = safe_len(entry)
    label_indices = list(entry.keys())
    out["label_indices_sample"] = label_indices[:5]
    if not label_indices:
        return out
    li = label_indices[0]
    tubes = entry.get(li, [])
    out["chosen_label_index"] = int(li)
    out["chosen_label_name"] = idx_to_label.get(int(li), "")
    out["num_tubes_for_label"] = safe_len(tubes)
    if not isinstance(tubes, list) or not tubes:
        return out
    tube0 = tubes[0]
    out["tube0_type"] = type(tube0).__name__
    if isinstance(tube0, np.ndarray):
        out["tube0_shape"] = list(tube0.shape)
        # head rows
        head = min(max_rows, tube0.shape[0])
        out["tube0_head_rows"] = [tube0[i].tolist() for i in range(head)]
        # column ranges to infer bbox columns
        try:
            out["tube0_col_mins"] = tube0.min(axis=0).tolist()
            out["tube0_col_maxs"] = tube0.max(axis=0).tolist()
        except Exception:
            pass
        out["note_bbox_columns"] = "Rows look like [frame, ..., x1, y1, x2, y2] in pixels"
    else:
        out["tube0_summary"] = summarize_container(tube0)
    return out


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(PKL_PATH):
        raise FileNotFoundError(f"Not found: {PKL_PATH}")

    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    # Expect basketball-only structure built from MultiSports
    # keys: labels, train_videos, test_videos, nframes, resolution, gttubes
    labels: List[str] = data.get("labels", [])
    train_videos: List[List[str]] = data.get("train_videos", [[]])
    test_videos: List[List[str]] = data.get("test_videos", [[]])
    nframes: Dict[str, int] = data.get("nframes", {})
    resolution: Dict[str, Tuple[int, int]] = data.get("resolution", {})
    gttubes: Dict[str, Dict[int, List[Any]]] = data.get("gttubes", {})

    idx_to_label = {i: l for i, l in enumerate(labels)}

    schema = {
        "top_level_keys": list(data.keys()) if isinstance(data, dict) else [],
        "labels": {
            "type": type(labels).__name__,
            "size": safe_len(labels),
            "sample": labels[:10],
        },
        "train_videos": {
            "type": type(train_videos).__name__,
            "num_splits": safe_len(train_videos),
            "split0_size": safe_len(train_videos[0]) if train_videos else 0,
            "split0_sample": (train_videos[0][:10] if train_videos and train_videos[0] else []),
        },
        "test_videos": {
            "type": type(test_videos).__name__,
            "num_splits": safe_len(test_videos),
            "split0_size": safe_len(test_videos[0]) if test_videos else 0,
            "split0_sample": (test_videos[0][:10] if test_videos and test_videos[0] else []),
        },
        "nframes": summarize_container(nframes),
        "resolution": summarize_container(resolution),
        "gttubes": summarize_container(gttubes),
    }

    # Example: choose one basketball video and one tube
    vid_example = pick_example_basketball_video(gttubes)
    example = extract_example_for_video(vid_example, gttubes, idx_to_label, max_rows=3)

    # Write results
    schema_path = os.path.join(RESULTS_DIR, "basketball_pkl_schema.json")
    example_path = os.path.join(RESULTS_DIR, "basketball_pkl_example.json")
    md_path = os.path.join(RESULTS_DIR, "README_basketball_pkl.md")

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(example, f, ensure_ascii=False, indent=2)

    # Short human-readable markdown
    lines: List[str] = []
    lines.append("# MultiSports 篮球注释PKL 标准结构与示例")
    lines.append("")
    lines.append("## 顶层结构")
    lines.append("- keys: labels, train_videos, test_videos, nframes, resolution, gttubes")
    lines.append("- labels: List[str]，篮球相关类别名")
    lines.append("- train_videos/test_videos: 形如 [List[str]]，通常只有一个子列表")
    lines.append("- nframes: Dict[video_id -> int]")
    lines.append("- resolution: Dict[video_id -> (H, W) 或 (W, H)]")
    lines.append("- gttubes: Dict[video_id -> Dict[label_index -> List[np.ndarray]]]")
    lines.append("")
    lines.append("## 示例（节选）")
    lines.append(f"- 示例视频: {example.get('video', '')}")
    lines.append(f"- 该视频包含标签索引数: {example.get('num_label_indices', 0)}")
    lines.append(f"- 选中标签: {example.get('chosen_label_index', '')} / {example.get('chosen_label_name', '')}")
    if isinstance(example.get("tube0_shape"), list):
        lines.append(f"- tube[0] 形状: {example['tube0_shape']}")
    if isinstance(example.get("tube0_head_rows"), list):
        lines.append("- tube[0] 前3行:")
        for row in example["tube0_head_rows"]:
            lines.append(f"  - {row}")
    if "note_bbox_columns" in example:
        lines.append(f"- 备注: {example['note_bbox_columns']}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote:")
    print("-", schema_path)
    print("-", example_path)
    print("-", md_path)


if __name__ == "__main__":
    main()


