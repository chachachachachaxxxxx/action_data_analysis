#!/usr/bin/env python3
import os
import json
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore


REPO_ROOT = "/storage/wangxinxing/code/action_data_analysis"
BB_ANN_PATH = os.path.join(
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


def load_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def count_segments_for_tube(tube: np.ndarray, gap_tolerance: int = 1) -> int:
    if getattr(tube, "ndim", None) != 2 or tube.shape[0] == 0:
        return 0
    # frame index assumed in column 0
    try:
        frames = sorted({int(row[0]) for row in tube})
    except Exception:
        return 0
    if not frames:
        return 0
    segments = 1
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > gap_tolerance:
            segments += 1
    return segments


def verify_counts(gap_tolerance: int = 1) -> Dict[str, Any]:
    data = load_pkl(BB_ANN_PATH)
    labels: List[str] = data.get("labels", [])
    gttubes: Dict[str, Dict[int, List[np.ndarray]]] = data.get("gttubes", {})

    idx_to_label = {i: l for i, l in enumerate(labels)}

    orig_total = 0
    split_total = 0
    per_class_orig: Dict[str, int] = {l: 0 for l in labels}
    per_class_split: Dict[str, int] = {l: 0 for l in labels}

    for vid, by_label in gttubes.items():
        for lab_idx, tubes in by_label.items():
            class_name = idx_to_label.get(int(lab_idx), str(lab_idx))
            if not isinstance(tubes, list):
                continue
            for tube in tubes:
                if getattr(tube, "ndim", None) != 2 or tube.shape[0] == 0:
                    continue
                orig_total += 1
                per_class_orig[class_name] = per_class_orig.get(class_name, 0) + 1
                seg = count_segments_for_tube(tube, gap_tolerance=gap_tolerance)
                split_total += seg
                per_class_split[class_name] = per_class_split.get(class_name, 0) + seg

    # diffs
    per_class_diff = {
        c: int(per_class_split.get(c, 0) - per_class_orig.get(c, 0)) for c in labels
    }

    result = {
        "gap_tolerance": int(gap_tolerance),
        "original_tube_total": int(orig_total),
        "split_tube_total": int(split_total),
        "total_diff": int(split_total - orig_total),
        "per_class_original": per_class_orig,
        "per_class_split": per_class_split,
        "per_class_diff": per_class_diff,
    }
    return result


def write_results(obj: Dict[str, Any]) -> Tuple[str, str]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "verify_tube_counts.json")
    md_path = os.path.join(RESULTS_DIR, "README_verify_tube_counts.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    lines: List[str] = []
    lines.append("# MultiSports 篮球 tube 核验（原始 vs 按跳帧拆段）")
    lines.append("")
    lines.append(f"- gap_tolerance: {obj['gap_tolerance']}")
    lines.append(f"- original_tube_total: {obj['original_tube_total']}")
    lines.append(f"- split_tube_total: {obj['split_tube_total']}")
    lines.append(f"- total_diff: {obj['total_diff']}")
    lines.append("")
    # Top increases
    diffs = sorted(obj["per_class_diff"].items(), key=lambda kv: kv[1], reverse=True)
    lines.append("## 按类别的增量（Top 15）")
    for c, d in diffs[:15]:
        if d <= 0:
            continue
        lines.append(f"- {c}: +{d}")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path


def main() -> None:
    res = verify_counts(gap_tolerance=1)
    jp, mp = write_results(res)
    print("Wrote:")
    print("-", jp)
    print("-", mp)


if __name__ == "__main__":
    main()


