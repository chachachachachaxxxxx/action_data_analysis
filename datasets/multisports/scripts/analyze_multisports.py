#!/usr/bin/env python3
import os
import pickle
import json
from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, List, Tuple, Any

import numpy as np


REPO_ROOT = "/storage/wangxinxing/code/action_data_analysis"
ANNOT_PATH = os.path.join(REPO_ROOT, "multisports", "annotations", "multisports_GT.pkl")
OUT_ANN_BASKETBALL_DIR = os.path.join(REPO_ROOT, "multisports", "annotations_basketball")
RESULTS_DIR = os.path.join(REPO_ROOT, "multisports", "results")


def load_annotations(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_label_maps(labels: List[str]) -> Tuple[Dict[int, str], Dict[str, int]]:
    idx_to_label = {i: l for i, l in enumerate(labels)}
    label_to_idx = {l: i for i, l in enumerate(labels)}
    return idx_to_label, label_to_idx


def filter_basketball_videos(videos: List[str]) -> List[str]:
    return [v for v in videos if isinstance(v, str) and v.startswith("basketball/")]


def compute_box_stats(boxes: np.ndarray) -> Tuple[float, float, float, float]:
    # boxes columns: [frame, x1, y1, x2, y2]
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return float(widths.mean()), float(widths.max()), float(widths.min()), float(heights.mean()), float(heights.max()), float(heights.min())


def compute_duration_frames(boxes: np.ndarray) -> int:
    return int(boxes.shape[0])


def analyze_basketball(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    labels: List[str] = data["labels"]
    idx_to_label, _ = build_label_maps(labels)

    train_videos: List[str] = data["train_videos"][0]
    test_videos: List[str] = data["test_videos"][0]
    train_basket = filter_basketball_videos(train_videos)
    test_basket = filter_basketball_videos(test_videos)

    nframes: Dict[str, int] = data["nframes"]
    resolution: Dict[str, Tuple[int, int]] = data["resolution"]
    gttubes: Dict[str, Dict[int, List[np.ndarray]]] = data["gttubes"]

    # collect basketball labels
    basketball_labels = [l for l in labels if l.startswith("basketball ")]
    basketball_idx_map = {i: l for i, l in enumerate(labels) if l in basketball_labels}
    idx_old_to_new: Dict[int, int] = {}
    new_labels: List[str] = []
    for old_idx, lab in basketball_idx_map.items():
        idx_old_to_new[old_idx] = len(new_labels)
        new_labels.append(lab)

    # aggregate stats
    train_frames_total = sum(nframes[v] for v in train_basket)
    test_frames_total = sum(nframes[v] for v in test_basket)

    train_res_counter = Counter([resolution[v] for v in train_basket])
    test_res_counter = Counter([resolution[v] for v in test_basket])

    per_class_instance_counts = Counter()
    per_class_durations: Dict[str, List[int]] = defaultdict(list)
    per_class_box_widths: Dict[str, List[float]] = defaultdict(list)
    per_class_box_heights: Dict[str, List[float]] = defaultdict(list)

    global_widths: List[float] = []
    global_heights: List[float] = []
    global_durations: List[int] = []

    def handle_tube_array(arr: np.ndarray, class_name: str):
        # arr shape (N,5): [frame, y1, x1, y2, x2] or [frame, x1, y1, x2, y2]
        # From inspection: columns look like [frame, y1, x1, y2, x2] given mins/maxs; derive robustly
        # We detect by checking which order gives positive widths/heights
        col1, col2, col3, col4 = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        # candidate A: x1=col1,y1=col2,x2=col3,y2=col4
        widths_a = col3 - col1
        heights_a = col4 - col2
        valid_a = (widths_a > 0).mean() > 0.9 and (heights_a > 0).mean() > 0.9
        # candidate B: x1=col2,y1=col1,x2=col4,y2=col3
        widths_b = col4 - col2
        heights_b = col3 - col1
        valid_b = (widths_b > 0).mean() > 0.9 and (heights_b > 0).mean() > 0.9
        if valid_b and (not valid_a or widths_b.mean() * heights_b.mean() > widths_a.mean() * heights_a.mean()):
            widths = widths_b
            heights = heights_b
        else:
            widths = widths_a
            heights = heights_a

        per_class_box_widths[class_name].extend(widths.tolist())
        per_class_box_heights[class_name].extend(heights.tolist())
        global_widths.extend(widths.tolist())
        global_heights.extend(heights.tolist())
        duration = compute_duration_frames(arr)
        per_class_durations[class_name].append(duration)
        global_durations.append(duration)

    # iterate over basketball videos
    for vid in train_basket + test_basket:
        if vid not in gttubes:
            continue
        vid_tubes = gttubes[vid]
        for old_label_idx, tube_list in vid_tubes.items():
            if old_label_idx not in idx_old_to_new:
                continue
            class_name = idx_to_label[old_label_idx]
            for arr in tube_list:
                if not isinstance(arr, np.ndarray) or arr.size == 0:
                    continue
                per_class_instance_counts[class_name] += 1
                handle_tube_array(arr, class_name)

    # summarize per-class stats
    per_class_duration_avg = {c: round(float(mean(v)), 2) if len(v) else 0.0 for c, v in per_class_durations.items()}
    per_class_total_boxes = {c: int(sum(v)) for c, v in per_class_durations.items()}
    per_class_box_stats = {}
    for c in new_labels:
        widths = per_class_box_widths.get(c, [])
        heights = per_class_box_heights.get(c, [])
        if widths and heights:
            w_avg, w_max, w_min = float(mean(widths)), float(max(widths)), float(min(widths))
            h_avg, h_max, h_min = float(mean(heights)), float(max(heights)), float(min(heights))
        else:
            w_avg = w_max = w_min = h_avg = h_max = h_min = 0.0
        per_class_box_stats[c] = {
            "width": {"avg": round(w_avg, 2), "max": round(w_max, 2), "min": round(w_min, 2)},
            "height": {"avg": round(h_avg, 2), "max": round(h_max, 2), "min": round(h_min, 2)},
        }

    global_box_stats = {
        "width": {
            "avg": round(float(mean(global_widths)), 2) if global_widths else 0.0,
            "max": round(float(max(global_widths)), 2) if global_widths else 0.0,
            "min": round(float(min(global_widths)), 2) if global_widths else 0.0,
        },
        "height": {
            "avg": round(float(mean(global_heights)), 2) if global_heights else 0.0,
            "max": round(float(max(global_heights)), 2) if global_heights else 0.0,
            "min": round(float(min(global_heights)), 2) if global_heights else 0.0,
        },
        "duration_frames": {
            "avg": round(float(mean(global_durations)), 2) if global_durations else 0.0,
            "max": int(max(global_durations)) if global_durations else 0,
            "min": int(min(global_durations)) if global_durations else 0,
        },
    }

    summary = {
        "basketball": {
            "num_classes": len(new_labels),
            "class_names": new_labels,
            "num_train_videos": len(train_basket),
            "num_test_videos": len(test_basket),
            "total_train_frames": int(train_frames_total),
            "total_test_frames": int(test_frames_total),
            "train_resolutions": {str(k): v for k, v in train_res_counter.items()},
            "test_resolutions": {str(k): v for k, v in test_res_counter.items()},
            "per_class_instance_counts": {c: int(per_class_instance_counts.get(c, 0)) for c in new_labels},
            "per_class_duration_avg": per_class_duration_avg,
            "per_class_box_stats": per_class_box_stats,
            "per_class_total_boxes": per_class_total_boxes,
            "global_box_stats": global_box_stats,
        }
    }

    # Build basketball-only annotation pkl structure (gttubes remapped)
    bb_gttubes = {}
    for vid in train_basket + test_basket:
        if vid not in gttubes:
            continue
        entry = {}
        for old_idx, tube_list in gttubes[vid].items():
            if old_idx not in idx_old_to_new:
                continue
            new_idx = idx_old_to_new[old_idx]
            entry[new_idx] = tube_list
        if entry:
            bb_gttubes[vid] = entry

    basketball_ann = {
        "labels": new_labels,
        "train_videos": [train_basket],
        "test_videos": [test_basket],
        "nframes": {v: nframes[v] for v in train_basket + test_basket},
        "resolution": {v: resolution[v] for v in train_basket + test_basket},
        "gttubes": bb_gttubes,
    }

    return basketball_ann, summary


def write_outputs(basketball_ann: Dict[str, Any], summary: Dict[str, Any]):
    os.makedirs(OUT_ANN_BASKETBALL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    bb_pkl_path = os.path.join(OUT_ANN_BASKETBALL_DIR, "multisports_basketball.pkl")
    with open(bb_pkl_path, "wb") as f:
        pickle.dump(basketball_ann, f)

    # structured files
    with open(os.path.join(RESULTS_DIR, "labels_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"basketball_labels": basketball_ann["labels"]}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(RESULTS_DIR, "split_sizes.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train": len(basketball_ann["train_videos"][0]),
            "test": len(basketball_ann["test_videos"][0])
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # boxes per class (standalone)
    with open(os.path.join(RESULTS_DIR, "boxes_per_class.json"), "w", encoding="utf-8") as f:
        json.dump(summary["basketball"]["per_class_total_boxes"], f, ensure_ascii=False, indent=2)

    # text outputs similar to original.txt
    txt_lines: List[str] = []
    bsum = summary["basketball"]
    txt_lines.append(f"篮球类别数量: {bsum['num_classes']}")
    txt_lines.append(f"篮球类别名称: {bsum['class_names']}")
    txt_lines.append("")
    txt_lines.append(f"篮球训练视频数量: {bsum['num_train_videos']}")
    txt_lines.append(f"篮球测试视频数量: {bsum['num_test_videos']}")
    txt_lines.append("")
    txt_lines.append(f"篮球训练视频总帧数: {bsum['total_train_frames']}")
    txt_lines.append(f"篮球测试视频总帧数: {bsum['total_test_frames']}")
    txt_lines.append("")
    txt_lines.append(f"篮球训练视频分辨率分布: Counter({Counter(basketball_ann['train_videos'][0] and [basketball_ann['resolution'][v] for v in basketball_ann['train_videos'][0]] )})")
    txt_lines.append(f"篮球测试视频分辨率分布: Counter({Counter(basketball_ann['test_videos'][0] and [basketball_ann['resolution'][v] for v in basketball_ann['test_videos'][0]] )})")
    txt_lines.append("")
    txt_lines.append("篮球各类别动作实例数统计：")
    for c in bsum['class_names']:
        txt_lines.append(f"{c}: {bsum['per_class_instance_counts'].get(c, 0)} 个实例")
    txt_lines.append("")
    txt_lines.append("篮球各类别检测框总数：")
    for c in bsum['class_names']:
        txt_lines.append(f"{c}: {bsum['per_class_total_boxes'].get(c, 0)} 个检测框")
    txt_lines.append("")
    txt_lines.append("篮球各类别动作实例平均持续帧数：")
    for c in bsum['class_names']:
        txt_lines.append(f"{c}: {bsum['per_class_duration_avg'].get(c, 0.0)} 帧")
    txt_lines.append("")
    txt_lines.append(f"篮球动作实例总数: {sum(bsum['per_class_instance_counts'].values())}")
    txt_lines.append("")
    txt_lines.append("篮球各类别空间检测框平均/最大/最小宽度和高度：")
    for c in bsum['class_names']:
        st = bsum['per_class_box_stats'][c]
        txt_lines.append(f"{c}: 宽度= 平均{st['width']['avg']}, 最大{st['width']['max']}, 最小{st['width']['min']}；高度= 平均{st['height']['avg']}, 最大{st['height']['max']}, 最小{st['height']['min']}")
    txt_lines.append("")
    txt_lines.append(f"所有tube检测框宽度: 平均{bsum['global_box_stats']['width']['avg']}, 最大{bsum['global_box_stats']['width']['max']}, 最小{bsum['global_box_stats']['width']['min']}")
    txt_lines.append(f"所有tube检测框高度: 平均{bsum['global_box_stats']['height']['avg']}, 最大{bsum['global_box_stats']['height']['max']}, 最小{bsum['global_box_stats']['height']['min']}")
    txt_lines.append(f"所有tube持续帧数: 平均{bsum['global_box_stats']['duration_frames']['avg']}, 最大{bsum['global_box_stats']['duration_frames']['max']}, 最小{bsum['global_box_stats']['duration_frames']['min']}")

    with open(os.path.join(RESULTS_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")


def main():
    data = load_annotations(ANNOT_PATH)
    basketball_ann, summary = analyze_basketball(data)
    write_outputs(basketball_ann, summary)
    print("Done. Outputs written to:")
    print("-", os.path.join(OUT_ANN_BASKETBALL_DIR, "multisports_basketball.pkl"))
    print("-", RESULTS_DIR)


if __name__ == "__main__":
    main()

