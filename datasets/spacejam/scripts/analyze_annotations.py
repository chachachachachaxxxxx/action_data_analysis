#!/usr/bin/env python3
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


ANNOTATION_DIR = "/storage/wangxinxing/code/action_data_analysis/spacejam/annotation"
RESULTS_DIR = "/storage/wangxinxing/code/action_data_analysis/spacejam/results"


def read_labels_mapping(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_split_csv(path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Expect path,label
            if len(row) == 1 and "," in row[0]:
                # Some csv readers may not split due to missing quoting; handle manually
                parts = row[0].rsplit(",", 1)
                if len(parts) == 2:
                    rows.append((parts[0], parts[1]))
                continue
            if len(row) >= 2:
                rows.append((row[0], row[1]))
    return rows


def count_by_label(rows: List[Tuple[str, str]]) -> Counter:
    counter: Counter = Counter()
    for _, label in rows:
        counter[str(label)] += 1
    return counter


def write_distribution_csv(
    out_path: str,
    label_counter: Counter,
    labels_map: Dict[str, str],
):
    total = sum(label_counter.values())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "label_name", "count", "proportion"])
        for label_id, count in sorted(label_counter.items(), key=lambda x: int(x[0])):
            name = labels_map.get(str(label_id), "UNKNOWN")
            prop = count / total if total else 0.0
            writer.writerow([label_id, name, count, f"{prop:.6f}"])


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_summary(
    out_path: str,
    sizes: Dict[str, int],
    uniques: Dict[str, int],
    intersections: Dict[str, int],
    overall_counter: Counter,
    labels_map: Dict[str, str],
):
    total = sizes.get("total", 0)
    def pct(n: int) -> str:
        return f"{(n / total * 100):.2f}%" if total else "0.00%"

    largest = max(overall_counter.values()) if overall_counter else 0
    smallest = min(overall_counter.values()) if overall_counter else 0
    imbalance = (largest / smallest) if smallest else 0.0

    lines: List[str] = []
    lines.append("SpaceJam 标注统计汇总\n")
    lines.append("[数据规模]")
    lines.append(f"- train: {sizes['train']} ({pct(sizes['train'])})")
    lines.append(f"- val: {sizes['val']} ({pct(sizes['val'])})")
    lines.append(f"- test: {sizes['test']} ({pct(sizes['test'])})")
    lines.append(f"- total: {sizes['total']}")
    lines.append("")
    lines.append("[去重检查]")
    lines.append(f"- train 唯一: {uniques['train']} (重复: {sizes['train'] - uniques['train']})")
    lines.append(f"- val 唯一: {uniques['val']} (重复: {sizes['val'] - uniques['val']})")
    lines.append(f"- test 唯一: {uniques['test']} (重复: {sizes['test'] - uniques['test']})")
    lines.append(f"- train∩val: {intersections['train_val']}")
    lines.append(f"- train∩test: {intersections['train_test']}")
    lines.append(f"- val∩test: {intersections['val_test']}")
    lines.append("")
    lines.append("[类别映射]")
    for k in sorted(labels_map.keys(), key=lambda x: int(x)):
        lines.append(f"- {k}: {labels_map[k]}")
    lines.append("")
    lines.append("[整体类别分布]")
    for label_id in sorted(overall_counter.keys(), key=lambda x: int(x)):
        count = overall_counter[label_id]
        name = labels_map.get(str(label_id), "UNKNOWN")
        lines.append(f"- {name} ({label_id}): {count}")
    lines.append("")
    lines.append(f"失衡度（最大/最小）: {imbalance:.2f}x")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ensure_dir(RESULTS_DIR)

    labels_map = read_labels_mapping(os.path.join(ANNOTATION_DIR, "labels_dict_back.json"))

    split_files = {
        "train": os.path.join(ANNOTATION_DIR, "train_back.csv"),
        "val": os.path.join(ANNOTATION_DIR, "val_back.csv"),
        "test": os.path.join(ANNOTATION_DIR, "test_back.csv"),
    }

    split_rows: Dict[str, List[Tuple[str, str]]] = {}
    split_counters: Dict[str, Counter] = {}
    sizes: Dict[str, int] = {}
    uniques: Dict[str, int] = {}

    for split, path in split_files.items():
        rows = read_split_csv(path)
        split_rows[split] = rows
        split_counters[split] = count_by_label(rows)
        sizes[split] = len(rows)
        uniques[split] = len({p for p, _ in rows})

    sizes["total"] = sum(sizes[s] for s in ("train", "val", "test"))

    # write per-split distributions
    for split in ("train", "val", "test"):
        out_csv = os.path.join(RESULTS_DIR, f"class_distribution_{split}.csv")
        write_distribution_csv(out_csv, split_counters[split], labels_map)

    # overall distribution
    overall_counter: Counter = Counter()
    for split in ("train", "val", "test"):
        overall_counter.update(split_counters[split])
    write_distribution_csv(
        os.path.join(RESULTS_DIR, "class_distribution_overall.csv"),
        overall_counter,
        labels_map,
    )

    # intersections across splits
    paths_train = {p for p, _ in split_rows["train"]}
    paths_val = {p for p, _ in split_rows["val"]}
    paths_test = {p for p, _ in split_rows["test"]}

    intersections = {
        "train_val": len(paths_train & paths_val),
        "train_test": len(paths_train & paths_test),
        "val_test": len(paths_val & paths_test),
    }

    # write sizes and checks
    write_json(os.path.join(RESULTS_DIR, "split_sizes.json"), sizes)
    write_json(os.path.join(RESULTS_DIR, "unique_counts.json"), uniques)
    write_json(os.path.join(RESULTS_DIR, "intersections.json"), intersections)
    write_json(os.path.join(RESULTS_DIR, "labels_mapping.json"), labels_map)

    # write summary txt
    write_summary(
        os.path.join(RESULTS_DIR, "summary.txt"),
        sizes,
        uniques,
        intersections,
        overall_counter,
        labels_map,
    )


if __name__ == "__main__":
    main()

