#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
from typing import Set, List, Dict, Tuple


def discover_std_samples(std_path: str) -> List[str]:
    """发现 std 数据集的样例列表（目录名），等价于 <std_root>/videos/*。

    - std_path 可传 std 根目录（包含 videos/）或直接传 videos 目录本身。
    - 返回样例目录名（不含路径）。
    """
    if not os.path.isdir(std_path):
        raise FileNotFoundError(f"路径不存在: {std_path}")

    # 若传入的不是 videos 目录，则尝试拼接 videos/
    videos_dir = std_path
    if os.path.basename(os.path.normpath(std_path)) != "videos":
        candidate = os.path.join(std_path, "videos")
        if os.path.isdir(candidate):
            videos_dir = candidate
        else:
            # 允许用户直接传 videos/* 的上层也行，但若没有 videos 目录则视为错误
            raise FileNotFoundError(
                f"未找到 videos 目录。请传入 std 根目录（包含 videos/）或 videos 目录本身: {std_path}"
            )

    samples: List[str] = []
    with os.scandir(videos_dir) as it:
        for entry in it:
            if entry.is_dir():
                samples.append(entry.name)
    samples.sort()
    return samples


essential_columns = ("id", "label_name")

def read_samples_with_non_tactical(pred_csv_path: str, non_tactical_label: str) -> Set[str]:
    """读取 predictions.csv，返回至少有一张图片被判为 `non_tactical_label` 的样例集合。

    - 通过 `id` 列提取样例名：取 `id` 的第 1 段（按 '/' 分割）。
    - 仅当 `label_name` 列精确等于 `non_tactical_label` 时计入。
    """
    if not os.path.isfile(pred_csv_path):
        raise FileNotFoundError(f"predictions.csv 不存在: {pred_csv_path}")

    samples_with_non_tactical: Set[str] = set()

    with open(pred_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in essential_columns if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                f"CSV 缺少必要列 {missing_cols}，需要列: {list(essential_columns)}；实际列: {reader.fieldnames}"
            )
        for row in reader:
            image_id = row.get("id", "").strip()
            label_name = row.get("label_name", "").strip()
            if not image_id:
                continue
            # 样例名是路径的第一段，例如：v_xxx/000001.jpg -> v_xxx
            sample = image_id.split("/")[0]
            if label_name == non_tactical_label:
                samples_with_non_tactical.add(sample)

    return samples_with_non_tactical


def compute_non_tactical_stats(pred_csv_path: str, non_tactical_label: str) -> Dict[str, Tuple[int, int]]:
    """统计每个样例（视频）的总帧数与非战术帧数。

    返回 Dict[sample -> (total_frames, non_tactical_frames)]。
    """
    if not os.path.isfile(pred_csv_path):
        raise FileNotFoundError(f"predictions.csv 不存在: {pred_csv_path}")

    totals: Dict[str, int] = {}
    non_counts: Dict[str, int] = {}

    with open(pred_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in essential_columns if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                f"CSV 缺少必要列 {missing_cols}，需要列: {list(essential_columns)}；实际列: {reader.fieldnames}"
            )
        for row in reader:
            image_id = row.get("id", "").strip()
            if not image_id:
                continue
            sample = image_id.split("/")[0]
            totals[sample] = totals.get(sample, 0) + 1
            if row.get("label_name", "").strip() == non_tactical_label:
                non_counts[sample] = non_counts.get(sample, 0) + 1

    stats: Dict[str, Tuple[int, int]] = {}
    for sample, total in totals.items():
        non_c = non_counts.get(sample, 0)
        stats[sample] = (total, non_c)
    return stats


def _resolve_videos_dir(std_path: str) -> str:
    """解析出 videos 目录绝对路径。"""
    if not os.path.isdir(std_path):
        raise FileNotFoundError(f"路径不存在: {std_path}")
    if os.path.basename(os.path.normpath(std_path)) == "videos":
        return std_path
    candidate = os.path.join(std_path, "videos")
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"未找到 videos 目录。请传入 std 根目录（包含 videos/）或 videos 目录本身: {std_path}"
    )


def count_total_frames_by_sample(std_path: str, image_exts: Tuple[str, ...]) -> Dict[str, int]:
    """统计每个样例(目录)下的图片帧总数（按扩展名匹配）。

    - image_exts: 元组，如 ('.jpg', '.jpeg', '.png')，大小写不敏感。
    - 只统计样例目录下的文件（不递归），忽略 JSON。
    """
    videos_dir = _resolve_videos_dir(std_path)
    exts = tuple(e.lower() for e in image_exts)
    totals: Dict[str, int] = {}
    with os.scandir(videos_dir) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            sample = entry.name
            cnt = 0
            try:
                with os.scandir(entry.path) as it2:
                    for f in it2:
                        if f.is_file():
                            _, ext = os.path.splitext(f.name)
                            if ext.lower() in exts:
                                cnt += 1
            except FileNotFoundError:
                cnt = 0
            totals[sample] = cnt
    return totals


def write_list(lines: List[str], out_txt: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_txt)), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(f"{x}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="找出 std 样例中没有任何'非战术视角'图片的样例，并输出为 txt。"
    )
    parser.add_argument(
        "std_path",
        help="std 根目录（包含 videos/）或直接传 videos 目录",
    )
    parser.add_argument(
        "--pred-csv",
        dest="pred_csv",
        default="/storage/wangxinxing/code/action_data_analysis/temp2/predictions.csv",
        help="预测结果 CSV 路径（包含列 id,label_name），默认指向 temp2/predictions.csv",
    )
    parser.add_argument(
        "--label",
        dest="non_tactical_label",
        default="非战术视角",
        help="非战术视角的类别名（与 CSV 的 label_name 对应），默认: 非战术视角",
    )
    parser.add_argument(
        "--out",
        dest="out_txt",
        default="/storage/wangxinxing/code/action_data_analysis/temp2/samples_without_non_tactical.txt",
        help="输出 txt 路径（每行一个样例名）",
    )
    parser.add_argument(
        "--stats-csv",
        dest="stats_csv",
        default="/storage/wangxinxing/code/action_data_analysis/temp2/non_tactical_stats.csv",
        help="输出每个样例的非战术帧计数与比例的 CSV 路径",
    )
    parser.add_argument(
        "--img-exts",
        dest="img_exts",
        default="jpg,jpeg,png",
        help="计作帧图片的扩展名（逗号分隔, 不含点）。默认: jpg,jpeg,png",
    )

    args = parser.parse_args()

    std_samples = discover_std_samples(args.std_path)
    samples_with_non = read_samples_with_non_tactical(args.pred_csv, args.non_tactical_label)

    # std 中没有任何 '非战术视角' 的样例 = std_samples - samples_with_non
    missing = [s for s in std_samples if s not in samples_with_non]

    write_list(missing, args.out_txt)

    # 统计每个样例的总帧（从 std 实际图片数）与非战术帧数（从 predictions.csv）
    image_exts = tuple("." + e.strip().lower() for e in args.img_exts.split(",") if e.strip())
    std_totals = count_total_frames_by_sample(args.std_path, image_exts)
    stats = compute_non_tactical_stats(args.pred_csv, args.non_tactical_label)
    rows = []
    for s in std_samples:
        total = std_totals.get(s, 0)
        _, non_cnt = stats.get(s, (0, 0))
        ratio = (non_cnt / total) if total > 0 else 0.0
        rows.append((s, total, non_cnt, ratio))
    rows.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(os.path.abspath(args.stats_csv)), exist_ok=True)
    with open(args.stats_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "total_frames", "non_tactical_frames", "non_tactical_ratio"])
        for s, total, non_cnt, ratio in rows:
            writer.writerow([s, total, non_cnt, f"{ratio:.6f}"])

    print(f"总样例数: {len(std_samples)}")
    print(f"含有'{args.non_tactical_label}'的样例数: {len(samples_with_non)}")
    print(f"没有'{args.non_tactical_label}'的样例数: {len(missing)}")
    print(f"已写入: {args.out_txt}")
    print(f"统计 CSV 已写入: {args.stats_csv}")

    # 汇总数据集层面总帧与非战术帧
    dataset_total = sum(std_totals.get(s, 0) for s in std_samples)
    dataset_non = 0
    for s in std_samples:
        _, non_cnt = stats.get(s, (0, 0))
        dataset_non += non_cnt
    print(f"数据集总帧数: {dataset_total}")
    print(f"数据集非战术帧数: {dataset_non}")
    if dataset_total > 0:
        print(f"数据集非战术帧比例: {dataset_non / dataset_total:.6f}")


if __name__ == "__main__":
    main()
