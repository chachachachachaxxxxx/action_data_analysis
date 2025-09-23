from __future__ import annotations

import argparse
import sys
from typing import List
from dataclasses import asdict

from action_data_analysis.analyze.export import (
  export_samples_with_context,
  merge_flatten_datasets,
  _discover_leaf_folders,
  _discover_std_sample_folders,
  merge_std_samples_into_single,
)
from action_data_analysis.analyze.visual import visualize_samples_with_context
from action_data_analysis.analyze.stats import (
  compute_labelme_folder_stats,
  compute_aggregate_stats,
  render_stats_markdown,
)
from action_data_analysis.analyze.tube_lengths import (
  compute_tube_lengths_for_labelme_dirs,
  compute_tube_lengths_for_finesports,
  compute_tube_lengths_for_multisports,
  save_results_json_csv,
  collect_overall_tube_lengths_labelme_dirs,
  bin_tube_lengths,
  plot_length_bins,
  compute_time_bins_for_labelme_dirs,
  save_time_bins_json_csv,
)
from action_data_analysis.analyze.overlap import (
  compute_folder_overlaps,
  compute_aggregate_overlaps,
  render_overlaps_markdown,
)
from action_data_analysis.convert.std_converter import convert_std_dataset
from action_data_analysis.convert.std_subset import create_std_subset
from action_data_analysis.convert.split_gt_csv import split_gt_csv


def _add_export_samples_parser(subparsers) -> None:
  p = subparsers.add_parser("export-samples", help="导出每类样例及上下文帧到结构化目录")
  p.add_argument("dirs", nargs="+", help="包含 LabelMe JSON 的目录或其上层目录")
  p.add_argument("--out", required=True, dest="out_dir")
  p.add_argument("--dataset-name", required=True)
  p.add_argument("--per-class", type=int, default=3)
  p.add_argument("--context", type=int, default=6)


def _add_merge_flatten_parser(subparsers) -> None:
  p = subparsers.add_parser("merge-flatten", help="合并多个数据集导出目录为单一平铺目录")
  p.add_argument("roots", nargs="+", help="各数据集导出根目录，如 output/json/FineSports")
  p.add_argument("--out", required=True, dest="out_dir")


def _add_visualize_samples_parser(subparsers) -> None:
  p = subparsers.add_parser("visualize-samples", help="按类别采样并可视化上下文帧")
  p.add_argument("dirs", nargs="+", help="包含 LabelMe JSON 的目录或其上层目录")
  p.add_argument("--out", required=True, dest="out_dir")
  p.add_argument("--per-class", type=int, default=3)
  p.add_argument("--context", type=int, default=25)


def _add_analyze_folder_parser(subparsers) -> None:
  p = subparsers.add_parser("analyze-folder", help="统计单个目录的标注情况并输出 JSON/Markdown（std: 传入样例文件夹路径）")
  p.add_argument("folder")
  p.add_argument("--out", dest="out_dir", default="")


def _add_analyze_dirs_parser(subparsers) -> None:
  p = subparsers.add_parser("analyze-dirs", help="统计多个 std 样例目录（从 <root>/videos/* 自动发现）")
  p.add_argument("dirs", nargs="+", help="std 根目录、videos 目录、或 videos/* 样例目录")
  p.add_argument("--out", required=True, dest="out_dir")


def _add_analyze_overlaps_parser(subparsers) -> None:
  p = subparsers.add_parser("analyze-overlaps", help="统计检测框重合（IoU）分布（std）")
  p.add_argument("dirs", nargs="+", help="std 根目录、videos 目录、或 videos/* 样例目录")
  p.add_argument("--out", required=True, dest="out_dir")
  p.add_argument("--thresholds", nargs="*", type=float, default=[0.1, 0.3, 0.5], help="IoU 阈值列表，如 0.1 0.5 0.7")


def _add_analyze_tube_lengths_parser(subparsers) -> None:
  p = subparsers.add_parser("analyze-tube-lengths", help="统计各数据集时空管长度分布（按动作）")
  p.add_argument("dataset", choices=["sportshhi", "multisports", "finesports", "labelme"], help="数据集或原始格式")
  p.add_argument("inputs", nargs="+", help="输入：dataset=labelme 时为包含 LabelMe JSON 的目录；finesports/multisports 为 PKL；sportshhi 为 CSV 标注目录")
  p.add_argument("--out", required=True, dest="out_dir", help="输出目录")


def _add_plot_tube_lengths_parser(subparsers) -> None:
  p = subparsers.add_parser("plot-tube-lengths", help="对 LabelMe 目录绘制时空管长度分箱计数图（std）")
  p.add_argument("dirs", nargs="+", help="std 根目录、videos 目录、或 videos/* 样例目录")
  p.add_argument("--out", required=True, dest="out_png", help="输出 PNG 文件路径")


def _add_time_bins_parser(subparsers) -> None:
  p = subparsers.add_parser("time-bins", help="基于秒的时长分箱计数（<=0.5, 0.5-1, 1+）（std）")
  p.add_argument("dirs", nargs="+", help="std 根目录、videos 目录、或 videos/* 样例目录")
  p.add_argument("--spf", required=True, type=float, help="每帧秒数 seconds-per-frame")
  p.add_argument("--out", required=True, dest="out_dir", help="输出目录")


def _add_convert_std_parser(subparsers) -> None:
  p = subparsers.add_parser("convert-std", help="将 std 数据集按标签映射转换为 converted 目录，并清理空样例")
  p.add_argument("std_root", help="std 根目录（包含 videos 与可选 stats）")
  p.add_argument("mapping_csv", help="包含列 label,label2 的映射 CSV")
  p.add_argument("--out", dest="out_root", default="", help="输出根目录（默认：<std_root>_converted）")
  p.add_argument("--no-copy-images", action="store_true", help="不复制图片，仅转换 JSON")


def _add_subset_std_parser(subparsers) -> None:
  p = subparsers.add_parser("subset-std", help="从 std 数据集抽取代表性子集（每类至少 N 个样例）")
  p.add_argument("dirs", nargs="+", help="std 根目录、videos 目录、或 videos/* 样例目录")
  p.add_argument("--out", dest="out_dir", default="", help="输出根目录（默认：<std_root>_subset）")
  p.add_argument("--per-class", type=int, default=3, help="每类至少样例数，默认 3")


def main() -> None:
  parser = argparse.ArgumentParser(
    prog="action-data-analysis",
    description="动作识别数据集分析 · CLI",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  _add_export_samples_parser(subparsers)
  _add_merge_flatten_parser(subparsers)
  def _add_merge_into_single_parser(subparsers) -> None:
    p = subparsers.add_parser("merge-into-single", help="将多个 std 样例合并为单一样例")
    p.add_argument("dirs", nargs="+", help="std 根目录、videos 目录、或 videos/* 样例目录")
    p.add_argument("--out", dest="out_root", default="", help="输出 std 根目录（默认：<std_root>_merged_single）")
    p.add_argument("--name", dest="sample_name", default="merged_all", help="合并后的样例名（默认 merged_all）")
  _add_merge_into_single_parser(subparsers)
  _add_visualize_samples_parser(subparsers)
  _add_analyze_folder_parser(subparsers)
  _add_analyze_dirs_parser(subparsers)
  _add_analyze_overlaps_parser(subparsers)
  _add_analyze_tube_lengths_parser(subparsers)
  _add_plot_tube_lengths_parser(subparsers)
  _add_time_bins_parser(subparsers)
  _add_convert_std_parser(subparsers)
  # 新增：导出 tube 视频（std）
  def _add_export_tubes_parser(subparsers) -> None:
    p = subparsers.add_parser("export-tubes", help="从 std 目录导出每条 tube 的视频 (25fps, 224x224)")
    p.add_argument("dirs", nargs="+", help="std 根目录、videos 目录或 videos/* 样例目录")
    p.add_argument("--out", required=True, dest="out_dir", help="输出根目录")
    p.add_argument("--strategy", choices=["letterbox", "square"], default="letterbox", help="导出策略")
    p.add_argument("--fps", type=int, default=25, help="输出帧率 (默认25)")
    p.add_argument("--size", type=int, default=224, help="输出分辨率，边长，默认224")
    p.add_argument("--min-len", type=int, default=1, help="最小 tube 帧数过滤")
    p.add_argument("--labels", dest="labels_json", default="", help="labels_dict.json 的路径（可选）。若不提供，将尝试使用 <out>/annotations/labels_dict.json")
  _add_export_tubes_parser(subparsers)
  _add_subset_std_parser(subparsers)
  # 新增：按 tube 拆分样例，生成新的 std 数据集
  def _add_split_by_tube_parser(subparsers) -> None:
    p = subparsers.add_parser("split-by-tube", help="将 std 样例按时空管拆分为新的 std 数据集")
    p.add_argument("dirs", nargs="+", help="std 根目录、videos 目录或 videos/* 样例目录")
    p.add_argument("--out", dest="out_dir", default="", help="输出根目录（默认：<std_root>_by_tube）")
    p.add_argument("--min-len", type=int, default=1, help="最小 tube 帧数过滤")
  _add_split_by_tube_parser(subparsers)
  # 新增：按 fps 窗口切分 tube，生成新的 std 数据集，并记录缺失标注掩码
  def _add_split_by_tube_fpswin_parser(subparsers) -> None:
    p = subparsers.add_parser("split-by-tube-fpswin", help="按 fps 窗口切分 tube（win=fps, stride=fps//2），缺失帧用最近标注补全并写 annotations/mask.csv")
    p.add_argument("dirs", nargs="+", help="std 根目录、videos 目录或 videos/* 样例目录")
    p.add_argument("--out", dest="out_dir", default="", help="输出根目录（默认：<std_root>_by_tube_fpswin）")
    p.add_argument("--fps", type=int, required=True, help="窗口长度（帧）= fps，步长= fps//2")
    p.add_argument("--min-len", type=int, default=1, help="最小 tube 帧数过滤")
  _add_split_by_tube_fpswin_parser(subparsers)
  # 新增：按比例切分 gt.csv
  def _add_split_gt_csv_parser(subparsers) -> None:
    p = subparsers.add_parser("split-gt-csv", help="按比例切分 gt.csv，去表头并删除 frames 列，视频路径加前缀")
    p.add_argument("gt_csv", help="输入 gt.csv 路径")
    p.add_argument("--out", dest="out_dir", default="", help="输出目录（默认与 gt.csv 同目录）")
    p.add_argument("--ratios", nargs=3, type=float, default=[8,1,1], help="切分比例 train val test，如 8 1 1")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--video-col", dest="video_col", default="", help="视频列名（留空自动识别）")
    p.add_argument("--prefix", dest="prefix", default="videos/", help="为视频路径添加的前缀（默认 videos/）")
  _add_split_gt_csv_parser(subparsers)
  # 新增：std <-> pkl 转换
  def _add_std_pkl_parsers(subparsers) -> None:
    p1 = subparsers.add_parser("std-to-pkl", help="从 std 数据集生成标准 PKL（annotations/gt.pkl）")
    p1.add_argument("std_root", help="std 根目录（包含 videos 与可选 stats）")
    p1.add_argument("--out", dest="out_root", default="", help="输出根目录（默认：std_root）")
    p1.add_argument("--name", dest="name", default="gt.pkl", help="输出 PKL 文件名（默认 gt.pkl）")

    p2 = subparsers.add_parser("pkl-to-std", help="从标准 PKL 重建 std（LabelMe JSON）")
    p2.add_argument("pkl_path", help="标准 PKL 路径（包含 version/labels/videos/nframes/resolution/gttubes）")
    p2.add_argument("videos_root", help="包含 <video_id>/ 帧图像的根目录（只读）")
    p2.add_argument("--out", dest="out_root", default="", help="输出 std 根目录（默认：与 PKL 同级 _std）")
    p2.add_argument("--inplace", action="store_true", help="就地在 videos_root/<video_id> 写 JSON，不复制图片")
    p2.add_argument("--copy-images", action="store_true", help="当非 inplace 时复制图片到输出目录")
  _add_std_pkl_parsers(subparsers)

  args = parser.parse_args()

  if args.command == "export-samples":
    leafs = _discover_leaf_folders(args.dirs)
    export_samples_with_context(
      folders=leafs,
      output_root=args.out_dir,
      dataset_name=args.dataset_name,
      per_class=int(args.per_class),
      context_frames=int(args.context),
    )
    return

  if args.command == "merge-flatten":
    merge_flatten_datasets(args.roots, args.out_dir)
    return

  if args.command == "merge-into-single":
    summary = merge_std_samples_into_single(
      inputs=list(args.dirs),
      out_root=(args.out_root if args.out_root else None),
      merged_sample_name=str(args.sample_name or "merged_all"),
    )
    import json as _json
    print(_json.dumps(summary, ensure_ascii=False, indent=2))
    return

  if args.command == "visualize-samples":
    leafs = _discover_leaf_folders(args.dirs)
    visualize_samples_with_context(
      folders=leafs,
      output_dir=args.out_dir,
      per_class=int(args.per_class),
      context_frames=int(args.context),
    )
    return

  if args.command == "analyze-folder":
    stats = compute_labelme_folder_stats(args.folder)
    if args.out_dir:
      import os, json
      os.makedirs(args.out_dir, exist_ok=True)
      with open(os.path.join(args.out_dir, 'stats.json'), 'w', encoding='utf-8') as f:
        import json as _json
        _json.dump(stats, f, ensure_ascii=False, indent=2)
      with open(os.path.join(args.out_dir, 'README_annotations.md'), 'w', encoding='utf-8') as f:
        f.write(render_stats_markdown(stats))
    else:
      import json as _json
      print(_json.dumps(stats, ensure_ascii=False, indent=2))
    return

  if args.command == "analyze-dirs":
    leafs = _discover_std_sample_folders(args.dirs)
    stats = compute_aggregate_stats(leafs)
    import os, json
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'stats.json'), 'w', encoding='utf-8') as f:
      import json as _json
      _json.dump(stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, 'README_annotations.md'), 'w', encoding='utf-8') as f:
      f.write(render_stats_markdown(stats))
    return

  if args.command == "analyze-overlaps":
    leafs = _discover_std_sample_folders(args.dirs)
    stats = compute_aggregate_overlaps(leafs, thresholds=list(args.thresholds))
    import os, json
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'overlaps.json'), 'w', encoding='utf-8') as f:
      import json as _json
      _json.dump(stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, 'README_overlaps.md'), 'w', encoding='utf-8') as f:
      f.write(render_overlaps_markdown(stats))
    return

  if args.command == "analyze-tube-lengths":
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset == "labelme":
      leafs = _discover_std_sample_folders(args.inputs)
      result = compute_tube_lengths_for_labelme_dirs(leafs)
      save_results_json_csv(result, args.out_dir, prefix="labelme")
      return
    # PKL-based datasets
    pkl_path = args.inputs[0]
    if args.dataset == "finesports":
      result = compute_tube_lengths_for_finesports(pkl_path)
      save_results_json_csv(result, args.out_dir, prefix="finesports")
      return
    if args.dataset == "multisports":
      result = compute_tube_lengths_for_multisports(pkl_path)
      save_results_json_csv(result, args.out_dir, prefix="multisports")
      return
    if args.dataset == "sportshhi":
      from action_data_analysis.analyze.tube_lengths import compute_tube_lengths_for_sportshhi
      result = compute_tube_lengths_for_sportshhi(args.inputs[0])
      save_results_json_csv(result, args.out_dir, prefix="sportshhi")
      return

  if args.command == "plot-tube-lengths":
    leafs = _discover_std_sample_folders(args.dirs)
    lengths = collect_overall_tube_lengths_labelme_dirs(leafs)
    counts = bin_tube_lengths(lengths)
    title = "Tube Length Counts"
    plot_length_bins(counts, title, args.out_png)
    return

  if args.command == "time-bins":
    leafs = _discover_std_sample_folders(args.dirs)
    result = compute_time_bins_for_labelme_dirs(leafs, seconds_per_frame=float(args.spf))
    save_time_bins_json_csv(result, args.out_dir, prefix="time")
    return
    return

  if args.command == "convert-std":
    summary = convert_std_dataset(
      std_root=args.std_root,
      mapping_csv=args.mapping_csv,
      out_root=(args.out_root if args.out_root else None),
      copy_images=(not args.no_copy_images),
    )
    import os, json
    os.makedirs(summary.output_root, exist_ok=True)
    with open(os.path.join(summary.output_root, 'conversion_summary.json'), 'w', encoding='utf-8') as f:
      json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
    print(f"[OK] converted std dataset: {summary.output_root}")
    return

  if args.command == "export-tubes":
    from action_data_analysis.analyze.export_tube_videos import export_std_tube_videos
    stats = export_std_tube_videos(
      inputs=list(args.dirs),
      output_root=args.out_dir,
      strategy=str(args.strategy),
      fps=int(args.fps),
      size=int(args.size),
      min_len=int(args.min_len),
      labels_json=str(args.labels_json or ""),
    )
    import json as _json
    print(_json.dumps(stats, ensure_ascii=False, indent=2))
    return

  if args.command == "subset-std":
    summary = create_std_subset(inputs=list(args.dirs), out_root=(args.out_dir if args.out_dir else None), per_class=int(args.per_class))
    import os, json
    os.makedirs(summary.output_root, exist_ok=True)
    with open(os.path.join(summary.output_root, 'subset_summary.json'), 'w', encoding='utf-8') as f:
      json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
    print(f"[OK] subset created: {summary.output_root}")
    return

  if args.command == "split-by-tube":
    from action_data_analysis.convert.split_tubes import split_std_samples_by_tube
    summary = split_std_samples_by_tube(
      inputs=list(args.dirs),
      out_root=(args.out_dir if args.out_dir else None),
      min_len=int(args.min_len),
    )
    import os, json
    os.makedirs(summary.output_root, exist_ok=True)
    with open(os.path.join(summary.output_root, 'split_summary.json'), 'w', encoding='utf-8') as f:
      json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
    print(f"[OK] split-by-tube dataset created: {summary.output_root}")
    return

  if args.command == "split-by-tube-fpswin":
    from action_data_analysis.convert.split_tubes import split_std_samples_by_tube_fpswin
    summary = split_std_samples_by_tube_fpswin(
      inputs=list(args.dirs),
      out_root=(args.out_dir if args.out_dir else None),
      fps=int(args.fps),
      min_len=int(args.min_len),
    )
    import os, json
    os.makedirs(summary.output_root, exist_ok=True)
    with open(os.path.join(summary.output_root, 'split_fpswin_summary.json'), 'w', encoding='utf-8') as f:
      json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
    print(f"[OK] split-by-tube-fpswin dataset created: {summary.output_root}")
    return

  if args.command == "split-gt-csv":
    ratios = list(args.ratios)
    summary = split_gt_csv(
      gt_csv_path=args.gt_csv,
      out_dir=(args.out_dir if args.out_dir else None),
      train_ratio=float(ratios[0]),
      val_ratio=float(ratios[1]),
      test_ratio=float(ratios[2]),
      seed=int(args.seed),
      video_col=(args.video_col if args.video_col else None),
      add_prefix=str(args.prefix),
    )
    import json as _json
    print(_json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return

  if args.command == "std-to-pkl":
    from action_data_analysis.convert.std_pkl import std_to_pkl
    out_pkl = std_to_pkl(
      std_root=args.std_root,
      out_root=(args.out_root if args.out_root else None),
      out_name=str(args.name or "gt.pkl"),
    )
    import json as _json
    print(_json.dumps({"out_pkl": out_pkl}, ensure_ascii=False))
    return

  if args.command == "pkl-to-std":
    from action_data_analysis.convert.std_pkl import pkl_to_std
    out_std = pkl_to_std(
      pkl_path=args.pkl_path,
      videos_root=args.videos_root,
      out_root=(args.out_root if args.out_root else None),
      inplace=bool(args.inplace),
      copy_images=bool(args.copy_images),
    )
    import json as _json
    print(_json.dumps({"out_std_root": out_std}, ensure_ascii=False))
    return

  # 未知命令（理论不会到这，因为 required=True）
  parser.print_help()
  sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
  main()