#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

"""
本脚本用于将 FineSports 风格目录转为 std 格式（videos/stats）。
后续类别映射与样例清理建议使用 src CLI：
  python -m action_data_analysis.cli.main convert-std <std_root> <mapping.csv>
"""

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dir_has_images(dir_path: Path) -> bool:
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return True
    return False


def default_out_dir(src_dir: Path) -> Path:
    name = src_dir.name
    if name.endswith("_json"):
        return src_dir.parent / f"{name}_std"
    return src_dir.parent / f"{name}_std"


def dedup_destination_dir(dest_dir: Path, desired_dirname: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / desired_dirname
    idx = 1
    while candidate.exists():
        candidate = dest_dir / f"{desired_dirname}_{idx}"
        idx += 1
    return candidate


def collect_samples(src_dir: Path) -> List[Path]:
    samples: List[Path] = []
    for action_dir in src_dir.iterdir():
        if not action_dir.is_dir():
            continue
        # Prefer second-level sample directories
        has_subdirs = False
        for sample_dir in action_dir.iterdir():
            if sample_dir.is_dir():
                has_subdirs = True
                samples.append(sample_dir)
        # Fallback: if files directly under action_dir form a sample
        if not has_subdirs and dir_has_images(action_dir):
            samples.append(action_dir)
    return samples


def transfer_samples(src_dir: Path, videos_dir: Path, move: bool) -> int:
    count = 0
    samples = collect_samples(src_dir)
    for sample_dir in samples:
        action_name = sample_dir.parent.name if sample_dir.parent != src_dir else sample_dir.name
        sample_id = sample_dir.name if sample_dir.parent != src_dir else "0"
        target_name = f"{action_name}_{sample_id}"
        target_dir = dedup_destination_dir(videos_dir, target_name)
        if move:
            shutil.move(str(sample_dir), str(target_dir))
        else:
            shutil.copytree(sample_dir, target_dir)
        count += 1
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert FineSports-style dataset to std layout (by sample folders)")
    parser.add_argument("src_dir", type=Path, help="Source directory (e.g., FineSports_json)")
    parser.add_argument("out_dir", type=Path, nargs="?", help="Output std directory (default: src_dir + _std)")
    parser.add_argument("--move", action="store_true", help="Move folders instead of copying (default: copy)")
    args = parser.parse_args(argv)

    src_dir: Path = args.src_dir
    if not src_dir.is_dir():
        print(f"[ERROR] SRC_DIR not found: {src_dir}", file=sys.stderr)
        return 1

    out_dir: Path = args.out_dir if args.out_dir is not None else default_out_dir(src_dir)
    videos_dir = out_dir / "videos"
    stats_dir = out_dir / "stats"

    videos_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    mode = "move" if args.move else "copy"
    print(f"[FineSports->std] src={src_dir} out={out_dir} mode={mode}")

    transferred = transfer_samples(src_dir, videos_dir, move=args.move)

    print(f"[OK] {mode}d {transferred} sample folders into {videos_dir}")
    print("[hint] std 转换完成，可继续执行类别映射：\n"
          "  python -m action_data_analysis.cli.main convert-std <std_root> <mapping.csv>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
