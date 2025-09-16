#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Set

"""
本脚本用于将 MultiSports 风格目录转为 std 格式（videos/stats）。
后续类别映射与样例清理建议使用 src CLI：
  python -m action_data_analysis.cli.main convert-std <std_root> <mapping.csv>
"""

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_json(path: Path) -> bool:
    return path.suffix.lower() == ".json"


def dir_has_images(dir_path: Path) -> bool:
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return True
    return False


def dedup_destination_file(dest_dir: Path, desired_name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = desired_name
    suffix = ""
    dot = desired_name.rfind(".")
    if dot != -1:
        stem = desired_name[:dot]
        suffix = desired_name[dot:]
    candidate = dest_dir / f"{stem}{suffix}"
    idx = 1
    while candidate.exists():
        candidate = dest_dir / f"{stem}_{idx}{suffix}"
        idx += 1
    return candidate


def dedup_destination_dir(dest_dir: Path, desired_dirname: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / desired_dirname
    idx = 1
    while candidate.exists():
        candidate = dest_dir / f"{desired_dirname}_{idx}"
        idx += 1
    return candidate


def default_out_dir(src_dir: Path) -> Path:
    name = src_dir.name
    if name.endswith("_json"):
        return src_dir.parent / f"{name}_std"
    return src_dir.parent / f"{name}_std"


def select_sample_dirs(src_dir: Path) -> List[Path]:
    # Select deepest directories that contain images
    all_dirs: List[Path] = [d for d in src_dir.rglob("*") if d.is_dir()]
    # Process deepest first
    all_dirs.sort(key=lambda d: len(d.relative_to(src_dir).parts), reverse=True)
    selected: List[Path] = []
    selected_set: Set[str] = set()
    for d in all_dirs:
        if not dir_has_images(d):
            continue
        # skip if a deeper selected sample is within this directory
        has_deeper = False
        d_prefix = str(d) + os.sep
        for s in selected:
            if str(s).startswith(d_prefix):
                has_deeper = True
                break
        if not has_deeper:
            selected.append(d)
            selected_set.add(str(d))
    return selected


def transfer_samples(sample_dirs: List[Path], videos_dir: Path, move: bool) -> int:
    count = 0
    for sdir in sample_dirs:
        target_dir = dedup_destination_dir(videos_dir, sdir.name)
        if move:
            shutil.move(str(sdir), str(target_dir))
        else:
            shutil.copytree(sdir, target_dir)
        count += 1
    return count


def transfer_non_json_files(src_dir: Path, videos_dir: Path, move: bool) -> int:
    # Kept for backward-compat if needed elsewhere; not used in main flow anymore
    moved = 0
    file_op = shutil.move if move else shutil.copy2
    for path in src_dir.rglob("*"):
        if path.is_file() and not is_json(path):
            target = dedup_destination_file(videos_dir, path.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            file_op(str(path), str(target))
            moved += 1
    return moved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert MultiSports-style dataset to std layout (by sample folders)")
    parser.add_argument("src_dir", type=Path, help="Source directory (e.g., MultiSports_json)")
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
    print(f"[MultiSports->std] src={src_dir} out={out_dir} mode={mode}")

    samples = select_sample_dirs(src_dir)
    transferred = transfer_samples(samples, videos_dir, move=args.move)

    print(f"[OK] {mode}d {transferred} sample folders into {videos_dir}")
    print("[hint] std 转换完成，可继续执行类别映射：\n"
          "  python -m action_data_analysis.cli.main convert-std <std_root> <mapping.csv>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
