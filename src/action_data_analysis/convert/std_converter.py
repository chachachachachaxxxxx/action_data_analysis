from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import shutil
import os
from tqdm import tqdm

from .label_map_converter import convert_dataset_dir, ConversionStats


@dataclass
class StdConversionSummary:
  source_root: str
  output_root: str
  samples_total: int
  samples_converted: int
  samples_deleted_after: int
  json_files_total: int
  json_files_converted: int
  json_files_skipped: int
  json_files_deleted: int
  annotations_before: int
  annotations_after: int
  images_copied: int


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_std_root(path: Path) -> bool:
  return path.is_dir() and (path / "videos").is_dir()


def _iter_std_samples(std_root: Path) -> List[Path]:
  videos = std_root / "videos"
  if not videos.exists():
    return []
  items: List[Path] = []
  for child in sorted(videos.iterdir()):
    if child.is_dir():
      items.append(child)
  return items


def _copy_all_images(src_dir: Path, dst_dir: Path) -> int:
  """Copy all image files (by extension) recursively from src_dir to dst_dir, preserving layout."""
  copied = 0
  for root, _dirs, files in os.walk(src_dir):
    rel = Path(root).relative_to(src_dir)
    out_sub = (dst_dir / rel)
    out_sub.mkdir(parents=True, exist_ok=True)
    for fname in files:
      ext = Path(fname).suffix.lower()
      if ext in IMG_EXTS:
        src_path = Path(root) / fname
        dst_path = out_sub / fname
        if not dst_path.exists():
          shutil.copy2(src_path, dst_path)
          copied += 1
  return copied


def convert_std_dataset(std_root: str | Path, mapping_csv: str | Path, out_root: str | Path | None = None, copy_images: bool = True) -> StdConversionSummary:
  std_root = Path(std_root)
  if not _is_std_root(std_root):
    raise ValueError(f"Not a std dataset root (missing 'videos'): {std_root}")

  if out_root is None:
    out_dir = std_root.parent / f"{std_root.name}_converted"
  else:
    out_dir = Path(out_root)
  videos_out = out_dir / "videos"
  stats_out = out_dir / "stats"
  videos_out.mkdir(parents=True, exist_ok=True)
  stats_out.mkdir(parents=True, exist_ok=True)

  samples = _iter_std_samples(std_root)

  json_total = 0
  json_converted = 0
  json_skipped = 0
  json_deleted = 0
  ann_before = 0
  ann_after = 0
  imgs_copied = 0
  samples_converted = 0
  samples_deleted_after = 0

  pbar = tqdm(total=len(samples), desc="convert std", unit="sample")
  for sdir in samples:
    # Count json files before
    json_before = len(list(sdir.rglob("*.json")))
    json_total += json_before

    # Convert this sample directory into target sample directory
    target_sample_dir = videos_out / sdir.name
    stats: ConversionStats = convert_dataset_dir(
      dataset_dir=sdir,
      mapping_csv=mapping_csv,
      output_dir=target_sample_dir,
      overwrite=False,
      copy_images=False,  # 先不按 JSON 复制，待样例保留后再复制整个样例的所有图片
    )

    json_converted += stats.converted_files
    json_skipped += stats.skipped_files
    ann_before += stats.total_annotations_before
    ann_after += stats.total_annotations_after
    json_deleted += stats.files_deleted

    # After conversion, if target sample has no JSON, remove it (样例级删除)
    has_json_after = any(p.suffix.lower() == ".json" for p in target_sample_dir.rglob("*.json")) if target_sample_dir.exists() else False
    if not has_json_after:
      if target_sample_dir.exists():
        shutil.rmtree(target_sample_dir, ignore_errors=True)
      samples_deleted_after += 1
    else:
      # 样例仍保留：复制该样例下的所有图片文件（不仅限于有 JSON 的）
      if copy_images:
        imgs_copied += _copy_all_images(sdir, target_sample_dir)
      samples_converted += 1
    try:
      pbar.update(1)
    except Exception:
      pass

  try:
    pbar.close()
  except Exception:
    pass

  return StdConversionSummary(
    source_root=str(std_root.resolve()),
    output_root=str(out_dir.resolve()),
    samples_total=len(samples),
    samples_converted=samples_converted,
    samples_deleted_after=samples_deleted_after,
    json_files_total=json_total,
    json_files_converted=json_converted,
    json_files_skipped=json_skipped,
    json_files_deleted=json_deleted,
    annotations_before=ann_before,
    annotations_after=ann_after,
    images_copied=imgs_copied,
  )
