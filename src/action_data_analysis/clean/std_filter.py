from __future__ import annotations

import os
import shutil
from typing import List, Optional, Sequence

from action_data_analysis.analyze.export import _discover_std_sample_folders


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def filter_std_dataset(
  inputs: Sequence[str],
  out_root: Optional[str] = None,
  include_prefixes: Optional[Sequence[str]] = None,
  exclude_prefixes: Optional[Sequence[str]] = None,
) -> dict:
  """按样例名 include/exclude 前缀过滤，输出新的 std 根目录，返回简要字典。"""
  sample_dirs = _discover_std_sample_folders(list(inputs))
  if not sample_dirs:
    raise ValueError("No std samples discovered from inputs")

  any_sample = sample_dirs[0]
  videos_dir = os.path.dirname(any_sample)
  std_root = os.path.dirname(videos_dir)
  if not out_root:
    out_root = std_root + "_filtered"

  videos_out = os.path.join(out_root, "videos")
  _ensure_dir(videos_out)

  includes = [str(p) for p in (include_prefixes or []) if str(p)]
  excludes = [str(p) for p in (exclude_prefixes or []) if str(p)]

  def _match_include(name: str) -> bool:
    if not includes:
      return True
    return any(name.startswith(pref) for pref in includes)

  def _match_exclude(name: str) -> bool:
    if not excludes:
      return False
    return any(name.startswith(pref) for pref in excludes)

  kept = 0
  removed = 0
  for sample_dir in sample_dirs:
    sample_name = os.path.basename(os.path.normpath(sample_dir))
    if not _match_include(sample_name) or _match_exclude(sample_name):
      removed += 1
      continue
    dst_dir = os.path.join(videos_out, sample_name)
    shutil.copytree(sample_dir, dst_dir, dirs_exist_ok=True)
    kept += 1

  return {
    "input_samples": len(sample_dirs),
    "kept": kept,
    "removed": removed,
    "output_root": out_root,
    "include_prefixes": includes,
    "exclude_prefixes": excludes,
  }




