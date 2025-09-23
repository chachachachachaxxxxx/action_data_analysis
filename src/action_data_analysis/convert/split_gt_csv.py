from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitSummary:
  input_csv: str
  output_dir: str
  num_rows_input: int
  num_rows_train: int
  num_rows_val: int
  num_rows_test: int
  ratios: Tuple[float, float, float]
  seed: int
  video_column: str
  prefix_added: str


def _detect_video_column(df: pd.DataFrame) -> str:
  candidates = [
    "video",
    "video_path",
    "path",
    "file",
    "filename",
    "name",
  ]
  lower_to_original = {c.lower(): c for c in df.columns}
  for cand in candidates:
    if cand in lower_to_original:
      return lower_to_original[cand]
  # fallback: pick first string-like column
  for col in df.columns:
    if pd.api.types.is_object_dtype(df[col]):
      return col
  # if none, just use first column
  return df.columns[0]


def _normalize_prefix(path_value: str, prefix: str) -> str:
  if not prefix:
    return path_value
  if path_value.startswith(prefix):
    return path_value
  return f"{prefix}{path_value}"


def split_gt_csv(
  gt_csv_path: str,
  out_dir: Optional[str] = None,
  train_ratio: float = 8.0,
  val_ratio: float = 1.0,
  test_ratio: float = 1.0,
  seed: int = 42,
  video_col: Optional[str] = None,
  add_prefix: str = "videos/",
) -> SplitSummary:
  if not os.path.isfile(gt_csv_path):
    raise FileNotFoundError(f"gt.csv 不存在: {gt_csv_path}")

  df = pd.read_csv(gt_csv_path)

  # 删除 frames 列（如存在）
  if "frames" in df.columns:
    df = df.drop(columns=["frames"])  # type: ignore[arg-type]

  # 检测或使用指定的视频列名
  video_column = video_col if video_col else _detect_video_column(df)
  if video_column not in df.columns:
    raise ValueError(f"未找到视频列: {video_column}，可用列: {list(df.columns)}")

  # 为视频路径添加前缀（若尚未添加）
  df[video_column] = df[video_column].astype(str).map(lambda s: _normalize_prefix(s, add_prefix))

  # 计算切分索引
  total = len(df)
  if total == 0:
    raise ValueError("输入 gt.csv 为空")

  ratios = np.array([float(train_ratio), float(val_ratio), float(test_ratio)], dtype=float)
  if np.any(ratios < 0):
    raise ValueError("比例必须为非负")
  if ratios.sum() <= 0:
    raise ValueError("比例之和必须大于 0")
  probs = ratios / ratios.sum()

  rng = np.random.default_rng(seed)
  perm = rng.permutation(total)

  n_train = int(round(probs[0] * total))
  n_val = int(round(probs[1] * total))
  n_test = total - n_train - n_val

  # 由于四舍五入可能溢出，做一次修正
  if n_test < 0:
    n_test = 0
  if n_train + n_val + n_test != total:
    # 将差额补到 test
    n_test = total - n_train - n_val

  idx_train = perm[:n_train]
  idx_val = perm[n_train : n_train + n_val]
  idx_test = perm[n_train + n_val :]

  df_train = df.iloc[idx_train]
  df_val = df.iloc[idx_val]
  df_test = df.iloc[idx_test]

  # 输出目录：默认与输入同目录
  output_dir = out_dir if out_dir else os.path.dirname(os.path.abspath(gt_csv_path))
  os.makedirs(output_dir, exist_ok=True)

  train_csv = os.path.join(output_dir, "train.csv")
  val_csv = os.path.join(output_dir, "val.csv")
  test_csv = os.path.join(output_dir, "test.csv")

  # 写出：无表头、无索引
  df_train.to_csv(train_csv, index=False, header=False)
  df_val.to_csv(val_csv, index=False, header=False)
  df_test.to_csv(test_csv, index=False, header=False)

  return SplitSummary(
    input_csv=os.path.abspath(gt_csv_path),
    output_dir=os.path.abspath(output_dir),
    num_rows_input=int(total),
    num_rows_train=int(len(df_train)),
    num_rows_val=int(len(df_val)),
    num_rows_test=int(len(df_test)),
    ratios=(float(train_ratio), float(val_ratio), float(test_ratio)),
    seed=int(seed),
    video_column=str(video_column),
    prefix_added=str(add_prefix),
  )


