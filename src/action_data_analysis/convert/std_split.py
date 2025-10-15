from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np  # type: ignore

from action_data_analysis.analyze.export import _discover_std_sample_folders


@dataclass
class StdSplitSummary:
  std_root: str
  annotations_dir: str
  num_samples: int
  num_train: int
  num_val: int
  num_test: int
  ratios: Tuple[float, float, float]
  seed: int


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def _infer_label_for_sample(sample_dir: str) -> str:
  """从样例目录内 JSON 的 labels 推断主标签（频次最高）。

  若无法解析，则返回 "unknown"。
  为避免引入新的依赖，这里延迟导入并复用已有的解析工具。
  """
  try:
    from collections import Counter
    from action_data_analysis.io.json import iter_labelme_dir, extract_bbox_and_action
    cnt: Counter[str] = Counter()
    for _json_path, rec in iter_labelme_dir(sample_dir):
      for sh in (rec.get("shapes") or []):
        parsed = extract_bbox_and_action(sh)
        if parsed is None:
          continue
        _bbox, action = parsed
        name = str(action or "").strip() or "unknown"
        cnt[name] += 1
    if not cnt:
      return "unknown"
    return max(cnt.items(), key=lambda kv: kv[1])[0]
  except Exception:
    return "unknown"


def _relative_tube_path(sample_dir: str, std_root: str) -> str:
  """返回相对路径，形如 videos/<sample>。

  下游常以该路径为基准拼接帧或窗口级资源。
  """
  std_root = os.path.abspath(std_root)
  sample_dir = os.path.abspath(sample_dir)
  try:
    rel = os.path.relpath(sample_dir, start=std_root)
  except Exception:
    rel = sample_dir
  # 期望以 "videos/" 开头
  if not rel.startswith("videos" + os.sep):
    # 若传入的是 videos/* 自身，rel 可能是 <sample>
    base = os.path.basename(sample_dir.rstrip(os.sep))
    return os.path.join("videos", base).replace("\\", "/")
  return rel.replace("\\", "/")


def _count_frames_in_sample(sample_dir: str) -> int:
  """统计样例目录中的最大帧号（按文件名尾部数字解析）。"""
  try:
    max_idx = 0
    for name in os.listdir(sample_dir):
      if not os.path.isfile(os.path.join(sample_dir, name)):
        continue
      stem, _ext = os.path.splitext(name)
      # 解析末尾数字
      digits = ""
      i = len(stem) - 1
      while i >= 0 and stem[i].isdigit():
        digits = stem[i] + digits
        i -= 1
      if digits:
        try:
          vi = int(digits)
          if vi > max_idx:
            max_idx = vi
        except Exception:
          pass
    return int(max_idx)
  except Exception:
    return 0


def _load_name_to_id(annotations_dir: str) -> Dict[str, int]:
  """读取 labels 映射，用于将标签名映射为 ID。

  兼容三种形式：
  - list: ["labelA", "labelB", ...] -> name_to_id
  - dict(id->name): {"0": "labelA", 1: "labelB"} -> 反向映射
  - dict(name->id): {"labelA": 0, "labelB": 1}
  优先读取 <annotations_dir>/labels_dict.json；若不存在，则尝试 <annotations_dir>/labels.json
  """
  import json
  name_to_id: Dict[str, int] = {}
  cand_paths = [
    os.path.join(annotations_dir, "labels_dict.json"),
    os.path.join(annotations_dir, "labels.json"),
  ]
  for p in cand_paths:
    if not os.path.exists(p):
      continue
    try:
      with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
      if isinstance(obj, list):
        for i, name in enumerate(obj):
          if isinstance(name, str):
            name_to_id[name] = i
        break
      if isinstance(obj, dict):
        # 判断键是否为 id
        if obj and all(isinstance(k, (int,)) or (isinstance(k, str) and k.isdigit()) for k in obj.keys()):
          # id -> name
          for k, name in obj.items():
            try:
              ki = int(k) if not isinstance(k, int) else k
            except Exception:
              continue
            if isinstance(name, str):
              name_to_id[name] = ki
        else:
          # name -> id
          for name, vid in obj.items():
            try:
              vi = int(vid)
            except Exception:
              continue
            if isinstance(name, str):
              name_to_id[name] = vi
        break
    except Exception:
      name_to_id = {}
  return name_to_id


def split_std_dataset(
  std_root: str,
  out_dir: Optional[str] = None,
  train_ratio: float = 8.0,
  val_ratio: float = 1.0,
  test_ratio: float = 1.0,
  seed: int = 42,
  infer_labels: bool = False,
) -> StdSplitSummary:
  """将 std 标准数据集按比例切分为 train/val/test CSV。

  - 输入：std_root（包含 videos/ 子目录）
  - 输出：<annotations>/train.csv、val.csv、test.csv（无表头）
    - 默认两列：path,label
    - 当 infer_labels=False 时，label 固定为 "unknown"
    - 当 infer_labels=True 时，label 为样例内出现频次最高的动作名
  - 返回：摘要信息
  """
  std_root = os.path.abspath(std_root)
  sample_dirs: List[str] = _discover_std_sample_folders([std_root])
  if not sample_dirs:
    raise ValueError(f"未在 {std_root} 下发现任何样例目录（期望 <root>/videos/* 且包含 JSON）")

  total = len(sample_dirs)
  ratios = np.asarray([float(train_ratio), float(val_ratio), float(test_ratio)], dtype=float)
  if np.any(ratios < 0):
    raise ValueError("比例必须为非负")
  if ratios.sum() <= 0:
    raise ValueError("比例之和必须大于 0")
  probs = ratios / ratios.sum()

  # 采样索引（固定随机种子）
  rng = np.random.default_rng(int(seed))
  perm = rng.permutation(total)
  n_train = int(round(probs[0] * total))
  n_val = int(round(probs[1] * total))
  n_test = total - n_train - n_val
  if n_test < 0:
    n_test = 0
  if n_train + n_val + n_test != total:
    n_test = total - n_train - n_val

  idx_train = perm[:n_train]
  idx_val = perm[n_train : n_train + n_val]
  idx_test = perm[n_train + n_val :]

  # 生成条目：tube,label,frames（tube 对应样例相对路径 videos/<sample>）
  def _rows(idxs: np.ndarray, name_to_id: Dict[str, int]) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    for i in map(int, idxs.tolist()):
      sdir = sample_dirs[i]
      path_rel = _relative_tube_path(sdir, std_root)
      if infer_labels:
        label_name = _infer_label_for_sample(sdir)
      else:
        label_name = "unknown"
      # 映射到 ID（若可）
      if label_name in name_to_id:
        label_str = str(name_to_id[label_name])
      else:
        label_str = label_name
      frames = _count_frames_in_sample(sdir)
      rows.append((path_rel, label_str, int(frames)))
    return rows

  # 输出目录：默认写到 <std_root>/annotations
  annotations_dir = out_dir if out_dir else os.path.join(std_root, "annotations")
  _ensure_dir(annotations_dir)

  # 读取标签映射（可选）
  name_to_id = _load_name_to_id(annotations_dir)

  rows_train = _rows(idx_train, name_to_id)
  rows_val = _rows(idx_val, name_to_id)
  rows_test = _rows(idx_test, name_to_id)

  import csv
  def _write(csv_path: str, rows: List[Tuple[str, str, int]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
      w = csv.writer(f)
      w.writerow(["tube", "label", "frames"])  # 与 export_tube_videos 风格对齐
      for p, l, fr in rows:
        w.writerow([p, l, int(fr)])

  _write(os.path.join(annotations_dir, "train.csv"), rows_train)
  _write(os.path.join(annotations_dir, "val.csv"), rows_val)
  _write(os.path.join(annotations_dir, "test.csv"), rows_test)

  return StdSplitSummary(
    std_root=os.path.abspath(std_root),
    annotations_dir=os.path.abspath(annotations_dir),
    num_samples=int(total),
    num_train=len(rows_train),
    num_val=len(rows_val),
    num_test=len(rows_test),
    ratios=(float(train_ratio), float(val_ratio), float(test_ratio)),
    seed=int(seed),
  )


