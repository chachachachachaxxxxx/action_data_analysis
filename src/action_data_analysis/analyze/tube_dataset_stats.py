from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple


def _read_labels_dict(labels_json: str) -> Dict[int, str]:
  with open(labels_json, "r", encoding="utf-8") as f:
    obj = json.load(f)
  if not obj:
    return {}
  # 支持 {"0":"label"} 或 {"label":0}
  first_key = next(iter(obj.keys()))
  try:
    int(first_key)
    return {int(k): str(v) for k, v in obj.items()}
  except Exception:
    return {int(v): str(k) for k, v in obj.items()}


def _read_split_csv(csv_path: str) -> List[Tuple[str, str]]:
  import csv
  rows: List[Tuple[str, str]] = []
  if not os.path.exists(csv_path):
    return rows
  with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
      if not row:
        continue
      cell0 = (row[0] or "").strip().lower()
      if cell0 in {"tube", "path", "video", "videos", "sample"}:
        continue
      if len(row) == 1 and "," in row[0]:
        row = row[0].split(",")
      if len(row) < 2:
        p = row[0].strip()
        if not p:
          continue
        rows.append((p, "unknown"))
      else:
        rows.append((row[0].strip(), row[1].strip()))
  return rows


def _normalize_relpath(p: str) -> str:
  p = p.strip()
  parts = p.replace("\\", "/").split("/")
  if "videos" in parts:
    idx = parts.index("videos")
    return "/".join(parts[idx:])
  return f"videos/{os.path.basename(p)}"


def compute_tube_dataset_stats(root: str) -> Dict[str, object]:
  """统计 tube 格式数据集的基本信息：

  - 类别分布（train/val/test 分别统计，既按 id 也按名称汇总）
  - 路径存在性检查（缺失文件/目录列表）
  - 总计样本数、去重后的样本集合大小
  """
  ann = os.path.join(root, "annotations")
  labels_json = os.path.join(ann, "labels_dict.json")
  labels: Dict[int, str] = _read_labels_dict(labels_json) if os.path.exists(labels_json) else {}

  splits = {"train": [], "val": [], "test": []}
  for sp in ("train", "val", "test"):
    rows = _read_split_csv(os.path.join(ann, f"{sp}.csv"))
    splits[sp] = [( _normalize_relpath(p), str(l)) for (p, l) in rows]

  # 统计类别分布
  per_split_counts_id: Dict[str, Dict[str, int]] = {sp: {} for sp in splits}
  per_split_counts_name: Dict[str, Dict[str, int]] = {sp: {} for sp in splits}
  all_relpaths: set[str] = set()
  missing: List[str] = []

  for sp, rows in splits.items():
    for rel, l in rows:
      # 统计按 id
      per_split_counts_id[sp][l] = per_split_counts_id[sp].get(l, 0) + 1
      # 统计按名称
      try:
        li = int(l)
        name = labels.get(li, str(li))
      except Exception:
        name = l
      per_split_counts_name[sp][name] = per_split_counts_name[sp].get(name, 0) + 1

      # 路径检查
      all_relpaths.add(rel)
      abs_path = os.path.join(root, rel)
      if not os.path.exists(abs_path):
        missing.append(rel)

  return {
    "root": os.path.abspath(root),
    "labels": labels,  # {id: name}
    "counts_by_id": per_split_counts_id,    # {split: {id_str: count}}
    "counts_by_name": per_split_counts_name,  # {split: {name: count}}
    "num_samples": {sp: len(rows) for sp, rows in splits.items()},
    "unique_samples": len(all_relpaths),
    "missing": sorted(set(missing)),
  }


