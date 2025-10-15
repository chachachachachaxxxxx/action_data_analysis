from __future__ import annotations

import os
import shutil
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass
class MergeSummary:
  inputs: List[str]
  output_root: str
  merged_labels: Dict[str, int]
  splits: Dict[str, int]
  videos_copied: int


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def _load_labels_dict(path: str) -> Dict[str, str]:
  with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
  # 允许 {"0":"contest"} 或 {"contest":0} 两种形式；统一为 {name: id}
  if not data:
    return {}
  sample_key = next(iter(data.keys()))
  try:
    int(sample_key)
    # 形如 {"0":"contest"}
    return {str(v): int(k) for k, v in data.items()}
  except Exception:
    # 形如 {"contest":0}
    return {str(k): int(v) for k, v in data.items()}


def _invert_labels(name_to_id: Dict[str, int]) -> Dict[int, str]:
  return {int(i): str(n) for n, i in name_to_id.items()}


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
      # 跳过常见表头
      if cell0 in {"tube", "path", "video", "videos", "sample"}:
        continue
      if len(row) == 1 and "," in row[0]:
        row = row[0].split(",")
      if len(row) < 2:
        # 兼容只有 path 的情况：label 设为 "unknown"
        p = row[0].strip()
        if not p:
          continue
        rows.append((p, "unknown"))
      else:
        rows.append((row[0].strip(), row[1].strip()))
  return rows


def _write_split_csv(rows: List[Tuple[str, str]], out_csv: str) -> None:
  import csv
  _ensure_dir(os.path.dirname(out_csv))
  with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    for p, l in rows:
      w.writerow([p, l])


def _write_labels_dict(name_to_newid: Dict[str, int], out_json: str) -> None:
  # 输出为 {"0":"name"} 形式，兼容现有脚本
  id_to_name = {int(i): str(n) for n, i in name_to_newid.items()}
  obj = {str(i): id_to_name[i] for i in sorted(id_to_name.keys())}
  _ensure_dir(os.path.dirname(out_json))
  with open(out_json, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)


def _normalize_video_relpath(p: str) -> str:
  p = p.strip()
  if p.startswith("videos/"):
    # 保留 videos/ 之后的完整层级
    parts = p.replace("\\", "/").split("/")
    idx = parts.index("videos")
    return "/".join(parts[idx:])
  # 允许传入绝对路径或相对到根的路径，统一取 "videos/<sample>"
  # 若是绝对路径，取最后两级；否则取 basename
  parts = p.replace("\\", "/").split("/")
  if "videos" in parts:
    idx = parts.index("videos")
    return "/".join(parts[idx:]) if idx < len(parts) else "videos"
  # 兜底：当作 sample 目录名
  return f"videos/{os.path.basename(p)}"


def merge_tube_datasets(
  inputs: List[str],
  out_root: str | Path,
) -> MergeSummary:
  """合并多个 tube 数据集：

  期望每个输入都具有结构：
    <in>/
      ├─ videos/
      └─ annotations/
          ├─ train.csv  # 两列：path,label
          ├─ val.csv
          ├─ test.csv
          └─ labels_dict.json  # {"0":"labelA", ...}

  输出：
    <out>/
      ├─ videos/  # 复制并去重
      └─ annotations/
          ├─ train.csv | val.csv | test.csv  # 路径重写为合并后的 "videos/<sample>"
          └─ labels_dict.json  # 新的全局标签表
  """
  if not inputs:
    raise ValueError("inputs 不能为空")

  out_dir = Path(out_root)
  videos_out = out_dir / "videos"
  ann_out = out_dir / "annotations"
  _ensure_dir(str(videos_out))
  _ensure_dir(str(ann_out))

  # 1) 读取并统一标签空间（按名称并集、稳定排序）
  all_label_names: List[str] = []
  seen: set[str] = set()
  labels_per_input: List[Dict[str, int]] = []  # name->old_id
  for root in inputs:
    ld_json = os.path.join(root, "annotations", "labels_dict.json")
    name_to_id = _load_labels_dict(ld_json)
    labels_per_input.append(name_to_id)
    for name in sorted(name_to_id.keys()):
      if name not in seen:
        seen.add(name)
        all_label_names.append(name)
  # 稳定：按名称排序
  all_label_names = sorted(all_label_names)
  name_to_newid: Dict[str, int] = {name: i for i, name in enumerate(all_label_names)}

  # 2) 合并 train/val/test csv（重映射 label，重写路径，去重）
  split_names = ["train", "val", "test"]
  merged_rows: Dict[str, List[Tuple[str, str]]] = {sp: [] for sp in split_names}
  existed_in_split: Dict[str, set[str]] = {sp: set() for sp in split_names}
  videos_copied = 0

  for root, name_to_oldid in zip(inputs, labels_per_input):
    # 从该输入的 CSV 读取
    for sp in split_names:
      csv_path = os.path.join(root, "annotations", f"{sp}.csv")
      rows = _read_split_csv(csv_path)
      for p, l in rows:
        # 规范化目标相对路径（以 videos/ 开头，保留其后的层级；若无，则降级为 basename）
        rel = _normalize_video_relpath(p)
        # 将标签以名称对齐：如果 l 是数字字符串，先用旧 labels 映射回名称；否则视为名称
        label_name: Optional[str]
        try:
          li = int(l)
          # 旧 id -> 名称
          inv = _invert_labels(name_to_oldid)
          label_name = inv.get(li, None)
        except Exception:
          label_name = l
        if not label_name:
          # 无法识别则跳过
          continue
        if label_name not in name_to_newid:
          # 不在并集（理论不应发生），跳过
          continue
        new_label = str(name_to_newid[label_name])
        # 按 split 内路径去重
        if rel in existed_in_split[sp]:
          continue
        existed_in_split[sp].add(rel)
        merged_rows[sp].append((rel, new_label))

        # 复制该条样例对应的视频（文件或目录）
        # 源可能是：绝对路径；相对（以 videos/ 开头）；或仅文件/目录名
        src_candidates: List[str] = []
        # 原 CSV 的 p 优先
        if os.path.isabs(p):
          src_candidates.append(p)
        # 相对到输入根
        if p.startswith("videos/"):
          src_candidates.append(os.path.join(root, p))
        src_candidates.append(os.path.join(root, rel))
        # 兼容仅名
        base = os.path.basename(rel)
        src_candidates.append(os.path.join(root, "videos", base))

        src_path: Optional[str] = None
        for cand in src_candidates:
          if os.path.exists(cand):
            src_path = cand
            break
        if src_path is None:
          continue  # 源不存在则跳过复制（但仍写入 CSV）

        dst_path = os.path.join(out_dir, rel)
        _ensure_dir(os.path.dirname(dst_path))
        if os.path.exists(dst_path):
          continue
        if os.path.isdir(src_path):
          shutil.copytree(src_path, dst_path)
          videos_copied += 1
        elif os.path.isfile(src_path):
          shutil.copy2(src_path, dst_path)
          videos_copied += 1

  # 3) 写出 labels_dict.json
  _write_labels_dict(name_to_newid, str(ann_out / "labels_dict.json"))

  # 4) 写出合并后的 CSV
  for sp in split_names:
    _write_split_csv(merged_rows[sp], str(ann_out / f"{sp}.csv"))

  return MergeSummary(
    inputs=[str(Path(p).resolve()) for p in inputs],
    output_root=str(out_dir.resolve()),
    merged_labels=name_to_newid,
    splits={sp: len(merged_rows[sp]) for sp in split_names},
    videos_copied=videos_copied,
  )


def main_cli() -> None:
  import argparse
  parser = argparse.ArgumentParser(
    description="Merge multiple tube datasets with structure: videos/ + annotations/{train,val,test}.csv + labels_dict.json",
  )
  parser.add_argument("inputs", nargs="+", help="待合并的数据集根目录（包含 videos 与 annotations）")
  parser.add_argument("--out", required=True, help="输出根目录")
  args = parser.parse_args()

  summary = merge_tube_datasets(args.inputs, args.out)
  print(json.dumps({
    "inputs": summary.inputs,
    "output_root": summary.output_root,
    "labels": summary.merged_labels,
    "splits": summary.splits,
    "videos_copied": summary.videos_copied,
  }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main_cli()


