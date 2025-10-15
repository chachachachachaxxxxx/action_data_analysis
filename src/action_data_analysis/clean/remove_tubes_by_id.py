from __future__ import annotations

import os
import json
from typing import Dict, Iterable, List, Optional, Set, Tuple
from tqdm import tqdm

from action_data_analysis.analyze.export import _discover_std_sample_folders
from action_data_analysis.io.json import iter_labelme_dir, extract_bbox_and_action


def _read_remove_list(txt_path: str) -> Dict[str, Set[str]]:
  """读取待移除列表，格式：video_id,track_id 每行一条。

  返回：{video_id -> set(track_id)}
  """
  video_to_tids: Dict[str, Set[str]] = {}
  with open(txt_path, 'r', encoding='utf-8') as f:
    for line in f:
      s = line.strip()
      if not s or s.startswith('#'):
        continue
      parts = [p.strip() for p in s.replace('\t', ',').split(',') if p.strip()]
      if len(parts) < 2:
        continue
      vid, tid = parts[0], parts[1]
      video_to_tids.setdefault(vid, set()).add(tid)
  return video_to_tids


def _extract_tid(shape: Dict[str, object]) -> Optional[str]:
  v = shape.get("group_id")
  if isinstance(v, (str, int)):
    tid = str(v)
  return tid


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def remove_tubes_by_id(
  std_inputs: List[str],
  remove_txt: str,
  out_root: str,
) -> Dict[str, object]:
  """从标准数据集（LabelMe JSON）中按 video_id,track_id 组合删除 tube（逐帧删除对应 tid 的 shapes）。

  - std_inputs：std 根目录 / videos 目录 / videos/* 样例目录；函数会自动发现样例目录
  - remove_txt：待删除清单的 txt 文件路径，格式为每行 `video_id,track_id`
  - out_root：输出新的 std 根目录，仅写 JSON（不复制图片）
  返回：摘要字典
  """
  targets = _read_remove_list(remove_txt)
  folders = _discover_std_sample_folders(std_inputs)

  # 目标 videos 结构
  out_videos = os.path.join(out_root, "videos")
  _ensure_dir(out_videos)

  num_samples = 0
  num_json = 0
  num_shapes_before = 0
  num_shapes_after = 0
  num_shapes_removed = 0
  affected_samples: List[str] = []

# 添加tqdm

  for folder in tqdm(folders, desc="remove tubes"):
    sample_id = os.path.basename(os.path.normpath(folder))
    dst_folder = os.path.join(out_videos, sample_id)
    _ensure_dir(dst_folder)
    num_samples += 1

    remove_tids = targets.get(sample_id) or set()
    if remove_tids:
      affected_samples.append(sample_id)

    for json_path, rec in tqdm(iter_labelme_dir(folder), desc="remove tubes", unit="json"):
      num_json += 1
      shapes_out: List[Dict[str, object]] = []
      for sh in rec.get("shapes", []) or []:
        tid = _extract_tid(sh)
        if tid and tid in remove_tids:
          num_shapes_removed += 1
          continue
        shapes_out.append(sh)
      num_shapes_before += len(rec.get("shapes", []) or [])
      num_shapes_after += len(shapes_out)
      out_rec = dict(rec)
      out_rec["shapes"] = shapes_out
      # 如果该帧没有标注，则跳过
      if  len(shapes_out) == 0:
        continue
      out_json = os.path.join(dst_folder, os.path.basename(json_path))
      with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out_rec, f, ensure_ascii=False, indent=2)

  summary = {
    "inputs": [os.path.abspath(p) for p in std_inputs],
    "remove_txt": os.path.abspath(remove_txt),
    "output_root": os.path.abspath(out_root),
    "samples": int(num_samples),
    "json_files": int(num_json),
    "shapes_before": int(num_shapes_before),
    "shapes_after": int(num_shapes_after),
    "shapes_removed": int(num_shapes_removed),
    "affected_samples": sorted(list(set(affected_samples))),
  }
  _ensure_dir(out_root)
  with open(os.path.join(out_root, 'remove_tubes_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
  return summary


