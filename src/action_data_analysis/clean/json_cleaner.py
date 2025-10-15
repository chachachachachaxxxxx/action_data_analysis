from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import os
import json
import shutil

import numpy as np

from action_data_analysis.io.json import iter_labelme_dir, extract_bbox_and_action
from action_data_analysis.analyze.export import _discover_std_sample_folders


def _parse_frame_index_from_path(path: str) -> Optional[int]:
  base = os.path.basename(path)
  name, _ = os.path.splitext(base)
  digits = ""
  i = len(name) - 1
  while i >= 0 and name[i].isdigit():
    digits = name[i] + digits
    i -= 1
  if digits:
    try:
      return int(digits)
    except Exception:
      return None
  return None


def _detect_dataset_stride(path: str) -> int:
  s = path.lower()
  if "sportshhi" in s:
    return 5
  if "finesports" in s:
    return 1
  if "multisports" in s:
    return 1
  return 1


def _extract_tid(shape: Dict[str, Any]) -> Optional[str]:
  attrs = shape.get("attributes") or {}
  flags = shape.get("flags") or {}
  tid: Optional[str] = None
  for key in ("id", "track_id", "player_id"):
    v = attrs.get(key)
    if isinstance(v, (str, int)):
      tid = str(v)
      break
  if tid is None:
    for key in ("id", "track_id", "player_id"):
      v = flags.get(key)
      if isinstance(v, (str, int)):
        tid = str(v)
        break
  if tid is None:
    v = shape.get("group_id")
    if isinstance(v, (str, int)):
      tid = str(v)
  return tid


def _collect_segments_for_folder(folder: str) -> List[Dict[str, Any]]:
  frames: List[Tuple[int, List[Tuple[str, str]]]] = []
  for json_path, rec in iter_labelme_dir(folder):
    pairs: List[Tuple[str, str]] = []
    for sh in rec.get("shapes", []) or []:
      parsed = extract_bbox_and_action(sh)
      if parsed is None:
        continue
      _bbox, action = parsed
      if not action:
        continue
      tid = _extract_tid(sh)
      if tid:
        pairs.append((tid, action))
    fidx = _parse_frame_index_from_path(json_path)
    frames.append((fidx if fidx is not None else len(frames), pairs))

  frames.sort(key=lambda t: t[0])

  diffs: Counter[int] = Counter()
  prev_idx: Optional[int] = None
  for fidx, _ in frames:
    if prev_idx is not None:
      d = fidx - prev_idx
      if d > 0:
        diffs[d] += 1
    prev_idx = fidx
  if diffs:
    stride = max(diffs.items(), key=lambda kv: (kv[1], -kv[0]))[0]
  else:
    stride = _detect_dataset_stride(folder)

  last_idx_by_key: Dict[Tuple[str, str], int] = {}
  cur_len_by_key: Dict[Tuple[str, str], int] = {}
  cur_start_by_key: Dict[Tuple[str, str], int] = {}
  segments: List[Dict[str, Any]] = []

  for fidx, pairs in frames:
    if not pairs:
      continue
    unique_pairs = set(pairs)
    for key in unique_pairs:
      if key in last_idx_by_key and (fidx - last_idx_by_key[key] == stride):
        cur_len_by_key[key] = cur_len_by_key.get(key, 0) + 1
        last_idx_by_key[key] = fidx
      else:
        if key in cur_len_by_key and cur_len_by_key[key] > 0:
          tid, act = key
          segments.append({
            "action": act,
            "tid": tid,
            "folder": os.path.abspath(folder),
            "start": int(cur_start_by_key[key]),
            "end": int(last_idx_by_key[key]),
            "length": int(cur_len_by_key[key]),
          })
        cur_len_by_key[key] = 1
        last_idx_by_key[key] = fidx
        cur_start_by_key[key] = fidx

  for key, L in cur_len_by_key.items():
    if L > 0:
      tid, act = key
      segments.append({
        "action": act,
        "tid": tid,
        "folder": os.path.abspath(folder),
        "start": int(cur_start_by_key[key]),
        "end": int(last_idx_by_key[key]),
        "length": int(L),
      })

  return segments


def _collect_wh_stats(folders: List[str]) -> Tuple[float, float, float, float]:
  widths: List[float] = []
  heights: List[float] = []
  for folder in folders:
    for _json_path, rec in iter_labelme_dir(folder):
      for sh in rec.get("shapes", []) or []:
        parsed = extract_bbox_and_action(sh)
        if parsed is None:
          continue
        (x1, y1, x2, y2), _act = parsed
        try:
          w = float(x2) - float(x1)
          h = float(y2) - float(y1)
        except Exception:
          continue
        if w > 0 and h > 0:
          widths.append(w)
          heights.append(h)
  if not widths or not heights:
    return 0.0, 0.0, 0.0, 0.0
  arr_w = np.asarray(widths, dtype=float)
  arr_h = np.asarray(heights, dtype=float)
  return float(arr_w.mean()), float(arr_w.std()), float(arr_h.mean()), float(arr_h.std())


def _collect_bigbox_frames_by_key(folder: str, w_thr: float, h_thr: float) -> Dict[Tuple[str, str], Set[int]]:
  key_to_frames: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
  for json_path, rec in iter_labelme_dir(folder):
    fidx = _parse_frame_index_from_path(json_path)
    if fidx is None:
      continue
    for sh in rec.get("shapes", []) or []:
      parsed = extract_bbox_and_action(sh)
      if parsed is None:
        continue
      (x1, y1, x2, y2), act = parsed
      if not act:
        continue
      tid = _extract_tid(sh)
      if not tid:
        continue
      try:
        w = float(x2) - float(x1)
        h = float(y2) - float(y1)
      except Exception:
        continue
      if w > w_thr or h > h_thr:
        key_to_frames[(tid, act)].add(int(fidx))
  return key_to_frames


def _should_remove_by_length(seg: Dict[str, Any], max_len: int, exceptions: Set[str]) -> bool:
  act = str(seg.get("action") or "")
  if act.lower() in exceptions:
    return False
  L = int(seg.get("length", 0))
  return L > int(max_len)


def _should_remove_by_bigbox(seg: Dict[str, Any], flagged_frames_by_key: Dict[Tuple[str, str], Set[int]]) -> bool:
  tid = str(seg.get("tid"))
  act = str(seg.get("action"))
  key = (tid, act)
  frames = flagged_frames_by_key.get(key)
  if not frames:
    return False
  start = int(seg.get("start", 0))
  end = int(seg.get("end", -1))
  for f in frames:
    if start <= int(f) <= end:
      return True
  return False


def _filter_shapes_in_record(rec: Dict[str, Any], fidx: int, removed_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
  shapes_out: List[Dict[str, Any]] = []
  for sh in rec.get("shapes", []) or []:
    parsed = extract_bbox_and_action(sh)
    if parsed is None:
      continue
    _bbox, act = parsed
    tid = _extract_tid(sh)
    if not tid:
      # 没有 tid 的 shape 暂不删
      shapes_out.append(sh)
      continue
    to_remove = False
    for seg in removed_segments:
      if str(seg.get("tid")) == tid and str(seg.get("action")) == str(act):
        if int(seg.get("start", 0)) <= int(fidx) <= int(seg.get("end", -1)):
          to_remove = True
          break
    if not to_remove:
      shapes_out.append(sh)
  out = dict(rec)
  out["shapes"] = shapes_out
  return out


def clean_labelme_json_dataset(
  inputs: List[str],
  out_root: str,
  sigma: float = 3.0,
  max_len: int = 64,
  exceptions: Optional[List[str]] = None,
) -> Dict[str, Any]:
  """清洗 LabelMe JSON 数据集（std/videos/*）。

  硬规则：
  1) 3σ 异常大框：若 tube 任一帧框宽或高超阈值，则剔除整条 tube 段；
  2) 过长 tube：除 exceptions 外，长度超过 max_len 的 tube 段剔除。

  返回：统计字典，并在 out_root 写出过滤后的 JSON 文件（仅 JSON，图片不复制）。
  """
  folders = _discover_std_sample_folders(inputs)
  os.makedirs(out_root, exist_ok=True)

  # 统计全局宽高分布
  w_mean, w_std, h_mean, h_std = _collect_wh_stats(folders)
  w_thr = float(w_mean + float(sigma) * w_std)
  h_thr = float(h_mean + float(sigma) * h_std)

  # 收集各样例的 segments 及大框帧
  removed_by_length: List[Dict[str, Any]] = []
  removed_by_bigbox: List[Dict[str, Any]] = []
  kept_segments: List[Dict[str, Any]] = []

  exceptions_set = set(x.strip().lower() for x in (exceptions or ["hold", "noball"]))

  segments_by_folder: Dict[str, List[Dict[str, Any]]] = {}
  flagged_frames_by_key_by_folder: Dict[str, Dict[Tuple[str, str], Set[int]]] = {}

  for folder in folders:
    segs = _collect_segments_for_folder(folder)
    segments_by_folder[folder] = segs
    flagged_frames_by_key_by_folder[folder] = _collect_bigbox_frames_by_key(folder, w_thr, h_thr)

  # 判定需移除的 segments
  for folder, segs in segments_by_folder.items():
    big_frames = flagged_frames_by_key_by_folder.get(folder, {})
    for seg in segs:
      rm_len = _should_remove_by_length(seg, max_len=max_len, exceptions=exceptions_set)
      rm_big = _should_remove_by_bigbox(seg, flagged_frames_by_key=big_frames)
      if rm_len:
        removed_by_length.append(seg)
      elif rm_big:
        removed_by_bigbox.append(seg)
      else:
        kept_segments.append(seg)

  # 写出过滤后的 JSON（复制目录结构，但仅写 JSON）
  # 约定输入样例路径：<std_root>/videos/<sample>/，输出：<out_root>/videos/<sample>/
  total_json = 0
  total_json_after = 0
  total_shapes_before = 0
  total_shapes_after = 0

  removed_index: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
  for seg in removed_by_length:
    removed_index[seg["folder"]].append(seg)
  for seg in removed_by_bigbox:
    removed_index[seg["folder"]].append(seg)

  for folder in folders:
    # 计算相对路径：找出包含 "/videos/" 的上层 root
    abs_folder = os.path.abspath(folder)
    parts = abs_folder.split(os.sep)
    if "videos" in parts:
      vid_idx = parts.index("videos")
      rel_sub = os.path.join(*parts[vid_idx:])
    else:
      # 若不含 videos，则用最后一级目录名
      rel_sub = os.path.join("videos", os.path.basename(folder))
    out_folder = os.path.join(out_root, rel_sub)
    os.makedirs(out_folder, exist_ok=True)

    for json_path, rec in iter_labelme_dir(folder):
      total_json += 1
      fidx = _parse_frame_index_from_path(json_path) or -1
      total_shapes_before += len(rec.get("shapes", []) or [])
      cleaned = _filter_shapes_in_record(rec, fidx=fidx, removed_segments=removed_index.get(os.path.abspath(folder), []))
      total_shapes_after += len(cleaned.get("shapes", []) or [])
      if not cleaned.get("shapes"):
        # 仍写空 JSON，以保持帧对齐；也可选择跳过
        pass
      fname = os.path.basename(json_path)
      out_path = os.path.join(out_folder, fname)
      with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
      total_json_after += 1

  summary: Dict[str, Any] = {
    "inputs": [os.path.abspath(p) for p in inputs],
    "output_root": os.path.abspath(out_root),
    "sigma": float(sigma),
    "w_mean": w_mean, "w_std": w_std, "h_mean": h_mean, "h_std": h_std,
    "w_threshold": w_thr, "h_threshold": h_thr,
    "max_len": int(max_len),
    "exceptions": sorted(list(exceptions_set)),
    "stats": {
      "num_folders": int(len(folders)),
      "num_segments_total": int(sum(len(v) for v in segments_by_folder.values())),
      "removed_by_length": int(len(removed_by_length)),
      "removed_by_bigbox": int(len(removed_by_bigbox)),
      "kept_segments": int(len(kept_segments)),
      "json_files_before": int(total_json),
      "json_files_after": int(total_json_after),
      "shapes_before": int(total_shapes_before),
      "shapes_after": int(total_shapes_after),
    },
  }

  # 写出概要
  with open(os.path.join(out_root, "clean_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

  return summary


