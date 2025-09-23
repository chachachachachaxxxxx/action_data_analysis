from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import os
import json

import numpy as np

from action_data_analysis.io.json import iter_labelme_dir, extract_bbox_and_action


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


def summarize_lengths(lengths: List[int]) -> Dict[str, float]:
  if not lengths:
    return {"min": 0.0, "p25": 0.0, "mean": 0.0, "p75": 0.0, "max": 0.0}
  arr = np.asarray(lengths, dtype=float)
  return {
    "min": float(np.min(arr)),
    "p25": float(np.percentile(arr, 25)),
    "mean": float(np.mean(arr)),
    "p75": float(np.percentile(arr, 75)),
    "max": float(np.max(arr)),
  }


def _collect_tubes_from_labelme_folder(folder: str) -> Dict[str, List[int]]:
  """从一个 LabelMe 帧目录中，以 (track_id, action) 为键统计tube长度（帧数）。

  返回: {action_name: [tube_len_frames, ...]}
  """
  # 记录每个 (tid, action) 的片段开始、最后一帧索引、当前长度
  frames: List[Tuple[int, List[Tuple[str, str]]]] = []
  for json_path, rec in iter_labelme_dir(folder):
    pairs: List[Tuple[str, str]] = []
    for sh in rec.get("shapes", []) or []:
      parsed = extract_bbox_and_action(sh)
      if parsed is None:
        continue
      _bbox, action = parsed
      # 提取 track/group id
      attrs = sh.get("attributes") or {}
      flags = sh.get("flags") or {}
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
        v = sh.get("group_id")
        if isinstance(v, (str, int)):
          tid = str(v)
      if tid and action:
        pairs.append((tid, action))
    fidx = _parse_frame_index_from_path(json_path)
    frames.append((fidx if fidx is not None else len(frames), pairs))

  frames.sort(key=lambda t: t[0])

  # 自动估计 stride：取相邻帧索引的最常见正向差值，若不可得则退回数据集规则
  diffs: Counter[int] = Counter()
  prev_idx: Optional[int] = None
  for fidx, pairs in frames:
    if prev_idx is not None:
      d = fidx - prev_idx
      if d > 0:
        diffs[d] += 1
    prev_idx = fidx
  if diffs:
    stride = max(diffs.items(), key=lambda kv: (kv[1], -kv[0]))[0]
  else:
    stride = _detect_dataset_stride(folder)

  # 累积长度
  lengths_by_key: Dict[Tuple[str, str], int] = {}
  last_idx_by_key: Dict[Tuple[str, str], int] = {}
  finished_lengths_by_action: Dict[str, List[int]] = defaultdict(list)

  for fidx, pairs in frames:
    if not pairs:
      continue
    for key in pairs:
      if key in last_idx_by_key and (fidx - last_idx_by_key[key] == stride):
        # 连续帧，延长
        lengths_by_key[key] = lengths_by_key.get(key, 0) + 1
        last_idx_by_key[key] = fidx
      else:
        # 与上一次不是恰好 stride，说明开启了新段。
        # 若之前已有长度，先结算到对应动作。
        if key in lengths_by_key and lengths_by_key[key] > 0:
          finished_lengths_by_action[key[1]].append(int(lengths_by_key[key]))
        lengths_by_key[key] = 1
        last_idx_by_key[key] = fidx

    # 对于未出现在本帧、但此前活跃且不是“刚好断开 stride”的键，我们在遇到新键时不立即结算，
    # 而是在遍历结束后统一结算。这里保持简单：结算在最后一次遍历后统一完成。

  # 统一结算所有仍在进行中的片段
  for (tid, action), L in lengths_by_key.items():
    if L > 0:
      finished_lengths_by_action[action].append(int(L))

  return finished_lengths_by_action


def _collect_tube_segments_from_labelme_folder(folder: str) -> List[Dict[str, Any]]:
  """从一个 LabelMe 帧目录中收集每条时空管的片段（含位置信息）。

  返回：List[{
    action: str,
    tid: str,             # 轨迹/实例 id（尽力从 attributes/flags/group_id 提取）
    folder: str,          # 样例目录绝对路径
    start: int,           # 片段起始帧索引（按文件名数值）
    end: int,             # 片段结束帧索引
    length: int,          # 帧数
  }]
  """
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
      # 提取 track/group id
      attrs = sh.get("attributes") or {}
      flags = sh.get("flags") or {}
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
        v = sh.get("group_id")
        if isinstance(v, (str, int)):
          tid = str(v)
      if tid:
        pairs.append((tid, action))
    fidx = _parse_frame_index_from_path(json_path)
    frames.append((fidx if fidx is not None else len(frames), pairs))

  frames.sort(key=lambda t: t[0])

  # 自动估计 stride，若失败则按数据集规则
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
    # 同一帧若重复同键，仅计一次
    unique_pairs = set(pairs)
    for key in unique_pairs:
      if key in last_idx_by_key and (fidx - last_idx_by_key[key] == stride):
        # 连续
        cur_len_by_key[key] = cur_len_by_key.get(key, 0) + 1
        last_idx_by_key[key] = fidx
      else:
        # 结算旧段
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
        # 开新段
        cur_len_by_key[key] = 1
        last_idx_by_key[key] = fidx
        cur_start_by_key[key] = fidx

  # 收尾：结算仍在进行中的段
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


def compute_tube_lengths_for_labelme_dirs(folders: List[str]) -> Dict[str, Any]:
  """聚合多个 LabelMe 帧目录的tube长度分布。

  返回字典包含：
  - per_action: {action: {count, lengths:[...可省略], summary:{min,p25,mean,p75,max}}}
  - overall: {count, summary}
  - folders: [...]
  """
  agg_by_action: Dict[str, List[int]] = defaultdict(list)
  segments_by_action: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
  for folder in folders:
    # 收集段及长度
    segs = _collect_tube_segments_from_labelme_folder(folder)
    for seg in segs:
      act = seg.get("action") or "__unknown__"
      L = int(seg.get("length", 0))
      if L > 0:
        agg_by_action[act].append(L)
        # 保存定位信息供异常输出
        segments_by_action[act].append(seg)
    # 进度更新在性能优化提交中统一添加

  per_action_stats: Dict[str, Any] = {}
  overall_lengths: List[int] = []
  outliers_by_action: Dict[str, Any] = {}

  for action, ls in agg_by_action.items():
    overall_lengths.extend(ls)
    # 统计基本分位
    summary = summarize_lengths(ls)
    per_action_stats[action] = {
      "count": int(len(ls)),
      "summary": summary,
    }
    # 计算 IQR 异常值
    if ls:
      arr = np.asarray(ls, dtype=float)
      q1 = float(np.percentile(arr, 25))
      q3 = float(np.percentile(arr, 75))
      iqr = q3 - q1
      lower = q1 - 1.5 * iqr
      upper = q3 + 1.5 * iqr
      low_list: List[Dict[str, Any]] = []
      high_list: List[Dict[str, Any]] = []
      for seg in segments_by_action.get(action, []):
        L = int(seg.get("length", 0))
        if L < lower:
          low_list.append({
            "folder": seg.get("folder", ""),
            "tid": seg.get("tid", ""),
            "start": int(seg.get("start", 0)),
            "end": int(seg.get("end", 0)),
            "length": L,
          })
        elif L > upper:
          high_list.append({
            "folder": seg.get("folder", ""),
            "tid": seg.get("tid", ""),
            "start": int(seg.get("start", 0)),
            "end": int(seg.get("end", 0)),
            "length": L,
          })
      # 限制每类最多输出一定数量，避免 JSON 过大
      low_list.sort(key=lambda s: s["length"])  # 短的在前
      high_list.sort(key=lambda s: -s["length"])  # 长的在前
      outliers_by_action[action] = {
        "thresholds": {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper},
        "low": low_list[:50],
        "high": high_list[:50],
      }

  result: Dict[str, Any] = {
    "folders": [os.path.abspath(f) for f in folders],
    "per_action": dict(sorted(per_action_stats.items(), key=lambda kv: (-kv[1]["count"], kv[0]))),
    "overall": {
      "count": int(len(overall_lengths)),
      "summary": summarize_lengths(overall_lengths),
    },
  }
  if outliers_by_action:
    result["outliers"] = dict(sorted(outliers_by_action.items(), key=lambda kv: kv[0]))
  return result


def compute_tube_lengths_for_finesports(pkl_path: str) -> Dict[str, Any]:
  import pickle
  with open(pkl_path, "rb") as f:
    d = pickle.load(f)
  labels: List[str] = d.get("labels", [])
  idx_to_label = {i: name for i, name in enumerate(labels)}
  gttubes = d.get("gttubes", {}) or {}

  per_action: Dict[str, List[int]] = defaultdict(list)
  for vid, cls_map in gttubes.items():
    if not isinstance(cls_map, dict):
      continue
    for cls_id, tubes in cls_map.items():
      name = idx_to_label.get(int(cls_id), str(cls_id))
      if not isinstance(tubes, list):
        continue
      for arr in tubes:
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > 0:
          per_action[name].append(int(arr.shape[0]))

  overall: List[int] = []
  per_action_stats: Dict[str, Any] = {}
  for name, ls in per_action.items():
    overall.extend(ls)
    per_action_stats[name] = {"count": int(len(ls)), "summary": summarize_lengths(ls)}

  return {
    "source": os.path.abspath(pkl_path),
    "per_action": dict(sorted(per_action_stats.items(), key=lambda kv: (-kv[1]["count"], kv[0]))),
    "overall": {"count": int(len(overall)), "summary": summarize_lengths(overall)},
  }


def compute_tube_lengths_for_multisports(pkl_path: str) -> Dict[str, Any]:
  """MultiSports PKL 结构与 FineSports 类似：labels + gttubes。"""
  return compute_tube_lengths_for_finesports(pkl_path)


def compute_tube_lengths_for_sportshhi(annotations_dir: str) -> Dict[str, Any]:
  """基于 SportsHHI CSV：按 (video_id, instance_id, action_id) 分组，长度=该组行数。"""
  import pandas as pd
  pbtxt = os.path.join(annotations_dir, "sports_action_list.pbtxt")

  # 读取类别映射
  id_to_name: Dict[int, str] = {}
  if os.path.exists(pbtxt):
    current: Dict[str, Any] = {}
    with open(pbtxt, "r", encoding="utf-8") as f:
      for raw in f:
        line = raw.strip()
        if line.startswith("name:"):
          parts = line.split('"')
          current["name"] = parts[1] if len(parts) > 1 else line.split(":", 1)[1].strip()
        elif line.startswith("label_id:"):
          try:
            current["id"] = int(line.split(":", 1)[1].strip())
          except Exception:
            continue
        elif line.startswith("}"):
          if "id" in current and "name" in current:
            id_to_name[int(current["id"])] = current["name"]
          current = {}
    if current and "id" in current and "name" in current and int(current["id"]) not in id_to_name:
      id_to_name[int(current["id"])] = current["name"]

  # 读取 CSV（train/val）
  cols = [
    "video_id", "frame_id", "x1", "y1", "x2", "y2",
    "x1_2", "y1_2", "x2_2", "y2_2", "action_id", "person_id", "instance_id",
  ]
  dfs: List[pd.DataFrame] = []
  for fname in ["sports_train_v1.csv", "sports_val_v1.csv"]:
    path = os.path.join(annotations_dir, fname)
    if os.path.exists(path):
      df = pd.read_csv(path, header=None, names=cols, dtype={"frame_id": str})
      dfs.append(df)
  df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=cols)
  if df_all.empty:
    return {"source": os.path.abspath(annotations_dir), "per_action": {}, "overall": {"count": 0, "summary": summarize_lengths([])}}

  # 清洗并构造分组键
  df = df_all[["video_id", "action_id", "instance_id"]].copy()
  df = df.dropna(subset=["action_id", "instance_id"])  # 仅保留有实例编号的行
  # action 映射为名称
  def _act_name(x):
    try:
      xi = int(x)
    except Exception:
      return str(x)
    return id_to_name.get(xi, str(xi))
  df["action_name"] = df["action_id"].map(_act_name)

  # 按 (video_id, instance_id, action_name) 分组计数 -> tube 长度
  grp = df.groupby(["video_id", "instance_id", "action_name"], dropna=False).size().reset_index(name="length")

  per_action: Dict[str, List[int]] = defaultdict(list)
  for _, row in grp.iterrows():
    per_action[str(row["action_name"])].append(int(row["length"]))

  overall: List[int] = []
  per_action_stats: Dict[str, Any] = {}
  for name, ls in per_action.items():
    overall.extend(ls)
    per_action_stats[name] = {"count": int(len(ls)), "summary": summarize_lengths(ls)}

  return {
    "source": os.path.abspath(annotations_dir),
    "per_action": dict(sorted(per_action_stats.items(), key=lambda kv: (-kv[1]["count"], kv[0]))),
    "overall": {"count": int(len(overall)), "summary": summarize_lengths(overall)},
  }


def save_results_json_csv(result: Dict[str, Any], out_dir: str, prefix: str) -> Tuple[str, str]:
  os.makedirs(out_dir, exist_ok=True)
  json_path = os.path.join(out_dir, f"{prefix}_tube_lengths.json")
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

  # 导出 CSV：每行一个动作及其计数和分位数
  import csv
  csv_path = os.path.join(out_dir, f"{prefix}_tube_lengths_per_action.csv")
  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["action", "count", "min", "p25", "mean", "p75", "max"])
    for action, meta in result.get("per_action", {}).items():
      s = meta.get("summary", {})
      w.writerow([
        action,
        int(meta.get("count", 0)),
        s.get("min", 0.0), s.get("p25", 0.0), s.get("mean", 0.0), s.get("p75", 0.0), s.get("max", 0.0)
      ])
  return json_path, csv_path


# ======== Plotting helpers ========

def collect_overall_tube_lengths_labelme_dirs(folders: List[str]) -> List[int]:
  """收集多个 LabelMe 目录的全部 tube 长度列表（帧数）。"""
  overall: List[int] = []
  for folder in folders:
    per_action = _collect_tubes_from_labelme_folder(folder)
    for _act, ls in per_action.items():
      overall.extend(int(x) for x in ls)
  # 进度更新在性能优化提交中统一添加
  return overall


def bin_tube_lengths(lengths: List[int]) -> Dict[str, int]:
  """将长度分箱计数：1..25 各一档，外加 "25+"（>=26）。"""
  labels_1_to_25 = [str(i) for i in range(1, 26)]
  bins: Dict[str, int] = {k: 0 for k in labels_1_to_25}
  bins["25+"] = 0
  for L in lengths:
    try:
      v = int(L)
    except Exception:
      continue
    if v <= 0:
      continue
    if 1 <= v <= 25:
      bins[str(v)] += 1
    elif v >= 26:
      bins["25+"] += 1
  return bins


def plot_length_bins(counts: Dict[str, int], title: str, out_png: str) -> None:
  import matplotlib
  matplotlib.use("Agg")  # 无显示环境下渲染
  import matplotlib.pyplot as plt

  labels = [str(i) for i in range(1, 26)] + ["25+"]
  values = [int(counts.get(k, 0)) for k in labels]

  plt.figure(figsize=(6, 4))
  bars = plt.bar(labels, values, color="#4C78A8")
  plt.title(title)
  plt.xlabel("Tube length (frames)")
  plt.ylabel("Count")
  for rect, v in zip(bars, values):
    plt.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), str(v), ha="center", va="bottom", fontsize=9)
  plt.tight_layout()
  os.makedirs(os.path.dirname(out_png), exist_ok=True)
  plt.savefig(out_png, dpi=150)
  plt.close()


# ======== Time-bin statistics (by seconds) ========

def compute_time_bins_for_labelme_dirs(folders: List[str], seconds_per_frame: float) -> Dict[str, Any]:
  """将 tube 帧长度转换为时长（秒），并统计三段计数：<=0.5, (0.5,1], >1。

  返回：{"bins": {"<=0.5": n1, "0.5-1": n2, "1+": n3}, "seconds_per_frame": spf, "num_tubes": N}
  """
  lengths = collect_overall_tube_lengths_labelme_dirs(folders)
  leq_05 = 0
  between_05_1 = 0
  gt_1 = 0
  for L in lengths:
    try:
      v = float(int(L)) * float(seconds_per_frame)
    except Exception:
      continue
    if v <= 0.5:
      leq_05 += 1
    elif v <= 1.0:
      between_05_1 += 1
    else:
      gt_1 += 1
  return {
    "bins": {"<=0.5": leq_05, "0.5-1": between_05_1, "1+": gt_1},
    "seconds_per_frame": float(seconds_per_frame),
    "num_tubes": int(leq_05 + between_05_1 + gt_1),
  }


def save_time_bins_json_csv(result: Dict[str, Any], out_dir: str, prefix: str) -> Tuple[str, str]:
  os.makedirs(out_dir, exist_ok=True)
  import csv
  json_path = os.path.join(out_dir, f"{prefix}_time_bins.json")
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
  csv_path = os.path.join(out_dir, f"{prefix}_time_bins.csv")
  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["bin", "count"])
    for k in ["<=0.5", "0.5-1", "1+"]:
      w.writerow([k, int(result.get("bins", {}).get(k, 0))])
  return json_path, csv_path


