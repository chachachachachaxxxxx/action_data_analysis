from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import shutil
from tqdm import tqdm

from action_data_analysis.analyze.export import _discover_std_sample_folders
from action_data_analysis.io.json import iter_labelme_dir, extract_bbox_and_action


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


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


def _find_track_id(shape: Dict[str, Any]) -> Optional[str]:
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


def _list_image_candidates(folder: str, stem: str) -> List[str]:
  # 按常见扩展名尝试
  for ext in (".jpg", ".jpeg", ".png"):
    p = os.path.join(folder, stem + ext)
    if os.path.exists(p):
      return [p]
  # 兜底：在目录内查找同 stem 的文件
  hits: List[str] = []
  for name in os.listdir(folder):
    if os.path.splitext(name)[0] == stem:
      hits.append(os.path.join(folder, name))
  return sorted(hits)


def _load_image_for_json(folder: str, json_path: str, rec: Dict[str, Any]) -> Optional[Tuple[str, np.ndarray]]:
  # 优先使用 JSON 的 imagePath
  img_name = str(rec.get("imagePath") or "").strip()
  if img_name:
    cand = os.path.join(folder, img_name)
    if os.path.exists(cand):
      img = cv2.imread(cand, cv2.IMREAD_COLOR)
      if img is not None:
        return cand, img
  # 退化：用 json 文件名的 stem 去找图像
  stem = os.path.splitext(os.path.basename(json_path))[0]
  for p in _list_image_candidates(folder, stem):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is not None:
      return p, img
  return None


class TubeExporterBase:
  def __init__(self, output_wh: Tuple[int, int] = (224, 224), fps: int = 25) -> None:
    self.out_w = int(max(1, output_wh[0]))
    self.out_h = int(max(1, output_wh[1]))
    self.fps = int(fps)

  def _letterbox_resize(self, crop: np.ndarray) -> np.ndarray:
    H, W = crop.shape[:2]
    if H <= 0 or W <= 0:
      return np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
    scale = min(self.out_w / float(W), self.out_h / float(H))
    new_w = max(1, int(round(W * scale)))
    new_h = max(1, int(round(H * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((self.out_h, self.out_w, 3), dtype=crop.dtype)
    x_off = (self.out_w - new_w) // 2
    y_off = (self.out_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

  def process(self, img: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    raise NotImplementedError

  def export(self, frames: List[Tuple[str, int, Tuple[float, float, float, float]]], out_path: str) -> None:
    # frames: [(img_path, frame_idx, (x1,y1,x2,y2)), ...] 已排序
    writer = cv2.VideoWriter(
      out_path,
      cv2.VideoWriter_fourcc(*'mp4v'),
      float(self.fps),
      (self.out_w, self.out_h),
    )
    for img_path, _fidx, box in frames:
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      if img is None:
        # 读取失败 -> 写入黑帧占位
        writer.write(np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8))
        continue
      frame_out = self.process(img, box)
      writer.write(frame_out)
    writer.release()


"""
统一到 128×176 的方式（同一 tube 内大小各异的检测框）
基本思路（默认 letterbox 策略，推荐）：
先在整段 tube 内找出“面积最大的框”作为参考框，取其宽高得到固定参考尺寸 (ref_w, ref_h)。
对每一帧，以“当前帧检测框的中心”为中心，裁出一个固定大小的窗口（宽 ref_w、高 ref_h）。靠边越界会被截断在图像范围内。
将该裁剪结果按比例缩放并 letterbox 到目标分辨率 128×176（保持原始长宽比，不够的边用黑边填充）。
结果：每帧输出都是 128×176，目标尺度在整段内基本一致、无形变。
另一个可选（square 策略）：
用 tube 内“最大框的最大边”作为参考边长，在每帧以当前框中心裁出固定正方形窗口，然后直接缩放到 128×176。
结果也统一到 128×176，但会发生非等比拉伸（可能形变）；优点是画面可铺满，无黑边。
该设计的目的：
“固定参考窗口 + 居中裁剪”把同一 tube 内不同大小的框统一到近似一致的视觉尺度；
letterbox 再负责把任意裁剪结果稳定地映射到指定输出分辨率 128×176。
"""

class LetterboxExporter(TubeExporterBase):
  def process(self, img: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    H, W = img.shape[:2]
    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
      return np.zeros((self.out_h, self.out_w, 3), dtype=img.dtype)
    crop = img[y1:y2, x1:x2]
    return self._letterbox_resize(crop)


class SquareCropExporter(TubeExporterBase):
  def process(self, img: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    size = max(w, h)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nx1 = int(round(cx - size / 2.0))
    ny1 = int(round(cy - size / 2.0))
    nx2 = int(round(cx + size / 2.0))
    ny2 = int(round(cy + size / 2.0))
    H, W = img.shape[:2]
    sx1 = max(0, nx1); sy1 = max(0, ny1)
    sx2 = min(W, nx2); sy2 = min(H, ny2)
    if sx2 <= sx1 or sy2 <= sy1:
      return np.zeros((self.out_h, self.out_w, 3), dtype=img.dtype)
    crop = img[sy1:sy2, sx1:sx2]
    return cv2.resize(crop, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)


class RefLetterboxExporter(TubeExporterBase):
  """基于参考尺寸（整条 tube 最大框的宽高）进行裁剪，中心按当前帧 box 中心，之后 letterbox 到固定大小。"""

  def __init__(self, ref_wh: Tuple[float, float], output_wh: Tuple[int, int] = (224, 224), fps: int = 25) -> None:
    super().__init__(output_wh=output_wh, fps=fps)
    self.ref_w = max(1.0, float(ref_wh[0]))
    self.ref_h = max(1.0, float(ref_wh[1]))

  def process(self, img: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx1 = int(round(cx - self.ref_w / 2.0))
    ry1 = int(round(cy - self.ref_h / 2.0))
    rx2 = int(round(cx + self.ref_w / 2.0))
    ry2 = int(round(cy + self.ref_h / 2.0))
    H, W = img.shape[:2]
    sx1 = max(0, min(W, rx1)); sy1 = max(0, min(H, ry1))
    sx2 = max(0, min(W, rx2)); sy2 = max(0, min(H, ry2))
    if sx2 <= sx1 or sy2 <= sy1:
      return np.zeros((self.out_h, self.out_w, 3), dtype=img.dtype)
    crop = img[sy1:sy2, sx1:sx2]
    return self._letterbox_resize(crop)


class RefSquareCropExporter(TubeExporterBase):
  """基于参考尺寸（整条 tube 最大框的最大边）生成正方形裁剪，中心按当前帧 box 中心，之后缩放到固定大小。"""

  def __init__(self, ref_side: float, output_wh: Tuple[int, int] = (224, 224), fps: int = 25) -> None:
    super().__init__(output_wh=output_wh, fps=fps)
    self.side = max(1.0, float(ref_side))

  def process(self, img: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    sx1 = int(round(cx - self.side / 2.0))
    sy1 = int(round(cy - self.side / 2.0))
    sx2 = int(round(cx + self.side / 2.0))
    sy2 = int(round(cy + self.side / 2.0))
    H, W = img.shape[:2]
    rx1 = max(0, min(W, sx1)); ry1 = max(0, min(H, sy1))
    rx2 = max(0, min(W, sx2)); ry2 = max(0, min(H, sy2))
    if rx2 <= rx1 or ry2 <= ry1:
      return np.zeros((self.out_h, self.out_w, 3), dtype=img.dtype)
    crop = img[ry1:ry2, rx1:rx2]
    return cv2.resize(crop, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)


def _build_tubes_from_folder(folder: str) -> Dict[Tuple[str, str], List[Tuple[int, str, Tuple[float, float, float, float]]]]:
  """从单个样例目录构建 tube：
  返回: {(track_id, action): [(frame_idx, img_path, (x1,y1,x2,y2)), ...]}
  """
  # 收集每帧的信息
  per_frame: List[Tuple[int, str, List[Tuple[str, Tuple[float, float, float, float]]]]] = []
  for json_path, rec in iter_labelme_dir(folder):
    fidx = _parse_frame_index_from_path(json_path)
    img_info = _load_image_for_json(folder, json_path, rec)
    if img_info is None:
      # 没有配套图像，跳过该帧
      continue
    img_path, _img = img_info
    items: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for sh in rec.get("shapes", []) or []:
      parsed = extract_bbox_and_action(sh)
      if parsed is None:
        continue
      (x1, y1, x2, y2), action = parsed
      tid = _find_track_id(sh)
      if not tid or not action:
        continue
      items.append((f"{tid}::{action}", (x1, y1, x2, y2)))
    if items:
      per_frame.append((fidx if fidx is not None else len(per_frame), img_path, items))

  per_frame.sort(key=lambda t: t[0])

  # 根据相邻帧差估计 stride
  stride = 1
  diffs: Dict[int, int] = {}
  last: Optional[int] = None
  for fidx, _p, _ in per_frame:
    if last is not None:
      d = fidx - last
      if d > 0:
        diffs[d] = diffs.get(d, 0) + 1
    last = fidx
  if diffs:
    stride = max(diffs.items(), key=lambda kv: (kv[1], -kv[0]))[0]

  # 按 (tid, action) 聚合，并按 stride 切分连续片段
  tubes: Dict[Tuple[str, str], List[Tuple[int, str, Tuple[float, float, float, float]]]] = {}
  last_fidx_by_key: Dict[Tuple[str, str], int] = {}

  def _append(key: Tuple[str, str], item: Tuple[int, str, Tuple[float, float, float, float]]) -> None:
    tubes.setdefault(key, []).append(item)

  for fidx, img_path, pairs in per_frame:
    for key_str, bbox in pairs:
      tid, action = key_str.split("::", 1)
      key = (tid, action)
      prev = last_fidx_by_key.get(key)
      if prev is not None and (fidx - prev) != stride:
        # 间断：用分隔符标记（后续切段）
        _append(key, (-1, "", (0.0, 0.0, 0.0, 0.0)))
      _append(key, (fidx, img_path, bbox))
      last_fidx_by_key[key] = fidx

  return tubes


def _split_by_separators(seq: List[Tuple[int, str, Tuple[float, float, float, float]]]) -> List[List[Tuple[int, str, Tuple[float, float, float, float]]]]:
  parts: List[List[Tuple[int, str, Tuple[float, float, float, float]]]] = []
  cur: List[Tuple[int, str, Tuple[float, float, float, float]]] = []
  for it in seq:
    if it[0] == -1:
      if cur:
        parts.append(cur)
        cur = []
      continue
    cur.append(it)
  if cur:
    parts.append(cur)
  return parts


def export_std_tube_videos(
  inputs: List[str],
  output_root: str,
  strategy: str = "letterbox",
  fps: int = 25,
  size: Any = 224,
  min_len: int = 1,
  labels_json: str = "",
) -> Dict[str, Any]:
  """将标准 std 格式（LabelMe 帧目录）导出为每条 tube 的 MP4。

  - inputs: std 根目录、videos 目录或具体样例目录；函数会自动发现样例目录
  - strategy: "letterbox" 或 "square"
  - fps: 输出帧率（默认 25）
  - size: 输出分辨率；可为 int（size x size）或 (width,height) 或 "WxH" 字符串
  - min_len: tube 最小帧数过滤
  输出：
  - 写入 <output_root>/videos/*.mp4
  - 写入 <output_root>/annotations/gt.csv （两列：tube,label）
  - 读取 <output_root>/annotations/labels_dict.json 进行标签名→ID 映射（若存在）
  返回：统计信息字典
  """
  folders = _discover_std_sample_folders(inputs)

  # 目标目录
  videos_dir = os.path.join(output_root, "videos")
  ann_dir = os.path.join(output_root, "annotations")
  _ensure_dir(videos_dir)
  _ensure_dir(ann_dir)

  # 读取标签映射（可选）
  name_to_id: Dict[str, int] = {}
  labels_path = labels_json if labels_json else os.path.join(ann_dir, 'labels_dict.json')
  if os.path.exists(labels_path):
    try:
      import json
      with open(labels_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
      # 兼容三种形式：list / {id->name} / {name->id}
      if isinstance(obj, list):
        for i, name in enumerate(obj):
          if isinstance(name, str):
            name_to_id[name] = i
      elif isinstance(obj, dict):
        # 判断键是否为 id（纯数字/数字字符串）
        numeric_keys = all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in obj.keys()) if obj else False
        if numeric_keys:
          for k, name in obj.items():
            try:
              ki = int(k) if not isinstance(k, int) else k
            except Exception:
              continue
            if isinstance(name, str):
              name_to_id[name] = ki
        else:
          for name, vid in obj.items():
            try:
              vi = int(vid)
            except Exception:
              continue
            if isinstance(name, str):
              name_to_id[name] = vi
    except Exception:
      name_to_id = {}
    # 将 labels_dict.json 复制到 annotations 目录
    try:
      dst_labels = os.path.join(ann_dir, 'labels_dict.json')
      if os.path.abspath(labels_path) != os.path.abspath(dst_labels):
        shutil.copy2(labels_path, dst_labels)
    except Exception:
      pass

  # 解析输出尺寸
  out_w, out_h = 224, 224
  try:
    if isinstance(size, (list, tuple)) and len(size) == 2:
      out_w, out_h = int(size[0]), int(size[1])
    elif isinstance(size, str):
      s = size.lower().replace(" ", "")
      if "x" in s:
        parts = s.split("x")
      elif "," in s:
        parts = s.split(",")
      else:
        v = int(s)
        out_w, out_h = v, v
        parts = []
      if parts:
        out_w, out_h = int(parts[0]), int(parts[1])
    else:
      v = int(size)
      out_w, out_h = v, v
  except Exception:
    out_w, out_h = 224, 224

  if strategy.lower() == "square":
    exporter_factory = "ref-square"
  else:
    exporter_factory = "ref-letterbox"

  num_folders = 0
  num_tubes = 0
  num_written = 0
  csv_rows: List[Tuple[str, str, int]] = []  # (tube_name, label, frames)
  # 外层：按样例的进度条
  pbar_samples = tqdm(total=len(folders), desc='samples', unit='sample')
  # 内层：按 tube 的总体进度条
  pbar = tqdm(desc='export tubes', unit='tube')

  for folder in folders:
    num_folders += 1
    tubes = _build_tubes_from_folder(folder)
    sample_id = os.path.basename(folder.rstrip(os.sep))
    for (tid, action), seq in sorted(tubes.items(), key=lambda kv: (kv[0][1], kv[0][0])):
      segments = _split_by_separators(seq)
      for seg_idx, seg in enumerate(segments):
        frames = [(p, f, b) for (f, p, b) in seg if f >= 0]
        if len(frames) < int(min_len):
          continue
        num_tubes += 1
        # 计算参考尺寸：整段中面积最大的框的宽和高、以及正方形边
        ref_box = None  # (x1,y1,x2,y2)
        ref_area = -1.0
        for _p, _f, b in frames:
          x1, y1, x2, y2 = b
          area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
          if area > ref_area:
            ref_area = area
            ref_box = (float(x1), float(y1), float(x2), float(y2))
        if ref_box is None:
          continue
        rw = float(ref_box[2] - ref_box[0])
        rh = float(ref_box[3] - ref_box[1])
        rside = max(rw, rh)
        # 基于参考尺寸创建导出器
        if exporter_factory == "ref-square":
          exporter: TubeExporterBase = RefSquareCropExporter(ref_side=rside, output_wh=(out_w, out_h), fps=fps)
        else:
          exporter = RefLetterboxExporter(ref_wh=(rw, rh), output_wh=(out_w, out_h), fps=fps)
        # 文件名：不再分动作子目录，统一放到 videos 下
        out_name = f"{sample_id}__tid{tid}__seg{seg_idx:02d}.mp4"
        out_path = os.path.join(videos_dir, out_name)
        exporter.export(frames, out_path)
        num_written += 1
        pbar.update(1)
        # 标签：优先映射到 ID，若无映射则使用原始动作名
        tube_len = int(len(frames))
        if action in name_to_id:
          csv_rows.append((out_name, str(name_to_id[action]), tube_len))
        else:
          csv_rows.append((out_name, action, tube_len))
    # 更新样例级进度
    pbar_samples.update(1)

  # 写出 annotations/gt.csv
  gt_csv = os.path.join(ann_dir, 'gt.csv')
  try:
    import csv
    with open(gt_csv, 'w', newline='', encoding='utf-8') as f:
      w = csv.writer(f)
      w.writerow(["tube", "label", "frames"])  # 表头：tube 名称、标签（ID 或名称）、帧数
      for name, lab, flen in csv_rows:
        w.writerow([name, lab, int(flen)])
  except Exception:
    pass
  finally:
    try:
      pbar.close()
    except Exception:
      pass
    try:
      pbar_samples.close()
    except Exception:
      pass

  return {
    "folders": len(folders),
    "visited_folders": int(num_folders),
    "tubes_total": int(num_tubes),
    "videos_written": int(num_written),
    "output_root": os.path.abspath(output_root),
    "videos_dir": os.path.abspath(videos_dir),
    "annotations_dir": os.path.abspath(ann_dir),
    "gt_csv": os.path.abspath(gt_csv),
    "strategy": strategy,
    "fps": int(fps),
    "size": (int(size) if isinstance(size, int) else -1),
    "width": int(out_w),
    "height": int(out_h),
    "labels_mapped": int(sum(1 for _n, lab, _l in csv_rows if isinstance(lab, str) and lab.isdigit())),
    "labels_unmapped": int(sum(1 for _n, lab, _l in csv_rows if not (isinstance(lab, str) and lab.isdigit()))),
  }


