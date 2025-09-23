from __future__ import annotations

import os
import shutil
from typing import Iterable, List, Tuple, Optional, Dict
import json

from action_data_analysis.analyze.visual import sample_per_class_examples
from action_data_analysis.io.json import iter_labelme_dir, extract_bbox_and_action
from tqdm import tqdm


def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)


def _list_sorted_images(folder: str) -> List[str]:
  allow_exts = (".jpg", ".jpeg", ".png")
  return sorted([f for f in os.listdir(folder) if f.lower().endswith(allow_exts)])


def _discover_leaf_folders(paths: List[str]) -> List[str]:
  """递归发现包含 LabelMe JSON 的叶子目录。

  定义：目录内若存在至少一个 .json 文件则视为叶子目录；否则继续向下递归。
  """
  leafs: List[str] = []
  for p in paths:
    if not os.path.isdir(p):
      continue
    entries = []
    try:
      entries = os.listdir(p)
    except Exception:
      continue
    has_json = any(name.lower().endswith('.json') for name in entries)
    if has_json:
      leafs.append(p)
      continue
    # 继续向下搜索
    subdirs = [os.path.join(p, name) for name in entries if os.path.isdir(os.path.join(p, name))]
    for sub in subdirs:
      leafs.extend(_discover_leaf_folders([sub]))
  # 去重
  return sorted(list(dict.fromkeys(leafs)))


def _folder_has_json(folder: str) -> bool:
  try:
    for name in os.listdir(folder):
      if name.lower().endswith('.json'):
        return True
  except Exception:
    return False
  return False


def _is_std_root(path: str) -> bool:
  return os.path.isdir(path) and os.path.isdir(os.path.join(path, 'videos'))


def _discover_std_sample_folders(paths: List[str]) -> List[str]:
  """发现 std 数据集的样例目录：<root>/videos/*。

  规则：
  - 若传入 std 根目录（包含 videos 子目录），则返回其中所有包含 JSON 的直接子目录；
  - 若传入的是 videos 目录本身，则返回其中所有包含 JSON 的直接子目录；
  - 若传入路径位于 .../videos/* 下，且该目录包含 JSON，则直接返回该目录；
  - 否则，若 <path>/videos 存在，则按 std 根目录处理。
  """
  result: List[str] = []
  seen = set()

  def _add_unique(dir_path: str) -> None:
    ap = os.path.abspath(dir_path)
    if ap not in seen:
      seen.add(ap)
      result.append(ap)

  for p in paths:
    if not os.path.isdir(p):
      continue
    p_abs = os.path.abspath(p)
    base = os.path.basename(os.path.normpath(p_abs))

    # case: videos/* (sample folder)
    parts = p_abs.split(os.sep)
    if len(parts) >= 2 and parts[-2] == 'videos' and os.path.isdir(p_abs):
      if _folder_has_json(p_abs):
        _add_unique(p_abs)
      continue

    # case: std root or any dir that has a 'videos' subdir
    if _is_std_root(p_abs) or os.path.isdir(os.path.join(p_abs, 'videos')):
      videos_dir = os.path.join(p_abs, 'videos')
      try:
        for name in os.listdir(videos_dir):
          sub = os.path.join(videos_dir, name)
          if os.path.isdir(sub) and _folder_has_json(sub):
            _add_unique(sub)
      except Exception:
        pass
      continue

    # case: p itself is 'videos' directory
    if base == 'videos' and os.path.isdir(p_abs):
      try:
        for name in os.listdir(p_abs):
          sub = os.path.join(p_abs, name)
          if os.path.isdir(sub) and _folder_has_json(sub):
            _add_unique(sub)
      except Exception:
        pass
      continue

  return sorted(result)


def export_samples_with_context(
  folders: List[str],
  output_root: str,
  dataset_name: str,
  per_class: int = 3,
  context_frames: int = 6,
) -> None:
  """导出每类随机样例及其前后上下文帧，复制原始 IMG 与 JSON。

  - folders: 若干个包含帧图与 LabelMe JSON 的目录
  - output_root: 输出根目录（例如 /.../output/json）
  - dataset_name: 数据集名称（用于输出路径）
  - per_class: 每类采样数量
  - context_frames: 采样帧两侧各扩展帧数，总计 2*context_frames+1，若不足则取全部
  """
  leaf_folders = _discover_leaf_folders(folders)
  picked = sample_per_class_examples(leaf_folders, per_class=per_class)

  dataset_root = os.path.join(output_root, dataset_name)
  _ensure_dir(dataset_root)

  # 总计条目数量（每个 items 里是一条中心帧样例）
  total_items = sum(len(v) for v in picked.values())
  pbar = tqdm(total=total_items, desc="export samples", unit="item")
  for action, items in picked.items():
    act_dir = os.path.join(dataset_root, action or "unknown")
    _ensure_dir(act_dir)
    for json_center_path, img_center_path in items:
      folder = os.path.dirname(json_center_path)
      center_fname = os.path.basename(img_center_path)

      frames = _list_sorted_images(folder)
      try:
        pos = frames.index(center_fname)
      except ValueError:
        pos = -1

      if pos >= 0:
        lo = max(0, pos - context_frames)
        hi = min(len(frames), pos + context_frames + 1)
        ctx = frames[lo:hi]
      else:
        # 找不到中心帧，则仅导出中心帧（若存在）
        ctx = [center_fname] if center_fname in frames else []
        if not ctx and frames:
          ctx = [frames[0]]

      sample_id = f"{os.path.basename(folder)}__{os.path.splitext(center_fname)[0]}"
      sample_dir = os.path.join(act_dir, sample_id)
      _ensure_dir(sample_dir)

      for fname in ctx:
        src_img_path = os.path.join(folder, fname)
        if os.path.exists(src_img_path):
          shutil.copy2(src_img_path, os.path.join(sample_dir, fname))

        src_json_path = os.path.join(folder, os.path.splitext(fname)[0] + ".json")
        if os.path.exists(src_json_path):
          shutil.copy2(src_json_path, os.path.join(sample_dir, os.path.basename(src_json_path)))
      pbar.update(1)
  try:
    pbar.close()
  except Exception:
    pass


def merge_flatten_datasets(
  dataset_roots: List[str],
  output_dir: str,
) -> None:
  """将若干数据集导出目录平铺合并到一个目录，并按规则重命名。

  约定输入结构：<root>/<dataset_name>/<action>/<sample_id>/*.{jpg,png,json}
  输出命名：{dataset_name}__{action}__{sample_id}__{basename}.{ext}
  若重名冲突，附加 _dup{n} 后缀。
  """
  _ensure_dir(output_dir)

  def _copy_with_unique_name(src_path: str, dst_dir: str, base_name: str) -> None:
    name = base_name
    stem, ext = os.path.splitext(base_name)
    k = 1
    while os.path.exists(os.path.join(dst_dir, name)):
      name = f"{stem}_dup{k}{ext}"
      k += 1
    shutil.copy2(src_path, os.path.join(dst_dir, name))

  from tqdm import tqdm as _tqdm  # 局部别名，避免与上方 pbar 名冲突
  pbar = _tqdm(desc="flatten", unit="file")
  for root in dataset_roots:
    if not os.path.isdir(root):
      continue
    dataset_name = os.path.basename(os.path.normpath(root))
    for action in sorted(os.listdir(root)):
      action_dir = os.path.join(root, action)
      if not os.path.isdir(action_dir):
        continue
      for sample_id in sorted(os.listdir(action_dir)):
        sample_dir = os.path.join(action_dir, sample_id)
        if not os.path.isdir(sample_dir):
          continue
        allow_img_exts = (".jpg", ".jpeg", ".png")
        for fname in sorted(os.listdir(sample_dir)):
          src_path = os.path.join(sample_dir, fname)
          if not os.path.isfile(src_path):
            continue
          flat_name = f"{dataset_name}__{action}__{sample_id}__{fname}"

          # 若为 JSON，重写其中的 imagePath 指向平铺后的新图片文件名
          if fname.lower().endswith('.json'):
            try:
              with open(src_path, 'r', encoding='utf-8') as f:
                rec = json.load(f)
            except Exception:
              # 读取失败则按原样复制
              _copy_with_unique_name(src_path, output_dir, flat_name)
              continue

            stem = os.path.splitext(fname)[0]
            # 优先根据同目录存在的配对图片确定扩展名
            img_ext: str = ""
            for ext in allow_img_exts:
              if os.path.exists(os.path.join(sample_dir, stem + ext)):
                img_ext = ext
                break
            # 其次根据 JSON 原有 imagePath 的扩展名推断
            if not img_ext:
              old_path = rec.get('imagePath') or ""
              if isinstance(old_path, str) and old_path:
                _, old_ext = os.path.splitext(old_path)
                if old_ext.lower() in allow_img_exts:
                  img_ext = old_ext
            # 兜底
            if not img_ext:
              img_ext = ".jpg"

            new_image_name = f"{dataset_name}__{action}__{sample_id}__{stem}{img_ext}"
            try:
              rec['imagePath'] = new_image_name
              # 写到目标位置（保持 flat_name 为 JSON 文件名）
              dst_path = os.path.join(output_dir, flat_name)
              name = flat_name
              stem2, ext2 = os.path.splitext(flat_name)
              k = 1
              while os.path.exists(os.path.join(output_dir, name)):
                name = f"{stem2}_dup{k}{ext2}"
                k += 1
              dst_path = os.path.join(output_dir, name)
              with open(dst_path, 'w', encoding='utf-8') as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            except Exception:
              # 写失败则回退为原样复制
              _copy_with_unique_name(src_path, output_dir, flat_name)
          else:
            _copy_with_unique_name(src_path, output_dir, flat_name)
          try:
            pbar.update(1)
          except Exception:
            pass
  try:
    pbar.close()
  except Exception:
    pass


def merge_std_samples_into_single(
  inputs: List[str],
  out_root: Optional[str] = None,
  merged_sample_name: str = "merged_all",
) -> Dict[str, int]:
  """将若干 std 样例目录合并为一个样例，输出新的 std 结构。

  规则：
  - 自动发现 inputs 下的样例目录（std 根 / videos / videos/* 均可）。
  - 在 out_root 下创建标准结构：out_root/videos/<merged_sample_name>/
  - 复制每个样例中的图片与 JSON 到上述单一样例目录。
  - 为避免重名冲突：目标文件名加前缀 "<orig_sample_id>__"。
  - 对 JSON：重写 imagePath 指向新文件名（同前缀），扩展名优先按配对图片推断，否则按原 imagePath 推断，兜底 .jpg。

  返回统计信息：{"num_samples": N, "num_images": I, "num_jsons": J}
  """
  sample_dirs = _discover_std_sample_folders(inputs)
  if not sample_dirs:
    return {"num_samples": 0, "num_images": 0, "num_jsons": 0}

  # 推断默认 out_root
  if not out_root:
    # 取第一个样例目录的两级上级作为 std 根（.../<std_root>/videos/<sample>）
    any_sample = sample_dirs[0]
    videos_dir = os.path.dirname(any_sample)
    std_root = os.path.dirname(videos_dir)
    out_root = std_root + "_merged_single"

  # 目标样例目录
  merged_dir = os.path.join(out_root, "videos", merged_sample_name)
  _ensure_dir(merged_dir)

  allow_img_exts = (".jpg", ".jpeg", ".png")
  num_images = 0
  num_jsons = 0

  # 确保唯一命名的帮助函数
  def _unique_dest_name(base_name: str) -> str:
    name = base_name
    stem, ext = os.path.splitext(base_name)
    k = 1
    while os.path.exists(os.path.join(merged_dir, name)):
      name = f"{stem}_dup{k}{ext}"
      k += 1
    return name

  from tqdm import tqdm as _tqdm  # 局部别名
  pbar = _tqdm(total=len(sample_dirs), desc="merge-into-single", unit="sample")
  for sample_dir in sample_dirs:
    orig_sample_id = os.path.basename(os.path.normpath(sample_dir))

    # 列出该样例下的所有文件
    try:
      file_names = sorted(os.listdir(sample_dir))
    except Exception:
      file_names = []

    # 第一遍复制图片（供 JSON 扩展名推断）
    for fname in file_names:
      src_path = os.path.join(sample_dir, fname)
      if not os.path.isfile(src_path):
        continue
      if not os.path.splitext(fname)[1].lower() in allow_img_exts:
        continue
      dst_base = f"{orig_sample_id}__{fname}"
      dst_name = _unique_dest_name(dst_base)
      shutil.copy2(src_path, os.path.join(merged_dir, dst_name))
      num_images += 1

    # 第二遍处理 JSON：重写 imagePath 并写入
    for fname in file_names:
      if not fname.lower().endswith('.json'):
        continue
      src_json_path = os.path.join(sample_dir, fname)
      try:
        with open(src_json_path, 'r', encoding='utf-8') as f:
          rec = json.load(f)
      except Exception:
        # 读取失败，按原样重命名复制
        dst_base = f"{orig_sample_id}__{fname}"
        dst_name = _unique_dest_name(dst_base)
        shutil.copy2(src_json_path, os.path.join(merged_dir, dst_name))
        num_jsons += 1
        continue

      stem = os.path.splitext(fname)[0]
      # 推断图片扩展名（优先存在的配对图片，其次 JSON 原 imagePath，兜底 .jpg）
      img_ext: str = ""
      for ext in allow_img_exts:
        if os.path.exists(os.path.join(sample_dir, stem + ext)):
          img_ext = ext
          break
      if not img_ext:
        old_path = rec.get('imagePath') or ""
        if isinstance(old_path, str) and old_path:
          _, old_ext = os.path.splitext(old_path)
          if old_ext.lower() in allow_img_exts:
            img_ext = old_ext
      if not img_ext:
        img_ext = ".jpg"

      new_image_name = f"{orig_sample_id}__{stem}{img_ext}"
      rec['imagePath'] = new_image_name

      # 确保 JSON 文件名唯一
      dst_json_base = f"{orig_sample_id}__{fname}"
      dst_json_name = _unique_dest_name(dst_json_base)
      dst_json_path = os.path.join(merged_dir, dst_json_name)
      try:
        with open(dst_json_path, 'w', encoding='utf-8') as f:
          json.dump(rec, f, ensure_ascii=False, indent=2)
        num_jsons += 1
      except Exception:
        # 写失败则原样复制
        shutil.copy2(src_json_path, dst_json_path)
        num_jsons += 1

    try:
      pbar.update(1)
    except Exception:
      pass

  try:
    pbar.close()
  except Exception:
    pass

  return {"num_samples": len(sample_dirs), "num_images": num_images, "num_jsons": num_jsons}


def _sanitize_for_dir(name: str) -> str:
  # 将不适合文件夹名的字符替换为下划线，保留常见可读字符
  import re
  s = name.strip()
  s = re.sub(r"\s+", "_", s)
  s = re.sub(r"[^0-9A-Za-z_\-\.\u4e00-\u9fa5]", "_", s)
  return s or "unknown"


def rename_std_samples_by_action(
  inputs: List[str],
  out_root: Optional[str] = None,
  name_format: str = "{action}__{orig}",
) -> Dict[str, int]:
  """按动作信息重命名样例目录（输出到新的 std 根）。

  过程：
  - 发现样例目录（支持 std 根 / videos / videos/*）。
  - 读取样例内所有 JSON，收集动作标签；取出现频次最高的动作作为该样例动作；若无则 "unknown"。
  - 目标目录结构：<out_root>/videos/<name_format(action, orig)>/
  - 复制（非移动）整个样例目录内容到新目录；若重名则追加 _dup{k}。

  返回统计：{"num_samples": N, "renamed": R}
  """
  sample_dirs = _discover_std_sample_folders(inputs)
  if not sample_dirs:
    return {"num_samples": 0, "renamed": 0}

  # 推断默认 out_root
  if not out_root:
    any_sample = sample_dirs[0]
    videos_dir = os.path.dirname(any_sample)
    std_root = os.path.dirname(videos_dir)
    out_root = std_root + "_renamed_by_action"

  videos_out = os.path.join(out_root, "videos")
  _ensure_dir(videos_out)

  def _copy_dir(src: str, dst: str) -> None:
    _ensure_dir(dst)
    for fname in os.listdir(src):
      sp = os.path.join(src, fname)
      dp = os.path.join(dst, fname)
      if os.path.isdir(sp):
        _copy_dir(sp, dp)
      else:
        shutil.copy2(sp, dp)

  renamed = 0
  from collections import Counter
  from tqdm import tqdm as _tqdm
  pbar = _tqdm(total=len(sample_dirs), desc="rename-by-action", unit="sample")
  for sample_dir in sample_dirs:
    orig = os.path.basename(os.path.normpath(sample_dir))

    # 收集动作频次
    cnt: Counter[str] = Counter()
    for json_path, rec in iter_labelme_dir(sample_dir):
      for sh in rec.get("shapes", []) or []:
        parsed = extract_bbox_and_action(sh)
        if parsed is None:
          continue
        _, act = parsed
        act_norm = _sanitize_for_dir(str(act or "").strip() or "unknown")
        cnt[act_norm] += 1

    if cnt:
      action = max(cnt.items(), key=lambda kv: kv[1])[0]
    else:
      action = "unknown"

    base_name = name_format.format(action=action, orig=orig)
    base_name = _sanitize_for_dir(base_name)

    # 确保唯一
    dst_name = base_name
    k = 1
    while os.path.exists(os.path.join(videos_out, dst_name)):
      dst_name = f"{base_name}_dup{k}"
      k += 1

    dst_dir = os.path.join(videos_out, dst_name)
    _copy_dir(sample_dir, dst_dir)
    renamed += 1

    try:
      pbar.update(1)
    except Exception:
      pass

  try:
    pbar.close()
  except Exception:
    pass

  return {"num_samples": len(sample_dirs), "renamed": renamed}

