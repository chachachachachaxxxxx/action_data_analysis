from __future__ import annotations

import os
import shutil
from typing import Iterable, List, Tuple
import json

from action_data_analysis.analyze.visual import sample_per_class_examples


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

