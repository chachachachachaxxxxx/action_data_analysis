from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import os
import shutil

from action_data_analysis.analyze.export import _discover_std_sample_folders
from action_data_analysis.io.json import iter_labelme_dir, extract_bbox_and_action
from action_data_analysis.analyze.stats import compute_labelme_folder_stats
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SubsetSummary:
  source_roots: List[str]
  output_root: str
  actions: List[str]
  per_class_required: int
  samples_total: int
  samples_selected: int
  per_action_counts: Dict[str, int]
  missing_actions: List[str]
  per_sample_action_tubes: Dict[str, Dict[str, int]]  # key: 样例名（目录名），值：{action: tubes}


def _ensure_dir(path: Path) -> None:
  path.mkdir(parents=True, exist_ok=True)


def _collect_actions_for_sample(sample_dir: str) -> Set[str]:
  """扫描单个样例目录，返回该样例中出现过的动作集合。"""
  actions: Set[str] = set()
  for _json_path, rec in iter_labelme_dir(sample_dir):
    for sh in rec.get("shapes", []) or []:
      parsed = extract_bbox_and_action(sh)
      if parsed is None:
        continue
      _bbox, action = parsed
      if action:
        actions.add(action)
  return actions


def _copy_sample_dir(src: Path, dst: Path) -> Tuple[int, int]:
  """复制一个样例目录：复制所有图片文件与全部 JSON/其他文件，保留相对子结构。

  返回 (images_copied, files_copied)
  """
  images = 0
  files = 0
  for root, _dirs, fnames in os.walk(src):
    rel = Path(root).relative_to(src)
    out_sub = (dst / rel)
    out_sub.mkdir(parents=True, exist_ok=True)
    for fname in fnames:
      sp = Path(root) / fname
      dp = out_sub / fname
      if dp.exists():
        continue
      shutil.copy2(sp, dp)
      files += 1
      if sp.suffix.lower() in IMG_EXTS:
        images += 1
  return images, files


def _compute_per_action_tubes_for_sample(sample_dir: str) -> Dict[str, int]:
  """基于统计实现，计算单个样例目录内按动作的 tube 数量。"""
  stats = compute_labelme_folder_stats(sample_dir)
  tubes = stats.get("spatiotemporal_tubes", {}).get("per_action", []) or []
  result: Dict[str, int] = {}
  for name, cnt in tubes:
    try:
      result[str(name)] = int(cnt)
    except Exception:
      continue
  return result


def _greedy_cover_samples(
  sample_to_actions: Dict[str, Set[str]],
  per_class_required: int,
) -> List[str]:
  """贪心选择样例：尽量覆盖每个动作至少 `per_class_required` 个样例。

  策略：
  - 维护每个动作的已选计数 needed[action] = per_class_required
  - 反复选择能最大化减少剩余需求总和的样例；若并列，优先动作多者，再按样例名排序；
  - 直到所有动作满足或无增益。
  """
  # 统计完整的动作集合
  all_actions: Set[str] = set()
  for acts in sample_to_actions.values():
    all_actions.update(acts)
  if not all_actions:
    return []

  remaining: Dict[str, int] = {a: per_class_required for a in sorted(all_actions)}
  selected: List[str] = []
  unused: Set[str] = set(sample_to_actions.keys())

  def _gain(sample: str) -> int:
    g = 0
    for a in sample_to_actions.get(sample, set()):
      need = remaining.get(a, 0)
      if need > 0:
        g += 1  # 该样例对该动作贡献 1
    return g

  while unused:
    # 选择增益最大的样例
    candidates = sorted(list(unused))
    best: Optional[str] = None
    best_gain = 0
    best_card = -1
    for s in candidates:
      g = _gain(s)
      if g > best_gain or (g == best_gain and len(sample_to_actions.get(s, set())) > best_card) or (
        g == best_gain and len(sample_to_actions.get(s, set())) == best_card and (best is None or s < best)
      ):
        best = s
        best_gain = g
        best_card = len(sample_to_actions.get(s, set()))
    if best is None or best_gain == 0:
      break
    # 选中 best
    selected.append(best)
    unused.remove(best)
    for a in sample_to_actions.get(best, set()):
      if remaining.get(a, 0) > 0:
        remaining[a] -= 1
    # 检查是否已经满足全部
    if all(v <= 0 for v in remaining.values()):
      break

  # 若仍有未满足动作，尝试追加：为每个仍缺的动作，找任一包含该动作且未选中的样例
  for a, need in remaining.items():
    if need <= 0:
      continue
    # 首先从未使用中挑选，若没有则允许使用已选（无实际增益，但保持输出稳定）
    pool = [s for s in sample_to_actions.keys() if a in sample_to_actions[s]]
    for s in sorted(pool):
      if need <= 0:
        break
      if s not in selected:
        selected.append(s)
        need -= 1
    remaining[a] = need

  return selected


def create_std_subset(
  inputs: List[str],
  out_root: str | Path | None,
  per_class: int = 3,
) -> SubsetSummary:
  """基于 std 数据集目录（根/videos/或具体样例）创建子集：保证每个动作至少 per_class 个样例。

  - inputs: std 根、videos 目录或 videos/* 样例目录；会自动发现样例。
  - out_root: 输出根目录；若为空，默认在首个 std 根旁创建 `<name>_subset`。
  - per_class: 每类至少样例数。
  """
  # 发现样例目录
  folders: List[str] = _discover_std_sample_folders(inputs)
  if not folders:
    raise ValueError("No std sample folders discovered from inputs")

  # 推断默认 out_root
  if out_root is None or str(out_root) == "":
    # 取第一个样例向上两级到 std 根（.../videos/<sample>）
    p = Path(folders[0])
    std_root = p.parent.parent
    out_dir = std_root.parent / f"{std_root.name}_subset"
  else:
    out_dir = Path(out_root)
  videos_out = out_dir / "videos"
  stats_out = out_dir / "stats"
  _ensure_dir(videos_out)
  _ensure_dir(stats_out)

  # 为每个样例统计动作集合
  sample_to_actions: Dict[str, Set[str]] = {}
  pbar_scan = tqdm(total=len(folders), desc="scan samples", unit="sample")
  for sdir in folders:
    acts = _collect_actions_for_sample(sdir)
    if acts:
      sample_to_actions[sdir] = acts
    try:
      pbar_scan.update(1)
    except Exception:
      pass
  try:
    pbar_scan.close()
  except Exception:
    pass

  # 收集全体动作
  all_actions: Set[str] = set()
  for acts in sample_to_actions.values():
    all_actions.update(acts)

  # 贪心挑选样例
  chosen: List[str] = _greedy_cover_samples(sample_to_actions, per_class_required=int(per_class))

  # 复制所选样例目录
  per_action_counts: Dict[str, int] = {a: 0 for a in sorted(all_actions)}
  per_sample_action_tubes: Dict[str, Dict[str, int]] = {}
  pbar_copy = tqdm(total=len(chosen), desc="copy subset", unit="sample")
  for s in chosen:
    src = Path(s)
    dst = videos_out / src.name
    imgs, _files = _copy_sample_dir(src, dst)
    # 计数：该样例包含的动作集合都 +1（样例级计数）
    for a in sample_to_actions.get(s, set()):
      per_action_counts[a] = per_action_counts.get(a, 0) + 1
    # 统计：该样例内每个动作的 tube 数
    per_sample_action_tubes[src.name] = _compute_per_action_tubes_for_sample(s)
    try:
      pbar_copy.update(1)
    except Exception:
      pass
  try:
    pbar_copy.close()
  except Exception:
    pass

  missing: List[str] = [a for a, cnt in per_action_counts.items() if cnt < int(per_class)]

  return SubsetSummary(
    source_roots=[str(Path(p).resolve()) for p in inputs],
    output_root=str(out_dir.resolve()),
    actions=sorted(all_actions),
    per_class_required=int(per_class),
    samples_total=len(folders),
    samples_selected=len(chosen),
    per_action_counts=per_action_counts,
    missing_actions=sorted(missing),
    per_sample_action_tubes=per_sample_action_tubes,
  )


