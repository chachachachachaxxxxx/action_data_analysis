#!/usr/bin/env python3
import os
import json
from typing import Any, Dict, List, Tuple, Optional

from action_data_analysis.io.json import iter_labelme_dir


REPO_ROOT = "/storage/wangxinxing/code/action_data_analysis"
EXPORT_ROOT = os.path.join(REPO_ROOT, "data", "MultiSports_json")
RESULTS_DIR = os.path.join(REPO_ROOT, "datasets", "multisports", "results")


def _parse_frame_index_from_name(fname: str) -> Optional[int]:
    name, _ = os.path.splitext(fname)
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


def _autodetect_stride(frame_indices: List[int]) -> int:
    if not frame_indices or len(frame_indices) < 2:
        return 1
    from collections import Counter
    diffs: Counter[int] = Counter()
    prev = frame_indices[0]
    for idx in frame_indices[1:]:
        d = idx - prev
        if d > 0:
            diffs[d] += 1
        prev = idx
    if not diffs:
        return 1
    return max(diffs.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def analyze_video_folder(video_dir: str) -> Dict[str, Any]:
    # pair key = (group_id_or_track, action)
    frames_by_pair: Dict[Tuple[str, str], List[int]] = {}
    total_json = 0
    for json_path, rec in iter_labelme_dir(video_dir):
        total_json += 1
        fidx = _parse_frame_index_from_name(os.path.basename(json_path))
        if fidx is None:
            # fallback: sequence index
            fidx = total_json
        for sh in rec.get("shapes", []) or []:
            action = ""
            attrs = sh.get("attributes") or {}
            if isinstance(attrs, dict) and isinstance(attrs.get("action"), str):
                action = attrs.get("action", "")
            if not action:
                flags = sh.get("flags") or {}
                if isinstance(flags, dict) and isinstance(flags.get("action"), str):
                    action = flags.get("action", "")
            gid = sh.get("group_id")
            tid = None
            if isinstance(attrs, dict):
                for key in ("track_id", "id", "player_id"):
                    v = attrs.get(key)
                    if isinstance(v, (str, int)):
                        tid = str(v)
                        break
            if tid is None:
                if isinstance(gid, (str, int)):
                    tid = str(gid)
                else:
                    # ignore shapes without any id
                    continue
            key = (tid, action)
            frames_by_pair.setdefault(key, []).append(int(fidx))

    # Check gaps per pair with local stride
    tubes_total = 0
    tubes_with_gaps = 0
    gap_examples: List[Dict[str, Any]] = []
    for (tid, action), flist in frames_by_pair.items():
        if not flist:
            continue
        flist.sort()
        local_stride = _autodetect_stride(flist)
        segments = 1
        has_gap = False
        for i in range(1, len(flist)):
            if flist[i] - flist[i - 1] > local_stride:
                segments += 1
                has_gap = True
        tubes_total += segments
        if has_gap:
            tubes_with_gaps += 1
            if len(gap_examples) < 20:
                gap_examples.append({
                    "track_id": tid,
                    "action": action,
                    "frames": flist[:20],
                    "local_stride": local_stride,
                })

    return {
        "video": os.path.basename(video_dir),
        "num_pairs": len(frames_by_pair),
        "tubes_total": tubes_total,
        "tubes_with_gaps": tubes_with_gaps,
        "gap_examples": gap_examples,
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # discover leaf video dirs
    if not os.path.isdir(EXPORT_ROOT):
        raise SystemExit(f"Not a directory: {EXPORT_ROOT}")
    leaf_dirs: List[str] = []
    for name in sorted(os.listdir(EXPORT_ROOT)):
        d = os.path.join(EXPORT_ROOT, name)
        if os.path.isdir(d):
            leaf_dirs.append(d)

    reports: List[Dict[str, Any]] = []
    total_tubes = 0
    total_pairs = 0
    total_gap_tubes = 0
    for i, d in enumerate(leaf_dirs):
        if i % 50 == 0:
            print(f"Analyzing {i+1}/{len(leaf_dirs)}: {os.path.basename(d)}")
        r = analyze_video_folder(d)
        reports.append(r)
        total_tubes += int(r["tubes_total"])
        total_pairs += int(r["num_pairs"])
        total_gap_tubes += int(r["tubes_with_gaps"])

    summary = {
        "root": EXPORT_ROOT,
        "num_videos": len(leaf_dirs),
        "total_pairs": total_pairs,
        "total_tubes": total_tubes,
        "tubes_with_gaps": total_gap_tubes,
        "has_gaps": total_gap_tubes > 0,
    }

    out_json = os.path.join(RESULTS_DIR, "exported_tubes_check.json")
    out_md = os.path.join(RESULTS_DIR, "README_exported_tubes_check.md")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "videos": reports}, f, ensure_ascii=False, indent=2)

    lines: List[str] = []
    lines.append("# MultiSports_json 导出数据的 tube 统计与跳帧检测")
    lines.append("")
    lines.append(f"- num_videos: {summary['num_videos']}")
    lines.append(f"- total_pairs(以id+action聚合): {summary['total_pairs']}")
    lines.append(f"- total_tubes(按局部stride分段): {summary['total_tubes']}")
    lines.append(f"- tubes_with_gaps: {summary['tubes_with_gaps']}")
    lines.append(f"- has_gaps: {summary['has_gaps']}")
    lines.append("")
    if summary["has_gaps"]:
        lines.append("## 示例（最多20条）")
        for r in reports:
            if r.get("tubes_with_gaps", 0) > 0:
                lines.append(f"- {r['video']}: gaps={r['tubes_with_gaps']}")
                for ex in r.get("gap_examples", [])[:3]:
                    lines.append(f"  - id={ex['track_id']} action={ex['action']} stride={ex['local_stride']} frames(head)={ex['frames']}")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote:")
    print("-", out_json)
    print("-", out_md)


if __name__ == "__main__":
    main()


