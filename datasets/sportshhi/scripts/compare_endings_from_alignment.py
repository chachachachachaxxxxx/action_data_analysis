#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple


def _digits_from_name(name: str) -> Optional[int]:
    digits = ''.join([c for c in name if c.isdigit()])
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _build_index_map_from_dir(video_dir: str) -> Dict[int, str]:
    try:
        files = [f for f in os.listdir(video_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    except FileNotFoundError:
        return {}
    idx_map: Dict[int, str] = {}
    for fn in files:
        name, _ = os.path.splitext(fn)
        idx = _digits_from_name(name)
        if idx is None:
            continue
        if idx not in idx_map:
            idx_map[idx] = fn
    return idx_map


def _iter_common_forward(ms_map: Dict[int, str], sh_map: Dict[int, str], ms_start: int, sh_start: int):
    t = 0
    while True:
        ms_idx = ms_start + t
        sh_idx = sh_start + t
        if ms_idx not in ms_map or sh_idx not in sh_map:
            break
        yield t, ms_idx, sh_idx
        t += 1


def main() -> None:
    ap = argparse.ArgumentParser(description="统计：自对齐起始帧后，哪个数据集先结束（逐视频）")
    ap.add_argument("--alignment_csv", type=str, required=True)
    ap.add_argument("--multisports_root", type=str, required=True)
    ap.add_argument("--sportshhi_root", type=str, required=True)
    ap.add_argument("--allow_status", type=str, default="ok,no_below_threshold")
    ap.add_argument("--video-id", type=str, default="", help="仅统计指定 video_id；为空则处理 CSV 中全部允许行")
    ap.add_argument("--out_csv", type=str, default="", help="若提供，则写出逐视频统计结果")
    ap.add_argument("--counts_csv", type=str, default="", help="若提供，则写出聚合计数：who_more,extra_frames,count")
    args = ap.parse_args()

    allow = {s.strip() for s in (args.allow_status or "").split(",") if s.strip()}

    rows: List[Dict[str, str]] = []
    with open(args.alignment_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if allow and (row.get("status") or "").strip() not in allow:
                continue
            if args.video_id and (row.get("video_id") or "").strip() != args.video_id:
                continue
            rows.append(row)

    total = 0
    ms_end_first = 0
    sh_end_first = 0
    tie_or_unknown = 0
    # (video_id, who_first, common_len, ms_total_len, sh_total_len, ms_extra, sh_extra)
    per_video: List[Tuple[str, str, int, int, int, int, int]] = []

    for r in rows:
        video_id = (r.get("video_id") or "").strip()
        ms_file = (r.get("ms_file") or "").strip()
        sh_file = (r.get("sh_file") or "").strip()
        if not video_id or not ms_file or not sh_file:
            continue

        try:
            sh_start = int(r.get("sh_index") or 0)
        except Exception:
            sh_start = _digits_from_name(os.path.splitext(sh_file)[0]) or 0
        try:
            ms_start = int(r.get("ms_index") or 0)
        except Exception:
            ms_start = _digits_from_name(os.path.splitext(ms_file)[0]) or 0
        if not sh_start or not ms_start:
            continue

        ms_dir = os.path.join(args.multisports_root, video_id)
        sh_dir = os.path.join(args.sportshhi_root, video_id)
        ms_map = _build_index_map_from_dir(ms_dir)
        sh_map = _build_index_map_from_dir(sh_dir)
        if not ms_map or not sh_map:
            continue

        # 计算从起始帧往后的各自可用长度
        ms_len = 0
        while (ms_start + ms_len) in ms_map:
            ms_len += 1
        sh_len = 0
        while (sh_start + sh_len) in sh_map:
            sh_len += 1

        t = 0
        while True:
            ms_idx = ms_start + t
            sh_idx = sh_start + t
            ms_has = ms_idx in ms_map
            sh_has = sh_idx in sh_map
            if not ms_has and not sh_has:
                # 双方同时结束：若 t==0 则起始即无公共帧，视为 unknown；否则为 tie
                who = "unknown" if t == 0 else "tie"
                common_len = t
                total += 1
                if who == "unknown":
                    tie_or_unknown += 1
                else:
                    tie_or_unknown += 1
                ms_extra = max(0, ms_len - common_len)
                sh_extra = max(0, sh_len - common_len)
                per_video.append((video_id, who, common_len, ms_len, sh_len, ms_extra, sh_extra))
                break
            if not ms_has or not sh_has:
                # 有一方缺失，另一方存在
                who = "ms" if (not ms_has and sh_has) else ("sh" if (not sh_has and ms_has) else "tie")
                common_len = t  # 已有公共长度
                total += 1
                if who == "ms":
                    ms_end_first += 1
                elif who == "sh":
                    sh_end_first += 1
                else:
                    tie_or_unknown += 1
                ms_extra = (ms_len - common_len) if who == "sh" else 0
                sh_extra = (sh_len - common_len) if who == "ms" else 0
                # 若为 tie/unknown，上面分支已处理，这里不会出现 tie（理论上）
                per_video.append((video_id, who, common_len, ms_len, sh_len, ms_extra, sh_extra))
                break
            t += 1

    # 汇总均值
    def _avg(vals: List[int]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0

    ms_extra_all = [rec[5] for rec in per_video]
    sh_extra_all = [rec[6] for rec in per_video]
    avg_ms_extra = _avg(ms_extra_all)
    avg_sh_extra = _avg(sh_extra_all)

    # 统计“各自比对方多出多少帧”的分布
    from collections import defaultdict
    ms_more_counts: Dict[int, int] = defaultdict(int)
    sh_more_counts: Dict[int, int] = defaultdict(int)
    for vid, who, clen, mlen, slen, mex, sex in per_video:
        diff = int(abs(mlen - slen))
        if diff <= 0:
            continue
        if mlen > slen:
            ms_more_counts[diff] += 1
        elif slen > mlen:
            sh_more_counts[diff] += 1

    def _to_sorted_dict(d: Dict[int, int]) -> Dict[str, int]:
        return {str(k): int(d[k]) for k in sorted(d.keys())}

    print(
        "{"
        + f"\n  \"videos_considered\": {total},"
        + f"\n  \"ms_end_first\": {ms_end_first},"
        + f"\n  \"sh_end_first\": {sh_end_first},"
        + f"\n  \"tie_or_unknown\": {tie_or_unknown},"
        + f"\n  \"avg_ms_extra\": {avg_ms_extra:.4f},"
        + f"\n  \"avg_sh_extra\": {avg_sh_extra:.4f},"
        + f"\n  \"ms_more_counts\": { _to_sorted_dict(ms_more_counts) },"
        + f"\n  \"sh_more_counts\": { _to_sorted_dict(sh_more_counts) }\n"
        + "}"
    )

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "who_end_first", "common_length", "ms_total_length", "sh_total_length", "ms_extra_frames", "sh_extra_frames"])
            for vid, who, clen, mlen, slen, mex, sex in per_video:
                w.writerow([vid, who, int(clen), int(mlen), int(slen), int(mex), int(sex)])

    if args.counts_csv:
        os.makedirs(os.path.dirname(args.counts_csv), exist_ok=True)
        with open(args.counts_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["who_more", "extra_frames", "count"])
            for k in sorted(ms_more_counts.keys()):
                w.writerow(["ms", int(k), int(ms_more_counts[k])])
            for k in sorted(sh_more_counts.keys()):
                w.writerow(["sh", int(k), int(sh_more_counts[k])])


if __name__ == "__main__":
    main()


