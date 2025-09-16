#!/usr/bin/env bash
# Ensure running under bash even if invoked via zsh/sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/align_merge_multisports_sportshhi.sh \
    [--multisports-root /abs/path/to/MultiSports_json] \
    [--sportshhi-root   /abs/path/to/SportsHHI_json] \
    [--out              /abs/path/to/MultiSports_SportsHHI_overlap_json] \
    [--iou 0.99] [--limit-videos N]

Notes:
  - 对齐 MultiSports 与 SportsHHI 的公共视频帧（按 stride 比例），合并两侧 LabelMe JSON，统计 IoU≥阈值 的匹配。
  - 合并 JSON 写入到 --out/<video_id>/*.json；统计写到 --out/__stats__/iou_counts.{json,csv}
USAGE
}

MULTISPORTS_ROOT="/storage/wangxinxing/code/action_data_analysis/data/MultiSports_json"
SPORTSHHI_ROOT="/storage/wangxinxing/code/action_data_analysis/data/SportsHHI_json"
OUT_DIR="/storage/wangxinxing/code/action_data_analysis/data/MultiSports_SportsHHI_overlap_json"
IOU="0.99"
LIMIT_VIDEOS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --multisports-root) MULTISPORTS_ROOT="$2"; shift 2;;
    --sportshhi-root)   SPORTSHHI_ROOT="$2"; shift 2;;
    --out)              OUT_DIR="$2"; shift 2;;
    --iou)              IOU="$2"; shift 2;;
    --limit-videos)     LIMIT_VIDEOS="$2"; shift 2;;
    -h|--help)          usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

mkdir -p "$OUT_DIR"

echo "MultiSports root: $MULTISPORTS_ROOT"
echo "SportsHHI root:   $SPORTSHHI_ROOT"
echo "Output dir:       $OUT_DIR"
echo "IoU threshold:    $IOU"
if [[ -n "$LIMIT_VIDEOS" ]]; then echo "Limit videos:     $LIMIT_VIDEOS"; fi

LIMIT_ARG=()
if [[ -n "$LIMIT_VIDEOS" ]]; then
  LIMIT_ARG=(--limit_videos "$LIMIT_VIDEOS")
fi

PYTHONPATH="${PYTHONPATH:-}:src" python -u \
  /storage/wangxinxing/code/action_data_analysis/datasets/sportshhi/scripts/align_merge_and_compare.py \
  --multisports_root "$MULTISPORTS_ROOT" \
  --sportshhi_root "$SPORTSHHI_ROOT" \
  --out_root "$OUT_DIR" \
  --iou "$IOU" \
  "${LIMIT_ARG[@]}"

echo "Done. Merged JSON at: $OUT_DIR; Stats at: $OUT_DIR/__stats__/"


