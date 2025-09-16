#!/usr/bin/env bash
# Ensure running under bash even if invoked via zsh/sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/analyze_overlaps_sportshhi.sh \
    --sportshhi-root /abs/path/to/SportsHHI_main_json \
    --out /abs/path/to/output/SportsHHI_main_analysis \
    [--thresholds "0.1 0.3 0.5"]

Notes:
  - 调用 CLI 子命令 analyze-overlaps，遍历叶子目录（含 .json 的帧目录）统计 IoU 重合情况。
  - 默认阈值为 0.1 0.3 0.5，可通过 --thresholds 覆盖，空格分隔。
USAGE
}

SPORTSHHI_ROOT="/storage/wangxinxing/code/action_data_analysis/data/SportsHHI_main_json"
OUT_DIR="/storage/wangxinxing/code/action_data_analysis/output/SportsHHI_main_analysis"
THRESHOLDS="0.99"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sportshhi-root) SPORTSHHI_ROOT="$2"; shift 2;;
    --out) OUT_DIR="$2"; shift 2;;
    --thresholds) THRESHOLDS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$SPORTSHHI_ROOT" || -z "$OUT_DIR" ]]; then
  usage; exit 1
fi

mkdir -p "$OUT_DIR"

# 组装 thresholds 数组参数
read -r -a TH_ARR <<< "$THRESHOLDS"

echo "Analyzing overlaps in: $SPORTSHHI_ROOT"
echo "Output dir: $OUT_DIR"
echo "Thresholds: ${TH_ARR[*]}"

# 调用 Python CLI
PYTHONPATH="${PYTHONPATH:-}:src" python -m action_data_analysis.cli.main analyze-overlaps \
  "$SPORTSHHI_ROOT" \
  --out "$OUT_DIR" \
  --thresholds ${TH_ARR[@]}

echo "Done. Results written to: $OUT_DIR/overlaps.json and $OUT_DIR/README_overlaps.md"


