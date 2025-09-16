#!/usr/bin/env bash
# Ensure running under bash even if invoked via zsh/sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/export_merged_samples_sportshhi.sh \
    --sportshhi-root /abs/path/to/SportsHHI_json \
    --out-json /abs/path/to/output/json \
    --out-merged /abs/path/to/output/merged \
    [--per-class 3] [--context 6]

Notes:
  - 每类采样 N=3，前后各扩展 K=6 帧（总 13 帧），不足则用全部
  - 先导出到 output/json/SportsHHI，再平铺合并到 output/merged
USAGE
}

SPORTSHHI_ROOT=""
OUT_JSON="/storage/wangxinxing/code/action_data_analysis/output/json"
OUT_MERGED="/storage/wangxinxing/code/action_data_analysis/output/merged"
PER_CLASS="3"
CONTEXT="6"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sportshhi-root) SPORTSHHI_ROOT="$2"; shift 2;;
    --out-json) OUT_JSON="$2"; shift 2;;
    --out-merged) OUT_MERGED="$2"; shift 2;;
    --per-class) PER_CLASS="$2"; shift 2;;
    --context) CONTEXT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$SPORTSHHI_ROOT" ]]; then usage; exit 1; fi

mkdir -p "$OUT_JSON" "$OUT_MERGED"

# 1) SportsHHI
scripts/export_samples.sh \
  --out "$OUT_JSON" \
  --dataset-name SportsHHI \
  --per-class "$PER_CLASS" \
  --context "$CONTEXT" \
  "$SPORTSHHI_ROOT"

# 2) Merge flatten (single dataset)
PYTHONPATH="${PYTHONPATH:-}:src" python -m action_data_analysis.cli.main merge-flatten \
  "$OUT_JSON"/SportsHHI \
  --out "$OUT_MERGED"

echo "Done. Exported SportsHHI to $OUT_JSON and merged to $OUT_MERGED"


