#!/usr/bin/env bash
# Ensure running under bash even if invoked via zsh/sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/export_samples.sh --out OUT_DIR --dataset-name NAME [--per-class N] [--context K] DIR1 [DIR2 ...]

Notes:
  - 为每个类别随机采样 N 个样例，导出中心帧及其前后 K 帧（总共 2K+1），不足则用全部帧
  - 复制原始 JSON 与图像，不进行绘制
USAGE
}

OUT_DIR=""
DATASET_NAME=""
PER_CLASS="3"
CONTEXT="6"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2;;
    --dataset-name) DATASET_NAME="$2"; shift 2;;
    --per-class) PER_CLASS="$2"; shift 2;;
    --context) CONTEXT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) ARGS+=("$1"); shift;;
  esac
done

if [[ -z "$OUT_DIR" || -z "$DATASET_NAME" || ${#ARGS[@]} -eq 0 ]]; then usage; exit 1; fi

PYTHONPATH="${PYTHONPATH:-}:src" python -m action_data_analysis.cli.main export-samples "${ARGS[@]}" --out "$OUT_DIR" --dataset-name "$DATASET_NAME" --per-class "$PER_CLASS" --context "$CONTEXT"

