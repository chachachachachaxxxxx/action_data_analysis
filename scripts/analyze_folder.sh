#!/usr/bin/env bash
# Ensure running under bash even if invoked via zsh/sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/analyze_folder.sh FOLDER [--out OUT_DIR]

Notes:
  - FOLDER: 包含帧图与 LabelMe JSON 的单个目录
  - 若指定 --out 则输出 stats.json 与 README_annotations.md 到该目录
USAGE
}

if [[ $# -lt 1 ]]; then usage; exit 1; fi

FOLDER="$1"; shift || true
OUT_DIR="/storage/wangxinxing/code/action_data_analysis/output/test_output"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  PYTHONPATH="${PYTHONPATH:-}:src" python -m action_data_analysis.cli.main analyze-folder "$FOLDER"
else
  PYTHONPATH="${PYTHONPATH:-}:src" python -m action_data_analysis.cli.main analyze-folder "$FOLDER" --out "$OUT_DIR"
fi

