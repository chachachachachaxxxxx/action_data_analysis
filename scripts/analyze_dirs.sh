#!/usr/bin/env bash
# Ensure running under bash even if invoked via zsh/sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/analyze_dirs.sh --out OUT_DIR DIR1 [DIR2 ...]

Notes:
  - 统计多个目录并输出聚合结果到 OUT_DIR
USAGE
}

OUT_DIR="/storage/wangxinxing/code/action_data_analysis/output/test_output2"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) ARGS+=("$1"); shift;;
  esac
done

if [[ -z "$OUT_DIR" || ${#ARGS[@]} -eq 0 ]]; then usage; exit 1; fi

PYTHONPATH="${PYTHONPATH:-}:src" python -m action_data_analysis.cli.main analyze-dirs "${ARGS[@]}" --out "$OUT_DIR"

