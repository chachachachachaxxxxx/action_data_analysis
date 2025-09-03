#!/usr/bin/env bash
# Ensure running under bash even if invoked via zsh/sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/visualize_samples.sh --out OUT_DIR [--per-class N] [--context N] DIR1 [DIR2 ...]

Notes:
  - 为每个类别随机采样若干帧作为样例，并导出前后 N 帧
USAGE
}

OUT_DIR=""
PER_CLASS="3"
CONTEXT="25"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2;;
    --per-class) PER_CLASS="$2"; shift 2;;
    --context) CONTEXT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) ARGS+=("$1"); shift;;
  esac
done

if [[ -z "$OUT_DIR" || ${#ARGS[@]} -eq 0 ]]; then usage; exit 1; fi

PYTHONPATH="${PYTHONPATH:-}:src" python -m action_data_analysis.cli.main visualize-samples "${ARGS[@]}" --out "$OUT_DIR" --per-class "$PER_CLASS" --context "$CONTEXT"

