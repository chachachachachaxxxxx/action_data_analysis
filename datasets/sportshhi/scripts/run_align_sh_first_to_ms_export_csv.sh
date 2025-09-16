#!/usr/bin/env bash
# Ensure running under bash even if invoked via zsh/sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  datasets/sportshhi/scripts/run_align_sh_first_to_ms_export_csv.sh \
    [--multisports-root /abs/path/to/MultiSports_json] \
    [--sportshhi-root   /abs/path/to/SportsHHI_json] \
    [--out-csv          /abs/path/to/alignment_search.csv] \
    [--max-k 20] [--threshold 3.0]
USAGE
}

MULTISPORTS_ROOT="/storage/wangxinxing/code/action_data_analysis/data/MultiSports_json"
SPORTSHHI_ROOT="/storage/wangxinxing/code/action_data_analysis/data/SportsHHI_json"
OUT_CSV="/storage/wangxinxing/code/action_data_analysis/output/start_refine/alignment_search.csv"
MAX_K="0"
THRESHOLD="3.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --multisports-root) MULTISPORTS_ROOT="$2"; shift 2;;
    --sportshhi-root)   SPORTSHHI_ROOT="$2"; shift 2;;
    --out-csv)          OUT_CSV="$2"; shift 2;;
    --max-k)            MAX_K="$2"; shift 2;;
    --threshold)        THRESHOLD="$2"; shift 2;;
    -h|--help)          usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

mkdir -p "$(dirname "$OUT_CSV")"

echo "MultiSports root: $MULTISPORTS_ROOT"
echo "SportsHHI root:   $SPORTSHHI_ROOT"
echo "Out CSV:          $OUT_CSV"
echo "Max K:            $MAX_K"
echo "Threshold:        $THRESHOLD"

PYTHONPATH="${PYTHONPATH:-}:src" python -u \
  /storage/wangxinxing/code/action_data_analysis/datasets/sportshhi/scripts/align_sh_first_to_ms_export_csv.py \
  --multisports_root "$MULTISPORTS_ROOT" \
  --sportshhi_root "$SPORTSHHI_ROOT" \
  --out_csv "$OUT_CSV" \
  --max-k "$MAX_K" \
  --threshold "$THRESHOLD"

echo "Done. CSV -> $OUT_CSV"


