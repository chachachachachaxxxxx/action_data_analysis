#!/usr/bin/env bash
# Ensure running under bash even if invoked via zsh/sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

# Default paths (edit if needed)
RAW_DEFAULT="/storage/wangxinxing/code/action_data_analysis/data/SportsHHI/rawframes"
ANN_DEFAULT="/storage/wangxinxing/code/action_data_analysis/sportshhi/basketball_annotations"
OUT_DEFAULT="/storage/wangxinxing/code/action_data_analysis/data/SportsHHI_main"

usage() {
  cat <<'USAGE'
Usage:
  sportshhi/scripts/convert_to_labelme.sh \
    --rawframes SRC_DIR \
    --out OUT_DIR \
    [--ann ANN_DIR] \
    [--limit_videos N] [--limit_frames N] \
    [--balanced_per_action N] \
    [--include_secondary] \
    [--filter_video vid1,vid2,...]

Notes:
  - By default, uses:
      RAW=${RAW_DEFAULT}
      ANN=${ANN_DEFAULT}
      OUT=${OUT_DEFAULT}
  - Logs are written under OUT as convert_YYYYmmdd_HHMMSS.log
USAGE
}

RAW="$RAW_DEFAULT"
ANN="$ANN_DEFAULT"
OUT="$OUT_DEFAULT"
# Store values; build argv array later to avoid word-splitting issues across shells
LIMIT_VIDEOS_VAL=""
LIMIT_FRAMES_VAL=""
BALANCED_VAL=""
INCLUDE_SECONDARY_FLAG=0
FILTER_VIDEO_VAL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rawframes) RAW="$2"; shift 2;;
    --ann) ANN="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --limit_videos) LIMIT_VIDEOS_VAL="$2"; shift 2;;
    --limit_frames) LIMIT_FRAMES_VAL="$2"; shift 2;;
    --balanced_per_action) BALANCED_VAL="$2"; shift 2;;
    --include_secondary) INCLUDE_SECONDARY_FLAG=1; shift 1;;
    --filter_video) FILTER_VIDEO_VAL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

mkdir -p "$OUT"
LOG="$OUT/convert_$(date +%Y%m%d_%H%M%S).log"
echo "RAW=$RAW" | tee "$LOG"
echo "ANN=$ANN" | tee -a "$LOG"
echo "OUT=$OUT" | tee -a "$LOG"

if [[ ! -d "$RAW" ]]; then echo "ERROR: RAW not found: $RAW" | tee -a "$LOG"; exit 1; fi
if [[ ! -d "$ANN" ]]; then echo "ERROR: ANN not found: $ANN" | tee -a "$LOG"; exit 1; fi

echo "Starting conversion..." | tee -a "$LOG"

# Build argv array for python call (robust across shells)
PY_ARGS=(
  /storage/wangxinxing/code/action_data_analysis/sportshhi/scripts/convert_to_labelme_json.py
  --rawframes_src "$RAW"
  --ann_dir "$ANN"
  --out_root "$OUT"
)
if [ -n "$LIMIT_VIDEOS_VAL" ]; then PY_ARGS+=( --limit_videos "$LIMIT_VIDEOS_VAL" ); fi
if [ -n "$LIMIT_FRAMES_VAL" ]; then PY_ARGS+=( --limit_frames "$LIMIT_FRAMES_VAL" ); fi
if [ -n "$BALANCED_VAL" ]; then PY_ARGS+=( --balanced_per_action "$BALANCED_VAL" ); fi
if [ $INCLUDE_SECONDARY_FLAG -eq 1 ]; then PY_ARGS+=( --include_secondary ); fi
if [ -n "$FILTER_VIDEO_VAL" ]; then PY_ARGS+=( --filter_video "$FILTER_VIDEO_VAL" ); fi

python "${PY_ARGS[@]}" | tee -a "$LOG"

echo "Done. See log: $LOG" | tee -a "$LOG"

