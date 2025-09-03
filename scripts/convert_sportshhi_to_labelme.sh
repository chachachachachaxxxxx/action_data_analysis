#!/usr/bin/env bash
# Ensure running under bash even if invoked via zsh/sh
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

# 轻量封装 sportshhi/scripts/convert_to_labelme.sh，透传参数

ROOT="/storage/wangxinxing/code/action_data_analysis"
"$ROOT"/sportshhi/scripts/convert_to_labelme.sh "$@"

