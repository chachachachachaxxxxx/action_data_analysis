#!/usr/bin/env bash
set -euo pipefail

show_help() {
	cat <<'EOF'
用法：stats.sh <STD_ROOT> [--out <OUT_DIR>] [--spf <SECONDS_PER_FRAME>] [--skip-dirs] [--skip-plot] [--skip-time]

- STD_ROOT: 标准数据集根目录（需包含 videos/ 与 stats/）
- --out: 输出目录（默认：<STD_ROOT>/stats）
- --spf: 每帧秒数，用于时长分箱（默认：0.04）
- --skip-dirs: 跳过目录聚合统计
- --skip-plot: 跳过 Tube 长度分箱图
- --skip-time: 跳过时长分箱（秒）
EOF
}

if [[ ${1-} == "-h" || ${1-} == "--help" || $# -eq 0 ]]; then
	show_help
	exit 0
fi

STD_ROOT="$1"; shift || true

OUT_DIR=""
SPF="0.04"
SKIP_DIRS="0"
SKIP_PLOT="0"
SKIP_TIME="0"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--out)
			OUT_DIR="$2"; shift 2;;
		--spf)
			SPF="$2"; shift 2;;
		--skip-dirs)
			SKIP_DIRS="1"; shift;;
		--skip-plot)
			SKIP_PLOT="1"; shift;;
		--skip-time)
			SKIP_TIME="1"; shift;;
		*)
			echo "未知参数: $1"; show_help; exit 1;;
	 esac
done

if command -v realpath >/dev/null 2>&1; then
	STD_ROOT="$(realpath "$STD_ROOT")"
else
	STD_ROOT="$(python - <<'PY'
import os, sys
print(os.path.realpath(sys.argv[1]))
PY
"$STD_ROOT")"
fi

if [[ -z "${OUT_DIR}" ]]; then
	OUT_DIR="${STD_ROOT}/stats"
fi
mkdir -p "${OUT_DIR}"

if [[ ! -d "${STD_ROOT}/videos" ]]; then
	echo "错误：未找到目录 ${STD_ROOT}/videos"
	exit 1
fi

# 保证可通过源码运行（未安装为包时）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH-}"

echo "STD_ROOT: ${STD_ROOT}"
echo "OUTPUT : ${OUT_DIR}"
echo "SPF    : ${SPF}"

# 1) 目录聚合统计
if [[ "${SKIP_DIRS}" == "0" ]]; then
	echo "[1/3] 目录聚合统计 → ${OUT_DIR}"
	python -m action_data_analysis.cli.main analyze-dirs "${STD_ROOT}" --out "${OUT_DIR}"
fi

# 2) Tube 长度分箱图
if [[ "${SKIP_PLOT}" == "0" ]]; then
	echo "[2/3] Tube 长度分箱图 → ${OUT_DIR}/tube_lengths.png"
	python -m action_data_analysis.cli.main plot-tube-lengths "${STD_ROOT}" --out "${OUT_DIR}/tube_lengths.png"
fi

# 3) 时长分箱（秒）
if [[ "${SKIP_TIME}" == "0" ]]; then
	echo "[3/3] 时长分箱（秒） → ${OUT_DIR}/time_bins (spf=${SPF})"
	python -m action_data_analysis.cli.main time-bins "${STD_ROOT}" --spf "${SPF}" --out "${OUT_DIR}/time_bins"
fi

echo "完成。输出位于：${OUT_DIR}"


