#!/usr/bin/env bash
# 一键启动：自动激活 conda 环境并运行检测程序
#
#   ./run.sh          # 默认摄像头 0
#   ./run.sh 1        # 使用 1 号摄像头（外接 USB 摄像头）
#
# 参数原样传给 detect.py。可用 CONDA_ENV_NAME 指定环境名，默认 task。

set -euo pipefail
cd "$(dirname "$0")"

# 定位 conda 安装目录：优先问 conda 自己，其次常见安装路径
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
else
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
        if [ -f "$d/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$d"
            break
        fi
    done
fi
: "${CONDA_BASE:?未找到 conda，请先安装 Miniconda 或修改本脚本的 CONDA_BASE}"

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_NAME="${CONDA_ENV_NAME:-task}"
conda activate "$ENV_NAME"

# 消除 OpenCV 5 的 Qt 字体告警（不影响画面，仅美化窗口）
FONT_DIR="$(python -c 'import cv2,os;print(os.path.join(os.path.dirname(cv2.__file__),"qt","fonts"))' 2>/dev/null || true)"
DEJAVU="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
if [ -n "$FONT_DIR" ] && [ -f "$DEJAVU" ] && [ ! -e "$FONT_DIR/DejaVuSans.ttf" ]; then
    mkdir -p "$FONT_DIR" 2>/dev/null && ln -sf "$DEJAVU" "$FONT_DIR/DejaVuSans.ttf" 2>/dev/null || true
fi

exec python detect.py "$@"
