#!/usr/bin/env bash
# 一键启动：自动激活 conda 环境 task，再运行检测程序
#
#   ./run.sh                     # 默认摄像头
#   ./run.sh --source 1          # 第二个摄像头
#   ./run.sh --model yolo11s.pt  # 换模型
#   ./run.sh --no-show           # 无窗口模式
#
# 所有参数原样传给 detect.py

set -euo pipefail
cd "$(dirname "$0")"

# 载入 conda，让 conda activate 在非交互 shell 里也能用
CONDA_BASE="$HOME/miniconda3"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate task

# 消除 OpenCV 5 的 Qt 字体告警（不影响画面，仅美化窗口）
FONT_DIR="$(python -c 'import cv2,os;print(os.path.join(os.path.dirname(cv2.__file__),"qt","fonts"))')"
if [ ! -e "$FONT_DIR/DejaVuSans.ttf" ]; then
    mkdir -p "$FONT_DIR"
    ln -sf /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf "$FONT_DIR/DejaVuSans.ttf" 2>/dev/null || true
fi

exec python detect.py "$@"
