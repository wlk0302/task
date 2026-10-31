#!/usr/bin/env bash
# 下载 YOLO11n 预训练权重；官方源（GitHub）失败时自动切换国内镜像。
# 用法：./download_model.sh

set -euo pipefail
cd "$(dirname "$0")"

FILE="yolo11n.pt"
OFFICIAL="https://github.com/ultralytics/assets/releases/download/v8.3.0/${FILE}"
MIRROR_GH="https://ghproxy.net/${OFFICIAL}"
MIRROR_HF="https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/${FILE}"

if [ -s "$FILE" ]; then
    echo "已存在 ${FILE}（$(du -h "$FILE" | cut -f1)），跳过下载。"
    exit 0
fi

for url in "$OFFICIAL" "$MIRROR_GH" "$MIRROR_HF"; do
    echo "尝试下载: $url"
    if curl -fL --connect-timeout 10 --retry 2 --retry-delay 2 -o "$FILE" "$url"; then
        echo "✅ 下载完成: ${FILE}（$(du -h "$FILE" | cut -f1)）"
        exit 0
    fi
    echo "   失败，换下一个源"
    rm -f "$FILE"
done

echo "❌ 所有源均失败。请手动下载 ${FILE} 后放到本目录：" >&2
echo "   ${MIRROR_GH}" >&2
exit 1
