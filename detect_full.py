#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 实时摄像头目标检测
========================

功能:
    1. 打开电脑内置摄像头 / 外接 USB 摄像头（也可以读视频文件、RTSP 流）
    2. 用 Ultralytics YOLO 预训练模型检测画面里的常见物品（COCO 80 类）
    3. 画出检测框、类别名、置信度，并叠加 FPS
    4. 按 q / ESC 或点窗口关闭按钮正常退出（自动释放摄像头）

用法:
    python detect.py                      # 默认摄像头 0 + yolo11n
    python detect.py --source 1           # 指定摄像头编号
    python detect.py --source test.mp4    # 读视频文件
    python detect.py --list-cameras       # 列出所有可用摄像头
    python detect.py --model yolo11s.pt   # 换更大的模型（更准、更慢）
    python detect.py --conf 0.4           # 提高置信度阈值
    python detect.py --no-show            # 不弹窗（无桌面环境调试用）
    python detect.py --save out.mp4       # 同时录制检测结果
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WINDOW_NAME = "YOLO Camera Detection  |  press q / ESC to quit"


# --------------------------------------------------------------------------- #
# 命令行参数
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO 实时摄像头目标检测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="0",
                   help="摄像头编号(0,1,2...) 或 视频/图片文件路径 或 RTSP 地址")
    p.add_argument("--model", default="yolo11n.pt",
                   help="模型权重，首次运行会自动下载")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="推理设备，auto = 有 GPU 就用 GPU")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="只检测指定类别 id，例如 --classes 0 2 7 (人/车/卡车)")
    p.add_argument("--width", type=int, default=1280, help="摄像头采集宽度")
    p.add_argument("--height", type=int, default=720, help="摄像头采集高度")
    p.add_argument("--fps", type=int, default=30, help="摄像头采集帧率")
    p.add_argument("--no-show", action="store_true", help="不创建窗口，只在终端输出")
    p.add_argument("--max-frames", type=int, default=0, help="处理 N 帧后自动退出(0=不限)")
    p.add_argument("--save", default="", help="把检测结果录成视频，例如 out.mp4")
    p.add_argument("--list-cameras", action="store_true", help="列出可用摄像头后退出")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 摄像头相关
# --------------------------------------------------------------------------- #
def probe_cameras(max_index: int = 6) -> list[tuple[int, int, int]]:
    """扫描 /dev/video0..N，返回 [(编号, 宽, 高), ...]"""
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if sys.platform.startswith("linux") \
            else cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append((idx, frame.shape[1], frame.shape[0]))
        cap.release()
    return found


def open_capture(source: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """打开视频源。数字 -> 摄像头，其它 -> 文件/流。"""
    is_camera = source.isdigit()

    if is_camera:
        idx = int(source)
        # 先试 V4L2 后端（Linux 上最稳），失败再退回默认后端
        backends = [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "默认后端")]
        for api, name in backends:
            cap = cv2.VideoCapture(idx, api)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 只留最新一帧，降低延迟
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                print(f"[摄像头] 已打开 /dev/video{idx}（{name}），实际分辨率 {w}x{h}")
                return cap
            cap.release()
        raise RuntimeError(
            f"无法打开摄像头 {idx}。请检查:\n"
            f"  1) 摄像头是否被其它程序占用（浏览器/微信/腾讯会议都会占用）\n"
            f"  2) 执行  ls -l /dev/video*  看设备是否存在\n"
            f"  3) 执行  python detect.py --list-cameras  看哪些编号可用\n"
            f"  4) 权限问题可加组:  sudo usermod -aG video $USER  然后重新登录"
        )

    path = Path(source)
    if not path.exists() and "://" not in source:
        raise FileNotFoundError(f"找不到视频文件: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    print(f"[视频] 已打开 {source}，"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
          f"{cap.get(cv2.CAP_PROP_FPS):.1f} FPS")
    return cap


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()

    # ---- 仅列摄像头 -------------------------------------------------------
    if args.list_cameras:
        print("正在扫描摄像头（0-5）...\n")
        cams = probe_cameras(6)
        if not cams:
            print("未发现可用摄像头。")
            print("  - 检查 ls -l /dev/video*")
            print("  - 检查是否被其它程序占用")
            return 1
        for idx, w, h in cams:
            print(f"  摄像头 {idx}: {w}x{h}")
            print(f"    启动: python detect.py --source {idx}")
        return 0

    # ---- 选设备 -----------------------------------------------------------
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}")
        print("请先执行:  conda activate task && pip install -r requirements.txt")
        return 1

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 62)
    print(f"PyTorch {torch.__version__} | 推理设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"模型: {args.model} | 置信度阈值: {args.conf} | 输入尺寸: {args.imgsz}")
    print("=" * 62)

    # ---- 加载模型并预热 ---------------------------------------------------
    print("正在加载模型（首次运行会自动下载权重）...")
    model = YOLO(args.model)

    t0 = time.time()
    model.predict(
        np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
        device=device, verbose=False,
    )
    print(f"模型就绪，预热耗时 {time.time() - t0:.2f}s\n")

    # ---- 打开视频源 -------------------------------------------------------
    try:
        cap = open_capture(args.source, args.width, args.height, args.fps)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[错误] {e}")
        return 1

    # ---- 可选：录制 -------------------------------------------------------
    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
        fps_out = cap.get(cv2.CAP_PROP_FPS)
        if not fps_out or fps_out != fps_out or fps_out <= 0:  # nan/0 兜底
            fps_out = 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_out, (w, h))
        print(f"[录制] 结果将保存到 {args.save} ({w}x{h} @ {fps_out:.0f}fps)")

    # ---- 交互提示 ---------------------------------------------------------
    show = not args.no_show
    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 540)
        print("[提示] 在视频窗口按 q 或 ESC 退出\n")

    # ---- 主循环 -----------------------------------------------------------
    frame_id = 0
    fps_display = 0.0
    t_prev = time.time()
    exit_reason = "未知"

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                exit_reason = "视频源读取结束或中断"
                break

            # 推理 + 绘制（results[0].plot() 会画框、类别名和置信度）
            results = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                classes=args.classes,
                verbose=False,
            )
            annotated = results[0].plot(line_width=2, font_size=9)

            n_obj = len(results[0].boxes) if results[0].boxes is not None else 0
            frame_id += 1

            # FPS（指数平滑）
            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                inst = 1.0 / dt
                fps_display = inst if fps_display == 0 else 0.9 * fps_display + 0.1 * inst

            cv2.putText(annotated, f"FPS {fps_display:5.1f}   Objects {n_obj}",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            if not args.no_show:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):      # q 或 ESC
                    exit_reason = "用户按键退出"
                    break
                # 点窗口右上角关闭按钮时 getWindowProperty 会变成 <1
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = "窗口被关闭"
                    break
            else:
                if frame_id % 10 == 0 or frame_id == 1:
                    names = results[0].names
                    cls_ids = results[0].boxes.cls.int().tolist() if results[0].boxes is not None else []
                    summary = {}
                    for c in cls_ids:
                        summary[names[c]] = summary.get(names[c], 0) + 1
                    print(f"帧 {frame_id:4d} | {fps_display:5.1f} FPS | 检出 {n_obj} 个: {summary}")

            if writer is not None:
                writer.write(annotated)

            if args.max_frames and frame_id >= args.max_frames:
                exit_reason = f"已达 --max-frames {args.max_frames} 帧"
                break

    except KeyboardInterrupt:
        exit_reason = "Ctrl+C 中断"
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[录制] 已保存 {args.save}")
        if show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # 让窗口真正关闭
        print(f"\n[退出] {exit_reason}，共处理 {frame_id} 帧")

    return 0


if __name__ == "__main__":
    sys.exit(main())
