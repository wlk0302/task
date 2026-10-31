import time

import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")                 # 预训练模型
cap = cv2.VideoCapture(0)                  # 0 = 默认摄像头，外接改 1
t_prev, fps = time.time(), 0.0

while True:
    ok, frame = cap.read()                 # 读一帧
    if not ok:
        break

    img = model.predict(frame, conf=0.25, verbose=False)[0].plot()  # 检测 + 画框

    now = time.time()
    inst = 1.0 / (now - t_prev)            # 相邻两帧间隔 → 瞬时帧率
    fps = inst if fps == 0 else fps * 0.9 + inst * 0.1   # 指数平滑，读数更稳
    t_prev = now

    cv2.putText(img, f"FPS {fps:5.1f}", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("YOLO", img)                # q/ESC 退出，s 截图

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("s"):
        name = time.strftime("shot_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(name, img)
        print(f"已保存截图 {name}")

cap.release()                              # 释放摄像头
