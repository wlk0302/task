import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")                 # 预训练模型
cap = cv2.VideoCapture(0)                  # 0 = 默认摄像头，外接改 1

while True:
    ok, frame = cap.read()                 # 读一帧
    if not ok:
        break
    # 检测 + 画框（plot() 自带类别名和置信度）+ 显示
    cv2.imshow("YOLO", model.predict(frame, conf=0.25, verbose=False)[0].plot())
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):   # q / ESC 退出
        break

cap.release()                              # 释放摄像头
