import cv2, time
from ultralytics import YOLO
import torch


DET_WEIGHT = "yolo11swindball.pt"         
URL = "http://10.159.48.206:4747/video"
IMSIZE = 640
CONF = 0.75


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = YOLO(DET_WEIGHT).to(device)

# DroidCam stream
cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    print("Error: Cannot open DroidCam stream")
    exit()

print(" YOLOv11 tracking on DroidCam | ESC to quit")
prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect + track 
    results = detector.track(
        source=frame,
        conf=CONF,
        imgsz=IMSIZE,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
        show=False
    )[0]

    if results.boxes and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        ids = results.boxes.id.cpu().numpy().astype(int)

        for i, ((x1, y1, x2, y2), track_id) in enumerate(zip(boxes, ids)):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            conf_score = results.boxes.conf[i]
            cv2.putText(frame, f"Wind Ball {conf_score:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    fps = 1 / (time.time() - prev); prev = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imshow("Wind Ball Tracker (DroidCam)", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()