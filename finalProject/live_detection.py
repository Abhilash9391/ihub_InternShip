from ultralytics import YOLO
import cv2
from collections import deque

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)

buffer = deque(maxlen=10)
last_letter = ""
sentence = ""

CONF_THRESH = 0.15

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESH, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue

            cls_id = int(box.cls[0])
            letter = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f"{letter} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            buffer.append(letter)

            if buffer.count(letter) > 7 and letter != last_letter:
                sentence += letter
                last_letter = letter
                print("Detected:", sentence)

    cv2.putText(frame, sentence, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)

    cv2.imshow("Live Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
