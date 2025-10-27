from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
source =0

results = model.predict(source=source,show=True,conf=0.5)

for result in results: 
    frame = result.orig_img
    annotated_frame = result.plot()
    cv2.imshow("detected objects",annotated_frame)

    if cv2.waitKey(1) &0XFF == ord('q'):
        break
cv2.destroyAllWindows()
