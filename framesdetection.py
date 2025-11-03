from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.predict(
          source = "frames/",
          save = True,
          project = "results",
          name = "frame_detect"

)