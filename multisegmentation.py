import glob 
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

extensions = ['jpg','svg','jpeg']

image_paths = []

for ext in extensions:
    image_paths.extend(glob.glob(f"images/*.{ext}"))

for path in image_paths:
    results = model(path,save=True,show=True)
    


