from ultralytics import YOLO

# Load pre-trained YOLOv8n
model = YOLO('yolov8n.pt')

# Train on COCO128 dataset
model.train(
    data='coco128.yaml',
    epochs=3,
    imgsz=640,
    batch=8,
    project='runs/detect',
    name='train_coco128'
)
