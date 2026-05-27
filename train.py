from ultralytics import YOLO
import torch

# Check device
device = "cpu"
print("Using device:", device)
print("Torch CUDA available:", torch.cuda.is_available())

# Load pretrained model
model = YOLO("yolo11n.pt")

# FAST TRAINING CONFIGURATION - Prioritizes speed over accuracy
model.train(
    data="data.yaml",
    epochs=5,              # Very fast - just 5 epochs
    imgsz=320,             # 320x320 images (4x faster than 640)
    batch=1,               # Batch size 1 for CPU
    device=device,
    workers=0,             # No multiprocessing
    cache=False,           # No caching
    pretrained=True,
    optimizer="SGD",
    verbose=False,         # Less output
    patience=3,
    save=True,
    hsv_h=0.0,             # No HSV augmentation
    hsv_s=0.0,
    hsv_v=0.0,
    fliplr=0.0,            # No flips
    flipud=0.0,
    degrees=0.0,           # No rotation
    translate=0.0,         # No translation
    scale=0.0,             # No scaling
    perspective=0.0,       # No perspective
    mixup=0.0,             # No mixup
    cutmix=0.0,            # No cutmix
    mosaic=0.0,            # No mosaic augmentation
    copy_paste=0.0,        # No copy-paste
    val=False              # Skip validation
)