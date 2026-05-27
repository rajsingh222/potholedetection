from ultralytics import YOLO
import torch

# ============================================
# AUTO-DETECT BEST DEVICE
# ============================================
if torch.cuda.is_available():
    device = 0  # GPU (much faster - 10-15x improvement!)
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU Available: {device_name}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    batch_size = 16  # GPU can handle larger batches
else:
    device = "cpu"
    print("⚠️  No GPU found - using CPU (slow)")
    batch_size = 1

print(f"Using device: {device}")
print(f"Batch size: {batch_size}")

# Load pretrained model
model = YOLO("yolo11n.pt")

# OPTIMIZED TRAINING - Fast + Balanced
model.train(
    data="data.yaml",
    epochs=10,                 # Increased epochs with GPU
    imgsz=320,                 # Keep 320x320 for speed
    batch=batch_size,          # Auto-adjusted for device
    device=device,             # Auto GPU/CPU
    workers=4,                 # Parallel data loading (if GPU)
    cache=False,               # Disable for stability
    pretrained=True,
    optimizer="SGD",
    verbose=True,              # See progress
    patience=5,                # Early stopping
    save=True,
    # Minimal augmentation for speed
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.0,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    perspective=0.0,
    mixup=0.0,
    cutmix=0.0,
    mosaic=1.0,                # Keep mosaic for data diversity
    copy_paste=0.0,
    val=True,                  # Validate with GPU available
    patience=5,                # Stop if no improvement
)

print("\n✅ Training complete!")
print("📊 Check 'runs/detect/train/results.png' for metrics")
