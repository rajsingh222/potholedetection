# 🚀 Training on Google Colab - Complete Guide

## Why Google Colab?
- **FREE GPU**: NVIDIA T4/L4 (10-100x faster than CPU)
- **No setup needed**: Pre-installed libraries
- **Easy data upload**: Automatic cloud storage integration
- **Training time**: ~5 mins instead of hours

---

## Step 1: Prepare Your Dataset

### Option A: Upload to Google Drive
1. Compress your dataset:
   ```bash
   # On your local machine
   # Zip: dataset/ folder containing:
   # - images/train/
   # - labels/train/
   # - data.yaml
   ```

2. Upload to Google Drive:
   - Go to [Google Drive](https://drive.google.com)
   - Create folder: `Pothole-Detection`
   - Upload `dataset.zip` and `data.yaml`

---

## Step 2: Create Google Colab Notebook

Go to [Google Colab](https://colab.research.google.com) and paste this code:

### Cell 1: Install Dependencies
```python
# Install required packages
!pip install ultralytics opencv-python torch torchvision

# Import libraries
from ultralytics import YOLO
import torch

# Check GPU availability
print("GPU Available:", torch.cuda.is_available())
print("GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

### Cell 2: Mount Google Drive
```python
from google.colab import drive

# Mount your Google Drive
drive.mount('/content/drive')

# Navigate to your dataset folder
import os
os.chdir('/content/drive/MyDrive/Pothole-Detection')

# List files
!ls -la
```

### Cell 3: Extract and Prepare Dataset
```python
# Extract dataset if compressed
!unzip -q dataset.zip

# Verify dataset structure
!ls -la dataset/
!ls -la dataset/images/
!ls -la dataset/labels/
```

### Cell 4: Download Pre-trained Model
```python
# Download YOLOv11 nano model
!wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt

# Verify
!ls -la *.pt
```

### Cell 5: Configure Training
```python
# Create training script
training_code = '''
from ultralytics import YOLO
import torch

# Load model
model = YOLO("yolo11n.pt")

# Train with GPU
model.train(
    data="data.yaml",
    epochs=50,              # More epochs for better accuracy
    imgsz=640,              # Full resolution with GPU
    batch=32,               # Large batch size (GPU can handle it)
    device=0,               # Use GPU
    workers=4,              # Parallel workers
    cache=True,             # Cache images in memory
    pretrained=True,
    optimizer="SGD",
    verbose=True,
    patience=10,
    save=True,
    # Data augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.0,
    degrees=10,
    translate=0.1,
    scale=0.5,
    perspective=0.0,
    mixup=0.0,
    cutmix=0.0,
    mosaic=1.0,
    copy_paste=0.0,
)

print("✅ Training Complete!")
print("Model saved to: runs/detect/train/weights/best.pt")
'''

with open('train.py', 'w') as f:
    f.write(training_code)

print("✅ Training script created!")
```

### Cell 6: Run Training
```python
!python train.py
```

### Cell 7: Check Results
```python
import matplotlib.pyplot as plt
from PIL import Image

# Display training results
results_img = Image.open('/content/drive/MyDrive/Pothole-Detection/runs/detect/train/results.png')
plt.figure(figsize=(15, 8))
plt.imshow(results_img)
plt.axis('off')
plt.title("Training Results")
plt.tight_layout()
plt.show()

# Show confusion matrix
confusion_img = Image.open('/content/drive/MyDrive/Pothole-Detection/runs/detect/train/confusion_matrix.png')
plt.figure(figsize=(10, 10))
plt.imshow(confusion_img)
plt.axis('off')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
```

### Cell 8: Download Trained Model
```python
# Copy best model to Drive for download
import shutil

shutil.copy(
    '/content/drive/MyDrive/Pothole-Detection/runs/detect/train/weights/best.pt',
    '/content/drive/MyDrive/Pothole-Detection/best_model.pt'
)

print("✅ Model saved to Google Drive!")
print("Download from: Google Drive -> Pothole-Detection -> best_model.pt")
```

### Cell 9: Test Model on Sample Image
```python
from ultralytics import YOLO
import cv2
from IPython.display import Image as IPImage

# Load trained model
model = YOLO('/content/drive/MyDrive/Pothole-Detection/runs/detect/train/weights/best.pt')

# Test on a sample image from dataset
test_image = '/content/drive/MyDrive/Pothole-Detection/dataset/images/train/Czech_000000.jpg'

# Run inference
results = model(test_image)

# Save result
results[0].save(filename='result.jpg')

# Display
display(IPImage('result.jpg'))
```

---

## Step 3: Advanced Tips

### Upload Dataset from Zip
```python
# If dataset is large, upload as zip for faster transfer
!unzip -q /content/drive/MyDrive/Pothole-Detection/dataset.zip
```

### Different Model Sizes
```python
# Nano (fastest, least accurate)
model = YOLO("yolo11n.pt")

# Small (balanced)
model = YOLO("yolo11s.pt")

# Medium (slower, more accurate)
model = YOLO("yolo11m.pt")

# Large (slowest, most accurate)
model = YOLO("yolo11l.pt")
```

### Monitor Training
```python
# During training, check progress
import time
time.sleep(60)  # Wait 60 seconds
# Then check tensorboard or results
```

### Download Results
```python
# After training, download entire results folder
!zip -r results.zip /content/drive/MyDrive/Pothole-Detection/runs/detect/train/
# Then download results.zip from Colab
```

---

## Step 4: Expected Performance

| Metric | CPU | Google Colab GPU |
|--------|-----|------------------|
| **Training Time (50 epochs)** | 4-6 hours | 10-15 minutes |
| **Speed per epoch** | ~5-8 min | ~10-20 sec |
| **Batch Size** | 1-4 | 16-32 |
| **Cost** | Free | Free |

---

## Step 5: Use Trained Model Back in Your App

```python
# Download best.pt from Google Drive
# Replace the best.pt in your local project
# Your Flask app will automatically use the new model

from flask import Flask
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("best.pt")  # Now uses trained model!

@app.route('/predict', methods=['POST'])
def predict():
    # Your inference code here
    results = model(image)
    return results
```

---

## Troubleshooting

### Out of Memory?
```python
# Reduce batch size
batch=16  # Instead of 32

# Or reduce image size
imgsz=416  # Instead of 640
```

### Training Too Slow?
```python
# Use smaller model
model = YOLO("yolo11n.pt")  # Nano is fastest

# Reduce epochs
epochs=20

# Disable validation
val=False
```

### Can't Upload Large Files?
```python
# Split dataset into smaller zip files
# Upload separately and unzip in Colab
!unzip -q dataset_part1.zip
!unzip -q dataset_part2.zip
```

---

## Quick Start (Copy-Paste)
Just paste this entire notebook in Colab:

```python
# 1. Install
!pip install ultralytics opencv-python -q

# 2. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Navigate
import os
os.chdir('/content/drive/MyDrive/Pothole-Detection')

# 4. Extract
!unzip -q dataset.zip

# 5. Download model
!wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt -q

# 6. Train
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    workers=4,
    cache=True
)

# 7. Download
import shutil
shutil.copy('runs/detect/train/weights/best.pt', '/content/drive/MyDrive/Pothole-Detection/best_model.pt')
print("✅ Done! Download best_model.pt from Google Drive")
```

---

## After Training on Colab

1. **Download** `best.pt` from Google Drive
2. **Replace** the `best.pt` in your local project
3. **Run** your Flask app - it will use the trained model
4. **Deploy** to production

---

That's it! You're ready to train on GPU! 🎉
