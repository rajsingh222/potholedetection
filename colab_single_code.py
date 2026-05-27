#!/usr/bin/env python3
"""
⚡ ULTRA-FAST YOLOv11 Training for Google Colab
Complete training in one script - 10-15 minutes on GPU
"""

import subprocess
import sys
import os
import shutil
import torch
from glob import glob
from pathlib import Path

# ==============================================================================
# 1️⃣ INSTALL DEPENDENCIES
# ==============================================================================
print("\n" + "="*80)
print("⚡ YOLOv11 Pothole Detection - Complete Training Pipeline")
print("="*80)

print("\n📦 Installing dependencies...")
packages = ['ultralytics>=8.0.0', 'opencv-python', 'torch', 'torchvision', 'tqdm']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

print("✅ Dependencies installed!")

# ==============================================================================
# 2️⃣ MOUNT GOOGLE DRIVE & EXPLORE
# ==============================================================================
print("\n📂 Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    os.chdir('/content/drive/MyDrive')
    print("✅ Google Colab detected - Drive mounted")
    is_colab = True
except:
    print("ℹ️  Local environment detected")
    is_colab = False

# ==============================================================================
# 3️⃣ FIND DATASET
# ==============================================================================
print("\n📁 Looking for dataset...")
print(f"Current directory: {os.getcwd()}\n")

# Try different paths
search_paths = [
    'dataset',
    './dataset',
    '../dataset',
    '/content/drive/MyDrive/dataset',
    '/content/drive/MyDrive/projects/dataset',
    '/content/drive/MyDrive/Resources/dataset',
    '/content/dataset',
]

dataset_found = False
for path in search_paths:
    if os.path.isdir(path):
        print(f"✅ Found dataset at: {os.path.abspath(path)}")
        os.chdir(os.path.dirname(os.path.abspath(path)))
        dataset_root = 'dataset'
        dataset_found = True
        break

if not dataset_found:
    print("❌ Dataset folder not found!")
    print("\n📍 Searched locations:")
    for p in search_paths:
        print(f"   • {p}")
    print("\n⚠️  Please ensure 'dataset' folder exists in one of these locations")
    sys.exit(1)

print(f"📍 Working directory: {os.getcwd()}")

# ==============================================================================
# 4️⃣ HARDWARE DETECTION
# ==============================================================================
print("\n🎮 Detecting Hardware...")

has_gpu = torch.cuda.is_available()
device = 0 if has_gpu else 'cpu'

if has_gpu:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(0.95)
    print(f"✅ GPU: {gpu_name}")
    print(f"💾 Memory: {gpu_mem:.1f} GB")
    print("⏱️  Expected Time: 10-15 minutes")
    batch_size = 64
    epochs = 50
    img_size = 640
    workers = 4
else:
    print("⚠️  CPU Mode (No GPU detected)")
    print("⏱️  Expected Time: 30-45 minutes")
    batch_size = 8
    epochs = 15
    img_size = 320
    workers = 0

print(f"📊 Batch: {batch_size} | Epochs: {epochs} | Image Size: {img_size}x{img_size}")

# ==============================================================================
# 5️⃣ VERIFY DATASET STRUCTURE
# ==============================================================================
print("\n📋 Verifying dataset structure...")

required_paths = [
    f'{dataset_root}/images/train',
    f'{dataset_root}/labels/train',
    'data.yaml'
]

all_valid = True
for path in required_paths:
    if os.path.exists(path):
        if 'images/train' in path:
            count = len(glob(os.path.join(path, '*.*')))
            print(f"✅ {path} ({count} images)")
        elif 'labels/train' in path:
            count = len(glob(os.path.join(path, '*.txt')))
            print(f"✅ {path} ({count} labels)")
        else:
            print(f"✅ {path}")
    else:
        print(f"❌ Missing: {path}")
        all_valid = False

if not all_valid:
    print("\n❌ Some required files are missing!")
    print("Expected structure:")
    print("   dataset/")
    print("   ├── images/train/")
    print("   └── labels/train/")
    print("   data.yaml")
    sys.exit(1)

print("✅ All files verified!")

# ==============================================================================
# 6️⃣ LOAD MODEL
# ==============================================================================
print("\n🤖 Loading YOLOv11 Nano model...")

model_file = 'yolo11n.pt'
if not os.path.exists(model_file):
    print(f"📥 Downloading {model_file}...")
    model = YOLO('yolo11n.pt')
    print("✅ Model downloaded and loaded!")
else:
    print(f"✅ {model_file} ready ({os.path.getsize(model_file)/1e6:.1f} MB)")
    model = YOLO('yolo11n.pt')

# ==============================================================================
# 7️⃣ CONFIGURE TRAINING
# ==============================================================================
print("\n⚙️  Configuring training parameters...")

training_params = {
    'data': 'data.yaml',
    'epochs': epochs,
    'imgsz': img_size,
    'batch': batch_size,
    'device': device,
    'workers': workers,
    'cache': False,
    'pretrained': True,
    'optimizer': 'SGD',
    'lr0': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'patience': 5,
    'save': True,
    'verbose': True,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'fliplr': 0.5,
    'degrees': 5,
    'translate': 0.1,
    'scale': 0.5,
    'val': True,
    'plots': True,
}

print("✅ Configuration ready!")

# ==============================================================================
# 8️⃣ TRAIN MODEL
# ==============================================================================
print("\n" + "="*80)
print("🚀 STARTING MODEL TRAINING")
print("="*80)
if has_gpu:
    print("⏱️  GPU Training in progress... (10-15 minutes)")
else:
    print("⏱️  CPU Training in progress... (30-45 minutes)")
print("="*80 + "\n")

try:
    results = model.train(**training_params)
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
except Exception as e:
    print(f"\n❌ Training error: {e}")
    sys.exit(1)

# ==============================================================================
# 9️⃣ VISUALIZE RESULTS
# ==============================================================================
print("\n📊 Displaying training results...")

results_dir = 'runs/detect/train'

try:
    if os.path.exists(f'{results_dir}/results.png'):
        img = Image.open(f'{results_dir}/results.png')
        plt.figure(figsize=(15, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Training Results - Metrics & Loss')
        plt.tight_layout()
        if is_colab:
            plt.show()
        print("✅ Training metrics saved")

    if os.path.exists(f'{results_dir}/confusion_matrix.png'):
        img = Image.open(f'{results_dir}/confusion_matrix.png')
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        if is_colab:
            plt.show()
        print("✅ Confusion matrix saved")
except Exception as e:
    print(f"⚠️  Could not display results: {e}")

# ==============================================================================
# 🔟 SAVE MODEL
# ==============================================================================
print("\n💾 Saving trained model...")

src_model = f'{results_dir}/weights/best.pt'
dst_model = 'best_trained_model.pt'

if os.path.exists(src_model):
    shutil.copy(src_model, dst_model)
    model_size = os.path.getsize(dst_model) / 1024 / 1024
    print(f"✅ Model saved: {dst_model} ({model_size:.1f} MB)")
    
    shutil.copy(src_model, 'best.pt')
    print(f"✅ Also saved as: best.pt")
    
    print("\n📦 Archiving training results...")
    try:
        subprocess.run(['zip', '-r', 'training_results.zip', results_dir, '-q'], check=True)
        print("✅ Results archived: training_results.zip")
    except:
        print("ℹ️  Could not create zip (optional)")
else:
    print(f"❌ Model not found at {src_model}")

# ==============================================================================
# TEST MODEL ON SAMPLES
# ==============================================================================
print("\n🔍 Testing model on sample images...")

try:
    test_model = YOLO(src_model)
    sample_images = glob(f'{dataset_root}/images/train/*.jpg')[:3] + glob(f'{dataset_root}/images/train/*.png')[:1]
    
    if sample_images:
        for idx, img_path in enumerate(sample_images[:3]):
            try:
                results_test = test_model(img_path, verbose=False)
                for result in results_test:
                    detections = len(result.boxes)
                    print(f"   Image {idx+1}: {detections} detections")
                    result.save(filename=f'test_detection_{idx+1}.jpg')
                    
                    if is_colab and idx < 2:
                        img = Image.open(f'test_detection_{idx+1}.jpg')
                        plt.figure(figsize=(12, 6))
                        plt.imshow(img)
                        plt.axis('off')
                        plt.title(f'Model Test - {os.path.basename(img_path)}')
                        plt.tight_layout()
                        plt.show()
            except Exception as e:
                print(f"   ⚠️  Could not test image {idx+1}: {e}")
    else:
        print("   ⚠️  No sample images found")
except Exception as e:
    print(f"   ⚠️  Testing failed: {e}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("✨ TRAINING COMPLETE ✨")
print("="*80)

summary = f"""
📊 TRAINING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model:              YOLOv11 Nano
Device:             {"GPU - " + torch.cuda.get_device_name(0) if has_gpu else "CPU"}
Epochs:             {epochs}
Batch Size:         {batch_size}
Image Size:         {img_size}x{img_size}
Training Time:      {("~10-15 minutes" if has_gpu else "~30-45 minutes")}
Training Images:    {len(glob(f'{dataset_root}/images/train/*.*'))}
Classes:            5 (Longitudinal, Transverse, Alligator, Pothole, Other)

📁 OUTPUT FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ best_trained_model.pt  - Your trained model
✅ best.pt               - For app.py
✅ training_results.zip  - All metrics & visualizations
✅ runs/detect/train/    - Full training folder

🚀 NEXT STEPS (For Local Use)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Download 'best.pt' from Colab (left sidebar → Files)
2. Place in your project root:
   e:\\Roadpotholedetection System\\best.pt
3. Run your Flask app:
   python app.py
4. Open: http://127.0.0.1:5000

✨ Your model is ready for deployment!
"""

print(summary)
print("="*80)
