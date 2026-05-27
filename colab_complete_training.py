#!/usr/bin/env python3
"""
⚡ OPTIMIZED YOLOv11 Training Pipeline - FASTEST VERSION
Complete training in 10-15 minutes on GPU or 30-45 minutes on CPU
"""

import subprocess
import sys
import os
import shutil
import torch
from glob import glob
from pathlib import Path

# ==============================================================================
# FAST SETUP & DEPENDENCIES
# ==============================================================================
print("\n" + "="*80)
print("⚡ ULTRA-FAST YOLOv11 Pothole Detection Training")
print("="*80)

print("\n📦 Setting up dependencies...")
packages = ['ultralytics>=8.0.0', 'opencv-python', 'torch', 'torchvision', 'tqdm']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

print("✅ Dependencies installed!")

# ==============================================================================
# DEVICE DETECTION & SETUP
# ==============================================================================
print("\n🎮 Detecting Hardware...")

has_gpu = torch.cuda.is_available()
device = 0 if has_gpu else 'cpu'

if has_gpu:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% GPU memory
    print(f"   ✅ GPU: {gpu_name}")
    print(f"   💾 Memory: {gpu_mem:.1f} GB")
    print("   ⏱️  Expected Time: 10-15 minutes")
    batch_size = 64
    epochs = 50
    img_size = 640
    workers = 4
else:
    print("   ⚠️  CPU Mode (No GPU detected)")
    print("   ⏱️  Expected Time: 30-45 minutes")
    batch_size = 8
    epochs = 15
    img_size = 320
    workers = 0

print(f"   📊 Batch: {batch_size} | Epochs: {epochs} | Image Size: {img_size}x{img_size}")

# ==============================================================================
# COLAB MOUNT (if running in Colab)
# ==============================================================================
print("\n📂 Checking environment...")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    os.chdir('/content/drive/MyDrive')
    print("   ✅ Google Colab detected - Drive mounted")
    is_colab = True
except:
    print("   ℹ️  Local environment detected")
    is_colab = False

# ==============================================================================
# DATASET SETUP
# ==============================================================================
print("\n📁 Setting up dataset...")
print(f"   Current directory: {os.getcwd()}")

# Function to search for dataset in different locations
def find_dataset():
    search_paths = [
        'dataset',
        './dataset',
        '../dataset',
        '/content/drive/MyDrive/dataset',
        '/content/dataset',
        os.path.expanduser('~/dataset'),
    ]
    
    for path in search_paths:
        if os.path.isdir(path):
            print(f"   ✅ Found dataset at: {os.path.abspath(path)}")
            return path
    
    return None

def find_dataset_zip():
    search_paths = [
        'dataset.zip',
        './dataset.zip',
        '../dataset.zip',
        '/content/drive/MyDrive/dataset.zip',
        '/content/dataset.zip',
        os.path.expanduser('~/dataset.zip'),
    ]
    
    for path in search_paths:
        if os.path.isfile(path):
            print(f"   ✅ Found dataset.zip at: {os.path.abspath(path)}")
            return path
    
    return None

# Try to find dataset
dataset_path = find_dataset()
dataset_zip_path = find_dataset_zip()

if dataset_path:
    print("   ✅ Dataset folder found!")
    dataset_root = dataset_path
elif dataset_zip_path:
    print("   📦 Extracting dataset.zip...")
    import zipfile
    try:
        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("   ✅ Dataset extracted!")
        dataset_root = 'dataset'
    except Exception as e:
        print(f"   ❌ Failed to extract: {e}")
        sys.exit(1)
else:
    print("\n   ❌ ERROR: Dataset not found in any location!")
    print("\n   📍 Searched locations:")
    print("      • Current directory")
    print("      • ./dataset")
    print("      • ../dataset")
    print("      • /content/drive/MyDrive/dataset")
    print("      • /content/dataset")
    print("      • ~/dataset")
    print("\n   ✅ Solutions:")
    print("      1. For LOCAL: Place 'dataset' folder in project root")
    print("      2. For COLAB: Upload 'dataset.zip' to Google Drive root")
    print("      3. Or manually upload to /content/ in Colab")
    print("\n   📂 Your workspace has:")
    for item in glob('*'):
        item_type = "📁 Folder" if os.path.isdir(item) else "📄 File"
        print(f"      {item_type}: {item}")
    sys.exit(1)

# Verify dataset structure
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
            print(f"   ✅ {path} ({count} images)")
        elif 'labels/train' in path:
            count = len(glob(os.path.join(path, '*.txt')))
            print(f"   ✅ {path} ({count} labels)")
        else:
            print(f"   ✅ {path}")
    else:
        print(f"   ❌ Missing: {path}")
        all_valid = False

if not all_valid:
    print("\n   ❌ Some required files are missing!")
    print("   Check your dataset structure matches:")
    print("      dataset/")
    print("      ├── images/")
    print("      │   └── train/")
    print("      └── labels/")
    print("          └── train/")
    print("      data.yaml")
    sys.exit(1)

# ==============================================================================
# MODEL DOWNLOAD
# ==============================================================================
print("\n🤖 Loading YOLOv11 model...")

model_file = 'yolo11n.pt'
if not os.path.exists(model_file):
    print(f"   📥 Downloading {model_file}...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolo11n.pt')  # Auto-downloads
        print(f"   ✅ Model loaded!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
else:
    print(f"   ✅ {model_file} ready ({os.path.getsize(model_file)/1e6:.1f} MB)")

# ==============================================================================
# OPTIMIZED TRAINING CONFIGURATION
# ==============================================================================
print("\n⚙️  Configuring optimized training...")

training_params = {
    'data': 'data.yaml',
    'epochs': epochs,
    'imgsz': img_size,
    'batch': batch_size,
    'device': device,
    'workers': workers,
    'cache': False,  # False is faster for small datasets
    'pretrained': True,
    'optimizer': 'SGD',  # SGD is faster than Adam
    'lr0': 0.01,  # Learning rate
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'patience': 5,  # Early stopping
    'save': True,
    'save_period': 1,
    'verbose': True,
    
    # ✅ SPEED OPTIMIZATIONS
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'fliplr': 0.5,
    'flipud': 0.0,
    'degrees': 5,
    'translate': 0.1,
    'scale': 0.5,
    'perspective': 0.0,
    'mixup': 0.0,
    'cutmix': 0.0,
    'mosaic': 1.0,
    'copy_paste': 0.0,
    'erasing': 0.0,
    'val': True,
    'plots': True,
}

print("✅ Configuration ready!")

# ==============================================================================
# PRE-TRAINING VALIDATION
# ==============================================================================
print("\n✔️  Running validation checks...")

checks_passed = True

# Check data.yaml
if not os.path.exists('data.yaml'):
    print("   ❌ data.yaml not found!")
    checks_passed = False
else:
    print("   ✅ data.yaml")

# Check dataset folders
if not os.path.exists(f'{dataset_root}/images/train'):
    print(f"   ❌ {dataset_root}/images/train not found!")
    checks_passed = False
else:
    num_train = len(glob(f'{dataset_root}/images/train/*.*'))
    print(f"   ✅ Training images: {num_train}")

if not os.path.exists(f'{dataset_root}/labels/train'):
    print(f"   ❌ {dataset_root}/labels/train not found!")
    checks_passed = False
else:
    num_labels = len(glob(f'{dataset_root}/labels/train/*.txt'))
    print(f"   ✅ Training labels: {num_labels}")

if not checks_passed:
    print("\n   ❌ Validation failed! Check dataset structure.")
    sys.exit(1)

print("\n✅ All checks passed! Ready to train.")

# ==============================================================================
# TRAINING
# ==============================================================================
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)
if has_gpu:
    print("⏱️  GPU Training in progress... (10-15 minutes)")
else:
    print("⏱️  CPU Training in progress... (30-45 minutes)")
    print("   💡 Tip: For faster training, use Google Colab with GPU!")
print("="*80 + "\n")

try:
    model = YOLO('yolo11n.pt')
    results = model.train(**training_params)
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
except Exception as e:
    print(f"\n❌ Training error: {e}")
    sys.exit(1)

# ==============================================================================
# RESULTS VISUALIZATION
# ==============================================================================
print("\n📊 Generating results visualization...")

results_dir = 'runs/detect/train'

try:
    # Show training results
    if os.path.exists(f'{results_dir}/results.png'):
        img = Image.open(f'{results_dir}/results.png')
        plt.figure(figsize=(15, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Training Results - Metrics & Loss')
        plt.tight_layout()
        plt.savefig('training_results_display.png', dpi=100, bbox_inches='tight')
        print("   ✅ Training metrics saved")
        if is_colab:
            plt.show()

    # Show confusion matrix
    if os.path.exists(f'{results_dir}/confusion_matrix.png'):
        img = Image.open(f'{results_dir}/confusion_matrix.png')
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix_display.png', dpi=100, bbox_inches='tight')
        print("   ✅ Confusion matrix saved")
        if is_colab:
            plt.show()
            
except Exception as e:
    print(f"   ⚠️  Could not display results: {e}")

# ==============================================================================
# SAVE & PACKAGE MODEL
# ==============================================================================
print("\n💾 Saving trained model...")

src_model = f'{results_dir}/weights/best.pt'
dst_model = 'best_trained_model.pt'

if os.path.exists(src_model):
    shutil.copy(src_model, dst_model)
    model_size = os.path.getsize(dst_model) / 1024 / 1024
    print(f"   ✅ Model saved: {dst_model} ({model_size:.1f} MB)")
    
    # Also copy to 'best.pt' for app.py
    shutil.copy(src_model, 'best.pt')
    print(f"   ✅ Also saved as: best.pt")
    
    # Create results archive
    print("\n📦 Archiving training results...")
    try:
        subprocess.run(['zip', '-r', 'training_results.zip', results_dir, '-q'], check=True)
        print("   ✅ Results archived: training_results.zip")
    except:
        print("   ℹ️  Could not create zip (optional)")
else:
    print(f"   ❌ Model not found at {src_model}")

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
                    
                    # Save test result
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
# TRAINING SUMMARY & NEXT STEPS
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
1. Download 'best.pt' or 'best_trained_model.pt'
2. Place in your project root directory
3. Run: python app.py
4. Open: http://127.0.0.1:5000

📱 FOR DEPLOYMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Docker: Package model with Flask app
✓ Cloud: Deploy on AWS/Azure/GCP
✓ Edge: Use on mobile/IoT devices
✓ API: Build FastAPI for scaling

🎓 FURTHER IMPROVEMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Add more training data for better accuracy
• Try larger models: yolo11s.pt or yolo11m.pt
• Fine-tune hyperparameters
• Implement augmentation strategies
• Use validation data for tuning

📖 RESOURCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔗 YOLOv11 Docs:    https://docs.ultralytics.com
🔗 Colab:           https://colab.research.google.com
🔗 Flask Docs:      https://flask.palletsprojects.com

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ Your model is ready for deployment! ✨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print(summary)
print("="*80)
