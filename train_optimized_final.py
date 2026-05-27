#!/usr/bin/env python3
"""
🔥 OPTIMIZED YOLOv11 Training - Full Power Mode
Train the model PROPERLY to get high confidence detections
"""

import subprocess
import sys
import os
import shutil
import torch
from glob import glob
from pathlib import Path


def main():
    """Main training function - required for Windows multiprocessing"""

    # ==============================================================================
    # 1️⃣ DEPENDENCIES
    # ==============================================================================
    print("\n" + "="*80)
    print("🔥 YOLOv11 OPTIMIZED Training - High Quality Mode")
    print("="*80)

    print("\n📦 Checking dependencies...")
    try:
        from ultralytics import YOLO
        from PIL import Image
        import matplotlib.pyplot as plt
        print("✅ All dependencies ready!")
    except ImportError:
        print("Installing missing packages...")
        packages = ['ultralytics>=8.0.0', 'opencv-python', 'torch', 'torchvision', 'tqdm']
        for pkg in packages:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
        from ultralytics import YOLO
        from PIL import Image
        import matplotlib.pyplot as plt

    # ==============================================================================
    # 2️⃣ HARDWARE DETECTION
    # ==============================================================================
    print("\n🎮 Detecting Hardware...")

    has_gpu = torch.cuda.is_available()
    device = 0 if has_gpu else 'cpu'

    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_mem_total:.1f}GB)")
        
        # OPTIMIZED settings for quality training
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        batch_size = 16  # Reduced for better gradient updates
        epochs = 100    # INCREASED for convergence
        img_size = 640  # Full resolution
        workers = 2
        
        print("⏱️  Expected Time: 30-45 minutes (quality training)")
    else:
        print("⚠️  CPU Mode detected")
        batch_size = 8
        epochs = 50
        img_size = 416
        workers = 0
        print("⏱️  Expected Time: 60+ minutes")

    print(f"\n📊 Configuration:")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: {img_size}x{img_size}")
    print(f"   Workers: {workers}")

    # ==============================================================================
    # 3️⃣ FIND DATASET
    # ==============================================================================
    print("\n📁 Looking for dataset...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if os.path.isdir('dataset'):
        dataset_root = 'dataset'
        print(f"✅ Found dataset: {os.path.abspath('dataset')}")
    else:
        print("❌ Dataset not found!")
        sys.exit(1)

    # ==============================================================================
    # 4️⃣ VERIFY DATASET
    # ==============================================================================
    print("\n📋 Verifying dataset...")

    required_paths = [
        f'{dataset_root}/images/train',
        f'{dataset_root}/labels/train',
        'data.yaml'
    ]

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
            sys.exit(1)

    # ==============================================================================
    # 5️⃣ LOAD MODEL
    # ==============================================================================
    print("\n🤖 Loading YOLOv11 Model...")

    model_file = 'yolo11n.pt'
    if not os.path.exists(model_file):
        print(f"📥 Downloading {model_file}...")
        model = YOLO('yolo11n.pt')
    else:
        model_size = os.path.getsize(model_file) / 1e6
        print(f"✅ {model_file} ready ({model_size:.1f}MB)")
        model = YOLO('yolo11n.pt')

    # ==============================================================================
    # 6️⃣ OPTIMIZED TRAINING PARAMETERS
    # ==============================================================================
    print("\n⚙️  Configuring OPTIMIZED training...")

    training_params = {
        # Basic settings
        'data': 'data.yaml',
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'workers': workers,
        
        # Optimization
        'optimizer': 'SGD',
        'lr0': 0.001,  # Lower learning rate for stability
        'lrf': 0.01,   # Final learning rate factor
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Scheduling
        'warmup_epochs': 5.0,  # Longer warmup
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Regularization
        'patience': 20,  # Early stopping patience
        'close_mosaic': 10,  # Turn off mosaic for last 10 epochs
        
        # Data augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'fliplr': 0.5,
        'flipud': 0.1,
        'degrees': 10,
        'translate': 0.2,
        'scale': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        
        # Validation & Saving
        'val': True,
        'cache': False,
        'pretrained': True,
        'save': True,
        'plots': True,
        'verbose': True,
    }

    print("✅ OPTIMIZED configuration ready!")

    # ==============================================================================
    # 7️⃣ TRAIN MODEL
    # ==============================================================================
    print("\n" + "="*80)
    print("🚀 STARTING OPTIMIZED TRAINING")
    print("="*80)
    print(f"⏱️  This is proper training - may take {('30-45 minutes' if has_gpu else '60+ minutes')}")
    print("="*80 + "\n")

    try:
        results = model.train(**training_params)
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED!")
        print("="*80)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # ==============================================================================
    # 8️⃣ SAVE & TEST MODEL
    # ==============================================================================
    print("\n📊 Post-training analysis...")

    # Find latest training folder
    import glob as glob_module
    train_folders = glob_module.glob('runs/detect/train*')
    if train_folders:
        results_dir = max(train_folders, key=os.path.getctime)
    else:
        results_dir = 'runs/detect/train'

    print(f"Training folder: {results_dir}")

    # Save model
    print("\n💾 Saving model...")
    src_model = f'{results_dir}/weights/best.pt'
    
    if os.path.exists(src_model):
        shutil.copy(src_model, 'best_optimized.pt')
        shutil.copy(src_model, 'best.pt')
        model_size = os.path.getsize(src_model) / 1024 / 1024
        print(f"✅ Model saved: best.pt ({model_size:.1f}MB)")
    else:
        print(f"❌ Model not found at {src_model}")

    # Test on sample images
    print("\n🔍 Testing on sample images...")
    test_model = YOLO('best.pt')
    test_images = glob(f'{dataset_root}/images/train/*.jpg')[:3]
    
    total_detections = 0
    for idx, img_path in enumerate(test_images):
        results_test = test_model(img_path, conf=0.05)  # Very low threshold
        detections = len(results_test[0].boxes)
        total_detections += detections
        print(f"   Image {idx+1}: {detections} detections")
        
        if detections > 0:
            for box in results_test[0].boxes:
                cls_name = test_model.names[int(box.cls)]
                conf = box.conf.item()
                print(f"      → {cls_name} ({conf:.3f})")
            results_test[0].save(filename=f'test_optimized_{idx+1}.jpg')

    # ==============================================================================
    # SUMMARY
    # ==============================================================================
    print("\n" + "="*80)
    print("✨ TRAINING COMPLETE - OPTIMIZED MODEL READY ✨")
    print("="*80)

    summary = f"""
📊 TRAINING RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model:              YOLOv11 Nano (OPTIMIZED)
Epochs:             {epochs}
Batch Size:         {batch_size}
Image Size:         {img_size}x{img_size}
Learning Rate:      SGD (adaptive)
Warmup:             5 epochs
Early Stopping:     patience=20
Augmentation:       Advanced (mosaic, mixup, rotation)

🎯 DETECTION TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sample Images Tested: 3
Total Detections:     {total_detections}
Avg per Image:        {total_detections/3:.1f}

📁 OUTPUT FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ best.pt              - Main model (use this!)
✅ best_optimized.pt    - Backup copy
✅ runs/detect/train-X/ - Full training folder
✅ test_optimized_*.jpg - Test results

🚀 DEPLOYMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Run: python app.py
2. Open: http://127.0.0.1:5000
3. Upload image to test
4. Get results with high confidence detections!

✨ OPTIMIZATION NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Lower learning rate (0.001) for stable convergence
✓ Longer warmup (5 epochs) for better initialization
✓ Advanced augmentation (mosaic + mixup) for robustness
✓ Lower batch size (16) for better gradient updates
✓ More epochs (100) for full convergence
✓ Mosaic disabled last 10 epochs for fine-tuning
✓ Close monitoring with early stopping
"""

    print(summary)
    print("="*80)


if __name__ == '__main__':
    main()
