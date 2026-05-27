#!/usr/bin/env python3
"""
⚡ ULTRA-FAST YOLOv11 Training - Local Version
Complete training in one script - 10-15 minutes on GPU or 30-45 minutes on CPU
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
    # 1️⃣ CHECK & INSTALL DEPENDENCIES
    # ==============================================================================
    print("\n" + "="*80)
    print("⚡ YOLOv11 Pothole Detection - Local Training Pipeline")
    print("="*80)

    print("\n📦 Checking dependencies...")
    try:
        from ultralytics import YOLO
        from PIL import Image
        import matplotlib.pyplot as plt
        print("✅ All dependencies already installed!")
    except ImportError:
        print("Installing missing packages...")
        packages = ['ultralytics>=8.0.0', 'opencv-python', 'torch', 'torchvision', 'tqdm']
        for pkg in packages:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
        
        from ultralytics import YOLO
        from PIL import Image
        import matplotlib.pyplot as plt
        print("✅ Dependencies installed!")

    # ==============================================================================
    # 2️⃣ DETECT HARDWARE & CONFIGURE
    # ==============================================================================
    print("\n🎮 Detecting Hardware...")

    has_gpu = torch.cuda.is_available()
    device = 0 if has_gpu else 'cpu'

    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"💾 Total Memory: {gpu_mem_total:.1f} GB")
        
        # Adaptive batch size based on GPU memory
        if gpu_mem_total >= 16:  # High-end GPU
            batch_size = 64
            img_size = 640
            workers = 4
            print("🚀 High-end GPU detected - using aggressive settings")
        elif gpu_mem_total >= 8:  # Mid-range GPU (RTX 3070, RTX 3080, etc.)
            batch_size = 32
            img_size = 640
            workers = 2
            print("💪 Mid-range GPU detected - balanced settings")
        elif gpu_mem_total >= 4:  # Lower GPU (RTX 3060, GTX 1080, etc.)
            batch_size = 16
            img_size = 512
            workers = 2
            print("⚠️  Lower VRAM GPU detected - reduced batch size")
        else:  # Very limited GPU
            batch_size = 8
            img_size = 320
            workers = 0
            print("⚠️  Limited VRAM - minimal batch size")
        
        torch.cuda.set_per_process_memory_fraction(0.9)
        epochs = 50
        print("⏱️  Expected Time: 10-15 minutes")
    else:
        print("⚠️  No GPU detected - using CPU (slower)")
        print("⏱️  Expected Time: 30-45 minutes")
        print("💡 Tip: For faster training, install CUDA/cuDNN or use Google Colab GPU")
        batch_size = 8
        epochs = 15
        img_size = 320
        workers = 0

    print(f"\n📊 Configuration:")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Image Size: {img_size}x{img_size}")
    print(f"   Workers: {workers}")

    # ==============================================================================
    # 3️⃣ FIND DATASET LOCALLY
    # ==============================================================================
    print("\n📁 Looking for dataset...")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"Script directory: {script_dir}\n")

    # Check if dataset exists in current directory
    if os.path.isdir('dataset'):
        dataset_root = 'dataset'
        print(f"✅ Found dataset in: {os.path.abspath('dataset')}")
    elif os.path.isdir('../dataset'):
        dataset_root = '../dataset'
        os.chdir('..')
        print(f"✅ Found dataset in: {os.path.abspath('dataset')}")
    else:
        print("❌ Dataset folder not found!")
        print(f"\n📍 Expected location: {os.path.join(script_dir, 'dataset')}")
        print("   Please ensure 'dataset' folder exists in the project root with structure:")
        print("      dataset/")
        print("      ├── images/train/")
        print("      └── labels/train/")
        sys.exit(1)

    # ==============================================================================
    # 4️⃣ VERIFY DATASET STRUCTURE
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
    # 5️⃣ LOAD MODEL
    # ==============================================================================
    print("\n🤖 Loading YOLOv11 Nano model...")

    model_file = 'yolo11n.pt'
    if not os.path.exists(model_file):
        print(f"📥 Downloading {model_file}...")
        model = YOLO('yolo11n.pt')
        print("✅ Model downloaded and loaded!")
    else:
        model_size = os.path.getsize(model_file) / 1e6
        print(f"✅ {model_file} ready ({model_size:.1f} MB)")
        model = YOLO('yolo11n.pt')

    # ==============================================================================
    # 6️⃣ CONFIGURE TRAINING PARAMETERS
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
    # 7️⃣ TRAIN MODEL
    # ==============================================================================
    print("\n" + "="*80)
    print("🚀 STARTING MODEL TRAINING")
    print("="*80)
    if has_gpu:
        print("⏱️  GPU Training in progress... (10-15 minutes)")
    else:
        print("⏱️  CPU Training in progress... (30-45 minutes)")
        print("   This will take a while, but it will work!")
    print("="*80 + "\n")

    try:
        results = model.train(**training_params)
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        sys.exit(1)

    # ==============================================================================
    # 8️⃣ VISUALIZE RESULTS
    # ==============================================================================
    print("\n📊 Displaying training results...")

    # Find the latest training folder (YOLO creates train, train-2, train-3, etc.)
    import glob as glob_module
    train_folders = glob_module.glob('runs/detect/train*')
    if train_folders:
        results_dir = max(train_folders, key=os.path.getctime)  # Get most recent
    else:
        results_dir = 'runs/detect/train'
    
    print(f"Using training folder: {results_dir}")

    try:
        if os.path.exists(f'{results_dir}/results.png'):
            img = Image.open(f'{results_dir}/results.png')
            plt.figure(figsize=(15, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Training Results - Metrics & Loss')
            plt.tight_layout()
            plt.savefig('training_results_display.png', dpi=100, bbox_inches='tight')
            print("✅ Training metrics saved to: training_results_display.png")
            
            # Try to display if not headless
            try:
                plt.show(block=False)
            except:
                pass

        if os.path.exists(f'{results_dir}/confusion_matrix.png'):
            img = Image.open(f'{results_dir}/confusion_matrix.png')
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig('confusion_matrix_display.png', dpi=100, bbox_inches='tight')
            print("✅ Confusion matrix saved to: confusion_matrix_display.png")
            
            try:
                plt.show(block=False)
            except:
                pass
    except Exception as e:
        print(f"⚠️  Could not display results: {e}")

    # ==============================================================================
    # 9️⃣ SAVE MODEL
    # ==============================================================================
    print("\n💾 Saving trained model...")

    src_model = f'{results_dir}/weights/best.pt'
    dst_model = 'best_trained_model.pt'

    if os.path.exists(src_model):
        shutil.copy(src_model, dst_model)
        model_size = os.path.getsize(dst_model) / 1024 / 1024
        print(f"✅ Model saved: {dst_model} ({model_size:.1f} MB)")
        
        # Also copy to 'best.pt' for app.py
        shutil.copy(src_model, 'best.pt')
        print(f"✅ Also saved as: best.pt")
        
        # Create results archive
        print("\n📦 Archiving training results...")
        try:
            subprocess.run(['tar', '-czf', 'training_results.tar.gz', results_dir], 
                          capture_output=True, timeout=60)
            print("✅ Results archived: training_results.tar.gz")
        except:
            try:
                subprocess.run(['zip', '-r', 'training_results.zip', results_dir, '-q'], 
                              timeout=60)
                print("✅ Results archived: training_results.zip")
            except:
                print("ℹ️  Could not create archive (optional)")
    else:
        print(f"❌ Model not found at {src_model}")
        print(f"Looking in: {os.path.abspath(results_dir)}")
        if os.path.exists(results_dir):
            print(f"Contents: {os.listdir(results_dir)}")

    # ==============================================================================
    # 🔟 TEST MODEL ON SAMPLES
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
                        
                        try:
                            img = Image.open(f'test_detection_{idx+1}.jpg')
                            plt.figure(figsize=(12, 6))
                            plt.imshow(img)
                            plt.axis('off')
                            plt.title(f'Model Test - {os.path.basename(img_path)}')
                            plt.tight_layout()
                            plt.show(block=False)
                        except:
                            pass
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

    num_images = len(glob(f'{dataset_root}/images/train/*.*'))

    summary = f"""
📊 TRAINING SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model:              YOLOv11 Nano
Device:             {"GPU - " + torch.cuda.get_device_name(0) if has_gpu else "CPU"}
Epochs:             {epochs}
Batch Size:         {batch_size}
Image Size:         {img_size}x{img_size}
Training Time:      {("~10-15 minutes" if has_gpu else "~30-45 minutes")}
Training Images:    {num_images}
Classes:            5 (Longitudinal, Transverse, Alligator, Pothole, Other)

📁 OUTPUT FILES IN PROJECT ROOT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ best.pt                  - Ready for app.py (use this!)
✅ best_trained_model.pt    - Backup copy
✅ training_results.*       - Metrics & visualizations
✅ runs/detect/train/       - Full training folder
✅ test_detection_*.jpg     - Sample predictions

🚀 NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. The model is saved as 'best.pt' in your project root
2. Run your Flask app:
   python app.py
3. Open: http://127.0.0.1:5000

✨ Your model is ready for deployment!
"""

    print(summary)
    print("="*80)


if __name__ == '__main__':
    # This is required for Windows multiprocessing support
    main()
