# 🚗 Road Pothole Detection System

YOLOv11-based real-time pothole detection with web interface and optional Google Colab training.

## ⚡ Quick Start (No Training Required!)

The pre-trained model is included. Just run:

```bash
# 1. Clone repository
git clone https://github.com/rajsingh222/potholedetection.git
cd potholedetection

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run web app
python app.py

# 5. Open browser
# Go to: http://127.0.0.1:5000
```

**That's it!** The trained model (`best.pt`) is ready to use. No training needed! 🎉

## 📊 Model Details

- **Model:** YOLOv11 Nano
- **Training Images:** 2,829 pothole images
- **Classes:** 5 damage types
  - 0: Longitudinal cracks
  - 1: Transverse cracks
  - 2: Alligator cracks
  - 3: Potholes
  - 4: Other corruptions
- **mAP50:** 0.298
- **File Size:** 5.5 MB
- **Speed:** ~7ms per image (GPU), ~15ms (CPU)

## 🎯 Features

✅ Web UI for image uploads  
✅ Real-time object detection  
✅ Damage classification  
✅ Detection counts by class  
✅ Annotated result images  
✅ Local GPU/CPU support  
✅ Optional Google Colab training  

## 📁 Project Structure

```
├── best.pt                      # ✅ Pre-trained model (ready to use!)
├── app.py                       # Flask web application
├── train_local_fast.py          # Fast local training script
├── train_optimized_final.py     # Optimized training (100 epochs)
├── colab_single_code.py         # Google Colab training
├── data.yaml                    # Dataset configuration
├── templates/
│   └── index.html               # Web UI
├── static/
│   ├── uploads/                 # Uploaded images
│   └── results/                 # Detection results
└── dataset/                     # Training dataset (optional)
    ├── images/train/
    └── labels/train/
```

## 🚀 Usage

### Web Interface
1. Go to `http://127.0.0.1:5000`
2. Upload image or video
3. Click "Detect"
4. View results with:
   - Annotated image with bounding boxes
   - Damage type counts
   - Confidence scores

### Python API

```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Predict
results = model('image.jpg', conf=0.1)

# Get detections
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf = box.conf.item()
        print(f"{cls_name}: {conf:.2f}")
```

## 🔧 Training (Optional)

If you want to retrain or improve the model:

### Fast Local Training
```bash
python train_local_fast.py
```
- **Time:** 10-15 min (GPU) / 30-45 min (CPU)
- **Epochs:** 50
- **Batch Size:** Adaptive

### Optimized Training (Best Quality)
```bash
python train_optimized_final.py
```
- **Time:** 30-45 min (GPU) / 60+ min (CPU)
- **Epochs:** 100
- **Features:** Advanced augmentation, optimal learning rate

### Google Colab (Fastest!)
```bash
# Copy colab_single_code.py to Google Colab
# Runtime: 10-15 min on Tesla T4 GPU
```

## 📋 Testing

```bash
# Test model on sample images
python test_model.py

# Test different confidence thresholds
python test_confidence.py
```

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| No detections | Lower confidence to 0.05-0.1 |
| Out of memory | Reduce batch size or image size |
| Slow inference | Enable GPU (check CUDA installation) |
| Model not found | Ensure `best.pt` exists (5.5 MB) |

## 📦 Requirements

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
flask>=2.0.0
pillow>=8.0.0
```

Install: `pip install -r requirements.txt`

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| mAP50 | 0.298 |
| mAP50-95 | 0.118 |
| Precision | 0.333 |
| Recall | 0.376 |
| Inference Speed (GPU) | 7ms |
| Inference Speed (CPU) | 15ms |

## 🎓 Dataset & Training

**Pre-trained on:** Czech pothole dataset (2,829 images)

**To use custom data:**
1. Organize: `dataset/images/train/` and `dataset/labels/train/`
2. Use YOLO format for labels (`.txt` files)
3. Update `data.yaml`
4. Run training script

## 🌐 Deployment

### Local
```bash
python app.py
```

### Production
```bash
pip install gunicorn
gunicorn app:app
```

## 📈 Training Timeline

- **May 27, 2026:** YOLOv11 trained, 2829 images, 22 min on GPU
- **mAP50:** 0.298 (all 5 pothole classes detected)
- **Confidence:** 0.1-0.3 (optimal for this dataset)

## 🤝 How to Contribute

1. Improve model accuracy by:
   - Collecting more training data
   - Fine-tuning hyperparameters
   - Using advanced augmentation
2. Enhance web UI
3. Add new features

## 📝 License

MIT License

## 👨‍💻 Author

Raj Singh

---

**🚀 Ready to go! Run:** `python app.py`