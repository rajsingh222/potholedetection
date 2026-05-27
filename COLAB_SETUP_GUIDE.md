# 🚀 Google Colab Setup Guide

## ⚠️ Fix: Dataset Not Found Error

If you're getting this error:
```
❌ ERROR: No dataset found!
   Place 'dataset.zip' or 'dataset/' folder in current directory
```

Follow these steps:

---

## **Option 1: Upload dataset.zip to Google Drive (RECOMMENDED)**

### Step 1: Prepare Your Dataset
- Make sure you have `dataset.zip` on your computer
- It should contain:
  ```
  dataset.zip
  ├── dataset/
  │   ├── images/
  │   │   └── train/
  │   └── labels/
  │       └── train/
  ```

### Step 2: Upload to Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Create a folder called `PotholeDetection` (or any name)
3. Upload `dataset.zip` into this folder
4. **Also upload `data.yaml`** to the same folder
5. **Upload `colab_complete_training.py`** to the same folder

### Step 3: Run in Google Colab
1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy-paste this code:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Change to your folder
import os
os.chdir('/content/drive/MyDrive/PotholeDetection')

# Run the training script
exec(open('colab_complete_training.py').read())
```

4. Run the cell and wait!

---

## **Option 2: Upload Files Directly to Colab**

### Step 1: In Colab Notebook
```python
from google.colab import files

print("Upload your files:")
print("1. dataset.zip")
print("2. data.yaml")
print("3. colab_complete_training.py")

uploaded = files.upload()

for filename in uploaded.keys():
    print(f'Uploaded: {filename}')
```

### Step 2: Extract and Run
```python
import zipfile
import os

# Extract dataset
if 'dataset.zip' in uploaded:
    with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    print("Dataset extracted!")

# Run training
exec(open('colab_complete_training.py').read())
```

---

## **Option 3: Local Run (No Colab)**

### On Your Computer:
```bash
# Make sure you're in the project folder
cd e:\Roadpotholedetection System

# Dataset should be here:
ls dataset/images/train/
ls dataset/labels/train/

# Run training
python colab_complete_training.py
```

---

## **Verification Checklist** ✅

Before running, make sure:

- [ ] `dataset/` folder exists with images and labels
- [ ] `data.yaml` exists in the working directory
- [ ] `colab_complete_training.py` exists
- [ ] All files are in the same directory
- [ ] Dataset structure:
  ```
  dataset/
  ├── images/
  │   ├── train/ (contains *.jpg, *.png files)
  │   └── val/ (optional)
  └── labels/
      ├── train/ (contains *.txt files)
      └── val/ (optional)
  ```

---

## **If Still Getting Error**

### Check current directory:
```python
import os
print("Current directory:", os.getcwd())
print("Files here:", os.listdir('.'))
print("Dataset exists:", os.path.exists('dataset'))
```

### Check dataset structure:
```python
from glob import glob
print("Train images:", len(glob('dataset/images/train/*.*')))
print("Train labels:", len(glob('dataset/labels/train/*.txt')))
```

### Manually specify path:
The script now automatically searches these locations:
- `./dataset`
- `../dataset`
- `/content/drive/MyDrive/dataset` (Colab)
- `/content/dataset` (Colab)
- `~/dataset`

---

## **Expected Output**

When running successfully, you'll see:

```
================================================================================
⚡ ULTRA-FAST YOLOv11 Pothole Detection Training
================================================================================

📦 Setting up dependencies...
✅ Dependencies installed!

🎮 Detecting Hardware...
   ✅ GPU: Tesla T4 (or your GPU name)
   💾 Memory: 15.0 GB
   ⏱️  Expected Time: 10-15 minutes

📁 Setting up dataset...
   ✅ Dataset found locally!

📋 Verifying dataset structure...
   ✅ dataset/images/train (80 images)
   ✅ dataset/labels/train (80 labels)
   ✅ data.yaml

...training starts...
```

---

## **Tips for Success**

1. **Use GPU in Colab**: Runtime → Change runtime type → GPU
2. **Dataset should be < 1GB** for faster training
3. **Check Colab specs**: High RAM CPU or GPU recommended
4. **Monitor GPU**: Watch GPU memory usage during training
5. **Save model**: Download `best.pt` after training

---

## **Next Steps After Training**

1. Download `best.pt` from Colab
2. Copy to your local project:
   ```
   e:\Roadpotholedetection System\best.pt
   ```
3. Run your Flask app:
   ```bash
   python app.py
   ```
4. Open http://127.0.0.1:5000

---

## **Troubleshooting**

| Error | Solution |
|-------|----------|
| `No dataset found` | Check Option 1/2/3 above |
| `CUDA out of memory` | Reduce batch size in script |
| `Training very slow` | Use Colab GPU (faster than CPU) |
| `data.yaml not found` | Upload it to the same folder |
| `Permission denied` | Check file permissions |

---

**Questions?** Check the script output messages - they're designed to help! 🎯
