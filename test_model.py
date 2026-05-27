#!/usr/bin/env python3
"""Test the trained model"""

from ultralytics import YOLO
from glob import glob
import cv2
import os

# Load model
model = YOLO('best.pt')
print(f"✅ Model loaded\n")
print(f"Classes: {model.names}\n")

# Get a test image
test_images = glob('dataset/images/train/*.jpg')
if not test_images:
    test_images = glob('dataset/images/train/*.png')

print(f"Found {len(test_images)} test images\n")

if test_images:
    test_img = test_images[0]
    print(f"Testing on: {test_img}")
    
    # Check if file exists
    if os.path.exists(test_img):
        print(f"✅ File exists\n")
        
        # Run inference
        print("Running inference...")
        results = model(test_img, conf=0.25)
        print(f"✅ Inference complete\n")
        
        # Check results
        for r in results:
            num_boxes = len(r.boxes)
            print(f"📦 Boxes detected: {num_boxes}")
            if num_boxes > 0:
                for i, box in enumerate(r.boxes):
                    cls_id = int(box.cls)
                    cls_name = model.names[cls_id]
                    conf = box.conf.item()
                    print(f"   {i+1}. {cls_name} (confidence: {conf:.2f})")
            else:
                print("   ⚠️  No objects detected in this image")
    else:
        print(f"❌ File not found: {test_img}")
else:
    print("❌ No test images found in dataset/images/train/")
