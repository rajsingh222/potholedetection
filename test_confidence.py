#!/usr/bin/env python3
"""Test model with different confidence levels"""

from ultralytics import YOLO
from glob import glob
import cv2

model = YOLO('best.pt')
print(f"✅ Model loaded\n")

# Get test images
test_images = glob('dataset/images/train/*.jpg')[:5]  # Test first 5

print("Testing with different confidence thresholds:\n")

for test_img in test_images:
    print(f"Image: {test_img.split('/')[-1]}")
    
    for conf_threshold in [0.1, 0.2, 0.3, 0.5]:
        results = model(test_img, conf=conf_threshold)
        num_detections = len(results[0].boxes)
        print(f"  conf={conf_threshold}: {num_detections} detections")
        if num_detections > 0:
            for box in results[0].boxes:
                cls_name = model.names[int(box.cls)]
                conf = box.conf.item()
                print(f"      → {cls_name} ({conf:.2f})")
    print()
