#!/usr/bin/env python3
"""Test with previously working images"""

from ultralytics import YOLO
import cv2

model = YOLO('best.pt')
print("Testing images that previously worked:\n")

test_images = ['dataset/images/train/Czech_000002.jpg', 'dataset/images/train/Czech_000006.jpg']

for img_path in test_images:
    try:
        print(f"Image: {img_path}")
        results = model(img_path, conf=0.05)
        r = results[0]
        num_boxes = len(r.boxes)
        print(f"  Detections: {num_boxes}")
        
        if num_boxes > 0:
            for box in r.boxes:
                cls_name = model.names[int(box.cls)]
                conf = box.conf.item()
                print(f"    → {cls_name} ({conf:.3f})")
        
        # Test annotation
        annotated = r.plot()
        fname = f'test_working_{img_path.split("/")[-1]}'
        cv2.imwrite(fname, annotated)
        print(f"  ✅ Annotated image saved: {fname}\n")
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
