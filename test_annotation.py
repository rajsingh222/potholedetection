#!/usr/bin/env python3
"""Quick test to verify annotation is working"""

from ultralytics import YOLO
import cv2
from glob import glob

print("🧪 Testing annotation capability...\n")

# Load model
model = YOLO('best.pt')
print(f"✅ Model loaded\n")

# Get test image
test_images = glob('dataset/images/train/*.jpg')[:1]

if test_images:
    img_path = test_images[0]
    print(f"Testing with: {img_path}")
    
    # Inference with VERY low threshold
    for conf in [0.01, 0.05, 0.1]:
        print(f"\n  Testing conf={conf}:")
        results = model(img_path, conf=conf)
        
        # Get result
        r = results[0]
        print(f"    Boxes detected: {len(r.boxes)}")
        
        # Test annotation
        annotated = r.plot()
        print(f"    Annotated shape: {annotated.shape}")
        print(f"    Data type: {annotated.dtype}")
        
        # Save test
        test_path = f'test_annotation_conf{conf}.jpg'
        success = cv2.imwrite(test_path, annotated)
        print(f"    Saved to {test_path}: {success}")
        
        if len(r.boxes) > 0:
            print(f"    ✅ FOUND DETECTIONS AT conf={conf}!")
            break
else:
    print("❌ No test images found")
