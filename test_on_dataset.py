#!/usr/bin/env python3
"""
Test the trained model on the test dataset
Evaluates detection performance and generates statistics
"""

import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import time

# Configuration
TEST_IMAGE_DIR = "Data/test"
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.05  # Detection confidence threshold
OUTPUT_DIR = "test_results"

# Class mapping
CLASS_MAP = {
    0: "Longitudinal cracks (D00)",
    1: "Transverse cracks (D10)",
    2: "Alligator cracks (D20)",
    3: "Potholes (D40)",
    4: "Other corruptions (OTHER)"
}

def setup_output_directory():
    """Create output directory for results"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/annotated", exist_ok=True)
    print(f"✅ Output directory created: {OUTPUT_DIR}/\n")

def load_model():
    """Load the trained YOLO model"""
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully\n")
    return model

def get_test_images():
    """Get all test images from the directory"""
    test_path = Path(TEST_IMAGE_DIR)
    images = sorted(list(test_path.glob("*.jpg")))
    print(f"Found {len(images)} test images")
    return images

def test_model(model, images):
    """Run inference on all test images"""
    print(f"\nRunning inference on {len(images)} test images...")
    print(f"Using confidence threshold: {CONF_THRESHOLD}\n")
    
    statistics = {
        'total_images': len(images),
        'images_with_detections': 0,
        'images_without_detections': 0,
        'total_detections': 0,
        'detections_by_class': defaultdict(int),
        'inference_times': []
    }
    
    annotated_count = 0
    max_samples = 50  # Save up to 50 annotated samples
    
    for idx, img_path in enumerate(images, 1):
        if idx % 500 == 0:
            print(f"  Processing... {idx}/{len(images)} ({idx/len(images)*100:.1f}%)")
        
        # Inference
        start_time = time.time()
        results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        inference_time = time.time() - start_time
        statistics['inference_times'].append(inference_time)
        
        # Extract results
        r = results[0]
        num_boxes = len(r.boxes)
        
        if num_boxes > 0:
            statistics['images_with_detections'] += 1
            statistics['total_detections'] += num_boxes
            
            # Count by class
            for box in r.boxes:
                class_idx = int(box.cls)
                class_name = CLASS_MAP.get(class_idx, f"Unknown({class_idx})")
                statistics['detections_by_class'][class_name] += 1
            
            # Save annotated samples (first 50)
            if annotated_count < max_samples:
                annotated = r.plot()
                output_path = os.path.join(
                    OUTPUT_DIR, 
                    "annotated", 
                    f"{idx:05d}_{img_path.stem}.jpg"
                )
                cv2.imwrite(output_path, annotated)
                annotated_count += 1
        else:
            statistics['images_without_detections'] += 1
    
    print(f"✅ Inference complete!\n")
    return statistics

def print_statistics(stats):
    """Print test statistics"""
    print("=" * 70)
    print("DETECTION TEST RESULTS")
    print("=" * 70)
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"  Total test images:          {stats['total_images']}")
    print(f"  Images with detections:     {stats['images_with_detections']} ({stats['images_with_detections']/stats['total_images']*100:.1f}%)")
    print(f"  Images without detections:  {stats['images_without_detections']} ({stats['images_without_detections']/stats['total_images']*100:.1f}%)")
    
    print(f"\n🎯 DETECTION STATISTICS:")
    print(f"  Total detections:           {stats['total_detections']}")
    print(f"  Avg detections/image:       {stats['total_detections']/stats['total_images']:.2f}")
    print(f"  Avg detections per positive: {stats['total_detections']/max(1, stats['images_with_detections']):.2f}")
    
    print(f"\n📈 DETECTIONS BY CLASS:")
    for class_name, count in sorted(
        stats['detections_by_class'].items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        pct = count / stats['total_detections'] * 100 if stats['total_detections'] > 0 else 0
        print(f"  {class_name:40s}: {count:6d} ({pct:5.1f}%)")
    
    avg_time = sum(stats['inference_times']) / len(stats['inference_times'])
    total_time = sum(stats['inference_times'])
    print(f"\n⏱️  PERFORMANCE:")
    print(f"  Avg inference time/image:   {avg_time*1000:.2f}ms")
    print(f"  Total inference time:       {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"  Images per second:          {len(stats['inference_times'])/total_time:.1f}")
    
    print("\n" + "=" * 70)

def save_report(stats):
    """Save statistics report to file"""
    report_path = os.path.join(OUTPUT_DIR, "test_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DETECTION TEST RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write(f"  Total test images:          {stats['total_images']}\n")
        f.write(f"  Images with detections:     {stats['images_with_detections']} ({stats['images_with_detections']/stats['total_images']*100:.1f}%)\n")
        f.write(f"  Images without detections:  {stats['images_without_detections']} ({stats['images_without_detections']/stats['total_images']*100:.1f}%)\n\n")
        
        f.write("DETECTION STATISTICS:\n")
        f.write(f"  Total detections:           {stats['total_detections']}\n")
        f.write(f"  Avg detections/image:       {stats['total_detections']/stats['total_images']:.2f}\n")
        f.write(f"  Avg detections per positive: {stats['total_detections']/max(1, stats['images_with_detections']):.2f}\n\n")
        
        f.write("DETECTIONS BY CLASS:\n")
        for class_name, count in sorted(
            stats['detections_by_class'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            pct = count / stats['total_detections'] * 100 if stats['total_detections'] > 0 else 0
            f.write(f"  {class_name:40s}: {count:6d} ({pct:5.1f}%)\n")
        
        avg_time = sum(stats['inference_times']) / len(stats['inference_times'])
        total_time = sum(stats['inference_times'])
        f.write(f"\nPERFORMANCE:\n")
        f.write(f"  Avg inference time/image:   {avg_time*1000:.2f}ms\n")
        f.write(f"  Total inference time:       {total_time:.1f}s ({total_time/60:.1f}m)\n")
        f.write(f"  Images per second:          {len(stats['inference_times'])/total_time:.1f}\n")
    
    print(f"\n✅ Report saved to: {report_path}")

def main():
    print("\n" + "=" * 70)
    print("ROAD POTHOLE DETECTION - TEST DATASET EVALUATION")
    print("=" * 70 + "\n")
    
    # Setup
    setup_output_directory()
    model = load_model()
    images = get_test_images()
    
    # Test
    stats = test_model(model, images)
    
    # Results
    print_statistics(stats)
    save_report(stats)
    
    print(f"\n✅ Annotated samples saved to: {OUTPUT_DIR}/annotated/")
    print(f"✅ Test complete! Check {OUTPUT_DIR}/ for results\n")

if __name__ == "__main__":
    main()
