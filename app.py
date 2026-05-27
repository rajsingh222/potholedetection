
# Import Libraries
import os
import uuid
import cv2
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

# 1. Correct Flask initialization
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO MODEL
model = YOLO("best.pt")

print(f"✅ Model loaded successfully!")
print(f"Model classes: {model.names}")


# Helper function
def count_detections(results):
    # Fixed dictionary syntax with initial zero counts
    counts = {"D00": 0, "D10": 0, "D20": 0, "D30": 0, "D40": 0, "OTHER": 0}

    # Fixed class map to match your trained model classes (from data.yaml)
    class_map = {
        "Longitudinal cracks": "D00",
        "Transverse cracks": "D10",
        "Alligator cracks": "D20",
        "Potholes": "D40",
        "Other corruptions": "OTHER",
    }

    # Indented the processing logic inside the function
    for r in results:
        if r.boxes is None:
            continue

        for cls in r.boxes.cls.tolist():
            name = model.names[int(cls)]
            print(f"  Detected: {name} (class {int(cls)})")

            if name in class_map:
                counts[class_map[name]] += 1
            else:
                counts["OTHER"] += 1

    return counts


# ROUTES
@app.route("/")
def index():
    return render_template("index.html")


# Image prediction route
@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file Uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "empty filename"}), 400

        # FIXED: Changed uuid.uuid() to uuid.uuid4()
        filename = str(uuid.uuid4()) + ".jpg"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        
        print(f"\n🔍 Processing image: {filename}")
        print(f"   Saved to: {upload_path}")

        # RUN YOLO DETECTION USING THE HELPER FUNCTION
        # Use VERY low confidence (0.05) to catch all detections
        results = model(upload_path, conf=0.05)
        print(f"   Got {len(results)} result(s) from model")
        print(f"   Boxes in result: {len(results[0].boxes)}")
        
        counts = count_detections(results)
        print(f"   Detections: {counts}")

        # annotated images
        annotated = results[0].plot()
        print(f"   Annotated image shape: {annotated.shape if annotated is not None else 'None'}")
        
        result_path = os.path.join(RESULT_FOLDER, filename)
        success = cv2.imwrite(result_path, annotated)
        print(f"   Image write success: {success}")
        
        print(f"   ✅ Result saved to: {result_path}")

        # FIXED: Added missing comma between dictionary elements
        return jsonify({"result_image": "/" + result_path, "counts": counts})
    
    except Exception as e:
        print(f"❌ Error in predict_image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Video prediction
# FIXED: Added missing leading slash in the route
@app.route("/predict/video", methods=["POST"])
def predict_video():
    try:
        # FIXED: Stripped accidental whitespace from key check
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        # FIXED: Changed "files" to "file" to match the guard block
        file = request.files["file"]
        
        # FIXED: Fixed typo from "confifence" to "confidence"
        # LOWERED default to 0.05 for maximum detection sensitivity
        conf = float(request.form.get("confidence", 0.05))
        
        print(f"\n🎥 Processing video with confidence: {conf}")

        filename = str(uuid.uuid4()) + ".mp4"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        
        print(f"   Saved to: {upload_path}")

        cap = cv2.VideoCapture(upload_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
        
        print(f"   Video: {width}x{height} @ {fps}fps")

        result_path = os.path.join(RESULT_FOLDER, "annotated_" + filename)

        out = cv2.VideoWriter(
            result_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        # FIXED: Added "D30" to ensure dictionary keys match frame_counts exactly
        counts = {"D00": 0, "D10": 0, "D20": 0, "D30": 0, "D40": 0, "OTHER": 0}

        frame_count = 0
        # Process Frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Use low confidence (0.05) for maximum detection
            results = model(frame, conf=conf)
            frame_counts = count_detections(results)

            # Aggregate counts
            for k in counts:
                counts[k] += frame_counts[k]

            annotated = results[0].plot()
            out.write(annotated)
            
            if frame_count % 30 == 0:
                print(f"   Processed {frame_count} frames...")

        cap.release()
        out.release()
        
        print(f"   ✅ Video processed: {frame_count} frames")
        print(f"   Detections: {counts}")

        return jsonify({"result_video": "/" + result_path, "counts": counts})
    
    except Exception as e:
        print(f"❌ Error in predict_video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# RunApp
# FIXED: Corrected spelling to standard "__main__" block
if __name__ == "__main__":
    app.run(debug=True)


    

