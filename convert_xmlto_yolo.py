# Import Libraries
import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image  # Accessing image dimensions
from tqdm import tqdm  # Live progress bars on terminal

# Final class mapping
CLASS_MAP = {
    "D00": 0,  # Longitudinal crack
    "D10": 1,  # Transverse
    "D20": 2,  # Alligator
    "D40": 3,  # Pothole
    "D43": 4,  # Other corruptions -> merged
    "D44": 4,
    "D50": 4,
}

OUTPUT_DIR = "dataset"


# Convert bounding boxes from Pascal VOC (xmin, xmax, ymin, ymax) to YOLO format
def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    xmin, xmax, ymin, ymax = box

    # Calculate center coordinates and width/height dimensions
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    # Return normalized values
    return (x * dw, y * dh, w * dw, h * dh)


# Helper function to find existing case-insensitive directories
def find_dir(parent, possibilities):
    for p in possibilities:
        path = os.path.join(parent, p)
        if os.path.exists(path):
            return path
    return None


# Convert Train
def convert_train():
    # Automatically locate raw Data directory ignoring uppercase/lowercase mistakes
    base_data_dir = find_dir(".", ["Data", "data", "DATA"])
    if not base_data_dir:
        print("❌ CRITICAL ERROR: Could not find your 'Data' or 'data' folder in the root directory!")
        print(f"Current working folder contents: {os.listdir('.')}")
        return

    train_dir = find_dir(base_data_dir, ["train", "Train", "TRAIN"])
    if not train_dir:
        print(f"❌ CRITICAL ERROR: Could not find a 'train' folder inside '{base_data_dir}'!")
        return

    img_dir = find_dir(train_dir, ["images", "Images", "IMAGES"])
    ann_root = find_dir(train_dir, ["annotations", "Annotations", "ANNOTATIONS"])

    if not img_dir or not ann_root:
        print(f"❌ CRITICAL ERROR: Ensure '{train_dir}' contains an 'images' folder AND an 'annotations' folder.")
        print(f"Found inside train folder: {os.listdir(train_dir)}")
        return

    out_img_dir = os.path.join(OUTPUT_DIR, "images", "train")
    out_lbl_dir = os.path.join(OUTPUT_DIR, "labels", "train")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    xml_files = []
    for root_dir, _, files in os.walk(ann_root):
        for file in files:
            if file.endswith(".xml") or file.endswith(".XML"):
                xml_files.append(os.path.join(root_dir, file))

    print(f"Found {len(xml_files)} XML FILES inside {ann_root}")
    if len(xml_files) == 0:
        print(f"⚠️ Warning: Checked '{ann_root}' but it contains no XML files.")
        return

    for xml_path in tqdm(xml_files, desc="Converting TRAIN"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            img_name_node = root.find("filename")
            if img_name_node is None or not img_name_node.text:
                # Fallback if filename node is broken: use xml filename with .jpg
                img_name = os.path.basename(xml_path).rsplit(".", 1)[0] + ".jpg"
            else:
                img_name = img_name_node.text

            img_path = os.path.join(img_dir, img_name)

            # EMERGENCY FALLBACK: If real file extension differs from XML text (e.g. PNG vs JPG)
            if not os.path.exists(img_path):
                base_no_ext = os.path.splitext(img_name)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    alt_path = os.path.join(img_dir, base_no_ext + ext)
                    if os.path.exists(alt_path):
                        img_path = alt_path
                        break

            if not os.path.exists(img_path):
                # Silent continue to keep execution clean, change to print() if debugging
                continue

            img = Image.open(img_path)
            w, h = img.size

            label_name = os.path.basename(xml_path).rsplit(".", 1)[0] + ".txt"
            label_path = os.path.join(out_lbl_dir, label_name)

            with open(label_path, "w") as f:
                for obj in root.findall("object"):
                    cls_name_node = obj.find("name")
                    if cls_name_node is None:
                        continue
                    cls_name = cls_name_node.text

                    if cls_name not in CLASS_MAP:
                        continue

                    cls_id = CLASS_MAP[cls_name]
                    xmlbox = obj.find("bndbox")
                    if xmlbox is None:
                        continue

                    xmin = float(xmlbox.find("xmin").text)
                    xmax = float(xmlbox.find("xmax").text)
                    ymin = float(xmlbox.find("ymin").text)
                    ymax = float(xmlbox.find("ymax").text)

                    bb = convert_box((w, h), (xmin, xmax, ymin, ymax))
                    f.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

            shutil.copy(img_path, out_img_dir)
        except Exception as e:
            # Skip corrupted individual XML profiles safely
            continue


# Convert test
def convert_test():
    base_data_dir = find_dir(".", ["Data", "data", "DATA"])
    if not base_data_dir:
        return

    test_dir = find_dir(base_data_dir, ["test", "Test", "TEST"])
    if not test_dir:
        print(f"ℹ️ Skipping test preparation: No 'test' split folder identified in '{base_data_dir}'")
        return

    img_dir = find_dir(test_dir, ["images", "Images", "IMAGES"])
    if not img_dir:
        # Fallback: check if test folder directly contains images
        img_dir = test_dir

    out_img_dir = os.path.join(OUTPUT_DIR, "images", "test")
    os.makedirs(out_img_dir, exist_ok=True)

    files = os.listdir(img_dir)
    valid_imgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    if not valid_imgs:
        print(f"⚠️ Warning: No compatible raw images found inside test directory: '{img_dir}'")
        return

    for img_file in tqdm(valid_imgs, desc="Copying TEST images"):
        img_path = os.path.join(img_dir, img_file)
        shutil.copy(img_path, out_img_dir)


# -----------------------------
# Main Execution Guard
# -----------------------------
if __name__ == "__main__":
    print("🚀 Auto-detecting folders and converting TRAIN (XML → YOLO)...")
    convert_train()

    print("🚀 Preparing TEST images...")
    convert_test()

    print("✅ Process cycle finished!")