# To save labels by using trained model

from ultralytics import YOLO
import os
from pathlib import Path

# ---- SETTINGS ----
MODEL_PATH = "best.pt"                   # your trained model
IMAGE_DIR = "/images/"                    # folder containing input images
OUTPUT_LABEL_DIR = "/labels"        # folder to store .txt labes

# create output directory if not exists
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# load model
model = YOLO(MODEL_PATH)

print(model.names)
# get image list
img_paths = list(Path(IMAGE_DIR).glob("*.*"))

for img_path in img_paths:
    results = model(img_path)
    boxes = results[0].boxes.xywhn  # normalized boxes
    
    label_path = Path(OUTPUT_LABEL_DIR) / (img_path.stem + ".txt")
    
    # Check if label file already exists & has content
    file_exists = label_path.exists() and label_path.stat().st_size > 0

    # open file in append ('a') or write ('w') mode
    mode = "a" if file_exists else "w"

    with open(label_path, mode) as f:
        for box in boxes:
            x, y, w, h = box[:4]
            class_id = 1  # only 1 class
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print(f"Processed: {img_path.name} → {'appended' if file_exists else 'created'}")

print("✔ Done!")
