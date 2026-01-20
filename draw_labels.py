# To visualize images with labels draw by model

import cv2
import os
from pathlib import Path

IMAGE_DIR = "/images/"
LABEL_DIR = "/labels"
OUTPUT_DIR = "/visualize/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class_names = {0: "product"}

img_paths = list(Path(IMAGE_DIR).glob("*.*"))

for img_path in img_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read image: {img_path}")
        continue

    h, w, _ = img.shape

    label_path = Path(LABEL_DIR) / (img_path.stem + ".txt")

    if not label_path.exists():
        print(f"No label for {img_path.name}")
        continue

    with open(label_path, "r") as f:
        annotations = f.readlines()

    for ann in annotations:
        values = ann.strip().split()

        if len(values) == 5:
            cls, x_c, y_c, bw, bh = map(float, values)
        elif len(values) == 6:
            cls, x_c, y_c, bw, bh, conf = map(float, values)
        else:
            print(f"Skipping malformed line in {label_path}: {values}")
            continue

        x_c *= w
        y_c *= h
        bw *= w
        bh *= h

        x1 = int(x_c - bw/2)
        y1 = int(y_c - bh/2)
        x2 = int(x_c + bw/2)
        y2 = int(y_c + bh/2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

        cls_name = class_names.get(int(cls), str(int(cls)))
        cv2.putText(img, cls_name, (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

    save_path = Path(OUTPUT_DIR) / img_path.name
    cv2.imwrite(str(save_path), img)
    print("✔ saved:", save_path)
