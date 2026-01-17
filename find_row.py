from ultralytics import YOLO
import cv2
import numpy as np
from utils import calculate_area, extract_empty_boxes,extract_row_boxes, map_empty_to_rows

# =========================
# CONFIG
# =========================
DET_MODEL_PATH = "/home/medprime/Music/Product-Detection-System/11n_b2/weights/best.pt"
SEG_MODEL_PATH = "/home/medprime/Music/Product-Detection-System/row_11n_b25/weights/best.pt"
IMAGE_PATH = "/home/medprime/Music/Product-Detection-System/data2/test/images/test_14_jpg.rf.7fa6f5873178bc7fbdf6253dedcf343d.jpg"



# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    # Load models
    det_model = YOLO(DET_MODEL_PATH)
    seg_model = YOLO(SEG_MODEL_PATH)

    # Read image
    img = cv2.imread(IMAGE_PATH)
    img_h = img.shape[0]

    # Inference
    det_res = det_model(IMAGE_PATH, save=False, show=False)[0]
    seg_res = seg_model(IMAGE_PATH, save=False, show=False, iou=0.5)[0]

    # Calculate area (optional, for debugging)
    empty_area, product_area = calculate_area(det_res)

    # Extract rows + empty slots
    row_boxes = extract_row_boxes(seg_res, img)
    empty_boxes = extract_empty_boxes(det_res)

    # Map empty to rows
    row_empty_map = map_empty_to_rows(empty_boxes, row_boxes, img_h)

    # Print result
    print("\n========== EMPTY SLOTS PER ROW ==========")
    for row_idx, slots in row_empty_map.items():
        print(f"Row {row_idx + 1}: {len(slots)} empty slots")

    # Optional visualization
    output = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(row_boxes):
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"Row {i+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for (x1, y1, x2, y2) in empty_boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(output, "Empty", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite("output_rows_empty.jpg", output)
    print("[INFO] Saved visualization as output_rows_empty.jpg")
