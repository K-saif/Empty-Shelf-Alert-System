# utils for main.py
from ultralytics import YOLO
import cv2
import numpy as np

def calculate_area(results):

    # Storage for total pixel areas
    area_class0 = 0  # empty slots
    area_class1 = 0  # products

    for r in results:
        boxes = r.boxes  # detections
        
        for box in boxes:
            cls = int(box.cls[0].item())     # class id
            x1, y1, x2, y2 = box.xyxy[0]      # pixel coordinates
            
            # compute area of bounding box
            area = (x2 - x1) * (y2 - y1)
            
            # accumulate by class
            if cls == 0:
                area_class0 += area
            elif cls == 1:
                area_class1 += area

    # Calculate percentages
    empty_area = area_class0/area_class1 *100
    print(f"Empty area percentage: {empty_area:.2f}%")

    product_area = area_class1/(area_class0 + area_class1) *100
    print(f"Product area percentage: {product_area:.2f}%")

    return empty_area, product_area






# =========================
# UTILITY FUNCTIONS
# =========================
def get_center(box):
    """Return the (cx, cy) center of a bounding box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def resize_mask_to_image(mask, orig_w, orig_h):
    """Resize a binary mask back to original image resolution."""
    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


# =========================
# PROCESS ROW MASKS
# =========================
def extract_row_boxes(seg_res, img):
    """Extract bounding boxes from segmentation masks and return filtered row boxes sorted top→bottom."""
    row_masks = seg_res.masks
    if row_masks is None:
        return None

    orig_h, orig_w = img.shape[:2]
    boxes = []

    for mask in row_masks.data:
        mask_np = resize_mask_to_image(mask.cpu().numpy().astype(np.uint8), orig_w, orig_h)
        ys, xs = np.where(mask_np == 1)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        boxes.append((x1, y1, x2, y2))

    if not boxes:
        return None

    # Filter out narrow noisy rows
    widths = [(x2 - x1) for (x1, y1, x2, y2) in boxes]
    avg_width = sum(widths) / len(widths)
    threshold = avg_width / 2

    print(f"[INFO] Average row width: {avg_width:.2f}, filter threshold: {threshold:.2f}")

    filtered_boxes = [box for box, w in zip(boxes, widths) if w >= threshold]
    removed = len(boxes) - len(filtered_boxes)
    if removed > 0:
        print(f"[INFO] Removed {removed} narrow row(s).")

    # Sort rows top → bottom
    filtered_boxes.sort(key=lambda r: r[1])
    print(f"[INFO] Total valid rows: {len(filtered_boxes)}")

    return filtered_boxes


# =========================
# DETECT EMPTY SLOTS
# =========================
def extract_empty_boxes(det_res):
    """Extract bounding boxes where class = 0 (empty slot)."""
    if det_res.boxes is None:
        return None
    empty = []
    for box in det_res.boxes:
        if int(box.cls[0]) == 0:  # class 0 = empty space
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            empty.append((x1, y1, x2, y2))
    print(f"[INFO] Detected {len(empty)} empty slots.")
    return empty


# =========================
# MAP EMPTY SLOTS TO ROWS
# =========================
def map_empty_to_rows(empty_boxes, row_boxes, img_h):
    """Map empty slot boxes to shelf rows using band logic."""
    row_centers = [get_center(box)[1] for box in row_boxes]
    row_centers.sort()

    # Create vertical bands
    bands = [(0, row_centers[0])]  # top → first row
    for i in range(len(row_centers) - 1):
        bands.append((row_centers[i], row_centers[i + 1]))
    bands.append((row_centers[-1], img_h))  # last row → bottom

    row_empty_map = {i: [] for i in range(len(row_centers))}

    for eb in empty_boxes:
        _, cy = get_center(eb)
        for band_index, (low, high) in enumerate(bands):
            if low <= cy < high:
                row_index = min(band_index, len(row_centers) - 1)
                row_empty_map[row_index].append(eb)
                break

    return row_empty_map
