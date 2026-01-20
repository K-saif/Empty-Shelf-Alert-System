import argparse
from ultralytics import YOLO
import cv2
import csv
import time
from datetime import datetime
from utils import calculate_area, extract_empty_boxes, extract_row_boxes, map_empty_to_rows

# =========================
# CONFIG
# =========================
DET_MODEL_PATH = "./Models/detect_product_empty_space.pt"
SEG_MODEL_PATH = "./Models/segment_shelf_rows.pt"
OUTPUT_VIDEO   = "./Output_result.mp4"
OUTPUT_CSV     = "./Results/Monitoring_results.csv"

# =========================
# ARG PARSER
# =========================
parser = argparse.ArgumentParser(description="Shelf Monitoring System")
parser.add_argument("--interval", type=int, required=True, help="Inference interval in seconds")
parser.add_argument("--source", type=str, required=True, help="Video source (URL or webcam index)")
args = parser.parse_args()

INFERENCE_INTERVAL = args.interval
VIDEO_SOURCE = args.source

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    det_model = YOLO(DET_MODEL_PATH)
    seg_model = YOLO(SEG_MODEL_PATH)

    # If source is digit, use webcam index
    try:
        VIDEO_SOURCE = int(VIDEO_SOURCE)
    except:
        pass

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Failed to open video stream: {VIDEO_SOURCE}")
        exit()

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Prepare CSV file with header
    with open(OUTPUT_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "row_number", "empty_slots", "empty_percentage"])

    print(f"▶ Saving Video: {OUTPUT_VIDEO}")
    print(f"▶ Saving CSV: {OUTPUT_CSV}")
    print(f"▶ Interval: {INFERENCE_INTERVAL} sec")
    print(f"▶ Source: {VIDEO_SOURCE}")

    last_inference_time = 0

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("❌ Stream ended or failed, stopping...")
                break

            current_time = time.time()
            out.write(frame)

            if (current_time - last_inference_time) >= INFERENCE_INTERVAL:
                print("⚙ Running inference...")
                last_inference_time = current_time

                img = frame.copy()
                img_h = img.shape[0]

                det_res = det_model(img, save=False, show=False)[0]
                seg_res = seg_model(img, save=False, show=False, iou=0.5)[0]

                empty_area, product_area = calculate_area(det_res)

                row_boxes = extract_row_boxes(seg_res, img)
                if not row_boxes:
                    print("❌Rows not detected, skipping...")
                    continue

                empty_boxes = extract_empty_boxes(det_res)
                if not empty_boxes:
                    print("❌ Empty boxes not detected, skipping...")
                    continue

                row_empty_map = map_empty_to_rows(empty_boxes, row_boxes, img_h)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open(OUTPUT_CSV, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    for row_idx, empty_list in row_empty_map.items():
                        empty_count = len(empty_list)
                        writer.writerow([
                            timestamp,
                            row_idx + 1,
                            empty_count,
                            round(float(empty_area), 4)
                        ])

                print("✔ Inference logged at:", timestamp)

        except Exception as e:
            print("❌ Error during processing:", e)
            continue

    cap.release()
    out.release()
    print("✅ Completed processing!")
