# Retail Shelf Stock & Empty Space Detection (YOLO11)

## 📌 Overview
This project presents a **production-ready retail shelf analytics system** that detects **products and empty shelf spaces** using a unified YOLO11-based object detection model.

The system is designed for **automated shelf monitoring in supermarkets and retail stores**, enabling real-time detection of:
- Product presence
- Empty shelf spaces
- Potential out-of-stock zones

This solution is especially relevant for **large retail chains and smart store deployments**, where manual shelf auditing is inefficient and error-prone.

---

## 🎯 Problem Statement
Retail stores frequently suffer from:
- Out-of-stock products
- Poor shelf utilization
- Delayed restocking decisions

Manual shelf inspection does not scale across large supermarkets.  
This project addresses this problem using **computer vision-driven shelf analytics**.

---

## 🚀 Key Features
✔ Product detection on retail shelves  
✔ Empty shelf space detection  
✔ Unified detection model (products + empty space)  
✔ Works on real supermarket shelf images  
✔ Designed for real-time inference on edge devices  

---

## 🧠 Dataset Strategy (Key Innovation)
This project uses a **hybrid dataset construction approach** to enable robust empty shelf detection.

### 1️⃣ Product Detection Dataset
- SKU-based retail product images
- Trained YOLO11 model to detect **products on shelves**

### 2️⃣ Empty Shelf Dataset
- Public dataset from Roboflow:
  - [*Supermarket Empty Shelf Detector*](https://universe.roboflow.com/fyp-ormnr/supermarket-empty-shelf-detector)
- Dataset contains **only empty space annotations**, without product labels

### 3️⃣ Weakly-Supervised Relabeling (Core Contribution)
To unify both datasets:
1. The pretrained **product detection YOLO11 model** is run on empty shelf images
2. Detected products are automatically saved as YOLO labels into existing .txt of empty shelf dataset.
3. Assigned:
   - `class_id = 0` → Empty shelf space (already present)
   - `class_id = 1` → Product (added)
4. Result: A **single, unified dataset** containing both classes

This approach avoids manual relabeling and closely mimics **real-world industrial workflows**.

---

## 🏷 Class Mapping
| Class ID | Label |
|--------|------|
| 0 | Empty Shelf Space |
| 1 | Product |

---

## 📊 Applications
- Automated shelf monitoring
- Out-of-stock detection
- Shelf utilization analysis
- Retail analytics dashboards
- Smart store deployments

---

## ⚙️ Deployment Ready
- Real-time inference supported
- Easily integrable with REST APIs

---
