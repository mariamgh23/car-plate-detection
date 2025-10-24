# 🚗 Car Plate Detection using Deep Learning

An end-to-end project for detecting vehicles and recognizing their license plates in video footage using deep learning, OpenCV, and YOLO.  
This repository includes preprocessing, interpolation, and visualization scripts to track cars and overlay license plate detections smoothly on video.

---

## 📁 Project Structure

car-plate-detection/

│

├── interpolate.py # Interpolates missing bounding boxes for smooth tracking..
├── visualize.py # Draws detection boxes and plate numbers on video frames.
├── test.csv # Raw detection results (input).
├── test_interpolated.csv # Output after interpolation (generated).
├── sample.mp4 # Input video for detection visualization.
├── out.mp4 # Final rendered output video.
└── README.md # Documentation file.


---

## 🎯 Features

- 🚘 Detects and tracks multiple vehicles frame by frame  
- 🔍 Identifies and crops license plates  
- 🧮 Fills in missing bounding box frames using interpolation  
- 🎨 Draws visually appealing bounding boxes and labels  
- 💾 Generates annotated output video (`out.mp4`)

---

## 🧠 Workflow Overview

1. **Detection Output (YOLO or similar)**  
   - A CSV file (`test.csv`) containing car and plate bounding boxes per frame.

2. **Interpolation (`interpolate.py`)**  
   - Fills missing bounding boxes for frames where detections were missed.
   - Produces `test_interpolated.csv`.

3. **Visualization (`visualize.py`)**  
   - Reads interpolated results and the original video.
   - Draws bounding boxes and overlays license plate crops + text.
   - Saves final video as `out.mp4`.

---

## 🧩 Example CSV Structure

| frame_nmr | car_id | car_bbox               | license_plate_bbox      | license_number | license_number_score |
|------------|---------|------------------------|--------------------------|----------------|----------------------|
| 10         | 1       | [320 180 520 340]     | [370 300 430 340]       | ABC123         | 0.98                 |

---

Got it ✅ — here’s your entire section rewritten as **one clean, properly structured Markdown block**, perfect to paste directly into your `README.md` (it’ll render beautifully on GitHub):

---

````markdown
## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mariamgh23/car-plate-detection.git
cd car-plate-detection
````

---

### 2️⃣ Install Dependencies

You can create a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

Then install required packages:

```bash
pip install -r requirements.txt
```

💡 *If you don’t have a `requirements.txt` yet, install these core libraries manually:*

```bash
pip install numpy pandas scipy opencv-python ultralytics torch
```

---

## ▶️ How to Run

### Step 1. Interpolate Missing Frames

```bash
python interpolate.py
```

### Step 2. Visualize Results

```bash
python visualize.py
```

✅ This reads `sample.mp4` and creates `out.mp4` with bounding boxes and license numbers.

---

## 🧰 Key Functions

### 🟦 `interpolate_bounding_boxes()` — *(in `interpolate.py`)*

* Takes raw detection data.
* Uses `scipy.interpolate.interp1d()` to fill missing bounding boxes linearly.
* Ensures every car has continuous frame data from first to last appearance.

### 🟩 `draw_border()` — *(in `visualize.py`)*

* Draws stylized corner borders around each detected vehicle.
* Enhances visual clarity with adjustable line length and color.

---

## 🧾 Requirements Summary

| Library                | Purpose                                      |
| ---------------------- | -------------------------------------------- |
| `opencv-python`        | Video I/O and drawing                        |
| `numpy`                | Numerical operations                         |
| `pandas`               | CSV data handling                            |
| `scipy`                | Bounding box interpolation                   |
| `torch`, `ultralytics` | YOLO model support (if using live detection) |




