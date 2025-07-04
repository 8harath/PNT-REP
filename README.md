# **Advanced Car Parking Space Detection System**

## **📌 Overview**

The **Advanced Car Parking Space Detection System** is a computer vision-powered application designed to automate the detection, classification, and reporting of parking slot occupancy from top-down images or video feeds. Leveraging state-of-the-art deep learning models and image processing techniques, it provides a scalable and intelligent solution for modern parking management systems.

---

## **🚀 Key Capabilities**

* Detects and tracks parking spaces in aerial/static/top-down video frames.
* Determines real-time occupancy status using ML-based object detection.
* Handles real-world edge cases: occlusions, misalignments, oversized vehicles, poor lighting.
* Generates multi-format reports: annotated images, textual summaries, and CSVs.
* Offers an interactive interface for defining and modifying slot layouts.

---

## **🎯 Use Cases**

Ideal for integration in:

* Urban parking lots (malls, airports, hospitals)
* Campus and commercial parking zones
* Surveillance systems with vehicle analytics
* Smart city and IoT parking networks

**Benefits:**

* Reduces manual labor and human error.
* Enables data-driven infrastructure planning.
* Seamlessly integrates into existing monitoring systems.
* Improves user experience with accurate availability updates.

---

## **🔧 Core Features**

### 1. **Interactive Slot Layout Editor**

* Drag-to-select multiple slots.
* Grid-based automatic parking slot suggestions.
* Right-click to remove individual slots.
* One-key commands for clearing, resetting, and saving layouts.

### 2. **ML-Based Vehicle Detection**

* Employs **YOLOv8** for real-time vehicle detection.
* Trained on the **COCO dataset** (detects cars, motorcycles, buses, trucks).
* Configurable confidence thresholds.
* Supports frame-wise and batch detection.

### 3. **Comprehensive Reporting Engine**

* **Visual Reports**: Annotated images with occupancy overlay and charts.
* **Text Reports**: Slot occupancy, counts, detection confidence stats.
* **CSV Reports**: Structured output for downstream analytics or dashboards.

### 4. **Robust Edge Case Handling**

* Detects occluded or partially visible vehicles.
* Adapts to varying parking orientations and lighting.
* Supports reserved and special-use zones (e.g., handicapped).

---

## **🧠 Technical Approach**

### Machine Learning

* **Model**: YOLOv8 via Ultralytics framework
* **Classes**: Cars, trucks, buses, motorcycles (from COCO dataset)

### Computer Vision (OpenCV)

* Image preprocessing: Grayscale, Gaussian/Medians blurs, dilation.
* Adaptive thresholding for improved contrast under variable lighting.

### Parking Slot Analysis

* Grid-based detection using pixel intensity and object overlap logic.
* Dynamic space sizing based on layout and image resolution.

---

## **🛠 Workflow Overview**

### Step 1: Setup & Initialization

* Clone the repository
* Install dependencies
* Launch the GUI interface

### Step 2: Slot Layout Configuration

* Use mouse controls to draw and edit parking slots on images.

### Step 3: Detection & Reporting

* Trigger ML-based detection
* Auto-generate:

  * `*.png` → Annotated visual report
  * `*.txt` → Slot-level statistics and summary
  * `*.csv` → Structured occupancy data

### Step 4: Analysis & Integration

* View live overlays or batch-process multiple inputs.
* Export CSVs for integration with dashboards or business logic.

---

## **📂 Project Structure**

```
PNT-REP/
├── attached_assets/
│   ├── enhanced_parking_detector.py   # Main GUI and logic controller
│   ├── car_detector.py                # ML integration and inference
│   ├── requirements.txt               # Python dependencies
│   ├── carPark.mp4                    # Sample video feed
│   └── carParkImg.png                 # Sample input image
├── static/                            # (Optional) Web UI static files
├── templates/                         # (Optional) Web template views
├── reports/                           # Output: PNG, TXT, CSV reports
└── ...
```

---

## **📦 Installation**

### 1. Clone the Repository

```bash
git clone https://github.com/8harath/PNT-REP.git
cd PNT-REP/attached_assets
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Note: On first use, the YOLOv8 model will auto-download.

---

## **▶️ Usage Guide**

### Run the Interface

```bash
python enhanced_parking_detector.py
```

### Controls & Shortcuts

| Action                    | Key/Mouse         |
| ------------------------- | ----------------- |
| Select parking slots      | Left-click + drag |
| Remove a slot             | Right-click       |
| Clear all slots           | `c`               |
| Run detection & reporting | `d`               |
| Reset layout              | `r`               |
| Undo previous slot        | `z`               |
| Save current layout       | `s`               |
| Exit application          | `q`               |

---

## **📊 Output Formats**

* `parking_report_<timestamp>.png`: Visual layout with stats overlay
* `parking_report_<timestamp>.txt`: Textual summary
* `parking_status.csv`: Machine-readable occupancy dataset

---

## **📚 Dependencies**

* **ML & Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* **Vision & Processing**: `OpenCV`
* **Visualization**: `Matplotlib`, `Seaborn`
* **Data Handling**: `Pandas`, `NumPy`

---

## **🙌 Credits & Acknowledgments**

* Built upon frameworks and ideas from:

  * [Murtaza’s Computer Vision Zone](https://www.computervision.zone/)
  * [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* Contributors: Bharath & collaborators
* Open-source libraries: OpenCV, YOLOv8, Matplotlib, Pandas, Seaborn

---

## **📁 Sample Outputs & Demos**

Explore sample visual and textual reports in the `/reports/` directory. For advanced usage or batch integration, refer to inline comments in the Python scripts.

---

## **📬 Feedback & Contributions**

This project is actively maintained. For suggestions, bugs, or feature requests, feel free to:

* Open an issue on the repository
* Submit a pull request
* Contact the maintainer via GitHub profile

