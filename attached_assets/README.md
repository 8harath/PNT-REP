# Advanced Car Parking Space Detection System

This project provides an intelligent parking space detection system that combines computer vision and machine learning to analyze parking lots. It can process both images and video feeds to detect parking spaces, identify vehicles, and generate comprehensive reports about parking lot utilization.

## Features

### 1. Interactive Parking Space Selection
- Drag-and-select multiple parking spaces at once
- Grid-based automatic space allocation
- Visual feedback during selection
- Right-click to remove individual spaces
- Clear all spaces with a single keypress

### 2. ML-Based Vehicle Detection
- Uses YOLOv8 for accurate vehicle detection
- Detects multiple vehicle types:
  - Cars
  - Motorcycles
  - Buses
  - Trucks
- Provides confidence scores for each detection

### 3. Comprehensive Reporting
- Visual reports with multiple plots:
  1. Parking lot with detections
  2. Parking statistics
  3. Detection confidence distribution
  4. Vehicle type distribution
- Detailed text reports
- CSV data export
- Real-time statistics display

### 4. Edge Case Handling
- Irregular parking detection
- Large vehicle handling
- Environmental variations
- Lighting condition adaptation
- Moving vehicle detection
- Special parking zones

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/car-parking-detection.git
   cd car-parking-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model (automatically handled on first run)

## Usage

### 1. Select Parking Spaces

Run the main script:
```bash
python enhanced_parking_detector.py
```

#### Interactive Controls:
- **Left-click and drag**: Select multiple parking spaces
- **Right-click**: Remove individual parking spaces
- **'c' key**: Clear all parking spaces
- **'d' key**: Detect cars and generate reports
- **'q' key**: Quit the application

### 2. Generate Reports

Press 'd' to:
- Detect vehicles in the parking lot
- Generate comprehensive reports
- Save visual and text reports
- Update CSV data

### 3. Report Contents

#### Visual Report (`parking_report_[timestamp].png`):
- Parking lot visualization with detections
- Parking statistics bar chart
- Detection confidence distribution
- Vehicle type distribution pie chart

#### Text Report (`parking_report_[timestamp].txt`):
- Total parking spaces
- Detected vehicles count
- Available spaces
- Vehicle type distribution
- Detection confidence statistics

#### CSV Data (`parking_status.csv`):
- Total slots
- Occupied slots
- Available slots
- Timestamp

## Project Structure

```
car-parking-detection/
├── enhanced_parking_detector.py  # Main detection script
├── car_detector.py              # ML-based car detection
├── requirements.txt             # Project dependencies
├── reports/                     # Generated reports directory
├── carPark.mp4                 # Sample video
└── carParkImg.png              # Sample image
```

## Technical Details

### ML Model
- Uses YOLOv8 for vehicle detection
- COCO dataset classes for vehicle types
- Confidence threshold for detections

### Image Processing
- Grayscale conversion
- Gaussian blur
- Adaptive thresholding
- Median blur for noise reduction
- Dilation for component connection

### Space Detection
- Grid-based space allocation
- Automatic space size calculation
- Occupancy detection using pixel analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- Original inspiration from [Murtaza's Computer Vision Zone](https://www.computervision.zone/)
- Enhanced by Bharath K
- YOLOv8 model from [Ultralytics](https://github.com/ultralytics/ultralytics)

## Acknowledgments

- OpenCV for computer vision capabilities
- YOLOv8 for object detection
- Matplotlib and Seaborn for visualization
- Pandas for data handling