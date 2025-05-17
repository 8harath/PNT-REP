import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import cvzone
from pathlib import Path
from car_detector import CarDetector

class EnhancedParkingDetector:
    def __init__(self, video_path=None, image_path=None):
        self.video_path = video_path
        self.image_path = image_path
        self.width, self.height = 120, 43
        self.posList = []
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_rectangles = []
        self.car_detector = CarDetector()
        self.load_parking_positions()
        self.show_help = True
        self.history = []  # Add history for undo functionality
        self.is_reset = False  # Add reset state flag
        
        # Add zoom functionality
        self.zoom_scale = 1.0
        self.zoom_step = 0.1
        self.min_zoom = 0.5
        self.max_zoom = 3.0
        self.pan_x = 0
        self.pan_y = 0
        self.pan_step = 50
        self.is_panning = False
        self.last_mouse_pos = None
        
    def load_parking_positions(self):
        
        try:
            with open('CarParkPos', 'rb') as f:
                self.posList = pickle.load(f)
        except:
            self.posList = []
            
    def save_parking_positions(self):
        # Save current state to history before saving
        self.history.append(self.posList.copy())
        with open('CarParkPos', 'wb') as f:
            pickle.dump(self.posList, f)

    def add_help_text(self, img):
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Add numbered help text in the right corner
        help_text = [
            "=== Available Shortcuts ===",
            "1. R: Reset all selections",
            "2. Z: Undo last selection",
            "3. D: Detect & generate report",
            "4. S: Save current layout",
            "",
            "=== Mouse Controls ===",
            "• Left-click + Drag: Select spaces",
            "• Right-click: Remove space"
        ]
        
        # Calculate starting position (right corner)
        start_x = width - 300  # 300 pixels from right edge
        y_offset = 30
        
        # Add text with background for better visibility
        for text in help_text:
            if text.startswith("==="):
                # Make section headers more prominent
                cvzone.putTextRect(img, text, (start_x, y_offset), scale=1.5,
                                 thickness=2, offset=5, colorR=(0,0,0))
            else:
                cvzone.putTextRect(img, text, (start_x, y_offset), scale=1.2,
                                 thickness=2, offset=5, colorR=(0,0,0))
            y_offset += 25  # Spacing between lines

    def apply_zoom_and_pan(self, img):
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Calculate zoomed dimensions
        zoomed_w = int(w * self.zoom_scale)
        zoomed_h = int(h * self.zoom_scale)
        
        # Resize image
        zoomed_img = cv2.resize(img, (zoomed_w, zoomed_h))
        
        # Calculate pan boundaries
        max_pan_x = max(0, zoomed_w - w)
        max_pan_y = max(0, zoomed_h - h)
        
        # Clamp pan values
        self.pan_x = max(0, min(self.pan_x, max_pan_x))
        self.pan_y = max(0, min(self.pan_y, max_pan_y))
        
        # Crop image based on pan
        if zoomed_w > w or zoomed_h > h:
            x1 = self.pan_x
            y1 = self.pan_y
            x2 = min(x1 + w, zoomed_w)
            y2 = min(y1 + h, zoomed_h)
            zoomed_img = zoomed_img[y1:y2, x1:x2]
            
            # If the cropped image is smaller than the window, pad it
            if zoomed_img.shape[0] < h or zoomed_img.shape[1] < w:
                padded_img = np.zeros((h, w, 3), dtype=np.uint8)
                padded_img[:zoomed_img.shape[0], :zoomed_img.shape[1]] = zoomed_img
                zoomed_img = padded_img
        
        return zoomed_img

    def mouse_callback(self, event, x, y, flags, param):
        if self.is_panning:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.last_mouse_pos = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.last_mouse_pos is not None:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                self.pan_x = max(0, self.pan_x - dx)
                self.pan_y = max(0, self.pan_y - dy)
                self.last_mouse_pos = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.last_mouse_pos = None
            return

        # Adjust coordinates for border
        border_size = 50
        x -= border_size
        y -= border_size

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)  # Initialize end_point immediately
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and self.end_point:
                # Calculate grid of parking spaces
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                
                # Ensure x1,y1 is top-left and x2,y2 is bottom-right
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Calculate number of spaces that fit
                num_spaces_x = (x2 - x1) // self.width
                num_spaces_y = (y2 - y1) // self.height
                
                # Add all spaces in the grid
                for i in range(num_spaces_x):
                    for j in range(num_spaces_y):
                        pos = (x1 + i * self.width, y1 + j * self.height)
                        if pos not in self.posList:
                            self.posList.append(pos)
                
                self.save_parking_positions()
                self.start_point = None
                self.end_point = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove spaces in the clicked area
            for pos in self.posList[:]:
                x1, y1 = pos
                if x1 < x < x1 + self.width and y1 < y < y1 + self.height:
                    self.posList.remove(pos)
            self.save_parking_positions()

    def process_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 16)
        # Apply median blur to remove noise
        median = cv2.medianBlur(thresh, 5)
        # Dilate to connect components
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(median, kernel, iterations=1)
        return dilated

    def check_parking_space(self, img_pro, img):
        space_counter = 0
        occupied_slots = 0
        
        for pos in self.posList:
            x, y = pos
            img_crop = img_pro[y:y + self.height, x:x + self.width]
            count = cv2.countNonZero(img_crop)
            
            # Enhanced threshold for better detection
            if count < 900:
                color = (0, 255, 0)  # Green for empty
                thickness = 5
                space_counter += 1
            else:
                color = (0, 0, 255)  # Red for occupied
                thickness = 2
                occupied_slots += 1
                
            # Draw rectangle and count
            cv2.rectangle(img, pos, (pos[0] + self.width, pos[1] + self.height), color, thickness)
            cvzone.putTextRect(img, str(count), (x, y + self.height - 3), scale=1,
                             thickness=2, offset=0, colorR=color)
            
        # Display statistics
        cvzone.putTextRect(img, f'Free: {space_counter}/{len(self.posList)}', (100, 50), scale=3,
                         thickness=5, offset=20, colorR=(0,200,0))
        
        return space_counter, occupied_slots

    def generate_csv_report(self, total_slots, occupied_slots, available_slots):
        data = {
            'Total Slots': [total_slots],
            'Occupied Slots': [occupied_slots],
            'Available Slots': [available_slots],
            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        df = pd.DataFrame(data)
        df.to_csv('parking_status.csv', index=False)

    def process_video(self):
        if not self.video_path:
            raise ValueError("Video path not provided")
            
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create a window that can be resized
        cv2.namedWindow("Parking Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking Detection", frame_width, frame_height)
        
        while True:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            success, img = cap.read()
            if not success:
                break
                
            processed_img = self.process_frame(img)
            available_slots, occupied_slots = self.check_parking_space(processed_img, img)
            
            # Generate CSV report every 30 frames
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
                self.generate_csv_report(
                    total_slots=len(self.posList),
                    occupied_slots=occupied_slots,
                    available_slots=available_slots
                )
            
            cv2.imshow("Parking Detection", img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def process_image(self):
        if not self.image_path:
            raise ValueError("Image path not provided")
            
        # Read image with full resolution
        img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image from {self.image_path}")
            
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Calculate window size with borders
        border_size = 50
        window_width = width + 2 * border_size
        window_height = height + 2 * border_size
        
        # Create a window that can be resized
        cv2.namedWindow("Parking Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Parking Detection", window_width, window_height)
        
        # Set mouse callback
        cv2.setMouseCallback("Parking Detection", self.mouse_callback)
        
        while True:
            # Create a bordered image
            bordered_img = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            bordered_img[border_size:border_size+height, border_size:border_size+width] = img.copy()
            
            # Only draw parking spaces if there are any and not in reset state
            if self.posList and not self.is_reset:
                for pos in self.posList:
                    x, y = pos
                    # Adjust coordinates for border
                    x += border_size
                    y += border_size
                    cv2.rectangle(bordered_img, (x, y), (x + self.width, y + self.height), (255, 0, 255), 2)
            
            # Draw selection rectangle if drawing
            if self.drawing and self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                # Adjust coordinates for border
                x1 += border_size
                y1 += border_size
                x2 += border_size
                y2 += border_size
                cv2.rectangle(bordered_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw grid preview
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                num_spaces_x = (x2 - x1) // self.width
                num_spaces_y = (y2 - y1) // self.height
                
                for i in range(num_spaces_x):
                    for j in range(num_spaces_y):
                        pos_x = x1 + i * self.width
                        pos_y = y1 + j * self.height
                        cv2.rectangle(bordered_img, (pos_x, pos_y), 
                                    (pos_x + self.width, pos_y + self.height), (0, 255, 0), 1)
            
            # Add essential information only if not in reset state
            if not self.is_reset:
                total_spaces = len(self.posList)
                
                # Only process image and show statistics if there are parking spaces
                if total_spaces > 0:
                    processed_img = self.process_frame(img)
                    available_slots, occupied_slots = self.check_parking_space(processed_img, img)
                    empty_spaces = available_slots
                    
                    # Display statistics in the top-right corner
                    cvzone.putTextRect(bordered_img, f'Total Slots: {total_spaces}', 
                                     (window_width - 250, 30), scale=2, thickness=3, offset=10, colorR=(0,200,0))
                    cvzone.putTextRect(bordered_img, f'Empty Slots: {empty_spaces}', 
                                     (window_width - 250, 80), scale=2, thickness=3, offset=10, colorR=(0,200,0))
            
            # Show help text if enabled
            if self.show_help:
                self.add_help_text(bordered_img)
            
            # Apply zoom and pan
            display_img = self.apply_zoom_and_pan(bordered_img)
            
            cv2.imshow("Parking Detection", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset all selections and state
                self.posList = []
                self.history = []
                self.save_parking_positions()
                self.drawing = False
                self.start_point = None
                self.end_point = None
                self.is_reset = True  # Set reset state
                
                # Show reset message
                cvzone.putTextRect(bordered_img, "All selections reset!", 
                                 (window_width//2 - 150, window_height - 50), 
                                 scale=3, thickness=3, offset=10, colorR=(0,255,0))
                cv2.imshow("Parking Detection", bordered_img)
                cv2.waitKey(1000)  # Show message for 1 second
            elif key == ord('z'):
                # Undo last selection
                if self.history:
                    self.posList = self.history.pop()
                    self.save_parking_positions()
                    self.is_reset = False  # Exit reset state
                    # Show undo message
                    cvzone.putTextRect(bordered_img, "Last selection undone!", 
                                     (window_width//2 - 150, window_height - 50), 
                                     scale=2, thickness=2, offset=10, colorR=(0,255,0))
                    cv2.imshow("Parking Detection", bordered_img)
                    cv2.waitKey(500)  # Show message for 0.5 seconds
            elif key == ord('s'):
                # Save current layout
                self.save_parking_positions()
                self.is_reset = False  # Exit reset state
                cvzone.putTextRect(bordered_img, "Layout Saved!", 
                                 (window_width//2 - 100, window_height - 50), 
                                 scale=2, thickness=2, offset=10, colorR=(0,255,0))
                cv2.imshow("Parking Detection", bordered_img)
                cv2.waitKey(1000)  # Show message for 1 second
            elif key == ord('d'):
                if len(self.posList) == 0:
                    print("Please select parking spaces first!")
                    continue
                    
                # Process the image and detect cars
                processed_img = self.process_frame(img)
                available_slots, occupied_slots = self.check_parking_space(processed_img, img)
                
                # Run ML-based car detection
                results = self.car_detector.detect_cars(img)
                detections, space_status = self.car_detector.process_detections(results, self.posList, img)
                
                # Generate comprehensive report
                report_path, text_report_path = self.car_detector.generate_report(
                    img, self.posList, detections, space_status
                )
                
                print(f"\nReport generated successfully!")
                print(f"Visual report saved to: {report_path}")
                print(f"Text report saved to: {text_report_path}")
                
                # Generate CSV report
                self.generate_csv_report(
                    total_slots=len(self.posList),
                    occupied_slots=occupied_slots,
                    available_slots=available_slots
                )
                
                # Show success message
                cvzone.putTextRect(bordered_img, "Report Generated!", 
                                 (window_width//2 - 150, window_height - 50), 
                                 scale=3, thickness=3, offset=10, colorR=(0,255,0))
                cv2.imshow("Parking Detection", bordered_img)
                cv2.waitKey(2000)  # Show message for 2 seconds
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    detector = EnhancedParkingDetector(
        video_path='carPark.mp4',
        image_path='carParkImg.png'
    )
    
    # Process image for parking space selection
    detector.process_image() 