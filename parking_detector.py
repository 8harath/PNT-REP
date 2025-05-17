import cv2
import numpy as np
import logging
from ultralytics import YOLO
import os

class ParkingDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Standard slot dimensions (can be adjusted based on image)
        self.slot_width = 80
        self.slot_height = 40
        
        # Load YOLOv8 model for vehicle detection
        try:
            self.model = YOLO("yolov8n.pt")
            self.logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading YOLOv8 model: {str(e)}")
            self.model = None
            
        # COCO classes relevant for vehicles
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Parameters for slot detection
        self.min_slot_confidence = 0.5
        self.max_slot_area = 20000
        self.min_slot_area = 500
        
    def process_image(self, image_path):
        """
        Process a parking lot image to detect slots and vehicles
        Returns a dictionary with slot information
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate slot dimensions based on image size
        self.adjust_slot_dimensions(width, height)
        
        # Detect parking slots using image processing
        slots = self.detect_parking_slots(image)
        
        # Detect vehicles using YOLOv8
        vehicles = self.detect_vehicles(image)
        
        # Combine slot and vehicle information
        result = self.analyze_slots_and_vehicles(image, slots, vehicles)
        
        return result
    
    def adjust_slot_dimensions(self, image_width, image_height):
        """Adjust slot dimensions based on image size"""
        # Calculate relative dimensions - typically parking slots are about 1/16 of image width
        self.slot_width = int(image_width / 16)
        self.slot_height = int(self.slot_width / 2)  # Parking slots typically have 2:1 aspect ratio
        
        self.logger.debug(f"Adjusted slot dimensions to: {self.slot_width}x{self.slot_height}")
    
    def detect_parking_slots(self, image):
        """Detect parking slots in the image using image processing techniques"""
        self.logger.info("Detecting parking slots...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to get binary image
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 15
        )
        
        # Apply morphological operations to remove noise and enhance slot lines
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape to identify potential parking slots
        parking_slots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_slot_area < area < self.max_slot_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if the aspect ratio matches a typical parking slot
                aspect_ratio = w / h if h > 0 else 0
                if 1.5 < aspect_ratio < 3.5:  # Typical parking slot aspect ratio
                    parking_slots.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area,
                        'contour': contour
                    })
        
        # Use grid analysis to detect missing slots
        grid_slots = self.detect_slots_by_grid(image, parking_slots)
        
        # Combine detected slots with the grid-inferred slots
        all_slots = self.merge_slot_detections(parking_slots, grid_slots)
        
        self.logger.info(f"Detected {len(all_slots)} potential parking slots")
        return all_slots
    
    def detect_slots_by_grid(self, image, detected_slots):
        """Use grid analysis to identify potential parking slots not detected directly"""
        height, width = image.shape[:2]
        
        # Create a mask with the detected slots
        slot_mask = np.zeros((height, width), dtype=np.uint8)
        for slot in detected_slots:
            x, y, w, h = slot['x'], slot['y'], slot['width'], slot['height']
            cv2.rectangle(slot_mask, (x, y), (x + w, y + h), 255, -1)
        
        # Find organized rows and columns of parking slots
        rows, cols = self.find_parking_grid(detected_slots)
        
        # Infer missing slots based on the grid pattern
        inferred_slots = []
        for row in rows:
            for col in cols:
                # Check if this grid cell might contain a slot
                center_x = col
                center_y = row
                
                # Skip if this area already has a detected slot
                region = slot_mask[center_y-self.slot_height//2:center_y+self.slot_height//2, 
                                  center_x-self.slot_width//2:center_x+self.slot_width//2]
                if region.size > 0 and np.sum(region) > 0:
                    continue
                
                # Check for special zone markings
                is_special = self.check_for_special_zone(
                    image, 
                    center_x-self.slot_width//2, 
                    center_y-self.slot_height//2,
                    self.slot_width, 
                    self.slot_height
                )
                
                inferred_slots.append({
                    'x': center_x - self.slot_width//2,
                    'y': center_y - self.slot_height//2,
                    'width': self.slot_width,
                    'height': self.slot_height,
                    'is_special': is_special,
                    'inferred': True
                })
        
        return inferred_slots
    
    def find_parking_grid(self, slots):
        """Find rows and columns in the parking lot (grid analysis)"""
        if not slots:
            return [], []
            
        # Extract centers of slots
        centers = [(slot['x'] + slot['width']//2, slot['y'] + slot['height']//2) for slot in slots]
        
        # Extract X and Y coordinates
        x_coords = [x for x, _ in centers]
        y_coords = [y for _, y in centers]
        
        # Cluster X coordinates to find columns
        cols = self.cluster_coordinates(x_coords, self.slot_width//2)
        
        # Cluster Y coordinates to find rows
        rows = self.cluster_coordinates(y_coords, self.slot_height//2)
        
        return rows, cols
    
    def cluster_coordinates(self, coords, threshold):
        """Cluster 1D coordinates based on proximity"""
        if not coords:
            return []
            
        # Sort coordinates
        sorted_coords = sorted(coords)
        
        # Initial cluster with the first coordinate
        clusters = [[sorted_coords[0]]]
        
        # Cluster remaining coordinates
        for coord in sorted_coords[1:]:
            # Check if the coordinate is close to the last cluster
            if coord - clusters[-1][-1] <= threshold:
                # Add to the last cluster
                clusters[-1].append(coord)
            else:
                # Start a new cluster
                clusters.append([coord])
        
        # Calculate the mean of each cluster
        cluster_means = [sum(cluster) // len(cluster) for cluster in clusters]
        
        return cluster_means
    
    def merge_slot_detections(self, direct_slots, grid_slots):
        """Merge directly detected slots with grid-inferred slots"""
        all_slots = direct_slots.copy()
        
        # Add grid slots that don't overlap with direct slots
        for grid_slot in grid_slots:
            # Check for overlap with existing slots
            overlaps = False
            for direct_slot in direct_slots:
                if self.rectangles_overlap(
                    (grid_slot['x'], grid_slot['y'], grid_slot['width'], grid_slot['height']),
                    (direct_slot['x'], direct_slot['y'], direct_slot['width'], direct_slot['height'])
                ):
                    overlaps = True
                    break
            
            if not overlaps:
                all_slots.append(grid_slot)
        
        return all_slots
    
    def rectangles_overlap(self, rect1, rect2):
        """Check if two rectangles overlap"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)
    
    def detect_vehicles(self, image):
        """Detect vehicles in the image using YOLOv8"""
        if self.model is None:
            self.logger.warning("YOLOv8 model not loaded, skipping vehicle detection")
            return []
        
        self.logger.info("Detecting vehicles with YOLOv8...")
        
        # Run YOLOv8 inference
        results = self.model(image)
        
        vehicles = []
        
        # Process detection results
        if results and len(results) > 0:
            for i, detection in enumerate(results[0].boxes):
                class_id = int(detection.cls)
                confidence = float(detection.conf)
                
                # Only consider vehicle classes with confidence above threshold
                if class_id in self.vehicle_classes and confidence >= self.min_slot_confidence:
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])
                    
                    vehicles.append({
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'class_id': class_id,
                        'confidence': confidence
                    })
        
        self.logger.info(f"Detected {len(vehicles)} vehicles")
        return vehicles
    
    def check_for_special_zone(self, image, x, y, width, height):
        """Check if a region contains special zone markings (e.g., handicapped parking)"""
        # Extract the region
        x = max(0, x)
        y = max(0, y)
        region = image[y:y+height, x:x+width]
        
        if region.size == 0:
            return False
            
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(region.copy(), cv2.COLOR_BGR2HSV)
        
        # Define blue color range for handicapped parking
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Define yellow color range for special zones
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        special_mask = cv2.bitwise_or(blue_mask, yellow_mask)
        
        # Calculate the percentage of special colors
        special_percentage = (np.sum(special_mask > 0) / (width * height)) * 100
        
        return special_percentage > 5  # Return True if more than 5% has special markings
    
    def analyze_slots_and_vehicles(self, image, slots, vehicles):
        """Analyze which slots are occupied by vehicles"""
        height, width = image.shape[:2]
        
        # Create an annotated image for visualization
        annotated_img = image.copy()
        
        # First, identify potential moving vehicles and large vehicles
        moving_vehicles = []
        large_vehicles_list = []
        vehicle_speeds = {}
        
        # Track center points for each vehicle to estimate movement
        for vehicle in vehicles:
            x, y = vehicle['x'], vehicle['y']
            w, h = vehicle['width'], vehicle['height']
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Store center point
            vehicle['center'] = (center_x, center_y)
            
            # Check if vehicle is in a drive lane (usually not in parking slot alignment)
            is_in_drive_lane = True
            for slot in slots:
                if self.calculate_overlap_area(
                    (vehicle['x'], vehicle['y'], vehicle['width'], vehicle['height']),
                    (slot['x'], slot['y'], slot['width'], slot['height'])
                ) > 0:
                    is_in_drive_lane = False
                    break
            
            # Calculate vehicle speed/movement factor (simulated as we don't have multi-frame data)
            # For now, we'll use angle of vehicle as a proxy for movement
            # Vehicles at an angle to the parking grid are more likely to be moving
            if w > h:
                aspect_ratio = w / h if h > 0 else 1
            else:
                aspect_ratio = h / w if w > 0 else 1
                
            # Typical parked cars have aspect ratio close to standard parking slot ratio
            # If very different, might be at an angle (moving or misaligned)
            movement_score = abs(aspect_ratio - 2.0) * 20  # Score from 0-100
            
            # If in drive lane and has a significant movement score
            if is_in_drive_lane and movement_score > 30:
                moving_vehicles.append(vehicle)
                vehicle['is_moving'] = True
                vehicle_speeds[id(vehicle)] = movement_score
            else:
                vehicle['is_moving'] = False
            
            # Check for large vehicles (buses and trucks)
            if vehicle['class_id'] in [5, 7]:  # bus or truck
                large_vehicles_list.append(vehicle)
                vehicle['is_large'] = True
            else:
                vehicle['is_large'] = False
        
        # Process each slot
        for i, slot in enumerate(slots):
            x, y = slot['x'], slot['y']
            w, h = slot['width'], slot['height']
            
            # Check for large vehicles in this area
            large_vehicle = False
            large_vehicle_class = None
            
            # Check if slot is occupied by any vehicle
            occupied = False
            occupying_vehicle = None
            
            # Is this slot affected by a moving vehicle?
            affected_by_moving = False
            
            for vehicle in vehicles:
                # Skip moving vehicles - they shouldn't count as occupying parking spaces
                if vehicle.get('is_moving', False) and vehicle in moving_vehicles:
                    # Check if moving vehicle is overlapping with this slot
                    if self.rectangles_overlap(
                        (x, y, w, h),
                        (vehicle['x'], vehicle['y'], vehicle['width'], vehicle['height'])
                    ):
                        affected_by_moving = True
                    continue
                
                # Check for overlap between slot and vehicle
                if self.rectangles_overlap(
                    (x, y, w, h),
                    (vehicle['x'], vehicle['y'], vehicle['width'], vehicle['height'])
                ):
                    # Calculate overlap area
                    overlap_area = self.calculate_overlap_area(
                        (x, y, w, h),
                        (vehicle['x'], vehicle['y'], vehicle['width'], vehicle['height'])
                    )
                    
                    # Calculate percentage of slot covered by vehicle
                    slot_area = w * h
                    overlap_percentage = (overlap_area / slot_area) * 100
                    
                    # Enhanced logic for different scenarios:
                    # 1. Regular parked vehicle: >30% overlap
                    # 2. Misaligned parking: even with less overlap, if center point is within slot
                    # 3. Large vehicles might partially overlap multiple slots
                    
                    # Check for center point of vehicle in the slot (misaligned parking)
                    center_in_slot = (x <= vehicle['center'][0] <= x+w and 
                                     y <= vehicle['center'][1] <= y+h)
                    
                    # Consider slot occupied if:
                    if (overlap_percentage > 30 or  # Significant overlap
                        (overlap_percentage > 15 and center_in_slot) or  # Misaligned but center in slot 
                        (vehicle.get('is_large', False) and overlap_percentage > 20)):  # Large vehicle
                        
                        occupied = True
                        occupying_vehicle = vehicle
                        
                        # Check if this is a large vehicle (bus or truck)
                        if vehicle['class_id'] in [5, 7]:  # bus or truck
                            large_vehicle = True
                            large_vehicle_class = vehicle['class_id']
                        
                        break
            
            # Check if this slot has special markings
            is_special = slot.get('is_special', False)
            if not is_special:
                is_special = self.check_for_special_zone(image, x, y, w, h)
            
            # Update slot information with enhanced edge case handling
            slot['occupied'] = occupied
            slot['is_special'] = is_special
            slot['large_vehicle'] = large_vehicle
            slot['vehicle_class'] = occupying_vehicle['class_id'] if occupied else None
            slot['affected_by_moving'] = affected_by_moving
            
            # Draw the slot on the annotated image with enhanced visual coding
            if is_special:
                color = (255, 255, 0)  # Yellow for special zones
                thickness = 2
                if occupied:
                    # Special zone that is occupied
                    thickness = 3
            elif large_vehicle and occupied:
                color = (255, 165, 0)  # Orange for large vehicles
                thickness = 3
            elif affected_by_moving:
                color = (180, 180, 180)  # Gray for slots affected by moving vehicles
                thickness = 1
                # Don't count these as occupied
                slot['occupied'] = False
            elif occupied:
                color = (0, 0, 255)    # Red for occupied
                thickness = 2
            else:
                color = (0, 255, 0)    # Green for available
                thickness = 2
                
            # Draw slot with appropriate color and style
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, thickness)
            
            # Add additional visual indicators for special cases
            if large_vehicle and occupied:
                # Mark large vehicle slots with a cross pattern
                cv2.line(annotated_img, (x, y), (x + w, y + h), color, 1)
                cv2.line(annotated_img, (x + w, y), (x, y + h), color, 1)
        
        # Draw vehicles on the image with enhanced visualizations
        for vehicle in vehicles:
            x, y = vehicle['x'], vehicle['y']
            w, h = vehicle['width'], vehicle['height']
            
            # Different colors for different vehicle types and movement states
            if vehicle.get('is_moving', False):
                # Moving vehicles in a distinct color with dashed outline
                color = (0, 165, 255)  # Orange for moving vehicles
                # Draw dashed rectangle to indicate movement
                # We'll create a dashed effect by drawing multiple small lines
                for i in range(0, w, 10):
                    if i + 10 <= w:
                        # Top edge dashed line
                        cv2.line(annotated_img, (x + i, y), (x + i + 5, y), color, 2)
                        # Bottom edge dashed line
                        cv2.line(annotated_img, (x + i, y + h), (x + i + 5, y + h), color, 2)
                        
                for i in range(0, h, 10):
                    if i + 10 <= h:
                        # Left edge dashed line
                        cv2.line(annotated_img, (x, y + i), (x, y + i + 5), color, 2)
                        # Right edge dashed line
                        cv2.line(annotated_img, (x + w, y + i), (x + w, y + i + 5), color, 2)
                        
                # Add "MOVING" label
                cv2.putText(annotated_img, "MOVING", (x, y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Different colors for different vehicle types
                if vehicle['class_id'] == 2:  # Car
                    color = (255, 0, 0)  # Blue
                elif vehicle['class_id'] == 3:  # Motorcycle
                    color = (255, 0, 255)  # Magenta
                elif vehicle['class_id'] == 5:  # Bus
                    color = (0, 255, 255)  # Yellow
                    # Add "LARGE VEHICLE" label for buses
                    cv2.putText(annotated_img, "LARGE VEHICLE", (x, y - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                elif vehicle['class_id'] == 7:  # Truck
                    color = (255, 165, 0)  # Orange
                    # Add "LARGE VEHICLE" label for trucks
                    cv2.putText(annotated_img, "LARGE VEHICLE", (x, y - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    color = (200, 200, 200)  # Gray for other classes
                
                # Draw solid rectangle for non-moving vehicles
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
            
            # Add label with confidence
            label = f"{self.get_class_name(vehicle['class_id'])}: {vehicle['confidence']:.2f}"
            cv2.putText(annotated_img, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate enhanced statistics with edge case handling
        total_slots = len(slots)
        # Don't count slots affected by moving vehicles as occupied
        occupied_slots = sum(1 for slot in slots if slot['occupied'] and not slot.get('affected_by_moving', False))
        available_slots = total_slots - occupied_slots
        special_slots = sum(1 for slot in slots if slot['is_special'])
        special_occupied = sum(1 for slot in slots if slot['is_special'] and slot['occupied'])
        special_available = special_slots - special_occupied
        large_vehicles = sum(1 for slot in slots if slot.get('large_vehicle', False))
        moving_vehicles_count = len([v for v in vehicles if v.get('is_moving', False)])
        
        # Count misaligned vehicles (detected through modified criteria)
        misaligned_vehicles = 0
        for slot in slots:
            if slot.get('misaligned', False):
                misaligned_vehicles += 1
        
        # Add a more detailed summary overlay to the image
        overlay = annotated_img.copy()
        # Draw a semi-transparent rectangle for the overlay background
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        # Apply the overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, annotated_img, 1 - alpha, 0, annotated_img)
        
        # Enhanced summary text with all edge cases addressed
        summary_text = [
            f"Total Slots: {total_slots}",
            f"Occupied: {occupied_slots}",
            f"Available: {available_slots}",
            f"Special Zones: {special_slots} ({special_occupied} occupied)",
            f"Large Vehicles: {large_vehicles}",
            f"Moving Vehicles: {moving_vehicles_count}",
            f"Misaligned Vehicles: {misaligned_vehicles}"
        ]
        
        for i, text in enumerate(summary_text):
            cv2.putText(annotated_img, text, (20, 40 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Prepare the enhanced result dictionary with additional edge case metrics
        return {
            'slots': slots,
            'vehicles': vehicles,
            'total_slots': total_slots,
            'occupied_slots': occupied_slots,
            'available_slots': available_slots,
            'special_slots': special_slots,
            'special_occupied': special_occupied, 
            'special_available': special_available,
            'large_vehicles': large_vehicles,
            'moving_vehicles': moving_vehicles_count,
            'misaligned_vehicles': misaligned_vehicles,
            'annotated_image': annotated_img
        }
    
    def calculate_overlap_area(self, rect1, rect2):
        """Calculate the overlap area between two rectangles"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Calculate overlap dimensions
        overlap_width = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_height = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        
        # Calculate overlap area
        overlap_area = overlap_width * overlap_height
        
        return overlap_area
    
    def get_class_name(self, class_id):
        """Get the class name for a COCO class ID"""
        class_names = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }
        return class_names.get(class_id, f"Class {class_id}")
