import cv2
import numpy as np
import pandas as pd
import logging
from datetime import datetime

class ParkingAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_parking_data(self, slots_data, image_path=None):
        """
        Analyze the parking data to generate statistics and visualizations
        """
        self.logger.info("Analyzing parking data...")
        
        # Extract the key information
        slots = slots_data['slots']
        vehicles = slots_data['vehicles']
        total_slots = slots_data['total_slots']
        occupied_slots = slots_data['occupied_slots']
        available_slots = slots_data['available_slots']
        special_slots = slots_data['special_slots']
        large_vehicles = slots_data['large_vehicles']
        annotated_image = slots_data['annotated_image']
        
        # Extract additional information for more detailed analysis
        slot_occupancy = []
        special_zones = []
        large_vehicle_slots = []
        
        for slot in slots:
            if slot['occupied']:
                slot_occupancy.append(1)
                
                if slot['large_vehicle']:
                    large_vehicle_slots.append(slot)
            else:
                slot_occupancy.append(0)
                
            if slot['is_special']:
                special_zones.append(slot)
        
        # Occupancy percentage
        occupancy_percentage = (occupied_slots / total_slots * 100) if total_slots > 0 else 0
        
        # Special zones statistics
        special_occupied = sum(1 for slot in special_zones if slot['occupied'])
        special_available = len(special_zones) - special_occupied
        
        # Add visualization enhancement
        self.enhance_visualization(
            annotated_image,
            total_slots,
            occupied_slots,
            available_slots,
            special_slots,
            large_vehicles
        )
        
        # Generate result
        result = {
            'total_slots': total_slots,
            'occupied_slots': occupied_slots,
            'available_slots': available_slots,
            'special_slots': special_slots,
            'large_vehicles': large_vehicles,
            'occupancy_percentage': occupancy_percentage,
            'special_occupied': special_occupied,
            'special_available': special_available,
            'annotated_image': annotated_image,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    def enhance_visualization(self, image, total, occupied, available, special, large_vehicles):
        """
        Enhance the visualization with more detailed information
        """
        height, width = image.shape[:2]
        
        # Create a semi-transparent overlay for the stats panel
        overlay = image.copy()
        panel_height = 180
        panel_width = 300
        
        # Position the panel at the bottom-right
        panel_x = width - panel_width - 20
        panel_y = height - panel_height - 20
        
        # Draw a rounded rectangle for the panel
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        
        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Stats to display
        stats = [
            f"Total Slots: {total}",
            f"Occupied Slots: {occupied}",
            f"Available Slots: {available}",
            f"Special Zones: {special}",
            f"Large Vehicles: {large_vehicles}",
            f"Occupancy Rate: {(occupied / total * 100):.1f}%" if total > 0 else "Occupancy Rate: N/A"
        ]
        
        # Add title
        cv2.putText(image, "Parking Analysis", 
                   (panel_x + 10, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw horizontal line
        cv2.line(image, 
                (panel_x + 10, panel_y + 40), 
                (panel_x + panel_width - 10, panel_y + 40), 
                (255, 255, 255), 2)
        
        # Add stats with color coding
        for i, stat in enumerate(stats):
            color = (255, 255, 255)  # Default white
            
            # Change color based on the stat
            if "Occupied" in stat:
                color = (0, 0, 255)  # Red for occupied
            elif "Available" in stat:
                color = (0, 255, 0)  # Green for available
            elif "Special" in stat:
                color = (255, 255, 0)  # Yellow for special zones
            elif "Large" in stat:
                color = (255, 165, 0)  # Orange for large vehicles
            
            cv2.putText(image, stat, 
                       (panel_x + 10, panel_y + 70 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add timestamp
        cv2.putText(image, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                   (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
