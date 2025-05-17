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
    
    def enhance_visualization(self, image, total, occupied, available, special, large_vehicles, 
                              special_occupied=0, moving_vehicles=0, misaligned_vehicles=0):
        """
        Enhance the visualization with more detailed information about all edge cases
        """
        height, width = image.shape[:2]
        
        # Create a copy of image to avoid modifying the original
        enhanced_image = image.copy()
        
        # Create a semi-transparent overlay for the stats panel
        overlay = enhanced_image.copy()
        panel_height = 220  # Increased height for more stats
        panel_width = 320   # Slightly wider for better readability
        
        # Position the panel at the bottom-right
        panel_x = width - panel_width - 20
        panel_y = height - panel_height - 20
        
        # Draw a filled rectangle for the panel background
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        
        # Apply transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, enhanced_image, 1 - alpha, 0, enhanced_image)
        
        # Enhanced stats to display with all edge cases
        stats = [
            f"Total Slots: {total}",
            f"Occupied Slots: {occupied}",
            f"Available Slots: {available}",
            f"Special Zones: {special} (Total)",
            f"Special Occupied: {special_occupied}",
            f"Large Vehicles: {large_vehicles}",
            f"Moving Vehicles: {moving_vehicles}",
            f"Misaligned Parking: {misaligned_vehicles}",
            f"Occupancy Rate: {(occupied / total * 100):.1f}%" if total > 0 else "Occupancy Rate: N/A"
        ]
        
        # Add title with improved styling
        cv2.putText(enhanced_image, "Parking Analysis", 
                   (panel_x + 10, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw horizontal line
        cv2.line(enhanced_image, 
                (panel_x + 10, panel_y + 40), 
                (panel_x + panel_width - 10, panel_y + 40), 
                (255, 255, 255), 2)
        
        # Add stats with enhanced color coding
        for i, stat in enumerate(stats):
            color = (255, 255, 255)  # Default white
            
            # Enhanced color coding for better visual differentiation
            if "Occupied Slots" in stat:
                color = (0, 0, 255)  # Red for occupied
            elif "Available Slots" in stat:
                color = (0, 255, 0)  # Green for available
            elif "Special Zones" in stat:
                color = (255, 255, 0)  # Yellow for special zones
            elif "Special Occupied" in stat:
                color = (200, 200, 0)  # Darker yellow for occupied special zones
            elif "Large Vehicles" in stat:
                color = (255, 165, 0)  # Orange for large vehicles
            elif "Moving Vehicles" in stat:
                color = (0, 165, 255)  # Light orange for moving vehicles
            elif "Misaligned" in stat:
                color = (180, 0, 180)  # Purple for misaligned vehicles
            elif "Occupancy Rate" in stat:
                # Color based on occupancy percentage
                if total > 0:
                    rate = occupied / total * 100
                    if rate < 50:
                        color = (0, 255, 0)  # Green for low occupancy
                    elif rate < 80:
                        color = (0, 255, 255)  # Yellow for moderate occupancy
                    else:
                        color = (0, 0, 255)  # Red for high occupancy
            
            cv2.putText(enhanced_image, stat, 
                       (panel_x + 10, panel_y + 70 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add legend to explain symbols and colors
        legend_x = 20
        legend_y = 30
        
        # Draw legend background
        legend_height = 180
        legend_width = 220
        cv2.rectangle(overlay, (legend_x - 10, legend_y - 25), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, enhanced_image, 1 - alpha, 0, enhanced_image)
        
        # Add legend title
        cv2.putText(enhanced_image, "Legend", 
                   (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add legend items
        legend_items = [
            ("Green Box", "Available Slot", (0, 255, 0)),
            ("Red Box", "Occupied Slot", (0, 0, 255)),
            ("Yellow Box", "Special Zone", (255, 255, 0)),
            ("Orange Box", "Large Vehicle", (255, 165, 0)),
            ("Dashed Box", "Moving Vehicle", (0, 165, 255)),
            ("Purple Box", "Misaligned Parking", (180, 0, 180)),
            ("X Pattern", "Multiple Slot Usage", (255, 255, 255))
        ]
        
        for i, (symbol, desc, color) in enumerate(legend_items):
            # Draw colored symbol
            cv2.rectangle(enhanced_image, 
                        (legend_x, legend_y + 20 + i * 20 - 10), 
                        (legend_x + 15, legend_y + 20 + i * 20 + 5), 
                        color, -1)
            
            # Draw label
            cv2.putText(enhanced_image, desc, 
                       (legend_x + 25, legend_y + 20 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp
        cv2.putText(enhanced_image, f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                   (width - 350, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return enhanced_image
