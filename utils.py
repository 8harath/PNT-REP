import os
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)

def create_temp_directory():
    """Create a temporary directory for file uploads"""
    try:
        temp_dir = tempfile.mkdtemp(prefix='parking_detection_')
        logger.info(f"Created temporary directory: {temp_dir}")
        return temp_dir
    except Exception as e:
        logger.error(f"Error creating temporary directory: {str(e)}")
        # Fallback to using a local directory
        temp_dir = os.path.join(os.getcwd(), 'temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

def cleanup_temp_files(directory):
    """Clean up temporary files when they're no longer needed"""
    try:
        if os.path.exists(directory) and os.path.isdir(directory):
            shutil.rmtree(directory)
            logger.info(f"Cleaned up temporary directory: {directory}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory: {str(e)}")

def get_vehicle_class_name(class_id):
    """Get the class name for a COCO class ID"""
    class_names = {
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }
    return class_names.get(class_id, f"Class {class_id}")

def format_timestamp(timestamp=None):
    """Format a timestamp for display"""
    from datetime import datetime
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')

def create_csv_report(data, output_path):
    """Create a CSV report from parking data"""
    import pandas as pd
    
    # Create DataFrame
    df = pd.DataFrame({
        'Total Slots': [data['total_slots']],
        'Occupied Slots': [data['occupied_slots']],
        'Available Slots': [data['available_slots']]
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"CSV report saved to: {output_path}")
    
    return output_path
