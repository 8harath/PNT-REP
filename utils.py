import os
import shutil
import tempfile
import logging

logger = logging.getLogger(__name__)

def create_temp_directory():
    """Create a temporary directory for file uploads"""
    try:
        # Use static/uploads if it exists, otherwise create a temp directory
        if os.path.exists('static/uploads'):
            os.makedirs('static/uploads', exist_ok=True)
            return 'static/uploads'
        else:
            temp_dir = tempfile.mkdtemp()
            return temp_dir
    except Exception as e:
        logger.error(f"Error creating temporary directory: {str(e)}")
        # Fallback to the current directory if we can't create a temp directory
        os.makedirs('uploads', exist_ok=True)
        return 'uploads'

def cleanup_temp_files(directory):
    """Clean up temporary files when they're no longer needed"""
    try:
        # Don't delete the static/uploads directory, just its contents
        if directory == 'static/uploads':
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            # Otherwise, delete the entire temp directory
            shutil.rmtree(directory, ignore_errors=True)
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {str(e)}")