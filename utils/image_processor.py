"""
Handle event flyer image uploads and processing
"""

import os
from PIL import Image
import io
from datetime import datetime
import streamlit as st

class ImageProcessor:
    """Handle image uploads and processing"""
    
    def __init__(self, upload_dir="static/uploads"):
        self.upload_dir = upload_dir
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def save_uploaded_file(self, uploaded_file, event_id=None):
        """Save uploaded file and return path"""
        if uploaded_file is None:
            return None
        
        # Validate file extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext not in self.allowed_extensions:
            st.error(f"Unsupported file type: {file_ext}. Allowed: {', '.join(self.allowed_extensions)}")
            return None
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if event_id:
            filename = f"event_{event_id}_{timestamp}{file_ext}"
        else:
            filename = f"temp_{timestamp}{file_ext}"
        
        file_path = os.path.join(self.upload_dir, filename)
        
        # Save file
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Optimize image if it's too large
            self._optimize_image(file_path)
            
            return file_path
            
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return None
    
    def _optimize_image(self, image_path, max_size=(1200, 1200), quality=85):
        """Optimize image size and quality"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = rgb_img
                
                # Resize if too large
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save optimized image
                img.save(image_path, 'JPEG' if image_path.lower().endswith('.jpg') else 'PNG', 
                        optimize=True, quality=quality)
                
        except Exception as e:
            st.warning(f"Image optimization failed: {e}")
    
    def display_image(self, image_path, width=400):
        """Display image in Streamlit"""
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                st.image(image, width=width, caption="Event Flyer")
            except Exception as e:
                st.error(f"Error displaying image: {e}")
    
    def delete_image(self, image_path):
        """Delete image file"""
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                return True
            except Exception as e:
                st.error(f"Error deleting image: {e}")
                return False
        return True
    
    def get_image_url(self, image_path):
        """Get URL for image (for web display)"""
        if not image_path:
            return None
        
        # For Streamlit Cloud, we need to serve static files differently
        # This is a simplified version - in production, you'd use a CDN or proper static file serving
        if image_path.startswith(self.upload_dir):
            # Return relative path
            return f"/{image_path}"
        
        return image_path
