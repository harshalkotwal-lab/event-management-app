"""
G H Raisoni College - Advanced Event Management System
Complete solution with AI, Image Uploads, Social Features
Deployable on Streamlit Cloud - Streamlit Native Version
"""

import streamlit as st
st.set_page_config(
    page_title="G H Raisoni Event Manager",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from datetime import datetime, date, timedelta
import json
import os
import hashlib
import uuid
import pandas as pd
from PIL import Image
import io
import re
import sqlite3
import base64
import traceback
import logging
from functools import lru_cache
import time
import atexit
import openai
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / "data"
USERS_FILE = DATA_DIR / "users.json"
EVENTS_FILE = DATA_DIR / "events.json"
REGISTRATIONS_FILE = DATA_DIR / "registrations.json"

# Default credentials (CHANGE THESE IN PRODUCTION)
DEFAULT_CREDENTIALS = {
    "admin": {
        "username": "admin@raisoni",
        "password": "admin123",
        "name": "Administrator",
        "role": "admin",
        "department": "Administration",
        "email": "admin@ghraisoni.edu"
    },
    "faculty": {
        "username": "faculty@raisoni",
        "password": "faculty123",
        "name": "Faculty Coordinator",
        "role": "faculty",
        "department": "Faculty",
        "email": "faculty@ghraisoni.edu"
    }
}

# College configuration
COLLEGE_CONFIG = {
    "name": "G H Raisoni College of Engineering and Management",
    "departments": [
        "Computer Science & Engineering",
        "Artificial Intelligence & Machine Learning",
        "Electronics & Communication",
        "Electrical & Electronics",
        "Mechanical Engineering",
        "Civil Engineering",
        "Information Technology",
        "Data Science"
    ],
    "academic_years": ["I", "II", "III", "IV"],
    "event_types": [
        "Workshop",
        "Hackathon",
        "Competition",
        "Bootcamp",
        "Seminar",
        "Conference",
        "Webinar",
        "Training",
        "Symposium",
        "Cultural Event"
    ]
}

# Feature flags
FEATURES = {
    "student_registration": True,
    "email_verification": False,
    "qr_checkin": False,
    "certificate_generation": False,
    "whatsapp_integration": False
}

# Email configuration (if enabled)
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "events@ghraisoni.edu",
    "use_tls": True
}

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



class Validators:
    """Collection of input validation methods"""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email format and college domain"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, email):
            return False, "Invalid email format"
        
        # Check if it's a college email (optional)
        college_domains = ['ghraisoni.edu', 'raisoni.net']
        domain = email.split('@')[-1]
        if not any(domain.endswith(college_domain) for college_domain in college_domains):
            return True, "Warning: Non-college email detected"
        
        return True, "Valid email"
    
    @staticmethod
    def validate_roll_number(roll_no: str) -> Tuple[bool, str]:
        """Validate roll number format"""
        # Example: CSE2023001, AIML2023056
        pattern = r'^[A-Z]{2,4}\d{7}$'
        
        if not re.match(pattern, roll_no):
            return False, "Invalid roll number format. Example: CSE2023001"
        
        return True, "Valid roll number"
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not any(char.isupper() for char in password):
            return False, "Password must contain at least one uppercase letter"
        
        if not any(char.isdigit() for char in password):
            return False, "Password must contain at least one number"
        
        return True, "Strong password"
    
    @staticmethod
    def validate_event_date(event_date: str) -> Tuple[bool, str]:
        """Validate event date is in future"""
        try:
            date_obj = datetime.fromisoformat(event_date)
            if date_obj < datetime.now():
                return False, "Event date must be in the future"
            return True, "Valid date"
        except ValueError:
            return False, "Invalid date format"
    
    @staticmethod
    def validate_url(url: str) -> Tuple[bool, str]:
        """Validate URL format"""
        pattern = r'^https?://(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/\S*)?$'
        
        if not url:  # URL is optional
            return True, "URL is optional"
        
        if not re.match(pattern, url):
            return False, "Invalid URL format"
        
        return True, "Valid URL"
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent XSS"""
        if not text:
            return ""
        
        # Remove potentially dangerous characters
        sanitized = text.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        
        return sanitized.strip()


class AuthManager:
    """Handles user authentication and session management"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.session_timeout = timedelta(hours=24)
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_default_credentials(self, username: str, password: str, role: str) -> bool:
        """Verify default admin/faculty credentials"""
        if role == 'admin':
            default = DEFAULT_CREDENTIALS['admin']
        elif role == 'faculty':
            default = DEFAULT_CREDENTIALS['faculty']
        else:
            return False
        
        return (username == default['username'] and 
                self.hash_password(password) == self.hash_password(default['password']))
    
    def verify_student_credentials(self, username: str, password: str) -> bool:
        """Verify student credentials from database"""
        user = self.db.get_user_by_username(username)
        if not user:
            return False
        
        hashed_input = self.hash_password(password)
        return user.get('password') == hashed_input and user.get('is_active', True)
    
    def create_session(self, username: str, role: str, user_data: Optional[Dict] = None):
        """Create user session"""
        st.session_state['authenticated'] = True
        st.session_state['username'] = username
        st.session_state['role'] = role
        st.session_state['login_time'] = datetime.now()
        
        if user_data:
            st.session_state['user_data'] = user_data
    
    def logout(self):
        """Clear user session"""
        for key in ['authenticated', 'username', 'role', 'login_time', 'user_data']:
            if key in st.session_state:
                del st.session_state[key]
    
    def is_session_valid(self) -> bool:
        """Check if session is still valid"""
        if not st.session_state.get('authenticated', False):
            return False
        
        login_time = st.session_state.get('login_time')
        if not login_time:
            return False
        
        if isinstance(login_time, str):
            login_time = datetime.fromisoformat(login_time)
        
        return datetime.now() - login_time < self.session_timeout
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current user data"""
        if not self.is_session_valid():
            return None
        
        username = st.session_state.get('username')
        if not username:
            return None
        
        # For admin/faculty, return default data
        role = st.session_state.get('role')
        if role == 'admin':
            return DEFAULT_CREDENTIALS['admin']
        elif role == 'faculty':
            return DEFAULT_CREDENTIALS['faculty']
        else:
            # For students, get from database
            return self.db.get_user_by_username(username)
    
    def require_auth(self, roles: list = None):
        """Decorator to require authentication for specific roles"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.is_session_valid():
                    st.error("Session expired. Please login again.")
                    st.stop()
                
                user_role = st.session_state.get('role')
                if roles and user_role not in roles:
                    st.error(f"Access denied. Required roles: {', '.join(roles)}")
                    st.stop()
                
                return func(*args, **kwargs)
            return wrapper
        return decorator


# ============================================
# AI EVENT GENERATOR CLASS
# ============================================
class AIEventGenerator:
    """Generate structured event data from unstructured text - Compatible with OpenAI v0.28"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.api_key = None
        self.is_configured = False
        
        try:
            # Get API key from secrets
            if hasattr(st, 'secrets'):
                # Try different possible secret names
                if 'OPENAI_API_KEY' in st.secrets:
                    self.api_key = st.secrets["OPENAI_API_KEY"]
                elif 'openai' in st.secrets and 'api_key' in st.secrets.openai:
                    self.api_key = st.secrets.openai.api_key
                
                if self.api_key and self.api_key.startswith("sk-"):
                    import openai
                    # For OpenAI v0.28.x, set the API key directly
                    openai.api_key = self.api_key
                    self.is_configured = True
                    st.success("✅ OpenAI configured successfully")
                elif self.api_key:
                    st.warning(f"⚠️ Invalid OpenAI API key format")
                    self.is_configured = False
                else:
                    st.warning("⚠️ OpenAI API key not found in secrets")
                    self.is_configured = False
            else:
                st.warning("⚠️ Streamlit secrets not available")
                self.is_configured = False
                
        except Exception as e:
            st.warning(f"⚠️ Could not initialize OpenAI: {str(e)[:100]}")
            self.is_configured = False
    
    def extract_event_info(self, text):
        """Extract event information from text using AI or regex fallback"""
        
        # Try OpenAI first if available
        if self.is_configured and self.api_key:
            try:
                with st.spinner("🤖 AI is processing your event..."):
                    return self._extract_with_openai(text)
            except Exception as e:
                st.warning(f"AI extraction failed: {str(e)[:100]}. Using regex fallback.")
        
        # Fallback to regex extraction
        return self._extract_with_regex(text)
    
    def _extract_with_openai(self, text):
        """Use OpenAI to extract structured event data"""
        prompt = f"""
        Extract event information from the following text and return as JSON with these fields:
        - title: Event title (string)
        - description: Detailed event description (string)
        - event_type: Type of event (workshop, hackathon, competition, bootcamp, seminar, conference, webinar)
        - event_date: Event date in YYYY-MM-DD format (extract from text or use reasonable default)
        - venue: Event venue/location (string)
        - organizer: Event organizer (string)
        - event_link: Event website/URL if mentioned (string or null)  # NEW
        - registration_link: Registration URL if mentioned (string or null)
        - max_participants: Maximum participants if mentioned (integer or 100)
        
        Text: {text}
        
        Return only valid JSON, no other text.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured event information from text. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=600
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean response (remove markdown code blocks if present)
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'\s*```', '', result_text)
            result_text = re.sub(r'^json\s*', '', result_text, flags=re.IGNORECASE)
            
            # Parse JSON
            event_data = json.loads(result_text)
            
            # Validate required fields
            required_fields = ['title', 'description', 'event_type', 'event_date', 'venue', 'organizer']
            for field in required_fields:
                if field not in event_data:
                    event_data[field] = ""
            
            # Add AI metadata
            event_data['ai_generated'] = True
            event_data['ai_prompt'] = text
            event_data['ai_extracted_at'] = datetime.now().isoformat()
            
            return event_data
            
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse AI response as JSON: {str(e)[:100]}")
            # Show the raw response for debugging
            st.code(f"Raw AI response: {result_text[:500]}...")
            return self._extract_with_regex(text)
        except Exception as e:
            st.error(f"OpenAI API error: {str(e)[:100]}")
            return self._extract_with_regex(text)
    
    def _extract_with_regex(self, text):
        """Fallback regex-based extraction"""
        event_data = {
            'title': 'New Event',
            'description': text[:200] + '...' if len(text) > 200 else text,
            'event_type': 'workshop',
            'event_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'venue': 'G H Raisoni College',
            'organizer': 'College Department',
            'event_link': None,  # NEW
            'registration_link': None,
            'max_participants': 100,
            'ai_generated': False,
            'ai_prompt': text
        }
        
        # Try to extract title (first line or sentence)
        lines = text.strip().split('\n')
        if lines and lines[0].strip():
            first_line = lines[0].strip()
            if len(first_line) < 100:  # Reasonable title length
                event_data['title'] = first_line
        
        # Try to extract date patterns
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # DD-MM-YYYY
            r'(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})',  # YYYY-MM-DD
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['event_date'] = match.group(1)
                break
        
        # Try to extract venue
        venue_keywords = ['at', 'venue', 'location', 'place', 'hall', 'room']
        for keyword in venue_keywords:
            pattern = rf'{keyword}[:\s]*([^.\n,;]{5,50})'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['venue'] = match.group(1).strip()
                break
        
        # Try to extract organizer
        organizer_keywords = ['by', 'organizer', 'organized by', 'conducted by', 'hosted by']
        for keyword in organizer_keywords:
            pattern = rf'{keyword}[:\s]*([^.\n,;]{5,50})'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['organizer'] = match.group(1).strip()
                break
        
        # Try to extract URLs (registration links)
        url_pattern = r'https?://[^\s<>"\'()]+'
        urls = re.findall(url_pattern, text)
        
        if urls:
            # Use first URL as event link
            event_data['event_link'] = urls[0]
            # If there's a second URL, use it as registration link
            if len(urls) > 1:
                event_data['registration_link'] = urls[1]
        
        return event_data
    
    def enhance_event_description(self, basic_info):
        """Use AI to enhance event description"""
        if not self.client:
            return basic_info
        
        prompt = f"""
        Enhance this event description to make it more engaging and professional:
        
        Title: {basic_info.get('title', 'Event')}
        Basic Description: {basic_info.get('description', '')}
        
        Please provide:
        1. A catchy, engaging title (max 10 words)
        2. A professional description (3-4 paragraphs)
        3. Key highlights/bullet points
        4. Who should attend
        5. What participants will learn/gain
        
        Return as JSON with fields: enhanced_title, enhanced_description, key_highlights, target_audience, learning_outcomes
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at creating engaging event descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'\s*```', '', result_text)
            
            enhanced_data = json.loads(result_text)
            
            # Merge enhanced data with basic info
            if 'enhanced_title' in enhanced_data:
                basic_info['title'] = enhanced_data['enhanced_title']
            if 'enhanced_description' in enhanced_data:
                basic_info['description'] = enhanced_data['enhanced_description']
            
            basic_info['ai_enhanced'] = True
            if 'ai_metadata' not in basic_info:
                basic_info['ai_metadata'] = {}
            basic_info['ai_metadata']['enhancement'] = enhanced_data
            
            return basic_info
            
        except Exception as e:
            st.warning(f"AI enhancement failed: {e}")
            return basic_info

# ============================================
# CONFIGURATION
# ============================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# ENHANCED CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        text-align: center;
        padding: 0.75rem;
        margin-bottom: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .college-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .event-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
        border-left: 3px solid #3B82F6;
    }
    
    .event-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        border-color: #2563EB;
    }

    /* Ensure image column has enough width for 200px image */
    .stColumn:first-child {
        min-width: 220px !important; /* 200px image + padding */
    }

    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 6px;
        line-height: 1.3;
    }

    .registration-section {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 8px;
        border-radius: 6px;
        margin-top: 8px;
        border-left: 3px solid #3B82F6;
        font-size: 0.9rem;
    }

    
    .role-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-left: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .admin-badge { background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); color: #DC2626; }
    .faculty-badge { background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); color: #1D4ED8; }
    .student-badge { background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); color: #065F46; }
    
    .ai-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .social-container {
        display: flex;
        gap: 0.5rem;
        margin: 1rem 0;
        padding: 0.75rem;
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.3rem 0;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .registration-section {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #3B82F6;
        font-size: 0.95rem;
    }
    
    .status-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    
    .status-upcoming { background: #D1FAE5; color: #065F46; }
    .status-ongoing { background: #FEF3C7; color: #92400E; }
    .status-past { background: #FEE2E2; color: #DC2626; }
    
    
    .card-description {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0.75rem 0;
    }
    
    .filter-row {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    
    .nav-button {
        width: 100%;
        margin: 0.25rem 0;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background: white;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .nav-button:hover {
        background: #f1f5f9;
        transform: translateX(4px);
    }
    
    .nav-button.active {
        background: #3B82F6;
        color: white;
        border-color: #3B82F6;
        font-weight: 600;
    }

    /* Engagement buttons */
    .engagement-button {
        transition: all 0.2s ease;
    }
    
    .engagement-button:hover {
        transform: scale(1.05);
    }
    
    /* Liked state */
    .liked-button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
        color: white !important;
    }
    
    /* Interested state */
    .interested-button {
        background: linear-gradient(135deg, #FFD93D 0%, #FF9F1C 100%) !important;
        color: white !important;
    }
    
    /* Engagement metrics */
    .engagement-metric {
        font-size: 0.9rem;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Event links styling */
    .event-links-container {
        background: #f8fafc;
        padding: 12px;
        border-radius: 8px;
        margin: 12px 0;
        border: 1px solid #e2e8f0;
    }
    
    .event-link-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px;
        margin: 4px 0;
        background: white;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
    }
    
    .event-link-item:hover {
        background: #f1f5f9;
        border-color: #3B82F6;
    }
    
    .event-link-icon {
        font-size: 1.2rem;
        min-width: 24px;
    }
    
    .event-link-text {
        flex: 1;
        word-break: break-all;
    }
    
    .event-link-text a {
        color: #1E40AF;
        text-decoration: none;
        font-weight: 500;
    }
    
    .event-link-text a:hover {
        text-decoration: underline;
        color: #1E3A8A;
    }
    
    .event-link-badge {
        background: #3B82F6;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    
    .registration-badge {
        background: #10B981;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================
def display_role_badge(role):
    """Display role badge"""
    badges = {
        "admin": ("👑 Admin", "admin-badge"),
        "faculty": ("👨‍🏫 Faculty", "faculty-badge"),
        "student": ("👨‍🎓 Student", "student-badge")
    }
    
    if role in badges:
        text, css_class = badges[role]
        st.markdown(f'<span class="role-badge {css_class}">{text}</span>', 
                   unsafe_allow_html=True)

def format_date(date_str):
    """Format date for display"""
    try:
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            dt = date_str
        return dt.strftime("%d %b %Y, %I:%M %p")
    except:
        return str(date_str)

def get_event_status(event_date):
    """Get event status badge"""
    try:
        if isinstance(event_date, str):
            dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        else:
            dt = event_date
        
        if dt > datetime.now():
            return '<span class="status-badge status-upcoming">🟢 Upcoming</span>'
        elif dt.date() == datetime.now().date():
            return '<span class="status-badge status-ongoing">🟡 Ongoing</span>'
        else:
            return '<span class="status-badge status-past">🔴 Past</span>'
    except:
        return '<span class="status-badge">Unknown</span>'

# ============================================
# DATABASE MANAGER
# ============================================
class DatabaseManager:
    """Simple database manager"""
    
    def __init__(self, db_path="event_management.db"):
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()
        self._add_default_users()
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
    
    def create_tables(self):
        """Create all necessary tables including likes and interested"""
        try:
            cursor = self.conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL,
                    roll_no TEXT,
                    department TEXT,
                    year TEXT,
                    email TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Users table (existing)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    event_type TEXT,
                    event_date TIMESTAMP,
                    venue TEXT,
                    organizer TEXT,
                    event_link TEXT,  -- NEW: Event URL/website
                    registration_link TEXT,  -- Registration URL
                    max_participants INTEGER DEFAULT 100,
                    current_participants INTEGER DEFAULT 0,
                    flyer_path TEXT,
                    created_by TEXT,
                    created_by_name TEXT,
                    ai_generated BOOLEAN DEFAULT 0,
                    status TEXT DEFAULT 'upcoming',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Events table (existing)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    event_type TEXT,
                    event_date TIMESTAMP,
                    venue TEXT,
                    organizer TEXT,
                    registration_link TEXT,
                    max_participants INTEGER DEFAULT 100,
                    current_participants INTEGER DEFAULT 0,
                    flyer_path TEXT,
                    created_by TEXT,
                    created_by_name TEXT,
                    ai_generated BOOLEAN DEFAULT 0,
                    status TEXT DEFAULT 'upcoming',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Registrations table (existing)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS registrations (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    event_title TEXT NOT NULL,
                    student_username TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    student_roll TEXT,
                    student_dept TEXT,
                    status TEXT DEFAULT 'pending',
                    attendance TEXT DEFAULT 'absent',
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(event_id, student_username)
                )
            ''')
            
            # Likes table (NEW)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS event_likes (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    student_username TEXT NOT NULL,
                    liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(event_id, student_username)
                )
            ''')
            
            # Interested table (NEW)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS event_interested (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    student_username TEXT NOT NULL,
                    interested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(event_id, student_username)
                )
            ''')
            self.conn.commit()
            logger.info("All tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    def _add_default_users(self):
        """Add default admin, faculty, and student users"""
        try:
            cursor = self.conn.cursor()
            
            # Check if admin exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ('admin@raisoni',))
            if cursor.fetchone()[0] == 0:
                admin_id = str(uuid.uuid4())
                hashed_pass = hashlib.sha256('admin123'.encode()).hexdigest()
                cursor.execute('''
                    INSERT INTO users (id, name, username, password, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (admin_id, 'Administrator', 'admin@raisoni', hashed_pass, 'admin', datetime.now().isoformat()))
            
            # Check if faculty exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ('faculty@raisoni',))
            if cursor.fetchone()[0] == 0:
                faculty_id = str(uuid.uuid4())
                hashed_pass = hashlib.sha256('faculty123'.encode()).hexdigest()
                cursor.execute('''
                    INSERT INTO users (id, name, username, password, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (faculty_id, 'Faculty Coordinator', 'faculty@raisoni', hashed_pass, 'faculty', datetime.now().isoformat()))
            
            # Add default student accounts
            default_students = [
                {
                    'name': 'Rohan Sharma',
                    'username': 'rohan@student',
                    'password': 'student123',
                    'roll_no': 'CSE2023001',
                    'department': 'Computer Science & Engineering',
                    'year': 'III',
                    'email': 'rohan.sharma@ghraisoni.edu'
                },
                {
                    'name': 'Priya Patel',
                    'username': 'priya@student',
                    'password': 'student123',
                    'roll_no': 'AIML2023002',
                    'department': 'Artificial Intelligence & Machine Learning',
                    'year': 'II',
                    'email': 'priya.patel@ghraisoni.edu'
                },
                {
                    'name': 'Amit Kumar',
                    'username': 'amit@student',
                    'password': 'student123',
                    'roll_no': 'IT2023003',
                    'department': 'Information Technology',
                    'year': 'IV',
                    'email': 'amit.kumar@ghraisoni.edu'
                },
                {
                    'name': 'Neha Singh',
                    'username': 'neha@student',
                    'password': 'student123',
                    'roll_no': 'DS2023004',
                    'department': 'Data Science',
                    'year': 'I',
                    'email': 'neha.singh@ghraisoni.edu'
                },
                {
                    'name': 'Vikram Verma',
                    'username': 'vikram@student',
                    'password': 'student123',
                    'roll_no': 'ECE2023005',
                    'department': 'Electronics & Communication',
                    'year': 'III',
                    'email': 'vikram.verma@ghraisoni.edu'
                }
            ]
            
            for student in default_students:
                cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (student['username'],))
                if cursor.fetchone()[0] == 0:
                    student_id = str(uuid.uuid4())
                    hashed_pass = hashlib.sha256(student['password'].encode()).hexdigest()
                    cursor.execute('''
                        INSERT INTO users (id, name, roll_no, department, year, email, username, password, role, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        student_id,
                        student['name'],
                        student['roll_no'],
                        student['department'],
                        student['year'],
                        student['email'],
                        student['username'],
                        hashed_pass,
                        'student',
                        datetime.now().isoformat()
                    ))
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error adding default users: {e}")
    
    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials"""
        try:
            if role in ['admin', 'faculty']:
                default_creds = {
                    'admin': {'username': 'admin@raisoni', 'password': 'admin123'},
                    'faculty': {'username': 'faculty@raisoni', 'password': 'faculty123'}
                }
                creds = default_creds[role]
                if username == creds['username']:
                    return hashlib.sha256(password.encode()).hexdigest() == hashlib.sha256(creds['password'].encode()).hexdigest()
                return False
            else:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT password FROM users WHERE username = ? AND role = 'student'",
                    (username,)
                )
                result = cursor.fetchone()
                if result:
                    stored_hash = result[0]
                    return hashlib.sha256(password.encode()).hexdigest() == stored_hash
                return False
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    def get_user(self, username):
        """Get user by username"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        return dict(result) if result else None
    
    def add_user(self, user_data):
        """Add new user"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO users (id, name, roll_no, department, year, email, username, password, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_data.get('id', str(uuid.uuid4())),
                user_data.get('name'),
                user_data.get('roll_no'),
                user_data.get('department'),
                user_data.get('year'),
                user_data.get('email'),
                user_data.get('username'),
                self._hash_password(user_data.get('password')),
                user_data.get('role', 'student'),
                datetime.now().isoformat()
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return False
    
    def add_event(self, event_data):
        """Add new event"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO events (
                    id, title, description, event_type, event_date, venue, organizer,
                    registration_link, max_participants, current_participants, flyer_path, created_by,
                    created_by_name, ai_generated, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_data.get('id', str(uuid.uuid4())),
                event_data.get('title'),
                event_data.get('description'),
                event_data.get('event_type'),
                event_data.get('event_date'),
                event_data.get('venue'),
                event_data.get('organizer'),
                event_data.get('registration_link', ''),
                event_data.get('max_participants', 100),
                event_data.get('current_participants', 0),
                event_data.get('flyer_path'),
                event_data.get('created_by'),
                event_data.get('created_by_name'),
                event_data.get('ai_generated', False),
                'upcoming',
                datetime.now().isoformat()
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding event: {e}")
            return False
    
    def get_all_events(self):
        """Get all events"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM events ORDER BY event_date DESC")
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def get_events_by_creator(self, username):
        """Get events created by specific user"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM events WHERE created_by = ? ORDER BY event_date DESC", (username,))
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def add_registration(self, reg_data):
        """Add new registration"""
        try:
            cursor = self.conn.cursor()
            
            # Check if already registered
            cursor.execute("SELECT id FROM registrations WHERE event_id = ? AND student_username = ?", 
                          (reg_data['event_id'], reg_data['student_username']))
            if cursor.fetchone():
                return None
            
            cursor.execute('''
                INSERT INTO registrations (
                    id, event_id, event_title, student_username, student_name,
                    student_roll, student_dept, status, attendance, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                reg_data.get('id', str(uuid.uuid4())),
                reg_data.get('event_id'),
                reg_data.get('event_title'),
                reg_data.get('student_username'),
                reg_data.get('student_name'),
                reg_data.get('student_roll'),
                reg_data.get('student_dept'),
                reg_data.get('status', 'pending'),
                reg_data.get('attendance', 'absent'),
                datetime.now().isoformat()
            ))
            
            # Update event participant count
            cursor.execute("UPDATE events SET current_participants = current_participants + 1 WHERE id = ?", 
                          (reg_data['event_id'],))
            
            self.conn.commit()
            return reg_data['id']
        except Exception as e:
            logger.error(f"Error adding registration: {e}")
            return None
    
    def get_registrations_by_student(self, username):
        """Get all registrations for a student"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT r.*, e.event_date, e.venue, e.status as event_status 
            FROM registrations r
            LEFT JOIN events e ON r.event_id = e.id
            WHERE r.student_username = ?
            ORDER BY r.registered_at DESC
        ''', (username,))
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def get_registrations_by_event(self, event_id):
        """Get all registrations for an event"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT r.*, u.department, u.year 
            FROM registrations r
            LEFT JOIN users u ON r.student_username = u.username
            WHERE r.event_id = ?
            ORDER BY r.registered_at DESC
        ''', (event_id,))
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def is_student_registered(self, event_id, username):
        """Check if student is registered for event"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM registrations WHERE event_id = ? AND student_username = ?", 
                      (event_id, username))
        return cursor.fetchone() is not None
    
    def update_event_status(self):
        """Update event status based on current time"""
        try:
            now = datetime.now().isoformat()
            cursor = self.conn.cursor()
            cursor.execute("UPDATE events SET status = 'past' WHERE event_date <= ? AND status != 'past'", (now,))
            cursor.execute("UPDATE events SET status = 'ongoing' WHERE event_date > ? AND status != 'ongoing'", (now,))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating event status: {e}")
            return False

    def add_like(self, event_id, student_username):
        """Add a like for an event"""
        try:
            cursor = self.conn.cursor()
            like_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO event_likes (id, event_id, student_username, liked_at)
                VALUES (?, ?, ?, ?)
            ''', (like_id, event_id, student_username, datetime.now().isoformat()))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Already liked
            return False
        except Exception as e:
            logger.error(f"Error adding like: {e}")
            return False
    
    def remove_like(self, event_id, student_username):
        """Remove a like for an event"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                DELETE FROM event_likes 
                WHERE event_id = ? AND student_username = ?
            ''', (event_id, student_username))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing like: {e}")
            return False
    
    def is_event_liked(self, event_id, student_username):
        """Check if student liked an event"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id FROM event_likes 
                WHERE event_id = ? AND student_username = ?
            ''', (event_id, student_username))
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking like: {e}")
            return False
    
    def get_event_likes_count(self, event_id):
        """Get total likes for an event"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM event_likes 
                WHERE event_id = ?
            ''', (event_id,))
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting likes count: {e}")
            return 0
    
    def get_student_liked_events(self, student_username):
        """Get all events liked by a student"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT e.* FROM events e
                JOIN event_likes l ON e.id = l.event_id
                WHERE l.student_username = ?
                ORDER BY l.liked_at DESC
            ''', (student_username,))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting student liked events: {e}")
            return []
    
    # INTERESTED METHODS
    def add_interested(self, event_id, student_username):
        """Add interested for an event"""
        try:
            cursor = self.conn.cursor()
            interested_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO event_interested (id, event_id, student_username, interested_at)
                VALUES (?, ?, ?, ?)
            ''', (interested_id, event_id, student_username, datetime.now().isoformat()))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Already interested
            return False
        except Exception as e:
            logger.error(f"Error adding interested: {e}")
            return False
    
    def remove_interested(self, event_id, student_username):
        """Remove interested for an event"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                DELETE FROM event_interested 
                WHERE event_id = ? AND student_username = ?
            ''', (event_id, student_username))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing interested: {e}")
            return False
    
    def is_event_interested(self, event_id, student_username):
        """Check if student is interested in an event"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id FROM event_interested 
                WHERE event_id = ? AND student_username = ?
            ''', (event_id, student_username))
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking interested: {e}")
            return False
    
    def get_event_interested_count(self, event_id):
        """Get total interested count for an event"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM event_interested 
                WHERE event_id = ?
            ''', (event_id,))
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting interested count: {e}")
            return 0
    
    def get_student_interested_events(self, student_username):
        """Get all events student is interested in"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT e.* FROM events e
                JOIN event_interested i ON e.id = i.event_id
                WHERE i.student_username = ?
                ORDER BY i.interested_at DESC
            ''', (student_username,))
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting student interested events: {e}")
            return []

# Initialize database
db = DatabaseManager()

# ============================================
# EVENT CARD DISPLAY
# ============================================
def display_event_card(event, current_user=None):
    """Display event card with Event Link, Registration Link, Like, and Interested buttons"""
    if not event or not event.get('id'):
        return
    
    event_id = event.get('id')
    
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        # Create horizontal layout
        col_img, col_info = st.columns([1, 2.5], gap="small")
        
        with col_img:
            # Display event flyer
            flyer = event.get('flyer_path')
            if flyer and flyer.startswith('data:image'):
                try:
                    st.image(flyer, width=200, use_column_width=False)
                except:
                    st.markdown('''
                    <div style="width: 200px; height: 150px; background: #f0f0f0; 
                    display: flex; align-items: center; justify-content: center; border-radius: 8px;">
                        <span style="font-size: 32px;">📷</span>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div style="width: 200px; height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                display: flex; align-items: center; justify-content: center; border-radius: 8px;">
                    <span style="font-size: 32px; color: white;">🎯</span>
                </div>
                ''', unsafe_allow_html=True)
        
        with col_info:
            # Header with title and badges
            title_col, badge_col = st.columns([4, 1])
            with title_col:
                title = event.get('title', 'Untitled Event')
                if len(title) > 60:
                    title = title[:57] + "..."
                st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
            with badge_col:
                if event.get('ai_generated'):
                    st.markdown('<span class="ai-badge">🤖 AI</span>', unsafe_allow_html=True)
            
            # Status and date
            col_status, col_date = st.columns([1, 2])
            with col_status:
                event_date = event.get('event_date')
                st.markdown(get_event_status(event_date), unsafe_allow_html=True)
            with col_date:
                st.caption(f"📅 {format_date(event_date)}")
            
            # Event details
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                venue = event.get('venue', 'TBD')
                if len(venue) > 25:
                    venue = venue[:22] + "..."
                st.caption(f"📍 {venue}")
            with row1_col2:
                event_type = event.get('event_type', 'Event')
                st.caption(f"🏷️ {event_type}")
            
            # Engagement metrics
            likes_count = db.get_event_likes_count(event_id)
            interested_count = db.get_event_interested_count(event_id)
            
            engagement_col1, engagement_col2 = st.columns(2)
            with engagement_col1:
                st.caption(f"❤️ {likes_count} Likes")
            with engagement_col2:
                st.caption(f"⭐ {interested_count} Interested")
            
            # ============================================
            # EVENT LINKS SECTION
            # ============================================
            event_link = event.get('event_link', '')  # New field for event URL
            registration_link = event.get('registration_link', '')
            
            if event_link or registration_link:
                st.markdown("### 🔗 Event Links")
                
                # Show event link if available
                if event_link:
                    st.markdown(f"""
                    **🌐 Event Page:**  
                    [Click here to visit event website]({event_link})  
                    *Official event page with detailed information*
                    """)
                
                # Show registration link if available
                if registration_link:
                    st.markdown(f"""
                    **📝 Registration:**  
                    [Click here to register]({registration_link})  
                    *Official registration link*
                    """)
            
            # Like and Interested buttons
            if current_user:
                col_like, col_interested, col_spacer = st.columns([1, 1, 2])
                
                with col_like:
                    is_liked = db.is_event_liked(event_id, current_user)
                    like_text = "❤️ Liked" if is_liked else "🤍 Like"
                    like_type = "secondary" if is_liked else "primary"
                    
                    if st.button(like_text, key=f"like_{event_id}", 
                               use_container_width=True, type=like_type):
                        if is_liked:
                            if db.remove_like(event_id, current_user):
                                st.rerun()
                        else:
                            if db.add_like(event_id, current_user):
                                st.rerun()
                
                with col_interested:
                    is_interested = db.is_event_interested(event_id, current_user)
                    interested_text = "⭐ Interested" if is_interested else "☆ Interested"
                    interested_type = "secondary" if is_interested else "primary"
                    
                    if st.button(interested_text, key=f"interested_{event_id}", 
                               use_container_width=True, type=interested_type):
                        if is_interested:
                            if db.remove_interested(event_id, current_user):
                                st.rerun()
                        else:
                            if db.add_interested(event_id, current_user):
                                st.rerun()
            
            # Description
            desc = event.get('description', '')
            if desc:
                if len(desc) > 100:
                    with st.expander("📝 Description", expanded=False):
                        st.write(desc)
                    st.caption(f"{desc[:100]}...")
                else:
                    st.caption(desc)
        
        # ============================================
        # REGISTRATION SECTION
        # ============================================
        if current_user:
            st.markdown('<div class="registration-section">', unsafe_allow_html=True)
            
            is_registered = db.is_student_registered(event_id, current_user)
            
            if is_registered:
                st.success("✅ You are already registered for this event")
                
                # Show links for reference
                if event_link or registration_link:
                    col_links = st.columns(2)
                    link_idx = 0
                    
                    if event_link:
                        with col_links[link_idx]:
                            st.markdown(f"""
                            **🌐 Event Page:**  
                            [Visit Event]({event_link})
                            """)
                        link_idx += 1
                    
                    if registration_link:
                        with col_links[link_idx]:
                            st.markdown(f"""
                            **📝 Registration:**  
                            [Register Here]({registration_link})
                            """)
            else:
                # Show registration options
                if registration_link:
                    # Two-column layout for registration options
                    col_ext, col_app = st.columns([1, 1])
                    
                    with col_ext:
                        st.markdown("### 🌐 **External Registration**")
                        st.markdown(f"""
                        **[Click to register externally]({registration_link})**
                        
                        *Use the official registration link*
                        """)
                        
                        # Button to confirm external registration
                        if st.button("✅ I Registered Externally", 
                                   key=f"ext_reg_{event_id}",
                                   use_container_width=True,
                                   type="secondary"):
                            student = db.get_user(current_user)
                            if student:
                                reg_data = {
                                    'id': str(uuid.uuid4()),
                                    'event_id': event_id,
                                    'event_title': event.get('title'),
                                    'student_username': current_user,
                                    'student_name': student.get('name', current_user),
                                    'student_roll': student.get('roll_no', 'N/A'),
                                    'student_dept': student.get('department', 'N/A'),
                                    'status': 'confirmed'
                                }
                                if db.add_registration(reg_data):
                                    st.success("✅ External registration recorded!")
                                    st.rerun()
                    
                    with col_app:
                        st.markdown("### 📱 **College App Registration**")
                        st.markdown("""
                        **Register in our system**
                        
                        *Track attendance & get certificates*
                        """)
                        
                        if st.button("📱 Register in App", 
                                   key=f"app_reg_{event_id}",
                                   use_container_width=True,
                                   type="primary"):
                            student = db.get_user(current_user)
                            if student:
                                reg_data = {
                                    'id': str(uuid.uuid4()),
                                    'event_id': event_id,
                                    'event_title': event.get('title'),
                                    'student_username': current_user,
                                    'student_name': student.get('name', current_user),
                                    'student_roll': student.get('roll_no', 'N/A'),
                                    'student_dept': student.get('department', 'N/A')
                                }
                                if db.add_registration(reg_data):
                                    st.success("✅ Registered in college system!")
                                    st.rerun()
                else:
                    # Only app registration available
                    st.markdown("### 📱 **Register via College App**")
                    
                    if st.button("Register Now", 
                               key=f"reg_{event_id}",
                               use_container_width=True,
                               type="primary"):
                        student = db.get_user(current_user)
                        if student:
                            reg_data = {
                                'id': str(uuid.uuid4()),
                                'event_id': event_id,
                                'event_title': event.get('title'),
                                'student_username': current_user,
                                'student_name': student.get('name', current_user),
                                'student_roll': student.get('roll_no', 'N/A'),
                                'student_dept': student.get('department', 'N/A')
                            }
                            if db.add_registration(reg_data):
                                st.success("✅ Registration successful!")
                                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Creator info
        created_by = event.get('created_by_name', 'Unknown')
        st.caption(f"👤 {created_by}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# LOGIN PAGE
# ============================================
def login_page():
    """Display login page"""
    st.markdown('<div class="college-header"><h2>G H Raisoni College of Engineering and Management</h2><p>Advanced Event Management System</p></div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👑 Admin Login")
        admin_user = st.text_input("Username", value="admin@raisoni", key="admin_user")
        admin_pass = st.text_input("Password", type="password", value="admin123", key="admin_pass")
        
        if st.button("Admin Login", use_container_width=True, type="primary"):
            if db.verify_credentials(admin_user, admin_pass, 'admin'):
                st.session_state.role = 'admin'
                st.session_state.username = admin_user
                st.session_state.name = "Administrator"
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with col2:
        st.subheader("👨‍🏫 Faculty Login")
        faculty_user = st.text_input("Username", value="faculty@raisoni", key="faculty_user")
        faculty_pass = st.text_input("Password", type="password", value="faculty123", key="faculty_pass")
        
        if st.button("Faculty Login", use_container_width=True, type="primary"):
            if db.verify_credentials(faculty_user, faculty_pass, 'faculty'):
                st.session_state.role = 'faculty'
                st.session_state.username = faculty_user
                st.session_state.name = "Faculty Coordinator"
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with col3:
        st.subheader("👨‍🎓 Student Portal")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            student_user = st.text_input("Username", key="student_user_login")
            student_pass = st.text_input("Password", type="password", key="student_pass_login")
            
            if st.button("Student Login", use_container_width=True, type="primary"):
                if db.verify_credentials(student_user, student_pass, 'student'):
                    student = db.get_user(student_user)
                    if student:
                        st.session_state.role = 'student'
                        st.session_state.username = student_user
                        st.session_state.name = student.get('name', student_user)
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("User not found")
                else:
                    st.error("Invalid credentials")
        
        with tab2:
            with st.form("student_registration"):
                st.markdown("### Create Student Account")
                
                name = st.text_input("Full Name *")
                roll_no = st.text_input("Roll Number *")
                department = st.selectbox("Department *", 
                                         ["CSE", "AIML", "IT", "EE", "BCA", "MCA", "BBA", "MBA", "EXTC", "MECH", "CIVIL", "DS", "Other"])
                year = st.selectbox("Year *", ["I", "II", "III", "IV"])
                email = st.text_input("Email *")
                username = st.text_input("Username *")
                password = st.text_input("Password *", type="password")
                confirm_pass = st.text_input("Confirm Password *", type="password")
                
                if st.form_submit_button("Register", use_container_width=True, type="primary"):
                    if password != confirm_pass:
                        st.error("Passwords don't match")
                    elif not all([name, roll_no, email, username, password]):
                        st.error("Please fill all required fields (*)")
                    else:
                        # Check if username exists
                        existing_user = db.get_user(username)
                        if existing_user:
                            st.error("Username already exists")
                        else:
                            user_data = {
                                'id': str(uuid.uuid4()),
                                'name': name,
                                'roll_no': roll_no,
                                'department': department,
                                'year': year,
                                'email': email,
                                'username': username,
                                'password': password,
                                'role': 'student'
                            }
                            
                            if db.add_user(user_data):
                                st.success("✅ Registration successful! Please login.")
                                st.rerun()
                            else:
                                st.error("Registration failed")

# ============================================
# STUDENT DASHBOARD
# ============================================
def student_dashboard():
    """Student dashboard"""
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    
    # Get student info
    student = db.get_user(st.session_state.username)
    if student:
        st.sidebar.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
        st.sidebar.markdown(f"**Department:** {student.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Year:** {student.get('year', 'N/A')}")
    
    display_role_badge('student')
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Events Feed", "My Registrations", "Liked Events", "Interested Events", "My Profile"]
        
        if 'student_page' not in st.session_state:
            st.session_state.student_page = "Events Feed"
        
        for option in nav_options:
            is_active = st.session_state.student_page == option
            button_class = "active" if is_active else ""
            button_text = f"▶ {option}" if is_active else option
            
            if st.button(button_text, key=f"student_{option}", use_container_width=True):
                st.session_state.student_page = option
                st.rerun()
        
        # Engagement statistics
        st.markdown("---")
        st.markdown("### My Engagement")
        
        # Get counts
        liked_events = db.get_student_liked_events(st.session_state.username)
        interested_events = db.get_student_interested_events(st.session_state.username)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("❤️ Liked", len(liked_events))
        with col_stat2:
            st.metric("⭐ Interested", len(interested_events))
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Page content
    selected = st.session_state.student_page
    
    if selected == "Events Feed":
        st.markdown('<h1 class="main-header">🎯 Discover Events</h1>', unsafe_allow_html=True)
        
        # Update event status
        db.update_event_status()
        
        # Filters
        col_filters = st.columns([2, 1, 1, 1])
        with col_filters[0]:
            search = st.text_input("🔍 Search events", placeholder="Search by title, description...")
        with col_filters[1]:
            event_type = st.selectbox("Type", ["All", "Workshop", "Hackathon", "Competition", 
                                              "Bootcamp", "Seminar", "Conference", "Webinar"])
        with col_filters[2]:
            show_only = st.selectbox("Show", ["All", "Upcoming", "Ongoing", "Past"])
        with col_filters[3]:
            ai_only = st.checkbox("🤖 AI-Generated Only")
        
        # Get events
        events = db.get_all_events()
        
        # Apply filters
        filtered_events = events
        
        if search:
            search_lower = search.lower()
            filtered_events = [e for e in filtered_events 
                             if search_lower in e.get('title', '').lower() or 
                             search_lower in e.get('description', '').lower()]
        
        if event_type != "All":
            filtered_events = [e for e in filtered_events if e.get('event_type') == event_type]
        
        if show_only == "Upcoming":
            filtered_events = [e for e in filtered_events if e.get('status') == 'upcoming']
        elif show_only == "Ongoing":
            filtered_events = [e for e in filtered_events if e.get('status') == 'ongoing']
        elif show_only == "Past":
            filtered_events = [e for e in filtered_events if e.get('status') == 'past']
        
        if ai_only:
            filtered_events = [e for e in filtered_events if e.get('ai_generated')]
        
        # Display events count
        st.caption(f"Found {len(filtered_events)} events")
        
        # Display events
        if filtered_events:
            for event in filtered_events:
                display_event_card(event, st.session_state.username)
        else:
            st.info("No events found matching your criteria.")

        pass
    
    elif selected == "My Registrations":
        st.header("📋 My Registrations")
        
        registrations = db.get_registrations_by_student(st.session_state.username)
        
        if not registrations:
            st.info("You haven't registered for any events yet.")
            if st.button("Browse Events", use_container_width=True, type="primary"):
                st.session_state.student_page = "Events Feed"
                st.rerun()
            return
        
        # Calculate statistics
        total = len(registrations)
        upcoming = len([r for r in registrations if r.get('event_status') == 'upcoming'])
        ongoing = len([r for r in registrations if r.get('event_status') == 'ongoing'])
        completed = len([r for r in registrations if r.get('event_status') == 'past'])
        
        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", total)
        with col2:
            st.metric("Upcoming", upcoming)
        with col3:
            st.metric("Ongoing", ongoing)
        with col4:
            st.metric("Completed", completed)
        
        # Display registrations
        for reg in registrations:
            with st.container():
                st.markdown('<div class="event-card">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    event_title = reg.get('event_title', 'Unknown Event')
                    st.markdown(f'<div class="card-title">{event_title}</div>', unsafe_allow_html=True)
                    
                    # Event details
                    event_date = reg.get('event_date')
                    if event_date:
                        st.caption(f"📅 {format_date(event_date)}")
                    
                    venue = reg.get('venue', 'N/A')
                    st.caption(f"📍 {venue}")
                    
                    # Registration details
                    reg_status = reg.get('status', 'pending').title()
                    st.caption(f"📝 Status: {reg_status}")
                
                with col2:
                    # Event status
                    event_status = reg.get('event_status', 'unknown')
                    if event_status == 'upcoming':
                        st.success("🟢 Upcoming")
                    elif event_status == 'ongoing':
                        st.warning("🟡 Ongoing")
                    else:
                        st.error("🔴 Completed")
                
                st.markdown('</div>', unsafe_allow_html=True)
        pass

    elif selected == "Liked Events":
        st.header("❤️ Liked Events")
        
        liked_events = db.get_student_liked_events(st.session_state.username)
        
        if not liked_events:
            st.info("You haven't liked any events yet.")
            st.markdown("""
            **How to like events:**
            1. Go to **Events Feed**
            2. Click the **🤍 Like** button on any event
            3. Your liked events will appear here
            """)
            if st.button("Browse Events", use_container_width=True, type="primary"):
                st.session_state.student_page = "Events Feed"
                st.rerun()
            return
        
        # Filter tabs
        tab1, tab2, tab3 = st.tabs(["All Liked", "Upcoming", "Past"])
        
        with tab1:
            st.caption(f"Total liked events: {len(liked_events)}")
            for event in liked_events:
                display_event_card(event, st.session_state.username)
        
        with tab2:
            upcoming = [e for e in liked_events if e.get('status') == 'upcoming']
            if upcoming:
                st.caption(f"Upcoming liked events: {len(upcoming)}")
                for event in upcoming:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("No upcoming liked events.")
        
        with tab3:
            past = [e for e in liked_events if e.get('status') == 'past']
            if past:
                st.caption(f"Past liked events: {len(past)}")
                for event in past:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("No past liked events.")
    
    elif selected == "Interested Events":
        st.header("⭐ Interested Events")
        
        interested_events = db.get_student_interested_events(st.session_state.username)
        
        if not interested_events:
            st.info("You haven't marked any events as interested yet.")
            st.markdown("""
            **How to mark interest:**
            1. Go to **Events Feed**
            2. Click the **☆ Interested** button on any event
            3. Your interested events will appear here
            """)
            if st.button("Browse Events", use_container_width=True, type="primary"):
                st.session_state.student_page = "Events Feed"
                st.rerun()
            return
        
        # Filter tabs
        tab1, tab2, tab3 = st.tabs(["All Interested", "Upcoming", "Past"])
        
        with tab1:
            st.caption(f"Total interested events: {len(interested_events)}")
            for event in interested_events:
                display_event_card(event, st.session_state.username)
        
        with tab2:
            upcoming = [e for e in interested_events if e.get('status') == 'upcoming']
            if upcoming:
                st.caption(f"Upcoming interested events: {len(upcoming)}")
                for event in upcoming:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("No upcoming interested events.")
        
        with tab3:
            past = [e for e in interested_events if e.get('status') == 'past']
            if past:
                st.caption(f"Past interested events: {len(past)}")
                for event in past:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("No past interested events.")
    
    elif selected == "My Profile":
        st.header("👤 My Profile")
        
        student = db.get_user(st.session_state.username)
        
        if not student:
            st.error("User not found!")
            return
        
        # Profile display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Personal Information")
            st.markdown(f"**Full Name:** {student.get('name', 'N/A')}")
            st.markdown(f"**Roll Number:** {student.get('roll_no', 'N/A')}")
            st.markdown(f"**Department:** {student.get('department', 'N/A')}")
            st.markdown(f"**Year:** {student.get('year', 'N/A')}")
        
        with col2:
            st.markdown("### Account Information")
            st.markdown(f"**Email:** {student.get('email', 'N/A')}")
            st.markdown(f"**Username:** {student.get('username', 'N/A')}")
            st.markdown(f"**Member Since:** {format_date(student.get('created_at'))}")
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 My Statistics")
        
        registrations = db.get_registrations_by_student(st.session_state.username) or []
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Events Registered", len(registrations))
        with col_stat2:
            attended = len([r for r in registrations if r.get('attendance') == 'present'])
            st.metric("Events Attended", attended)

        pass

# ============================================
# FACULTY DASHBOARD
# ============================================
def faculty_dashboard():
    """Faculty coordinator dashboard"""
    st.sidebar.title("👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('faculty')
    
    # Debug section in sidebar
    with st.sidebar.expander("🔧 Debug OpenAI"):
        if st.button("Test OpenAI Connection"):
            try:
                import openai
                st.write(f"OpenAI version: {openai.__version__}")
                
                if hasattr(st, 'secrets'):
                    if 'OPENAI_API_KEY' in st.secrets:
                        api_key = st.secrets["OPENAI_API_KEY"]
                        st.write(f"API key found: {api_key[:10]}...")
                        
                        # Test the API
                        openai.api_key = api_key
                        test_response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": "Say 'Hello'"}],
                            max_tokens=10
                        )
                        st.success("✅ OpenAI connection successful!")
                        st.write(f"Response: {test_response.choices[0].message.content}")
                    else:
                        st.error("OPENAI_API_KEY not found in secrets")
                else:
                    st.error("Secrets not available")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Dashboard", "Create Event", "My Events", "Registrations"]
        
        if 'faculty_page' not in st.session_state:
            st.session_state.faculty_page = "Dashboard"
        
        for option in nav_options:
            is_active = st.session_state.faculty_page == option
            button_class = "active" if is_active else ""
            button_text = f"▶ {option}" if is_active else option
            
            if st.button(button_text, key=f"faculty_{option}", use_container_width=True):
                st.session_state.faculty_page = option
                st.rerun()
        
        st.markdown("---")
        if st.button("Logout", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Page content
    selected = st.session_state.faculty_page
    
    if selected == "Dashboard":
        st.markdown('<h1 class="main-header">Faculty Dashboard</h1>', unsafe_allow_html=True)
        
        # Statistics
        events = db.get_events_by_creator(st.session_state.username)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("My Events", len(events))
        with col2:
            upcoming = len([e for e in events if e.get('status') == 'upcoming'])
            st.metric("Upcoming", upcoming)
        
        # Recent events
        st.subheader("📅 My Recent Events")
        if events:
            for event in events[-3:]:
                display_event_card(event, None)
        else:
            st.info("No events created yet. Create your first event!")
    
    elif selected == "Create Event":
        st.header("➕ Create New Event")
        
        tab1, tab2 = st.tabs(["📝 Manual Entry", "🤖 AI Generator"])
        
        with tab1:
            with st.form("create_event_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    title = st.text_input("Event Title *")
                    event_type = st.selectbox("Event Type *", 
                                            ["Workshop", "Hackathon", "Competition", 
                                             "Bootcamp", "Seminar", "Conference", "Webinar"])
                    event_date = st.date_input("Event Date *", min_value=date.today())
                    event_time = st.time_input("Event Time *")
                    max_participants = st.number_input("Max Participants", min_value=1, value=100)
                
                with col2:
                    venue = st.text_input("Venue *")
                    organizer = st.text_input("Organizer *", value="G H Raisoni College")
                    # NEW: Event link field
                    event_link = st.text_input("Event Website/URL", 
                                             placeholder="https://example.com/event-details")
                    # Existing registration link
                    registration_link = st.text_input("Registration Link", 
                                                    placeholder="https://forms.google.com/registration")
                    
                    # Flyer upload
                    st.subheader("Event Flyer (Optional)")
                    flyer = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png', 'gif', 'webp'], key="faculty_flyer")
                    if flyer:
                        st.image(flyer, width=200)
                
                description = st.text_area("Event Description *", height=150)
                
                submit_button = st.form_submit_button("Create Event", use_container_width=True, type="primary")
                
                if submit_button:
                    if not all([title, event_type, venue, organizer, description]):
                        st.error("Please fill all required fields (*)")
                    else:
                        # Save flyer
                        flyer_path = None
                        if flyer:
                            try:
                                flyer.seek(0)
                                image_bytes = flyer.getvalue()
                                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                file_ext = os.path.splitext(flyer.name)[1].lower()
                                mime_types = {
                                    '.jpg': 'image/jpeg',
                                    '.jpeg': 'image/jpeg',
                                    '.png': 'image/png',
                                    '.gif': 'image/gif'
                                }
                                mime_type = mime_types.get(file_ext, 'image/jpeg')
                                flyer_path = f"data:{mime_type};base64,{image_base64}"
                            except Exception as e:
                                logger.error(f"Error processing image: {e}")
                        
                        # Combine date and time
                        event_datetime = datetime.combine(event_date, event_time)
                        
                        event_data = {
                            'title': title,
                            'description': description,
                            'event_type': event_type,
                            'event_date': event_datetime.isoformat(),
                            'venue': venue,
                            'organizer': organizer,
                            'event_link': event_link,  # NEW
                            'registration_link': registration_link,
                            'max_participants': max_participants,
                            'flyer_path': flyer_path,
                            'created_by': st.session_state.username,
                            'created_by_name': st.session_state.name,
                            'ai_generated': False
                        }
                        
                        if db.add_event(event_data):
                            st.success(f"Event '{title}' created successfully! 🎉")
                            st.rerun()
                        else:
                            st.error("Failed to create event")
        
        with tab2:
            st.subheader("🤖 AI-Powered Event Generator")
            st.markdown("""
            **Instructions:**
            1. Paste text from WhatsApp, email, or any event announcement
            2. AI will automatically extract event details
            3. Review and edit the generated event
            4. Click "Create Event" to save
            """)
    
            # Text input for AI processing
            event_text = st.text_area("Paste event text here:", 
                             placeholder="Example: Join us for a Python Workshop on 15th Dec 2023 at Seminar Hall. Organized by CSE Department...",
                             height=200,
                             key="ai_text_input")
    
            if st.button("🤖 Generate Event with AI", use_container_width=True, type="primary", key="ai_generate_btn"):
                if event_text:
                    # Initialize AI Event Generator
                    ai_generator = AIEventGenerator()
            
                    # Extract event info using AI
                    event_data = ai_generator.extract_event_info(event_text)
            
                    # Store in session state for editing
                    st.session_state.ai_generated_event = event_data
            
                    # Show success message
                    if event_data.get('ai_generated'):
                        st.success("✅ Event details extracted successfully using AI!")
                    else:
                        st.info("⚠️ Using regex fallback for event extraction")
            
                    # Display preview
                    st.subheader("📋 Generated Event Preview")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Title:** {event_data.get('title')}")
                        st.write(f"**Type:** {event_data.get('event_type')}")
                        st.write(f"**Date:** {event_data.get('event_date')}")
                    with col2:
                        st.write(f"**Venue:** {event_data.get('venue')}")
                        st.write(f"**Organizer:** {event_data.get('organizer')}")
            
                    with st.expander("📝 Description Preview"):
                        st.write(event_data.get('description'))
                else:
                    st.error("Please paste some event text first")
            
            # Display and edit AI-generated event
            if 'ai_generated_event' in st.session_state:
                event_data = st.session_state.ai_generated_event
                
                st.markdown("---")
                st.subheader("✏️ Review & Edit AI-Generated Event")
                
                with st.form("ai_event_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        ai_title = st.text_input("Event Title", value=event_data.get('title', ''))
                        ai_event_type = st.selectbox("Event Type", 
                                                   ["Workshop", "Hackathon", "Competition", 
                                                    "Bootcamp", "Seminar", "Conference", "Webinar"],
                                                   index=["Workshop", "Hackathon", "Competition", "Bootcamp", "Seminar", "Conference", "Webinar"].index(event_data.get('event_type', 'workshop').capitalize()) if event_data.get('event_type') in ["Workshop", "Hackathon", "Competition", "Bootcamp", "Seminar", "Conference", "Webinar"] else 0)
                        
                        # Parse date from AI
                        try:
                            ai_date_str = event_data.get('event_date', '')
                            if isinstance(ai_date_str, str):
                                ai_date = datetime.strptime(ai_date_str, '%Y-%m-%d').date()
                            else:
                                ai_date = date.today()
                        except:
                            ai_date = date.today()
                        
                        ai_date = st.date_input("Event Date", value=ai_date, min_value=date.today())
                        ai_time = st.time_input("Event Time", value=datetime.now().time())
                        ai_max_participants = st.number_input("Max Participants", min_value=1, value=event_data.get('max_participants', 100))
                    
                    with col2:
                        ai_venue = st.text_input("Venue", value=event_data.get('venue', 'G H Raisoni College'))
                        ai_organizer = st.text_input("Organizer", value=event_data.get('organizer', 'G H Raisoni College'))
                        ai_reg_link = st.text_input("Registration Link", value=event_data.get('registration_link', ''))
                    
                    ai_description = st.text_area("Event Description", 
                                                value=event_data.get('description', ''),
                                                height=150)
                    
                    # Flyer upload for AI events too
                    st.subheader("Event Flyer (Optional)")
                    ai_flyer = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png', 'gif', 'webp'], key="ai_flyer")
                    if ai_flyer:
                        st.image(ai_flyer, width=200)
                    
                    ai_submit = st.form_submit_button("✅ Create AI-Generated Event", use_container_width=True)
                    
                    if ai_submit:
                        if not all([ai_title, ai_venue, ai_organizer, ai_description]):
                            st.error("Please fill all required fields (*)")
                        else:
                            # Save flyer
                            flyer_path = None
                            if ai_flyer:
                                try:
                                    ai_flyer.seek(0)
                                    image_bytes = ai_flyer.getvalue()
                                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                    file_ext = os.path.splitext(ai_flyer.name)[1].lower()
                                    mime_types = {
                                        '.jpg': 'image/jpeg',
                                        '.jpeg': 'image/jpeg',
                                        '.png': 'image/png',
                                        '.gif': 'image/gif'
                                    }
                                    mime_type = mime_types.get(file_ext, 'image/jpeg')
                                    flyer_path = f"data:{mime_type};base64,{image_base64}"
                                except Exception as e:
                                    logger.error(f"Error processing image: {e}")
                            
                            # Combine date and time
                            event_datetime = datetime.combine(ai_date, ai_time)
                            
                            final_event_data = {
                                'title': ai_title,
                                'description': ai_description,
                                'event_type': ai_event_type,
                                'event_date': event_datetime.isoformat(),
                                'venue': ai_venue,
                                'organizer': ai_organizer,
                                'registration_link': ai_reg_link,
                                'max_participants': ai_max_participants,
                                'flyer_path': flyer_path,
                                'created_by': st.session_state.username,
                                'created_by_name': st.session_state.name,
                                'ai_generated': True,
                                'ai_metadata': event_data.get('ai_metadata', {})
                            }
                            
                            if db.add_event(final_event_data):
                                st.success(f"✅ AI-generated event '{ai_title}' created successfully! 🎉")
                                
                                # Clear session state
                                if 'ai_generated_event' in st.session_state:
                                    del st.session_state.ai_generated_event
                                
                                st.rerun()
                            else:
                                st.error("Failed to create event")
    
    elif selected == "My Events":
        st.header("📋 My Events")
        
        events = db.get_events_by_creator(st.session_state.username)
        
        if not events:
            st.info("You haven't created any events yet.")
            return
        
        # Show engagement statistics
        st.subheader("📊 Event Engagement")
        total_likes = sum(db.get_event_likes_count(e['id']) for e in events)
        total_interested = sum(db.get_event_interested_count(e['id']) for e in events)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Likes", total_likes)
        with col2:
            st.metric("Total Interested", total_interested)
            
        # Filter tabs
        tab1, tab2, tab3 = st.tabs(["Upcoming Events", "Ongoing Events", "Past Events"])
        
        with tab1:
            upcoming = [e for e in events if e.get('status') == 'upcoming']
            if upcoming:
                for event in upcoming:
                    display_event_card(event, None)
            else:
                st.info("No upcoming events.")
        
        with tab2:
            ongoing = [e for e in events if e.get('status') == 'ongoing']
            if ongoing:
                for event in ongoing:
                    display_event_card(event, None)
            else:
                st.info("No ongoing events.")
        
        with tab3:
            past = [e for e in events if e.get('status') == 'past']
            if past:
                for event in past:
                    display_event_card(event, None)
            else:
                st.info("No past events.")
    
    elif selected == "Registrations":
        st.header("📝 Event Registrations")
        
        events = db.get_events_by_creator(st.session_state.username)
        
        if not events:
            st.info("You haven't created any events yet.")
            return
        
        # Select event
        event_titles = [e['title'] for e in events]
        selected_title = st.selectbox("Select Event", event_titles)
        
        if selected_title:
            selected_event = next(e for e in events if e['title'] == selected_title)
            event_id = selected_event['id']
            
            # Get registrations for the selected event
            registrations = db.get_registrations_by_event(event_id)
            
            st.info(f"📊 Registrations for: **{selected_title}**")
            st.caption(f"Total Registrations: {len(registrations)}")
            
            if registrations:
                # Display registrations in a table
                df_data = []
                for reg in registrations:
                    df_data.append({
                        'Student Name': reg.get('student_name'),
                        'Roll No': reg.get('student_roll', 'N/A'),
                        'Department': reg.get('student_dept', 'N/A'),
                        'Year': reg.get('year', 'N/A'),
                        'Status': reg.get('status', 'pending').title(),
                        'Registered On': format_date(reg.get('registered_at')),
                        'Attendance': reg.get('attendance', 'absent').title()
                    })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Export option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"registrations_{selected_title.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                    
                    # Attendance management
                    st.subheader("📋 Mark Attendance")
                    student_options = {f"{reg['student_name']} ({reg['student_roll']})": reg['id'] for reg in registrations}
                    selected_student = st.selectbox("Select Student", list(student_options.keys()))
                    
                    if selected_student:
                        reg_id = student_options[selected_student]
                        col_at1, col_at2 = st.columns(2)
                        with col_at1:
                            if st.button("✅ Mark as Present", use_container_width=True):
                                cursor = db.conn.cursor()
                                cursor.execute("UPDATE registrations SET attendance = 'present' WHERE id = ?", (reg_id,))
                                db.conn.commit()
                                st.success("Attendance marked as Present!")
                                st.rerun()
                        with col_at2:
                            if st.button("❌ Mark as Absent", use_container_width=True):
                                cursor = db.conn.cursor()
                                cursor.execute("UPDATE registrations SET attendance = 'absent' WHERE id = ?", (reg_id,))
                                db.conn.commit()
                                st.success("Attendance marked as Absent!")
                                st.rerun()
            else:
                st.info("No registrations for this event yet.")

# ============================================
# ADMIN DASHBOARD
# ============================================
def admin_dashboard():
    """Admin dashboard"""
    st.sidebar.title("👑 Admin Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('admin')
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Dashboard", "Manage Events", "Manage Users"]
        
        if 'admin_page' not in st.session_state:
            st.session_state.admin_page = "Dashboard"
        
        for option in nav_options:
            is_active = st.session_state.admin_page == option
            button_class = "active" if is_active else ""
            button_text = f"▶ {option}" if is_active else option
            
            if st.button(button_text, key=f"admin_{option}", use_container_width=True):
                st.session_state.admin_page = option
                st.rerun()
        
        st.markdown("---")
        if st.button("Logout", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Page content
    selected = st.session_state.admin_page
    
    if selected == "Dashboard":
        st.markdown('<h1 class="main-header">Admin Dashboard</h1>', unsafe_allow_html=True)
        
        # Update event status
        db.update_event_status()
        
        # Get data
        events = db.get_all_events()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", len(events))
        with col2:
            upcoming = len([e for e in events if e.get('status') == 'upcoming'])
            st.metric("Upcoming Events", upcoming)
        with col3:
            ai_events = len([e for e in events if e.get('ai_generated')])
            st.metric("🤖 AI Events", ai_events)
        with col4:
            today = date.today()
            recent_events = len([e for e in events if datetime.fromisoformat(e.get('event_date').replace('Z', '+00:00')).date() >= today])
            st.metric("Recent Events", recent_events)
        
        # Recent events
        st.subheader("📅 Recent Events")
        if events:
            for event in events[:5]:
                display_event_card(event, None)
        else:
            st.info("No events found.")
    
    elif selected == "Manage Events":
        st.header("📋 Manage Events")
        
        events = db.get_all_events()
        
        if events:
            for event in events:
                with st.container():
                    col_view, col_actions = st.columns([3, 1])
                    
                    with col_view:
                        display_event_card(event, None)
                    
                    with col_actions:
                        st.markdown("### Actions")
                        if st.button("Delete", key=f"delete_{event['id']}", use_container_width=True, type="secondary"):
                            # Delete event from database
                            cursor = db.conn.cursor()
                            try:
                                # First delete registrations
                                cursor.execute("DELETE FROM registrations WHERE event_id = ?", (event['id'],))
                                # Then delete event
                                cursor.execute("DELETE FROM events WHERE id = ?", (event['id'],))
                                db.conn.commit()
                                st.success("Event deleted successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting event: {e}")
        else:
            st.info("No events found.")
    
    elif selected == "Manage Users":
        st.header("👥 Manage Users")
        
        # Get all users
        cursor = db.conn.cursor()
        cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
        users = cursor.fetchall()
        users = [dict(user) for user in users]
        
        if users:
            # Display user statistics
            admin_count = len([u for u in users if u['role'] == 'admin'])
            faculty_count = len([u for u in users if u['role'] == 'faculty'])
            student_count = len([u for u in users if u['role'] == 'student'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Admins", admin_count)
            with col2:
                st.metric("Faculty", faculty_count)
            with col3:
                st.metric("Students", student_count)
            
            # User table
            df_data = []
            for user in users:
                df_data.append({
                    'Name': user.get('name'),
                    'Username': user.get('username'),
                    'Role': user.get('role').title(),
                    'Department': user.get('department', 'N/A'),
                    'Roll No': user.get('roll_no', 'N/A'),
                    'Created': format_date(user.get('created_at')),
                    'Status': 'Active'
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                # User actions
                st.subheader("User Actions")
                user_options = {f"{user['name']} ({user['username']})": user['id'] for user in users}
                selected_user = st.selectbox("Select User", list(user_options.keys()))
                
                if selected_user:
                    user_id = user_options[selected_user]
                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        if st.button("Reset Password", use_container_width=True):
                            cursor = db.conn.cursor()
                            default_pass = hashlib.sha256('password123'.encode()).hexdigest()
                            cursor.execute("UPDATE users SET password = ? WHERE id = ?", (default_pass, user_id))
                            db.conn.commit()
                            st.success("Password reset to 'password123'")
                    with col_act2:
                        if st.button("Delete User", use_container_width=True, type="secondary"):
                            # Don't allow deleting default admin and faculty
                            cursor = db.conn.cursor()
                            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
                            result = cursor.fetchone()
                            if result and result['username'] in ['admin@raisoni', 'faculty@raisoni']:
                                st.error("Cannot delete default admin/faculty accounts")
                            else:
                                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                                db.conn.commit()
                                st.success("User deleted successfully!")
                                st.rerun()
        else:
            st.info("No users found.")

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application"""
    
    # Initialize session state
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'name' not in st.session_state:
        st.session_state.name = None
    
    # Update event status
    db.update_event_status()
    
    # Route based on login status
    if st.session_state.role is None:
        login_page()
    elif st.session_state.role == 'admin':
        admin_dashboard()
    elif st.session_state.role == 'faculty':
        faculty_dashboard()
    elif st.session_state.role == 'student':
        student_dashboard()

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    main()
