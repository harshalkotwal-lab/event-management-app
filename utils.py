# utils.py
"""
Utility functions for the Event Management System
"""
import re
import base64
import hashlib
import uuid
import logging
from datetime import datetime, date, timedelta
from typing import Tuple, Optional
import streamlit as st

# Setup logging
logger = logging.getLogger(__name__)

class Validators:
    """Collection of input validation methods"""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email format and college domain"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, email):
            return False, "Invalid email format"
        
        # Check if it's a college email (optional)
        college_domains = ['ghraisoni.edu', 'raisoni.net', 'ghrce.raisoni.net']
        domain = email.split('@')[-1]
        if not any(domain.endswith(college_domain) for college_domain in college_domains):
            return True, "Warning: Non-college email detected"
        
        return True, "Valid email"
    
    @staticmethod
    def validate_mobile(mobile: str) -> Tuple[bool, str]:
        """Validate Indian mobile number"""
        if not mobile:
            return False, "Mobile number required"
        
        mobile = re.sub(r'\D', '', str(mobile))
        if len(mobile) != 10:
            return False, "Must be 10 digits"
        if mobile[0] not in ['6', '7', '8', '9']:
            return False, "Invalid mobile prefix"
        return True, "Valid mobile number"
    
    @staticmethod
    def validate_password(password: str, min_length=8) -> Tuple[bool, str]:
        """Validate password strength"""
        if len(password) < min_length:
            return False, f"Password must be at least {min_length} characters"
        
        checks = [
            (any(c.isupper() for c in password), "Must contain uppercase letter"),
            (any(c.islower() for c in password), "Must contain lowercase letter"),
            (any(c.isdigit() for c in password), "Must contain number"),
            (any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?/' for c in password), "Must contain special character")
        ]
        
        for check, message in checks:
            if not check:
                return False, message
        
        return True, "Strong password"
    
    @staticmethod
    def format_mobile(mobile: str) -> str:
        """Format mobile number consistently"""
        digits = re.sub(r'\D', '', str(mobile))
        if len(digits) == 10:
            return f"+91 {digits[:5]} {digits[5:]}"
        return mobile
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent XSS and SQL injection"""
        if not text:
            return ""
        
        # Remove potentially dangerous characters
        dangerous_patterns = [
            r'<script.*?>.*?</script>',  # Script tags
            r'on\w+\s*=',  # Event handlers
            r'javascript:',  # JavaScript protocol
            r'vbscript:',  # VBScript protocol
            r'expression\(',  # CSS expressions
            r'url\(',  # CSS URL
            r'--',  # SQL comment
            r';',  # SQL injection
            r'\/\*',  # SQL comment start
            r'\*\/',  # SQL comment end
            r'xp_',  # SQL extended procedure
            r'@@',  # SQL variable
            r'UNION.*SELECT',  # SQL UNION attack
            r'SELECT.*FROM',  # SQL SELECT attack
            r'INSERT.*INTO',  # SQL INSERT attack
            r'DELETE.*FROM',  # SQL DELETE attack
            r'DROP.*TABLE',  # SQL DROP attack
        ]
        
        sanitized = text
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # HTML escape
        sanitized = sanitized.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        sanitized = sanitized.replace('&', '&amp;')
        
        return sanitized.strip()

# Helper functions
def display_role_badge(role):
    """Display role badge"""
    badges = {
        "admin": ("👑 Admin", "background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); color: #DC2626;"),
        "faculty": ("👨‍🏫 Faculty", "background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); color: #1D4ED8;"),
        "student": ("👨‍🎓 Student", "background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); color: #065F46;"),
        "mentor": ("👨‍🏫 Mentor", "background: linear-gradient(135deg, #8B5CF6 0%, #A855F7 100%); color: white;")
    }
    
    if role in badges:
        text, style = badges[role]
        st.markdown(f'<span style="{style} padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 700;">{text}</span>', 
                   unsafe_allow_html=True)

def format_date(date_str):
    """Format date for display"""
    try:
        if isinstance(date_str, str):
            # Handle different date string formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                try:
                    dt = datetime.strptime(date_str.split('+')[0], fmt)
                    break
                except:
                    continue
            else:
                return str(date_str)
        else:
            dt = date_str
        
        return dt.strftime("%d %b %Y, %I:%M %p")
    except:
        return str(date_str)

def get_event_status(event_date):
    """Get event status badge"""
    try:
        if isinstance(event_date, str):
            # Handle different date string formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                try:
                    dt = datetime.strptime(event_date.split('+')[0], fmt)
                    break
                except:
                    continue
            else:
                return '<span style="background: #E5E7EB; color: #374151; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">Unknown</span>'
        else:
            dt = event_date
        
        now = datetime.now()
        if dt > now:
            return '<span style="background: #D1FAE5; color: #065F46; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🟢 Upcoming</span>'
        elif dt.date() == now.date():
            return '<span style="background: #FEF3C7; color: #92400E; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🟡 Ongoing</span>'
        else:
            return '<span style="background: #FEE2E2; color: #DC2626; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🔴 Past</span>'
    except:
        return '<span style="background: #E5E7EB; color: #374151; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">Unknown</span>'

def save_flyer_image(uploaded_file):
    """Save uploaded flyer image and return base64 string"""
    if uploaded_file is None:
        return None
    
    try:
        image_bytes = uploaded_file.getvalue()
        if len(image_bytes) > 5 * 1024 * 1024:  # 5MB limit
            return None
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        
        mime_type = mime_types.get(file_ext, 'image/jpeg')
        return f"data:{mime_type};base64,{image_base64}"
    except Exception as e:
        logger.error(f"Error processing flyer image: {e}")
        return None

def get_custom_css():
    """Return custom CSS styles"""
    return """
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
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .college-header {
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 1rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .event-card {
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 16px;
            margin: 10px 0;
            background: white;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            border-left: 4px solid #3B82F6;
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 6px;
            line-height: 1.3;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            border: 1px solid #E5E7EB;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0.3rem 0;
        }
        
        @media (max-width: 768px) {
            .main-header {
                font-size: 1.8rem;
            }
            
            .event-card {
                padding: 12px;
                margin: 8px 0;
            }
            
            .metric-card {
                padding: 1rem;
            }
            
            .metric-value {
                font-size: 1.8rem;
            }
        }
    </style>
    """
