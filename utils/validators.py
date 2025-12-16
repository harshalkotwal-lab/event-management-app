"""
Input validation utilities
"""

import re
from datetime import datetime
from typing import Tuple, Optional


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
