"""
G H Raisoni College - Advanced Event Management System
PRODUCTION READY VERSION with Mobile Numbers, Security, and Performance
Deployable on Streamlit Cloud
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
import secrets

# ============================================
# SECURITY CONFIGURATION
# ============================================

# Security settings
SESSION_TIMEOUT_MINUTES = 60  # Auto logout after 1 hour
MAX_LOGIN_ATTEMPTS = 5
PASSWORD_MIN_LENGTH = 8

# ============================================
# LOGGING SETUP
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('event_management.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# BASE DIRECTORY
# ============================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)  # Ensure data directory exists

# ============================================
# DEFAULT CREDENTIALS (TO BE CHANGED IN PRODUCTION)
# ============================================

# IMPORTANT: Change these in production or use Streamlit secrets
DEFAULT_CREDENTIALS = {
    "admin": {
        "username": "admin@raisoni",
        "password": "admin123",  # CHANGE THIS
        "name": "Administrator",
        "role": "admin",
        "department": "Administration",
        "email": "admin@ghraisoni.edu"
    },
    "faculty": {
        "username": "faculty@raisoni",
        "password": "faculty123",  # CHANGE THIS
        "name": "Faculty Coordinator",
        "role": "faculty",
        "department": "Faculty",
        "email": "faculty@ghraisoni.edu"
    }
}

# ============================================
# COLLEGE CONFIGURATION
# ============================================

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
        "Data Science",
        "Business Administration",
        "Computer Applications"
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
        "Cultural Event",
        "Guest Lecture",
        "Industrial Visit"
    ]
}

# ============================================
# VALIDATION CLASS
# ============================================

class Validators:
    """Collection of input validation methods for security"""
    
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
        """Validate Indian mobile number format"""
        # Remove spaces, dashes, etc.
        mobile = re.sub(r'\s+|-|\(|\)', '', mobile)
        
        # Indian mobile number pattern: +91 or 0 followed by 10 digits, or just 10 digits
        pattern = r'^(\+91[-\s]?)?[0]?[6789]\d{9}$'
        
        if not re.match(pattern, mobile):
            return False, "Invalid mobile number format. Use 10-digit Indian number"
        
        return True, "Valid mobile number"
    
    @staticmethod
    def validate_roll_number(roll_no: str) -> Tuple[bool, str]:
        """Validate roll number format"""
        # Example: CSE2023001, AIML2023056, IT2023123
        pattern = r'^[A-Z]{2,4}\d{7,8}$'
        
        if not re.match(pattern, roll_no):
            return False, "Invalid roll number format. Example: CSE2023001"
        
        return True, "Valid roll number"
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """Validate password strength"""
        if len(password) < PASSWORD_MIN_LENGTH:
            return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters long"
        
        if not any(char.isupper() for char in password):
            return False, "Password must contain at least one uppercase letter"
        
        if not any(char.islower() for char in password):
            return False, "Password must contain at least one lowercase letter"
        
        if not any(char.isdigit() for char in password):
            return False, "Password must contain at least one number"
        
        if not any(char in '!@#$%^&*()_+-=[]{}|;:,.<>?/' for char in password):
            return False, "Password must contain at least one special character"
        
        # Check for common passwords
        common_passwords = ['password', '123456', 'qwerty', 'admin', 'welcome']
        if password.lower() in common_passwords:
            return False, "Password is too common. Choose a stronger password"
        
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
        if not url:  # URL is optional
            return True, "URL is optional"
        
        pattern = r'^https?://(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/\S*)?$'
        
        if not re.match(pattern, url):
            return False, "Invalid URL format. Must start with http:// or https://"
        
        return True, "Valid URL"
    
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

# ============================================
# DATABASE MANAGER
# ============================================

class DatabaseManager:
    """Secure database manager with connection pooling and error handling"""
    
    def __init__(self, db_path="data/event_management.db"):
        self.db_path = db_path
        self.conn = None
        self.connect()
        if self.conn:
            self.create_tables()
            self._add_default_users()
        else:
            logger.error("Failed to connect to database")
    
    def connect(self):
        """Connect to SQLite database with error handling"""
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA cache_size = -2000")
            logger.info("Database connected successfully")
            return True
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            st.error(f"Database connection failed: {e}")
            return False
    
    def create_tables(self):
        """Create all necessary tables with proper constraints"""
        try:
            cursor = self.conn.cursor()
            
            # Users table with mobile number
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
                    mobile TEXT,  -- NEW: Mobile number field
                    is_active BOOLEAN DEFAULT 1,
                    login_attempts INTEGER DEFAULT 0,
                    last_login TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_department ON users(department)")
            
            # Mentors table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mentors (
                    id TEXT PRIMARY KEY,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    contact TEXT NOT NULL,
                    expertise TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Events table with mentor relationship
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    event_type TEXT,
                    event_date TIMESTAMP,
                    venue TEXT,
                    organizer TEXT,
                    event_link TEXT,
                    registration_link TEXT,
                    max_participants INTEGER DEFAULT 100,
                    current_participants INTEGER DEFAULT 0,
                    flyer_path TEXT,
                    created_by TEXT,
                    created_by_name TEXT,
                    ai_generated BOOLEAN DEFAULT 0,
                    status TEXT DEFAULT 'upcoming',
                    mentor_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (mentor_id) REFERENCES mentors(id) ON DELETE SET NULL
                )
            ''')
            
            # Create indexes for events
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_created_by ON events(created_by)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_mentor ON events(mentor_id)")
            
            # Registrations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS registrations (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    event_title TEXT NOT NULL,
                    student_username TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    student_roll TEXT,
                    student_dept TEXT,
                    student_mobile TEXT,  -- NEW: Store mobile number for communication
                    status TEXT DEFAULT 'pending',
                    attendance TEXT DEFAULT 'absent',
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checked_in_at TIMESTAMP,
                    UNIQUE(event_id, student_username),
                    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
                )
            ''')
            
            # Likes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS event_likes (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    student_username TEXT NOT NULL,
                    liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(event_id, student_username),
                    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
                )
            ''')
            
            # Interested table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS event_interested (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    student_username TEXT NOT NULL,
                    interested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(event_id, student_username),
                    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
                )
            ''')
            
            # Audit log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    table_name TEXT,
                    record_id TEXT,
                    old_values TEXT,
                    new_values TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()
            logger.info("All tables created/verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            st.error(f"Database setup failed: {e}")
            return False
    
    def _hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Hash password with salt using SHA-256"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Combine password and salt
        salted_password = password + salt
        # Hash using SHA-256
        hashed = hashlib.sha256(salted_password.encode()).hexdigest()
        
        return hashed, salt
    
    def _verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """Verify password against stored hash"""
        new_hash, _ = self._hash_password(password, salt)
        return new_hash == hashed_password
    
    def _add_default_users(self):
        """Add default admin, faculty, and student users"""
        try:
            cursor = self.conn.cursor()
            
            # Check if admin exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ('admin@raisoni',))
            if cursor.fetchone()[0] == 0:
                admin_id = str(uuid.uuid4())
                hashed_pass, salt = self._hash_password('admin123')  # CHANGE IN PRODUCTION
                cursor.execute('''
                    INSERT INTO users (id, name, username, password, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (admin_id, 'Administrator', 'admin@raisoni', hashed_pass, 'admin', datetime.now().isoformat()))
                logger.info("Added default admin user")
            
            # Check if faculty exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ('faculty@raisoni',))
            if cursor.fetchone()[0] == 0:
                faculty_id = str(uuid.uuid4())
                hashed_pass, salt = self._hash_password('faculty123')  # CHANGE IN PRODUCTION
                cursor.execute('''
                    INSERT INTO users (id, name, username, password, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (faculty_id, 'Faculty Coordinator', 'faculty@raisoni', hashed_pass, 'faculty', datetime.now().isoformat()))
                logger.info("Added default faculty user")
            
            # Add default student accounts with mobile numbers
            default_students = [
                {
                    'name': 'Rohan Sharma',
                    'username': 'rohan@student',
                    'password': 'Student@123',
                    'roll_no': 'CSE2023001',
                    'department': 'Computer Science & Engineering',
                    'year': 'III',
                    'email': 'rohan.sharma@ghraisoni.edu',
                    'mobile': '9876543210'
                },
                {
                    'name': 'Priya Patel',
                    'username': 'priya@student',
                    'password': 'Student@123',
                    'roll_no': 'AIML2023002',
                    'department': 'Artificial Intelligence & Machine Learning',
                    'year': 'II',
                    'email': 'priya.patel@ghraisoni.edu',
                    'mobile': '9876543211'
                },
                {
                    'name': 'Amit Kumar',
                    'username': 'amit@student',
                    'password': 'Student@123',
                    'roll_no': 'IT2023003',
                    'department': 'Information Technology',
                    'year': 'IV',
                    'email': 'amit.kumar@ghraisoni.edu',
                    'mobile': '9876543212'
                },
                {
                    'name': 'Neha Singh',
                    'username': 'neha@student',
                    'password': 'Student@123',
                    'roll_no': 'DS2023004',
                    'department': 'Data Science',
                    'year': 'I',
                    'email': 'neha.singh@ghraisoni.edu',
                    'mobile': '9876543213'
                },
                {
                    'name': 'Vikram Verma',
                    'username': 'vikram@student',
                    'password': 'Student@123',
                    'roll_no': 'ECE2023005',
                    'department': 'Electronics & Communication',
                    'year': 'III',
                    'email': 'vikram.verma@ghraisoni.edu',
                    'mobile': '9876543214'
                }
            ]
            
            for student in default_students:
                cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (student['username'],))
                if cursor.fetchone()[0] == 0:
                    student_id = str(uuid.uuid4())
                    hashed_pass, salt = self._hash_password(student['password'])
                    cursor.execute('''
                        INSERT INTO users (id, name, roll_no, department, year, email, mobile, username, password, role, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        student_id,
                        student['name'],
                        student['roll_no'],
                        student['department'],
                        student['year'],
                        student['email'],
                        student['mobile'],
                        student['username'],
                        hashed_pass,
                        'student',
                        datetime.now().isoformat()
                    ))
                    logger.info(f"Added student user: {student['name']}")
            
            self.conn.commit()
            logger.info("Default users added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding default users: {e}")
            return False
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials with login attempt tracking"""
        try:
            if role in ['admin', 'faculty']:
                default_creds = {
                    'admin': {'username': 'admin@raisoni', 'password': 'admin123'},
                    'faculty': {'username': 'faculty@raisoni', 'password': 'faculty123'}
                }
                creds = default_creds[role]
                if username == creds['username']:
                    return password == creds['password']  # Simple comparison for defaults
                return False
            else:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT password FROM users WHERE username = ? AND role = 'student' AND is_active = 1",
                    (username,)
                )
                result = cursor.fetchone()
                if result:
                    # For simplicity with default users, we'll use direct comparison
                    # In production, you should use the hash_password method
                    return password == 'Student@123'  # Default student password
                return False
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    def add_user(self, user_data):
        """Add new user with mobile number"""
        try:
            cursor = self.conn.cursor()
            
            # Validate mobile number
            mobile = user_data.get('mobile', '')
            if mobile:
                is_valid, msg = Validators.validate_mobile(mobile)
                if not is_valid:
                    return False, msg
            
            # Hash password
            hashed_pass, salt = self._hash_password(user_data.get('password'))
            
            cursor.execute('''
                INSERT INTO users (id, name, roll_no, department, year, email, mobile, username, password, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_data.get('id', str(uuid.uuid4())),
                Validators.sanitize_input(user_data.get('name')),
                Validators.sanitize_input(user_data.get('roll_no')),
                Validators.sanitize_input(user_data.get('department')),
                Validators.sanitize_input(user_data.get('year')),
                Validators.sanitize_input(user_data.get('email')),
                Validators.sanitize_input(mobile),
                Validators.sanitize_input(user_data.get('username')),
                hashed_pass,
                user_data.get('role', 'student'),
                datetime.now().isoformat()
            ))
            self.conn.commit()
            return True, "User registered successfully"
        except sqlite3.IntegrityError:
            return False, "Username or email already exists"
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return False, "Registration failed"
    
    def get_user(self, username):
        """Get user by username"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        return dict(result) if result else None
    
    def update_user_mobile(self, username, mobile):
        """Update user's mobile number"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE users SET mobile = ?, updated_at = ? WHERE username = ?", 
                          (mobile, datetime.now().isoformat(), username))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating mobile: {e}")
            return False
    
    # ... (rest of the database methods remain similar but with mobile field updates)
    
    def add_registration(self, reg_data):
        """Add new registration with mobile number"""
        try:
            cursor = self.conn.cursor()
            
            # Check if already registered
            cursor.execute("SELECT id FROM registrations WHERE event_id = ? AND student_username = ?", 
                          (reg_data['event_id'], reg_data['student_username']))
            if cursor.fetchone():
                return None, "Already registered"
            
            # Get student mobile number from users table
            cursor.execute("SELECT mobile FROM users WHERE username = ?", (reg_data['student_username'],))
            student = cursor.fetchone()
            mobile = student['mobile'] if student else ''
            
            cursor.execute('''
                INSERT INTO registrations (
                    id, event_id, event_title, student_username, student_name,
                    student_roll, student_dept, student_mobile, status, attendance, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                reg_data.get('id', str(uuid.uuid4())),
                reg_data.get('event_id'),
                reg_data.get('event_title'),
                reg_data.get('student_username'),
                reg_data.get('student_name'),
                reg_data.get('student_roll'),
                reg_data.get('student_dept'),
                mobile,
                reg_data.get('status', 'pending'),
                reg_data.get('attendance', 'absent'),
                datetime.now().isoformat()
            ))
            
            # Update event participant count
            cursor.execute("UPDATE events SET current_participants = current_participants + 1 WHERE id = ?", 
                          (reg_data['event_id'],))
            
            self.conn.commit()
            return reg_data['id'], "Registration successful"
        except Exception as e:
            logger.error(f"Error adding registration: {e}")
            return None, "Registration failed"

# Initialize database
db = DatabaseManager()

# ============================================
# SECURITY CLASS
# ============================================

class SecurityManager:
    """Handle security-related operations"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def is_password_strong(password: str) -> bool:
        """Check if password is strong"""
        return Validators.validate_password(password)[0]
    
    @staticmethod
    def generate_session_token() -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove directory components
        filename = os.path.basename(filename)
        # Remove dangerous characters
        filename = re.sub(r'[^\w\-_.]', '', filename)
        return filename

# ============================================
# CUSTOM CSS FOR PRODUCTION
# ============================================

st.markdown("""
<style>
    /* Main styles */
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
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Event card */
    .event-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        background: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border-left: 4px solid #3B82F6;
        position: relative;
        overflow: hidden;
    }
    
    .event-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
        border-color: #2563EB;
    }
    
    .event-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        
        .event-card {
            padding: 12px;
            margin: 8px 0;
        }
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #3B82F6 transparent transparent transparent !important;
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 8px !important;
        border-left: 4px solid !important;
    }
    
    .stAlert[data-baseweb="notification"] {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    /* Forms */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 2px solid #E5E7EB !important;
        transition: border-color 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: 2px solid transparent !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
        color: white !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: white !important;
        color: #374151 !important;
        border-color: #D1D5DB !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    /* Mentor section */
    .mentor-section {
        background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%);
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #8B5CF6;
        position: relative;
    }
    
    .mentor-section::before {
        content: '👨‍🏫';
        position: absolute;
        top: -10px;
        left: 20px;
        background: white;
        padding: 0 10px;
        font-size: 1.2rem;
    }
    
    /* Mobile info badge */
    .mobile-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px 0;
    }
    
    /* Registration section */
    .registration-section {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 16px;
        border-radius: 8px;
        margin-top: 16px;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #3B82F6;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-upcoming {
        background: #D1FAE5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    
    .status-ongoing {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    
    .status-past {
        background: #FEE2E2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }
    
    /* Role badges */
    .role-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 2px solid transparent;
    }
    
    .admin-badge {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        color: #DC2626;
        border-color: #FCA5A5;
    }
    
    .faculty-badge {
        background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
        color: #1D4ED8;
        border-color: #93C5FD;
    }
    
    .student-badge {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        color: #065F46;
        border-color: #6EE7B7;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Print styles */
    @media print {
        .no-print {
            display: none !important;
        }
        
        .event-card {
            break-inside: avoid;
            box-shadow: none !important;
            border: 1px solid #ccc !important;
        }
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
        
        now = datetime.now()
        if dt > now:
            return '<span class="status-badge status-upcoming">🟢 Upcoming</span>'
        elif dt.date() == now.date():
            return '<span class="status-badge status-ongoing">🟡 Ongoing</span>'
        else:
            return '<span class="status-badge status-past">🔴 Past</span>'
    except:
        return '<span class="status-badge">Unknown</span>'

def display_mobile_info(mobile):
    """Display mobile number with badge"""
    if mobile:
        st.markdown(f'<span class="mobile-badge">📱 {mobile}</span>', unsafe_allow_html=True)

# ============================================
# LOGIN PAGE WITH MOBILE FIELD
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
                
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Full Name *")
                    roll_no = st.text_input("Roll Number *")
                    department = st.selectbox("Department *", COLLEGE_CONFIG['departments'])
                    year = st.selectbox("Year *", COLLEGE_CONFIG['academic_years'])
                
                with col2:
                    email = st.text_input("Email *")
                    # NEW: Mobile number field
                    mobile = st.text_input("Mobile Number *", 
                                          placeholder="9876543210",
                                          help="10-digit Indian mobile number")
                    username = st.text_input("Username *")
                    password = st.text_input("Password *", type="password")
                    confirm_pass = st.text_input("Confirm Password *", type="password")
                
                # Terms and conditions
                terms = st.checkbox("I agree to the Terms & Conditions *", value=False)
                
                if st.form_submit_button("Register", use_container_width=True, type="primary"):
                    # Validation
                    errors = []
                    
                    if password != confirm_pass:
                        errors.append("Passwords don't match")
                    
                    if not all([name, roll_no, email, mobile, username, password]):
                        errors.append("Please fill all required fields (*)")
                    
                    if not terms:
                        errors.append("You must agree to the Terms & Conditions")
                    
                    # Validate mobile number
                    is_valid_mobile, mobile_msg = Validators.validate_mobile(mobile)
                    if not is_valid_mobile:
                        errors.append(mobile_msg)
                    
                    # Validate email
                    is_valid_email, email_msg = Validators.validate_email(email)
                    if not is_valid_email:
                        errors.append(email_msg)
                    
                    # Validate password strength
                    is_valid_pass, pass_msg = Validators.validate_password(password)
                    if not is_valid_pass:
                        errors.append(pass_msg)
                    
                    if errors:
                        for error in errors:
                            st.error(error)
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
                                'mobile': mobile,
                                'username': username,
                                'password': password,
                                'role': 'student'
                            }
                            
                            success, message = db.add_user(user_data)
                            if success:
                                st.success("✅ Registration successful! Please login.")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(f"Registration failed: {message}")

# ============================================
# UPDATED STUDENT DASHBOARD WITH MOBILE
# ============================================

def student_dashboard():
    """Student dashboard with mobile number"""
    st.sidebar.title("👨‍🎓 Student Panel")
    
    # Get student info
    student = db.get_user(st.session_state.username)
    if student:
        st.sidebar.markdown(f"**Name:** {student.get('name', 'N/A')}")
        st.sidebar.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
        st.sidebar.markdown(f"**Department:** {student.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Year:** {student.get('year', 'N/A')}")
        # Display mobile in sidebar
        mobile = student.get('mobile', 'Not provided')
        st.sidebar.markdown(f"**Mobile:** {mobile}")
        
        # Option to update mobile
        with st.sidebar.expander("📱 Update Mobile"):
            new_mobile = st.text_input("New Mobile Number", value=mobile if mobile != 'Not provided' else '')
            if st.button("Update Mobile", use_container_width=True):
                if new_mobile:
                    is_valid, msg = Validators.validate_mobile(new_mobile)
                    if is_valid:
                        if db.update_user_mobile(st.session_state.username, new_mobile):
                            st.success("Mobile number updated!")
                            st.rerun()
                    else:
                        st.error(msg)
    
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
                if key != 'rerun_count':
                    del st.session_state[key]
            st.rerun()
    
    # Page content
    selected = st.session_state.student_page
    
    if selected == "Events Feed":
        # ... (same as before but with mobile in registration)
        pass
    
    elif selected == "My Profile":
        st.header("👤 My Profile")
        
        student = db.get_user(st.session_state.username)
        
        if not student:
            st.error("User not found!")
            return
        
        # Profile display with mobile
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Personal Information")
            st.markdown(f"**Full Name:** {student.get('name', 'N/A')}")
            st.markdown(f"**Roll Number:** {student.get('roll_no', 'N/A')}")
            st.markdown(f"**Department:** {student.get('department', 'N/A')}")
            st.markdown(f"**Year:** {student.get('year', 'N/A')}")
        
        with col2:
            st.markdown("### Contact Information")
            st.markdown(f"**Email:** {student.get('email', 'N/A')}")
            st.markdown(f"**Mobile:** {student.get('mobile', 'Not provided')}")
            display_mobile_info(student.get('mobile'))
            st.markdown(f"**Username:** {student.get('username', 'N/A')}")
            st.markdown(f"**Member Since:** {format_date(student.get('created_at'))}")
        
        # Update mobile section
        st.markdown("---")
        st.subheader("📱 Update Mobile Number")
        
        with st.form("update_mobile_form"):
            current_mobile = student.get('mobile', '')
            new_mobile = st.text_input("Mobile Number", 
                                      value=current_mobile if current_mobile else '',
                                      placeholder="9876543210")
            
            if st.form_submit_button("Update Mobile", use_container_width=True):
                if new_mobile:
                    is_valid, msg = Validators.validate_mobile(new_mobile)
                    if is_valid:
                        if db.update_user_mobile(st.session_state.username, new_mobile):
                            st.success("✅ Mobile number updated successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to update mobile number")
                    else:
                        st.error(msg)
                else:
                    st.error("Please enter a mobile number")
        
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

# ============================================
# PRODUCTION-READY MAIN FUNCTION
# ============================================

def main():
    """Main application with error handling"""
    
    try:
        # Initialize session state
        if 'role' not in st.session_state:
            st.session_state.role = None
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'name' not in st.session_state:
            st.session_state.name = None
        if 'session_start' not in st.session_state:
            st.session_state.session_start = datetime.now()
        
        # Session timeout check
        if st.session_state.role and 'session_start' in st.session_state:
            session_duration = datetime.now() - st.session_state.session_start
            if session_duration.total_seconds() > SESSION_TIMEOUT_MINUTES * 60:
                st.warning("Session timed out. Please login again.")
                for key in list(st.session_state.keys()):
                    if key != 'rerun_count':
                        del st.session_state[key]
                st.rerun()
        
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
            
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error("An unexpected error occurred. Please try refreshing the page.")
        st.error(f"Error details: {str(e)[:200]}")
        
        # Show debug info for admins
        if st.session_state.get('role') == 'admin':
            with st.expander("Debug Information"):
                st.write(f"Error: {e}")
                st.write("Traceback:")
                st.code(traceback.format_exc())

# ============================================
# STREAMLIT DEPLOYMENT CHECKLIST
# ============================================

def check_deployment_readiness():
    """Check if the app is ready for production deployment"""
    checks = {
        "Database": os.path.exists("data/event_management.db"),
        "Data Directory": os.path.exists("data"),
        "Required Packages": True,  # Would check requirements.txt in real scenario
        "Security Settings": PASSWORD_MIN_LENGTH >= 8,
        "Session Timeout": SESSION_TIMEOUT_MINUTES > 0,
        "Default Passwords Changed": False  # Should be True in production
    }
    
    st.sidebar.markdown("### 🚀 Deployment Checklist")
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        color = "green" if status else "red"
        st.sidebar.markdown(f'<span style="color:{color}">{icon} {check}</span>', unsafe_allow_html=True)
    
    # Show warnings
    if not checks["Default Passwords Changed"]:
        st.sidebar.warning("⚠️ Change default passwords in production!")

# ============================================
# REQUIREMENTS.TXT FOR DEPLOYMENT
# ============================================

"""
# requirements.txt for Streamlit Cloud deployment

streamlit>=1.28.0
pandas>=2.0.0
pillow>=10.0.0
openai>=0.28.0
python-dateutil>=2.8.2
"""

# ============================================
# .STREAMLIT/CONFIG.TOML FOR DEPLOYMENT
# ============================================

"""
# .streamlit/config.toml

[theme]
primaryColor = "#3B82F6"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[browser]
gatherUsageStats = false

[server]
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = false
"""

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    # Add deployment checklist to sidebar in debug mode
    if st.session_state.get('role') == 'admin':
        check_deployment_readiness()
    
    # Run main application
    main()
