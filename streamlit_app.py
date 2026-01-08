"""
G H Raisoni College - Advanced Event Management System
PRODUCTION READY with Supabase PostgreSQL (Free Forever)
Deployable on Streamlit Cloud
WITH GAMIFICATION LEADERBOARD SYSTEM
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
import base64
import re
import traceback
import logging
import time
import secrets
from typing import Tuple, Optional
from pathlib import Path
import sqlite3

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
# CONFIGURATION
# ============================================

# Set to True for Supabase PostgreSQL, False for SQLite
USE_SUPABASE = True  # Change this to switch databases

# Security settings
SESSION_TIMEOUT_MINUTES = 60
MAX_LOGIN_ATTEMPTS = 5
PASSWORD_MIN_LENGTH = 8

# College configuration
COLLEGE_CONFIG = {
    "name": "G H Raisoni College of Engineering and Management, Jalgaon",
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
        "Workshop", "Hackathon", "Competition", "Bootcamp", "Seminar",
        "Conference", "Webinar", "Training", "Symposium", "Cultural Event",
        "Guest Lecture", "Industrial Visit"
    ]
}

# ============================================
# GAMIFICATION CONFIGURATION
# ============================================

GAMIFICATION_CONFIG = {
    "points": {
        "registration": 50,
        "shortlisted": 100,
        "winner": 200
    },
    "badges": {
        "registration": "🏅 Participant",
        "shortlisted": "⭐ Shortlisted",
        "winner": "🏆 Winner",
        "top_10": "👑 Top 10 Leader",
        "top_25": "🎖️ Top 25 Achiever",
        "top_50": "🌟 Rising Star",
        "mentor_choice": "💎 Mentor's Choice",
        "most_active": "🚀 Most Active"
    },
    "leaderboard": {
        "top_n": 15,
        "update_interval": 3600,
        "department_top_n": 5
    }
}

# ============================================
# VALIDATION CLASS
# ============================================

class Validators:
    """Collection of input validation methods"""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email format and college domain"""
        if not email:
            return False, "Email is required"
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, email):
            return False, "Invalid email format"
        
        college_domains = ['ghraisoni.edu', 'raisoni.net', 'ghrce.raisoni.net']
        domain = email.split('@')[-1].lower()
        
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
    def validate_password(password: str) -> Tuple[bool, str]:
        """Validate password strength"""
        if len(password) < PASSWORD_MIN_LENGTH:
            return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters"
        
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
        
        dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'on\w+\s*=',
            r'javascript:',
            r'vbscript:',
            r'expression\(',
            r'url\(',
            r'--',
            r';',
            r'\/\*',
            r'\*\/',
            r'xp_',
            r'@@',
            r'UNION.*SELECT',
            r'SELECT.*FROM',
            r'INSERT.*INTO',
            r'DELETE.*FROM',
            r'DROP.*TABLE',
        ]
        
        sanitized = text
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        sanitized = sanitized.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        sanitized = sanitized.replace('&', '&amp;')
        
        return sanitized.strip()

# ============================================
# SUPABASE CLIENT (PostgreSQL)
# ============================================

class SupabaseClient:
    """Supabase PostgreSQL client using HTTP REST API"""
    
    def __init__(self):
        self.url = None
        self.key = None
        self.headers = None
        self.is_configured = False
        
        self._initialize_supabase()
    
    def _initialize_supabase(self):
        """Initialize Supabase connection"""
        try:
            if hasattr(st, 'secrets'):
                if 'SUPABASE' in st.secrets:
                    self.url = st.secrets.SUPABASE.get('url')
                    self.key = st.secrets.SUPABASE.get('key')
                    
                    if self.url and self.key:
                        if not self.url.startswith(('http://', 'https://')):
                            self.url = f'https://{self.url}'
                        
                        self.url = self.url.rstrip('/')
                        
                        self.headers = {
                            'apikey': self.key,
                            'Authorization': f'Bearer {self.key}',
                            'Content-Type': 'application/json',
                            'Prefer': 'return=representation'
                        }
                        self.is_configured = True
                        logger.info("✅ Supabase configured successfully")
                else:
                    logger.warning("⚠️ Supabase credentials not found in secrets")
            else:
                logger.warning("⚠️ Streamlit secrets not available")
                
        except Exception as e:
            logger.error(f"Supabase initialization error: {e}")
    
    def execute_query(self, table, method='GET', data=None, filters=None, limit=1000, order_by=None):
        """Execute REST API query to Supabase"""
        if not self.is_configured:
            logger.error("Supabase not configured")
            return None
        
        try:
            import requests
            
            url = f"{self.url}/rest/v1/{table}"
            params = []
            
            if filters:
                for k, v in filters.items():
                    if v is not None:
                        params.append(f"{k}=eq.{v}")
            
            if order_by:
                params.append(f"order={order_by}")
            
            params.append(f"limit={limit}")
            
            if params:
                url = f"{url}?{'&'.join(params)}"
            
            timeout = 30
            if method == 'GET':
                response = requests.get(url, headers=self.headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=data, timeout=timeout)
            elif method == 'PATCH':
                response = requests.patch(url, headers=self.headers, json=data, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers, timeout=timeout)
            else:
                logger.error(f"Unsupported method: {method}")
                return None
            
            if response.status_code >= 400:
                logger.error(f"Supabase API error {response.status_code}: {response.text}")
                return None
            
            if method == 'GET':
                return response.json() if response.text else []
            elif method in ['POST', 'PATCH']:
                return response.json() if response.text else True
            elif method == 'DELETE':
                return response.status_code in [200, 204]
            return True
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Supabase API request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Supabase query: {e}")
            return None
    
    def insert(self, table, data):
        """Insert data into table"""
        result = self.execute_query(table, 'POST', data)
        return bool(result) if result is not None else False
    
    def select(self, table, filters=None, limit=1000, order_by=None):
        """Select data from table"""
        return self.execute_query(table, 'GET', filters=filters, limit=limit, order_by=order_by)
    
    def update(self, table, filters, data):
        """Update data in table"""
        result = self.execute_query(table, 'PATCH', data, filters)
        return bool(result) if result is not None else False
    
    def delete(self, table, filters):
        """Delete data from table"""
        result = self.execute_query(table, 'DELETE', filters=filters)
        return result if result is not None else False

# ============================================
# SQLITE CLIENT (Fallback)
# ============================================

class SQLiteClient:
    """SQLite client for local development"""
    
    def __init__(self, db_path="data/event_management.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize()
    
    def _initialize(self):
        """Initialize SQLite database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA cache_size = -2000")
            logger.info("✅ SQLite database initialized")
        except Exception as e:
            logger.error(f"SQLite initialization error: {e}")
            raise
    
    def execute_query(self, query, params=None, fetchone=False, fetchall=False, commit=False):
        """Execute SQL query"""
        try:
            cursor = self.conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if commit:
                self.conn.commit()
            
            if fetchone:
                result = cursor.fetchone()
                return dict(result) if result else None
            
            if fetchall:
                results = cursor.fetchall()
                return [dict(row) for row in results]
            
            return cursor
        except Exception as e:
            logger.error(f"SQLite error: {e}")
            return None
    
    def insert(self, table, data):
        """Insert data into table"""
        try:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            cursor = self.execute_query(query, tuple(data.values()))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Insert error: {e}")
            return False
    
    def select(self, table, filters=None, limit=1000):
        """Select data from table"""
        try:
            query = f"SELECT * FROM {table}"
            if filters:
                conditions = ' AND '.join([f"{k} = ?" for k in filters.keys()])
                query = f"{query} WHERE {conditions}"
            query = f"{query} LIMIT {limit}"
            
            if filters:
                return self.execute_query(query, tuple(filters.values()), fetchall=True)
            return self.execute_query(query, fetchall=True)
        except Exception as e:
            logger.error(f"Select error: {e}")
            return []
    
    def update(self, table, filters, data):
        """Update data in table"""
        try:
            set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
            conditions = ' AND '.join([f"{k} = ?" for k in filters.keys()])
            query = f"UPDATE {table} SET {set_clause} WHERE {conditions}"
            params = tuple(data.values()) + tuple(filters.values())
            
            cursor = self.execute_query(query, params)
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Update error: {e}")
            return False
    
    def delete(self, table, filters):
        """Delete data from table"""
        try:
            conditions = ' AND '.join([f"{k} = ?" for k in filters.keys()])
            query = f"DELETE FROM {table} WHERE {conditions}"
            cursor = self.execute_query(query, tuple(filters.values()))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

# ============================================
# AI EVENT GENERATOR
# ============================================

class AIEventGenerator:
    """Generate structured event data from unstructured text"""
    
    def __init__(self):
        self.api_key = None
        self.is_configured = False
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            if hasattr(st, 'secrets'):
                if 'OPENAI_API_KEY' in st.secrets:
                    self.api_key = st.secrets["OPENAI_API_KEY"]
                elif 'openai' in st.secrets and 'api_key' in st.secrets.openai:
                    self.api_key = st.secrets.openai.api_key
                
                if self.api_key and self.api_key.startswith("sk-"):
                    import openai
                    openai.api_key = self.api_key
                    self.is_configured = True
                    logger.info("✅ OpenAI configured successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize OpenAI: {str(e)[:100]}")
    
    def extract_event_info(self, text):
        """Extract event information from text using AI or regex fallback"""
        if self.is_configured and self.api_key:
            try:
                return self._extract_with_openai(text)
            except Exception as e:
                logger.warning(f"AI extraction failed: {str(e)[:100]}")
        
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
        - event_link: Event website/URL if mentioned (string or null)
        - registration_link: Registration URL if mentioned (string or null)
        - max_participants: Maximum participants if mentioned (integer or 100)
        
        Text: {text}
        
        Return only valid JSON, no other text.
        """
        
        try:
            import openai
            
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
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'\s*```', '', result_text)
            result_text = re.sub(r'^json\s*', '', result_text, flags=re.IGNORECASE)
            
            event_data = json.loads(result_text)
            
            required_fields = ['title', 'description', 'event_type', 'event_date', 'venue', 'organizer']
            for field in required_fields:
                if field not in event_data:
                    event_data[field] = ""
            
            event_data['ai_generated'] = True
            event_data['ai_prompt'] = text
            event_data['ai_extracted_at'] = datetime.now().isoformat()
            
            return event_data
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)[:100]}")
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
            'event_link': None,
            'registration_link': None,
            'max_participants': 100,
            'ai_generated': False,
            'ai_prompt': text
        }
        
        lines = text.strip().split('\n')
        if lines and lines[0].strip():
            first_line = lines[0].strip()
            if len(first_line) < 100:
                event_data['title'] = first_line
        
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['event_date'] = match.group(1)
                break
        
        venue_keywords = ['at', 'venue', 'location', 'place', 'hall', 'room']
        for keyword in venue_keywords:
            pattern = rf'{keyword}[:\s]*([^.\n,;]{5,50})'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['venue'] = match.group(1).strip()
                break
        
        organizer_keywords = ['by', 'organizer', 'organized by', 'conducted by', 'hosted by']
        for keyword in organizer_keywords:
            pattern = rf'{keyword}[:\s]*([^.\n,;]{5,50})'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['organizer'] = match.group(1).strip()
                break
        
        url_pattern = r'https?://[^\s<>"\'()]+'
        urls = re.findall(url_pattern, text)
        
        if urls:
            event_data['event_link'] = urls[0]
            if len(urls) > 1:
                event_data['registration_link'] = urls[1]
        
        return event_data

# ============================================
# PASSWORD RESET MANAGER
# ============================================

class PasswordResetManager:
    """Manage password reset functionality"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.reset_tokens = {}
    
    def generate_reset_token(self, username: str) -> str:
        """Generate password reset token"""
        token = secrets.token_urlsafe(32)
        expires = datetime.now() + timedelta(hours=1)
        self.reset_tokens[token] = {
            'username': username,
            'expires': expires
        }
        return token
    
    def validate_reset_token(self, token: str) -> Tuple[bool, str]:
        """Validate reset token"""
        if token not in self.reset_tokens:
            return False, "Invalid or expired token"
        
        token_data = self.reset_tokens[token]
        if datetime.now() > token_data['expires']:
            del self.reset_tokens[token]
            return False, "Token has expired"
        
        return True, token_data['username']
    
    def reset_password(self, token: str, new_password: str) -> Tuple[bool, str]:
        """Reset password using token"""
        valid, result = self.validate_reset_token(token)
        if not valid:
            return False, result
        
        username = result
        user = self.db.get_user(username)
        
        if not user:
            return False, "User not found"
        
        is_valid, msg = Validators.validate_password(new_password)
        if not is_valid:
            return False, msg
        
        hashed_pass = hashlib.sha256(new_password.encode()).hexdigest()
        try:
            success = self.db.update_user_password(username, hashed_pass)
            if success:
                del self.reset_tokens[token]
                return True, "Password reset successful"
            return False, "Failed to reset password"
        except Exception as e:
            logger.error(f"Password reset error: {e}")
            return False, "Failed to reset password"

# ============================================
# UNIFIED DATABASE MANAGER
# ============================================

class DatabaseManager:
    """Unified database manager supporting both Supabase and SQLite"""
    
    def __init__(self, use_supabase=True):
        self.use_supabase = use_supabase
        
        if self.use_supabase:
            self.client = SupabaseClient()
            if not self.client.is_configured:
                logger.warning("⚠️ Supabase not configured. Falling back to SQLite.")
                self.use_supabase = False
                self.client = SQLiteClient()
        else:
            self.client = SQLiteClient()
        
        self._initialize_database()
        self._add_default_users()
    
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            if not self.use_supabase:
                self._create_sqlite_tables()
            logger.info(f"✅ Database initialized with {'Supabase' if self.use_supabase else 'SQLite'}")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def _create_sqlite_tables(self):
        """Create SQLite tables"""
        tables_sql = [
            """CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL,
                roll_no TEXT,
                department TEXT,
                year TEXT,
                email TEXT,
                mobile TEXT,
                reset_token TEXT,
                reset_token_expiry TEXT,
                last_activity TEXT,
                is_active INTEGER DEFAULT 1,
                login_attempts INTEGER DEFAULT 0,
                last_login TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                total_points INTEGER DEFAULT 0,
                last_points_update TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS mentors (
                id TEXT PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                full_name TEXT NOT NULL,
                department TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                contact TEXT NOT NULL,
                expertise TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                event_type TEXT,
                event_date TEXT,
                venue TEXT,
                organizer TEXT,
                event_link TEXT,
                registration_link TEXT,
                max_participants INTEGER DEFAULT 100,
                current_participants INTEGER DEFAULT 0,
                flyer_path TEXT,
                created_by TEXT,
                created_by_name TEXT,
                ai_generated INTEGER DEFAULT 0,
                status TEXT DEFAULT 'upcoming',
                mentor_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (mentor_id) REFERENCES mentors(id) ON DELETE SET NULL
            )""",
            """CREATE TABLE IF NOT EXISTS registrations (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                event_title TEXT NOT NULL,
                student_username TEXT NOT NULL,
                student_name TEXT NOT NULL,
                student_roll TEXT,
                student_dept TEXT,
                student_mobile TEXT,
                status TEXT DEFAULT 'pending',
                attendance TEXT DEFAULT 'absent',
                points_awarded INTEGER DEFAULT 0,
                badges_awarded TEXT DEFAULT '',
                mentor_notes TEXT,
                registered_at TEXT DEFAULT CURRENT_TIMESTAMP,
                checked_in_at TEXT,
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                UNIQUE(event_id, student_username)
            )""",
            """CREATE TABLE IF NOT EXISTS event_likes (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                student_username TEXT NOT NULL,
                liked_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                UNIQUE(event_id, student_username)
            )""",
            """CREATE TABLE IF NOT EXISTS event_interested (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                student_username TEXT NOT NULL,
                interested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                UNIQUE(event_id, student_username)
            )""",
            """CREATE TABLE IF NOT EXISTS student_badges (
                id TEXT PRIMARY KEY,
                student_username TEXT NOT NULL,
                badge_name TEXT NOT NULL,
                badge_type TEXT,
                awarded_for TEXT,
                event_id TEXT,
                event_title TEXT,
                awarded_by TEXT,
                awarded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS points_history (
                id TEXT PRIMARY KEY,
                student_username TEXT NOT NULL,
                points INTEGER NOT NULL,
                reason TEXT NOT NULL,
                event_id TEXT,
                event_title TEXT,
                awarded_by TEXT,
                awarded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        ]
        
        for sql in tables_sql:
            self.client.execute_query(sql, commit=True)
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials"""
        try:
            user = self.get_user(username)
            if not user or user.get('role') != role:
                return False
            
            stored_pass = user.get('password', '')
            input_hash = hashlib.sha256(password.encode()).hexdigest().lower()
            
            if stored_pass == input_hash or stored_pass == password:
                self.update_user_activity(username)
                return True
            return False
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    def get_user(self, username):
        """Get user by username"""
        try:
            if self.use_supabase:
                results = self.client.select('users', {'username': username}, limit=1)
                return results[0] if results else None
            else:
                return self.client.execute_query(
                    "SELECT * FROM users WHERE username = ?",
                    (username,), fetchone=True
                )
        except Exception as e:
            logger.error(f"Error getting user {username}: {e}")
            return None
    
    def add_user(self, user_data):
        """Add new user"""
        try:
            password = user_data.get('password', '')
            if not password:
                return False, "Password is required"
            
            hashed_pass = hashlib.sha256(password.encode()).hexdigest().lower()
            
            user_record = {
                'id': user_data.get('id', str(uuid.uuid4())),
                'name': user_data.get('name'),
                'username': user_data.get('username'),
                'password': hashed_pass,
                'role': user_data.get('role', 'student'),
                'roll_no': user_data.get('roll_no', ''),
                'department': user_data.get('department', ''),
                'year': user_data.get('year', ''),
                'email': user_data.get('email', ''),
                'mobile': user_data.get('mobile', ''),
                'total_points': 0
            }
            
            success = self.client.insert('users', user_record)
            if success:
                logger.info(f"✅ User '{user_data.get('username')}' added")
                return True, "User registered successfully"
            return False, "Registration failed"
        except Exception as e:
            logger.error(f"❌ Error adding user: {e}")
            return False, str(e)
    
    def update_user_activity(self, username):
        """Update user's last activity"""
        try:
            update_data = {
                'last_activity': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            return self.client.update('users', {'username': username}, update_data)
        except Exception as e:
            logger.error(f"Error updating user activity: {e}")
            return False
    
    def update_user_password(self, username, hashed_password):
        """Update user's password"""
        try:
            update_data = {
                'password': hashed_password,
                'updated_at': datetime.now().isoformat()
            }
            return self.client.update('users', {'username': username}, update_data)
        except Exception as e:
            logger.error(f"Error updating password: {e}")
            return False
    
    def get_user_by_email(self, email):
        """Get user by email"""
        try:
            if self.use_supabase:
                results = self.client.select('users', {'email': email})
                return results[0] if results else None
            else:
                return self.client.execute_query(
                    "SELECT * FROM users WHERE email = ?",
                    (email,), fetchone=True
                )
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    def set_remember_token(self, username, token, expiry):
        """Set remember me token"""
        try:
            update_data = {
                'reset_token': token,
                'reset_token_expiry': expiry,
                'updated_at': datetime.now().isoformat()
            }
            return self.client.update('users', {'username': username}, update_data)
        except Exception as e:
            logger.error(f"Error setting remember token: {e}")
            return False
    
    def verify_remember_token(self, username, token):
        """Verify remember me token"""
        try:
            user = self.get_user(username)
            if not user or not user.get('reset_token') or not user.get('reset_token_expiry'):
                return False
            
            stored_token = user.get('reset_token')
            expiry_str = user.get('reset_token_expiry')
            
            expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
            return stored_token == token and datetime.now() < expiry
        except Exception as e:
            logger.error(f"Error verifying remember token: {e}")
            return False
    
    def clear_reset_token(self, username):
        """Clear password reset token"""
        try:
            update_data = {
                'reset_token': None,
                'reset_token_expiry': None,
                'updated_at': datetime.now().isoformat()
            }
            return self.client.update('users', {'username': username}, update_data)
        except Exception as e:
            logger.error(f"Error clearing reset token: {e}")
            return False
    
    def add_event(self, event_data):
        """Add new event"""
        try:
            event_record = {
                'id': event_data.get('id', str(uuid.uuid4())),
                'title': event_data.get('title'),
                'description': event_data.get('description'),
                'event_type': event_data.get('event_type'),
                'event_date': event_data.get('event_date'),
                'venue': event_data.get('venue'),
                'organizer': event_data.get('organizer'),
                'event_link': event_data.get('event_link', ''),
                'registration_link': event_data.get('registration_link', ''),
                'max_participants': event_data.get('max_participants', 100),
                'current_participants': event_data.get('current_participants', 0),
                'flyer_path': event_data.get('flyer_path'),
                'created_by': event_data.get('created_by'),
                'created_by_name': event_data.get('created_by_name'),
                'ai_generated': event_data.get('ai_generated', False),
                'status': 'upcoming',
                'mentor_id': event_data.get('mentor_id'),
                'created_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('events', event_record)
            if success:
                logger.info(f"New event created: {event_data.get('title')}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding event: {e}")
            return False
    
    def get_all_events(self):
        """Get all events"""
        try:
            if self.use_supabase:
                events = self.client.select('events', limit=1000, order_by='event_date.desc')
            else:
                events = self.client.execute_query(
                    "SELECT * FROM events ORDER BY event_date DESC",
                    fetchall=True
                )
            return events or []
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    def get_events_by_creator(self, username):
        """Get events created by specific user"""
        try:
            if self.use_supabase:
                events = self.client.select('events', {'created_by': username}, limit=1000, order_by='event_date.desc')
            else:
                events = self.client.execute_query(
                    "SELECT * FROM events WHERE created_by = ? ORDER BY event_date DESC",
                    (username,), fetchall=True
                )
            return events or []
        except Exception as e:
            logger.error(f"Error getting events by creator: {e}")
            return []
    
    def get_event_with_mentor(self, event_id):
        """Get event with mentor details"""
        try:
            if self.use_supabase:
                events = self.client.select('events', {'id': event_id})
                if events:
                    event = events[0]
                    if event.get('mentor_id'):
                        mentors = self.client.select('mentors', {'id': event['mentor_id']})
                        if mentors:
                            mentor = mentors[0]
                            event['mentor_name'] = mentor.get('full_name')
                            event['mentor_contact'] = mentor.get('contact')
                            event['mentor_expertise'] = mentor.get('expertise')
                    return event
                return None
            else:
                return self.client.execute_query(
                    """SELECT e.*, m.full_name as mentor_name, m.contact as mentor_contact, 
                       m.expertise as mentor_expertise 
                       FROM events e LEFT JOIN mentors m ON e.mentor_id = m.id 
                       WHERE e.id = ?""",
                    (event_id,), fetchone=True
                )
        except Exception as e:
            logger.error(f"Error getting event with mentor: {e}")
            return None
    
    def assign_mentor_to_event(self, event_id, mentor_id):
        """Assign mentor to event"""
        try:
            update_data = {
                'mentor_id': mentor_id,
                'updated_at': datetime.now().isoformat()
            }
            return self.client.update('events', {'id': event_id}, update_data)
        except Exception as e:
            logger.error(f"Error assigning mentor: {e}")
            return False
    
    def update_event_status(self):
        """Update event status based on current time"""
        try:
            now = datetime.now().isoformat()
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            today_end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()
            
            if self.use_supabase:
                # Update past events
                past_update = {'status': 'past', 'updated_at': now}
                self.client.update('events', 
                                 {'event_date': {'lt': now}, 'status': {'neq': 'past'}}, 
                                 past_update)
                
                # Update ongoing events
                ongoing_update = {'status': 'ongoing', 'updated_at': now}
                self.client.update('events', 
                                 {'event_date': {'gte': today_start, 'lte': today_end}, 'status': 'upcoming'}, 
                                 ongoing_update)
            else:
                # Update past events
                self.client.execute_query(
                    "UPDATE events SET status = 'past', updated_at = ? WHERE event_date <= ? AND status != 'past'",
                    (now, now), commit=True
                )
                
                # Update ongoing events
                self.client.execute_query(
                    "UPDATE events SET status = 'ongoing', updated_at = ? WHERE event_date BETWEEN ? AND ? AND status = 'upcoming'",
                    (now, today_start, today_end), commit=True
                )
            
            return True
        except Exception as e:
            logger.error(f"Error updating event status: {e}")
            return False
    
    def add_mentor(self, mentor_data):
        """Add new mentor"""
        try:
            full_name = f"{mentor_data.get('first_name')} {mentor_data.get('last_name')}"
            mentor_id = mentor_data.get('id', str(uuid.uuid4()))
            password = mentor_data.get('password', secrets.token_urlsafe(12))
            hashed_pass = hashlib.sha256(password.encode()).hexdigest()
            contact = Validators.format_mobile(mentor_data.get('contact', ''))
            
            # Add to users table
            user_record = {
                'id': mentor_id,
                'name': full_name,
                'username': mentor_data.get('email'),
                'password': hashed_pass,
                'role': 'mentor',
                'email': mentor_data.get('email'),
                'department': mentor_data.get('department'),
                'mobile': contact,
                'created_at': datetime.now().isoformat(),
                'total_points': 0
            }
            
            # Add to mentors table
            mentor_record = {
                'id': mentor_id,
                'first_name': mentor_data.get('first_name'),
                'last_name': mentor_data.get('last_name'),
                'full_name': full_name,
                'department': mentor_data.get('department'),
                'email': mentor_data.get('email'),
                'contact': contact,
                'expertise': mentor_data.get('expertise', ''),
                'is_active': mentor_data.get('is_active', True),
                'created_at': datetime.now().isoformat(),
                'created_by': mentor_data.get('created_by')
            }
            
            user_success = self.client.insert('users', user_record)
            mentor_success = self.client.insert('mentors', mentor_record)
            
            if user_success and mentor_success:
                logger.info(f"Added new mentor: {full_name}")
                return True, password
            return False, "Failed to add mentor"
        except Exception as e:
            logger.error(f"Error adding mentor: {e}")
            return False, str(e)
    
    def get_all_mentors(self):
        """Get all mentors"""
        try:
            if self.use_supabase:
                mentors = self.client.select('mentors', limit=1000, order_by='full_name')
            else:
                mentors = self.client.execute_query(
                    "SELECT * FROM mentors ORDER BY full_name",
                    fetchall=True
                )
            return mentors or []
        except Exception as e:
            logger.error(f"Error getting mentors: {e}")
            return []
    
    def get_active_mentors(self):
        """Get active mentors only"""
        try:
            if self.use_supabase:
                mentors = self.client.select('mentors', {'is_active': True}, limit=1000, order_by='full_name')
            else:
                mentors = self.client.execute_query(
                    "SELECT * FROM mentors WHERE is_active = 1 ORDER BY full_name",
                    fetchall=True
                )
            return mentors or []
        except Exception as e:
            logger.error(f"Error getting active mentors: {e}")
            return []
    
    def get_mentor_by_id(self, mentor_id):
        """Get mentor by ID"""
        try:
            if self.use_supabase:
                results = self.client.select('mentors', {'id': mentor_id})
                return results[0] if results else None
            else:
                return self.client.execute_query(
                    "SELECT * FROM mentors WHERE id = ?",
                    (mentor_id,), fetchone=True
                )
        except Exception as e:
            logger.error(f"Error getting mentor by ID: {e}")
            return None
    
    def get_mentor_by_email(self, email):
        """Get mentor by email"""
        try:
            if self.use_supabase:
                results = self.client.select('mentors', {'email': email})
                return results[0] if results else None
            else:
                return self.client.execute_query(
                    "SELECT * FROM mentors WHERE email = ?",
                    (email,), fetchone=True
                )
        except Exception as e:
            logger.error(f"Error getting mentor by email: {e}")
            return None
    
    def get_events_by_mentor(self, mentor_id):
        """Get events assigned to a mentor"""
        try:
            if self.use_supabase:
                events = self.client.select('events', {'mentor_id': mentor_id}, limit=1000, order_by='event_date.desc')
            else:
                events = self.client.execute_query(
                    "SELECT * FROM events WHERE mentor_id = ? ORDER BY event_date DESC",
                    (mentor_id,), fetchall=True
                )
            return events or []
        except Exception as e:
            logger.error(f"Error getting events by mentor: {e}")
            return []
    
    def update_mentor(self, mentor_id, mentor_data):
        """Update mentor information"""
        try:
            return self.client.update('mentors', {'id': mentor_id}, mentor_data)
        except Exception as e:
            logger.error(f"Error updating mentor: {e}")
            return False
    
    def add_registration(self, reg_data):
        """Add new registration"""
        try:
            # Check if already registered
            if self.use_supabase:
                existing = self.client.select('registrations', {
                    'event_id': reg_data['event_id'],
                    'student_username': reg_data['student_username']
                })
                if existing:
                    return None, "Already registered"
            else:
                existing = self.client.execute_query(
                    "SELECT id FROM registrations WHERE event_id = ? AND student_username = ?",
                    (reg_data['event_id'], reg_data['student_username']), fetchone=True
                )
                if existing:
                    return None, "Already registered"
            
            # Get student info
            student = self.get_user(reg_data['student_username'])
            if not student:
                return None, "Student not found"
            
            registration_record = {
                'id': reg_data.get('id', str(uuid.uuid4())),
                'event_id': reg_data.get('event_id'),
                'event_title': reg_data.get('event_title'),
                'student_username': reg_data.get('student_username'),
                'student_name': student.get('name', reg_data.get('student_username')),
                'student_roll': student.get('roll_no', 'N/A'),
                'student_dept': student.get('department', 'N/A'),
                'student_mobile': student.get('mobile', ''),
                'registered_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('registrations', registration_record)
            if success:
                self._update_event_participant_count(reg_data['event_id'])
                logger.info(f"New registration: {reg_data['student_username']}")
                return registration_record['id'], "Registration successful"
            return None, "Registration failed"
        except Exception as e:
            logger.error(f"Error adding registration: {e}")
            return None, "Registration failed"
    
    def _update_event_participant_count(self, event_id):
        """Update event participant count"""
        try:
            if self.use_supabase:
                registrations = self.client.select('registrations', {'event_id': event_id})
                count = len(registrations) if registrations else 0
                update_data = {'current_participants': count, 'updated_at': datetime.now().isoformat()}
                return self.client.update('events', {'id': event_id}, update_data)
            else:
                cursor = self.client.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM registrations WHERE event_id = ?", (event_id,))
                count = cursor.fetchone()[0]
                cursor.execute("UPDATE events SET current_participants = ?, updated_at = ? WHERE id = ?", 
                             (count, datetime.now().isoformat(), event_id))
                self.client.conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating participant count: {e}")
            return False
    
    def get_registrations_by_student(self, username):
        """Get all registrations for a student"""
        try:
            if self.use_supabase:
                registrations = self.client.select('registrations', {'student_username': username})
            else:
                registrations = self.client.execute_query(
                    "SELECT r.*, e.event_date, e.venue, e.status as event_status FROM registrations r LEFT JOIN events e ON r.event_id = e.id WHERE r.student_username = ? ORDER BY r.registered_at DESC",
                    (username,), fetchall=True
                )
            return registrations or []
        except Exception as e:
            logger.error(f"Error getting registrations by student: {e}")
            return []
    
    def get_registrations_by_event(self, event_id):
        """Get all registrations for an event"""
        try:
            if self.use_supabase:
                return self.client.select('registrations', {'event_id': event_id})
            else:
                return self.client.execute_query(
                    "SELECT r.*, u.department, u.year, u.mobile FROM registrations r LEFT JOIN users u ON r.student_username = u.username WHERE r.event_id = ? ORDER BY r.registered_at DESC",
                    (event_id,), fetchall=True
                )
        except Exception as e:
            logger.error(f"Error getting registrations by event: {e}")
            return []
    
    def is_student_registered(self, event_id, username):
        """Check if student is registered for event"""
        try:
            if self.use_supabase:
                results = self.client.select('registrations', {
                    'event_id': event_id,
                    'student_username': username
                })
                return bool(results)
            else:
                result = self.client.execute_query(
                    "SELECT id FROM registrations WHERE event_id = ? AND student_username = ?",
                    (event_id, username), fetchone=True
                )
                return result is not None
        except Exception as e:
            logger.error(f"Error checking registration: {e}")
            return False
    
    def update_registration_status(self, registration_id, status, points=None, badge=None, mentor_notes=None):
        """Update registration status and award points/badges"""
        try:
            # Get registration details
            if self.use_supabase:
                reg_results = self.client.select('registrations', {'id': registration_id})
                if not reg_results:
                    return False, "Registration not found"
                registration = reg_results[0]
            else:
                registration = self.client.execute_query(
                    "SELECT * FROM registrations WHERE id = ?",
                    (registration_id,), fetchone=True
                )
                if not registration:
                    return False, "Registration not found"
            
            student_username = registration['student_username']
            event_id = registration['event_id']
            event_title = registration['event_title']
            
            update_data = {'status': status}
            
            # Award points if specified
            if points is not None and points > 0:
                update_data['points_awarded'] = points
                
                # Update student's total points
                student = self.get_user(student_username)
                if student:
                    current_points = student.get('total_points', 0)
                    new_points = current_points + points
                    
                    student_update_data = {
                        'total_points': new_points,
                        'last_points_update': datetime.now().isoformat()
                    }
                    self.client.update('users', {'username': student_username}, student_update_data)
                    
                    # Record points history
                    points_record = {
                        'id': str(uuid.uuid4()),
                        'student_username': student_username,
                        'points': points,
                        'reason': f"Awarded for {status.lower()} in {event_title}",
                        'event_id': event_id,
                        'event_title': event_title,
                        'awarded_by': st.session_state.username if 'username' in st.session_state else 'system',
                        'awarded_at': datetime.now().isoformat()
                    }
                    self.client.insert('points_history', points_record)
            
            # Award badge if specified
            if badge:
                current_badges = registration.get('badges_awarded', '')
                badges_list = current_badges.split(',') if current_badges else []
                if badge not in badges_list:
                    badges_list.append(badge)
                    update_data['badges_awarded'] = ','.join(badges_list)
                    
                    # Add to student badges table
                    badge_record = {
                        'id': str(uuid.uuid4()),
                        'student_username': student_username,
                        'badge_name': badge,
                        'badge_type': status.lower(),
                        'awarded_for': f"{status} in {event_title}",
                        'event_id': event_id,
                        'event_title': event_title,
                        'awarded_by': st.session_state.username if 'username' in st.session_state else 'system',
                        'awarded_at': datetime.now().isoformat()
                    }
                    self.client.insert('student_badges', badge_record)
            
            if mentor_notes:
                update_data['mentor_notes'] = mentor_notes
            
            # Update registration
            success = self.client.update('registrations', {'id': registration_id}, update_data)
            if success:
                return True, "Registration updated successfully"
            return False, "Failed to update registration"
        except Exception as e:
            logger.error(f"Error updating registration status: {e}")
            return False, str(e)
    
    def award_points_and_badge(self, student_username, event_id, event_title, achievement_type, awarded_by=None):
        """Award points and badge to student"""
        try:
            points = GAMIFICATION_CONFIG['points'].get(achievement_type, 0)
            badge = GAMIFICATION_CONFIG['badges'].get(achievement_type, '')
            
            if points <= 0:
                return False, "No points to award"
            
            # Update student's total points
            student = self.get_user(student_username)
            if not student:
                return False, "Student not found"
            
            current_points = student.get('total_points', 0)
            new_points = current_points + points
            
            update_data = {
                'total_points': new_points,
                'last_points_update': datetime.now().isoformat()
            }
            
            success = self.client.update('users', {'username': student_username}, update_data)
            if not success:
                return False, "Failed to update points"
            
            # Record points history
            points_record = {
                'id': str(uuid.uuid4()),
                'student_username': student_username,
                'points': points,
                'reason': f"Awarded for {achievement_type} in {event_title}",
                'event_id': event_id,
                'event_title': event_title,
                'awarded_by': awarded_by or 'system',
                'awarded_at': datetime.now().isoformat()
            }
            self.client.insert('points_history', points_record)
            
            # Award badge if available
            if badge:
                badge_record = {
                    'id': str(uuid.uuid4()),
                    'student_username': student_username,
                    'badge_name': badge,
                    'badge_type': achievement_type,
                    'awarded_for': f"{achievement_type} in {event_title}",
                    'event_id': event_id,
                    'event_title': event_title,
                    'awarded_by': awarded_by or 'system',
                    'awarded_at': datetime.now().isoformat()
                }
                self.client.insert('student_badges', badge_record)
            
            return True, f"Awarded {points} points and badge: {badge}"
        except Exception as e:
            logger.error(f"Error awarding points: {e}")
            return False, str(e)
    
    def get_student_points(self, username):
        """Get student's total points"""
        try:
            user = self.get_user(username)
            return user.get('total_points', 0) if user else 0
        except Exception as e:
            logger.error(f"Error getting student points: {e}")
            return 0
    
    def get_student_badges(self, username):
        """Get all badges earned by student"""
        try:
            if self.use_supabase:
                badges = self.client.select('student_badges', {'student_username': username})
            else:
                badges = self.client.execute_query(
                    "SELECT * FROM student_badges WHERE student_username = ? ORDER BY awarded_at DESC",
                    (username,), fetchall=True
                )
            return badges or []
        except Exception as e:
            logger.error(f"Error getting student badges: {e}")
            return []
    
    def get_points_history(self, username):
        """Get points history for student"""
        try:
            if self.use_supabase:
                history = self.client.select('points_history', {'student_username': username})
            else:
                history = self.client.execute_query(
                    "SELECT * FROM points_history WHERE student_username = ? ORDER BY awarded_at DESC",
                    (username,), fetchall=True
                )
            return history or []
        except Exception as e:
            logger.error(f"Error getting points history: {e}")
            return []
    
    def get_leaderboard(self, limit=15, department=None):
        """Get leaderboard of top students"""
        try:
            if self.use_supabase:
                # Get all students
                filters = {'role': 'student'}
                if department:
                    filters['department'] = department
                
                users = self.client.select('users', filters, limit=1000)
                if not users:
                    return []
                
                # Sort by points
                users.sort(key=lambda x: x.get('total_points', 0), reverse=True)
                
                # Add rank
                for i, user in enumerate(users[:limit], 1):
                    user['rank'] = i
                
                return users[:limit]
            else:
                query = """
                SELECT name, username, roll_no, department, year, total_points, 
                       (SELECT COUNT(*) FROM points_history WHERE student_username = users.username) as events_count,
                       (SELECT COUNT(*) FROM student_badges WHERE student_username = users.username) as badges_count
                FROM users 
                WHERE role = 'student'
                """
                
                params = []
                if department:
                    query += " AND department = ?"
                    params.append(department)
                
                query += " ORDER BY total_points DESC, name ASC LIMIT ?"
                params.append(limit)
                
                results = self.client.execute_query(query, tuple(params), fetchall=True)
                
                for i, result in enumerate(results, 1):
                    result['rank'] = i
                
                return results
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []
    
    def get_student_rank(self, username):
        """Get student's rank in leaderboard"""
        try:
            leaderboard = self.get_leaderboard(limit=1000)
            for i, student in enumerate(leaderboard, 1):
                if student['username'] == username:
                    return i
            return None
        except Exception as e:
            logger.error(f"Error getting student rank: {e}")
            return None
    
    def add_like(self, event_id, student_username):
        """Add a like for an event"""
        try:
            like_record = {
                'id': str(uuid.uuid4()),
                'event_id': event_id,
                'student_username': student_username,
                'liked_at': datetime.now().isoformat()
            }
            return self.client.insert('event_likes', like_record)
        except Exception as e:
            logger.error(f"Error adding like: {e}")
            return False
    
    def remove_like(self, event_id, student_username):
        """Remove a like for an event"""
        try:
            return self.client.delete('event_likes', {
                'event_id': event_id,
                'student_username': student_username
            })
        except Exception as e:
            logger.error(f"Error removing like: {e}")
            return False
    
    def is_event_liked(self, event_id, student_username):
        """Check if student liked an event"""
        try:
            if self.use_supabase:
                results = self.client.select('event_likes', {
                    'event_id': event_id,
                    'student_username': student_username
                })
                return bool(results)
            else:
                result = self.client.execute_query(
                    "SELECT id FROM event_likes WHERE event_id = ? AND student_username = ?",
                    (event_id, student_username), fetchone=True
                )
                return result is not None
        except Exception as e:
            logger.error(f"Error checking like: {e}")
            return False
    
    def get_event_likes_count(self, event_id):
        """Get total likes for an event"""
        try:
            if self.use_supabase:
                likes = self.client.select('event_likes', {'event_id': event_id})
                return len(likes) if likes else 0
            else:
                result = self.client.execute_query(
                    "SELECT COUNT(*) as count FROM event_likes WHERE event_id = ?",
                    (event_id,), fetchone=True
                )
                return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting likes count: {e}")
            return 0
    
    def add_interested(self, event_id, student_username):
        """Add interested for an event"""
        try:
            interested_record = {
                'id': str(uuid.uuid4()),
                'event_id': event_id,
                'student_username': student_username,
                'interested_at': datetime.now().isoformat()
            }
            return self.client.insert('event_interested', interested_record)
        except Exception as e:
            logger.error(f"Error adding interested: {e}")
            return False
    
    def remove_interested(self, event_id, student_username):
        """Remove interested for an event"""
        try:
            return self.client.delete('event_interested', {
                'event_id': event_id,
                'student_username': student_username
            })
        except Exception as e:
            logger.error(f"Error removing interested: {e}")
            return False
    
    def is_event_interested(self, event_id, student_username):
        """Check if student is interested in an event"""
        try:
            if self.use_supabase:
                results = self.client.select('event_interested', {
                    'event_id': event_id,
                    'student_username': student_username
                })
                return bool(results)
            else:
                result = self.client.execute_query(
                    "SELECT id FROM event_interested WHERE event_id = ? AND student_username = ?",
                    (event_id, student_username), fetchone=True
                )
                return result is not None
        except Exception as e:
            logger.error(f"Error checking interested: {e}")
            return False
    
    def get_event_interested_count(self, event_id):
        """Get total interested count for an event"""
        try:
            if self.use_supabase:
                interested = self.client.select('event_interested', {'event_id': event_id})
                return len(interested) if interested else 0
            else:
                result = self.client.execute_query(
                    "SELECT COUNT(*) as count FROM event_interested WHERE event_id = ?",
                    (event_id,), fetchone=True
                )
                return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting interested count: {e}")
            return 0
    
    def get_student_liked_events(self, student_username):
        """Get all events liked by a student"""
        try:
            if self.use_supabase:
                likes = self.client.select('event_likes', {'student_username': student_username})
                if not likes:
                    return []
                
                event_ids = [like['event_id'] for like in likes]
                liked_events = []
                for event_id in event_ids:
                    events = self.client.select('events', {'id': event_id})
                    if events:
                        liked_events.append(events[0])
                
                return liked_events
            else:
                return self.client.execute_query(
                    "SELECT e.* FROM events e JOIN event_likes l ON e.id = l.event_id WHERE l.student_username = ? ORDER BY l.liked_at DESC",
                    (student_username,), fetchall=True
                )
        except Exception as e:
            logger.error(f"Error getting liked events: {e}")
            return []
    
    def get_student_interested_events(self, student_username):
        """Get all events student is interested in"""
        try:
            if self.use_supabase:
                interests = self.client.select('event_interested', {'student_username': student_username})
                if not interests:
                    return []
                
                event_ids = [interest['event_id'] for interest in interests]
                interested_events = []
                for event_id in event_ids:
                    events = self.client.select('events', {'id': event_id})
                    if events:
                        interested_events.append(events[0])
                
                return interested_events
            else:
                return self.client.execute_query(
                    "SELECT e.* FROM events e JOIN event_interested i ON e.id = i.event_id WHERE i.student_username = ? ORDER BY i.interested_at DESC",
                    (student_username,), fetchall=True
                )
        except Exception as e:
            logger.error(f"Error getting interested events: {e}")
            return []
    
    def _add_default_users(self):
        """Add default admin and faculty users"""
        try:
            default_users = [
                {
                    'id': '00000000-0000-0000-0000-000000000001',
                    'username': 'admin@raisoni',
                    'password': 'Admin@12345',
                    'name': 'Administrator',
                    'role': 'admin',
                    'email': 'admin@ghraisoni.edu',
                    'department': 'Administration'
                },
                {
                    'id': '00000000-0000-0000-0000-000000000002',
                    'username': 'faculty@raisoni',
                    'password': 'Faculty@12345',
                    'name': 'Faculty Coordinator',
                    'role': 'faculty',
                    'email': 'faculty@ghraisoni.edu',
                    'department': 'Faculty'
                }
            ]
            
            for user_data in default_users:
                existing = self.get_user(user_data['username'])
                if not existing:
                    success, message = self.add_user(user_data)
                    if success:
                        logger.info(f"✅ Added default user: {user_data['username']}")
            
            self._add_default_students()
            return True
        except Exception as e:
            logger.error(f"Error adding default users: {e}")
            return False
    
    def _add_default_students(self):
        """Add default student accounts"""
        try:
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
                }
            ]
            
            for student in default_students:
                existing = self.get_user(student['username'])
                if not existing:
                    user_data = {
                        'name': student['name'],
                        'username': student['username'],
                        'password': student['password'],
                        'roll_no': student['roll_no'],
                        'department': student['department'],
                        'year': student['year'],
                        'email': student['email'],
                        'mobile': student['mobile'],
                        'role': 'student'
                    }
                    success, message = self.add_user(user_data)
                    if success:
                        logger.info(f"✅ Added default student: {student['name']}")
            return True
        except Exception as e:
            logger.error(f"Error adding default students: {e}")
            return False
    
    def get_system_stats(self):
        """Get system statistics"""
        try:
            stats = {}
            
            # Get users
            users = self.client.select('users') if self.use_supabase else self.client.execute_query("SELECT * FROM users", fetchall=True)
            if users:
                role_counts = {}
                total_points = 0
                for user in users:
                    role = user.get('role', 'unknown')
                    role_counts[role] = role_counts.get(role, 0) + 1
                    total_points += user.get('total_points', 0)
                stats['user_counts'] = role_counts
                stats['total_points'] = total_points
            
            # Get events
            events = self.client.select('events') if self.use_supabase else self.client.execute_query("SELECT * FROM events", fetchall=True)
            if events:
                status_counts = {}
                for event in events:
                    status = event.get('status', 'upcoming')
                    status_counts[status] = status_counts.get(status, 0) + 1
                stats['event_counts'] = status_counts
            
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

# ============================================
# DATABASE INITIALIZATION
# ============================================

db = DatabaseManager(use_supabase=USE_SUPABASE)
password_reset_manager = PasswordResetManager(db)

# ============================================
# HELPER FUNCTIONS
# ============================================

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
        if len(image_bytes) > 5 * 1024 * 1024:
            return None
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        
        mime_type = mime_types.get(file_ext, 'image/jpeg')
        return f"data:{mime_type};base64,{image_base64}"
    except Exception as e:
        logger.error(f"Error processing flyer image: {e}")
        return None

def display_event_card(event, current_user=None):
    """Display improved event card"""
    if not event or not event.get('id'):
        return
    
    event_id = event.get('id')
    
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        col_img, col_info = st.columns([1, 3], gap="medium")
        
        with col_img:
            flyer = event.get('flyer_path')
            if flyer and flyer.startswith('data:image'):
                try:
                    st.image(flyer, use_column_width=True)
                except:
                    st.markdown('<div style="width: 100%; height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; border-radius: 8px;"><span style="font-size: 32px; color: white;">🎯</span></div>', 
                               unsafe_allow_html=True)
            else:
                st.markdown('<div style="width: 100%; height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; border-radius: 8px;"><span style="font-size: 32px; color: white;">🎯</span></div>', 
                           unsafe_allow_html=True)
        
        with col_info:
            title = event.get('title', 'Untitled Event')
            if len(title) > 60:
                title = title[:57] + "..."
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
            
            event_date = event.get('event_date')
            status_html = get_event_status(event_date)
            formatted_date = format_date(event_date)
            
            st.markdown(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>{status_html}</div>
                <div style="color: #666; font-size: 0.9rem;">📅 {formatted_date}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            venue = event.get('venue', 'TBD')
            if len(venue) > 25:
                venue = venue[:22] + "..."
            
            event_type = event.get('event_type', 'Event')
            max_participants = event.get('max_participants', 100)
            current_participants = event.get('current_participants', 0)
            
            st.caption(f"📍 {venue} | 🏷️ {event_type} | 👥 {current_participants}/{max_participants}")
            
            if event.get('mentor_id'):
                mentor = db.get_mentor_by_id(event['mentor_id'])
                if mentor:
                    st.markdown('<div style="background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%); padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid #E5E7EB; border-left: 3px solid #8B5CF6; font-size: 0.9rem;">', unsafe_allow_html=True)
                    st.markdown(f"**Mentor:** {mentor['full_name']} | **Contact:** {mentor['contact']}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            likes_count = db.get_event_likes_count(event_id)
            interested_count = db.get_event_interested_count(event_id)
            
            if current_user:
                button_col1, button_col2, button_col3 = st.columns(3)
                
                with button_col1:
                    is_liked = db.is_event_liked(event_id, current_user)
                    like_text = "❤️ Liked" if is_liked else "🤍 Like"
                    like_type = "secondary" if is_liked else "primary"
                    
                    unique_key = f"like_{event_id}_{int(time.time() * 1000) % 10000}"
                    if st.button(like_text, key=unique_key, use_container_width=True, type=like_type):
                        if is_liked:
                            if db.remove_like(event_id, current_user):
                                st.rerun()
                        else:
                            if db.add_like(event_id, current_user):
                                st.rerun()
                
                with button_col2:
                    is_interested = db.is_event_interested(event_id, current_user)
                    interested_text = "⭐ Interested" if is_interested else "☆ Interested"
                    interested_type = "secondary" if is_interested else "primary"
                    
                    unique_key_interested = f"interested_{event_id}_{int(time.time() * 1000) % 10000}"
                    if st.button(interested_text, key=unique_key_interested, use_container_width=True, type=interested_type):
                        if is_interested:
                            if db.remove_interested(event_id, current_user):
                                st.rerun()
                        else:
                            if db.add_interested(event_id, current_user):
                                st.rerun()
                
                with button_col3:
                    unique_key_share = f"share_{event_id}_{int(time.time() * 1000) % 10000}"
                    if st.button("📤 Share", key=unique_key_share, use_container_width=True, type="secondary"):
                        event_title = event.get('title', 'Cool Event')
                        share_text = f"Check out '{event_title}' at G H Raisoni College Event Manager! 🎓\n\nJoin the platform to discover more events: [Event Manager App]"
                        st.code(share_text)
                        st.success("📋 Share message copied! Share with your friends.")
            
            st.caption(f"❤️ {likes_count} Likes | ⭐ {interested_count} Interested")
            
            event_link = event.get('event_link', '')
            registration_link = event.get('registration_link', '')
            
            if event_link or registration_link:
                with st.expander("🔗 Event Links", expanded=False):
                    if event_link:
                        st.markdown(f"**🌐 Event Page:** [Click here]({event_link})")
                    if registration_link:
                        st.markdown(f"**📝 Registration:** [Click here]({registration_link})")
            
            desc = event.get('description', '')
            if desc:
                if len(desc) > 150:
                    with st.expander("📝 Description", expanded=False):
                        st.write(desc)
                else:
                    st.caption(desc[:150] + "..." if len(desc) > 150 else desc)
        
        if current_user and st.session_state.get('role') == 'student':
            st.markdown('<div style="background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); padding: 8px; border-radius: 6px; margin-top: 8px; border-left: 3px solid #3B82F6; font-size: 0.9rem;">', unsafe_allow_html=True)
            
            is_registered = db.is_student_registered(event_id, current_user)
            
            if is_registered:
                st.success("✅ You are already registered for this event")
                
                if registration_link:
                    unique_key_ext_reg = f"ext_reg_{event_id}_{int(time.time() * 1000) % 10000}"
                    if st.button("✅ I Have Registered Externally", key=unique_key_ext_reg, use_container_width=True, type="secondary"):
                        if db.use_supabase:
                            success = db.client.update('registrations', 
                                                     {'event_id': event_id, 'student_username': current_user},
                                                     {'status': 'confirmed', 'updated_at': datetime.now().isoformat()})
                        else:
                            cursor = db.client.conn.cursor()
                            cursor.execute("UPDATE registrations SET status = 'confirmed', updated_at = ? WHERE event_id = ? AND student_username = ?",
                                         (datetime.now().isoformat(), event_id, current_user))
                            db.client.conn.commit()
                            success = cursor.rowcount > 0
                        
                        if success:
                            st.success("✅ External registration recorded!")
                            st.rerun()
            else:
                reg_col1, reg_col2 = st.columns([1, 1])
                
                with reg_col1:
                    unique_key_app_reg = f"app_reg_{event_id}_{int(time.time() * 1000) % 10000}"
                    if st.button("📱 Register in App", key=unique_key_app_reg, use_container_width=True, type="primary"):
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
                            reg_id, message = db.add_registration(reg_data)
                            if reg_id:
                                st.success("✅ Registered in college system!")
                                st.rerun()
                            else:
                                st.error(message)
                
                with reg_col2:
                    if registration_link:
                        st.markdown(f"[🌐 Register Externally]({registration_link})")
                        st.caption("Click to register on external site")
                    else:
                        st.info("No external registration link available")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        created_by = event.get('created_by_name', 'Unknown')
        st.caption(f"👤 Created by: {created_by}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PAGE FUNCTIONS
# ============================================

def forgot_password_page():
    """Password reset page"""
    st.markdown('<div class="college-header"><h2>🔐 Password Recovery</h2></div>', 
                unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Request Reset", "Reset Password"])
    
    with tab1:
        st.markdown("### Request Password Reset")
        reset_email = st.text_input("Email Address", placeholder="your.email@ghraisoni.edu")
        
        if st.button("Send Reset Link", use_container_width=True, type="primary"):
            if reset_email:
                user = db.get_user_by_email(reset_email)
                if user:
                    token = password_reset_manager.generate_reset_token(user['username'])
                    expiry = datetime.now() + timedelta(hours=1)
                    
                    if db.set_remember_token(user['username'], token, expiry.isoformat()):
                        st.success(f"✅ Reset link sent to {reset_email}")
                        st.info(f"**Test Token (for development):** `{token}`")
    
    with tab2:
        st.markdown("### Reset Your Password")
        
        reset_token = st.text_input("Reset Token", placeholder="Paste the token from your email")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        if st.button("Reset Password", use_container_width=True, type="primary"):
            if not all([reset_token, new_password, confirm_password]):
                st.error("Please fill all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                is_valid, msg = Validators.validate_password(new_password)
                if not is_valid:
                    st.error(msg)
                else:
                    success, result = password_reset_manager.reset_password(reset_token, new_password)
                    if success:
                        st.success("✅ Password reset successful!")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"Reset failed: {result}")
    
    st.markdown("---")
    if st.button("← Back to Login", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()

def landing_page():
    """Landing page with app info and login"""
    st.markdown(f'<div class="college-header"><h2>{COLLEGE_CONFIG["name"]}</h2><p>Advanced Event Management System with Gamification</p></div>', 
                unsafe_allow_html=True)

    with st.expander("📱 About This App", expanded=True):
        st.markdown("""
        ### Welcome to G H Raisoni Event Management System
        
        **New Gamification Features:**
        - 🏆 **Leaderboard:** Compete with other students for top rankings
        - 🎮 **Points System:** Earn points for participation and achievements
        - 🏅 **Badges:** Collect badges for various accomplishments
        - 📊 **Progress Tracking:** Monitor your growth and achievements
        
        **Points System:**
        - **50 Points** 🏅 - Register for an event
        - **100 Points** ⭐ - Get shortlisted in an event
        - **200 Points** 🏆 - Win an event
        
        **User Roles:**
        - **👑 Admin:** Full system control, manage users and events
        - **👨‍🏫 Faculty:** Create and manage events, track registrations
        - **👨‍🏫 Mentor:** Monitor assigned events and student engagement
        - **👨‍🎓 Student:** Browse events, register, and track participation
        """)
    
    st.markdown("---")
    st.subheader("🔐 Login to Your Account")
    
    role = st.selectbox(
        "Select Your Role",
        ["Select Role", "Admin", "Faculty", "Mentor", "Student"],
        key="login_role"
    )
    
    if role != "Select Role":
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username", key="login_username")
        
        with col2:
            password = st.text_input("Password", type="password", key="login_password")
        
        remember_me = st.checkbox("Remember Me", help="Stay logged in on this device for 30 days")
        
        col_forgot = st.columns([2, 1])[1]
        with col_forgot:
            if st.button("Forgot Password?", use_container_width=True):
                st.session_state.page = "forgot_password"
                st.rerun()
        
        if st.button("Login", use_container_width=True, type="primary"):
            if not username or not password:
                st.error("Please enter username and password")
            else:
                role_map = {
                    "Admin": "admin",
                    "Faculty": "faculty",
                    "Mentor": "mentor",
                    "Student": "student"
                }
                
                db_role = role_map[role]
                
                if db.verify_credentials(username, password, db_role):
                    user = db.get_user(username)
                    if user:
                        st.session_state.role = db_role
                        st.session_state.username = username
                        st.session_state.name = user.get('name', username)
                        st.session_state.session_start = datetime.now()
                        st.session_state.remember_me = remember_me
                        
                        if remember_me:
                            token = secrets.token_urlsafe(32)
                            expiry = datetime.now() + timedelta(days=30)
                            if db.set_remember_token(username, token, expiry.isoformat()):
                                if hasattr(st, 'query_params'):
                                    st.query_params.clear()
                                    st.query_params["remember_token"] = token
                                    st.query_params["remember_user"] = username
                        
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("User not found in database")
                else:
                    st.error("Invalid credentials")
        
        if role == "Student":
            st.markdown("---")
            st.subheader("👨‍🎓 New Student Registration")
            
            if st.button("Create New Student Account", use_container_width=True, type="secondary"):
                st.session_state.page = "student_register"
                st.rerun()

def student_registration_page():
    """Student registration page with auto-login"""
    st.markdown('<div class="college-header"><h2>👨‍🎓 Student Registration</h2></div>', 
                unsafe_allow_html=True)
    
    with st.form("student_registration"):
        st.markdown("### Create Your Student Account")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *")
            roll_no = st.text_input("Roll Number *")
            department = st.selectbox("Department *", COLLEGE_CONFIG['departments'])
            year = st.selectbox("Year *", COLLEGE_CONFIG['academic_years'])
        
        with col2:
            email = st.text_input("Email *")
            mobile = st.text_input("Mobile Number *", placeholder="9876543210")
            username = st.text_input("Username *")
            password = st.text_input("Password *", type="password")
            confirm_pass = st.text_input("Confirm Password *", type="password")
        
        terms = st.checkbox("I agree to the Terms & Conditions *", value=False)
        remember_me = st.checkbox("Remember Me on this device", value=True)
        
        col_submit, col_back = st.columns(2)
        with col_submit:
            submit = st.form_submit_button("Register", use_container_width=True, type="primary")
        with col_back:
            back = st.form_submit_button("← Back to Login", use_container_width=True, type="secondary")
        
        if back:
            st.session_state.page = "login"
            st.rerun()
        
        if submit:
            errors = []
            
            if password != confirm_pass:
                errors.append("Passwords don't match")
            
            if not all([name, roll_no, email, mobile, username, password]):
                errors.append("Please fill all required fields (*)")
            
            if not terms:
                errors.append("You must agree to the Terms & Conditions")
            
            is_valid_mobile, mobile_msg = Validators.validate_mobile(mobile)
            if not is_valid_mobile:
                errors.append(mobile_msg)
            
            is_valid_email, email_msg = Validators.validate_email(email)
            if not is_valid_email:
                errors.append(email_msg)
            
            is_valid_pass, pass_msg = Validators.validate_password(password)
            if not is_valid_pass:
                errors.append(pass_msg)
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
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
                        st.success("✅ Registration successful!")
                        
                        st.session_state.role = 'student'
                        st.session_state.username = username
                        st.session_state.name = name
                        st.session_state.session_start = datetime.now()
                        st.session_state.remember_me = remember_me
                        
                        if remember_me:
                            token = secrets.token_urlsafe(32)
                            expiry = datetime.now() + timedelta(days=30)
                            if db.set_remember_token(username, token, expiry.isoformat()):
                                if hasattr(st, 'query_params'):
                                    st.query_params.clear()
                                    st.query_params["remember_token"] = token
                                    st.query_params["remember_user"] = username
                        
                        st.balloons()
                        st.info("You have been automatically logged in. Redirecting to dashboard...")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {message}")

def student_dashboard():
    """Student dashboard with leaderboard"""
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    
    student = db.get_user(st.session_state.username)
    if student:
        st.sidebar.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
        st.sidebar.markdown(f"**Department:** {student.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Year:** {student.get('year', 'N/A')}")
        points = db.get_student_points(st.session_state.username)
        st.sidebar.markdown(f"**🏆 Points:** {points}")
        
        rank = db.get_student_rank(st.session_state.username)
        if rank:
            st.sidebar.markdown(f"**📊 Rank:** #{rank}")
    
    display_role_badge('student')
    
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Events Feed", "My Registrations", "Liked Events", "Interested Events", "Leaderboard", "My Profile", "My Achievements"]
        
        if 'student_page' not in st.session_state:
            st.session_state.student_page = "Events Feed"
        
        for option in nav_options:
            if st.button(option, key=f"student_{option}", use_container_width=True):
                st.session_state.student_page = option
                st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            if st.session_state.username:
                db.clear_reset_token(st.session_state.username)
            
            if hasattr(st, 'query_params'):
                st.query_params.clear()
            
            for key in list(st.session_state.keys()):
                if key != 'rerun_count':
                    del st.session_state[key]
            st.rerun()
    
    selected = st.session_state.student_page
    
    if selected == "Events Feed":
        st.markdown('<h1 class="main-header">🎯 Discover Events</h1>', unsafe_allow_html=True)
        
        db.update_event_status()
        
        col_filters = st.columns([2, 1, 1, 1])
        with col_filters[0]:
            search = st.text_input("🔍 Search events", placeholder="Search by title, description...")
        with col_filters[1]:
            event_type = st.selectbox("Type", ["All", "Workshop", "Hackathon", "Competition", "Bootcamp", "Seminar", "Conference", "Webinar"])
        with col_filters[2]:
            show_only = st.selectbox("Show", ["All", "Upcoming", "Ongoing", "Past"])
        with col_filters[3]:
            ai_only = st.checkbox("🤖 AI-Generated")
        
        events = db.get_all_events()
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
        
        st.caption(f"Found {len(filtered_events)} events")
        
        if filtered_events:
            for event in filtered_events:
                display_event_card(event, st.session_state.username)
        else:
            st.info("No events found matching your criteria.")
    
    elif selected == "My Registrations":
        st.header("📋 My Registrations")
        
        registrations = db.get_registrations_by_student(st.session_state.username)
        
        if not registrations:
            st.info("You haven't registered for any events yet.")
            return
        
        for reg in registrations:
            with st.container():
                st.markdown('<div class="event-card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    event_title = reg.get('event_title', 'Unknown Event')
                    st.markdown(f'<div class="card-title">{event_title}</div>', unsafe_allow_html=True)
                    
                    event_date = reg.get('event_date')
                    if event_date:
                        st.caption(f"📅 {format_date(event_date)}")
                    
                    venue = reg.get('venue', 'N/A')
                    st.caption(f"📍 {venue}")
                    
                    reg_status = reg.get('status', 'pending').title()
                    st.caption(f"📝 Status: {reg_status}")
                    
                    points = reg.get('points_awarded', 0)
                    badges = reg.get('badges_awarded', '')
                    if points > 0:
                        st.caption(f"🏆 Points: {points}")
                    if badges:
                        badge_list = badges.split(',')
                        for badge in badge_list:
                            st.markdown(f'<span style="background: #FFD700; color: #000; padding: 2px 6px; border-radius: 10px; font-size: 0.7rem; margin-right: 4px;">{badge}</span>', unsafe_allow_html=True)
                
                with col2:
                    event_status = reg.get('event_status', 'unknown')
                    if event_status == 'upcoming':
                        st.success("🟢 Upcoming")
                    elif event_status == 'ongoing':
                        st.warning("🟡 Ongoing")
                    else:
                        st.error("🔴 Completed")
                
                with col3:
                    if points > 0:
                        st.markdown(f'<div style="font-size: 1.5rem; font-weight: bold; text-align: center; color: #3B82F6;">{points}</div>', unsafe_allow_html=True)
                        st.caption("Points")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected == "Leaderboard":
        st.markdown('<h1 class="main-header">🏆 College Leaderboard</h1>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🏆 Overall Leaderboard", "📈 My Progress"])
        
        with tab1:
            leaderboard = db.get_leaderboard(limit=15)
            
            if leaderboard:
                st.markdown(f'<h2 style="text-align: center; margin-bottom: 2rem;">🏆 College Leaderboard</h2>', unsafe_allow_html=True)
                
                for student in leaderboard:
                    rank = student.get('rank', 0)
                    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
                    medal = medals.get(rank, f"{rank}.")
                    
                    with st.container():
                        st.markdown('<div class="leaderboard-card">', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
                        
                        with col1:
                            if rank <= 3:
                                st.markdown(f'<div style="font-size: 2rem; text-align: center;">{medal}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div style="font-size: 1.5rem; text-align: center; font-weight: bold;">{rank}.</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f'<div style="font-weight: bold; font-size: 1.1rem;">{student.get("name")}</div>', unsafe_allow_html=True)
                            st.caption(f"{student.get('roll_no', '')} | {student.get('department', '')} | Year {student.get('year', '')}")
                        
                        with col3:
                            points = student.get('total_points', 0)
                            st.markdown(f'<div style="font-size: 1.8rem; font-weight: bold; text-align: center; color: #3B82F6;">{points}</div>', unsafe_allow_html=True)
                            st.caption("Points")
                        
                        with col4:
                            events_count = student.get('events_count', 0)
                            st.metric("Events", events_count)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("---")
            else:
                st.info("No students found in leaderboard.")
    
    elif selected == "My Profile":
        st.header("👤 My Profile")
        
        student = db.get_user(st.session_state.username)
        
        if not student:
            st.error("User not found!")
            return
        
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
            mobile = student.get('mobile', 'Not provided')
            st.markdown(f"**Mobile:** {mobile}")
            st.markdown(f"**Username:** {student.get('username', 'N/A')}")
            st.markdown(f"**Member Since:** {format_date(student.get('created_at'))}")

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application function"""
    
    # Initialize session state
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'name' not in st.session_state:
        st.session_state.name = None
    if 'page' not in st.session_state:
        st.session_state.page = "login"
    if 'remember_me' not in st.session_state:
        st.session_state.remember_me = False
    
    # Session timeout check
    if (st.session_state.role and 'session_start' in st.session_state and 
        not st.session_state.remember_me):
        session_duration = datetime.now() - st.session_state.session_start
        if session_duration.total_seconds() > SESSION_TIMEOUT_MINUTES * 60:
            st.warning("Session timed out. Please login again.")
            if st.session_state.username:
                db.clear_reset_token(st.session_state.username)
            
            for key in list(st.session_state.keys()):
                if key != 'rerun_count':
                    del st.session_state[key]
            st.rerun()
    
    # Update event status periodically
    if 'last_status_update' not in st.session_state:
        st.session_state.last_status_update = datetime.now()
    
    if (datetime.now() - st.session_state.last_status_update).total_seconds() > 300:
        db.update_event_status()
        st.session_state.last_status_update = datetime.now()
    
    # Route based on page
    if st.session_state.page == "forgot_password":
        forgot_password_page()
    elif st.session_state.page == "student_register":
        student_registration_page()
    elif st.session_state.role is None:
        landing_page()
    elif st.session_state.role == 'student':
        student_dashboard()
    # Note: Other dashboards (admin, faculty, mentor) would be implemented similarly

# ============================================
# CUSTOM CSS
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
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 6px;
        line-height: 1.3;
    }
    
    .leaderboard-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .leaderboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()
