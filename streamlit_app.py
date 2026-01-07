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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import csv
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
        "top_n": 15,  # Display top 15 students in leaderboard
        "update_interval": 3600,  # Update leaderboard every hour
        "department_top_n": 5  # Top 5 per department
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
                        # Ensure URL has https:// scheme
                        if not self.url.startswith(('http://', 'https://')):
                            self.url = f'https://{self.url}'
                        
                        # Remove trailing slash if present
                        self.url = self.url.rstrip('/')
                        
                        self.headers = {
                            'apikey': self.key,
                            'Authorization': f'Bearer {self.key}',
                            'Content-Type': 'application/json',
                            'Prefer': 'return=minimal'
                        }
                        self.is_configured = True
                        logger.info("✅ Supabase configured successfully")
                        logger.info(f"Supabase URL: {self.url}")
                    else:
                        logger.warning("⚠️ Supabase credentials incomplete")
                else:
                    logger.warning("⚠️ Supabase credentials not found in secrets")
            else:
                logger.warning("⚠️ Streamlit secrets not available")
                
        except Exception as e:
            logger.error(f"Supabase initialization error: {e}")
    
    def execute_query(self, table, method='GET', data=None, filters=None, limit=1000):
        """Execute REST API query to Supabase"""
        if not self.is_configured:
            logger.error("Supabase not configured")
            return None
        
        try:
            import requests
            
            url = f"{self.url}/rest/v1/{table}"
            
            # Add filters
            params = []
            if filters:
                for k, v in filters.items():
                    # URL encode the value
                    import urllib.parse
                    encoded_value = urllib.parse.quote(str(v))
                    params.append(f"{k}=eq.{encoded_value}")
            
            # Add limit
            params.append(f"limit={limit}")
            
            # Combine params
            if params:
                url = f"{url}?{'&'.join(params)}"
            
            # Make request
            if method == 'GET':
                response = requests.get(url, headers=self.headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=data, timeout=10)
            elif method == 'PATCH':
                response = requests.patch(url, headers=self.headers, json=data, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers, timeout=10)
            
            response.raise_for_status()
            
            if method == 'GET':
                return response.json() if response.text else []
            else:
                return response.status_code in [200, 201, 204]
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Supabase API error: {e}")
            logger.error(f"URL attempted: {url[:100]}...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Supabase query: {e}")
            return None
    
    def insert(self, table, data):
        """Insert data into table"""
        return self.execute_query(table, 'POST', data)
    
    def select(self, table, filters=None, limit=1000):
        """Select data from table"""
        return self.execute_query(table, 'GET', filters=filters, limit=limit)
    
    def update(self, table, filters, data):
        """Update data in table"""
        return self.execute_query(table, 'PATCH', data, filters)
    
    def delete(self, table, filters):
        """Delete data from table"""
        return self.execute_query(table, 'DELETE', filters=filters)

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
            logger.error(f"SQLite error: {e}, Query: {query[:100]}")
            return None
    
    def insert(self, table, data):
        """Insert data into table"""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        try:
            cursor = self.execute_query(query, tuple(data.values()))
            self.conn.commit()
            return cursor.rowcount > 0
        except:
            return False
    
    def select(self, table, filters=None, limit=1000):
        """Select data from table"""
        query = f"SELECT * FROM {table}"
        
        if filters:
            conditions = ' AND '.join([f"{k} = ?" for k in filters.keys()])
            query = f"{query} WHERE {conditions}"
        
        query = f"{query} LIMIT {limit}"
        
        if filters:
            return self.execute_query(query, tuple(filters.values()), fetchall=True)
        else:
            return self.execute_query(query, fetchall=True)
    
    def update(self, table, filters, data):
        """Update data in table"""
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        conditions = ' AND '.join([f"{k} = ?" for k in filters.keys()])
        
        query = f"UPDATE {table} SET {set_clause} WHERE {conditions}"
        params = tuple(data.values()) + tuple(filters.values())
        
        try:
            cursor = self.execute_query(query, params)
            self.conn.commit()
            return cursor.rowcount > 0
        except:
            return False
    
    def delete(self, table, filters):
        """Delete data from table"""
        conditions = ' AND '.join([f"{k} = ?" for k in filters.keys()])
        query = f"DELETE FROM {table} WHERE {conditions}"
        
        try:
            cursor = self.execute_query(query, tuple(filters.values()))
            self.conn.commit()
            return cursor.rowcount > 0
        except:
            return False

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
                    logger.info("✅ OpenAI configured successfully")
                elif self.api_key:
                    logger.warning(f"⚠️ Invalid OpenAI API key format")
                    self.is_configured = False
                else:
                    logger.warning("⚠️ OpenAI API key not found in secrets")
                    self.is_configured = False
            else:
                logger.warning("⚠️ Streamlit secrets not available")
                self.is_configured = False
                
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize OpenAI: {str(e)[:100]}")
            self.is_configured = False
    
    def extract_event_info(self, text):
        """Extract event information from text using AI or regex fallback"""
        
        # Try OpenAI first if available
        if self.is_configured and self.api_key:
            try:
                with st.spinner("🤖 AI is processing your event..."):
                    return self._extract_with_openai(text)
            except Exception as e:
                logger.warning(f"AI extraction failed: {str(e)[:100]}. Using regex fallback.")
        
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
        - event_link: Event website/URL if mentioned (string or null)
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
            logger.error(f"Failed to parse AI response as JSON: {str(e)[:100]}")
            return self._extract_with_regex(text)
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

# ============================================
# PASSWORD RESET MANAGER
# ============================================

class PasswordResetManager:
    """Manage password reset functionality"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.reset_tokens = {}  # In production, use Redis or database
    
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
        
        # Validate password strength
        is_valid, msg = Validators.validate_password(new_password)
        if not is_valid:
            return False, msg
        
        # Hash and update password
        hashed_pass = hashlib.sha256(new_password.encode()).hexdigest()
        try:
            if self.db.use_supabase:
                success = self.db.client.update('users', {'username': username}, {'password': hashed_pass})
            else:
                cursor = self.db.conn.cursor()
                cursor.execute("UPDATE users SET password = ? WHERE username = ?", (hashed_pass, username))
                self.db.conn.commit()
                success = cursor.rowcount > 0
            
            if success:
                # Remove used token
                del self.reset_tokens[token]
                return True, "Password reset successful"
            else:
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
                st.warning("⚠️ Supabase not configured. Falling back to SQLite.")
                self.use_supabase = False
                self.client = SQLiteClient()
            else:
                # Test connection
                if not self._test_supabase_connection():
                    st.warning("⚠️ Supabase connection test failed. Falling back to SQLite.")
                    self.use_supabase = False
                    self.client = SQLiteClient()
        else:
            self.client = SQLiteClient()
        
        # Initialize database
        self._initialize_database()
        self._add_default_users()
    
    def _test_supabase_connection(self):
        """Test Supabase connection"""
        try:
            # Try a simple query to test connection
            result = self.client.execute_query('users', limit=1)
            logger.info("✅ Supabase connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {e}")
            return False
    
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
            # Users table
            """
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
                mobile TEXT,
                reset_token TEXT,
                reset_token_expiry TIMESTAMP,
                last_activity TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                login_attempts INTEGER DEFAULT 0,
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_points INTEGER DEFAULT 0,
                last_points_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Mentors table
            """
            CREATE TABLE IF NOT EXISTS mentors (
                id TEXT PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                full_name TEXT NOT NULL,
                department TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                contact TEXT NOT NULL,
                expertise TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # Events table
            """
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
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (mentor_id) REFERENCES mentors(id) ON DELETE SET NULL
            )
            """,
            
            # Registrations table
            """
            CREATE TABLE IF NOT EXISTS registrations (
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
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checked_in_at TIMESTAMP,
                UNIQUE(event_id, student_username),
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
            )
            """,
            
            # Likes table
            """
            CREATE TABLE IF NOT EXISTS event_likes (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                student_username TEXT NOT NULL,
                liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_id, student_username),
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
            )
            """,
            
            # Interested table
            """
            CREATE TABLE IF NOT EXISTS event_interested (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                student_username TEXT NOT NULL,
                interested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_id, student_username),
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
            )
            """,
            
            # Student badges table
            """
            CREATE TABLE IF NOT EXISTS student_badges (
                id TEXT PRIMARY KEY,
                student_username TEXT NOT NULL,
                badge_name TEXT NOT NULL,
                badge_type TEXT,
                awarded_for TEXT,
                event_id TEXT,
                event_title TEXT,
                awarded_by TEXT,
                awarded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_username) REFERENCES users(username) ON DELETE CASCADE
            )
            """,
            
            # Points history table
            """
            CREATE TABLE IF NOT EXISTS points_history (
                id TEXT PRIMARY KEY,
                student_username TEXT NOT NULL,
                points INTEGER NOT NULL,
                reason TEXT NOT NULL,
                event_id TEXT,
                event_title TEXT,
                awarded_by TEXT,
                awarded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_username) REFERENCES users(username) ON DELETE CASCADE
            )
            """
        ]
        
        for sql in tables_sql:
            self.client.execute_query(sql, commit=True)

    # ============================================
    # USER MANAGEMENT METHODS
    # ============================================
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials - SIMPLE AND RELIABLE"""
        try:
            logger.info(f"🔐 Login attempt: {username} as {role}")
        
            # Get user
            user = self.get_user(username)
            if not user:
                logger.warning(f"User not found: {username}")
                return False
        
            # Check role
            if user.get('role') != role:
                logger.warning(f"Role mismatch: {user.get('role')} != {role}")
                return False
        
            # Get stored password
            stored_pass = user.get('password', '')
        
            # Hash the input password
            input_hash = hashlib.sha256(password.encode()).hexdigest().lower()
        
            # For backward compatibility, also check plain text
            if stored_pass == input_hash or stored_pass == password:
                logger.info(f"✅ Login successful for {username}")
                return True
            else:
                logger.warning(f"❌ Password mismatch for {username}")
                return False
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def get_user(self, username):
        """Get user by username - IMPROVED"""
        try:
            if self.use_supabase:
                results = self.client.select('users', {'username': username})
                if results:
                    user = results[0]
                    # Ensure password field exists
                    if 'password' not in user:
                        user['password'] = ''
                    # Ensure points field exists
                    if 'total_points' not in user:
                        user['total_points'] = 0
                    return user
                return None
            else:
                return self.client.execute_query(
                    "SELECT * FROM users WHERE username = ?",
                    (username,), fetchone=True
                )
        except Exception as e:
            logger.error(f"Error getting user {username}: {e}")
            return None

    def add_user(self, user_data):
        """Add new user - ALWAYS HASHES PASSWORD"""
        try:
            # Get and validate password
            password = user_data.get('password', '')
            if not password:
                return False, "Password is required"
        
            # Always hash the password
            hashed_pass = hashlib.sha256(password.encode()).hexdigest().lower()
        
            user_record = {
                'id': str(uuid.uuid4()),
                'name': user_data.get('name'),
                'username': user_data.get('username'),
                'password': hashed_pass,
                'role': user_data.get('role', 'student'),
                'roll_no': user_data.get('roll_no', ''),
                'department': user_data.get('department', ''),
                'year': user_data.get('year', ''),
                'email': user_data.get('email', ''),
                'mobile': user_data.get('mobile', ''),
                'created_at': datetime.now().isoformat(),
                'total_points': 0,
                'last_points_update': datetime.now().isoformat()
            }
        
            if self.use_supabase:
                success = self.client.insert('users', user_record)
            else:
                success = self.client.insert('users', user_record)
        
            if success:
                logger.info(f"✅ User '{user_data.get('username')}' added")
                return True, "User registered successfully"
            else:
                logger.error(f"❌ Failed to add user: {user_data.get('username')}")
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
            
            if self.use_supabase:
                return self.client.update('users', {'username': username}, update_data)
            else:
                return self.client.update('users', {'username': username}, update_data)
        except:
            return False
    
    def update_user_mobile(self, username, mobile):
        """Update user's mobile number"""
        try:
            is_valid, msg = Validators.validate_mobile(mobile)
            if not is_valid:
                return False, msg
            
            formatted_mobile = Validators.format_mobile(mobile)
            
            update_data = {
                'mobile': formatted_mobile,
                'updated_at': datetime.now().isoformat()
            }
            
            if self.use_supabase:
                success = self.client.update('users', {'username': username}, update_data)
            else:
                success = self.client.update('users', {'username': username}, update_data)
            
            if success:
                return True, "Mobile number updated successfully"
            else:
                return False, "Failed to update mobile number"
                
        except Exception as e:
            logger.error(f"Error updating mobile: {e}")
            return False, "Failed to update mobile number"
    
    def update_user_password(self, username, new_password):
        """Update user's password"""
        try:
            is_valid, msg = Validators.validate_password(new_password)
            if not is_valid:
                return False, msg
            
            hashed_pass = hashlib.sha256(new_password.encode()).hexdigest()
            
            update_data = {
                'password': hashed_pass,
                'updated_at': datetime.now().isoformat(),
                'reset_token': None,
                'reset_token_expiry': None
            }
            
            if self.use_supabase:
                success = self.client.update('users', {'username': username}, update_data)
            else:
                success = self.client.update('users', {'username': username}, update_data)
            
            if success:
                return True, "Password updated successfully"
            else:
                return False, "Failed to update password"
                
        except Exception as e:
            logger.error(f"Error updating password: {e}")
            return False, "Failed to update password"
    
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
            
            if self.use_supabase:
                return self.client.update('users', {'username': username}, update_data)
            else:
                return self.client.update('users', {'username': username}, update_data)
        except Exception as e:
            logger.error(f"Error setting remember token: {e}")
            return False
    
    def verify_remember_token(self, username, token):
        """Verify remember me token"""
        try:
            if self.use_supabase:
                results = self.client.select('users', {'username': username})
                if not results:
                    return False
                user = results[0]
            else:
                user = self.client.execute_query(
                    "SELECT reset_token, reset_token_expiry FROM users WHERE username = ?",
                    (username,), fetchone=True
                )
            
            if user and user.get('reset_token') and user.get('reset_token_expiry'):
                stored_token = user.get('reset_token')
                expiry_str = user.get('reset_token_expiry')
                
                if isinstance(expiry_str, str):
                    expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                else:
                    expiry = expiry_str
                
                if stored_token == token and datetime.now() < expiry:
                    return True
            return False
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
            
            if self.use_supabase:
                return self.client.update('users', {'username': username}, update_data)
            else:
                return self.client.update('users', {'username': username}, update_data)
        except Exception as e:
            logger.error(f"Error clearing reset token: {e}")
            return False
    
    # ============================================
    # EVENT MANAGEMENT METHODS
    # ============================================
    
    def add_event(self, event_data):
        """Add new event"""
        try:
            event_id = event_data.get('id', str(uuid.uuid4()))
            
            event_record = {
                'id': event_id,
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
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            if self.use_supabase:
                success = self.client.insert('events', event_record)
            else:
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
                events = self.client.select('events', limit=1000)
                if events:
                    # Sort by date manually
                    events.sort(key=lambda x: x.get('event_date', ''), reverse=True)
                return events
            else:
                return self.client.execute_query(
                    "SELECT * FROM events ORDER BY event_date DESC",
                    fetchall=True
                )
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    def get_events_by_creator(self, username):
        """Get events created by specific user"""
        try:
            if self.use_supabase:
                events = self.client.select('events', {'created_by': username}, limit=1000)
                if events:
                    events.sort(key=lambda x: x.get('event_date', ''), reverse=True)
                return events
            else:
                return self.client.execute_query(
                    "SELECT * FROM events WHERE created_by = ? ORDER BY event_date DESC",
                    (username,), fetchall=True
                )
        except:
            return []
    
    def get_event_with_mentor(self, event_id):
        """Get event with mentor details"""
        try:
            if self.use_supabase:
                event = self.client.select('events', {'id': event_id})
                if event:
                    event = event[0]
                    if event.get('mentor_id'):
                        mentor = self.client.select('mentors', {'id': event['mentor_id']})
                        if mentor:
                            event['mentor_name'] = mentor[0].get('full_name')
                            event['mentor_contact'] = mentor[0].get('contact')
                            event['mentor_expertise'] = mentor[0].get('expertise')
                return event
            else:
                return self.client.execute_query(
                    "SELECT e.*, m.full_name as mentor_name, m.contact as mentor_contact, m.expertise as mentor_expertise FROM events e LEFT JOIN mentors m ON e.mentor_id = m.id WHERE e.id = ?",
                    (event_id,), fetchone=True
                )
        except:
            return None
    
    def assign_mentor_to_event(self, event_id, mentor_id):
        """Assign mentor to event"""
        try:
            update_data = {
                'mentor_id': mentor_id,
                'updated_at': datetime.now().isoformat()
            }
            
            if self.use_supabase:
                return self.client.update('events', {'id': event_id}, update_data)
            else:
                return self.client.update('events', {'id': event_id}, update_data)
        except:
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
                self.client.update('events', {'event_date': {'lt': now}, 'status': {'neq': 'past'}}, past_update)
                
                # Update ongoing events
                ongoing_update = {'status': 'ongoing', 'updated_at': now}
                self.client.update('events', {'event_date': {'gte': today_start, 'lte': today_end}, 'status': 'upcoming'}, ongoing_update)
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
    
    # ============================================
    # MENTOR MANAGEMENT
    # ============================================
    
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
            
            if self.use_supabase:
                user_success = self.client.insert('users', user_record)
                mentor_success = self.client.insert('mentors', mentor_record)
            else:
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
                mentors = self.client.select('mentors', limit=1000)
                if mentors:
                    mentors.sort(key=lambda x: x.get('full_name', ''))
                return mentors
            else:
                return self.client.execute_query(
                    "SELECT * FROM mentors ORDER BY full_name",
                    fetchall=True
                )
        except:
            return []
    
    def get_active_mentors(self):
        """Get active mentors only"""
        try:
            if self.use_supabase:
                mentors = self.client.select('mentors', {'is_active': True}, limit=1000)
                if mentors:
                    mentors.sort(key=lambda x: x.get('full_name', ''))
                return mentors
            else:
                return self.client.execute_query(
                    "SELECT * FROM mentors WHERE is_active = 1 ORDER BY full_name",
                    fetchall=True
                )
        except:
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
        except:
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
        except:
            return None
    
    def get_events_by_mentor(self, mentor_id):
        """Get events assigned to a mentor"""
        try:
            if self.use_supabase:
                events = self.client.select('events', {'mentor_id': mentor_id}, limit=1000)
                if events:
                    events.sort(key=lambda x: x.get('event_date', ''), reverse=True)
                return events
            else:
                return self.client.execute_query(
                    "SELECT * FROM events WHERE mentor_id = ? ORDER BY event_date DESC",
                    (mentor_id,), fetchall=True
                )
        except:
            return []
    
    def update_mentor(self, mentor_id, mentor_data):
        """Update mentor information"""
        try:
            if self.use_supabase:
                return self.client.update('mentors', {'id': mentor_id}, mentor_data)
            else:
                return self.client.update('mentors', {'id': mentor_id}, mentor_data)
        except:
            return False
    
    def delete_mentor(self, mentor_id):
        """Delete mentor (soft delete - set inactive)"""
        try:
            update_data = {'is_active': False, 'updated_at': datetime.now().isoformat()}
            
            if self.use_supabase:
                mentor_success = self.client.update('mentors', {'id': mentor_id}, update_data)
                user_success = self.client.update('users', {'username': mentor_id}, update_data)
            else:
                mentor_success = self.client.update('mentors', {'id': mentor_id}, update_data)
                user_success = self.client.update('users', {'username': mentor_id}, update_data)
            
            return mentor_success and user_success
        except:
            return False
    
    # ============================================
    # REGISTRATION METHODS
    # ============================================
    
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
            
            # Get student mobile
            student = self.get_user(reg_data['student_username'])
            mobile = student.get('mobile', '') if student else ''
            
            reg_id = reg_data.get('id', str(uuid.uuid4()))
            
            registration_record = {
                'id': reg_id,
                'event_id': reg_data.get('event_id'),
                'event_title': reg_data.get('event_title'),
                'student_username': reg_data.get('student_username'),
                'student_name': reg_data.get('student_name'),
                'student_roll': reg_data.get('student_roll'),
                'student_dept': reg_data.get('student_dept'),
                'student_mobile': mobile,
                'status': reg_data.get('status', 'pending'),
                'attendance': reg_data.get('attendance', 'absent'),
                'points_awarded': 0,
                'badges_awarded': '',
                'mentor_notes': '',
                'registered_at': datetime.now().isoformat()
            }
            
            if self.use_supabase:
                success = self.client.insert('registrations', registration_record)
            else:
                success = self.client.insert('registrations', registration_record)
            
            if success:
                # Update event participant count
                self._update_event_participant_count(reg_data['event_id'])
                
                logger.info(f"New registration: {reg_data['student_username']}")
                return reg_id, "Registration successful"
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
                return registrations
            else:
                return self.client.execute_query(
                    "SELECT r.*, e.event_date, e.venue, e.status as event_status FROM registrations r LEFT JOIN events e ON r.event_id = e.id WHERE r.student_username = ? ORDER BY r.registered_at DESC",
                    (username,), fetchall=True
                )
        except:
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
        except:
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
        except:
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
            
            update_data = {
                'status': status,
                'updated_at': datetime.now().isoformat()
            }
            
            # Award points if specified
            if points is not None and points > 0:
                update_data['points_awarded'] = points
                
                # Update student's total points
                student = self.get_user(student_username)
                if student:
                    current_points = student.get('total_points', 0)
                    new_points = current_points + points
                    
                    # Update student's total points
                    student_update_data = {
                        'total_points': new_points,
                        'last_points_update': datetime.now().isoformat(),
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    if self.use_supabase:
                        self.client.update('users', {'username': student_username}, student_update_data)
                    else:
                        self.client.update('users', {'username': student_username}, student_update_data)
                    
                    # Record points history
                    points_record = {
                        'id': str(uuid.uuid4()),
                        'student_username': student_username,
                        'points': points,
                        'reason': f"Awarded for {status.lower()} in {event_title}",
                        'event_id': event_id,
                        'event_title': event_title,
                        'awarded_by': st.session_state.username if hasattr(st, 'session_state') and hasattr(st.session_state, 'username') else 'system',
                        'awarded_at': datetime.now().isoformat()
                    }
                    
                    if self.use_supabase:
                        self.client.insert('points_history', points_record)
                    else:
                        self.client.insert('points_history', points_record)
            
            # Award badge if specified
            if badge:
                current_badges = registration.get('badges_awarded', '')
                if current_badges:
                    badges_list = current_badges.split(',')
                    if badge not in badges_list:
                        badges_list.append(badge)
                        update_data['badges_awarded'] = ','.join(badges_list)
                else:
                    update_data['badges_awarded'] = badge
                
                # Add to student badges table
                badge_record = {
                    'id': str(uuid.uuid4()),
                    'student_username': student_username,
                    'badge_name': badge,
                    'badge_type': status.lower(),
                    'awarded_for': f"{status} in {event_title}",
                    'event_id': event_id,
                    'event_title': event_title,
                    'awarded_by': st.session_state.username if hasattr(st, 'session_state') and hasattr(st.session_state, 'username') else 'system',
                    'awarded_at': datetime.now().isoformat()
                }
                
                if self.use_supabase:
                    self.client.insert('student_badges', badge_record)
                else:
                    self.client.insert('student_badges', badge_record)
            
            # Add mentor notes
            if mentor_notes:
                update_data['mentor_notes'] = mentor_notes
            
            # Update registration
            if self.use_supabase:
                success = self.client.update('registrations', {'id': registration_id}, update_data)
            else:
                success = self.client.update('registrations', {'id': registration_id}, update_data)
            
            if success:
                return True, "Registration updated successfully"
            else:
                return False, "Failed to update registration"
                
        except Exception as e:
            logger.error(f"Error updating registration status: {e}")
            return False, str(e)
    
    # ============================================
    # GAMIFICATION & LEADERBOARD METHODS
    # ============================================
    
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
                'last_points_update': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            if self.use_supabase:
                success = self.client.update('users', {'username': student_username}, update_data)
            else:
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
            
            if self.use_supabase:
                self.client.insert('points_history', points_record)
            else:
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
                
                if self.use_supabase:
                    self.client.insert('student_badges', badge_record)
                else:
                    self.client.insert('student_badges', badge_record)
            
            # Update registration points if exists
            try:
                if self.use_supabase:
                    regs = self.client.select('registrations', {
                        'event_id': event_id,
                        'student_username': student_username
                    })
                    if regs:
                        reg_update = {
                            'points_awarded': points,
                            'badges_awarded': badge,
                            'updated_at': datetime.now().isoformat()
                        }
                        self.client.update('registrations', {'id': regs[0]['id']}, reg_update)
                else:
                    reg = self.client.execute_query(
                        "SELECT id FROM registrations WHERE event_id = ? AND student_username = ?",
                        (event_id, student_username), fetchone=True
                    )
                    if reg:
                        reg_update = {
                            'points_awarded': points,
                            'badges_awarded': badge
                        }
                        self.client.update('registrations', {'id': reg['id']}, reg_update)
            except Exception as e:
                logger.warning(f"Could not update registration points: {e}")
            
            return True, f"Awarded {points} points and badge: {badge}"
            
        except Exception as e:
            logger.error(f"Error awarding points: {e}")
            return False, str(e)
    
    def get_student_points(self, username):
        """Get student's total points"""
        try:
            user = self.get_user(username)
            if user:
                return user.get('total_points', 0)
            return 0
        except:
            return 0
    
    def get_student_badges(self, username):
        """Get all badges earned by student"""
        try:
            if self.use_supabase:
                badges = self.client.select('student_badges', {'student_username': username})
                return badges if badges else []
            else:
                return self.client.execute_query(
                    "SELECT * FROM student_badges WHERE student_username = ? ORDER BY awarded_at DESC",
                    (username,), fetchall=True
                )
        except:
            return []
    
    def get_points_history(self, username):
        """Get points history for student"""
        try:
            if self.use_supabase:
                history = self.client.select('points_history', {'student_username': username})
                return history if history else []
            else:
                return self.client.execute_query(
                    "SELECT * FROM points_history WHERE student_username = ? ORDER BY awarded_at DESC",
                    (username,), fetchall=True
                )
        except:
            return []
    
    def get_leaderboard(self, limit=15, department=None):
        """Get leaderboard of top students"""
        try:
            if self.use_supabase:
                # For Supabase, we need to get all students and sort manually
                users = self.client.select('users', {'role': 'student'}, limit=1000)
                if not users:
                    return []
                
                # Filter by department if specified
                if department:
                    users = [u for u in users if u.get('department') == department]
                
                # Sort by points (descending), then by name
                users.sort(key=lambda x: (x.get('total_points', 0), x.get('name', '')), reverse=True)
                
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
                
                # Add rank
                for i, result in enumerate(results, 1):
                    result['rank'] = i
                
                return results
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []
    
    def get_department_leaderboard(self, limit=5):
        """Get top students per department"""
        try:
            all_students = []
            departments = COLLEGE_CONFIG['departments']
            
            for dept in departments:
                dept_students = self.get_leaderboard(limit=limit, department=dept)
                for student in dept_students:
                    student['department_rank'] = dept_students.index(student) + 1
                all_students.extend(dept_students)
            
            return all_students
        except Exception as e:
            logger.error(f"Error getting department leaderboard: {e}")
            return []
    
    def get_top_badges(self, limit=10):
        """Get students with most badges"""
        try:
            if self.use_supabase:
                # This is more complex in Supabase without direct SQL
                # We'll implement a simplified version
                users = self.client.select('users', {'role': 'student'}, limit=1000)
                if not users:
                    return []
                
                # Get badge counts for each user
                for user in users:
                    badges = self.get_student_badges(user['username'])
                    user['badges_count'] = len(badges)
                
                # Sort by badge count
                users.sort(key=lambda x: x.get('badges_count', 0), reverse=True)
                return users[:limit]
            else:
                return self.client.execute_query(
                    """
                    SELECT u.name, u.username, u.roll_no, u.department, 
                           COUNT(sb.id) as badges_count,
                           GROUP_CONCAT(DISTINCT sb.badge_name) as badges_list
                    FROM users u
                    LEFT JOIN student_badges sb ON u.username = sb.student_username
                    WHERE u.role = 'student'
                    GROUP BY u.username
                    ORDER BY badges_count DESC
                    LIMIT ?
                    """,
                    (limit,), fetchall=True
                )
        except Exception as e:
            logger.error(f"Error getting top badges: {e}")
            return []
    
    def get_student_rank(self, username):
        """Get student's rank in leaderboard"""
        try:
            leaderboard = self.get_leaderboard(limit=1000)  # Get all students
            for i, student in enumerate(leaderboard, 1):
                if student['username'] == username:
                    return i
            return None
        except:
            return None
    
    # ============================================
    # LIKES & INTEREST METHODS (FIXED)
    # ============================================
    
    def add_like(self, event_id, student_username):
        """Add a like for an event"""
        try:
            like_record = {
                'id': str(uuid.uuid4()),
                'event_id': event_id,
                'student_username': student_username,
                'liked_at': datetime.now().isoformat()
            }
            
            if self.use_supabase:
                return self.client.insert('event_likes', like_record)
            else:
                return self.client.insert('event_likes', like_record)
        except:
            return False
    
    def remove_like(self, event_id, student_username):
        """Remove a like for an event"""
        try:
            if self.use_supabase:
                return self.client.delete('event_likes', {
                    'event_id': event_id,
                    'student_username': student_username
                })
            else:
                return self.client.delete('event_likes', {
                    'event_id': event_id,
                    'student_username': student_username
                })
        except:
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
        except:
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
        except:
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
            
            if self.use_supabase:
                return self.client.insert('event_interested', interested_record)
            else:
                return self.client.insert('event_interested', interested_record)
        except:
            return False
    
    def remove_interested(self, event_id, student_username):
        """Remove interested for an event"""
        try:
            if self.use_supabase:
                return self.client.delete('event_interested', {
                    'event_id': event_id,
                    'student_username': student_username
                })
            else:
                return self.client.delete('event_interested', {
                    'event_id': event_id,
                    'student_username': student_username
                })
        except:
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
        except:
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
        except:
            return 0
    
    def get_student_liked_events(self, student_username):
        """Get all events liked by a student - FIXED"""
        try:
            if self.use_supabase:
                # For Supabase, we need to get liked events first, then get event details
                likes = self.client.select('event_likes', {'student_username': student_username})
                if not likes:
                    return []
                
                # Get all event IDs that the student liked
                event_ids = [like['event_id'] for like in likes]
                
                # Get event details for each event ID
                liked_events = []
                for event_id in event_ids:
                    event = self.client.select('events', {'id': event_id})
                    if event:
                        liked_events.append(event[0])
                
                # Sort by liked date (most recent first)
                liked_events.sort(key=lambda x: next(
                    (like['liked_at'] for like in likes if like['event_id'] == x['id']),
                    ''), reverse=True)
                
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
        """Get all events student is interested in - FIXED"""
        try:
            if self.use_supabase:
                # For Supabase, we need to get interested events first, then get event details
                interests = self.client.select('event_interested', {'student_username': student_username})
                if not interests:
                    return []
                
                # Get all event IDs that the student is interested in
                event_ids = [interest['event_id'] for interest in interests]
                
                # Get event details for each event ID
                interested_events = []
                for event_id in event_ids:
                    event = self.client.select('events', {'id': event_id})
                    if event:
                        interested_events.append(event[0])
                
                # Sort by interested date (most recent first)
                interested_events.sort(key=lambda x: next(
                    (interest['interested_at'] for interest in interests if interest['event_id'] == x['id']),
                    ''), reverse=True)
                
                return interested_events
            else:
                return self.client.execute_query(
                    "SELECT e.* FROM events e JOIN event_interested i ON e.id = i.event_id WHERE i.student_username = ? ORDER BY i.interested_at DESC",
                    (student_username,), fetchall=True
                )
        except Exception as e:
            logger.error(f"Error getting interested events: {e}")
            return []
    
    # ============================================
    # DEFAULT USERS
    # ============================================

    def _add_default_users(self):
        """Add default admin and faculty users - SIMPLIFIED"""
        try:
            # Always add default users with correct passwords
            default_users = [
                {
                    'username': 'admin@raisoni',
                    'password': 'Admin@12345',
                    'name': 'Administrator',
                    'role': 'admin',
                    'email': 'admin@ghraisoni.edu',
                    'department': 'Administration'
                },
                {
                    'username': 'faculty@raisoni',
                    'password': 'Faculty@12345',
                    'name': 'Faculty Coordinator',
                    'role': 'faculty',
                    'email': 'faculty@ghraisoni.edu',
                    'department': 'Faculty'
                }
            ]    
        
            for user_data in default_users:
                # Delete if exists first (to ensure clean state)
                if self.use_supabase:
                    self.client.delete('users', {'username': user_data['username']})
                else:
                    self.client.execute_query(
                        "DELETE FROM users WHERE username = ?",
                        (user_data['username'],), commit=True
                    )
            
                # Add fresh user
                self.add_user(user_data)
                logger.info(f"Added/replaced default user: {user_data['username']}")
        
            # Add default students
            self._add_default_students()
        
            return True
        
        except Exception as e:
            logger.error(f"Error adding default users: {e}")
            return False
    
    def _add_default_students(self):
        """Add default student accounts - FIXED VERSION"""
        try:
            default_students = [
                {
                    'name': 'Rohan Sharma',
                    'username': 'rohan@student',
                    'password': 'Student@123',  # Plain text, will be hashed
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
                    # Create user data with plain password
                    user_data = {
                        'name': student['name'],
                        'username': student['username'],
                        'password': student['password'],  # Plain text
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
                    else:
                        logger.error(f"❌ Failed to add student {student['name']}: {message}")
                else:
                    logger.info(f"Student already exists: {student['name']}")
        
            return True
        
        except Exception as e:
            logger.error(f"Error adding default students: {e}")
            traceback.print_exc()
            return False
    
    # ============================================
    # SYSTEM STATISTICS
    # ============================================
    
    def get_system_stats(self):
        """Get system statistics"""
        try:
            stats = {}
            
            # Get all users
            users = self.client.select('users') if self.use_supabase else self.client.execute_query("SELECT role FROM users", fetchall=True)
            if users:
                role_counts = {}
                total_points = 0
                for user in users:
                    role = user.get('role', 'unknown')
                    role_counts[role] = role_counts.get(role, 0) + 1
                    total_points += user.get('total_points', 0)
                stats['user_counts'] = role_counts
                stats['total_points'] = total_points
            
            # Get all events
            events = self.client.select('events') if self.use_supabase else self.client.execute_query("SELECT status FROM events", fetchall=True)
            if events:
                status_counts = {}
                for event in events:
                    status = event.get('status', 'upcoming')
                    status_counts[status] = status_counts.get(status, 0) + 1
                stats['event_counts'] = status_counts
            
            # Count AI events
            ai_count = 0
            for event in events:
                if event.get('ai_generated'):
                    ai_count += 1
            stats['ai_events'] = ai_count
            
            # Count registrations
            if self.use_supabase:
                regs = self.client.select('registrations')
                stats['total_registrations'] = len(regs) if regs else 0
            else:
                result = self.client.execute_query("SELECT COUNT(*) as count FROM registrations", fetchone=True)
                stats['total_registrations'] = result['count'] if result else 0
            
            # Count active mentors
            mentors = self.get_active_mentors()
            stats['active_mentors'] = len(mentors)
            
            # Recent events (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            recent_count = 0
            for event in events:
                created_at = event.get('created_at', '')
                if created_at > week_ago:
                    recent_count += 1
            stats['recent_events'] = recent_count
            
            # Leaderboard stats
            top_students = self.get_leaderboard(limit=5)
            stats['top_students'] = top_students
            
            # Total badges awarded
            if self.use_supabase:
                badges = self.client.select('student_badges')
                stats['total_badges'] = len(badges) if badges else 0
            else:
                result = self.client.execute_query("SELECT COUNT(*) as count FROM student_badges", fetchone=True)
                stats['total_badges'] = result['count'] if result else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

# Initialize database
db = DatabaseManager(use_supabase=USE_SUPABASE)

def fix_existing_passwords(db_instance):
    """Fix existing passwords by hashing any plain text passwords"""
    logger.info("=== FIXING EXISTING PASSWORDS ===")
    
    # Get all users
    if db_instance.use_supabase:
        users = db_instance.client.select('users')
    else:
        users = db_instance.client.execute_query("SELECT * FROM users", fetchall=True)
    
    if users:
        logger.info(f"Found {len(users)} users to check")
        
        for user in users:
            username = user.get('username')
            stored_pass = user.get('password', '')
            
            # Skip if already hashed (64 hex chars, lowercase)
            is_hashed = (len(stored_pass) == 64 and 
                        all(c in '0123456789abcdef' for c in stored_pass.lower()))
            
            if not is_hashed and stored_pass:
                logger.info(f"⚠️ Plain text password found for {username}: '{stored_pass[:20]}...'")
                
                # Hash the password
                new_hash = hashlib.sha256(stored_pass.encode()).hexdigest().lower()
                
                # Update in database
                if db_instance.use_supabase:
                    success = db_instance.client.update('users', {'username': username}, {'password': new_hash})
                else:
                    success = db_instance.client.update('users', {'username': username}, {'password': new_hash})
                
                if success:
                    logger.info(f"✅ Fixed {username}: Hashed to {new_hash[:20]}...")
                else:
                    logger.error(f"❌ Failed to fix password for {username}")
            else:
                logger.info(f"✅ {username} already has hashed password")
    
    logger.info("=== PASSWORD FIX COMPLETE ===")

# Fix existing passwords
fix_existing_passwords(db)

# Initialize password reset manager
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
        if len(image_bytes) > 5 * 1024 * 1024:  # 5MB limit
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

def check_remember_me_cookie():
    """Check for remember me cookie and auto-login"""
    if 'remember_me' not in st.session_state:
        st.session_state.remember_me = False
    
    # Check for cookie in query parameters
    try:
        # Try to get query parameters from URL
        if hasattr(st, 'query_params'):
            # Streamlit 1.28+ way
            params = st.query_params.to_dict()
        else:
            # Fallback for older Streamlit versions
            # Try to parse from URL manually
            params = {}
            import urllib.parse
            from urllib.parse import urlparse, parse_qs
            
            # Get current URL from st (if available)
            if hasattr(st, 'get_current_url'):
                url = st.get_current_url()
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                # Convert list values to single values
                params = {k: v[0] if len(v) == 1 else v for k, v in params.items()}
        
        if 'remember_token' in params and 'remember_user' in params:
            token = params['remember_token']
            username = params['remember_user']
            
            if db.verify_remember_token(username, token):
                user = db.get_user(username)
                if user:
                    st.session_state.role = user.get('role')
                    st.session_state.username = username
                    st.session_state.name = user.get('name', username)
                    st.session_state.session_start = datetime.now()
                    st.session_state.remember_me = True
                    st.success(f"Welcome back, {st.session_state.name}!")
                    st.rerun()
                    
    except Exception as e:
        logger.debug(f"Error checking remember me cookie: {e}")

# ============================================
# LEADERBOARD DISPLAY FUNCTIONS
# ============================================

def display_leaderboard(leaderboard_data, title="🏆 College Leaderboard"):
    """Display leaderboard with rankings"""
    st.markdown(f'<h2 style="text-align: center; margin-bottom: 2rem;">{title}</h2>', unsafe_allow_html=True)
    
    if not leaderboard_data:
        st.info("No students found in leaderboard.")
        return
    
    # Create medal emojis for top 3
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    
    for student in leaderboard_data:
        rank = student.get('rank', 0)
        medal = medals.get(rank, f"{rank}.")
        
        with st.container():
            st.markdown('<div class="leaderboard-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
            
            with col1:
                # Display rank with medal for top 3
                if rank <= 3:
                    st.markdown(f'<div style="font-size: 2rem; text-align: center;">{medal}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="font-size: 1.5rem; text-align: center; font-weight: bold;">{rank}.</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div style="font-weight: bold; font-size: 1.1rem;">{student.get("name")}</div>', unsafe_allow_html=True)
                st.caption(f"{student.get('roll_no', '')} | {student.get('department', '')} | Year {student.get('year', '')}")
                
                # Display badges if available
                if student.get('badges_list'):
                    badges = student['badges_list'].split(',')[:3]  # Show only first 3 badges
                    badge_html = ' '.join([f'<span style="background: #FFD700; color: #000; padding: 2px 6px; border-radius: 10px; font-size: 0.7rem; margin-right: 4px;">{badge}</span>' for badge in badges])
                    st.markdown(badge_html, unsafe_allow_html=True)
            
            with col3:
                # Points display
                points = student.get('total_points', 0)
                st.markdown(f'<div style="font-size: 1.8rem; font-weight: bold; text-align: center; color: #3B82F6;">{points}</div>', unsafe_allow_html=True)
                st.caption("Points", help="Total points earned")
            
            with col4:
                # Events count
                events_count = student.get('events_count', 0)
                st.metric("Events", events_count)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

def display_student_gamification_profile(username):
    """Display student's gamification profile"""
    student = db.get_user(username)
    if not student:
        st.error("Student not found!")
        return
    
    # Get student data
    points = db.get_student_points(username)
    badges = db.get_student_badges(username)
    points_history = db.get_points_history(username)
    rank = db.get_student_rank(username)
    
    # Display profile header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f'<h2>🎮 Gamification Profile</h2>', unsafe_allow_html=True)
        st.markdown(f'**Student:** {student.get("name")}')
        st.markdown(f'**Roll No:** {student.get("roll_no", "N/A")}')
        st.markdown(f'**Department:** {student.get("department", "N/A")}')
    
    with col2:
        st.metric("🏆 Total Points", points)
    
    with col3:
        if rank:
            st.metric("📊 Rank", f"#{rank}")
        else:
            st.metric("📊 Rank", "Not Ranked")
    
    # Badges section
    st.markdown("### 🏅 Badges Earned")
    if badges:
        cols = st.columns(4)
        for i, badge in enumerate(badges[:8]):  # Show up to 8 badges
            with cols[i % 4]:
                badge_name = badge.get('badge_name', '')
                event_title = badge.get('event_title', '')
                awarded_at = format_date(badge.get('awarded_at', ''))
                
                st.markdown(f'''
                <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center; 
                            margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="font-size: 2rem;">{badge_name.split()[0]}</div>
                    <div style="font-weight: bold; margin: 0.5rem 0;">{badge_name}</div>
                    <div style="font-size: 0.8rem; color: #666;">{event_title}</div>
                    <div style="font-size: 0.7rem; color: #888;">{awarded_at}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        if len(badges) > 8:
            st.caption(f"And {len(badges) - 8} more badges...")
    else:
        st.info("No badges earned yet. Participate in events to earn badges!")
    
    # Points history
    st.markdown("### 📈 Points History")
    if points_history:
        history_data = []
        for history in points_history[:10]:  # Show last 10 entries
            history_data.append({
                'Date': format_date(history.get('awarded_at')),
                'Points': f"+{history.get('points', 0)}",
                'Reason': history.get('reason', ''),
                'Event': history.get('event_title', '')
            })
        
        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No points history yet.")
    
    # Progress section
    st.markdown("### 🚀 Your Progress")
    
    # Calculate next milestone
    milestones = [100, 250, 500, 1000, 2000, 5000]
    next_milestone = next((m for m in milestones if m > points), None)
    
    if next_milestone:
        progress = (points / next_milestone) * 100
        st.progress(progress / 100)
        st.caption(f"{points} / {next_milestone} points to next milestone")
        
        col_a, col_b = st.columns(2)
        with col_a:
            events_registered = len(db.get_registrations_by_student(username))
            st.metric("Events Registered", events_registered)
        
        with col_b:
            st.metric("Next Milestone", f"{next_milestone - points} points needed")
    else:
        st.success("🎉 You've reached all milestones! You're a top achiever!")

# ============================================
# EVENT CARD DISPLAY (FIXED VERSION WITHOUT NESTED COLUMNS)
# ============================================
def display_event_card(event, current_user=None):
    """Display improved event card with flyer, mentor info, and registration links"""
    if not event or not event.get('id'):
        return
    
    event_id = event.get('id')
    
    # Create a container for the entire card
    card_container = st.container()
    
    with card_container:
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        # Create two-column layout: image on left, details on right
        col_img, col_info = st.columns([1, 3], gap="medium")
        
        with col_img:
            # Display event flyer if available
            flyer = event.get('flyer_path')
            if flyer and flyer.startswith('data:image'):
                try:
                    st.image(flyer, use_column_width=True)
                except:
                    # Fallback if image fails to load
                    st.markdown('<div style="width: 100%; height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; border-radius: 8px;"><span style="font-size: 32px; color: white;">🎯</span></div>', 
                               unsafe_allow_html=True)
            else:
                # Default placeholder
                st.markdown('<div style="width: 100%; height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; border-radius: 8px;"><span style="font-size: 32px; color: white;">🎯</span></div>', 
                           unsafe_allow_html=True)
        
        with col_info:
            # Header with title and badges - use HTML for layout
            title = event.get('title', 'Untitled Event')
            if len(title) > 60:
                title = title[:57] + "..."
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
            
            # Status and date row - using HTML layout instead of nested columns
            event_date = event.get('event_date')
            status_html = get_event_status(event_date)
            formatted_date = format_date(event_date)
            
            # Use markdown for status and date side by side
            st.markdown(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>{status_html}</div>
                <div style="color: #666; font-size: 0.9rem;">📅 {formatted_date}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Event details - use single row
            venue = event.get('venue', 'TBD')
            if len(venue) > 25:
                venue = venue[:22] + "..."
            
            event_type = event.get('event_type', 'Event')
            max_participants = event.get('max_participants', 100)
            current_participants = event.get('current_participants', 0)
            
            st.caption(f"📍 {venue} | 🏷️ {event_type} | 👥 {current_participants}/{max_participants}")
            
            # Mentor information (if assigned)
            if event.get('mentor_id'):
                mentor = db.get_mentor_by_id(event['mentor_id'])
                if mentor:
                    st.markdown('<div style="background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%); padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid #E5E7EB; border-left: 3px solid #8B5CF6; font-size: 0.9rem;">', unsafe_allow_html=True)
                    st.markdown(f"**Mentor:** {mentor['full_name']} | **Contact:** {mentor['contact']}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Engagement metrics
            likes_count = db.get_event_likes_count(event_id)
            interested_count = db.get_event_interested_count(event_id)
            
            # Engagement row - use a separate container for buttons
            if current_user:
                # Create a container for engagement buttons
                engagement_container = st.container()
                with engagement_container:
                    # Use horizontal layout for buttons without creating nested columns
                    button_col1, button_col2, button_col3 = st.columns(3)
                    
                    with button_col1:
                        is_liked = db.is_event_liked(event_id, current_user)
                        like_text = "❤️ Liked" if is_liked else "🤍 Like"
                        like_type = "secondary" if is_liked else "primary"
                        
                        import time
                        unique_key = f"like_{event_id}_{int(time.time() * 1000) % 10000}"
                        if st.button(like_text, key=unique_key, 
                                   use_container_width=True, type=like_type, 
                                   help="Like this event"):
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
                        if st.button(interested_text, key=unique_key_interested, 
                                    use_container_width=True, type=interested_type,
                                    help="Mark as interested"):
                            if is_interested:
                                if db.remove_interested(event_id, current_user):
                                    st.rerun()
                            else:
                                if db.add_interested(event_id, current_user):
                                    st.rerun()
                    
                    with button_col3:
                        # Share button to promote the app
                        unique_key_share = f"share_{event_id}_{int(time.time() * 1000) % 10000}"
                        if st.button("📤 Share", key=unique_key_share,
                                   use_container_width=True, type="secondary",
                                   help="Share this event with friends"):
                            # Create a share message
                            event_title = event.get('title', 'Cool Event')
                            share_text = f"Check out '{event_title}' at G H Raisoni College Event Manager! 🎓\n\nJoin the platform to discover more events: [Event Manager App]"
                            
                            # Copy to clipboard
                            st.code(share_text)
                            st.success("📋 Share message copied! Share with your friends.")
            
            # Show engagement counts
            st.caption(f"❤️ {likes_count} Likes | ⭐ {interested_count} Interested")
            
            # Event links (if available)
            event_link = event.get('event_link', '')
            registration_link = event.get('registration_link', '')
            
            if event_link or registration_link:
                with st.expander("🔗 Event Links", expanded=False):
                    if event_link:
                        st.markdown(f"**🌐 Event Page:** [Click here]({event_link})")
                    if registration_link:
                        st.markdown(f"**📝 Registration:** [Click here]({registration_link})")
            
            # Description (collapsible)
            desc = event.get('description', '')
            if desc:
                if len(desc) > 150:
                    with st.expander("📝 Description", expanded=False):
                        st.write(desc)
                else:
                    st.caption(desc[:150] + "..." if len(desc) > 150 else desc)
        
        # ============================================
        # REGISTRATION SECTION (For students only)
        # ============================================
        if current_user and st.session_state.role == 'student':
            st.markdown('<div style="background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); padding: 8px; border-radius: 6px; margin-top: 8px; border-left: 3px solid #3B82F6; font-size: 0.9rem;">', unsafe_allow_html=True)
            
            is_registered = db.is_student_registered(event_id, current_user)
            
            if is_registered:
                st.success("✅ You are already registered for this event")
                
                # Show "I Have Registered Externally" button
                if registration_link:
                    unique_key_ext_reg = f"ext_reg_{event_id}_{int(time.time() * 1000) % 10000}"
                    if st.button("✅ I Have Registered Externally", 
                               key=unique_key_ext_reg,
                               use_container_width=True,
                               type="secondary",
                               help="Mark that you have registered externally"):
                        # Update registration status
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
                # Registration options
                reg_col1, reg_col2 = st.columns([1, 1])
                
                with reg_col1:
                    # Register in App button
                    unique_key_app_reg = f"app_reg_{event_id}_{int(time.time() * 1000) % 10000}"
                    if st.button("📱 Register in App", 
                                key=unique_key_app_reg,
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
                            reg_id, message = db.add_registration(reg_data)
                            if reg_id:
                                st.success("✅ Registered in college system!")
                                st.rerun()
                            else:
                                st.error(message)
                
                with reg_col2:
                    # External registration link button (if available)
                    if registration_link:
                        st.markdown(f"[🌐 Register Externally]({registration_link})")
                        st.caption("Click to register on external site")
                    else:
                        st.info("No external registration link available")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Creator info
        created_by = event.get('created_by_name', 'Unknown')
        st.caption(f"👤 Created by: {created_by}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PASSWORD RESET PAGE
# ============================================
def forgot_password_page():
    """Password reset page"""
    st.markdown('<div class="college-header"><h2>🔐 Password Recovery</h2></div>', 
                unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Request Reset", "Reset Password"])
    
    with tab1:
        st.markdown("### Request Password Reset")
        st.info("Enter your registered email address to receive a password reset link.")
        
        reset_email = st.text_input("Email Address", placeholder="your.email@ghraisoni.edu")
        
        if st.button("Send Reset Link", use_container_width=True, type="primary"):
            if reset_email:
                user = db.get_user_by_email(reset_email)
                if user:
                    # Generate reset token
                    token = password_reset_manager.generate_reset_token(user['username'])
                    expiry = datetime.now() + timedelta(hours=1)
                    
                    # Store token in database
                    if db.set_remember_token(user['username'], token, expiry.isoformat()):
                        # In production, send email with reset link
                        reset_url = f"https://yourapp.com/reset?token={token}"
                        
                        st.success(f"✅ Reset link sent to {reset_email}")
                        st.info(f"**Test Token (for development):** `{token}`")
                        st.markdown(f"""
                        **In production, an email would be sent with:**
                        - Reset link: {reset_url}
                        - Token expires: {expiry.strftime('%I:%M %p, %d %b %Y')}
                        """)
                        
                        # Development: Show direct link
                        st.markdown("---")
                        st.markdown("**For testing:**")
                        st.code(f"Reset token: {token}")
                        st.markdown(f"Copy this token and use it in the 'Reset Password' tab")
                    else:
                        st.error("Failed to generate reset token")
                else:
                    st.error("Email not found in our system")
            else:
                st.error("Please enter your email address")
    
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
                # Validate password strength
                is_valid, msg = Validators.validate_password(new_password)
                if not is_valid:
                    st.error(msg)
                else:
                    success, result = password_reset_manager.reset_password(reset_token, new_password)
                    if success:
                        st.success("✅ Password reset successful! You can now login with your new password.")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"Reset failed: {result}")
    
    st.markdown("---")
    if st.button("← Back to Login", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()

# ============================================
# LANDING PAGE WITH LOGIN
# ============================================
def landing_page():
    """Landing page with app info and login"""
    # Display logo
    try:
        logo_path = "ghribmjal-logo.jpg"
        if os.path.exists(logo_path):
            st.image(logo_path, width=150)
    except:
        pass
    
    st.markdown(f'<div class="college-header"><h2>{COLLEGE_CONFIG["name"]}</h2><p>Advanced Event Management System with Gamification</p></div>', 
                unsafe_allow_html=True)

    # App Information
    with st.expander("📱 About This App", expanded=True):
        st.markdown("""
        ### Welcome to G H Raisoni Event Management System
        
        **New Gamification Features:**
        - 🏆 **Leaderboard:** Compete with other students for top rankings
        - 🎮 **Points System:** Earn points for participation and achievements
        - 🏅 **Badges:** Collect badges for various accomplishments
        - 📊 **Progress Tracking:** Monitor your growth and achievements
        - 🏆 **Department Rankings:** See how your department performs
        
        **Points System:**
        - **50 Points** 🏅 - Register for an event
        - **100 Points** ⭐ - Get shortlisted in an event
        - **200 Points** 🏆 - Win an event
        
        **Features:**
        - 🎯 **Discover Events:** Browse workshops, hackathons, seminars, and more
        - 📝 **Easy Registration:** Register for events with one click
        - 👨‍🏫 **Mentor Guidance:** Get guidance from experienced mentors
        - 🤖 **AI-Powered:** Generate events from text using AI
        - 📊 **Analytics:** Track your event participation
        - 📱 **Mobile-Friendly:** Access from any device
        - 🔐 **Remember Me:** Stay logged in on this device
        
        **User Roles:**
        - **👑 Admin:** Full system control, manage users and events
        - **👨‍🏫 Faculty:** Create and manage events, track registrations
        - **👨‍🏫 Mentor:** Monitor assigned events and student engagement
        - **👨‍🎓 Student:** Browse events, register, and track participation
        
        **Getting Started:**
        1. Select your role from the dropdown
        2. Enter your credentials
        3. Check "Remember Me" to stay logged in
        4. Students can register for new accounts
        5. Start exploring events and earning points!
        """)
    
    # Check for remember me cookie
    check_remember_me_cookie()
    
    # Login Section
    st.markdown("---")
    st.subheader("🔐 Login to Your Account")
    
    # Role selection
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
        
        # Remember me checkbox
        remember_me = st.checkbox("Remember Me", 
                                 help="Stay logged in on this device for 30 days")
        
        # Forgot password link
        col_forgot = st.columns([2, 1])[1]
        with col_forgot:
            if st.button("Forgot Password?", use_container_width=True):
                st.session_state.page = "forgot_password"
                st.rerun()
        
        # Login button
        if st.button("Login", use_container_width=True, type="primary"):
            if not username or not password:
                st.error("Please enter username and password")
            else:
                # Map role to database role
                role_map = {
                    "Admin": "admin",
                    "Faculty": "faculty",
                    "Mentor": "mentor",
                    "Student": "student"
                }
                
                db_role = role_map[role]
                
                if db.verify_credentials(username, password, db_role):
                    # Get user details
                    user = db.get_user(username)
                    if user:
                        st.session_state.role = db_role
                        st.session_state.username = username
                        st.session_state.name = user.get('name', username)
                        st.session_state.session_start = datetime.now()
                        st.session_state.remember_me = remember_me
                        
                        # Set remember me token if requested
                        if remember_me:
                            token = secrets.token_urlsafe(32)
                            expiry = datetime.now() + timedelta(days=30)
                            if db.set_remember_token(username, token, expiry.isoformat()):
                                # Set query parameters for auto-login
                                if hasattr(st, 'query_params'):
                                    # Clear existing params and set new ones
                                    st.query_params.clear()
                                    st.query_params["remember_token"] = token
                                    st.query_params["remember_user"] = username
                        
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("User not found in database")
                else:
                    st.error("Invalid credentials")
        
        # Student registration
        if role == "Student":
            st.markdown("---")
            st.subheader("👨‍🎓 New Student Registration")
            
            if st.button("Create New Student Account", use_container_width=True, type="secondary"):
                st.session_state.page = "student_register"
                st.rerun()



# ============================================
# STUDENT REGISTRATION PAGE
# ============================================
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
            mobile = st.text_input("Mobile Number *", 
                                  placeholder="9876543210",
                                  help="10-digit Indian mobile number")
            username = st.text_input("Username *")
            password = st.text_input("Password *", type="password")
            confirm_pass = st.text_input("Confirm Password *", type="password")
        
        # Terms and conditions
        terms = st.checkbox("I agree to the Terms & Conditions *", value=False)
        
        # Remember me checkbox for registration
        remember_me = st.checkbox("Remember Me on this device", 
                                 value=True,
                                 help="Stay logged in after registration")
        
        col_submit, col_back = st.columns(2)
        with col_submit:
            submit = st.form_submit_button("Register", use_container_width=True, type="primary")
        with col_back:
            back = st.form_submit_button("← Back to Login", use_container_width=True, type="secondary")
        
        if back:
            st.session_state.page = "login"
            st.rerun()
        
        if submit:
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
                        st.success("✅ Registration successful!")
                        
                        # Auto-login after registration
                        st.session_state.role = 'student'
                        st.session_state.username = username
                        st.session_state.name = name
                        st.session_state.session_start = datetime.now()
                        st.session_state.remember_me = remember_me
                        
                        # Set remember me token if requested
                        if remember_me:
                            token = secrets.token_urlsafe(32)
                            expiry = datetime.now() + timedelta(days=30)
                            if db.set_remember_token(username, token, expiry.isoformat()):
                                # Set query parameters for auto-login
                                if hasattr(st, 'query_params'):
                                    # Clear existing params and set new ones
                                    st.query_params.clear()
                                    st.query_params["remember_token"] = token
                                    st.query_params["remember_user"] = username
                        
                        st.balloons()
                        st.info("You have been automatically logged in. Redirecting to dashboard...")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {message}")

# ============================================
# CUSTOM CSS (Updated with leaderboard styles)
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
    
    /* Improved Event Card */
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
    
    /* Leaderboard Card */
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
    
    /* Top 3 Leaderboard Cards */
    .leaderboard-gold {
        background: linear-gradient(135deg, #FFD700 0%, #FFEC8B 100%);
        border: 2px solid #FFD700;
    }
    
    .leaderboard-silver {
        background: linear-gradient(135deg, #C0C0C0 0%, #E8E8E8 100%);
        border: 2px solid #C0C0C0;
    }
    
    .leaderboard-bronze {
        background: linear-gradient(135deg, #CD7F32 0%, #E6B17E 100%);
        border: 2px solid #CD7F32;
    }
    
    /* Badge Styles */
    .badge-gold {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000;
        font-weight: bold;
    }
    
    .badge-silver {
        background: linear-gradient(135deg, #C0C0C0 0%, #FFFFFF 100%);
        color: #000;
        font-weight: bold;
    }
    
    .badge-bronze {
        background: linear-gradient(135deg, #CD7F32 0%, #FFA500 100%);
        color: #000;
        font-weight: bold;
    }
    
    .badge-participant {
        background: linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%);
        color: white;
    }
    
    .badge-shortlisted {
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
        color: white;
    }
    
    .badge-winner {
        background: linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%);
        color: #000;
    }
    
    /* Points Display */
    .points-display {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Progress Bar */
    .progress-container {
        background: #E5E7EB;
        border-radius: 10px;
        height: 20px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        transition: width 0.5s ease;
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
    
    .mentor-badge {
        background: linear-gradient(135deg, #8B5CF6 0%, #A855F7 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
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

    /* Engagement buttons - smaller */
    .engagement-button {
        transition: all 0.2s ease;
        font-size: 0.85rem !important;
        padding: 0.4rem 0.8rem !important;
        min-height: auto !important;
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
    
    /* Mentor info in event card */
    .mentor-info {
        background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%);
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #E5E7EB;
        border-left: 3px solid #8B5CF6;
        font-size: 0.9rem;
    }
    
    /* Smaller buttons */
    .small-button {
        font-size: 0.85rem !important;
        padding: 0.4rem 0.8rem !important;
        min-height: auto !important;
    }
    
    /* Registration buttons */
    .reg-button {
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
        min-height: auto !important;
    }
    
    /* Share button */
    .share-button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
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
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.8rem;
        }
        
        .leaderboard-card {
            padding: 0.75rem;
        }
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
# DASHBOARD FUNCTIONS (Updated with Leaderboard)
# ============================================

def student_dashboard():
    """Student dashboard with leaderboard"""
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    
    # Get student info
    student = db.get_user(st.session_state.username)
    if student:
        st.sidebar.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
        st.sidebar.markdown(f"**Department:** {student.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Year:** {student.get('year', 'N/A')}")
        mobile = student.get('mobile', 'Not provided')
        st.sidebar.markdown(f"**Mobile:** {mobile}")
        
        # Display points in sidebar
        points = db.get_student_points(st.session_state.username)
        st.sidebar.markdown(f"**🏆 Points:** {points}")
        
        # Display rank in sidebar
        rank = db.get_student_rank(st.session_state.username)
        if rank:
            st.sidebar.markdown(f"**📊 Rank:** #{rank}")
    
    display_role_badge('student')
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Events Feed", "My Registrations", "Liked Events", "Interested Events", "Leaderboard", "My Profile", "My Achievements"]
        
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
        registrations = db.get_registrations_by_student(st.session_state.username)
        badges = db.get_student_badges(st.session_state.username)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("❤️ Liked", len(liked_events))
        with col_stat2:
            st.metric("⭐ Interested", len(interested_events))
        
        col_stat3, col_stat4 = st.columns(2)
        with col_stat3:
            st.metric("📝 Registered", len(registrations))
        with col_stat4:
            st.metric("🏅 Badges", len(badges))
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            # Clear remember me token from database
            if st.session_state.username:
                db.clear_reset_token(st.session_state.username)
    
            # Clear query parameters
            if hasattr(st, 'query_params'):
                st.query_params.clear()
    
            # Clear session state
            for key in list(st.session_state.keys()):
                if key != 'rerun_count':
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
            ai_only = st.checkbox("🤖 AI-Generated")
        
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
        total_points = sum(r.get('points_awarded', 0) for r in registrations)
        
        # Display stats
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total", total)
        with col2:
            st.metric("Upcoming", upcoming)
        with col3:
            st.metric("Ongoing", ongoing)
        with col4:
            st.metric("Completed", completed)
        with col5:
            st.metric("Points Earned", total_points)
        
        # Display registrations
        for reg in registrations:
            with st.container():
                st.markdown('<div class="event-card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
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
                    
                    # Points and badges
                    points = reg.get('points_awarded', 0)
                    badges = reg.get('badges_awarded', '')
                    if points > 0:
                        st.caption(f"🏆 Points: {points}")
                    if badges:
                        badge_list = badges.split(',')
                        for badge in badge_list:
                            st.markdown(f'<span style="background: #FFD700; color: #000; padding: 2px 6px; border-radius: 10px; font-size: 0.7rem; margin-right: 4px;">{badge}</span>', unsafe_allow_html=True)
                
                with col2:
                    # Event status
                    event_status = reg.get('event_status', 'unknown')
                    if event_status == 'upcoming':
                        st.success("🟢 Upcoming")
                    elif event_status == 'ongoing':
                        st.warning("🟡 Ongoing")
                    else:
                        st.error("🔴 Completed")
                
                with col3:
                    # Points display
                    if points > 0:
                        st.markdown(f'<div style="font-size: 1.5rem; font-weight: bold; text-align: center; color: #3B82F6;">{points}</div>', unsafe_allow_html=True)
                        st.caption("Points")
                
                st.markdown('</div>', unsafe_allow_html=True)

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
    
    elif selected == "Leaderboard":
        st.markdown('<h1 class="main-header">🏆 College Leaderboard</h1>', unsafe_allow_html=True)
        
        # Leaderboard tabs
        tab1, tab2, tab3 = st.tabs(["🏆 Overall Leaderboard", "🎖️ Department Rankings", "📈 My Progress"])
        
        with tab1:
            # Overall leaderboard
            st.subheader("Top 15 Students")
            leaderboard = db.get_leaderboard(limit=15)
            
            if leaderboard:
                display_leaderboard(leaderboard, "🏆 College Leaderboard")
            else:
                st.info("No students found in leaderboard.")
        
        with tab2:
            # Department rankings
            st.subheader("🎖️ Top 5 Per Department")
            dept_leaderboard = db.get_department_leaderboard(limit=5)
            
            if dept_leaderboard:
                # Group by department
                departments = COLLEGE_CONFIG['departments']
                
                for dept in departments:
                    dept_students = [s for s in dept_leaderboard if s.get('department') == dept]
                    if dept_students:
                        st.markdown(f"### {dept}")
                        display_leaderboard(dept_students[:5], f"🏆 {dept}")
            else:
                st.info("No department rankings available yet.")
        
        with tab3:
            # My progress
            display_student_gamification_profile(st.session_state.username)
    
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
            st.markdown("### Contact Information")
            st.markdown(f"**Email:** {student.get('email', 'N/A')}")
            mobile = student.get('mobile', 'Not provided')
            st.markdown(f"**Mobile:** {mobile}")
            st.markdown(f"**Username:** {student.get('username', 'N/A')}")
            st.markdown(f"**Member Since:** {format_date(student.get('created_at'))}")
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 My Statistics")
        
        registrations = db.get_registrations_by_student(st.session_state.username) or []
        badges = db.get_student_badges(st.session_state.username) or []
        points = db.get_student_points(st.session_state.username)
        rank = db.get_student_rank(st.session_state.username)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Events Registered", len(registrations))
        with col_stat2:
            attended = len([r for r in registrations if r.get('attendance') == 'present'])
            st.metric("Events Attended", attended)
        with col_stat3:
            st.metric("🏅 Badges Earned", len(badges))
        with col_stat4:
            if rank:
                st.metric("📊 Rank", f"#{rank}")
            else:
                st.metric("📊 Rank", "Not Ranked")
        
        # Points display
        st.markdown(f'<div class="points-display">{points} Points</div>', unsafe_allow_html=True)
    
    elif selected == "My Achievements":
        display_student_gamification_profile(st.session_state.username)

def mentor_dashboard():
    """Mentor dashboard with gamification features"""
    st.sidebar.title("👨‍🏫 Mentor Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    
    # Get mentor info
    mentor = db.get_mentor_by_email(st.session_state.username)
    if mentor:
        st.sidebar.markdown(f"**Department:** {mentor.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Email:** {mentor.get('email', 'N/A')}")
        st.sidebar.markdown(f"**Contact:** {mentor.get('contact', 'N/A')}")
        if mentor.get('expertise'):
            st.sidebar.markdown(f"**Expertise:** {mentor.get('expertise', 'N/A')}")
    
    display_role_badge('mentor')
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["My Events", "Student Engagement", "Award Points", "My Profile"]
        
        if 'mentor_page' not in st.session_state:
            st.session_state.mentor_page = "My Events"
        
        for option in nav_options:
            is_active = st.session_state.mentor_page == option
            button_class = "active" if is_active else ""
            button_text = f"▶ {option}" if is_active else option
            
            if st.button(button_text, key=f"mentor_{option}", use_container_width=True):
                st.session_state.mentor_page = option
                st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            # Clear remember me token from database
            if st.session_state.username:
                db.clear_reset_token(st.session_state.username)
    
            # Clear query parameters
            if hasattr(st, 'query_params'):
                st.query_params.clear()
    
            # Clear session state
            for key in list(st.session_state.keys()):
                if key != 'rerun_count':
                    del st.session_state[key]
            st.rerun()
    
    # Page content
    selected = st.session_state.mentor_page
    
    if selected == "My Events":
        st.markdown('<h1 class="main-header">📅 My Assigned Events</h1>', unsafe_allow_html=True)
        
        # Get mentor ID
        mentor = db.get_mentor_by_email(st.session_state.username)
        if not mentor:
            st.error("Mentor profile not found!")
            return
        
        mentor_id = mentor['id']
        
        # Get events assigned to this mentor
        events = db.get_events_by_mentor(mentor_id)
        
        if not events:
            st.info("No events assigned to you yet. Events will appear here when admin assigns them.")
            return
        
        # Statistics
        total = len(events)
        upcoming = len([e for e in events if e.get('status') == 'upcoming'])
        ongoing = len([e for e in events if e.get('status') == 'ongoing'])
        past = len([e for e in events if e.get('status') == 'past'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", total)
        with col2:
            st.metric("Upcoming", upcoming)
        with col3:
            st.metric("Ongoing", ongoing)
        with col4:
            st.metric("Past", past)
        
        # Display events
        st.subheader("📋 Event Details")
        for event in events:
            display_event_card(event, None)
    
    elif selected == "Student Engagement":
        st.markdown('<h1 class="main-header">📊 Student Engagement</h1>', unsafe_allow_html=True)
        
        # Get mentor ID
        mentor = db.get_mentor_by_email(st.session_state.username)
        if not mentor:
            st.error("Mentor profile not found!")
            return
        
        mentor_id = mentor['id']
        
        # Get events assigned to this mentor
        events = db.get_events_by_mentor(mentor_id)
        
        if not events:
            st.info("No events assigned to monitor engagement.")
            return
        
        # Select event to view engagement
        event_options = {e['title']: e['id'] for e in events}
        selected_event_title = st.selectbox("Select Event", list(event_options.keys()))
        
        if selected_event_title:
            event_id = event_options[selected_event_title]
            selected_event = next(e for e in events if e['id'] == event_id)
            
            # Get engagement data
            likes_count = db.get_event_likes_count(event_id)
            interested_count = db.get_event_interested_count(event_id)
            registrations = db.get_registrations_by_event(event_id)
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Likes", likes_count)
            with col2:
                st.metric("Interested", interested_count)
            with col3:
                st.metric("Registrations", len(registrations))
            
            # Display registrations
            st.subheader("📋 Registered Students")
            if registrations:
                df_data = []
                for reg in registrations:
                    df_data.append({
                        'Student Name': reg.get('student_name'),
                        'Roll No': reg.get('student_roll', 'N/A'),
                        'Department': reg.get('department', 'N/A'),
                        'Mobile': reg.get('mobile', 'N/A'),
                        'Status': reg.get('status', 'pending').title(),
                        'Points': reg.get('points_awarded', 0),
                        'Badges': reg.get('badges_awarded', ''),
                        'Registered On': format_date(reg.get('registered_at'))
                    })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("No students have registered for this event yet.")
    
    elif selected == "Award Points":
        st.markdown('<h1 class="main-header">🎮 Award Points & Badges</h1>', unsafe_allow_html=True)
        
        # Get mentor ID
        mentor = db.get_mentor_by_email(st.session_state.username)
        if not mentor:
            st.error("Mentor profile not found!")
            return
        
        mentor_id = mentor['id']
        
        # Get events assigned to this mentor
        events = db.get_events_by_mentor(mentor_id)
        
        if not events:
            st.info("No events assigned to you. You can only award points for your assigned events.")
            return
        
        # Select event
        event_options = {e['title']: e['id'] for e in events}
        selected_event_title = st.selectbox("Select Event", list(event_options.keys()))
        
        if selected_event_title:
            event_id = event_options[selected_event_title]
            selected_event = next(e for e in events if e['id'] == event_id)
            
            # Get registrations for the selected event
            registrations = db.get_registrations_by_event(event_id)
            
            if not registrations:
                st.info("No students have registered for this event yet.")
                return
            
            # Award points section
            st.subheader("🎯 Award Points to Students")
            
            # Select student
            student_options = {r['student_name']: r['student_username'] for r in registrations}
            selected_student_name = st.selectbox("Select Student", list(student_options.keys()))
            
            if selected_student_name:
                student_username = student_options[selected_student_name]
                student = db.get_user(student_username)
                
                if student:
                    # Display student info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Student:** {student.get('name')}")
                        st.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
                    with col2:
                        current_points = db.get_student_points(student_username)
                        st.metric("Current Points", current_points)
                    
                    # Award options
                    st.subheader("Select Achievement")
                    
                    achievement_col1, achievement_col2, achievement_col3 = st.columns(3)
                    
                    with achievement_col1:
                        if st.button("🏅 Registration", use_container_width=True, type="primary"):
                            success, message = db.award_points_and_badge(
                                student_username,
                                event_id,
                                selected_event['title'],
                                'registration',
                                st.session_state.username
                            )
                            if success:
                                st.success(f"Awarded 50 points and Participant badge!")
                                st.rerun()
                            else:
                                st.error(f"Failed to award points: {message}")
                    
                    with achievement_col2:
                        if st.button("⭐ Shortlisted", use_container_width=True, type="primary"):
                            success, message = db.award_points_and_badge(
                                student_username,
                                event_id,
                                selected_event['title'],
                                'shortlisted',
                                st.session_state.username
                            )
                            if success:
                                st.success(f"Awarded 100 points and Shortlisted badge!")
                                st.rerun()
                            else:
                                st.error(f"Failed to award points: {message}")
                    
                    with achievement_col3:
                        if st.button("🏆 Winner", use_container_width=True, type="primary"):
                            success, message = db.award_points_and_badge(
                                student_username,
                                event_id,
                                selected_event['title'],
                                'winner',
                                st.session_state.username
                            )
                            if success:
                                st.success(f"Awarded 200 points and Winner badge!")
                                st.rerun()
                            else:
                                st.error(f"Failed to award points: {message}")
                    
                    # Custom award option
                    st.markdown("---")
                    st.subheader("Custom Award")
                    
                    custom_col1, custom_col2 = st.columns(2)
                    with custom_col1:
                        custom_points = st.number_input("Points to Award", min_value=1, max_value=500, value=50)
                        custom_reason = st.text_input("Reason for Award", placeholder="Excellent participation, Outstanding performance...")
                    
                    with custom_col2:
                        custom_badge = st.text_input("Badge Name (Optional)", placeholder="Mentor's Choice, Most Creative...")
                        custom_badge_type = st.selectbox("Badge Type", ["custom", "recognition", "achievement"])
                    
                    if st.button("🎖️ Award Custom Points", use_container_width=True, type="secondary"):
                        if custom_points > 0:
                            # Update student's points
                            current_points = db.get_student_points(student_username)
                            new_points = current_points + custom_points
                            
                            # Update points
                            update_data = {
                                'total_points': new_points,
                                'last_points_update': datetime.now().isoformat(),
                                'updated_at': datetime.now().isoformat()
                            }
                            
                            if db.use_supabase:
                                db.client.update('users', {'username': student_username}, update_data)
                            else:
                                db.client.update('users', {'username': student_username}, update_data)
                            
                            # Record points history
                            points_record = {
                                'id': str(uuid.uuid4()),
                                'student_username': student_username,
                                'points': custom_points,
                                'reason': custom_reason or f"Custom award from mentor",
                                'event_id': event_id,
                                'event_title': selected_event['title'],
                                'awarded_by': st.session_state.username,
                                'awarded_at': datetime.now().isoformat()
                            }
                            
                            if db.use_supabase:
                                db.client.insert('points_history', points_record)
                            else:
                                db.client.insert('points_history', points_record)
                            
                            # Award custom badge if specified
                            if custom_badge:
                                badge_record = {
                                    'id': str(uuid.uuid4()),
                                    'student_username': student_username,
                                    'badge_name': custom_badge,
                                    'badge_type': custom_badge_type,
                                    'awarded_for': custom_reason or f"Custom award from mentor",
                                    'event_id': event_id,
                                    'event_title': selected_event['title'],
                                    'awarded_by': st.session_state.username,
                                    'awarded_at': datetime.now().isoformat()
                                }
                                
                                if db.use_supabase:
                                    db.client.insert('student_badges', badge_record)
                                else:
                                    db.client.insert('student_badges', badge_record)
                            
                            st.success(f"Awarded {custom_points} points to {student.get('name')}!")
                            st.rerun()
    
    elif selected == "My Profile":
        st.header("👤 My Profile")
        
        mentor = db.get_mentor_by_email(st.session_state.username)
        
        if not mentor:
            st.error("Mentor profile not found!")
            return
        
        # Profile display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Personal Information")
            st.markdown(f"**Full Name:** {mentor.get('full_name', 'N/A')}")
            st.markdown(f"**Department:** {mentor.get('department', 'N/A')}")
            st.markdown(f"**Email:** {mentor.get('email', 'N/A')}")
            st.markdown(f"**Contact:** {mentor.get('contact', 'N/A')}")
        
        with col2:
            st.markdown("### Professional Information")
            if mentor.get('expertise'):
                st.markdown(f"**Expertise:** {mentor.get('expertise', 'N/A')}")
            st.markdown(f"**Status:** {'Active' if mentor.get('is_active') else 'Inactive'}")
            st.markdown(f"**Member Since:** {format_date(mentor.get('created_at'))}")
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 My Statistics")
        
        mentor_id = mentor['id']
        events = db.get_events_by_mentor(mentor_id)
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Events", len(events))
        with col_stat2:
            upcoming = len([e for e in events if e.get('status') == 'upcoming'])
            st.metric("Upcoming", upcoming)
        with col_stat3:
            past = len([e for e in events if e.get('status') == 'past'])
            st.metric("Past", past)

def faculty_dashboard():
    """Faculty coordinator dashboard"""
    st.sidebar.title("👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('faculty')
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Dashboard", "Create Event", "My Events", "Registrations", "Leaderboard"]
        
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
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            # Clear remember me token from database
            if st.session_state.username:
                db.clear_reset_token(st.session_state.username)
    
            # Clear query parameters
            if hasattr(st, 'query_params'):
                st.query_params.clear()
    
            # Clear session state
            for key in list(st.session_state.keys()):
                if key != 'rerun_count':
                    del st.session_state[key]
            st.rerun()
    
    # Page content
    selected = st.session_state.faculty_page
    
    if selected == "Dashboard":
        st.markdown('<h1 class="main-header">Faculty Dashboard</h1>', unsafe_allow_html=True)
        
        # Statistics
        events = db.get_events_by_creator(st.session_state.username)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("My Events", len(events))
        with col2:
            upcoming = len([e for e in events if e.get('status') == 'upcoming'])
            st.metric("Upcoming", upcoming)
        with col3:
            total_points = sum(db.get_event_likes_count(e['id']) + db.get_event_interested_count(e['id']) for e in events)
            st.metric("Total Engagement", total_points)
        
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
                    event_link = st.text_input("Event Website/URL", 
                                             placeholder="https://example.com/event-details")
                    registration_link = st.text_input("Registration Link", 
                                                    placeholder="https://forms.google.com/registration")
                    
                    # Mentor selection
                    st.subheader("👨‍🏫 Assign Mentor (Optional)")
                    active_mentors = db.get_active_mentors()
                    if active_mentors:
                        mentor_options = ["None"] + [f"{m['full_name']} ({m['department']})" for m in active_mentors]
                        selected_mentor = st.selectbox("Select Mentor", mentor_options)
                    else:
                        st.info("No active mentors available. Admin can add mentors.")
                        selected_mentor = "None"
                    
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
                        # Get mentor ID if selected
                        mentor_id = None
                        if selected_mentor != "None" and active_mentors:
                            mentor_name = selected_mentor.split(" (")[0]
                            mentor = next((m for m in active_mentors if m['full_name'] == mentor_name), None)
                            if mentor:
                                mentor_id = mentor['id']
                        
                        # Save flyer
                        flyer_path = save_flyer_image(flyer)
                        
                        # Combine date and time
                        event_datetime = datetime.combine(event_date, event_time)
                        
                        event_data = {
                            'title': title,
                            'description': description,
                            'event_type': event_type,
                            'event_date': event_datetime.isoformat(),
                            'venue': venue,
                            'organizer': organizer,
                            'event_link': event_link,
                            'registration_link': registration_link,
                            'max_participants': max_participants,
                            'flyer_path': flyer_path,
                            'created_by': st.session_state.username,
                            'created_by_name': st.session_state.name,
                            'ai_generated': False,
                            'mentor_id': mentor_id
                        }
                        
                        if db.add_event(event_data):
                            st.success(f"Event '{title}' created successfully! 🎉")
                            if mentor_id:
                                st.info(f"✅ Mentor assigned: {selected_mentor}")
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
                        ai_event_link = st.text_input("Event Website", value=event_data.get('event_link', ''))
                        ai_reg_link = st.text_input("Registration Link", value=event_data.get('registration_link', ''))
                        
                        # Mentor selection for AI-generated events
                        st.subheader("👨‍🏫 Assign Mentor (Optional)")
                        active_mentors = db.get_active_mentors()
                        if active_mentors:
                            mentor_options = ["None"] + [f"{m['full_name']} ({m['department']})" for m in active_mentors]
                            ai_selected_mentor = st.selectbox("Select Mentor", mentor_options, key="ai_mentor_select")
                        else:
                            st.info("No active mentors available. Admin can add mentors.")
                            ai_selected_mentor = "None"
                    
                    ai_description = st.text_area("Event Description", 
                                                value=event_data.get('description', ''),
                                                height=150)
                    
                    # Flyer upload for AI events
                    st.subheader("Event Flyer (Optional)")
                    ai_flyer = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png', 'gif', 'webp'], key="ai_flyer")
                    if ai_flyer:
                        st.image(ai_flyer, width=200)
                    
                    ai_submit = st.form_submit_button("✅ Create AI-Generated Event", use_container_width=True)
                    
                    if ai_submit:
                        if not all([ai_title, ai_venue, ai_organizer, ai_description]):
                            st.error("Please fill all required fields (*)")
                        else:
                            # Get mentor ID if selected
                            ai_mentor_id = None
                            if ai_selected_mentor != "None" and active_mentors:
                                mentor_name = ai_selected_mentor.split(" (")[0]
                                mentor = next((m for m in active_mentors if m['full_name'] == mentor_name), None)
                                if mentor:
                                    ai_mentor_id = mentor['id']
                            
                            # Save flyer
                            flyer_path = save_flyer_image(ai_flyer)
                            
                            # Combine date and time
                            event_datetime = datetime.combine(ai_date, ai_time)
                            
                            final_event_data = {
                                'title': ai_title,
                                'description': ai_description,
                                'event_type': ai_event_type,
                                'event_date': event_datetime.isoformat(),
                                'venue': ai_venue,
                                'organizer': ai_organizer,
                                'event_link': ai_event_link,
                                'registration_link': ai_reg_link,
                                'max_participants': ai_max_participants,
                                'flyer_path': flyer_path,
                                'created_by': st.session_state.username,
                                'created_by_name': st.session_state.name,
                                'ai_generated': True,
                                'ai_metadata': event_data.get('ai_metadata', {}),
                                'mentor_id': ai_mentor_id
                            }
                            
                            if db.add_event(final_event_data):
                                st.success(f"✅ AI-generated event '{ai_title}' created successfully! 🎉")
                                if ai_mentor_id:
                                    st.info(f"✅ Mentor assigned: {ai_selected_mentor}")
                                
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
                # Display registrations in a table with mobile numbers
                df_data = []
                for reg in registrations:
                    df_data.append({
                        'Student Name': reg.get('student_name'),
                        'Roll No': reg.get('student_roll', 'N/A'),
                        'Mobile': reg.get('mobile', 'N/A'),
                        'Department': reg.get('department', 'N/A'),
                        'Year': reg.get('year', 'N/A'),
                        'Status': reg.get('status', 'pending').title(),
                        'Points': reg.get('points_awarded', 0),
                        'Badges': reg.get('badges_awarded', ''),
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
            else:
                st.info("No registrations for this event yet.")
    
    elif selected == "Leaderboard":
        st.markdown('<h1 class="main-header">🏆 College Leaderboard</h1>', unsafe_allow_html=True)
        
        # Overall leaderboard
        st.subheader("Top 15 Students")
        leaderboard = db.get_leaderboard(limit=15)
        
        if leaderboard:
            display_leaderboard(leaderboard, "🏆 College Leaderboard")
        else:
            st.info("No students found in leaderboard.")

def admin_dashboard():
    """Admin dashboard with gamification analytics"""
    st.sidebar.title("👑 Admin Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('admin')
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Dashboard", "Manage Events", "Manage Users", "Manage Mentors", "Gamification Analytics"]
        
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
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            # Clear remember me token from database
            if st.session_state.username:
                db.clear_reset_token(st.session_state.username)
    
            # Clear query parameters
            if hasattr(st, 'query_params'):
                st.query_params.clear()
    
            # Clear session state
            for key in list(st.session_state.keys()):
                if key != 'rerun_count':
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
        mentors = db.get_all_mentors()
        stats = db.get_system_stats()
        
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
            active_mentors = len([m for m in mentors if m.get('is_active')])
            st.metric("👨‍🏫 Active Mentors", active_mentors)
        
        # Gamification statistics
        st.markdown("---")
        st.subheader("🎮 Gamification Statistics")
        
        total_points = stats.get('total_points', 0)
        total_badges = stats.get('total_badges', 0)
        top_students = stats.get('top_students', [])
        
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.metric("🏆 Total Points", total_points)
        with col_g2:
            st.metric("🏅 Total Badges", total_badges)
        with col_g3:
            st.metric("📊 Active Students", len(top_students) if top_students else 0)
        
        # Top 3 students
        if top_students:
            st.subheader("🏆 Top 3 Students")
            cols = st.columns(3)
            medals = ["🥇", "🥈", "🥉"]
            
            for i, student in enumerate(top_students[:3]):
                with cols[i]:
                    points = student.get('total_points', 0)
                    st.markdown(f'''
                    <div style="background: {'linear-gradient(135deg, #FFD700 0%, #FFEC8B 100%)' if i == 0 else 'linear-gradient(135deg, #C0C0C0 0%, #E8E8E8 100%)' if i == 1 else 'linear-gradient(135deg, #CD7F32 0%, #E6B17E 100%)'}; 
                                padding: 1.5rem; border-radius: 15px; text-align: center; 
                                margin: 0.5rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                        <div style="font-size: 3rem;">{medals[i]}</div>
                        <div style="font-size: 1.5rem; font-weight: bold; margin: 1rem 0;">{student.get('name')}</div>
                        <div style="font-size: 0.9rem; color: #666;">{student.get('department', '')}</div>
                        <div style="font-size: 2.5rem; font-weight: bold; color: #1E3A8A; margin: 1rem 0;">{points}</div>
                        <div style="font-size: 0.9rem;">Points</div>
                    </div>
                    ''', unsafe_allow_html=True)
        
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
                            if db.use_supabase:
                                # Delete registrations first
                                db.client.delete('registrations', {'event_id': event['id']})
                                # Delete likes
                                db.client.delete('event_likes', {'event_id': event['id']})
                                # Delete interested
                                db.client.delete('event_interested', {'event_id': event['id']})
                                # Delete event
                                success = db.client.delete('events', {'id': event['id']})
                            else:
                                cursor = db.client.conn.cursor()
                                try:
                                    # First delete registrations
                                    cursor.execute("DELETE FROM registrations WHERE event_id = ?", (event['id'],))
                                    # Delete likes
                                    cursor.execute("DELETE FROM event_likes WHERE event_id = ?", (event['id'],))
                                    # Delete interested
                                    cursor.execute("DELETE FROM event_interested WHERE event_id = ?", (event['id'],))
                                    # Then delete event
                                    cursor.execute("DELETE FROM events WHERE id = ?", (event['id'],))
                                    db.client.conn.commit()
                                    success = cursor.rowcount > 0
                                except Exception as e:
                                    st.error(f"Error deleting event: {e}")
                                    success = False
                            
                            if success:
                                st.success("Event deleted successfully!")
                                st.rerun()
        else:
            st.info("No events found.")
    
    elif selected == "Manage Users":
        st.header("👥 Manage Users")
        
        # Get all users
        if db.use_supabase:
            users = db.client.select('users')
        else:
            cursor = db.client.conn.cursor()
            cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
            users = [dict(row) for row in cursor.fetchall()]
        
        if users:
            # Display user statistics
            admin_count = len([u for u in users if u['role'] == 'admin'])
            faculty_count = len([u for u in users if u['role'] == 'faculty'])
            student_count = len([u for u in users if u['role'] == 'student'])
            mentor_count = len([u for u in users if u['role'] == 'mentor'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Admins", admin_count)
            with col2:
                st.metric("Faculty", faculty_count)
            with col3:
                st.metric("Students", student_count)
            with col4:
                st.metric("Mentors", mentor_count)
            
            # User table with points
            df_data = []
            for user in users:
                df_data.append({
                    'Name': user.get('name'),
                    'Username': user.get('username'),
                    'Role': user.get('role').title(),
                    'Department': user.get('department', 'N/A'),
                    'Roll No': user.get('roll_no', 'N/A'),
                    'Mobile': user.get('mobile', 'N/A'),
                    'Points': user.get('total_points', 0) if user.get('role') == 'student' else 'N/A',
                    'Created': format_date(user.get('created_at')),
                    'Status': 'Active' if user.get('is_active') else 'Inactive'
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
                            default_pass = hashlib.sha256('password123'.encode()).hexdigest()
                            if db.use_supabase:
                                success = db.client.update('users', {'id': user_id}, {'password': default_pass})
                            else:
                                cursor = db.client.conn.cursor()
                                cursor.execute("UPDATE users SET password = ? WHERE id = ?", (default_pass, user_id))
                                db.client.conn.commit()
                                success = cursor.rowcount > 0
                            
                            if success:
                                st.success("Password reset to 'password123'")
                    with col_act2:
                        if st.button("Delete User", use_container_width=True, type="secondary"):
                            # Get user info
                            selected_user_data = next(u for u in users if u['id'] == user_id)
                            
                            # Don't allow deleting default admin and faculty
                            if selected_user_data['username'] in ['admin@raisoni', 'faculty@raisoni']:
                                st.error("Cannot delete default admin/faculty accounts")
                            else:
                                if db.use_supabase:
                                    # Delete user
                                    success = db.client.delete('users', {'id': user_id})
                                    # If it's a mentor, also delete from mentors table
                                    if selected_user_data['role'] == 'mentor':
                                        db.client.delete('mentors', {'email': selected_user_data['username']})
                                else:
                                    cursor = db.client.conn.cursor()
                                    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                                    # If it's a mentor, also delete from mentors table
                                    if selected_user_data['role'] == 'mentor':
                                        cursor.execute("DELETE FROM mentors WHERE email = ?", (selected_user_data['username'],))
                                    db.client.conn.commit()
                                    success = cursor.rowcount > 0
                                
                                if success:
                                    st.success("User deleted successfully!")
                                    st.rerun()
        else:
            st.info("No users found.")
    
    elif selected == "Manage Mentors":
        st.header("👨‍🏫 Manage Mentors")
        
        tab1, tab2, tab3 = st.tabs(["Add New Mentor", "View All Mentors", "Assign to Events"])
        
        with tab1:
            st.subheader("➕ Add New Mentor")
            
            with st.form("add_mentor_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    first_name = st.text_input("First Name *")
                    last_name = st.text_input("Last Name *")
                    department = st.selectbox("Department *", COLLEGE_CONFIG['departments'])
                    email = st.text_input("Email *", help="This will be the username for login")
                
                with col2:
                    contact = st.text_input("Contact Number *")
                    expertise = st.text_area("Expertise/Areas", placeholder="Python, Machine Learning, Web Development...")
                    is_active = st.checkbox("Active", value=True)
                    
                    # Password options
                    password_option = st.radio("Password", ["Auto-generate", "Custom"])
                    if password_option == "Custom":
                        custom_password = st.text_input("Set Password", type="password")
                    else:
                        custom_password = None
                
                submit = st.form_submit_button("Add Mentor", use_container_width=True, type="primary")
                
                if submit:
                    if not all([first_name, last_name, department, email, contact]):
                        st.error("Please fill all required fields (*)")
                    else:
                        mentor_data = {
                            'first_name': first_name,
                            'last_name': last_name,
                            'department': department,
                            'email': email,
                            'contact': contact,
                            'expertise': expertise,
                            'is_active': is_active,
                            'created_by': st.session_state.username
                        }
                        
                        # Add custom password if provided
                        if custom_password:
                            mentor_data['password'] = custom_password
                        
                        success, result = db.add_mentor(mentor_data)
                        if success:
                            password = result
                            st.success(f"✅ Mentor {first_name} {last_name} added successfully!")
                            st.info(f"**Login credentials:**\nUsername: {email}\nPassword: {password}")
                            st.warning("⚠️ Please save this password securely. It won't be shown again.")
                            st.rerun()
                        else:
                            st.error(f"Failed to add mentor: {result}")
        
        with tab2:
            st.subheader("📋 All Mentors")
            
            mentors = db.get_all_mentors()
            
            if not mentors:
                st.info("No mentors found. Add your first mentor!")
                return
            
            # Search and filter
            col_search, col_filter = st.columns(2)
            with col_search:
                search_term = st.text_input("🔍 Search mentors", placeholder="Search by name, department...")
            
            with col_filter:
                show_active = st.selectbox("Status", ["All", "Active Only", "Inactive Only"])
            
            # Filter mentors
            filtered_mentors = mentors
            if search_term:
                search_term = search_term.lower()
                filtered_mentors = [m for m in filtered_mentors 
                                  if search_term in m.get('full_name', '').lower() or 
                                  search_term in m.get('department', '').lower() or
                                  search_term in m.get('expertise', '').lower()]
            
            if show_active == "Active Only":
                filtered_mentors = [m for m in filtered_mentors if m.get('is_active')]
            elif show_active == "Inactive Only":
                filtered_mentors = [m for m in filtered_mentors if not m.get('is_active')]
            
            # Display mentors
            st.caption(f"Found {len(filtered_mentors)} mentors")
            
            for mentor in filtered_mentors:
                with st.container():
                    st.markdown('<div class="event-card">', unsafe_allow_html=True)
                    
                    col_info, col_actions = st.columns([3, 1])
                    
                    with col_info:
                        # Mentor status badge
                        status_color = "🟢" if mentor.get('is_active') else "🔴"
                        status_text = "Active" if mentor.get('is_active') else "Inactive"
                        
                        st.markdown(f'<div class="card-title">{mentor.get("full_name")} {status_color}</div>', unsafe_allow_html=True)
                        st.caption(f"**Department:** {mentor.get('department')}")
                        st.caption(f"**Email:** {mentor.get('email')}")
                        st.caption(f"**Contact:** {mentor.get('contact')}")
                        
                        if mentor.get('expertise'):
                            st.caption(f"**Expertise:** {mentor.get('expertise')}")
                    
                    with col_actions:
                        st.markdown("### Actions")
                        
                        # Edit button
                        if st.button("✏️ Edit", key=f"edit_{mentor['id']}", use_container_width=True):
                            st.session_state.editing_mentor = mentor['id']
                            st.rerun()
                        
                        # Delete/Activate button
                        if mentor.get('is_active'):
                            if st.button("❌ Deactivate", key=f"deact_{mentor['id']}", use_container_width=True, type="secondary"):
                                if db.delete_mentor(mentor['id']):
                                    st.success("Mentor deactivated!")
                                    st.rerun()
                        else:
                            if st.button("✅ Activate", key=f"act_{mentor['id']}", use_container_width=True, type="secondary"):
                                if db.update_mentor(mentor['id'], {'is_active': True}):
                                    st.success("Mentor activated!")
                                    st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Edit mentor form (appears when editing)
            if 'editing_mentor' in st.session_state:
                mentor_id = st.session_state.editing_mentor
                mentor = db.get_mentor_by_id(mentor_id)
                
                if mentor:
                    st.markdown("---")
                    st.subheader("✏️ Edit Mentor")
                    
                    with st.form("edit_mentor_form"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            edit_first_name = st.text_input("First Name", value=mentor.get('first_name', ''))
                            edit_last_name = st.text_input("Last Name", value=mentor.get('last_name', ''))
                            edit_department = st.selectbox("Department", COLLEGE_CONFIG['departments'], 
                                                         index=COLLEGE_CONFIG['departments'].index(mentor.get('department', '')) 
                                                         if mentor.get('department') in COLLEGE_CONFIG['departments'] else 0)
                        
                        with col2:
                            edit_email = st.text_input("Email", value=mentor.get('email', ''))
                            edit_contact = st.text_input("Contact", value=mentor.get('contact', ''))
                            edit_expertise = st.text_area("Expertise", value=mentor.get('expertise', ''))
                            edit_active = st.checkbox("Active", value=bool(mentor.get('is_active', True)))
                        
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            save = st.form_submit_button("💾 Save Changes", use_container_width=True, type="primary")
                        with col_cancel:
                            cancel = st.form_submit_button("❌ Cancel", use_container_width=True, type="secondary")
                        
                        if save:
                            update_data = {
                                'first_name': edit_first_name,
                                'last_name': edit_last_name,
                                'department': edit_department,
                                'email': edit_email,
                                'contact': edit_contact,
                                'expertise': edit_expertise,
                                'is_active': edit_active
                            }
                            
                            if db.update_mentor(mentor_id, update_data):
                                st.success("✅ Mentor updated successfully!")
                                del st.session_state.editing_mentor
                                st.rerun()
                            else:
                                st.error("Failed to update mentor.")
                        
                        if cancel:
                            del st.session_state.editing_mentor
                            st.rerun()
        
        with tab3:
            st.subheader("📅 Assign Mentors to Events")
            
            # Get all active mentors
            active_mentors = db.get_active_mentors()
            if not active_mentors:
                st.info("No active mentors available. Please add mentors first.")
                return
            
            # Get all events without mentors
            events = db.get_all_events()
            events_without_mentors = [e for e in events if not e.get('mentor_id')]
            
            if not events_without_mentors:
                st.success("🎉 All events have mentors assigned!")
                st.info("To reassign mentors, go to Faculty dashboard.")
                return
            
            # Select event to assign mentor
            event_options = {f"{e['title']} ({format_date(e['event_date'])})": e['id'] for e in events_without_mentors}
            selected_event_label = st.selectbox("Select Event (without mentor)", list(event_options.keys()))
            
            if selected_event_label:
                event_id = event_options[selected_event_label]
                selected_event = next(e for e in events_without_mentors if e['id'] == event_id)
                
                # Display event details
                st.markdown(f"**Selected Event:** {selected_event['title']}")
                st.caption(f"Date: {format_date(selected_event['event_date'])}")
                st.caption(f"Type: {selected_event.get('event_type', 'N/A')}")
                st.caption(f"Venue: {selected_event.get('venue', 'N/A')}")
                
                # Select mentor
                mentor_options = {f"{m['full_name']} ({m['department']})": m['id'] for m in active_mentors}
                selected_mentor_label = st.selectbox("Select Mentor", list(mentor_options.keys()))
                
                if selected_mentor_label:
                    mentor_id = mentor_options[selected_mentor_label]
                    selected_mentor = next(m for m in active_mentors if m['id'] == mentor_id)
                    
                    # Display mentor details
                    st.markdown("**Selected Mentor Details:**")
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.caption(f"Name: {selected_mentor['full_name']}")
                        st.caption(f"Department: {selected_mentor['department']}")
                    with col_m2:
                        st.caption(f"Email: {selected_mentor['email']}")
                        st.caption(f"Contact: {selected_mentor['contact']}")
                    
                    if selected_mentor.get('expertise'):
                        st.caption(f"Expertise: {selected_mentor['expertise']}")
                    
                    # Assign button
                    if st.button("✅ Assign Mentor to Event", use_container_width=True, type="primary"):
                        if db.assign_mentor_to_event(event_id, mentor_id):
                            st.success(f"✅ {selected_mentor['full_name']} assigned to '{selected_event['title']}'!")
                            st.rerun()
                        else:
                            st.error("Failed to assign mentor.")
    
    elif selected == "Gamification Analytics":
        st.markdown('<h1 class="main-header">📊 Gamification Analytics</h1>', unsafe_allow_html=True)
        
        # Overall statistics
        stats = db.get_system_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏆 Total Points", stats.get('total_points', 0))
        with col2:
            st.metric("🏅 Total Badges", stats.get('total_badges', 0))
        with col3:
            student_count = stats.get('user_counts', {}).get('student', 0)
            st.metric("👨‍🎓 Students", student_count)
        with col4:
            active_students = len(db.get_leaderboard(limit=1000))
            st.metric("📊 Active Students", active_students)
        
        # Leaderboard
        st.subheader("🏆 Top 15 Leaderboard")
        leaderboard = db.get_leaderboard(limit=15)
        
        if leaderboard:
            display_leaderboard(leaderboard, "🏆 College Leaderboard")
        else:
            st.info("No students in leaderboard yet.")
        
        # Department-wise statistics
        st.subheader("🎖️ Department Performance")
        departments = COLLEGE_CONFIG['departments']
        
        dept_stats = []
        for dept in departments:
            dept_students = db.get_leaderboard(limit=1000, department=dept)
            if dept_students:
                total_points = sum(s.get('total_points', 0) for s in dept_students)
                avg_points = total_points / len(dept_students) if dept_students else 0
                top_student = dept_students[0] if dept_students else None
                
                dept_stats.append({
                    'Department': dept,
                    'Students': len(dept_students),
                    'Total Points': total_points,
                    'Avg Points': round(avg_points, 1),
                    'Top Student': top_student['name'] if top_student else 'N/A',
                    'Top Points': top_student['total_points'] if top_student else 0
                })
        
        if dept_stats:
            dept_df = pd.DataFrame(dept_stats)
            dept_df = dept_df.sort_values('Total Points', ascending=False)
            st.dataframe(dept_df, use_container_width=True)
        
        # Points distribution
        st.subheader("📈 Points Distribution")
        all_students = db.get_leaderboard(limit=1000)
        
        if all_students:
            points_data = [s.get('total_points', 0) for s in all_students]
            
            # Create histogram
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[go.Histogram(x=points_data, nbinsx=20)])
            fig.update_layout(
                title="Points Distribution Among Students",
                xaxis_title="Points",
                yaxis_title="Number of Students",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Max Points", max(points_data) if points_data else 0)
            with col_s2:
                st.metric("Average Points", round(sum(points_data)/len(points_data), 1) if points_data else 0)
            with col_s3:
                st.metric("Median Points", sorted(points_data)[len(points_data)//2] if points_data else 0)
            with col_s4:
                st.metric("Total Students", len(points_data))

    
# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application function"""
    
    # Check for remember me cookie first
    if 'role' not in st.session_state or st.session_state.role is None:
        check_remember_me_cookie()
    
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
    
    # Session timeout check (skip if remember me is enabled)
    if (st.session_state.role and 'session_start' in st.session_state and 
        not st.session_state.remember_me):
        session_duration = datetime.now() - st.session_state.session_start
        if session_duration.total_seconds() > SESSION_TIMEOUT_MINUTES * 60:
            st.warning("Session timed out. Please login again.")
            # Clear remember me token
            if st.session_state.username:
                db.clear_reset_token(st.session_state.username)
            st.query_params = {}
            
            for key in list(st.session_state.keys()):
                if key != 'rerun_count':
                    del st.session_state[key]
            st.rerun()

    # Display database info in sidebar
    if db.use_supabase:
        st.sidebar.success("✅ Using Supabase PostgreSQL")
    else:
        st.sidebar.info("💾 Using SQLite (Local)")
    
    # Show setup guide if Supabase is not configured
    if USE_SUPABASE and not db.use_supabase:
        with st.sidebar.expander("🚀 Setup Supabase (Free Forever)", expanded=True):
            st.markdown("""
            ### Get Free PostgreSQL Database:
            
            1. **Go to [supabase.com](https://supabase.com)**
            2. **Create free account**
            3. **Create new project**
            4. **Go to Settings > API**
            5. **Copy URL and anon key**
            
            ### Set Streamlit Secrets:
            ```toml
            # .streamlit/secrets.toml
            [SUPABASE]
            url = "https://your-project.supabase.co"
            key = "your-anon-key"
            ```
            """)
    
    # Update event status
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
    elif st.session_state.role == 'admin':
        admin_dashboard()
    elif st.session_state.role == 'faculty':
        faculty_dashboard()
    elif st.session_state.role == 'mentor':
        mentor_dashboard()
    elif st.session_state.role == 'student':
        student_dashboard()

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()
