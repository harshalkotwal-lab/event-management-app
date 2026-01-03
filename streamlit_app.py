"""
G H Raisoni College - Advanced Event Management System
PRODUCTION READY with Supabase PostgreSQL (Free Forever)
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
# VALIDATION CLASS
# ============================================

class Validators:
    """Collection of input validation methods"""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email format"""
        if not email:
            return False, "Email required"
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False, "Invalid email format"
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
                        self.headers = {
                            'apikey': self.key,
                            'Authorization': f'Bearer {self.key}',
                            'Content-Type': 'application/json'
                        }
                        self.is_configured = True
                        logger.info("✅ Supabase configured successfully")
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
            return None
        
        try:
            import requests
            
            url = f"{self.url}/rest/v1/{table}"
            
            # Add filters
            if filters:
                filter_str = '&'.join([f"{k}=eq.{v}" for k, v in filters.items()])
                url = f"{url}?{filter_str}"
            
            # Add limit
            if '?' in url:
                url = f"{url}&limit={limit}"
            else:
                url = f"{url}?limit={limit}"
            
            # Make request
            if method == 'GET':
                response = requests.get(url, headers=self.headers)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=data)
            elif method == 'PATCH':
                response = requests.patch(url, headers=self.headers, json=data)
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers)
            
            response.raise_for_status()
            
            if method == 'GET':
                return response.json()
            else:
                return response.status_code in [200, 201, 204]
                
        except Exception as e:
            logger.error(f"Supabase API error: {e}")
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
            self.client = SQLiteClient()
        
        # Initialize database
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
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            """
        ]
        
        for sql in tables_sql:
            self.client.execute_query(sql, commit=True)
    
    # ============================================
    # USER MANAGEMENT METHODS
    # ============================================
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials"""
        try:
            # Check default admin/faculty first
            if role == 'admin' and username == 'admin@raisoni':
                return hashlib.sha256(password.encode()).hexdigest() == hashlib.sha256('Admin@12345'.encode()).hexdigest()
            elif role == 'faculty' and username == 'faculty@raisoni':
                return hashlib.sha256(password.encode()).hexdigest() == hashlib.sha256('Faculty@12345'.encode()).hexdigest()
            
            # Check database
            user = self.get_user(username)
            
            if user:
                stored_hash = user['password']
                user_role = user['role']
                is_active = user.get('is_active', True)
                
                if not is_active:
                    return False
                
                if user_role != role:
                    return False
                
                # Update last activity
                self.update_user_activity(username)
                
                return hashlib.sha256(password.encode()).hexdigest() == stored_hash
            return False
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    def get_user(self, username):
        """Get user by username"""
        try:
            if self.use_supabase:
                results = self.client.select('users', {'username': username})
                return results[0] if results else None
            else:
                return self.client.execute_query(
                    "SELECT * FROM users WHERE username = ?",
                    (username,), fetchone=True
                )
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def add_user(self, user_data):
        """Add new user"""
        try:
            # Validate mobile number
            if 'mobile' in user_data:
                is_valid, msg = Validators.validate_mobile(user_data['mobile'])
                if not is_valid:
                    return False, msg
            
            # Format mobile number
            mobile = Validators.format_mobile(user_data.get('mobile', ''))
            
            hashed_pass = hashlib.sha256(user_data.get('password').encode()).hexdigest()
            user_id = user_data.get('id', str(uuid.uuid4()))
            
            user_record = {
                'id': user_id,
                'name': user_data.get('name'),
                'username': user_data.get('username'),
                'password': hashed_pass,
                'role': user_data.get('role', 'student'),
                'roll_no': user_data.get('roll_no'),
                'department': user_data.get('department'),
                'year': user_data.get('year'),
                'email': user_data.get('email'),
                'mobile': mobile,
                'created_at': datetime.now().isoformat()
            }
            
            if self.use_supabase:
                success = self.client.insert('users', user_record)
            else:
                success = self.client.insert('users', user_record)
            
            if success:
                logger.info(f"New user registered: {user_data.get('username')}")
                return True, "User registered successfully"
            else:
                return False, "Registration failed"
                
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return False, "Registration failed"
    
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
                'created_at': datetime.now().isoformat()
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
    
    def update_mentor(self, mentor_id, mentor_data):
        """Update mentor information"""
        try:
            if self.use_supabase:
                return self.client.update('mentors', {'id': mentor_id}, mentor_data)
            else:
                return self.client.update('mentors', {'id': mentor_id}, mentor_data)
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
                'registered_at': datetime.now().isoformat()
            }
            
            if self.use_supabase:
                success = self.client.insert('registrations', registration_record)
            else:
                success = self.client.insert('registrations', registration_record)
            
            if success:
                logger.info(f"New registration: {reg_data['student_username']}")
                return reg_id, "Registration successful"
            return None, "Registration failed"
            
        except Exception as e:
            logger.error(f"Error adding registration: {e}")
            return None, "Registration failed"
    
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
    
    # ============================================
    # LIKES & INTEREST METHODS
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
        """Get all events liked by a student"""
        try:
            if self.use_supabase:
                # This is complex with REST API, simplified version
                return []
            else:
                return self.client.execute_query(
                    "SELECT e.* FROM events e JOIN event_likes l ON e.id = l.event_id WHERE l.student_username = ? ORDER BY l.liked_at DESC",
                    (student_username,), fetchall=True
                )
        except:
            return []
    
    def get_student_interested_events(self, student_username):
        """Get all events student is interested in"""
        try:
            if self.use_supabase:
                return []
            else:
                return self.client.execute_query(
                    "SELECT e.* FROM events e JOIN event_interested i ON e.id = i.event_id WHERE i.student_username = ? ORDER BY i.interested_at DESC",
                    (student_username,), fetchall=True
                )
        except:
            return []
    
    # ============================================
    # DEFAULT USERS
    # ============================================
    
    def _add_default_users(self):
        """Add default admin and faculty users"""
        try:
            default_users = [
                {
                    'id': str(uuid.uuid4()),
                    'name': 'Administrator',
                    'username': 'admin@raisoni',
                    'password': hashlib.sha256('Admin@12345'.encode()).hexdigest(),
                    'role': 'admin',
                    'email': 'admin@ghraisoni.edu',
                    'department': 'Administration',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': str(uuid.uuid4()),
                    'name': 'Faculty Coordinator',
                    'username': 'faculty@raisoni',
                    'password': hashlib.sha256('Faculty@12345'.encode()).hexdigest(),
                    'role': 'faculty',
                    'email': 'faculty@ghraisoni.edu',
                    'department': 'Faculty',
                    'created_at': datetime.now().isoformat()
                }
            ]
            
            for user in default_users:
                # Check if user exists
                existing = self.get_user(user['username'])
                if not existing:
                    if self.use_supabase:
                        self.client.insert('users', user)
                    else:
                        self.client.insert('users', user)
                    logger.info(f"Added default user: {user['username']}")
            
            # Add default students
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
                    student_data = {
                        'id': str(uuid.uuid4()),
                        'name': student['name'],
                        'username': student['username'],
                        'password': student['password'],
                        'role': 'student',
                        'roll_no': student['roll_no'],
                        'department': student['department'],
                        'year': student['year'],
                        'email': student['email'],
                        'mobile': student['mobile'],
                        'created_at': datetime.now().isoformat()
                    }
                    
                    if self.use_supabase:
                        self.client.insert('users', student_data)
                    else:
                        # Hash password for SQLite
                        student_data['password'] = hashlib.sha256(student['password'].encode()).hexdigest()
                        self.client.insert('users', student_data)
                    
                    logger.info(f"Added default student: {student['name']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding default students: {e}")
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
                for user in users:
                    role = user.get('role', 'unknown')
                    role_counts[role] = role_counts.get(role, 0) + 1
                stats['user_counts'] = role_counts
            
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
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

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
    }
    
    .college-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    
    .event-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        background: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #3B82F6;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 6px;
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
    
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Initialize database
    global db
    try:
        db = DatabaseManager(use_supabase=USE_SUPABASE)
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        st.stop()
    
    # Update event status periodically
    if 'last_status_update' not in st.session_state:
        st.session_state.last_status_update = datetime.now()
    
    if (datetime.now() - st.session_state.last_status_update).total_seconds() > 300:
        db.update_event_status()
        st.session_state.last_status_update = datetime.now()
    
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
    
    # Simple landing page for demo
    if st.session_state.role is None:
        st.markdown(f'<div class="college-header"><h2>{COLLEGE_CONFIG["name"]}</h2><p>Advanced Event Management System</p></div>', 
                    unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">Welcome to Event Manager</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 👑 Admin Login")
            if st.button("Admin Login", use_container_width=True):
                st.session_state.role = 'admin'
                st.session_state.username = 'admin@raisoni'
                st.session_state.name = 'Administrator'
                st.rerun()
        
        with col2:
            st.markdown("### 👨‍🏫 Faculty Login")
            if st.button("Faculty Login", use_container_width=True):
                st.session_state.role = 'faculty'
                st.session_state.username = 'faculty@raisoni'
                st.session_state.name = 'Faculty Coordinator'
                st.rerun()
        
        with col3:
            st.markdown("### 👨‍🎓 Student Login")
            if st.button("Student Login", use_container_width=True):
                st.session_state.role = 'student'
                st.session_state.username = 'rohan@student'
                st.session_state.name = 'Rohan Sharma'
                st.rerun()
        
        # Show statistics
        stats = db.get_system_stats()
        if stats:
            st.markdown("---")
            st.subheader("📊 System Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_events = sum(stats.get('event_counts', {}).values())
                st.metric("Total Events", total_events)
            with col2:
                total_users = sum(stats.get('user_counts', {}).values())
                st.metric("Total Users", total_users)
            with col3:
                st.metric("Registrations", stats.get('total_registrations', 0))
            with col4:
                st.metric("AI Events", stats.get('ai_events', 0))
    
    else:
        # User is logged in - show dashboard
        st.sidebar.title(f"{st.session_state.role.title()} Panel")
        st.sidebar.markdown(f"**User:** {st.session_state.name}")
        display_role_badge(st.session_state.role)
        
        # Dashboard based on role
        if st.session_state.role == 'admin':
            admin_dashboard()
        elif st.session_state.role == 'faculty':
            faculty_dashboard()
        elif st.session_state.role == 'student':
            student_dashboard()
        elif st.session_state.role == 'mentor':
            mentor_dashboard()
        
        # Logout button
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ============================================
# DASHBOARD FUNCTIONS (Simplified for demo)
# ============================================

def admin_dashboard():
    """Admin dashboard"""
    st.markdown('<h1 class="main-header">👑 Admin Dashboard</h1>', unsafe_allow_html=True)
    
    # Statistics
    stats = db.get_system_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_users = sum(stats.get('user_counts', {}).values())
        st.metric("Total Users", total_users)
    with col2:
        total_events = sum(stats.get('event_counts', {}).values())
        st.metric("Total Events", total_events)
    with col3:
        st.metric("Active Mentors", stats.get('active_mentors', 0))
    with col4:
        st.metric("Database", "Supabase" if db.use_supabase else "SQLite")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Users", "📅 Events", "⚙️ Settings"])
    
    with tab1:
        st.subheader("User Management")
        users = db.client.select('users') if db.use_supabase else db.client.execute_query("SELECT * FROM users", fetchall=True)
        
        if users:
            df_data = []
            for user in users:
                df_data.append({
                    'Name': user.get('name'),
                    'Username': user.get('username'),
                    'Role': user.get('role').title(),
                    'Department': user.get('department', 'N/A'),
                    'Mobile': user.get('mobile', 'N/A')
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No users found")
    
    with tab2:
        st.subheader("Event Management")
        events = db.get_all_events()
        
        if events:
            for event in events[:5]:  # Show first 5 events
                with st.container():
                    st.markdown('<div class="event-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f'<div class="card-title">{event.get("title", "Untitled")}</div>', unsafe_allow_html=True)
                        st.caption(f"📅 {format_date(event.get('event_date'))}")
                        st.caption(f"📍 {event.get('venue', 'TBD')}")
                        st.caption(f"👤 Created by: {event.get('created_by_name', 'Unknown')}")
                    
                    with col2:
                        status_html = get_event_status(event.get('event_date'))
                        st.markdown(status_html, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No events found")
    
    with tab3:
        st.subheader("System Settings")
        
        st.info(f"Current database: **{'Supabase PostgreSQL' if db.use_supabase else 'SQLite'}**")
        
        if st.button("Update Event Status", use_container_width=True):
            if db.update_event_status():
                st.success("Event status updated")
            else:
                st.error("Failed to update event status")
        
        if st.button("Export User Data (CSV)", use_container_width=True):
            users = db.client.select('users') if db.use_supabase else db.client.execute_query("SELECT * FROM users", fetchall=True)
            if users:
                df = pd.DataFrame(users)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="users_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )

def faculty_dashboard():
    """Faculty dashboard"""
    st.markdown('<h1 class="main-header">👨‍🏫 Faculty Dashboard</h1>', unsafe_allow_html=True)
    
    # Quick stats
    events = db.get_events_by_creator(st.session_state.username)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("My Events", len(events))
    with col2:
        upcoming = len([e for e in events if e.get('status') == 'upcoming'])
        st.metric("Upcoming", upcoming)
    with col3:
        total_participants = sum(e.get('current_participants', 0) for e in events)
        st.metric("Participants", total_participants)
    
    # Create event form
    with st.expander("➕ Create New Event", expanded=False):
        with st.form("create_event_form"):
            title = st.text_input("Event Title")
            event_type = st.selectbox("Event Type", COLLEGE_CONFIG['event_types'])
            event_date = st.date_input("Event Date", min_value=date.today())
            venue = st.text_input("Venue")
            description = st.text_area("Description")
            
            if st.form_submit_button("Create Event", use_container_width=True):
                if title and venue and description:
                    event_data = {
                        'title': title,
                        'description': description,
                        'event_type': event_type,
                        'event_date': datetime.combine(event_date, datetime.now().time()).isoformat(),
                        'venue': venue,
                        'organizer': 'G H Raisoni College',
                        'created_by': st.session_state.username,
                        'created_by_name': st.session_state.name
                    }
                    
                    if db.add_event(event_data):
                        st.success(f"Event '{title}' created successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to create event")
                else:
                    st.error("Please fill all required fields")
    
    # My events
    st.subheader("📋 My Events")
    if events:
        for event in events:
            with st.container():
                st.markdown('<div class="event-card">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f'<div class="card-title">{event.get("title", "Untitled")}</div>', unsafe_allow_html=True)
                    st.caption(f"📅 {format_date(event.get('event_date'))}")
                    st.caption(f"📍 {event.get('venue', 'TBD')}")
                    st.caption(f"👥 {event.get('current_participants', 0)}/{event.get('max_participants', 100)} participants")
                
                with col2:
                    status_html = get_event_status(event.get('event_date'))
                    st.markdown(status_html, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("You haven't created any events yet")

def student_dashboard():
    """Student dashboard"""
    st.markdown('<h1 class="main-header">👨‍🎓 Student Dashboard</h1>', unsafe_allow_html=True)
    
    # Student info
    student = db.get_user(st.session_state.username)
    if student:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {student.get('name')}")
            st.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
            st.markdown(f"**Department:** {student.get('department', 'N/A')}")
        with col2:
            st.markdown(f"**Year:** {student.get('year', 'N/A')}")
            st.markdown(f"**Mobile:** {student.get('mobile', 'Not provided')}")
    
    # Statistics
    registrations = db.get_registrations_by_student(st.session_state.username)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Events Registered", len(registrations))
    with col2:
        liked_events = db.get_student_liked_events(st.session_state.username)
        st.metric("Liked Events", len(liked_events))
    with col3:
        interested_events = db.get_student_interested_events(st.session_state.username)
        st.metric("Interested Events", len(interested_events))
    
    # Browse events
    st.subheader("🎯 Browse Events")
    events = db.get_all_events()
    
    if events:
        for event in events[:10]:  # Show first 10 events
            with st.container():
                st.markdown('<div class="event-card">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f'<div class="card-title">{event.get("title", "Untitled")}</div>', unsafe_allow_html=True)
                    st.caption(f"📅 {format_date(event.get('event_date'))}")
                    st.caption(f"📍 {event.get('venue', 'TBD')}")
                    st.caption(f"🏷️ {event.get('event_type', 'Event')}")
                
                with col2:
                    status_html = get_event_status(event.get('event_date'))
                    st.markdown(status_html, unsafe_allow_html=True)
                    
                    # Registration button
                    is_registered = db.is_student_registered(event['id'], st.session_state.username)
                    if is_registered:
                        st.success("✅ Registered")
                    else:
                        if st.button("Register", key=f"reg_{event['id']}", use_container_width=True):
                            student_info = db.get_user(st.session_state.username)
                            if student_info:
                                reg_data = {
                                    'event_id': event['id'],
                                    'event_title': event.get('title'),
                                    'student_username': st.session_state.username,
                                    'student_name': student_info.get('name'),
                                    'student_roll': student_info.get('roll_no', 'N/A'),
                                    'student_dept': student_info.get('department', 'N/A')
                                }
                                reg_id, message = db.add_registration(reg_data)
                                if reg_id:
                                    st.success("Registered successfully!")
                                    st.rerun()
                                else:
                                    st.error(message)
                
                # Like and Interested buttons
                col_like, col_interested = st.columns(2)
                with col_like:
                    is_liked = db.is_event_liked(event['id'], st.session_state.username)
                    if is_liked:
                        if st.button("❤️ Liked", key=f"like_{event['id']}", use_container_width=True):
                            db.remove_like(event['id'], st.session_state.username)
                            st.rerun()
                    else:
                        if st.button("🤍 Like", key=f"unlike_{event['id']}", use_container_width=True):
                            db.add_like(event['id'], st.session_state.username)
                            st.rerun()
                
                with col_interested:
                    is_interested = db.is_event_interested(event['id'], st.session_state.username)
                    if is_interested:
                        if st.button("⭐ Interested", key=f"interested_{event['id']}", use_container_width=True):
                            db.remove_interested(event['id'], st.session_state.username)
                            st.rerun()
                    else:
                        if st.button("☆ Interested", key=f"not_interested_{event['id']}", use_container_width=True):
                            db.add_interested(event['id'], st.session_state.username)
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No events available")

def mentor_dashboard():
    """Mentor dashboard"""
    st.markdown('<h1 class="main-header">👨‍🏫 Mentor Dashboard</h1>', unsafe_allow_html=True)
    
    # Mentor info
    mentor = db.get_user(st.session_state.username)
    if mentor:
        st.markdown(f"**Name:** {mentor.get('name')}")
        st.markdown(f"**Department:** {mentor.get('department', 'N/A')}")
        st.markdown(f"**Email:** {mentor.get('email', 'N/A')}")
    
    # Get mentor ID
    mentors = db.client.select('mentors', {'email': st.session_state.username}) if db.use_supabase else db.client.execute_query("SELECT * FROM mentors WHERE email = ?", (st.session_state.username,), fetchall=True)
    
    if mentors:
        mentor_info = mentors[0]
        mentor_id = mentor_info['id']
        
        # Get events assigned to this mentor
        events = db.client.select('events', {'mentor_id': mentor_id}) if db.use_supabase else db.client.execute_query("SELECT * FROM events WHERE mentor_id = ? ORDER BY event_date DESC", (mentor_id,), fetchall=True)
        
        st.subheader("📅 My Assigned Events")
        
        if events:
            for event in events:
                with st.container():
                    st.markdown('<div class="event-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f'<div class="card-title">{event.get("title", "Untitled")}</div>', unsafe_allow_html=True)
                        st.caption(f"📅 {format_date(event.get('event_date'))}")
                        st.caption(f"📍 {event.get('venue', 'TBD')}")
                        st.caption(f"👥 {event.get('current_participants', 0)} participants")
                    
                    with col2:
                        status_html = get_event_status(event.get('event_date'))
                        st.markdown(status_html, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No events assigned to you yet")
    else:
        st.info("Mentor profile not found")

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()
