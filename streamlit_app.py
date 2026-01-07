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
# SUPABASE CLIENT (PostgreSQL) - OPTIMIZED
# ============================================

class SupabaseClient:
    """Supabase PostgreSQL client using HTTP REST API - OPTIMIZED"""
    
    def __init__(self):
        self.url = None
        self.key = None
        self.headers = None
        self.is_configured = False
        self._cache = {}
        
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
                            'Prefer': 'return=representation'
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
    
    def _get_cache_key(self, table, method='GET', filters=None):
        """Generate cache key"""
        key_parts = [table, method]
        if filters:
            key_parts.append(json.dumps(filters, sort_keys=True))
        return hashlib.md5('_'.join(key_parts).encode()).hexdigest()
    
    def execute_query(self, table, method='GET', data=None, filters=None, limit=1000, order_by=None, cache_ttl=60):
        """Execute REST API query to Supabase - OPTIMIZED"""
        if not self.is_configured:
            logger.error("Supabase not configured")
            return None
        
        # Check cache for GET requests
        if method == 'GET':
            cache_key = self._get_cache_key(table, method, filters)
            if cache_key in self._cache:
                cache_data, timestamp = self._cache[cache_key]
                if time.time() - timestamp < cache_ttl:
                    return cache_data
        
        try:
            import requests
            
            url = f"{self.url}/rest/v1/{table}"
            
            # Add query parameters
            params = {}
            
            # Add filters using proper eq. syntax
            if filters:
                for k, v in filters.items():
                    if v is not None:
                        params[k] = f'eq.{v}'
            
            # Add ordering
            if order_by:
                params['order'] = order_by
            
            # Add limit
            params['limit'] = str(limit)
            
            # Make request
            timeout = 15  # Increased timeout
            if method == 'GET':
                response = requests.get(url, headers=self.headers, params=params, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=data, timeout=timeout)
            elif method == 'PATCH':
                response = requests.patch(url, headers=self.headers, json=data, params=params, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers, params=params, timeout=timeout)
            else:
                logger.error(f"Unsupported method: {method}")
                return None
            
            # Check for errors
            if response.status_code >= 400:
                logger.error(f"Supabase API error {response.status_code}: {response.text[:200]}")
                return None
            
            result = None
            if method == 'GET':
                result = response.json() if response.text else []
                # Cache the result
                cache_key = self._get_cache_key(table, method, filters)
                self._cache[cache_key] = (result, time.time())
            elif method in ['POST', 'PATCH']:
                result = response.json() if response.text else True
            elif method == 'DELETE':
                result = response.status_code in [200, 204]
            
            return result
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Supabase API request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Supabase query: {e}")
            return None
    
    def clear_cache(self):
        """Clear cache"""
        self._cache.clear()
    
    def insert(self, table, data):
        """Insert data into table"""
        result = self.execute_query(table, 'POST', data)
        if result is None:
            return False
        return bool(result)
    
    def select(self, table, filters=None, limit=1000, order_by=None, cache_ttl=60):
        """Select data from table"""
        return self.execute_query(table, 'GET', filters=filters, limit=limit, order_by=order_by, cache_ttl=cache_ttl)
    
    def update(self, table, filters, data):
        """Update data in table"""
        result = self.execute_query(table, 'PATCH', data, filters)
        if result is None:
            return False
        return bool(result)
    
    def delete(self, table, filters):
        """Delete data from table"""
        return self.execute_query(table, 'DELETE', filters=filters)

# ============================================
# SQLITE CLIENT (Fallback) - OPTIMIZED
# ============================================

class SQLiteClient:
    """SQLite client for local development - OPTIMIZED"""
    
    def __init__(self, db_path="data/event_management.db"):
        self.db_path = db_path
        self.conn = None
        self._cache = {}
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
            self.conn.execute("PRAGMA cache_size = -10000")  # Increased cache
            logger.info("✅ SQLite database initialized")
        except Exception as e:
            logger.error(f"SQLite initialization error: {e}")
            raise
    
    def _get_cache_key(self, query, params=None):
        """Generate cache key"""
        key_parts = [query]
        if params:
            key_parts.append(str(params))
        return hashlib.md5('_'.join(key_parts).encode()).hexdigest()
    
    def execute_query(self, query, params=None, fetchone=False, fetchall=False, commit=False, cache_ttl=60):
        """Execute SQL query - OPTIMIZED"""
        # Check cache for SELECT queries
        if 'SELECT' in query.upper() and fetchall:
            cache_key = self._get_cache_key(query, params)
            if cache_key in self._cache:
                cache_data, timestamp = self._cache[cache_key]
                if time.time() - timestamp < cache_ttl:
                    return cache_data
        
        try:
            cursor = self.conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if commit:
                self.conn.commit()
            
            result = None
            if fetchone:
                result = cursor.fetchone()
                result = dict(result) if result else None
            elif fetchall:
                results = cursor.fetchall()
                result = [dict(row) for row in results]
                # Cache the result
                cache_key = self._get_cache_key(query, params)
                self._cache[cache_key] = (result, time.time())
            else:
                result = cursor
            
            return result
        except Exception as e:
            logger.error(f"SQLite error: {e}, Query: {query[:100]}")
            return None
    
    def clear_cache(self):
        """Clear cache"""
        self._cache.clear()
    
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
    
    def select(self, table, filters=None, limit=1000, cache_ttl=60):
        """Select data from table"""
        query = f"SELECT * FROM {table}"
        
        if filters:
            conditions = ' AND '.join([f"{k} = ?" for k in filters.keys()])
            query = f"{query} WHERE {conditions}"
        
        query = f"{query} LIMIT {limit}"
        
        if filters:
            return self.execute_query(query, tuple(filters.values()), fetchall=True, cache_ttl=cache_ttl)
        else:
            return self.execute_query(query, fetchall=True, cache_ttl=cache_ttl)
    
    def update(self, table, filters, data):
        """Update data in table"""
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        conditions = ' AND '.join([f"{k} = ?" for k in filters.keys()])
        
        query = f"UPDATE {table} SET {set_clause} WHERE {conditions}"
        params = tuple(data.values()) + tuple(filters.values())
        
        try:
            cursor = self.execute_query(query, params)
            self.conn.commit()
            # Clear cache for this table
            self.clear_cache()
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
            # Clear cache for this table
            self.clear_cache()
            return cursor.rowcount > 0
        except:
            return False

# ============================================
# AI EVENT GENERATOR - OPTIMIZED
# ============================================

class AIEventGenerator:
    """Generate structured event data from unstructured text"""
    
    def __init__(self):
        self.api_key = None
        self.is_configured = False
        
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
            
            # Clean response
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
            if len(first_line) < 100:
                event_data['title'] = first_line
        
        # Try to extract date patterns
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
        
        # Try to extract URLs
        url_pattern = r'https?://[^\s<>"\'()]+'
        urls = re.findall(url_pattern, text)
        
        if urls:
            event_data['event_link'] = urls[0]
            if len(urls) > 1:
                event_data['registration_link'] = urls[1]
        
        return event_data

# ============================================
# UNIFIED DATABASE MANAGER - OPTIMIZED
# ============================================

class DatabaseManager:
    """Unified database manager supporting both Supabase and SQLite - OPTIMIZED"""
    
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
        """Initialize database tables - OPTIMIZED"""
        try:
            if not self.use_supabase:
                self._create_sqlite_tables()
            else:
                # For Supabase, we just verify tables exist
                self._verify_supabase_tables()
            
            logger.info(f"✅ Database initialized with {'Supabase' if self.use_supabase else 'SQLite'}")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def _verify_supabase_tables(self):
        """Verify that all required tables exist in Supabase"""
        # Simple check - if we can query users table, we assume all tables exist
        try:
            result = self.client.select('users', limit=1)
            if result is not None:
                logger.info("✅ Supabase tables verified")
            else:
                logger.warning("⚠️ Supabase tables might not exist")
        except:
            logger.warning("⚠️ Could not verify Supabase tables")
    
    def _create_sqlite_tables(self):
        """Create SQLite tables"""
        # Users table
        users_table = """
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
            reset_token_expiry TEXT,
            last_activity TEXT,
            is_active INTEGER DEFAULT 1,
            login_attempts INTEGER DEFAULT 0,
            last_login TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            total_points INTEGER DEFAULT 0,
            last_points_update TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Events table
        events_table = """
        CREATE TABLE IF NOT EXISTS events (
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
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Registrations table
        registrations_table = """
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
            registered_at TEXT DEFAULT CURRENT_TIMESTAMP,
            checked_in_at TEXT,
            UNIQUE(event_id, student_username),
            FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
        )
        """
        
        # Event likes table
        likes_table = """
        CREATE TABLE IF NOT EXISTS event_likes (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            student_username TEXT NOT NULL,
            liked_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(event_id, student_username),
            FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
        )
        """
        
        # Event interested table
        interested_table = """
        CREATE TABLE IF NOT EXISTS event_interested (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            student_username TEXT NOT NULL,
            interested_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(event_id, student_username),
            FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
        )
        """
        
        tables = [users_table, events_table, registrations_table, likes_table, interested_table]
        
        for table_sql in tables:
            self.client.execute_query(table_sql, commit=True)
    
    # ============================================
    # USER MANAGEMENT METHODS - OPTIMIZED
    # ============================================
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials - OPTIMIZED"""
        try:
            user = self.get_user(username)
            if not user:
                return False
            
            if user.get('role') != role:
                return False
            
            stored_pass = user.get('password', '')
            input_hash = hashlib.sha256(password.encode()).hexdigest().lower()
            
            if stored_pass == input_hash or stored_pass == password:
                # Update last login
                self.client.update('users', {'username': username}, {
                    'last_login': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                })
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    def get_user(self, username):
        """Get user by username - OPTIMIZED"""
        try:
            if self.use_supabase:
                results = self.client.select('users', {'username': username}, limit=1, cache_ttl=30)
                if results:
                    return results[0]
            else:
                return self.client.execute_query(
                    "SELECT * FROM users WHERE username = ?",
                    (username,), fetchone=True, cache_ttl=30
                )
        except Exception as e:
            logger.error(f"Error getting user {username}: {e}")
        return None
    
    def add_user(self, user_data):
        """Add new user - OPTIMIZED"""
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
                'total_points': 0,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('users', user_record)
            
            if success:
                # Clear cache
                if hasattr(self.client, 'clear_cache'):
                    self.client.clear_cache()
                logger.info(f"✅ User '{user_data.get('username')}' added")
                return True, "User registered successfully"
            else:
                return False, "Registration failed"
            
        except Exception as e:
            logger.error(f"❌ Error adding user: {e}")
            return False, str(e)
    
    # ============================================
    # EVENT MANAGEMENT METHODS - OPTIMIZED
    # ============================================
    
    def add_event(self, event_data):
        """Add new event - OPTIMIZED"""
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
                'current_participants': 0,
                'flyer_path': event_data.get('flyer_path'),
                'created_by': event_data.get('created_by'),
                'created_by_name': event_data.get('created_by_name'),
                'ai_generated': event_data.get('ai_generated', False),
                'status': 'upcoming',
                'mentor_id': event_data.get('mentor_id'),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('events', event_record)
            
            if success:
                # Clear cache
                if hasattr(self.client, 'clear_cache'):
                    self.client.clear_cache()
                logger.info(f"New event created: {event_data.get('title')}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error adding event: {e}")
            return False
    
    def get_all_events(self, cache_ttl=60):
        """Get all events - OPTIMIZED"""
        try:
            if self.use_supabase:
                events = self.client.select('events', limit=1000, order_by='event_date.desc', cache_ttl=cache_ttl)
            else:
                events = self.client.execute_query(
                    "SELECT * FROM events ORDER BY event_date DESC",
                    fetchall=True, cache_ttl=cache_ttl
                )
            return events or []
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    def get_events_by_creator(self, username, cache_ttl=60):
        """Get events created by specific user - OPTIMIZED"""
        try:
            if self.use_supabase:
                events = self.client.select('events', {'created_by': username}, limit=1000, cache_ttl=cache_ttl)
            else:
                events = self.client.execute_query(
                    "SELECT * FROM events WHERE created_by = ? ORDER BY event_date DESC",
                    (username,), fetchall=True, cache_ttl=cache_ttl
                )
            return events or []
        except Exception as e:
            logger.error(f"Error getting events by creator: {e}")
            return []
    
    # ============================================
    # REGISTRATION METHODS - OPTIMIZED
    # ============================================
    
    def add_registration(self, reg_data):
        """Add new registration - OPTIMIZED"""
        try:
            # Check if already registered
            existing = self.is_student_registered(reg_data['event_id'], reg_data['student_username'])
            if existing:
                return None, "Already registered"
            
            # Get student info
            student = self.get_user(reg_data['student_username'])
            mobile = student.get('mobile', '') if student else ''
            
            registration_record = {
                'id': reg_data.get('id', str(uuid.uuid4())),
                'event_id': reg_data.get('event_id'),
                'event_title': reg_data.get('event_title'),
                'student_username': reg_data.get('student_username'),
                'student_name': reg_data.get('student_name'),
                'student_roll': reg_data.get('student_roll'),
                'student_dept': reg_data.get('student_dept'),
                'student_mobile': mobile,
                'status': 'pending',
                'registered_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('registrations', registration_record)
            
            if success:
                # Clear cache
                if hasattr(self.client, 'clear_cache'):
                    self.client.clear_cache()
                # Update participant count
                self._update_event_participant_count(reg_data['event_id'])
                return registration_record['id'], "Registration successful"
            
            return None, "Registration failed"
            
        except Exception as e:
            logger.error(f"Error adding registration: {e}")
            return None, "Registration failed"
    
    def _update_event_participant_count(self, event_id):
        """Update event participant count - OPTIMIZED"""
        try:
            if self.use_supabase:
                registrations = self.client.select('registrations', {'event_id': event_id}, cache_ttl=0)
                count = len(registrations) if registrations else 0
                self.client.update('events', {'id': event_id}, {
                    'current_participants': count,
                    'updated_at': datetime.now().isoformat()
                })
            else:
                cursor = self.client.conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM registrations WHERE event_id = ?", (event_id,))
                count = cursor.fetchone()[0]
                cursor.execute("UPDATE events SET current_participants = ?, updated_at = ? WHERE id = ?", 
                             (count, datetime.now().isoformat(), event_id))
                self.client.conn.commit()
            
            # Clear cache
            if hasattr(self.client, 'clear_cache'):
                self.client.clear_cache()
            return True
        except Exception as e:
            logger.error(f"Error updating participant count: {e}")
            return False
    
    def get_registrations_by_student(self, username, cache_ttl=60):
        """Get all registrations for a student - OPTIMIZED"""
        try:
            if self.use_supabase:
                registrations = self.client.select('registrations', {'student_username': username}, cache_ttl=cache_ttl)
            else:
                registrations = self.client.execute_query(
                    "SELECT r.*, e.event_date, e.venue, e.status as event_status FROM registrations r LEFT JOIN events e ON r.event_id = e.id WHERE r.student_username = ? ORDER BY r.registered_at DESC",
                    (username,), fetchall=True, cache_ttl=cache_ttl
                )
            return registrations or []
        except Exception as e:
            logger.error(f"Error getting registrations: {e}")
            return []
    
    def is_student_registered(self, event_id, username):
        """Check if student is registered for event - OPTIMIZED"""
        try:
            if self.use_supabase:
                results = self.client.select('registrations', {
                    'event_id': event_id,
                    'student_username': username
                }, limit=1, cache_ttl=30)
                return bool(results)
            else:
                result = self.client.execute_query(
                    "SELECT id FROM registrations WHERE event_id = ? AND student_username = ?",
                    (event_id, username), fetchone=True, cache_ttl=30
                )
                return result is not None
        except Exception as e:
            logger.error(f"Error checking registration: {e}")
            return False
    
    # ============================================
    # LIKES & INTEREST METHODS - FIXED & OPTIMIZED
    # ============================================
    
    def add_like(self, event_id, student_username):
        """Add a like for an event - FIXED"""
        try:
            like_record = {
                'id': str(uuid.uuid4()),
                'event_id': event_id,
                'student_username': student_username,
                'liked_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('event_likes', like_record)
            
            if success:
                # Clear cache
                if hasattr(self.client, 'clear_cache'):
                    self.client.clear_cache()
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding like: {e}")
            return False
    
    def remove_like(self, event_id, student_username):
        """Remove a like for an event - FIXED"""
        try:
            success = self.client.delete('event_likes', {
                'event_id': event_id,
                'student_username': student_username
            })
            
            if success:
                # Clear cache
                if hasattr(self.client, 'clear_cache'):
                    self.client.clear_cache()
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing like: {e}")
            return False
    
    def is_event_liked(self, event_id, student_username):
        """Check if student liked an event - FIXED"""
        try:
            if self.use_supabase:
                results = self.client.select('event_likes', {
                    'event_id': event_id,
                    'student_username': student_username
                }, limit=1, cache_ttl=30)
                return bool(results)
            else:
                result = self.client.execute_query(
                    "SELECT id FROM event_likes WHERE event_id = ? AND student_username = ?",
                    (event_id, student_username), fetchone=True, cache_ttl=30
                )
                return result is not None
        except Exception as e:
            logger.error(f"Error checking like: {e}")
            return False
    
    def get_event_likes_count(self, event_id):
        """Get total likes for an event - OPTIMIZED"""
        try:
            if self.use_supabase:
                likes = self.client.select('event_likes', {'event_id': event_id}, cache_ttl=30)
                return len(likes) if likes else 0
            else:
                result = self.client.execute_query(
                    "SELECT COUNT(*) as count FROM event_likes WHERE event_id = ?",
                    (event_id,), fetchone=True, cache_ttl=30
                )
                return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting likes count: {e}")
            return 0
    
    def add_interested(self, event_id, student_username):
        """Add interested for an event - FIXED"""
        try:
            interested_record = {
                'id': str(uuid.uuid4()),
                'event_id': event_id,
                'student_username': student_username,
                'interested_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('event_interested', interested_record)
            
            if success:
                # Clear cache
                if hasattr(self.client, 'clear_cache'):
                    self.client.clear_cache()
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding interested: {e}")
            return False
    
    def remove_interested(self, event_id, student_username):
        """Remove interested for an event - FIXED"""
        try:
            success = self.client.delete('event_interested', {
                'event_id': event_id,
                'student_username': student_username
            })
            
            if success:
                # Clear cache
                if hasattr(self.client, 'clear_cache'):
                    self.client.clear_cache()
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing interested: {e}")
            return False
    
    def is_event_interested(self, event_id, student_username):
        """Check if student is interested in an event - FIXED"""
        try:
            if self.use_supabase:
                results = self.client.select('event_interested', {
                    'event_id': event_id,
                    'student_username': student_username
                }, limit=1, cache_ttl=30)
                return bool(results)
            else:
                result = self.client.execute_query(
                    "SELECT id FROM event_interested WHERE event_id = ? AND student_username = ?",
                    (event_id, student_username), fetchone=True, cache_ttl=30
                )
                return result is not None
        except Exception as e:
            logger.error(f"Error checking interested: {e}")
            return False
    
    def get_event_interested_count(self, event_id):
        """Get total interested count for an event - OPTIMIZED"""
        try:
            if self.use_supabase:
                interested = self.client.select('event_interested', {'event_id': event_id}, cache_ttl=30)
                return len(interested) if interested else 0
            else:
                result = self.client.execute_query(
                    "SELECT COUNT(*) as count FROM event_interested WHERE event_id = ?",
                    (event_id,), fetchone=True, cache_ttl=30
                )
                return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting interested count: {e}")
            return 0
    
    def get_student_liked_events(self, student_username):
        """Get all events liked by a student - OPTIMIZED"""
        try:
            if self.use_supabase:
                # Get all liked event IDs first
                likes = self.client.select('event_likes', {'student_username': student_username}, cache_ttl=60)
                if not likes:
                    return []
                
                # Get all events and filter
                all_events = self.get_all_events(cache_ttl=60)
                liked_event_ids = {like['event_id'] for like in likes}
                liked_events = [event for event in all_events if event.get('id') in liked_event_ids]
                
                # Sort by most recent like
                liked_events.sort(key=lambda x: next(
                    (like.get('liked_at', '') for like in likes if like.get('event_id') == x.get('id')),
                    ''
                ), reverse=True)
                
                return liked_events
            else:
                return self.client.execute_query(
                    "SELECT e.* FROM events e JOIN event_likes l ON e.id = l.event_id WHERE l.student_username = ? ORDER BY l.liked_at DESC",
                    (student_username,), fetchall=True, cache_ttl=60
                ) or []
        except Exception as e:
            logger.error(f"Error getting liked events: {e}")
            return []
    
    def get_student_interested_events(self, student_username):
        """Get all events student is interested in - OPTIMIZED"""
        try:
            if self.use_supabase:
                # Get all interested event IDs first
                interests = self.client.select('event_interested', {'student_username': student_username}, cache_ttl=60)
                if not interests:
                    return []
                
                # Get all events and filter
                all_events = self.get_all_events(cache_ttl=60)
                interested_event_ids = {interest['event_id'] for interest in interests}
                interested_events = [event for event in all_events if event.get('id') in interested_event_ids]
                
                # Sort by most recent interest
                interested_events.sort(key=lambda x: next(
                    (interest.get('interested_at', '') for interest in interests if interest.get('event_id') == x.get('id')),
                    ''
                ), reverse=True)
                
                return interested_events
            else:
                return self.client.execute_query(
                    "SELECT e.* FROM events e JOIN event_interested i ON e.id = i.event_id WHERE i.student_username = ? ORDER BY i.interested_at DESC",
                    (student_username,), fetchall=True, cache_ttl=60
                ) or []
        except Exception as e:
            logger.error(f"Error getting interested events: {e}")
            return []
    
    # ============================================
    # GAMIFICATION METHODS - OPTIMIZED
    # ============================================
    
    def get_student_points(self, username):
        """Get student's total points - OPTIMIZED"""
        try:
            user = self.get_user(username)
            if user:
                return user.get('total_points', 0)
            return 0
        except Exception as e:
            logger.error(f"Error getting points: {e}")
            return 0
    
    def get_leaderboard(self, limit=15, department=None, cache_ttl=300):
        """Get leaderboard of top students - OPTIMIZED"""
        try:
            if self.use_supabase:
                # Get all students
                filters = {'role': 'student'}
                if department:
                    filters['department'] = department
                
                users = self.client.select('users', filters, limit=1000, cache_ttl=cache_ttl)
                if not users:
                    return []
                
                # Sort by points
                users.sort(key=lambda x: x.get('total_points', 0), reverse=True)
                
                # Add rank
                for i, user in enumerate(users[:limit], 1):
                    user['rank'] = i
                
                return users[:limit]
            else:
                query = "SELECT * FROM users WHERE role = 'student'"
                params = []
                
                if department:
                    query += " AND department = ?"
                    params.append(department)
                
                query += " ORDER BY total_points DESC, name ASC LIMIT ?"
                params.append(limit)
                
                results = self.client.execute_query(query, tuple(params), fetchall=True, cache_ttl=cache_ttl) or []
                
                # Add rank
                for i, result in enumerate(results, 1):
                    result['rank'] = i
                
                return results
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []
    
    def get_student_rank(self, username):
        """Get student's rank in leaderboard - OPTIMIZED"""
        try:
            leaderboard = self.get_leaderboard(limit=1000, cache_ttl=300)
            for i, student in enumerate(leaderboard, 1):
                if student['username'] == username:
                    return i
            return None
        except Exception as e:
            logger.error(f"Error getting student rank: {e}")
            return None
    
    # ============================================
    # DEFAULT USERS - OPTIMIZED
    # ============================================
    
    def _add_default_users(self):
        """Add default admin and faculty users - OPTIMIZED"""
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
                    else:
                        logger.error(f"❌ Failed to add default user {user_data['username']}: {message}")
            
            # Add default students
            self._add_default_students()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding default users: {e}")
            return False
    
    def _add_default_students(self):
        """Add default student accounts - OPTIMIZED"""
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

# ============================================
# DATABASE INITIALIZATION
# ============================================

# Initialize database
db = DatabaseManager(use_supabase=USE_SUPABASE)

# ============================================
# HELPER FUNCTIONS - OPTIMIZED
# ============================================

@st.cache_data(ttl=300)
def format_date(date_str):
    """Format date for display - CACHED"""
    try:
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            dt = date_str
        
        return dt.strftime("%d %b %Y, %I:%M %p")
    except:
        return str(date_str)

def get_event_status(event_date):
    """Get event status badge - OPTIMIZED"""
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
# EVENT CARD DISPLAY - FIXED VERSION
# ============================================

def display_event_card(event, current_user=None):
    """Display improved event card with all interactions - FIXED"""
    if not event or not event.get('id'):
        return
    
    event_id = event.get('id')
    
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        # Create two-column layout
        col_img, col_info = st.columns([1, 3], gap="medium")
        
        with col_img:
            # Display event flyer
            flyer = event.get('flyer_path')
            if flyer and flyer.startswith('data:image'):
                try:
                    st.image(flyer, use_column_width=True)
                except:
                    st.markdown('<div style="width: 100%; height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; border-radius: 8px;"><span style="font-size: 32px; color: white;">🎯</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="width: 100%; height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; border-radius: 8px;"><span style="font-size: 32px; color: white;">🎯</span></div>', unsafe_allow_html=True)
        
        with col_info:
            # Header with title
            title = event.get('title', 'Untitled Event')
            if len(title) > 60:
                title = title[:57] + "..."
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
            
            # Status and date
            event_date = event.get('event_date')
            status_html = get_event_status(event_date)
            formatted_date = format_date(event_date)
            
            st.markdown(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>{status_html}</div>
                <div style="color: #666; font-size: 0.9rem;">📅 {formatted_date}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Event details
            venue = event.get('venue', 'TBD')
            if len(venue) > 25:
                venue = venue[:22] + "..."
            
            event_type = event.get('event_type', 'Event')
            max_participants = event.get('max_participants', 100)
            current_participants = event.get('current_participants', 0)
            
            st.caption(f"📍 {venue} | 🏷️ {event_type} | 👥 {current_participants}/{max_participants}")
            
            # Engagement metrics and buttons
            likes_count = db.get_event_likes_count(event_id)
            interested_count = db.get_event_interested_count(event_id)
            
            # Create buttons container
            if current_user:
                # Engagement buttons
                button_col1, button_col2, button_col3 = st.columns(3)
                
                with button_col1:
                    # Like button
                    is_liked = db.is_event_liked(event_id, current_user)
                    like_btn_text = "❤️ Liked" if is_liked else "🤍 Like"
                    like_btn_type = "secondary" if is_liked else "primary"
                    
                    if st.button(like_btn_text, key=f"like_{event_id}_{current_user}", 
                               use_container_width=True, type=like_btn_type,
                               help="Like this event"):
                        if is_liked:
                            if db.remove_like(event_id, current_user):
                                st.success("Like removed!")
                                st.rerun()
                        else:
                            if db.add_like(event_id, current_user):
                                st.success("Event liked!")
                                st.rerun()
                
                with button_col2:
                    # Interested button
                    is_interested = db.is_event_interested(event_id, current_user)
                    interested_btn_text = "⭐ Interested" if is_interested else "☆ Interested"
                    interested_btn_type = "secondary" if is_interested else "primary"
                    
                    if st.button(interested_btn_text, key=f"interested_{event_id}_{current_user}",
                               use_container_width=True, type=interested_btn_type,
                               help="Mark as interested"):
                        if is_interested:
                            if db.remove_interested(event_id, current_user):
                                st.success("Removed from interested!")
                                st.rerun()
                        else:
                            if db.add_interested(event_id, current_user):
                                st.success("Marked as interested!")
                                st.rerun()
                
                with button_col3:
                    # Share button with improved feedback
                    if st.button("📤 Share", key=f"share_{event_id}_{current_user}",
                               use_container_width=True, type="secondary",
                               help="Share this event"):
                        event_title = event.get('title', 'Cool Event')
                        share_text = f"Check out '{event_title}' at G H Raisoni College Event Manager!\n\nEvent Date: {formatted_date}\nVenue: {venue}\n\nJoin the platform to discover more events!"
                        
                        # Create a copy to clipboard button
                        st.markdown(f"""
                        <div style="background: #f0f9ff; padding: 10px; border-radius: 5px; border: 1px solid #bae6fd; margin: 5px 0;">
                            <p style="margin: 0 0 10px 0; font-weight: bold;">📋 Share message copied to clipboard!</p>
                            <div style="background: white; padding: 8px; border-radius: 3px; border: 1px solid #e2e8f0; font-size: 12px; color: #475569;">
                                {share_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add JavaScript to copy to clipboard
                        st.markdown(f"""
                        <script>
                        function copyToClipboard() {{
                            const text = `{share_text}`;
                            navigator.clipboard.writeText(text);
                        }}
                        copyToClipboard();
                        </script>
                        """, unsafe_allow_html=True)
            
            # Show engagement counts
            st.caption(f"❤️ {likes_count} Likes | ⭐ {interested_count} Interested")
            
            # Event links
            event_link = event.get('event_link', '')
            registration_link = event.get('registration_link', '')
            
            if event_link or registration_link:
                with st.expander("🔗 Event Links", expanded=False):
                    if event_link:
                        st.markdown(f"**🌐 Event Page:** [Click here]({event_link})")
                    if registration_link:
                        st.markdown(f"**📝 Registration:** [Click here]({registration_link})")
            
            # Description
            desc = event.get('description', '')
            if desc:
                if len(desc) > 150:
                    with st.expander("📝 Description", expanded=False):
                        st.write(desc)
                else:
                    st.caption(desc[:150] + "..." if len(desc) > 150 else desc)
        
        # Registration Section - FIXED
        if current_user and st.session_state.get('role') == 'student':
            st.markdown('<div class="registration-section">', unsafe_allow_html=True)
            
            is_registered = db.is_student_registered(event_id, current_user)
            
            if is_registered:
                st.success("✅ You are already registered for this event")
                
                # External registration button
                if registration_link:
                    if st.button("✅ I Have Registered Externally", 
                               key=f"ext_reg_{event_id}_{current_user}",
                               use_container_width=True,
                               type="secondary",
                               help="Mark that you have registered externally"):
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
                reg_col1, reg_col2 = st.columns(2)
                
                with reg_col1:
                    # Register in App button
                    if st.button("📱 Register in App", 
                                key=f"app_reg_{event_id}_{current_user}",
                                use_container_width=True,
                                type="primary"):
                        student = db.get_user(current_user)
                        if student:
                            reg_data = {
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
                    # External registration link button
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
# STUDENT DASHBOARD - OPTIMIZED
# ============================================

def student_dashboard():
    """Student dashboard with all features - OPTIMIZED"""
    # Sidebar
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    
    # Get student info
    student = db.get_user(st.session_state.username)
    if student:
        st.sidebar.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
        st.sidebar.markdown(f"**Department:** {student.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Year:** {student.get('year', 'N/A')}")
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Events Feed", "My Registrations", "Liked Events", "Interested Events", "Leaderboard", "My Profile"]
        
        if 'student_page' not in st.session_state:
            st.session_state.student_page = "Events Feed"
        
        for option in nav_options:
            if st.button(option, key=f"student_{option}", use_container_width=True):
                st.session_state.student_page = option
                st.rerun()
        
        # Quick stats
        st.markdown("---")
        st.markdown("### My Stats")
        
        # Get counts efficiently
        if 'student_stats' not in st.session_state:
            with st.spinner("Loading stats..."):
                registrations = db.get_registrations_by_student(st.session_state.username)
                liked_events = db.get_student_liked_events(st.session_state.username)
                interested_events = db.get_student_interested_events(st.session_state.username)
                points = db.get_student_points(st.session_state.username)
                
                st.session_state.student_stats = {
                    'registrations': len(registrations),
                    'liked': len(liked_events),
                    'interested': len(interested_events),
                    'points': points
                }
        
        stats = st.session_state.student_stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 Events", stats['registrations'])
        with col2:
            st.metric("🏆 Points", stats['points'])
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            # Clear session
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    selected = st.session_state.student_page
    
    if selected == "Events Feed":
        st.markdown('<h1 class="main-header">🎯 Discover Events</h1>', unsafe_allow_html=True)
        
        # Get events with cache
        if 'all_events' not in st.session_state:
            with st.spinner("Loading events..."):
                st.session_state.all_events = db.get_all_events(cache_ttl=60)
        
        events = st.session_state.all_events
        
        if not events:
            st.info("No events found.")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            search = st.text_input("🔍 Search events", placeholder="Search...")
        with col2:
            event_type = st.selectbox("Type", ["All"] + COLLEGE_CONFIG['event_types'])
        with col3:
            show_status = st.selectbox("Status", ["All", "Upcoming", "Ongoing", "Past"])
        
        # Apply filters
        filtered_events = events
        
        if search:
            search_lower = search.lower()
            filtered_events = [e for e in filtered_events 
                             if search_lower in e.get('title', '').lower() or 
                             search_lower in e.get('description', '').lower()]
        
        if event_type != "All":
            filtered_events = [e for e in filtered_events if e.get('event_type') == event_type]
        
        if show_status != "All":
            # Determine status based on date
            now = datetime.now()
            for event in filtered_events:
                event_date = event.get('event_date')
                if isinstance(event_date, str):
                    try:
                        event_dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                        if show_status == "Upcoming" and event_dt > now:
                            continue
                        elif show_status == "Ongoing" and event_dt.date() == now.date():
                            continue
                        elif show_status == "Past" and event_dt < now:
                            continue
                    except:
                        pass
        
        # Display events
        st.caption(f"Found {len(filtered_events)} events")
        
        for event in filtered_events:
            display_event_card(event, st.session_state.username)
    
    elif selected == "My Registrations":
        st.header("📋 My Registrations")
        
        registrations = db.get_registrations_by_student(st.session_state.username)
        
        if not registrations:
            st.info("You haven't registered for any events yet.")
            return
        
        for reg in registrations:
            with st.container():
                st.markdown('<div class="event-card">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    event_title = reg.get('event_title', 'Unknown Event')
                    st.markdown(f'<div class="card-title">{event_title}</div>', unsafe_allow_html=True)
                    
                    event_date = reg.get('event_date')
                    if event_date:
                        st.caption(f"📅 {format_date(event_date)}")
                    
                    reg_status = reg.get('status', 'pending').title()
                    st.caption(f"📝 Status: {reg_status}")
                
                with col2:
                    points = reg.get('points_awarded', 0)
                    if points > 0:
                        st.markdown(f'<div style="font-size: 1.5rem; font-weight: bold; text-align: center; color: #3B82F6;">{points}</div>', unsafe_allow_html=True)
                        st.caption("Points")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected == "Liked Events":
        st.header("❤️ Liked Events")
        
        liked_events = db.get_student_liked_events(st.session_state.username)
        
        if not liked_events:
            st.info("You haven't liked any events yet.")
            return
        
        for event in liked_events:
            display_event_card(event, st.session_state.username)
    
    elif selected == "Interested Events":
        st.header("⭐ Interested Events")
        
        interested_events = db.get_student_interested_events(st.session_state.username)
        
        if not interested_events:
            st.info("You haven't marked any events as interested yet.")
            return
        
        for event in interested_events:
            display_event_card(event, st.session_state.username)
    
    elif selected == "Leaderboard":
        st.markdown('<h1 class="main-header">🏆 College Leaderboard</h1>', unsafe_allow_html=True)
        
        leaderboard = db.get_leaderboard(limit=15)
        
        if not leaderboard:
            st.info("No students found in leaderboard.")
            return
        
        # Display leaderboard
        for student in leaderboard:
            with st.container():
                st.markdown('<div class="leaderboard-card">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    rank = student.get('rank', 0)
                    if rank <= 3:
                        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
                        st.markdown(f'<div style="font-size: 2rem; text-align: center;">{medals.get(rank, f"{rank}.")}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="font-size: 1.5rem; text-align: center; font-weight: bold;">{rank}.</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'<div style="font-weight: bold; font-size: 1.1rem;">{student.get("name")}</div>', unsafe_allow_html=True)
                    st.caption(f"{student.get('roll_no', '')} | {student.get('department', '')}")
                
                with col3:
                    points = student.get('total_points', 0)
                    st.markdown(f'<div style="font-size: 1.8rem; font-weight: bold; text-align: center; color: #3B82F6;">{points}</div>', unsafe_allow_html=True)
                    st.caption("Points")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
    
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
            st.markdown(f"**Mobile:** {student.get('mobile', 'N/A')}")
            st.markdown(f"**Username:** {student.get('username', 'N/A')}")
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 My Statistics")
        
        points = db.get_student_points(st.session_state.username)
        rank = db.get_student_rank(st.session_state.username)
        registrations = db.get_registrations_by_student(st.session_state.username)
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Events Registered", len(registrations))
        with col_stat2:
            st.metric("🏆 Points", points)
        with col_stat3:
            if rank:
                st.metric("📊 Rank", f"#{rank}")
            else:
                st.metric("📊 Rank", "Not Ranked")

# ============================================
# LANDING PAGE WITH LOGIN - OPTIMIZED
# ============================================

def landing_page():
    """Landing page with app info and login - OPTIMIZED"""
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
        
        **New Features:**
        - 🏆 **Leaderboard:** Compete with other students
        - 🎮 **Points System:** Earn points for participation
        - 🏅 **Badges:** Collect achievements
        - 📊 **Progress Tracking:** Monitor your growth
        
        **Getting Started:**
        1. Select your role
        2. Enter your credentials
        3. Students can register for new accounts
        4. Start exploring events!
        """)
    
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
        
        # Remember me
        remember_me = st.checkbox("Remember Me", help="Stay logged in on this device")
        
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
                    user = db.get_user(username)
                    if user:
                        st.session_state.role = db_role
                        st.session_state.username = username
                        st.session_state.name = user.get('name', username)
                        st.session_state.session_start = datetime.now()
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
# STUDENT REGISTRATION PAGE - OPTIMIZED
# ============================================

def student_registration_page():
    """Student registration page - OPTIMIZED"""
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
            
            # Validate mobile
            is_valid_mobile, mobile_msg = Validators.validate_mobile(mobile)
            if not is_valid_mobile:
                errors.append(mobile_msg)
            
            # Validate password
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
                        
                        # Auto-login
                        st.session_state.role = 'student'
                        st.session_state.username = username
                        st.session_state.name = name
                        st.session_state.session_start = datetime.now()
                        
                        st.balloons()
                        st.info("You have been automatically logged in. Redirecting...")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {message}")

# ============================================
# CUSTOM CSS - OPTIMIZED
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
    
    /* Event Card */
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
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 6px;
        line-height: 1.3;
    }
    
    /* Registration Section */
    .registration-section {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 8px;
        border-radius: 6px;
        margin-top: 8px;
        border-left: 3px solid #3B82F6;
        font-size: 0.9rem;
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
    
    /* Metrics */
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
    
    /* Status badges */
    .status-upcoming {
        background: #D1FAE5;
        color: #065F46;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-ongoing {
        background: #FEF3C7;
        color: #92400E;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-past {
        background: #FEE2E2;
        color: #DC2626;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
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
    
    /* Button improvements */
    .stButton > button {
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION - OPTIMIZED
# ============================================

def main():
    """Main application function - OPTIMIZED"""
    
    # Initialize session state
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'name' not in st.session_state:
        st.session_state.name = None
    if 'page' not in st.session_state:
        st.session_state.page = "login"
    
    # Show database info
    if db.use_supabase:
        st.sidebar.success("✅ Using Supabase PostgreSQL")
    else:
        st.sidebar.info("💾 Using SQLite (Local)")
    
    # Route based on page/role
    if st.session_state.page == "student_register":
        student_registration_page()
    elif st.session_state.role is None:
        landing_page()
    elif st.session_state.role == 'student':
        student_dashboard()
    else:
        # For other roles (admin, faculty, mentor), show student dashboard as example
        # You can add proper implementations for these roles
        st.warning(f"⚠️ {st.session_state.role.title()} dashboard is under development.")
        st.info("For now, you can use the student features.")
        student_dashboard()

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()
