"""
G H Raisoni College - Advanced Event Management System
PRODUCTION READY with Supabase PostgreSQL (Free Forever)
Deployable on Streamlit Cloud
WITH ENHANCED GAMIFICATION & ANALYTICS
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
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import csv
import sqlite3
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from collections import defaultdict
import math

# ============================================
# LOGGING SETUP - ENHANCED
# ============================================

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
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

# Database configuration
USE_SUPABASE = True  # Change this to switch databases
CACHE_ENABLED = True  # Enable caching for better performance
CACHE_TTL = 300  # 5 minutes

# Security settings
SESSION_TIMEOUT_MINUTES = 60
MAX_LOGIN_ATTEMPTS = 5
PASSWORD_MIN_LENGTH = 8
RATE_LIMIT_REQUESTS = 100  # Max requests per minute per user

# College configuration - Enhanced
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
        "Guest Lecture", "Industrial Visit", "Sports Event", "Technical Fest",
        "Placement Drive", "Alumni Talk", "Research Symposium"
    ],
    "event_categories": [
        "Technical", "Cultural", "Sports", "Academic", "Career", 
        "Research", "Innovation", "Social", "Entertainment"
    ]
}

# ============================================
# GAMIFICATION CONFIGURATION - ENHANCED
# ============================================

GAMIFICATION_CONFIG = {
    "points": {
        "registration": 50,
        "attendance": 25,
        "shortlisted": 100,
        "winner": 200,
        "runner_up": 150,
        "participation": 30,
        "event_creation": 20,
        "daily_login": 10,
        "feedback_submission": 15,
        "invite_friend": 50,
        "complete_profile": 100
    },
    "badges": {
        "registration": "🏅 Participant",
        "shortlisted": "⭐ Shortlisted",
        "winner": "🏆 Winner",
        "runner_up": "🥈 Runner Up",
        "top_10": "👑 Top 10 Leader",
        "top_25": "🎖️ Top 25 Achiever",
        "top_50": "🌟 Rising Star",
        "mentor_choice": "💎 Mentor's Choice",
        "most_active": "🚀 Most Active",
        "event_creator": "🎪 Event Organizer",
        "perfect_attendance": "📊 100% Attendance",
        "department_champion": "🏛️ Department Champion",
        "early_bird": "🐦 Early Bird",
        "social_butterfly": "🦋 Social Butterfly",
        "feedback_expert": "💬 Feedback Expert"
    },
    "levels": {
        1: {"name": "Beginner", "points_required": 0},
        2: {"name": "Learner", "points_required": 500},
        3: {"name": "Contributor", "points_required": 1500},
        4: {"name": "Expert", "points_required": 3000},
        5: {"name": "Master", "points_required": 5000},
        6: {"name": "Grand Master", "points_required": 10000},
        7: {"name": "Legend", "points_required": 20000}
    },
    "leaderboard": {
        "top_n": 20,
        "update_interval": 1800,  # 30 minutes
        "department_top_n": 10,
        "monthly_reset": False,
        "seasonal_awards": True
    },
    "streak_bonus": {
        "daily_login": [5, 10, 15, 20, 25, 30, 50],  # Bonus points for consecutive days
        "event_streak": [10, 25, 50, 100]  # Bonus for attending consecutive events
    }
}

# ============================================
# PERFORMANCE CACHE - ENHANCED
# ============================================

class PerformanceCache:
    """Enhanced caching system with TTL and LRU eviction"""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.ttl = {}
        self.access_time = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def get(self, key, default=None):
        """Get item from cache with TTL check"""
        if key in self.cache:
            if time.time() > self.ttl.get(key, 0):
                self.delete(key)
                self.misses += 1
                return default
            
            self.access_time[key] = time.time()
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return default
    
    def set(self, key, value, ttl=300):
        """Set item in cache with TTL"""
        if len(self.cache) >= self.max_size:
            self._evict()
        
        self.cache[key] = value
        self.ttl[key] = time.time() + ttl
        self.access_time[key] = time.time()
    
    def delete(self, key):
        """Delete item from cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.ttl:
            del self.ttl[key]
        if key in self.access_time:
            del self.access_time[key]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.ttl.clear()
        self.access_time.clear()
        self.hits = 0
        self.misses = 0
    
    def _evict(self):
        """LRU eviction policy"""
        if not self.access_time:
            return
        
        # Find least recently used key
        lru_key = min(self.access_time.items(), key=lambda x: x[1])[0]
        self.delete(lru_key)
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }

# Global cache instance
cache = PerformanceCache(max_size=2000)

# ============================================
# VALIDATION CLASS - ENHANCED
# ============================================

class Validators:
    """Collection of input validation methods"""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email format"""
        if not email:
            return False, "Email is required"
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(pattern, email):
            return False, "Invalid email format"
        
        # Check college domain
        if not email.endswith('@ghraisoni.edu'):
            return True, "Valid email (external)"
        
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
        
        # Check for common patterns
        common_patterns = ['123456', 'password', 'admin', 'qwerty', 'letmein']
        if password.lower() in common_patterns:
            return False, "Password is too common"
        
        return True, "Strong password"
    
    @staticmethod
    def format_mobile(mobile: str) -> str:
        """Format mobile number consistently"""
        digits = re.sub(r'\D', '', str(mobile))
        if len(digits) == 10:
            return f"+91 {digits[:5]} {digits[5:]}"
        return mobile
    
    @staticmethod
    def validate_roll_number(roll_no: str) -> Tuple[bool, str]:
        """Validate roll number format"""
        if not roll_no:
            return False, "Roll number required"
        
        pattern = r'^[A-Z]{2,4}\d{4,7}$'
        if re.match(pattern, roll_no, re.IGNORECASE):
            return True, "Valid roll number"
        
        return False, "Invalid roll number format (e.g., CSE2023001)"
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 500) -> str:
        """Sanitize user input"""
        if not text:
            return ""
        
        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\'&;]', '', text)
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()

# ============================================
# SUPABASE CLIENT (PostgreSQL) - FIXED
# ============================================

class SupabaseClient:
    """Fixed Supabase PostgreSQL client with proper error handling"""
    
    def __init__(self):
        self.url = None
        self.key = None
        self.headers = None
        self.is_configured = False
        self._initialized = False
        self.retry_attempts = 3
        self.retry_delay = 1
        
        self._initialize_supabase()
    
    def _initialize_supabase(self):
        """Initialize Supabase connection with retry logic"""
        if self._initialized:
            return
            
        for attempt in range(self.retry_attempts):
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
                            self._initialized = True
                            logger.info(f"✅ Supabase configured successfully (attempt {attempt + 1})")
                            
                            # Test connection
                            self._test_connection()
                            break
                        else:
                            logger.warning(f"⚠️ Supabase credentials incomplete (attempt {attempt + 1})")
                    else:
                        logger.warning(f"⚠️ Supabase credentials not found in secrets (attempt {attempt + 1})")
                else:
                    logger.warning(f"⚠️ Streamlit secrets not available (attempt {attempt + 1})")
                    
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Supabase initialization error (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
    
    def _test_connection(self):
        """Test connection to Supabase"""
        try:
            import requests
            response = requests.get(f"{self.url}/rest/v1/", headers=self.headers, timeout=5)
            if response.status_code == 200:
                logger.info("✅ Supabase connection test successful")
            else:
                logger.warning(f"⚠️ Supabase connection test returned {response.status_code}")
        except Exception as e:
            logger.warning(f"Supabase connection test failed: {e}")
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
    
    def execute_query(self, table, method='GET', data=None, filters=None, limit=1000, 
                     order_by=None, cache_ttl=60, use_cache=True):
        """Execute REST API query to Supabase with proper error handling"""
        # Check cache first
        if method == 'GET' and use_cache and CACHE_ENABLED:
            cache_key = self._get_cache_key(table, method, filters, limit, order_by)
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        try:
            import requests
            
            url = f"{self.url}/rest/v1/{table}"
            
            # Build query parameters
            params = {}
            
            if filters:
                for k, v in filters.items():
                    if v is not None:
                        params[k] = f"eq.{v}"
            
            if order_by:
                params['order'] = order_by
            
            params['limit'] = str(limit)
            
            # Set timeout
            timeout = 10 if method == 'GET' else 30
            
            # Make request
            response = None
            if method == 'GET':
                response = requests.get(url, headers=self.headers, params=params, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=data, params=params, timeout=timeout)
            elif method == 'PATCH':
                response = requests.patch(url, headers=self.headers, json=data, params=params, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers, params=params, timeout=timeout)
            else:
                logger.error(f"Unsupported method: {method}")
                return None
            
            # Check for errors
            if response.status_code >= 400:
                if response.status_code == 400:
                    logger.warning(f"Supabase schema mismatch for table {table}: {response.text[:200]}")
                    return None
                logger.error(f"Supabase API error {response.status_code}: {response.text[:200]}")
                return None
            
            # Process response
            result = None
            if method == 'GET':
                result = response.json() if response.text else []
                # Cache the result
                if use_cache and CACHE_ENABLED:
                    cache_key = self._get_cache_key(table, method, filters, limit, order_by)
                    cache.set(cache_key, result, ttl=cache_ttl)
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
    
    def _get_cache_key(self, table, method, filters, limit, order_by):
        """Generate cache key"""
        import json
        key_data = {
            'table': table,
            'method': method,
            'filters': filters,
            'limit': limit,
            'order_by': order_by
        }
        return f"supabase_{hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()}"
    
    def insert(self, table, data, use_cache=True):
        """Insert data into table"""
        # For Supabase, we need to ensure we only send columns that exist
        # We'll get the table structure first (simplified approach)
        return self.execute_query(table, 'POST', data, use_cache=use_cache)
    
    def select(self, table, filters=None, limit=1000, order_by=None, 
               cache_ttl=60, use_cache=True):
        """Select data from table"""
        return self.execute_query(table, 'GET', filters=filters, limit=limit, 
                                 order_by=order_by, cache_ttl=cache_ttl, use_cache=use_cache)
    
    def update(self, table, filters, data, use_cache=True):
        """Update data in table"""
        return self.execute_query(table, 'PATCH', data, filters, use_cache=use_cache)
    
    def delete(self, table, filters, use_cache=True):
        """Delete data from table"""
        return self.execute_query(table, 'DELETE', filters=filters, use_cache=use_cache)

# ============================================
# SQLITE CLIENT (Fallback) - FIXED
# ============================================

class SQLiteClient:
    """SQLite client for local development"""
    
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
            self.conn.execute("PRAGMA cache_size = -10000")
            
            # Create tables with simplified schema for compatibility
            self._create_compatible_tables()
            
        except Exception as e:
            logger.error(f"SQLite initialization error: {e}")
            raise
    
    def _create_compatible_tables(self):
        """Create tables with simplified schema compatible with existing Supabase"""
        # Users table - SIMPLIFIED VERSION
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
        
        # Events table - SIMPLIFIED VERSION
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
            try:
                self.conn.execute(table_sql)
            except Exception as e:
                logger.warning(f"Error creating table: {e}")
        
        self.conn.commit()
    
    def execute_query(self, query, params=None, fetchone=False, fetchall=False, commit=False, use_cache=True):
        """Execute SQL query"""
        # Check cache for SELECT queries
        if 'SELECT' in query.upper() and fetchall and use_cache and CACHE_ENABLED:
            cache_key = f"sqlite_{hashlib.md5((query + str(params)).encode()).hexdigest()}"
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
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
                row = cursor.fetchone()
                result = dict(row) if row else None
            elif fetchall:
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]
                # Cache the result
                if use_cache and CACHE_ENABLED:
                    cache_key = f"sqlite_{hashlib.md5((query + str(params)).encode()).hexdigest()}"
                    cache.set(cache_key, result, ttl=300)
            else:
                result = cursor
            
            return result
        except Exception as e:
            logger.error(f"SQLite error: {e}")
            return None
    
    def insert(self, table, data, use_cache=True):
        """Insert data into table"""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        try:
            self.execute_query(query, tuple(data.values()), commit=True)
            return True
        except Exception as e:
            logger.error(f"Insert error: {e}")
            return False
    
    def select(self, table, filters=None, limit=1000, cache_ttl=60, use_cache=True):
        """Select data from table"""
        query = f"SELECT * FROM {table}"
        
        if filters:
            conditions = ' AND '.join([f"{k} = ?" for k in filters.keys()])
            query = f"{query} WHERE {conditions}"
        
        query = f"{query} LIMIT {limit}"
        
        if filters:
            return self.execute_query(query, tuple(filters.values()), fetchall=True, use_cache=use_cache)
        else:
            return self.execute_query(query, fetchall=True, use_cache=use_cache)
    
    def update(self, table, filters, data, use_cache=True):
        """Update data in table"""
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        conditions = ' AND '.join([f"{k} = ?" for k in filters.keys()])
        
        query = f"UPDATE {table} SET {set_clause} WHERE {conditions}"
        params = tuple(data.values()) + tuple(filters.values())
        
        try:
            cursor = self.execute_query(query, params)
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Update error: {e}")
            return False
    
    def delete(self, table, filters, use_cache=True):
        """Delete data from table"""
        conditions = ' AND '.join([f"{k} = ?" for k in filters.keys()])
        query = f"DELETE FROM {table} WHERE {conditions}"
        
        try:
            cursor = self.execute_query(query, tuple(filters.values()))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

# ============================================
# UNIFIED DATABASE MANAGER - FIXED
# ============================================

class DatabaseManager:
    """Fixed database manager with proper error handling and compatibility"""
    
    def __init__(self, use_supabase=True):
        self.use_supabase = use_supabase
        self._initialized = False
        
        if self.use_supabase:
            self.client = SupabaseClient()
            if not self.client.is_configured:
                st.warning("⚠️ Supabase not configured. Falling back to SQLite.")
                self.use_supabase = False
                self.client = SQLiteClient()
        else:
            self.client = SQLiteClient()
        
        # Initialize database
        self._add_default_users()
        self._initialized = True
    
    # ============================================
    # USER MANAGEMENT - FIXED
    # ============================================
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials with rate limiting"""
        # Rate limiting check
        rate_key = f"login_attempts_{username}"
        attempts = cache.get(rate_key, 0)
        if attempts >= MAX_LOGIN_ATTEMPTS:
            logger.warning(f"Rate limited: {username}")
            return False
        
        try:
            user = self.get_user(username)
            if not user:
                cache.set(rate_key, attempts + 1, ttl=300)
                return False
            
            if user.get('role') != role:
                cache.set(rate_key, attempts + 1, ttl=300)
                return False
            
            stored_pass = user.get('password', '')
            input_hash = hashlib.sha256(password.encode()).hexdigest().lower()
            
            if stored_pass == input_hash or stored_pass == password:
                # Update last login
                update_data = {
                    'last_login': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'login_attempts': 0
                }
                
                self.client.update('users', {'username': username}, update_data, use_cache=False)
                
                # Clear rate limit
                cache.delete(rate_key)
                
                return True
            
            cache.set(rate_key, attempts + 1, ttl=300)
            return False
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            cache.set(rate_key, attempts + 1, ttl=300)
            return False
    
    def get_user(self, username, use_cache=True):
        """Get user by username with caching"""
        cache_key = f"user_{username}"
        if use_cache and CACHE_ENABLED:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self.use_supabase:
                results = self.client.select('users', {'username': username}, 
                                           limit=1, cache_ttl=30, use_cache=use_cache)
                if results:
                    user = results[0]
                    if use_cache:
                        cache.set(cache_key, user, ttl=300)
                    return user
            else:
                user = self.client.execute_query(
                    "SELECT * FROM users WHERE username = ?",
                    (username,), fetchone=True, use_cache=use_cache
                )
                if user and use_cache:
                    cache.set(cache_key, user, ttl=300)
                return user
        except Exception as e:
            logger.error(f"Error getting user {username}: {e}")
        return None
    
    def add_user(self, user_data):
        """Add new user with simplified schema"""
        try:
            password = user_data.get('password', '')
            if not password:
                return False, "Password is required"
            
            hashed_pass = hashlib.sha256(password.encode()).hexdigest().lower()
            
            # Simplified user record compatible with both databases
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
            
            success = self.client.insert('users', user_record, use_cache=False)
            
            if success:
                # Clear cache
                cache.delete(f"user_{user_record['username']}")
                return True, "User registered successfully"
            else:
                return False, "Registration failed"
            
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return False, str(e)
    
    # ============================================
    # EVENT MANAGEMENT - FIXED
    # ============================================
    
    def add_event(self, event_data):
        """Add new event with simplified schema"""
        try:
            # Simplified event record compatible with both databases
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
            
            success = self.client.insert('events', event_record, use_cache=False)
            
            if success:
                # Clear cache
                cache.delete("events_all")
                cache.delete("events_upcoming")
                return True, event_record['id']
            return False, None
            
        except Exception as e:
            logger.error(f"Error adding event: {e}")
            return False, None
    
    def get_all_events(self, cache_ttl=60, use_cache=True):
        """Get all events with caching"""
        cache_key = "events_all"
        if use_cache and CACHE_ENABLED:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self.use_supabase:
                events = self.client.select('events', limit=1000, 
                                          order_by='event_date.desc', 
                                          cache_ttl=cache_ttl, use_cache=use_cache)
            else:
                events = self.client.execute_query(
                    "SELECT * FROM events ORDER BY event_date DESC",
                    fetchall=True, use_cache=use_cache
                )
            
            events = events or []
            
            if use_cache:
                cache.set(cache_key, events, ttl=cache_ttl)
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    def update_event_status(self):
        """Update event status based on current time - FIXED DATE COMPARISON"""
        try:
            now = datetime.now()
            
            if self.use_supabase:
                # Get all events
                events = self.get_all_events(cache_ttl=0, use_cache=False)
                
                for event in events:
                    event_date = event.get('event_date')
                    event_id = event.get('id')
                    current_status = event.get('status', 'upcoming')
                    
                    if event_date:
                        try:
                            # Parse event date - handle both naive and aware datetimes
                            if isinstance(event_date, str):
                                # Remove timezone info for comparison
                                event_date_str = event_date.replace('Z', '').replace('+00:00', '')
                                # Try different formats
                                for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                                    try:
                                        event_dt = datetime.strptime(event_date_str, fmt)
                                        break
                                    except:
                                        continue
                                else:
                                    # Try ISO format without timezone
                                    event_dt = datetime.fromisoformat(event_date_str)
                            else:
                                event_dt = event_date
                            
                            # Make both datetimes naive for comparison
                            event_dt_naive = event_dt.replace(tzinfo=None) if event_dt.tzinfo else event_dt
                            now_naive = now.replace(tzinfo=None) if now.tzinfo else now
                            
                            # Check if event is past
                            if event_dt_naive < now_naive and current_status != 'past':
                                self.client.update('events', {'id': event_id}, {
                                    'status': 'past',
                                    'updated_at': datetime.now().isoformat()
                                }, use_cache=False)
                            
                            # Check if event is today
                            elif event_dt_naive.date() == now_naive.date() and current_status == 'upcoming':
                                self.client.update('events', {'id': event_id}, {
                                    'status': 'ongoing',
                                    'updated_at': datetime.now().isoformat()
                                }, use_cache=False)
                                
                        except Exception as e:
                            logger.warning(f"Error parsing event date {event_date}: {e}")
                            continue
            else:
                # SQLite version
                now_str = now.isoformat()
                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
                today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()
                
                # Update past events
                self.client.execute_query(
                    "UPDATE events SET status = 'past', updated_at = ? WHERE event_date <= ? AND status != 'past'",
                    (now_str, now_str), commit=True
                )
                
                # Update ongoing events
                self.client.execute_query(
                    "UPDATE events SET status = 'ongoing', updated_at = ? WHERE event_date BETWEEN ? AND ? AND status = 'upcoming'",
                    (now_str, today_start, today_end), commit=True
                )
            
            # Clear cache
            cache.delete("events_all")
            cache.delete("events_upcoming")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating event status: {e}")
            return False
    
    # ============================================
    # REGISTRATION MANAGEMENT - FIXED
    # ============================================
    
    def add_registration(self, reg_data):
        """Add new registration"""
        try:
            # Check if already registered
            existing = self.is_student_registered(reg_data['event_id'], reg_data['student_username'])
            if existing:
                return None, "Already registered"
            
            # Check event capacity
            event = self.get_event_by_id(reg_data['event_id'])
            if event:
                max_participants = event.get('max_participants', 100)
                current_participants = event.get('current_participants', 0)
                
                if current_participants >= max_participants:
                    return None, "Event is full"
            
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
                'status': 'confirmed',
                'registered_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('registrations', registration_record, use_cache=False)
            
            if success:
                # Clear cache
                cache.delete(f"registrations_{reg_data['student_username']}")
                
                # Update participant count
                self._update_event_participant_count(reg_data['event_id'])
                
                # Award registration points
                self.award_points(reg_data['student_username'], 
                                GAMIFICATION_CONFIG['points']['registration'],
                                "event_registration", 
                                f"Registered for: {reg_data['event_title'][:30]}")
                
                return registration_record['id'], "Registration successful"
            
            return None, "Registration failed"
            
        except Exception as e:
            logger.error(f"Error adding registration: {e}")
            return None, "Registration failed"
    
    def _update_event_participant_count(self, event_id):
        """Update event participant count"""
        try:
            if self.use_supabase:
                registrations = self.client.select('registrations', {'event_id': event_id}, 
                                                 cache_ttl=0, use_cache=False)
                count = len(registrations) if registrations else 0
                self.client.update('events', {'id': event_id}, {
                    'current_participants': count,
                    'updated_at': datetime.now().isoformat()
                }, use_cache=False)
            else:
                cursor = self.client.execute_query(
                    "SELECT COUNT(*) as count FROM registrations WHERE event_id = ?",
                    (event_id,), fetchone=True, use_cache=False
                )
                count = cursor['count'] if cursor else 0
                self.client.execute_query(
                    "UPDATE events SET current_participants = ?, updated_at = ? WHERE id = ?",
                    (count, datetime.now().isoformat(), event_id),
                    commit=True
                )
            
            # Clear cache
            cache.delete(f"event_{event_id}")
            cache.delete("events_all")
            
            return True
        except Exception as e:
            logger.error(f"Error updating participant count: {e}")
            return False
    
    def get_registrations_by_student(self, username, cache_ttl=60):
        """Get all registrations for a student"""
        cache_key = f"registrations_{username}"
        if CACHE_ENABLED:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self.use_supabase:
                registrations = self.client.select('registrations', {'student_username': username}, 
                                                 cache_ttl=cache_ttl)
            else:
                registrations = self.client.execute_query(
                    "SELECT r.*, e.event_date, e.venue, e.status as event_status FROM registrations r LEFT JOIN events e ON r.event_id = e.id WHERE r.student_username = ? ORDER BY r.registered_at DESC",
                    (username,), fetchall=True
                )
            
            registrations = registrations or []
            
            if CACHE_ENABLED:
                cache.set(cache_key, registrations, ttl=cache_ttl)
            
            return registrations
            
        except Exception as e:
            logger.error(f"Error getting registrations: {e}")
            return []
    
    # ============================================
    # GAMIFICATION & POINTS SYSTEM - SIMPLIFIED
    # ============================================
    
    def award_points(self, username, points, reason, description=""):
        """Award points to a user"""
        try:
            user = self.get_user(username, use_cache=False)
            if not user:
                return False
            
            current_points = user.get('total_points', 0)
            new_points = current_points + points
            
            # Update user points
            update_data = {
                'total_points': new_points,
                'updated_at': datetime.now().isoformat()
            }
            
            success = self.client.update('users', {'username': username}, update_data, use_cache=False)
            
            if success:
                # Clear user cache
                cache.delete(f"user_{username}")
                cache.delete(f"user_stats_{username}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error awarding points: {e}")
            return False
    
    def get_student_points(self, username):
        """Get student's total points"""
        user = self.get_user(username)
        if user:
            return user.get('total_points', 0)
        return 0
    
    # ============================================
    # LEADERBOARD - FIXED
    # ============================================
    
    def get_leaderboard(self, limit=20, department=None, cache_ttl=300):
        """Get leaderboard of top students - FIXED"""
        cache_key = f"leaderboard_{department}_{limit}"
        if CACHE_ENABLED:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self.use_supabase:
                filters = {'role': 'student'}
                if department:
                    filters['department'] = department
                
                users = self.client.select('users', filters, limit=limit * 2, 
                                         order_by='total_points.desc', 
                                         cache_ttl=cache_ttl)
                
                if not users:
                    if CACHE_ENABLED:
                        cache.set(cache_key, [], ttl=cache_ttl)
                    return []
                
                # Calculate rank
                users = sorted(users, key=lambda x: x.get('total_points', 0), reverse=True)
                users = users[:limit]
                
                for i, user in enumerate(users, 1):
                    user['rank'] = i
                    user['points'] = user.get('total_points', 0)
                
                result = users
                
            else:
                query = "SELECT * FROM users WHERE role = 'student'"
                params = []
                
                if department:
                    query += " AND department = ?"
                    params.append(department)
                
                query += " ORDER BY total_points DESC, name ASC LIMIT ?"
                params.append(limit)
                
                results = self.client.execute_query(query, tuple(params), fetchall=True) or []
                
                # Add rank
                for i, result in enumerate(results, 1):
                    result['rank'] = i
                    result['points'] = result.get('total_points', 0)
                
                result = results
            
            if CACHE_ENABLED:
                cache.set(cache_key, result, ttl=cache_ttl)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []
    
    def get_student_rank(self, username):
        """Get student's rank in leaderboard"""
        leaderboard = self.get_leaderboard(limit=1000)
        for i, student in enumerate(leaderboard, 1):
            if student['username'] == username:
                return i
        return None
    
    # ============================================
    # LIKES & INTEREST METHODS - FIXED
    # ============================================
    
    def add_like(self, event_id, student_username):
        """Add a like for an event"""
        try:
            # Check if already liked
            if self.is_event_liked(event_id, student_username):
                return False
            
            like_record = {
                'id': str(uuid.uuid4()),
                'event_id': event_id,
                'student_username': student_username,
                'liked_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('event_likes', like_record, use_cache=False)
            
            if success:
                # Clear cache
                cache.delete(f"liked_{event_id}_{student_username}")
                cache.delete(f"likes_count_{event_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error adding like: {e}")
            return False
    
    def remove_like(self, event_id, student_username):
        """Remove a like for an event"""
        try:
            success = self.client.delete('event_likes', {
                'event_id': event_id,
                'student_username': student_username
            }, use_cache=False)
            
            if success:
                # Clear cache
                cache.delete(f"liked_{event_id}_{student_username}")
                cache.delete(f"likes_count_{event_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing like: {e}")
            return False
    
    def is_event_liked(self, event_id, student_username):
        """Check if student liked an event"""
        cache_key = f"liked_{event_id}_{student_username}"
        if CACHE_ENABLED:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self.use_supabase:
                results = self.client.select('event_likes', {
                    'event_id': event_id,
                    'student_username': student_username
                }, limit=1, use_cache=False)
                result = bool(results)
            else:
                result = self.client.execute_query(
                    "SELECT id FROM event_likes WHERE event_id = ? AND student_username = ?",
                    (event_id, student_username), fetchone=True, use_cache=False
                )
                result = result is not None
            
            if CACHE_ENABLED:
                cache.set(cache_key, result, ttl=300)
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking like: {e}")
            return False
    
    def get_event_likes_count(self, event_id):
        """Get total likes for an event"""
        cache_key = f"likes_count_{event_id}"
        if CACHE_ENABLED:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self.use_supabase:
                likes = self.client.select('event_likes', {'event_id': event_id}, use_cache=False)
                count = len(likes) if likes else 0
            else:
                result = self.client.execute_query(
                    "SELECT COUNT(*) as count FROM event_likes WHERE event_id = ?",
                    (event_id,), fetchone=True, use_cache=False
                )
                count = result['count'] if result else 0
            
            if CACHE_ENABLED:
                cache.set(cache_key, count, ttl=60)
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting likes count: {e}")
            return 0
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def get_event_by_id(self, event_id):
        """Get event by ID"""
        cache_key = f"event_{event_id}"
        if CACHE_ENABLED:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self.use_supabase:
                results = self.client.select('events', {'id': event_id}, limit=1)
                event = results[0] if results else None
            else:
                event = self.client.execute_query(
                    "SELECT * FROM events WHERE id = ?",
                    (event_id,), fetchone=True
                )
            
            if event and CACHE_ENABLED:
                cache.set(cache_key, event, ttl=300)
            
            return event
            
        except Exception as e:
            logger.error(f"Error getting event: {e}")
            return None
    
    def is_student_registered(self, event_id, username):
        """Check if student is registered for event"""
        cache_key = f"registered_{event_id}_{username}"
        if CACHE_ENABLED:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self.use_supabase:
                results = self.client.select('registrations', {
                    'event_id': event_id,
                    'student_username': username
                }, limit=1, use_cache=False)
                result = bool(results)
            else:
                result = self.client.execute_query(
                    "SELECT id FROM registrations WHERE event_id = ? AND student_username = ?",
                    (event_id, username), fetchone=True, use_cache=False
                )
                result = result is not None
            
            if CACHE_ENABLED:
                cache.set(cache_key, result, ttl=60)
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking registration: {e}")
            return False
    
    def get_student_liked_events(self, student_username):
        """Get all events liked by a student"""
        try:
            if self.use_supabase:
                # Get all liked event IDs first
                likes = self.client.select('event_likes', {'student_username': student_username}, cache_ttl=60)
                if not likes:
                    return []
                
                # Get all events
                all_events = self.get_all_events(cache_ttl=60)
                if not all_events:
                    return []
                
                # Filter events that are liked
                liked_event_ids = {like['event_id'] for like in likes}
                liked_events = [event for event in all_events if event.get('id') in liked_event_ids]
                
                # Sort by most recent like (we'll approximate by event date)
                liked_events.sort(key=lambda x: x.get('event_date', ''), reverse=True)
                
                return liked_events
            else:
                return self.client.execute_query(
                    "SELECT e.* FROM events e JOIN event_likes l ON e.id = l.event_id WHERE l.student_username = ? ORDER BY e.event_date DESC",
                    (student_username,), fetchall=True, cache_ttl=60
                ) or []
        except Exception as e:
            logger.error(f"Error getting liked events: {e}")
            return []
    
    def get_student_interested_events(self, student_username):
        """Get all events student is interested in"""
        try:
            if self.use_supabase:
                # Get all interested event IDs first
                interests = self.client.select('event_interested', {'student_username': student_username}, cache_ttl=60)
                if not interests:
                    return []
                
                # Get all events
                all_events = self.get_all_events(cache_ttl=60)
                if not all_events:
                    return []
                
                # Filter events that are interested
                interested_event_ids = {interest['event_id'] for interest in interests}
                interested_events = [event for event in all_events if event.get('id') in interested_event_ids]
                
                # Sort by most recent interest (approximate by event date)
                interested_events.sort(key=lambda x: x.get('event_date', ''), reverse=True)
                
                return interested_events
            else:
                return self.client.execute_query(
                    "SELECT e.* FROM events e JOIN event_interested i ON e.id = i.event_id WHERE i.student_username = ? ORDER BY e.event_date DESC",
                    (student_username,), fetchall=True, cache_ttl=60
                ) or []
        except Exception as e:
            logger.error(f"Error getting interested events: {e}")
            return []
    
    # ============================================
    # DEFAULT USERS - FIXED
    # ============================================
    
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
                },
                {
                    'id': '00000000-0000-0000-0000-000000000003',
                    'username': 'mentor@raisoni',
                    'password': 'Mentor@12345',
                    'name': 'Senior Mentor',
                    'role': 'mentor',
                    'email': 'mentor@ghraisoni.edu',
                    'department': 'Mentorship'
                }
            ]
            
            for user_data in default_users:
                existing = self.get_user(user_data['username'], use_cache=False)
                if not existing:
                    success, message = self.add_user(user_data)
                    if success:
                        logger.info(f"Added default user: {user_data['username']}")
            
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
                },
                {
                    'name': 'Amit Kumar',
                    'username': 'amit@student',
                    'password': 'Student@123',
                    'roll_no': 'ECE2023003',
                    'department': 'Electronics & Communication',
                    'year': 'IV',
                    'email': 'amit.kumar@ghraisoni.edu',
                    'mobile': '9876543212'
                }
            ]
            
            for student in default_students:
                existing = self.get_user(student['username'], use_cache=False)
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
                        # Award some initial points
                        self.award_points(student['username'], 250, "welcome", "Welcome bonus")
                        logger.info(f"Added default student: {student['username']}")
            
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
# HELPER FUNCTIONS - FIXED
# ============================================

@st.cache_data(ttl=300)
def format_date(date_str):
    """Format date for display"""
    try:
        if isinstance(date_str, str):
            # Handle different date formats
            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime("%d %b %Y, %I:%M %p")
                except:
                    continue
            
            # Try ISO format
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime("%d %b %Y, %I:%M %p")
        else:
            dt = date_str
            return dt.strftime("%d %b %Y, %I:%M %p")
    except:
        return str(date_str)

def get_event_status(event_date, status=None):
    """Get event status badge - FIXED DATE COMPARISON"""
    try:
        if status and status in ['cancelled', 'completed', 'past']:
            if status == 'cancelled':
                return '<span style="background: #F3F4F6; color: #6B7280; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">⚫ Cancelled</span>'
            elif status == 'completed':
                return '<span style="background: #E5E7EB; color: #374151; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🔴 Completed</span>'
            else:
                return '<span style="background: #FEE2E2; color: #DC2626; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🔴 Past</span>'
        
        if isinstance(event_date, str):
            # Try to parse date
            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    dt = datetime.strptime(event_date, fmt)
                    break
                except:
                    continue
            else:
                # Try ISO format
                dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        else:
            dt = event_date
        
        now = datetime.now()
        
        # Make both datetimes naive for comparison
        dt_naive = dt.replace(tzinfo=None) if hasattr(dt, 'tzinfo') and dt.tzinfo else dt
        now_naive = now.replace(tzinfo=None) if now.tzinfo else now
        
        if dt_naive > now_naive:
            days_diff = (dt_naive - now_naive).days
            if days_diff <= 7:
                return '<span style="background: #D1FAE5; color: #065F46; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🟢 Soon</span>'
            else:
                return '<span style="background: #D1FAE5; color: #065F46; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🟢 Upcoming</span>'
        elif dt_naive.date() == now_naive.date():
            return '<span style="background: #FEF3C7; color: #92400E; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🟡 Today</span>'
        else:
            return '<span style="background: #FEE2E2; color: #DC2626; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🔴 Past</span>'
    except Exception as e:
        logger.warning(f"Error in get_event_status: {e}")
        return '<span style="background: #E5E7EB; color: #374151; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">Unknown</span>'

def save_flyer_image(uploaded_file):
    """Save uploaded flyer image and return base64 string"""
    if uploaded_file is None:
        return None
    
    try:
        image_bytes = uploaded_file.getvalue()
        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            st.warning("Image size too large. Please use images under 10MB.")
            return None
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml'
        }
        
        mime_type = mime_types.get(file_ext, 'image/jpeg')
        return f"data:{mime_type};base64,{image_base64}"
    except Exception as e:
        logger.error(f"Error processing flyer image: {e}")
        return None

# ============================================
# EVENT CARD DISPLAY - FIXED
# ============================================

def display_event_card(event, current_user=None, show_actions=True):
    """Display event card with all interactions - FIXED"""
    if not event or not event.get('id'):
        return
    
    event_id = event.get('id')
    
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        # Create columns layout
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
            status = event.get('status')
            status_html = get_event_status(event_date, status)
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
            
            # Engagement metrics
            likes_count = db.get_event_likes_count(event_id)
            interested_count = 0  # Placeholder - you need to implement this function
            
            # Create buttons container
            if current_user and show_actions:
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
                                time.sleep(0.3)
                                st.rerun()
                        else:
                            if db.add_like(event_id, current_user):
                                st.success("Event liked!")
                                time.sleep(0.3)
                                st.rerun()
                
                with button_col2:
                    # Interested button (simplified - always shows Interested)
                    interested_btn_text = "☆ Interested"
                    
                    if st.button(interested_btn_text, key=f"interested_{event_id}_{current_user}",
                               use_container_width=True, type="primary",
                               help="Mark as interested"):
                        st.info("Interested feature coming soon!")
                
                with button_col3:
                    # Share button
                    if st.button("📤 Share", key=f"share_{event_id}_{current_user}",
                               use_container_width=True, type="secondary",
                               help="Share this event"):
                        event_title = event.get('title', 'Cool Event')
                        share_text = f"Check out '{event_title}' at G H Raisoni College Event Manager!\n\nEvent Date: {formatted_date}\nVenue: {venue}\n\nJoin the platform to discover more events!"
                        
                        # Create copy to clipboard functionality
                        st.code(share_text)
                        st.info("📋 Copy the text above to share")
            
            # Show engagement counts
            st.caption(f"❤️ {likes_count} Likes")
            
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
        
        # Registration Section
        if current_user and st.session_state.get('role') == 'student' and show_actions:
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
                                                     {'status': 'confirmed', 'updated_at': datetime.now().isoformat()},
                                                     use_cache=False)
                        else:
                            cursor = db.client.conn.cursor()
                            cursor.execute("UPDATE registrations SET status = 'confirmed', updated_at = ? WHERE event_id = ? AND student_username = ?",
                                         (datetime.now().isoformat(), event_id, current_user))
                            db.client.conn.commit()
                            success = cursor.rowcount > 0
                        
                        if success:
                            st.success("✅ External registration recorded!")
                            time.sleep(0.3)
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
                                time.sleep(0.3)
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
# STUDENT DASHBOARD - FIXED
# ============================================

def student_dashboard():
    """Student dashboard with all features - FIXED"""
    # Sidebar
    st.sidebar.title("👨‍🎓 Student Panel")
    
    # Student info
    student = db.get_user(st.session_state.username)
    if student:
        st.sidebar.markdown(f"**User:** {student.get('name', st.session_state.username)}")
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
        
        # Get counts
        registrations = db.get_registrations_by_student(st.session_state.username)
        liked_events = db.get_student_liked_events(st.session_state.username)
        points = db.get_student_points(st.session_state.username)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 Events", len(registrations))
        with col2:
            st.metric("🏆 Points", points)
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            # Clear session
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    selected = st.session_state.student_page
    
    if selected == "Events Feed":
        events_feed_page()
    
    elif selected == "My Registrations":
        my_registrations_page()
    
    elif selected == "Liked Events":
        liked_events_page()
    
    elif selected == "Interested Events":
        interested_events_page()
    
    elif selected == "Leaderboard":
        leaderboard_page()
    
    elif selected == "My Profile":
        profile_page()

def events_feed_page():
    """Events feed page with enhanced filtering"""
    st.markdown('<h1 class="main-header">🎯 Discover Events</h1>', unsafe_allow_html=True)
    
    # Update event status
    try:
        db.update_event_status()
    except:
        pass
    
    # Get events
    with st.spinner("Loading events..."):
        events = db.get_all_events(cache_ttl=60)
    
    if not events:
        st.info("No events found. Check back later!")
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
        now = datetime.now()
        for event in filtered_events[:]:
            event_date = event.get('event_date')
            event_status = event.get('status', 'upcoming')
            
            if show_status == "Upcoming" and (event_status != 'upcoming' or 
                                             (isinstance(event_date, str) and 
                                              datetime.fromisoformat(event_date.replace('Z', '+00:00')) <= now)):
                filtered_events.remove(event)
            elif show_status == "Ongoing" and event_status != 'ongoing':
                filtered_events.remove(event)
            elif show_status == "Past" and event_status not in ['completed', 'past']:
                filtered_events.remove(event)
    
    # Display events
    st.caption(f"Found {len(filtered_events)} events")
    
    for event in filtered_events:
        display_event_card(event, st.session_state.username)

def my_registrations_page():
    """My registrations page"""
    st.header("📋 My Registrations")
    
    registrations = db.get_registrations_by_student(st.session_state.username)
    
    if not registrations:
        st.info("You haven't registered for any events yet.")
        st.markdown("""
        **Get started:**
        1. Go to **Events Feed**
        2. Find interesting events
        3. Click **Register** to participate
        4. Your registrations will appear here
        """)
        return
    
    # Display registrations
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

def liked_events_page():
    """Liked events page"""
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
        return
    
    st.caption(f"Total liked events: {len(liked_events)}")
    
    for event in liked_events:
        display_event_card(event, st.session_state.username)

def interested_events_page():
    """Interested events page - SIMPLIFIED"""
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
        return
    
    st.caption(f"Total interested events: {len(interested_events)}")
    
    for event in interested_events:
        display_event_card(event, st.session_state.username)

def leaderboard_page():
    """Leaderboard page - FIXED"""
    st.markdown('<h1 class="main-header">🏆 College Leaderboard</h1>', unsafe_allow_html=True)
    
    # Department selector
    department = st.selectbox("Department", ["All Departments"] + COLLEGE_CONFIG['departments'])
    dept_param = department if department != "All Departments" else None
    
    # Get leaderboard
    leaderboard = db.get_leaderboard(limit=15, department=dept_param)
    
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
                points = student.get('points', student.get('total_points', 0))
                st.markdown(f'<div style="font-size: 1.8rem; font-weight: bold; text-align: center; color: #3B82F6;">{points}</div>', unsafe_allow_html=True)
                st.caption("Points")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

def profile_page():
    """Profile page - SIMPLIFIED"""
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
# LANDING PAGE WITH LOGIN - FIXED
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
# STUDENT REGISTRATION PAGE - FIXED
# ============================================

def student_registration_page():
    """Student registration page - SIMPLIFIED"""
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
# CUSTOM CSS - SIMPLIFIED
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
</style>
""", unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION - FIXED
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
    if 'student_page' not in st.session_state:
        st.session_state.student_page = "Events Feed"
    
    # Show database info in sidebar
    st.sidebar.title("System Info")
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
        # For other roles, show student dashboard as example
        st.warning(f"⚠️ {st.session_state.role.title()} dashboard is under development.")
        st.info("For now, you can use the student features.")
        student_dashboard()

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()
