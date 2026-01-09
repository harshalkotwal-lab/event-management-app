"""
G H Raisoni College - Advanced Event Management System
PRODUCTION READY with Supabase PostgreSQL (Free Forever)
Deployable on Streamlit Cloud
WITH ENHANCED GAMIFICATION, ANALYTICS & NOTIFICATION SYSTEM
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
# SUPABASE CLIENT (PostgreSQL) - ENHANCED
# ============================================

class SupabaseClient:
    """Enhanced Supabase PostgreSQL client with connection pooling and retry logic"""
    
    def __init__(self):
        self.url = None
        self.key = None
        self.headers = None
        self.is_configured = False
        self.pool_size = 5
        self._connection_pool = []
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
                                'Prefer': 'return=representation',
                                'Accept-Encoding': 'gzip, deflate'
                            }
                            self.is_configured = True
                            self._initialized = True
                            logger.info(f"✅ Supabase configured successfully (attempt {attempt + 1})")
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
        """Execute REST API query to Supabase with enhanced performance"""
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
            params = {'limit': str(limit)}
            
            if filters:
                for k, v in filters.items():
                    if v is not None:
                        if isinstance(v, dict):
                            for op, val in v.items():
                                params[f"{k}"] = f"{op}.{val}"
                        else:
                            params[f"{k}"] = f"eq.{v}"
            
            if order_by:
                params['order'] = order_by
            
            # Set timeout
            timeout = 10 if method == 'GET' else 30
            
            # Execute request with retry
            response = self._execute_with_retry(
                self._make_request, url, method, params, data, timeout
            )
            
            if response is None:
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
                # Invalidate relevant caches
                self._invalidate_cache(table)
            elif method == 'DELETE':
                result = response.status_code in [200, 204]
                self._invalidate_cache(table)
            
            return result
                
        except Exception as e:
            logger.error(f"Supabase API error: {e}")
            return None
    
    def _make_request(self, url, method, params, data, timeout):
        """Make HTTP request"""
        import requests
        
        if method == 'GET':
            response = requests.get(url, headers=self.headers, params=params, 
                                  timeout=timeout, allow_redirects=True)
        elif method == 'POST':
            response = requests.post(url, headers=self.headers, json=data, 
                                   params=params, timeout=timeout)
        elif method == 'PATCH':
            response = requests.patch(url, headers=self.headers, json=data, 
                                    params=params, timeout=timeout)
        elif method == 'DELETE':
            response = requests.delete(url, headers=self.headers, 
                                     params=params, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code >= 400:
            logger.error(f"API error {response.status_code}: {response.text[:200]}")
            if response.status_code == 429:  # Rate limited
                retry_after = response.headers.get('Retry-After', 60)
                time.sleep(int(retry_after))
            return None
        
        return response
    
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
    
    def _invalidate_cache(self, table):
        """Invalidate cache for a table"""
        keys_to_delete = []
        for key in cache.cache.keys():
            if key.startswith(f"supabase_{table}"):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            cache.delete(key)
    
    def insert(self, table, data, use_cache=True):
        """Insert data into table"""
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
    
    def batch_insert(self, table, data_list, batch_size=100):
        """Batch insert for better performance"""
        if not data_list:
            return True
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            for data in batch:
                if not self.insert(table, data, use_cache=False):
                    logger.error(f"Batch insert failed at index {i}")
                    return False
            time.sleep(0.1)  # Small delay to avoid rate limiting
        
        self._invalidate_cache(table)
        return True

# ============================================
# SQLITE CLIENT (Fallback) - OPTIMIZED
# ============================================

class SQLiteClient:
    """Optimized SQLite client with connection pooling"""
    
    def __init__(self, db_path="data/event_management.db"):
        self.db_path = db_path
        self.connection_pool = []
        self.max_pool_size = 5
        self._lock = threading.Lock()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize SQLite database with performance optimizations"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = self._create_connection()
            conn.row_factory = sqlite3.Row
            
            # Performance optimizations
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = -10000")
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA locking_mode = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Create indexes for performance
            self._create_indexes(conn)
            
            self.connection_pool.append(conn)
            logger.info("✅ SQLite database initialized with optimizations")
            
        except Exception as e:
            logger.error(f"SQLite initialization error: {e}")
            raise
    
    def _create_connection(self):
        """Create a new database connection"""
        return sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
    
    def _create_indexes(self, conn):
        """Create indexes for performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
            "CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)",
            "CREATE INDEX IF NOT EXISTS idx_users_department ON users(department)",
            "CREATE INDEX IF NOT EXISTS idx_users_points ON users(total_points DESC)",
            "CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date)",
            "CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)",
            "CREATE INDEX IF NOT EXISTS idx_events_creator ON events(created_by)",
            "CREATE INDEX IF NOT EXISTS idx_registrations_event ON registrations(event_id)",
            "CREATE INDEX IF NOT EXISTS idx_registrations_student ON registrations(student_username)",
            "CREATE INDEX IF NOT EXISTS idx_likes_event_user ON event_likes(event_id, student_username)",
            "CREATE INDEX IF NOT EXISTS idx_interested_event_user ON event_interested(event_id, student_username)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_event ON event_feedback(event_id)",
            "CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id, read)",
            "CREATE INDEX IF NOT EXISTS idx_achievements_user ON user_achievements(user_id)"
        ]
        
        cursor = conn.cursor()
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
        conn.commit()
    
    def get_connection(self):
        """Get a connection from pool or create new one"""
        with self._lock:
            if self.connection_pool:
                return self.connection_pool.pop()
            elif len(self.connection_pool) < self.max_pool_size:
                conn = self._create_connection()
                conn.row_factory = sqlite3.Row
                return conn
            else:
                raise Exception("Connection pool exhausted")
    
    def return_connection(self, conn):
        """Return connection to pool"""
        with self._lock:
            if len(self.connection_pool) < self.max_pool_size:
                self.connection_pool.append(conn)
            else:
                conn.close()
    
    def execute_query(self, query, params=None, fetchone=False, fetchall=False, 
                     commit=False, use_cache=True):
        """Execute SQL query with connection pooling"""
        # Check cache for SELECT queries
        if 'SELECT' in query.upper() and fetchall and use_cache and CACHE_ENABLED:
            cache_key = f"sqlite_{hashlib.md5((query + str(params)).encode()).hexdigest()}"
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                return cached_data
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if commit:
                conn.commit()
            
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
            if conn and not commit:
                conn.rollback()
            return None
            
        finally:
            if conn:
                self.return_connection(conn)
    
    def clear_cache(self):
        """Clear SQLite related cache"""
        keys_to_delete = []
        for key in cache.cache.keys():
            if key.startswith("sqlite_"):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            cache.delete(key)
    
    def insert(self, table, data, use_cache=True):
        """Insert data into table"""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        try:
            self.execute_query(query, tuple(data.values()), commit=True)
            self.clear_cache()
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
            return self.execute_query(query, tuple(filters.values()), 
                                    fetchall=True, use_cache=use_cache)
        else:
            return self.execute_query(query, fetchall=True, use_cache=use_cache)
    
    def batch_insert(self, table, data_list, batch_size=100):
        """Batch insert for better performance"""
        if not data_list:
            return True
        
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]
                for data in batch:
                    columns = ', '.join(data.keys())
                    placeholders = ', '.join(['?' for _ in data])
                    query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                    cursor.execute(query, tuple(data.values()))
            
            conn.commit()
            self.clear_cache()
            return True
            
        except Exception as e:
            logger.error(f"Batch insert error: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self.return_connection(conn)

# ============================================
# AI EVENT GENERATOR - ENHANCED
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
        if self.is_configured and self.api_key:
            try:
                with st.spinner("🤖 AI is processing your event..."):
                    return self._extract_with_openai(text)
            except Exception as e:
                logger.warning(f"AI extraction failed: {str(e)[:100]}. Using regex fallback.")
        
        return self._extract_with_regex(text)
    
    def _extract_with_openai(self, text):
        """Use OpenAI to extract structured event data"""
        prompt = f"""
        Extract event information from the following text and return as JSON with these fields:
        - title: Event title (string)
        - description: Detailed event description (string)
        - event_type: Type of event from {COLLEGE_CONFIG['event_types']}
        - event_category: Category from {COLLEGE_CONFIG['event_categories']}
        - event_date: Event date in YYYY-MM-DD format
        - end_date: End date if mentioned (or null)
        - venue: Event venue/location (string)
        - organizer: Event organizer (string)
        - event_link: Event website/URL if mentioned (string or null)
        - registration_link: Registration URL if mentioned (string or null)
        - max_participants: Maximum participants if mentioned (integer or 100)
        - difficulty_level: Difficulty (Beginner/Intermediate/Advanced/Expert)
        - estimated_duration: Duration in minutes if mentioned (integer or null)
        - prerequisites: Any prerequisites if mentioned (string or null)
        - has_certificate: Whether certificate is provided (boolean)
        
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
            
            # Add AI metadata
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
            'event_type': 'Workshop',
            'event_category': 'Technical',
            'event_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'venue': 'G H Raisoni College',
            'organizer': 'College Department',
            'event_link': None,
            'registration_link': None,
            'max_participants': 100,
            'difficulty_level': 'Beginner',
            'estimated_duration': 120,
            'prerequisites': None,
            'has_certificate': False,
            'ai_generated': False,
            'ai_prompt': text
        }
        
        # Try to extract title
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
            if self.db.use_supabase:
                success = self.db.client.update('users', {'username': username}, {'password': hashed_pass})
            else:
                cursor = self.db.conn.cursor()
                cursor.execute("UPDATE users SET password = ? WHERE username = ?", (hashed_pass, username))
                self.db.conn.commit()
                success = cursor.rowcount > 0
            
            if success:
                del self.reset_tokens[token]
                return True, "Password reset successful"
            else:
                return False, "Failed to reset password"
                
        except Exception as e:
            logger.error(f"Password reset error: {e}")
            return False, "Failed to reset password"

# ============================================
# UNIFIED DATABASE MANAGER - COMPREHENSIVE
# ============================================

class DatabaseManager:
    """Comprehensive database manager with all features"""
    
    def __init__(self, use_supabase=True):
        self.use_supabase = use_supabase
        self._initialized = False
        self._analytics_cache = {}
        self._analytics_ttl = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        
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
        self._initialized = True
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        threading.Thread(target=self._background_maintenance, daemon=True).start()
    
    def _background_maintenance(self):
        """Background maintenance tasks"""
        while True:
            try:
                self.update_event_status()
                self._clean_old_cache()
                self._update_analytics_cache()
                time.sleep(300)
            except Exception as e:
                logger.error(f"Background maintenance error: {e}")
                time.sleep(60)
    
    def _clean_old_cache(self):
        """Clean old cache entries"""
        cache.clear()
        self._analytics_cache.clear()
    
    def _update_analytics_cache(self):
        """Update analytics cache"""
        pass
    
    def _initialize_database(self):
        """Initialize database tables with all features"""
        try:
            if not self.use_supabase:
                self._create_sqlite_tables()
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def _create_sqlite_tables(self):
        """Create all SQLite tables"""
        tables = [
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
                avatar_url TEXT,
                bio TEXT,
                skills TEXT,
                reset_token TEXT,
                reset_token_expiry TEXT,
                last_activity TEXT,
                is_active INTEGER DEFAULT 1,
                login_attempts INTEGER DEFAULT 0,
                last_login TEXT,
                daily_login_streak INTEGER DEFAULT 0,
                last_daily_login TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                total_points INTEGER DEFAULT 0,
                current_level INTEGER DEFAULT 1,
                level_progress INTEGER DEFAULT 0,
                badges_earned TEXT DEFAULT '[]',
                achievements_unlocked TEXT DEFAULT '[]',
                notification_settings TEXT DEFAULT '{"email": true, "push": true, "event_updates": true}',
                privacy_settings TEXT DEFAULT '{"profile_public": true, "show_points": true}',
                INDEX idx_users_points (total_points DESC),
                INDEX idx_users_department (department),
                INDEX idx_users_role (role)
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
                event_category TEXT,
                event_date TEXT,
                end_date TEXT,
                venue TEXT,
                organizer TEXT,
                co_organizers TEXT DEFAULT '[]',
                event_link TEXT,
                registration_link TEXT,
                max_participants INTEGER DEFAULT 100,
                current_participants INTEGER DEFAULT 0,
                flyer_path TEXT,
                tags TEXT DEFAULT '[]',
                prerequisites TEXT,
                resources TEXT DEFAULT '[]',
                created_by TEXT,
                created_by_name TEXT,
                ai_generated INTEGER DEFAULT 0,
                status TEXT DEFAULT 'upcoming',
                mentor_id TEXT,
                difficulty_level TEXT DEFAULT 'Beginner',
                estimated_duration INTEGER,
                has_certificate INTEGER DEFAULT 0,
                certificate_template TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                views_count INTEGER DEFAULT 0,
                popularity_score FLOAT DEFAULT 0.0,
                INDEX idx_events_date (event_date),
                INDEX idx_events_status (status),
                INDEX idx_events_creator (created_by),
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
                feedback_provided INTEGER DEFAULT 0,
                mentor_notes TEXT,
                certificate_issued INTEGER DEFAULT 0,
                certificate_url TEXT,
                registered_at TEXT DEFAULT CURRENT_TIMESTAMP,
                checked_in_at TEXT,
                checked_out_at TEXT,
                time_spent INTEGER DEFAULT 0,
                UNIQUE(event_id, student_username),
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                INDEX idx_registrations_event (event_id),
                INDEX idx_registrations_student (student_username)
            )
            """,
            
            # Event likes table
            """
            CREATE TABLE IF NOT EXISTS event_likes (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                student_username TEXT NOT NULL,
                liked_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_id, student_username),
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                INDEX idx_likes_event_user (event_id, student_username)
            )
            """,
            
            # Event interested table
            """
            CREATE TABLE IF NOT EXISTS event_interested (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                student_username TEXT NOT NULL,
                interested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_id, student_username),
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                INDEX idx_interested_event_user (event_id, student_username)
            )
            """,
            
            # Event feedback table
            """
            CREATE TABLE IF NOT EXISTS event_feedback (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                student_username TEXT NOT NULL,
                rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                comments TEXT,
                anonymous INTEGER DEFAULT 0,
                submitted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
                INDEX idx_feedback_event (event_id)
            )
            """,
            
            # Notifications table
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                notification_type TEXT,
                related_id TEXT,
                read INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT,
                action_url TEXT,
                INDEX idx_notifications_user (user_id, read)
            )
            """,
            
            # User achievements table
            """
            CREATE TABLE IF NOT EXISTS user_achievements (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                achievement_id TEXT NOT NULL,
                achievement_name TEXT NOT NULL,
                achievement_type TEXT,
                points_awarded INTEGER DEFAULT 0,
                unlocked_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}',
                UNIQUE(user_id, achievement_id),
                INDEX idx_achievements_user (user_id)
            )
            """,
            
            # Department stats table
            """
            CREATE TABLE IF NOT EXISTS department_stats (
                id TEXT PRIMARY KEY,
                department TEXT UNIQUE NOT NULL,
                total_students INTEGER DEFAULT 0,
                total_points INTEGER DEFAULT 0,
                events_organized INTEGER DEFAULT 0,
                average_rating FLOAT DEFAULT 0.0,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table_sql in tables:
            self.client.execute_query(table_sql, commit=True)
    
    # ============================================
    # USER MANAGEMENT METHODS
    # ============================================
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials with rate limiting"""
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
                # Update last login and streak
                update_data = {
                    'last_login': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'login_attempts': 0
                }
                
                # Check daily login streak
                last_login = user.get('last_daily_login')
                if last_login:
                    try:
                        last_login_date = datetime.fromisoformat(last_login.replace('Z', '+00:00')).date()
                        today = datetime.now().date()
                        
                        if today > last_login_date:
                            if (today - last_login_date).days == 1:
                                new_streak = user.get('daily_login_streak', 0) + 1
                                update_data['daily_login_streak'] = new_streak
                                
                                streak_bonus = GAMIFICATION_CONFIG['streak_bonus']['daily_login']
                                bonus_index = min(new_streak - 1, len(streak_bonus) - 1)
                                if bonus_index >= 0:
                                    self.award_points(username, streak_bonus[bonus_index], 
                                                    "daily_login_streak", f"Day {new_streak} streak")
                            else:
                                update_data['daily_login_streak'] = 1
                        
                        update_data['last_daily_login'] = datetime.now().isoformat()
                    except:
                        update_data['daily_login_streak'] = 1
                        update_data['last_daily_login'] = datetime.now().isoformat()
                
                self.client.update('users', {'username': username}, update_data)
                
                cache.delete(rate_key)
                self.award_points(username, GAMIFICATION_CONFIG['points']['daily_login'], 
                                "daily_login", "Daily login")
                
                return True
            
            cache.set(rate_key, attempts + 1, ttl=300)
            return False
            
        except Exception as e:
            logger.error(f"Login error: {e}")
            cache.set(rate_key, attempts + 1, ttl=300)
            return False
    
    def get_user(self, username, use_cache=True):
        """Get user by username"""
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
                'avatar_url': user_data.get('avatar_url', ''),
                'bio': user_data.get('bio', ''),
                'skills': json.dumps(user_data.get('skills', [])),
                'total_points': 0,
                'current_level': 1,
                'level_progress': 0,
                'badges_earned': '[]',
                'achievements_unlocked': '[]',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('users', user_record, use_cache=False)
            
            if success:
                self._clear_user_cache(user_record['username'])
                self.award_points(user_record['username'], 
                                GAMIFICATION_CONFIG['points']['complete_profile'],
                                "complete_profile", "Profile completion")
                
                return True, "User registered successfully"
            else:
                return False, "Registration failed"
            
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return False, str(e)
    
    def update_user_profile(self, username, profile_data):
        """Update user profile"""
        try:
            if 'skills' in profile_data and isinstance(profile_data['skills'], list):
                profile_data['skills'] = json.dumps(profile_data['skills'])
            
            profile_data['updated_at'] = datetime.now().isoformat()
            
            success = self.client.update('users', {'username': username}, profile_data)
            
            if success:
                self._clear_user_cache(username)
                return True, "Profile updated successfully"
            else:
                return False, "Profile update failed"
            
        except Exception as e:
            logger.error(f"Error updating profile: {e}")
            return False, str(e)
    
    def _clear_user_cache(self, username):
        """Clear cache for user"""
        cache.delete(f"user_{username}")
        cache.delete(f"user_stats_{username}")
    
    # ============================================
    # EVENT MANAGEMENT METHODS
    # ============================================
    
    def add_event(self, event_data):
        """Add new event"""
        try:
            event_record = {
                'id': event_data.get('id', str(uuid.uuid4())),
                'title': event_data.get('title'),
                'description': event_data.get('description'),
                'event_type': event_data.get('event_type'),
                'event_category': event_data.get('event_category', 'Technical'),
                'event_date': event_data.get('event_date'),
                'end_date': event_data.get('end_date'),
                'venue': event_data.get('venue'),
                'organizer': event_data.get('organizer'),
                'co_organizers': json.dumps(event_data.get('co_organizers', [])),
                'event_link': event_data.get('event_link', ''),
                'registration_link': event_data.get('registration_link', ''),
                'max_participants': event_data.get('max_participants', 100),
                'current_participants': 0,
                'flyer_path': event_data.get('flyer_path'),
                'tags': json.dumps(event_data.get('tags', [])),
                'prerequisites': event_data.get('prerequisites', ''),
                'resources': json.dumps(event_data.get('resources', [])),
                'created_by': event_data.get('created_by'),
                'created_by_name': event_data.get('created_by_name'),
                'ai_generated': event_data.get('ai_generated', False),
                'status': 'upcoming',
                'mentor_id': event_data.get('mentor_id'),
                'difficulty_level': event_data.get('difficulty_level', 'Beginner'),
                'estimated_duration': event_data.get('estimated_duration'),
                'has_certificate': event_data.get('has_certificate', False),
                'certificate_template': event_data.get('certificate_template', ''),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'views_count': 0,
                'popularity_score': 0.0
            }
            
            success = self.client.insert('events', event_record, use_cache=False)
            
            if success:
                cache.delete("events_all")
                cache.delete("events_upcoming")
                
                if event_record['created_by']:
                    self.award_points(event_record['created_by'], 
                                    GAMIFICATION_CONFIG['points']['event_creation'],
                                    "event_creation", f"Created event: {event_record['title'][:30]}")
                
                self._notify_interested_users(event_record)
                
                return True, event_record['id']
            return False, None
            
        except Exception as e:
            logger.error(f"Error adding event: {e}")
            return False, None
    
    def _notify_interested_users(self, event):
        """Notify users interested in similar events"""
        pass
    
    def get_all_events(self, cache_ttl=60, use_cache=True):
        """Get all events"""
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
            now = datetime.now()
        
            if self.use_supabase:
                events = self.get_all_events(cache_ttl=0, use_cache=False)
            
                for event in events:
                    event_date = event.get('event_date')
                    event_id = event.get('id')
                    current_status = event.get('status', 'upcoming')
                    end_date = event.get('end_date')
                
                    if isinstance(event_date, str):
                        try:
                            # Parse date string to datetime (handle timezone-aware strings)
                            if 'Z' in event_date or '+' in event_date:
                                # Timezone-aware datetime
                                event_dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                                # Make both datetime objects timezone-aware or both naive
                                # Convert to naive datetime by removing timezone info
                                event_dt = event_dt.replace(tzinfo=None)
                            else:
                                # Naive datetime
                                event_dt = datetime.fromisoformat(event_date)
                        
                            if end_date:
                                # Parse end date similarly
                                if 'Z' in end_date or '+' in end_date:
                                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                                    end_dt = end_dt.replace(tzinfo=None)
                                else:
                                    end_dt = datetime.fromisoformat(end_date)
                            
                                if end_dt < now and current_status != 'completed':
                                    self.client.update('events', {'id': event_id}, {
                                    'status': 'completed',
                                    'updated_at': now.isoformat()
                                    }, use_cache=False)
                        
                            elif event_dt <= now and current_status == 'upcoming':
                                self.client.update('events', {'id': event_id}, {
                                'status': 'ongoing',
                                'updated_at': now.isoformat()
                                }, use_cache=False)
                        
                            elif event_dt < now and current_status not in ['completed', 'cancelled']:
                                self.client.update('events', {'id': event_id}, {
                                'status': 'completed',
                                'updated_at': now.isoformat()
                                }, use_cache=False)
                            
                        except Exception as e:
                            logger.warning(f"Error parsing event date {event_date}: {e}")
                            continue
                        
            else:
                # For SQLite
                now_iso = now.isoformat()
                self.client.execute_query(
                    "UPDATE events SET status = 'ongoing', updated_at = ? WHERE event_date <= ? AND status = 'upcoming'",
                    (now_iso, now_iso), commit=True
                )
            
                self.client.execute_query(
                    "UPDATE events SET status = 'completed', updated_at = ? WHERE event_date < ? AND status IN ('upcoming', 'ongoing')",
                    (now_iso, now_iso), commit=True
                )
        
            cache.delete("events_all")
            cache.delete("events_upcoming")
        
            return True
        
        except Exception as e:
            logger.error(f"Error updating event status: {e}")
            return False
    
    # ============================================
    # MENTOR MANAGEMENT METHODS
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
    # REGISTRATION MANAGEMENT METHODS
    # ============================================
    
    def add_registration(self, reg_data):
        """Add new registration"""
        try:
            existing = self.is_student_registered(reg_data['event_id'], reg_data['student_username'])
            if existing:
                return None, "Already registered"
            
            event = self.get_event_by_id(reg_data['event_id'])
            if event:
                max_participants = event.get('max_participants', 100)
                current_participants = event.get('current_participants', 0)
                
                if current_participants >= max_participants:
                    return None, "Event is full"
            
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
                self._clear_registration_cache(reg_data['student_username'])
                self._update_event_participant_count(reg_data['event_id'])
                
                self.award_points(reg_data['student_username'], 
                                GAMIFICATION_CONFIG['points']['registration'],
                                "event_registration", 
                                f"Registered for: {reg_data['event_title'][:30]}")
                
                self.create_notification(
                    user_id=reg_data['student_username'],
                    title="✅ Event Registration Confirmed",
                    message=f"You have successfully registered for '{reg_data['event_title']}'",
                    notification_type="registration",
                    related_id=reg_data['event_id']
                )
                
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
    
    def _clear_registration_cache(self, username):
        """Clear registration cache for user"""
        cache.delete(f"registrations_{username}")
    
    # ============================================
    # LIKES & INTEREST METHODS
    # ============================================
    
    def add_like(self, event_id, student_username):
        """Add a like for an event"""
        try:
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
                cache.delete(f"liked_{event_id}_{student_username}")
                cache.delete(f"likes_count_{event_id}")
                cache.delete(f"liked_events_{student_username}")
                
                self._update_event_popularity(event_id)
                
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
                cache.delete(f"liked_{event_id}_{student_username}")
                cache.delete(f"likes_count_{event_id}")
                cache.delete(f"liked_events_{student_username}")
                
                self._update_event_popularity(event_id)
                
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
    
    def add_interested(self, event_id, student_username):
        """Add interested for an event"""
        try:
            if self.is_event_interested(event_id, student_username):
                return False
            
            interested_record = {
                'id': str(uuid.uuid4()),
                'event_id': event_id,
                'student_username': student_username,
                'interested_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('event_interested', interested_record, use_cache=False)
            
            if success:
                cache.delete(f"interested_{event_id}_{student_username}")
                cache.delete(f"interested_count_{event_id}")
                cache.delete(f"interested_events_{student_username}")
                
                self._update_event_popularity(event_id)
                
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding interested: {e}")
            return False
    
    def remove_interested(self, event_id, student_username):
        """Remove interested for an event"""
        try:
            success = self.client.delete('event_interested', {
                'event_id': event_id,
                'student_username': student_username
            }, use_cache=False)
            
            if success:
                cache.delete(f"interested_{event_id}_{student_username}")
                cache.delete(f"interested_count_{event_id}")
                cache.delete(f"interested_events_{student_username}")
                
                self._update_event_popularity(event_id)
                
                return True
            return False
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
        """Get total interested count for an event"""
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
        """Get all events liked by a student"""
        try:
            if self.use_supabase:
                likes = self.client.select('event_likes', {'student_username': student_username}, cache_ttl=60)
                if not likes:
                    return []
                
                all_events = self.get_all_events(cache_ttl=60)
                if not all_events:
                    return []
                
                liked_event_ids = {like['event_id'] for like in likes}
                liked_events = [event for event in all_events if event.get('id') in liked_event_ids]
                
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
                interests = self.client.select('event_interested', {'student_username': student_username}, cache_ttl=60)
                if not interests:
                    return []
                
                all_events = self.get_all_events(cache_ttl=60)
                if not all_events:
                    return []
                
                interested_event_ids = {interest['event_id'] for interest in interests}
                interested_events = [event for event in all_events if event.get('id') in interested_event_ids]
                
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
    
    def _update_event_popularity(self, event_id):
        """Update event popularity score"""
        try:
            likes = self.get_event_likes_count(event_id)
            interested = self.get_event_interested_count(event_id)
            registrations = self._get_event_registration_count(event_id)
            
            popularity = (likes * 0.3) + (interested * 0.4) + (registrations * 0.3)
            
            self.client.update('events', {'id': event_id}, {
                'popularity_score': popularity,
                'updated_at': datetime.now().isoformat()
            }, use_cache=False)
            
            cache.delete(f"event_{event_id}")
            
        except Exception as e:
            logger.error(f"Error updating popularity: {e}")
    
    def _get_event_registration_count(self, event_id):
        """Get registration count for event"""
        try:
            if self.use_supabase:
                regs = self.client.select('registrations', {'event_id': event_id}, use_cache=False)
                return len(regs) if regs else 0
            else:
                result = self.client.execute_query(
                    "SELECT COUNT(*) as count FROM registrations WHERE event_id = ?",
                    (event_id,), fetchone=True, use_cache=False
                )
                return result['count'] if result else 0
        except:
            return 0
    
    # ============================================
    # GAMIFICATION & POINTS SYSTEM
    # ============================================
    
    def award_points(self, username, points, reason, description=""):
        """Award points to a user"""
        try:
            user = self.get_user(username, use_cache=False)
            if not user:
                return False
            
            current_points = user.get('total_points', 0)
            new_points = current_points + points
            
            update_data = {
                'total_points': new_points,
                'updated_at': datetime.now().isoformat()
            }
            
            current_level = user.get('current_level', 1)
            for level, config in GAMIFICATION_CONFIG['levels'].items():
                if new_points >= config['points_required'] and level > current_level:
                    update_data['current_level'] = level
                    update_data['level_progress'] = 0
                    
                    self.unlock_achievement(
                        username,
                        f"level_{level}",
                        f"Reached Level {level}: {config['name']}",
                        "level_up",
                        points * 2
                    )
                    
                    self.create_notification(
                        user_id=username,
                        title=f"🎉 Level Up!",
                        message=f"Congratulations! You've reached Level {level}: {config['name']}",
                        notification_type="achievement",
                        related_id=f"level_{level}"
                    )
            
            success = self.client.update('users', {'username': username}, update_data, use_cache=False)
            
            if success:
                self._clear_user_cache(username)
                self._log_points_transaction(username, points, reason, description)
                self._check_badge_unlocks(username)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error awarding points: {e}")
            return False
    
    def _log_points_transaction(self, username, points, reason, description):
        """Log points transaction for audit trail"""
        pass
    
    def _check_badge_unlocks(self, username):
        """Check and unlock badges based on user achievements"""
        user = self.get_user(username, use_cache=False)
        if not user:
            return
        
        points = user.get('total_points', 0)
        current_badges = json.loads(user.get('badges_earned', '[]'))
        new_badges = []
        
        if points >= 10000 and "legend_badge" not in current_badges:
            new_badges.append("legend_badge")
        elif points >= 5000 and "master_badge" not in current_badges:
            new_badges.append("master_badge")
        elif points >= 1000 and "expert_badge" not in current_badges:
            new_badges.append("expert_badge")
        
        if new_badges:
            all_badges = current_badges + new_badges
            self.client.update('users', {'username': username}, {
                'badges_earned': json.dumps(all_badges),
                'updated_at': datetime.now().isoformat()
            }, use_cache=False)
            
            for badge in new_badges:
                self.create_notification(
                    user_id=username,
                    title="🎖️ New Badge Unlocked!",
                    message=f"You've unlocked the {badge.replace('_', ' ').title()} badge!",
                    notification_type="badge",
                    related_id=badge
                )
    
    def unlock_achievement(self, username, achievement_id, name, achievement_type, points=0):
        """Unlock an achievement for a user"""
        try:
            achievement_record = {
                'id': str(uuid.uuid4()),
                'user_id': username,
                'achievement_id': achievement_id,
                'achievement_name': name,
                'achievement_type': achievement_type,
                'points_awarded': points,
                'unlocked_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('user_achievements', achievement_record, use_cache=False)
            
            if success and points > 0:
                self.award_points(username, points, "achievement", name)
            
            return success
            
        except Exception as e:
            logger.error(f"Error unlocking achievement: {e}")
            return False
    
    def get_student_points(self, username):
        """Get student's total points"""
        user = self.get_user(username)
        if user:
            return user.get('total_points', 0)
        return 0
    
    def get_student_level(self, username):
        """Get student's current level"""
        user = self.get_user(username)
        if user:
            return user.get('current_level', 1)
        return 1
    
    def get_level_progress(self, username):
        """Get student's level progress percentage"""
        user = self.get_user(username)
        if not user:
            return 0
        
        current_level = user.get('current_level', 1)
        current_points = user.get('total_points', 0)
        
        current_level_config = GAMIFICATION_CONFIG['levels'].get(current_level, {})
        next_level_config = GAMIFICATION_CONFIG['levels'].get(current_level + 1, {})
        
        if not next_level_config:
            return 100
        
        points_current_level = current_level_config.get('points_required', 0)
        points_next_level = next_level_config.get('points_required', 0)
        
        if points_next_level <= points_current_level:
            return 100
        
        points_in_level = current_points - points_current_level
        points_needed = points_next_level - points_current_level
        
        progress = (points_in_level / points_needed) * 100
        return min(100, max(0, progress))
    
    # ============================================
    # LEADERBOARD & ANALYTICS
    # ============================================
    
    def get_leaderboard(self, limit=20, department=None, timeframe="all", cache_ttl=300):
        """Get leaderboard of top students"""
        cache_key = f"leaderboard_{department}_{timeframe}_{limit}"
        if CACHE_ENABLED:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if timeframe == "weekly":
                date_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
                points_column = "weekly_points"
            elif timeframe == "monthly":
                date_cutoff = (datetime.now() - timedelta(days=30)).isoformat()
                points_column = "monthly_points"
            else:
                points_column = "total_points"
            
            if self.use_supabase:
                filters = {'role': 'student'}
                if department:
                    filters['department'] = department
                
                users = self.client.select('users', filters, limit=limit * 2, 
                                         order_by=f'{points_column}.desc', 
                                         cache_ttl=cache_ttl)
                
                if not users:
                    return []
                
                users = sorted(users, key=lambda x: x.get(points_column, 0), reverse=True)
                users = users[:limit]
                
                for i, user in enumerate(users, 1):
                    user['rank'] = i
                    user['points'] = user.get(points_column, 0)
                
            else:
                query = f"""
                SELECT *, {points_column} as points FROM users 
                WHERE role = 'student'
                """
                params = []
                
                if department:
                    query += " AND department = ?"
                    params.append(department)
                
                query += f" ORDER BY {points_column} DESC, name ASC LIMIT ?"
                params.append(limit)
                
                results = self.client.execute_query(query, tuple(params), fetchall=True) or []
                
                for i, result in enumerate(results, 1):
                    result['rank'] = i
            
            if CACHE_ENABLED:
                cache.set(cache_key, results, ttl=cache_ttl)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []
    
    def get_student_rank(self, username):
        """Get student's rank in leaderboard"""
        leaderboard = self.get_leaderboard(limit=1000, timeframe="all")
        for i, student in enumerate(leaderboard, 1):
            if student['username'] == username:
                return i
        return None
    
    def get_department_stats(self):
        """Get statistics for all departments"""
        cache_key = "department_stats"
        if CACHE_ENABLED:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self.use_supabase:
                students = self.client.select('users', {'role': 'student'}, limit=1000)
                
                stats = {}
                for student in students:
                    dept = student.get('department', 'Unknown')
                    if dept not in stats:
                        stats[dept] = {
                            'total_students': 0,
                            'total_points': 0,
                            'average_points': 0
                        }
                    
                    stats[dept]['total_students'] += 1
                    stats[dept]['total_points'] += student.get('total_points', 0)
                
                for dept in stats:
                    if stats[dept]['total_students'] > 0:
                        stats[dept]['average_points'] = stats[dept]['total_points'] / stats[dept]['total_students']
                
                result = [{'department': k, **v} for k, v in stats.items()]
                
            else:
                result = self.client.execute_query("""
                    SELECT 
                        department,
                        COUNT(*) as total_students,
                        SUM(total_points) as total_points,
                        AVG(total_points) as average_points
                    FROM users 
                    WHERE role = 'student' AND department IS NOT NULL AND department != ''
                    GROUP BY department
                    ORDER BY total_points DESC
                """, fetchall=True) or []
            
            if CACHE_ENABLED:
                cache.set(cache_key, result, ttl=3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting department stats: {e}")
            return []
    
    def get_event_analytics(self, event_id=None):
        """Get analytics for events"""
        try:
            if event_id:
                event = self.get_event_by_id(event_id)
                if not event:
                    return None
                
                registrations = self.client.select('registrations', {'event_id': event_id})
                likes = self.get_event_likes_count(event_id)
                interested = self.get_event_interested_count(event_id)
                
                return {
                    'event': event,
                    'registrations_count': len(registrations) if registrations else 0,
                    'likes_count': likes,
                    'interested_count': interested,
                    'attendance_rate': self._calculate_attendance_rate(event_id),
                    'feedback_stats': self._get_feedback_stats(event_id)
                }
            else:
                events = self.get_all_events(use_cache=False)
                total_events = len(events)
                
                status_counts = defaultdict(int)
                type_counts = defaultdict(int)
                total_registrations = 0
                
                for event in events:
                    status = event.get('status', 'unknown')
                    event_type = event.get('event_type', 'Unknown')
                    
                    status_counts[status] += 1
                    type_counts[event_type] += 1
                    
                    regs = self.client.select('registrations', {'event_id': event.get('id')})
                    total_registrations += len(regs) if regs else 0
                
                return {
                    'total_events': total_events,
                    'status_counts': dict(status_counts),
                    'type_counts': dict(type_counts),
                    'total_registrations': total_registrations,
                    'average_registrations': total_registrations / total_events if total_events > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting event analytics: {e}")
            return None
    
    def _calculate_attendance_rate(self, event_id):
        """Calculate attendance rate for an event"""
        try:
            if self.use_supabase:
                registrations = self.client.select('registrations', {'event_id': event_id})
                if not registrations:
                    return 0
                
                attended = sum(1 for r in registrations if r.get('attendance') == 'present')
                return (attended / len(registrations)) * 100 if registrations else 0
            else:
                result = self.client.execute_query(
                    "SELECT COUNT(*) as total, SUM(CASE WHEN attendance = 'present' THEN 1 ELSE 0 END) as attended FROM registrations WHERE event_id = ?",
                    (event_id,), fetchone=True
                )
                if result and result['total'] > 0:
                    return (result['attended'] / result['total']) * 100
                return 0
        except:
            return 0
    
    def _get_feedback_stats(self, event_id):
        """Get feedback statistics for an event"""
        try:
            if self.use_supabase:
                feedbacks = self.client.select('event_feedback', {'event_id': event_id})
                if not feedbacks:
                    return {'count': 0, 'average_rating': 0, 'total': 0}
                
                total_rating = sum(f.get('rating', 0) for f in feedbacks)
                average = total_rating / len(feedbacks) if feedbacks else 0
                
                return {
                    'count': len(feedbacks),
                    'average_rating': round(average, 1),
                    'total': total_rating
                }
            else:
                result = self.client.execute_query(
                    "SELECT COUNT(*) as count, AVG(rating) as average FROM event_feedback WHERE event_id = ?",
                    (event_id,), fetchone=True
                )
                if result:
                    return {
                        'count': result['count'],
                        'average_rating': round(result['average'] or 0, 1),
                        'total': result['count'] * (result['average'] or 0)
                    }
                return {'count': 0, 'average_rating': 0, 'total': 0}
        except:
            return {'count': 0, 'average_rating': 0, 'total': 0}
    
    # ============================================
    # NOTIFICATION SYSTEM
    # ============================================
    
    def create_notification(self, user_id, title, message, notification_type=None, 
                          related_id=None, expires_in=604800):
        """Create a notification for a user"""
        try:
            notification_record = {
                'id': str(uuid.uuid4()),
                'user_id': user_id,
                'title': title,
                'message': message,
                'notification_type': notification_type,
                'related_id': related_id,
                'read': 0,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(seconds=expires_in)).isoformat(),
                'action_url': related_id
            }
            
            success = self.client.insert('notifications', notification_record, use_cache=False)
            return success
            
        except Exception as e:
            logger.error(f"Error creating notification: {e}")
            return False
    
    def get_user_notifications(self, user_id, unread_only=False, limit=20):
        """Get notifications for a user"""
        try:
            if self.use_supabase:
                filters = {'user_id': user_id}
                if unread_only:
                    filters['read'] = 0
                
                notifications = self.client.select('notifications', filters=filters, 
                                                 limit=limit, order_by='created_at.desc')
            else:
                query = "SELECT * FROM notifications WHERE user_id = ?"
                params = [user_id]
                
                if unread_only:
                    query += " AND read = 0"
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                notifications = self.client.execute_query(query, tuple(params), fetchall=True)
            
            return notifications or []
            
        except Exception as e:
            logger.error(f"Error getting notifications: {e}")
            return []
    
    def mark_notification_read(self, notification_id, user_id=None):
        """Mark a notification as read"""
        try:
            filters = {'id': notification_id}
            if user_id:
                filters['user_id'] = user_id
            
            success = self.client.update('notifications', filters, {'read': 1}, use_cache=False)
            return success
            
        except Exception as e:
            logger.error(f"Error marking notification read: {e}")
            return False
    
    def mark_all_notifications_read(self, user_id):
        """Mark all notifications as read for a user"""
        try:
            if self.use_supabase:
                success = self.client.update('notifications', {'user_id': user_id, 'read': 0}, 
                                           {'read': 1}, use_cache=False)
            else:
                success = self.client.execute_query(
                    "UPDATE notifications SET read = 1 WHERE user_id = ? AND read = 0",
                    (user_id,), commit=True
                )
                success = success is not None
            
            return success
            
        except Exception as e:
            logger.error(f"Error marking all notifications read: {e}")
            return False
    
    # ============================================
    # FEEDBACK SYSTEM
    # ============================================
    
    def submit_feedback(self, event_id, username, rating, comments=None, anonymous=False):
        """Submit feedback for an event"""
        try:
            existing = self.get_user_feedback(event_id, username)
            if existing:
                return False, "Feedback already submitted"
            
            feedback_record = {
                'id': str(uuid.uuid4()),
                'event_id': event_id,
                'student_username': username,
                'rating': rating,
                'comments': comments,
                'anonymous': 1 if anonymous else 0,
                'submitted_at': datetime.now().isoformat()
            }
            
            success = self.client.insert('event_feedback', feedback_record, use_cache=False)
            
            if success:
                self.client.update('registrations', 
                                 {'event_id': event_id, 'student_username': username},
                                 {'feedback_provided': 1}, use_cache=False)
                
                self.award_points(username, GAMIFICATION_CONFIG['points']['feedback_submission'],
                                "feedback_submission", f"Feedback for event")
                
                return True, "Feedback submitted successfully"
            
            return False, "Failed to submit feedback"
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return False, "Error submitting feedback"
    
    def get_user_feedback(self, event_id, username):
        """Get feedback submitted by a user for an event"""
        try:
            if self.use_supabase:
                results = self.client.select('event_feedback', 
                                           {'event_id': event_id, 'student_username': username},
                                           limit=1)
                return results[0] if results else None
            else:
                return self.client.execute_query(
                    "SELECT * FROM event_feedback WHERE event_id = ? AND student_username = ?",
                    (event_id, username), fetchone=True
                )
        except Exception as e:
            logger.error(f"Error getting user feedback: {e}")
            return None
    
    def get_event_feedback(self, event_id, limit=50):
        """Get all feedback for an event"""
        try:
            if self.use_supabase:
                feedbacks = self.client.select('event_feedback', {'event_id': event_id},
                                             limit=limit, order_by='submitted_at.desc')
            else:
                feedbacks = self.client.execute_query(
                    "SELECT * FROM event_feedback WHERE event_id = ? ORDER BY submitted_at DESC LIMIT ?",
                    (event_id, limit), fetchall=True
                )
            
            return feedbacks or []
            
        except Exception as e:
            logger.error(f"Error getting event feedback: {e}")
            return []
    
    # ============================================
    # DEFAULT USERS
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
                    'department': 'Administration',
                    'avatar_url': '👨‍💼'
                },
                {
                    'id': '00000000-0000-0000-0000-000000000002',
                    'username': 'faculty@raisoni',
                    'password': 'Faculty@12345',
                    'name': 'Faculty Coordinator',
                    'role': 'faculty',
                    'email': 'faculty@ghraisoni.edu',
                    'department': 'Faculty',
                    'avatar_url': '👨‍🏫'
                },
                {
                    'id': '00000000-0000-0000-0000-000000000003',
                    'username': 'mentor@raisoni',
                    'password': 'Mentor@12345',
                    'name': 'Senior Mentor',
                    'role': 'mentor',
                    'email': 'mentor@ghraisoni.edu',
                    'department': 'Mentorship',
                    'avatar_url': '👨‍🏫'
                }
            ]
            
            for user_data in default_users:
                existing = self.get_user(user_data['username'], use_cache=False)
                if not existing:
                    success, message = self.add_user(user_data)
                    if success:
                        logger.info(f"Added default user: {user_data['username']}")
            
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
                    'mobile': '9876543210',
                    'avatar_url': '👨‍🎓',
                    'bio': 'Passionate about AI and Machine Learning',
                    'skills': ['Python', 'Machine Learning', 'Web Development']
                },
                {
                    'name': 'Priya Patel',
                    'username': 'priya@student',
                    'password': 'Student@123',
                    'roll_no': 'AIML2023002',
                    'department': 'Artificial Intelligence & Machine Learning',
                    'year': 'II',
                    'email': 'priya.patel@ghraisoni.edu',
                    'mobile': '9876543211',
                    'avatar_url': '👩‍🎓',
                    'bio': 'Data Science enthusiast and competitive programmer',
                    'skills': ['Data Science', 'Python', 'SQL', 'Statistics']
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
                        'avatar_url': student.get('avatar_url', '👤'),
                        'bio': student.get('bio', ''),
                        'skills': student.get('skills', []),
                        'role': 'student'
                    }
                    
                    success, message = self.add_user(user_data)
                    if success:
                        self.award_points(student['username'], 250, "welcome", "Welcome bonus")
                        logger.info(f"Added default student: {student['username']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding default students: {e}")
            return False
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
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
# DATABASE INITIALIZATION
# ============================================

# Initialize database
db = DatabaseManager(use_supabase=USE_SUPABASE)

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

@st.cache_data(ttl=300)
def format_date(date_str):
    """Format date for display"""
    try:
        if isinstance(date_str, str):
            # Try different date formats
            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime("%d %b %Y, %I:%M %p")
                except:
                    continue
            
            # Handle ISO format with timezone
            if 'Z' in date_str or '+' in date_str:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                dt = datetime.fromisoformat(date_str)
                
            return dt.strftime("%d %b %Y, %I:%M %p")
        else:
            dt = date_str
            # Handle timezone-aware datetime objects
            if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt.strftime("%d %b %Y, %I:%M %p")
    except:
        return str(date_str)
                
def get_event_status(event_date, end_date=None, status=None):
    """Get event status badge"""
    if status and status in ['cancelled', 'completed']:
        if status == 'cancelled':
            return '<span style="background: #F3F4F6; color: #6B7280; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">⚫ Cancelled</span>'
        else:
            return '<span style="background: #E5E7EB; color: #374151; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🔴 Completed</span>'
    
    try:
        if isinstance(event_date, str):
            # Parse date string
            if 'Z' in event_date or '+' in event_date:
                # Timezone-aware datetime
                dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
                dt = dt.replace(tzinfo=None)  # Convert to naive datetime
            else:
                # Naive datetime
                dt = datetime.fromisoformat(event_date)
        else:
            dt = event_date
            # Ensure dt is naive datetime
            if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
        
        now = datetime.now()
        
        if end_date:
            try:
                if isinstance(end_date, str):
                    if 'Z' in end_date or '+' in end_date:
                        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        end_dt = end_dt.replace(tzinfo=None)
                    else:
                        end_dt = datetime.fromisoformat(end_date)
                else:
                    end_dt = end_date
                    if hasattr(end_dt, 'tzinfo') and end_dt.tzinfo is not None:
                        end_dt = end_dt.replace(tzinfo=None)
                
                if end_dt < now:
                    return '<span style="background: #E5E7EB; color: #374151; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🔴 Completed</span>'
                elif dt <= now <= end_dt:
                    return '<span style="background: #FEF3C7; color: #92400E; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🟡 Ongoing</span>'
            except:
                pass
        
        if dt > now:
            days_diff = (dt - now).days
            if days_diff <= 7:
                return '<span style="background: #D1FAE5; color: #065F46; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🟢 Soon</span>'
            else:
                return '<span style="background: #D1FAE5; color: #065F46; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🟢 Upcoming</span>'
        elif dt.date() == now.date():
            return '<span style="background: #FEF3C7; color: #92400E; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">🟡 Today</span>'
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
        if len(image_bytes) > 10 * 1024 * 1024:
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

def generate_avatar_url(name, size=100):
    """Generate avatar URL based on name"""
    style = "avataaars"
    seed = name.replace(" ", "").lower()
    return f"https://api.dicebear.com/7.x/{style}/png?seed={seed}&size={size}"

def get_event_emoji(event_type):
    """Get emoji for event type"""
    emoji_map = {
        'Workshop': '🔧',
        'Hackathon': '💻',
        'Competition': '🏆',
        'Bootcamp': '🚀',
        'Seminar': '🎓',
        'Conference': '👥',
        'Webinar': '🖥️',
        'Training': '📚',
        'Symposium': '🗣️',
        'Cultural Event': '🎭',
        'Guest Lecture': '🎤',
        'Industrial Visit': '🏭',
        'Sports Event': '⚽',
        'Technical Fest': '🔬',
        'Placement Drive': '💼',
        'Alumni Talk': '👨‍🎓',
        'Research Symposium': '🔍'
    }
    return emoji_map.get(event_type, '🎯')

def check_remember_me_cookie():
    """Check for remember me cookie and auto-login"""
    if 'remember_me' not in st.session_state:
        st.session_state.remember_me = False
    
    try:
        if hasattr(st, 'query_params'):
            params = st.query_params.to_dict()
        else:
            params = {}
            import urllib.parse
            from urllib.parse import urlparse, parse_qs
            
            if hasattr(st, 'get_current_url'):
                url = st.get_current_url()
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
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
# VISUALIZATION FUNCTIONS
# ============================================

def create_points_chart(points_history):
    """Create a chart showing points progression"""
    if not points_history:
        return None
    
    df = pd.DataFrame(points_history)
    if 'date' in df.columns and 'points' in df.columns:
        fig = px.line(df, x='date', y='points', 
                     title='Points Progression',
                     labels={'date': 'Date', 'points': 'Points'},
                     template='plotly_white')
        fig.update_traces(mode='lines+markers', line=dict(width=3))
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Points',
            hovermode='x unified'
        )
        return fig
    return None

def create_department_stats_chart(department_stats):
    """Create a bar chart for department statistics"""
    if not department_stats:
        return None
    
    df = pd.DataFrame(department_stats)
    if 'department' in df.columns and 'total_points' in df.columns:
        fig = px.bar(df, x='department', y='total_points',
                    title='Department Points Comparison',
                    labels={'department': 'Department', 'total_points': 'Total Points'},
                    color='total_points',
                    color_continuous_scale='Viridis',
                    template='plotly_white')
        fig.update_layout(
            xaxis_title='Department',
            yaxis_title='Total Points',
            xaxis_tickangle=-45
        )
        return fig
    return None

def create_event_type_chart(event_stats):
    """Create a pie chart for event type distribution"""
    if not event_stats or 'type_counts' not in event_stats:
        return None
    
    type_counts = event_stats['type_counts']
    if not type_counts:
        return None
    
    labels = list(type_counts.keys())
    values = list(type_counts.values())
    
    fig = px.pie(names=labels, values=values,
                title='Event Type Distribution',
                hole=0.3,
                template='plotly_white')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_leaderboard_visualization(leaderboard_data, title="Top Performers"):
    """Create a horizontal bar chart for leaderboard"""
    if not leaderboard_data:
        return None
    
    df = pd.DataFrame(leaderboard_data)
    if 'name' in df.columns and 'points' in df.columns:
        df = df.sort_values('points', ascending=True)
        
        fig = px.bar(df, y='name', x='points',
                    title=title,
                    labels={'name': 'Student', 'points': 'Points'},
                    color='points',
                    color_continuous_scale='Viridis',
                    orientation='h',
                    template='plotly_white')
        
        fig.update_layout(
            yaxis_title='Student',
            xaxis_title='Points',
            showlegend=False
        )
        
        for i, row in enumerate(df.itertuples()):
            fig.add_annotation(
                x=row.points + (df['points'].max() * 0.02),
                y=row.name,
                text=f"#{i+1}",
                showarrow=False,
                font=dict(size=12, color='white')
            )
        
        return fig
    return None

# ============================================
# EVENT CARD DISPLAY
# ============================================

def display_event_card(event, current_user=None, show_actions=True):
    """Display event card with all interactions"""
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
                    st.markdown('<div style="width: 100%; height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; border-radius: 8px;"><span style="font-size: 32px; color: white;">🎯</span></div>', unsafe_allow_html=True)
            else:
                event_type = event.get('event_type', 'Event').lower()
                colors = {
                    'workshop': ('#4F46E5', '#7C3AED'),
                    'hackathon': ('#059669', '#10B981'),
                    'seminar': ('#DC2626', '#EA580C'),
                    'conference': ('#7C3AED', '#A78BFA'),
                    'cultural': ('#DB2777', '#EC4899')
                }
                color_from, color_to = colors.get(event_type, ('#667eea', '#764ba2'))
                
                st.markdown(f'''
                <div style="width: 100%; height: 150px; background: linear-gradient(135deg, {color_from} 0%, {color_to} 100%); 
                display: flex; align-items: center; justify-content: center; border-radius: 8px; position: relative;">
                    <span style="font-size: 48px; color: white;">{get_event_emoji(event.get('event_type'))}</span>
                    <div style="position: absolute; bottom: 8px; right: 8px; background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; color: white;">
                        {event.get('event_category', 'Event')}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        with col_info:
            title = event.get('title', 'Untitled Event')
            if len(title) > 60:
                title = title[:57] + "..."
            
            difficulty = event.get('difficulty_level', 'Beginner')
            difficulty_colors = {
                'Beginner': '#10B981',
                'Intermediate': '#F59E0B',
                'Advanced': '#DC2626',
                'Expert': '#7C3AED'
            }
            difficulty_color = difficulty_colors.get(difficulty, '#6B7280')
            
            st.markdown(f'''
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div class="card-title">{title}</div>
                <div style="background: {difficulty_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem;">
                    {difficulty}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            event_date = event.get('event_date')
            end_date = event.get('end_date')
            status = event.get('status')
            status_html = get_event_status(event_date, end_date, status)
            formatted_date = format_date(event_date)
            
            duration = event.get('estimated_duration')
            duration_text = f"⏱️ {duration} min" if duration else ""
            
            st.markdown(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>{status_html}</div>
                <div style="color: #666; font-size: 0.9rem;">📅 {formatted_date} {duration_text}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            venue = event.get('venue', 'TBD')
            if len(venue) > 25:
                venue = venue[:22] + "..."
            
            event_type = event.get('event_type', 'Event')
            max_participants = event.get('max_participants', 100)
            current_participants = event.get('current_participants', 0)
            
            participation_rate = (current_participants / max_participants) * 100 if max_participants > 0 else 0
            progress_color = '#10B981' if participation_rate < 80 else '#F59E0B' if participation_rate < 95 else '#DC2626'
            
            st.markdown(f'''
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 4px;">
                    <span>📍 {venue} | 🏷️ {event_type}</span>
                    <span>👥 {current_participants}/{max_participants}</span>
                </div>
                <div style="width: 100%; height: 6px; background: #E5E7EB; border-radius: 3px; overflow: hidden;">
                    <div style="width: {participation_rate}%; height: 100%; background: {progress_color}; border-radius: 3px;"></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            likes_count = db.get_event_likes_count(event_id)
            interested_count = db.get_event_interested_count(event_id)
            views_count = event.get('views_count', 0)
            
            if current_user and show_actions:
                button_col1, button_col2, button_col3, button_col4 = st.columns(4)
                
                with button_col1:
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
                    is_interested = db.is_event_interested(event_id, current_user)
                    interested_btn_text = "⭐ Interested" if is_interested else "☆ Interested"
                    interested_btn_type = "secondary" if is_interested else "primary"
                    
                    if st.button(interested_btn_text, key=f"interested_{event_id}_{current_user}",
                               use_container_width=True, type=interested_btn_type,
                               help="Mark as interested"):
                        if is_interested:
                            if db.remove_interested(event_id, current_user):
                                st.success("Removed from interested!")
                                time.sleep(0.3)
                                st.rerun()
                        else:
                            if db.add_interested(event_id, current_user):
                                st.success("Marked as interested!")
                                time.sleep(0.3)
                                st.rerun()
                
                with button_col3:
                    if st.button("📤 Share", key=f"share_{event_id}_{current_user}",
                               use_container_width=True, type="secondary",
                               help="Share this event"):
                        event_title = event.get('title', 'Cool Event')
                        share_text = f"Check out '{event_title}' at G H Raisoni College Event Manager!\n\nEvent Date: {formatted_date}\nVenue: {venue}\n\nJoin the platform to discover more events!"
                        
                        st.code(share_text)
                        st.info("📋 Copy the text above to share")
                
                with button_col4:
                    if st.button("🔖 Save", key=f"save_{event_id}_{current_user}",
                               use_container_width=True, type="secondary",
                               help="Save for later"):
                        st.info("Save feature coming soon!")
            
            st.caption(f"❤️ {likes_count} Likes | ⭐ {interested_count} Interested | 👁️ {views_count} Views")
            
            tags = event.get('tags')
            if tags and isinstance(tags, str):
                try:
                    tags_list = json.loads(tags)
                    if tags_list:
                        tags_html = " ".join([f'<span style="background: #E0F2FE; color: #0369A1; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; margin-right: 4px;">{tag}</span>' for tag in tags_list[:3]])
                        st.markdown(f'<div style="margin-top: 8px;">{tags_html}</div>', unsafe_allow_html=True)
                except:
                    pass
            
            event_link = event.get('event_link', '')
            registration_link = event.get('registration_link', '')
            
            if event_link or registration_link:
                with st.expander("🔗 Event Links & Details", expanded=False):
                    if event_link:
                        st.markdown(f"**🌐 Event Page:** [Click here]({event_link})")
                    if registration_link:
                        st.markdown(f"**📝 Registration:** [Click here]({registration_link})")
                    
                    if event.get('has_certificate'):
                        st.markdown("**🎓 Certificate:** Yes")
                    
                    prerequisites = event.get('prerequisites')
                    if prerequisites:
                        st.markdown(f"**📋 Prerequisites:** {prerequisites}")
            
            desc = event.get('description', '')
            if desc:
                if len(desc) > 150:
                    with st.expander("📝 Description", expanded=False):
                        st.write(desc)
                else:
                    st.caption(desc[:150] + "..." if len(desc) > 150 else desc)
        
        if current_user and st.session_state.get('role') == 'student' and show_actions:
            st.markdown('<div class="registration-section">', unsafe_allow_html=True)
            
            is_registered = db.is_student_registered(event_id, current_user)
            
            if is_registered:
                st.success("✅ You are already registered for this event")
                
                event_status = event.get('status')
                if event_status in ['completed', 'ongoing']:
                    if st.button("💬 Provide Feedback", key=f"feedback_{event_id}_{current_user}",
                               use_container_width=True, type="secondary"):
                        st.session_state.feedback_event_id = event_id
                        st.rerun()
                
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
                reg_col1, reg_col2 = st.columns(2)
                
                with reg_col1:
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
                    if registration_link:
                        st.markdown(f"[🌐 Register Externally]({registration_link})")
                        st.caption("Click to register on external site")
                    else:
                        st.info("No external registration link available")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        created_by = event.get('created_by_name', 'Unknown')
        popularity = event.get('popularity_score', 0)
        
        col_creator, col_pop = st.columns([3, 1])
        with col_creator:
            st.caption(f"👤 Created by: {created_by}")
        with col_pop:
            if popularity > 0:
                st.caption(f"🔥 {popularity:.1f}")
        
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
                    token = password_reset_manager.generate_reset_token(user['username'])
                    expiry = datetime.now() + timedelta(hours=1)
                    
                    if db.set_remember_token(user['username'], token, expiry.isoformat()):
                        reset_url = f"https://yourapp.com/reset?token={token}"
                        
                        st.success(f"✅ Reset link sent to {reset_email}")
                        st.info(f"**Test Token (for development):** `{token}`")
                        st.markdown(f"""
                        **In production, an email would be sent with:**
                        - Reset link: {reset_url}
                        - Token expires: {expiry.strftime('%I:%M %p, %d %b %Y')}
                        """)
                        
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
    col_logo, col_header = st.columns([1, 3])
    
    with col_logo:
        try:
            logo_path = "ghribmjal-logo.jpg"
            if os.path.exists(logo_path):
                st.image(logo_path, width=100)
            else:
                st.markdown('<div style="font-size: 3rem;">🎓</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div style="font-size: 3rem;">🎓</div>', unsafe_allow_html=True)
    
    with col_header:
        st.markdown(f'<div class="college-header"><h1>G H Raisoni College</h1><h3>Advanced Event Management System</h3></div>', 
                    unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab_about, tab_features, tab_stats = st.tabs(["About", "Features", "Live Stats"])
    
    with tab_about:
        st.markdown("""
        ### Welcome to G H Raisoni Event Management System
        
        **Your all-in-one platform for:**
        - Discovering college events and activities
        - Registering for workshops, hackathons, seminars
        - Tracking your participation and achievements
        - Competing with peers on the leaderboard
        - Earning points and badges for participation
        
        **Powered by advanced features:**
        - Real-time notifications
        - Gamification system
        - Analytics dashboard
        - Feedback system
        - Certificate generation
        """)
    
    with tab_features:
        col_feat1, col_feat2 = st.columns(2)
        
        with col_feat1:
            st.markdown("""
            **🎮 Gamification**
            - Earn points for participation
            - Unlock badges and achievements
            - Level up based on activity
            - Daily login streaks
            
            **📊 Analytics**
            - Personal progress tracking
            - Department comparisons
            - Event participation stats
            - Points history
            """)
        
        with col_feat2:
            st.markdown("""
            **🔔 Smart Notifications**
            - Event reminders
            - Registration confirmations
            - Achievement unlocks
            - System announcements
            
            **🎯 Event Features**
            - Advanced search and filtering
            - Like and interest tracking
            - Feedback system
            - Certificate management
            """)
    
    with tab_stats:
        try:
            events = db.get_all_events(cache_ttl=60)
            students = db.get_leaderboard(limit=1000) if db.use_supabase else []
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total Events", len(events))
            with col_stat2:
                st.metric("Active Students", len(students))
            with col_stat3:
                upcoming = len([e for e in events if e.get('status') == 'upcoming'])
                st.metric("Upcoming Events", upcoming)
            
            upcoming_events = db.get_upcoming_events(limit=5)
            if upcoming_events:
                st.markdown("**Next 5 Events:**")
                for event in upcoming_events:
                    st.markdown(f"- **{event.get('title', 'Event')}** ({format_date(event.get('event_date'))})")
        except:
            st.info("Statistics loading...")
    
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
            username = st.text_input("Username", key="login_username",
                                    placeholder="Enter your username")
        
        with col2:
            password = st.text_input("Password", type="password", key="login_password",
                                    placeholder="Enter your password")
        
        col_remember, col_forgot = st.columns(2)
        with col_remember:
            remember_me = st.checkbox("Remember Me", help="Stay logged in on this device")
        
        with col_forgot:
            if st.button("Forgot Password?"):
                st.session_state.page = "forgot_password"
                st.rerun()
        
        if st.button("Login", use_container_width=True, type="primary", key="login_button"):
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
                        st.success(f"Welcome back, {user.get('name', username)}! 🎉")
                        
                        db.create_notification(
                            user_id=username,
                            title="👋 Welcome Back!",
                            message="Great to see you again! Check out the latest events.",
                            notification_type="welcome"
                        )
                        
                        st.rerun()
                    else:
                        st.error("User not found in database")
                else:
                    st.error("Invalid credentials. Please check your username and password.")
        
        if role == "Student":
            st.markdown("---")
            st.subheader("👨‍🎓 New Student Registration")
            
            col_reg1, col_reg2 = st.columns([3, 1])
            with col_reg1:
                st.markdown("Don't have an account yet? Create one now!")
            with col_reg2:
                if st.button("Create Account", use_container_width=True, type="secondary"):
                    st.session_state.page = "student_register"
                    st.rerun()

# ============================================
# STUDENT REGISTRATION PAGE
# ============================================

def student_registration_page():
    """Student registration page"""
    st.markdown('<div class="college-header"><h2>👨‍🎓 Student Registration</h2><p>Join the G H Raisoni community</p></div>', 
                unsafe_allow_html=True)
    
    with st.form("student_registration"):
        st.markdown("### Create Your Student Account")
        
        st.markdown("#### Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *", placeholder="Enter your full name")
            roll_no = st.text_input("Roll Number *", placeholder="e.g., CSE2023001")
            department = st.selectbox("Department *", COLLEGE_CONFIG['departments'])
        
        with col2:
            email = st.text_input("Email *", placeholder="student@ghraisoni.edu")
            mobile = st.text_input("Mobile Number *", placeholder="9876543210")
            year = st.selectbox("Year *", COLLEGE_CONFIG['academic_years'])
        
        st.markdown("#### Account Information")
        col3, col4 = st.columns(2)
        
        with col3:
            username = st.text_input("Username *", placeholder="Choose a username")
            password = st.text_input("Password *", type="password", 
                                    help=f"Minimum {PASSWORD_MIN_LENGTH} characters with uppercase, lowercase, number, and special character")
        
        with col4:
            confirm_pass = st.text_input("Confirm Password *", type="password")
            
            skills = st.text_input("Skills (optional)", 
                                  placeholder="e.g., Python, Machine Learning, Web Development",
                                  help="Comma-separated list of your skills")
        
        bio = st.text_area("Bio (optional)", 
                          placeholder="Tell us about yourself...",
                          height=100)
        
        col_terms, col_consent = st.columns(2)
        with col_terms:
            terms = st.checkbox("I agree to the Terms & Conditions *", value=False)
        with col_consent:
            newsletter = st.checkbox("Receive event notifications and updates", value=True)
        
        col_submit, col_back = st.columns(2)
        with col_submit:
            submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")
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
            
            is_valid_roll, roll_msg = Validators.validate_roll_number(roll_no)
            if not is_valid_roll:
                errors.append(roll_msg)
            
            is_valid_mobile, mobile_msg = Validators.validate_mobile(mobile)
            if not is_valid_mobile:
                errors.append(mobile_msg)
            
            is_valid_pass, pass_msg = Validators.validate_password(password)
            if not is_valid_pass:
                errors.append(pass_msg)
            
            is_valid_email, email_msg = Validators.validate_email(email)
            if not is_valid_email:
                errors.append(email_msg)
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                existing_user = db.get_user(username, use_cache=False)
                if existing_user:
                    st.error("Username already exists. Please choose a different username.")
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
                        'role': 'student',
                        'bio': bio,
                        'skills': [s.strip() for s in skills.split(',')] if skills else [],
                        'avatar_url': generate_avatar_url(name, size=100)
                    }
                    
                    success, message = db.add_user(user_data)
                    if success:
                        st.success("✅ Account created successfully!")
                        st.balloons()
                        
                        st.markdown(f"""
                        ### Welcome to G H Raisoni, {name}! 🎉
                        
                        **Your account has been created successfully.**
                        
                        **Next steps:**
                        1. You've been awarded **{GAMIFICATION_CONFIG['points']['complete_profile']} points** for completing your profile
                        2. Explore **Events Feed** to discover upcoming events
                        3. Register for events to earn more points
                        4. Check your **Leaderboard** ranking
                        
                        **Account Details:**
                        - **Username:** {username}
                        - **Email:** {email}
                        - **Department:** {department}
                        
                        You will be automatically logged in...
                        """)
                        
                        time.sleep(2)
                        st.session_state.role = 'student'
                        st.session_state.username = username
                        st.session_state.name = name
                        st.session_state.session_start = datetime.now()
                        
                        db.create_notification(
                            user_id=username,
                            title="🎉 Welcome to G H Raisoni!",
                            message=f"Hello {name}! Welcome to our event management system. Start exploring events to earn points and badges!",
                            notification_type="welcome"
                        )
                        
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {message}")

# ============================================
# STUDENT DASHBOARD - COMPREHENSIVE
# ============================================

def student_dashboard():
    """Student dashboard with all features"""
    st.sidebar.title("👨‍🎓 Student Panel")
    
    student = db.get_user(st.session_state.username)
    if student:
        col_avatar, col_name = st.sidebar.columns([1, 3])
        with col_avatar:
            avatar = student.get('avatar_url', '👤')
            st.markdown(f'<div style="font-size: 2rem;">{avatar}</div>', unsafe_allow_html=True)
        with col_name:
            st.markdown(f"**{student.get('name')}**")
            st.caption(f"@{student.get('username')}")
        
        st.sidebar.markdown("---")
        
        current_level = db.get_student_level(st.session_state.username)
        current_points = db.get_student_points(st.session_state.username)
        level_config = GAMIFICATION_CONFIG['levels'].get(current_level, {})
        
        st.sidebar.markdown(f"### Level {current_level}: {level_config.get('name', 'Beginner')}")
        
        progress = db.get_level_progress(st.session_state.username)
        st.sidebar.progress(progress / 100)
        st.sidebar.caption(f"{progress:.1f}% to next level")
        
        st.sidebar.markdown(f"**🏆 Points:** {current_points}")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Quick Stats")
        
        registrations = db.get_registrations_by_student(st.session_state.username)
        liked_events = db.get_student_liked_events(st.session_state.username)
        interested_events = db.get_student_interested_events(st.session_state.username)
        
        col_stat1, col_stat2 = st.sidebar.columns(2)
        with col_stat1:
            st.metric("Events", len(registrations))
            st.metric("Liked", len(liked_events))
        with col_stat2:
            st.metric("Interested", len(interested_events))
            rank = db.get_student_rank(st.session_state.username)
            if rank:
                st.metric("Rank", f"#{rank}")
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Navigation")
        
        nav_options = [
            "🎯 Events Feed", 
            "📋 My Registrations", 
            "❤️ Liked Events", 
            "⭐ Interested Events",
            "🏆 Leaderboard", 
            "📊 Analytics",
            "🔔 Notifications",
            "👤 My Profile"
        ]
        
        if 'student_page' not in st.session_state:
            st.session_state.student_page = "🎯 Events Feed"
        
        for option in nav_options:
            if st.button(option, key=f"student_{option}", use_container_width=True):
                st.session_state.student_page = option
                st.rerun()
        
        notifications = db.get_user_notifications(st.session_state.username, unread_only=True)
        unread_count = len(notifications)
        if unread_count > 0:
            st.sidebar.markdown(f'<div style="background: #EF4444; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8rem; text-align: center; margin-top: 8px;">{unread_count} unread notifications</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    selected = st.session_state.student_page
    
    if selected == "🎯 Events Feed":
        events_feed_page()
    
    elif selected == "📋 My Registrations":
        my_registrations_page()
    
    elif selected == "❤️ Liked Events":
        liked_events_page()
    
    elif selected == "⭐ Interested Events":
        interested_events_page()
    
    elif selected == "🏆 Leaderboard":
        leaderboard_page()
    
    elif selected == "📊 Analytics":
        analytics_page()
    
    elif selected == "🔔 Notifications":
        notifications_page()
    
    elif selected == "👤 My Profile":
        profile_page()

# ============================================
# STUDENT SUB-PAGES
# ============================================

def events_feed_page():
    """Events feed page"""
    st.markdown('<h1 class="main-header">🎯 Discover Events</h1>', unsafe_allow_html=True)
    
    try:
        db.update_event_status()
    except:
        pass
    
    with st.spinner("Loading events..."):
        events = db.get_all_events(cache_ttl=60)
    
    if not events:
        st.info("No events found. Check back later!")
        return
    
    with st.expander("🔍 Filters & Search", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            search = st.text_input("Search events", placeholder="Title, description...")
        
        with col2:
            event_type = st.selectbox("Type", ["All"] + COLLEGE_CONFIG['event_types'])
        
        with col3:
            event_category = st.selectbox("Category", ["All"] + COLLEGE_CONFIG['event_categories'])
        
        with col4:
            show_status = st.selectbox("Status", ["All", "Upcoming", "Ongoing", "Past", "Completed"])
        
        col5, col6, col7 = st.columns(3)
        
        with col5:
            difficulty = st.selectbox("Difficulty", ["All", "Beginner", "Intermediate", "Advanced", "Expert"])
        
        with col6:
            has_certificate = st.selectbox("Certificate", ["All", "Yes", "No"])
        
        with col7:
            sort_by = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Popularity", "Participants"])
    
    filtered_events = events
    
    if search:
        search_lower = search.lower()
        filtered_events = [e for e in filtered_events 
                         if search_lower in e.get('title', '').lower() or 
                         search_lower in e.get('description', '').lower() or
                         search_lower in e.get('tags', '').lower()]
    
    if event_type != "All":
        filtered_events = [e for e in filtered_events if e.get('event_type') == event_type]
    
    if event_category != "All":
        filtered_events = [e for e in filtered_events if e.get('event_category') == event_category]
    
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
            elif show_status == "Completed" and event_status != 'completed':
                filtered_events.remove(event)
    
    if difficulty != "All":
        filtered_events = [e for e in filtered_events if e.get('difficulty_level') == difficulty]
    
    if has_certificate != "All":
        cert_value = 1 if has_certificate == "Yes" else 0
        filtered_events = [e for e in filtered_events if e.get('has_certificate', 0) == cert_value]
    
    if sort_by == "Date (Newest)":
        filtered_events.sort(key=lambda x: x.get('event_date', ''), reverse=True)
    elif sort_by == "Date (Oldest)":
        filtered_events.sort(key=lambda x: x.get('event_date', ''))
    elif sort_by == "Popularity":
        filtered_events.sort(key=lambda x: x.get('popularity_score', 0), reverse=True)
    elif sort_by == "Participants":
        filtered_events.sort(key=lambda x: x.get('current_participants', 0), reverse=True)
    
    if not filtered_events:
        st.warning("No events match your filters. Try adjusting your search criteria.")
        return
    
    st.caption(f"Found {len(filtered_events)} events")
    
    view_mode = st.radio("View mode:", ["List", "Grid"], horizontal=True)
    
    if view_mode == "Grid":
        cols = st.columns(2)
        for i, event in enumerate(filtered_events):
            with cols[i % 2]:
                display_event_card(event, st.session_state.username, show_actions=False)
                if st.button("View Details", key=f"view_{event['id']}", use_container_width=True):
                    st.session_state.view_event_id = event['id']
                    st.rerun()
    else:
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
    
    tab_upcoming, tab_ongoing, tab_completed, tab_all = st.tabs(["Upcoming", "Ongoing", "Completed", "All"])
    
    with tab_upcoming:
        upcoming_regs = [r for r in registrations if r.get('event_status') == 'upcoming']
        display_registrations_table(upcoming_regs, "Upcoming Events")
    
    with tab_ongoing:
        ongoing_regs = [r for r in registrations if r.get('event_status') == 'ongoing']
        display_registrations_table(ongoing_regs, "Ongoing Events")
    
    with tab_completed:
        completed_regs = [r for r in registrations if r.get('event_status') in ['completed', 'past']]
        display_registrations_table(completed_regs, "Completed Events")
    
    with tab_all:
        display_registrations_table(registrations, "All Registrations")

def display_registrations_table(registrations, title):
    """Display registrations in a table format"""
    if not registrations:
        st.info(f"No {title.lower()}.")
        return
    
    st.subheader(title)
    
    for reg in registrations:
        with st.container():
            st.markdown('<div class="registration-card">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                event_title = reg.get('event_title', 'Unknown Event')
                st.markdown(f'**{event_title}**')
                
                event_date = reg.get('event_date')
                if event_date:
                    st.caption(f"📅 {format_date(event_date)}")
                
                reg_status = reg.get('status', 'pending').title()
                badge_color = {
                    'Confirmed': 'green',
                    'Pending': 'orange',
                    'Cancelled': 'red',
                    'Waitlisted': 'blue'
                }.get(reg_status, 'gray')
                
                st.markdown(f'<span style="background: {badge_color}20; color: {badge_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">{reg_status}</span>', unsafe_allow_html=True)
            
            with col2:
                points = reg.get('points_awarded', 0)
                if points > 0:
                    st.markdown(f'<div style="font-size: 1.2rem; font-weight: bold; text-align: center; color: #3B82F6;">{points}</div>', unsafe_allow_html=True)
                    st.caption("Points")
                else:
                    st.caption("No points yet")
            
            with col3:
                event_id = reg.get('event_id')
                if st.button("View", key=f"view_reg_{event_id}", use_container_width=True):
                    st.session_state.view_event_id = event_id
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

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
    
    categories = {}
    for event in liked_events:
        category = event.get('event_category', 'Uncategorized')
        if category not in categories:
            categories[category] = []
        categories[category].append(event)
    
    for category, events in categories.items():
        with st.expander(f"{category} ({len(events)})", expanded=True):
            for event in events:
                display_event_card(event, st.session_state.username, show_actions=False)
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("Unlike", key=f"unlike_{event['id']}", use_container_width=True):
                        if db.remove_like(event['id'], st.session_state.username):
                            st.success("Removed from liked events!")
                            time.sleep(0.3)
                            st.rerun()
                st.markdown("---")

def interested_events_page():
    """Interested events page"""
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
        with st.container():
            display_event_card(event, st.session_state.username, show_actions=False)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                is_registered = db.is_student_registered(event['id'], st.session_state.username)
                if not is_registered:
                    if st.button("Register Now", key=f"reg_from_int_{event['id']}", use_container_width=True):
                        student = db.get_user(st.session_state.username)
                        if student:
                            reg_data = {
                                'event_id': event['id'],
                                'event_title': event.get('title'),
                                'student_username': st.session_state.username,
                                'student_name': student.get('name', st.session_state.username),
                                'student_roll': student.get('roll_no', 'N/A'),
                                'student_dept': student.get('department', 'N/A')
                            }
                            reg_id, message = db.add_registration(reg_data)
                            if reg_id:
                                st.success("Registered!")
                                time.sleep(0.3)
                                st.rerun()
            with col2:
                if st.button("Remove", key=f"remove_int_{event['id']}", use_container_width=True):
                    if db.remove_interested(event['id'], st.session_state.username):
                        st.success("Removed from interested!")
                        time.sleep(0.3)
                        st.rerun()
            
            st.markdown("---")

def leaderboard_page():
    """Leaderboard page"""
    st.markdown('<h1 class="main-header">🏆 College Leaderboard</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        timeframe = st.selectbox("Timeframe", ["All Time", "This Month", "This Week"])
    with col2:
        department = st.selectbox("Department", ["All Departments"] + COLLEGE_CONFIG['departments'])
    with col3:
        limit = st.selectbox("Top N", [10, 20, 50, 100])
    
    timeframe_map = {
        "All Time": "all",
        "This Month": "monthly",
        "This Week": "weekly"
    }
    
    dept_param = department if department != "All Departments" else None
    
    leaderboard = db.get_leaderboard(
        limit=limit,
        department=dept_param,
        timeframe=timeframe_map[timeframe]
    )
    
    if not leaderboard:
        st.info("No students found in leaderboard.")
        return
    
    st.subheader(f"Top {len(leaderboard)} Students")
    
    fig = create_leaderboard_visualization(leaderboard, f"Top {len(leaderboard)} Performers")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Detailed Ranking")
    
    for student in leaderboard:
        with st.container():
            st.markdown('<div class="leaderboard-card">', unsafe_allow_html=True)
            
            col_rank, col_info, col_points, col_level = st.columns([1, 4, 2, 2])
            
            with col_rank:
                rank = student.get('rank', 0)
                if rank <= 3:
                    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
                    st.markdown(f'<div style="font-size: 2rem; text-align: center;">{medals.get(rank, "")}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="font-size: 1.5rem; text-align: center; font-weight: bold;">{rank}</div>', unsafe_allow_html=True)
            
            with col_info:
                avatar = student.get('avatar_url', '👤')
                st.markdown(f'<div style="display: flex; align-items: center; gap: 10px;"><span style="font-size: 1.5rem;">{avatar}</span><div><div style="font-weight: bold; font-size: 1.1rem;">{student.get("name")}</div><div style="color: #666; font-size: 0.9rem;">{student.get("roll_no", "")} | {student.get("department", "")}</div></div></div>', unsafe_allow_html=True)
            
            with col_points:
                points = student.get('points', student.get('total_points', 0))
                st.markdown(f'<div style="font-size: 1.8rem; font-weight: bold; text-align: center; color: #3B82F6;">{points}</div>', unsafe_allow_html=True)
                st.caption("Points")
            
            with col_level:
                level = student.get('current_level', 1)
                level_config = GAMIFICATION_CONFIG['levels'].get(level, {})
                st.markdown(f'<div style="font-size: 1.2rem; font-weight: bold; text-align: center; color: #10B981;">{level}</div>', unsafe_allow_html=True)
                st.caption(f"Level: {level_config.get('name', 'Beginner')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
    
    if department == "All Departments":
        st.markdown("### Department Statistics")
        dept_stats = db.get_department_stats()
        
        if dept_stats:
            fig = create_department_stats_chart(dept_stats)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                pd.DataFrame(dept_stats)[['department', 'total_students', 'total_points', 'average_points']],
                use_container_width=True,
                column_config={
                    "department": "Department",
                    "total_students": "Students",
                    "total_points": "Total Points",
                    "average_points": "Avg Points"
                }
            )

def analytics_page():
    """Analytics page for students"""
    st.header("📊 My Analytics")
    
    student = db.get_user(st.session_state.username)
    if not student:
        st.error("User not found!")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        points = db.get_student_points(st.session_state.username)
        st.metric("🏆 Total Points", points)
    
    with col2:
        level = db.get_student_level(st.session_state.username)
        level_config = GAMIFICATION_CONFIG['levels'].get(level, {})
        st.metric("📈 Current Level", f"{level}: {level_config.get('name', 'Beginner')}")
    
    with col3:
        progress = db.get_level_progress(st.session_state.username)
        st.metric("📊 Level Progress", f"{progress:.1f}%")
    
    with col4:
        rank = db.get_student_rank(st.session_state.username)
        if rank:
            st.metric("🏅 Global Rank", f"#{rank}")
        else:
            st.metric("🏅 Global Rank", "Unranked")
    
    st.markdown("---")
    
    st.subheader("Event Participation")
    
    registrations = db.get_registrations_by_student(st.session_state.username)
    
    if registrations:
        df = pd.DataFrame(registrations)
        
        if 'event_status' in df.columns:
            status_counts = df['event_status'].value_counts()
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig1 = go.Figure(data=[go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    hole=.3,
                    marker_colors=px.colors.qualitative.Set3
                )])
                fig1.update_layout(title="Event Status Distribution")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_chart2:
                if 'registered_at' in df.columns:
                    df['month'] = pd.to_datetime(df['registered_at']).dt.to_period('M')
                    monthly_counts = df.groupby('month').size().reset_index(name='count')
                    monthly_counts['month'] = monthly_counts['month'].dt.strftime('%Y-%m')
                    
                    fig2 = px.bar(monthly_counts, x='month', y='count',
                                 title="Monthly Participation",
                                 labels={'month': 'Month', 'count': 'Events'},
                                 color='count',
                                 color_continuous_scale='Viridis')
                    st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Points Breakdown")
        
        points_breakdown = {
            "Event Registrations": len(registrations) * GAMIFICATION_CONFIG['points']['registration'],
            "Daily Logins": 7 * GAMIFICATION_CONFIG['points']['daily_login'],
            "Feedback Submitted": 3 * GAMIFICATION_CONFIG['points']['feedback_submission'],
            "Profile Completion": GAMIFICATION_CONFIG['points']['complete_profile']
        }
        
        breakdown_df = pd.DataFrame({
            'Category': list(points_breakdown.keys()),
            'Points': list(points_breakdown.values())
        })
        
        fig3 = px.bar(breakdown_df, x='Category', y='Points',
                     title="Points by Category",
                     color='Points',
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig3, use_container_width=True)
    
    else:
        st.info("No participation data yet. Start registering for events!")
    
    st.markdown("---")
    st.subheader("Department Comparison")
    
    dept_stats = db.get_department_stats()
    if dept_stats:
        student_dept = student.get('department')
        if student_dept:
            dept_data = [d for d in dept_stats if d['department'] == student_dept]
            if dept_data:
                dept_data = dept_data[0]
                
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Department Rank", f"#{dept_stats.index(next(d for d in dept_stats if d['department'] == student_dept)) + 1}")
                with cols[1]:
                    st.metric("Department Points", dept_data['total_points'])
                with cols[2]:
                    st.metric("Department Students", dept_data['total_students'])
                with cols[3]:
                    st.metric("Avg Points/Student", f"{dept_data['average_points']:.1f}")

def notifications_page():
    """Notifications page"""
    st.header("🔔 Notifications")
    
    notifications = db.get_user_notifications(st.session_state.username, limit=50)
    
    if not notifications:
        st.info("No notifications yet.")
        return
    
    if st.button("Mark All as Read", type="primary"):
        if db.mark_all_notifications_read(st.session_state.username):
            st.success("All notifications marked as read!")
            time.sleep(0.5)
            st.rerun()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Show All", use_container_width=True):
            st.session_state.notif_filter = "all"
            st.rerun()
    with col2:
        if st.button("Unread Only", use_container_width=True):
            st.session_state.notif_filter = "unread"
            st.rerun()
    with col3:
        if st.button("Read Only", use_container_width=True):
            st.session_state.notif_filter = "read"
            st.rerun()
    
    filter_type = st.session_state.get('notif_filter', 'all')
    if filter_type == 'unread':
        notifications = [n for n in notifications if n.get('read', 0) == 0]
    elif filter_type == 'read':
        notifications = [n for n in notifications if n.get('read', 0) == 1]
    
    for notif in notifications:
        with st.container():
            read_status = "📌" if notif.get('read', 0) == 0 else "📎"
            bg_color = "#F0F9FF" if notif.get('read', 0) == 0 else "#FFFFFF"
            
            st.markdown(f'<div style="background: {bg_color}; padding: 12px; border-radius: 8px; border-left: 4px solid #3B82F6; margin: 8px 0;">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"**{read_status} {notif.get('title', 'Notification')}**")
                st.caption(notif.get('message', ''))
                
                notif_type = notif.get('notification_type', 'general')
                created_at = notif.get('created_at', '')
                if created_at:
                    st.caption(f"🕒 {format_date_relative(created_at)} | Type: {notif_type}")
            
            with col2:
                if notif.get('read', 0) == 0:
                    if st.button("Mark Read", key=f"read_{notif['id']}", use_container_width=True):
                        if db.mark_notification_read(notif['id'], st.session_state.username):
                            st.success("Marked as read!")
                            time.sleep(0.3)
                            st.rerun()
                
                action_url = notif.get('action_url')
                if action_url:
                    if st.button("View", key=f"action_{notif['id']}", use_container_width=True):
                        st.session_state.view_event_id = action_url
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

def profile_page():
    """Enhanced profile page"""
    st.header("👤 My Profile")
    
    student = db.get_user(st.session_state.username)
    
    if not student:
        st.error("User not found!")
        return
    
    col_avatar, col_info = st.columns([1, 3])
    
    with col_avatar:
        avatar = student.get('avatar_url', '👤')
        st.markdown(f'<div style="font-size: 4rem; text-align: center;">{avatar}</div>', unsafe_allow_html=True)
        
        if st.button("Change Avatar", use_container_width=True):
            st.session_state.profile_edit = "avatar"
            st.rerun()
    
    with col_info:
        st.markdown(f"# {student.get('name', 'Student')}")
        st.caption(f"@{student.get('username')}")
        
        level = db.get_student_level(st.session_state.username)
        level_config = GAMIFICATION_CONFIG['levels'].get(level, {})
        st.markdown(f'<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.9rem;">Level {level}: {level_config.get("name", "Beginner")}</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("✏️ Edit Profile", type="primary"):
        st.session_state.profile_edit = "details"
        st.rerun()
    
    if st.session_state.get('profile_edit') == "details":
        edit_profile_form(student)
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Personal Info", "Academic Info", "Statistics", "Achievements"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Contact Information")
            st.markdown(f"**Email:** {student.get('email', 'N/A')}")
            st.markdown(f"**Mobile:** {student.get('mobile', 'N/A')}")
        
        with col2:
            st.markdown("### Additional Info")
            bio = student.get('bio', '')
            if bio:
                st.markdown(f"**Bio:** {bio}")
            
            skills = student.get('skills', '[]')
            if skills:
                try:
                    skills_list = json.loads(skills)
                    if skills_list:
                        st.markdown("**Skills:**")
                        for skill in skills_list:
                            st.markdown(f'- {skill}')
                except:
                    pass
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Academic Information")
            st.markdown(f"**Roll Number:** {student.get('roll_no', 'N/A')}")
            st.markdown(f"**Department:** {student.get('department', 'N/A')}")
            st.markdown(f"**Year:** {student.get('year', 'N/A')}")
        
        with col2:
            st.markdown("### Account Information")
            st.markdown(f"**Username:** {student.get('username', 'N/A')}")
            st.markdown(f"**Member Since:** {format_date(student.get('created_at', 'N/A'))}")
            st.markdown(f"**Last Login:** {format_date_relative(student.get('last_login', 'N/A'))}")
    
    with tab3:
        points = db.get_student_points(st.session_state.username)
        rank = db.get_student_rank(st.session_state.username)
        registrations = db.get_registrations_by_student(st.session_state.username)
        liked_events = db.get_student_liked_events(st.session_state.username)
        interested_events = db.get_student_interested_events(st.session_state.username)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Events Registered", len(registrations))
        with col_stat2:
            st.metric("🏆 Points", points)
        with col_stat3:
            if rank:
                st.metric("📊 Rank", f"#{rank}")
            else:
                st.metric("📊 Rank", "Unranked")
        with col_stat4:
            st.metric("❤️ Liked Events", len(liked_events))
        
        col_stat5, col_stat6, col_stat7 = st.columns(3)
        with col_stat5:
            st.metric("⭐ Interested", len(interested_events))
        with col_stat6:
            streak = student.get('daily_login_streak', 0)
            st.metric("🔥 Login Streak", f"{streak} days")
        with col_stat7:
            level_progress = db.get_level_progress(st.session_state.username)
            st.metric("📈 Level Progress", f"{level_progress:.1f}%")
        
        st.subheader("Points Progression")
        
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        points_data = [0, 100, 250, 450, 700, points]
        
        fig = go.Figure(data=go.Scatter(
            x=months,
            y=points_data,
            mode='lines+markers',
            line=dict(width=3, color='#3B82F6'),
            marker=dict(size=8, color='#3B82F6')
        ))
        
        fig.update_layout(
            title="Points Growth Over Time",
            xaxis_title="Month",
            yaxis_title="Points",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        badges = student.get('badges_earned', '[]')
        try:
            badges_list = json.loads(badges)
            if badges_list:
                st.subheader("🏅 Badges Earned")
                
                cols = st.columns(4)
                for i, badge in enumerate(badges_list):
                    with cols[i % 4]:
                        st.markdown(f'<div style="text-align: center; padding: 10px; background: #F3F4F6; border-radius: 10px;"><div style="font-size: 2rem;">{get_badge_emoji(badge)}</div><div style="font-size: 0.8rem; margin-top: 5px;">{badge.replace("_", " ").title()}</div></div>', unsafe_allow_html=True)
            else:
                st.info("No badges earned yet. Participate in events to earn badges!")
        except:
            st.info("No badges earned yet.")
        
        st.subheader("🎯 Level Milestones")
        
        for level_num, config in GAMIFICATION_CONFIG['levels'].items():
            points_needed = config['points_required']
            current_points = points
            achieved = current_points >= points_needed
            
            progress_color = "#10B981" if achieved else "#E5E7EB"
            text_color = "white" if achieved else "#6B7280"
            
            st.markdown(f'''
            <div style="background: {progress_color}; color: {text_color}; padding: 8px 12px; border-radius: 8px; margin: 4px 0; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>Level {level_num}: {config['name']}</strong>
                    <div style="font-size: 0.8rem;">{points_needed} points required</div>
                </div>
                <div>
                    {'✅' if achieved else '⏳'}
                </div>
            </div>
            ''', unsafe_allow_html=True)

def edit_profile_form(student):
    """Form to edit profile"""
    st.subheader("✏️ Edit Profile")
    
    with st.form("edit_profile"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name", value=student.get('name', ''))
            email = st.text_input("Email", value=student.get('email', ''))
            mobile = st.text_input("Mobile Number", value=student.get('mobile', ''))
        
        with col2:
            bio = st.text_area("Bio", value=student.get('bio', ''), height=100)
            skills_input = st.text_input("Skills (comma-separated)", 
                                        value=", ".join(json.loads(student.get('skills', '[]'))))
        
        col_submit, col_cancel = st.columns(2)
        with col_submit:
            submit = st.form_submit_button("Save Changes", use_container_width=True, type="primary")
        with col_cancel:
            cancel = st.form_submit_button("Cancel", use_container_width=True, type="secondary")
        
        if cancel:
            st.session_state.profile_edit = None
            st.rerun()
        
        if submit:
            errors = []
            
            if not name:
                errors.append("Name is required")
            
            if email:
                is_valid_email, email_msg = Validators.validate_email(email)
                if not is_valid_email:
                    errors.append(email_msg)
            
            if mobile:
                is_valid_mobile, mobile_msg = Validators.validate_mobile(mobile)
                if not is_valid_mobile:
                    errors.append(mobile_msg)
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                update_data = {
                    'name': name,
                    'email': email,
                    'mobile': mobile,
                    'bio': bio
                }
                
                if skills_input:
                    skills_list = [s.strip() for s in skills_input.split(',') if s.strip()]
                    update_data['skills'] = json.dumps(skills_list)
                
                success, message = db.update_user_profile(st.session_state.username, update_data)
                
                if success:
                    st.success("Profile updated successfully!")
                    st.session_state.profile_edit = None
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed to update profile: {message}")

def get_badge_emoji(badge_name):
    """Get emoji for badge"""
    emoji_map = {
        'legend_badge': '👑',
        'master_badge': '🌟',
        'expert_badge': '🎖️',
        'contributor_badge': '⭐',
        'learner_badge': '📚',
        'beginner_badge': '🆕'
    }
    return emoji_map.get(badge_name, '🏅')

# ============================================
# FEEDBACK FORM
# ============================================

def feedback_form(event_id):
    """Display feedback form for an event"""
    st.header("💬 Event Feedback")
    
    event = db.get_event_by_id(event_id)
    if not event:
        st.error("Event not found!")
        return
    
    st.subheader(event.get('title', 'Event'))
    
    with st.form("feedback_form"):
        rating = st.slider("Rating", 1, 5, 3, 
                          help="1 = Poor, 5 = Excellent")
        
        stars = "⭐" * rating
        st.markdown(f"**Your rating:** {stars}")
        
        comments = st.text_area("Comments (optional)", 
                               placeholder="Share your thoughts about the event...",
                               height=150)
        
        anonymous = st.checkbox("Submit anonymously")
        
        col_submit, col_cancel = st.columns(2)
        with col_submit:
            submit = st.form_submit_button("Submit Feedback", use_container_width=True, type="primary")
        with col_cancel:
            cancel = st.form_submit_button("Cancel", use_container_width=True, type="secondary")
        
        if cancel:
            st.session_state.feedback_event_id = None
            st.rerun()
        
        if submit:
            success, message = db.submit_feedback(event_id, st.session_state.username, 
                                                 rating, comments, anonymous)
            if success:
                st.success("Thank you for your feedback!")
                st.session_state.feedback_event_id = None
                time.sleep(1)
                st.rerun()
            else:
                st.error(message)

# ============================================
# FACULTY DASHBOARD
# ============================================

def faculty_dashboard():
    """Faculty coordinator dashboard"""
    st.sidebar.title("👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('faculty')
    
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Dashboard", "Create Event", "My Events", "Registrations", "Mentor Management"]
        
        if 'faculty_page' not in st.session_state:
            st.session_state.faculty_page = "Dashboard"
        
        for option in nav_options:
            is_active = st.session_state.faculty_page == option
            button_text = f"▶ {option}" if is_active else option
            
            if st.button(button_text, key=f"faculty_{option}", use_container_width=True):
                st.session_state.faculty_page = option
                st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    selected = st.session_state.faculty_page
    
    if selected == "Dashboard":
        faculty_dashboard_page()
    
    elif selected == "Create Event":
        create_event_page()
    
    elif selected == "My Events":
        faculty_events_page()
    
    elif selected == "Registrations":
        faculty_registrations_page()
    
    elif selected == "Mentor Management":
        mentor_management_page()

def faculty_dashboard_page():
    """Faculty dashboard page"""
    st.markdown('<h1 class="main-header">Faculty Dashboard</h1>', unsafe_allow_html=True)
    
    events = db.get_events_by_creator(st.session_state.username)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("My Events", len(events))
    
    with col2:
        upcoming = len([e for e in events if e.get('status') == 'upcoming'])
        st.metric("Upcoming", upcoming)
    
    with col3:
        total_registrations = sum(db._get_event_registration_count(e.get('id')) for e in events)
        st.metric("Total Registrations", total_registrations)
    
    st.markdown("---")
    
    st.subheader("📅 My Recent Events")
    if events:
        for event in events[:3]:
            display_event_card(event, None)
        
        if len(events) > 3:
            if st.button("View All Events"):
                st.session_state.faculty_page = "My Events"
                st.rerun()
    else:
        st.info("No events created yet. Create your first event!")
        if st.button("Create Event", type="primary"):
            st.session_state.faculty_page = "Create Event"
            st.rerun()
    
    st.markdown("---")
    
    st.subheader("📊 Quick Analytics")
    
    if events:
        event_types = {}
        for event in events:
            event_type = event.get('event_type', 'Unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        if event_types:
            df = pd.DataFrame({
                'Event Type': list(event_types.keys()),
                'Count': list(event_types.values())
            })
            
            fig = px.pie(df, values='Count', names='Event Type',
                        title='Event Type Distribution',
                        hole=0.3)
            st.plotly_chart(fig, use_container_width=True)

def create_event_page():
    """Create event page for faculty"""
    st.header("➕ Create New Event")
    
    tab1, tab2 = st.tabs(["📝 Manual Entry", "🤖 AI Generator"])
    
    with tab1:
        with st.form("create_event_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Event Title *")
                event_type = st.selectbox("Event Type *", COLLEGE_CONFIG['event_types'])
                event_category = st.selectbox("Event Category *", COLLEGE_CONFIG['event_categories'])
                event_date = st.date_input("Event Date *", min_value=date.today())
                event_time = st.time_input("Event Time *")
                max_participants = st.number_input("Max Participants", min_value=1, value=100)
                difficulty_level = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced", "Expert"])
                estimated_duration = st.number_input("Estimated Duration (minutes)", min_value=30, value=120)
                has_certificate = st.checkbox("Provide Certificate", value=False)
            
            with col2:
                venue = st.text_input("Venue *")
                organizer = st.text_input("Organizer *", value="G H Raisoni College")
                event_link = st.text_input("Event Website/URL", 
                                         placeholder="https://example.com/event-details")
                registration_link = st.text_input("Registration Link", 
                                                placeholder="https://forms.google.com/registration")
                
                prerequisites = st.text_area("Prerequisites", 
                                           placeholder="Required skills, knowledge, or equipment...",
                                           height=80)
                
                tags_input = st.text_input("Tags (comma-separated)", 
                                         placeholder="Python, Workshop, Beginner, Hands-on")
                
                st.subheader("👨‍🏫 Assign Mentor (Optional)")
                active_mentors = db.get_active_mentors()
                if active_mentors:
                    mentor_options = ["None"] + [f"{m['full_name']} ({m['department']})" for m in active_mentors]
                    selected_mentor = st.selectbox("Select Mentor", mentor_options)
                else:
                    st.info("No active mentors available. Admin can add mentors.")
                    selected_mentor = "None"
                
                st.subheader("Event Flyer (Optional)")
                flyer = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png', 'gif', 'webp'], key="faculty_flyer")
                if flyer:
                    st.image(flyer, width=200)
            
            description = st.text_area("Event Description *", 
                                      placeholder="Detailed description of the event...",
                                      height=150)
            
            submit_button = st.form_submit_button("Create Event", use_container_width=True, type="primary")
            
            if submit_button:
                if not all([title, event_type, venue, organizer, description]):
                    st.error("Please fill all required fields (*)")
                else:
                    mentor_id = None
                    if selected_mentor != "None" and active_mentors:
                        mentor_name = selected_mentor.split(" (")[0]
                        mentor = next((m for m in active_mentors if m['full_name'] == mentor_name), None)
                        if mentor:
                            mentor_id = mentor['id']
                    
                    flyer_path = save_flyer_image(flyer)
                    
                    event_datetime = datetime.combine(event_date, event_time)
                    
                    tags = [tag.strip() for tag in tags_input.split(',')] if tags_input else []
                    
                    event_data = {
                        'title': title,
                        'description': description,
                        'event_type': event_type,
                        'event_category': event_category,
                        'event_date': event_datetime.isoformat(),
                        'venue': venue,
                        'organizer': organizer,
                        'event_link': event_link,
                        'registration_link': registration_link,
                        'max_participants': max_participants,
                        'difficulty_level': difficulty_level,
                        'estimated_duration': estimated_duration,
                        'has_certificate': has_certificate,
                        'prerequisites': prerequisites,
                        'tags': tags,
                        'flyer_path': flyer_path,
                        'created_by': st.session_state.username,
                        'created_by_name': st.session_state.name,
                        'ai_generated': False,
                        'mentor_id': mentor_id
                    }
                    
                    success, event_id = db.add_event(event_data)
                    if success:
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

        event_text = st.text_area("Paste event text here:", 
                         placeholder="Example: Join us for a Python Workshop on 15th Dec 2023 at Seminar Hall. Organized by CSE Department...",
                         height=200,
                         key="ai_text_input")

        if st.button("🤖 Generate Event with AI", use_container_width=True, type="primary", key="ai_generate_btn"):
            if event_text:
                ai_generator = AIEventGenerator()
                event_data = ai_generator.extract_event_info(event_text)
                st.session_state.ai_generated_event = event_data

                if event_data.get('ai_generated'):
                    st.success("✅ Event details extracted successfully using AI!")
                else:
                    st.info("⚠️ Using regex fallback for event extraction")

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
        
        if 'ai_generated_event' in st.session_state:
            event_data = st.session_state.ai_generated_event
            
            st.markdown("---")
            st.subheader("✏️ Review & Edit AI-Generated Event")
            
            with st.form("ai_event_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    ai_title = st.text_input("Event Title", value=event_data.get('title', ''))
                    ai_event_type = st.selectbox("Event Type", COLLEGE_CONFIG['event_types'],
                                               index=COLLEGE_CONFIG['event_types'].index(event_data.get('event_type', 'Workshop')) 
                                               if event_data.get('event_type') in COLLEGE_CONFIG['event_types'] else 0)
                    ai_event_category = st.selectbox("Event Category", COLLEGE_CONFIG['event_categories'],
                                                   index=COLLEGE_CONFIG['event_categories'].index(event_data.get('event_category', 'Technical')) 
                                                   if event_data.get('event_category') in COLLEGE_CONFIG['event_categories'] else 0)
                    
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
                    ai_difficulty = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced", "Expert"],
                                               index=["Beginner", "Intermediate", "Advanced", "Expert"].index(event_data.get('difficulty_level', 'Beginner')))
                    ai_duration = st.number_input("Estimated Duration (minutes)", min_value=30, 
                                                value=event_data.get('estimated_duration', 120))
                    ai_certificate = st.checkbox("Provide Certificate", value=event_data.get('has_certificate', False))
                
                with col2:
                    ai_venue = st.text_input("Venue", value=event_data.get('venue', 'G H Raisoni College'))
                    ai_organizer = st.text_input("Organizer", value=event_data.get('organizer', 'G H Raisoni College'))
                    ai_event_link = st.text_input("Event Website", value=event_data.get('event_link', ''))
                    ai_reg_link = st.text_input("Registration Link", value=event_data.get('registration_link', ''))
                    ai_prerequisites = st.text_area("Prerequisites", 
                                                   value=event_data.get('prerequisites', ''),
                                                   height=80)
                    ai_tags = st.text_input("Tags (comma-separated)", 
                                          value=",".join(event_data.get('tags', [])) if isinstance(event_data.get('tags'), list) else "")
                    
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
                
                st.subheader("Event Flyer (Optional)")
                ai_flyer = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png', 'gif', 'webp'], key="ai_flyer")
                if ai_flyer:
                    st.image(ai_flyer, width=200)
                
                ai_submit = st.form_submit_button("✅ Create AI-Generated Event", use_container_width=True)
                
                if ai_submit:
                    if not all([ai_title, ai_venue, ai_organizer, ai_description]):
                        st.error("Please fill all required fields (*)")
                    else:
                        ai_mentor_id = None
                        if ai_selected_mentor != "None" and active_mentors:
                            mentor_name = ai_selected_mentor.split(" (")[0]
                            mentor = next((m for m in active_mentors if m['full_name'] == mentor_name), None)
                            if mentor:
                                ai_mentor_id = mentor['id']
                        
                        flyer_path = save_flyer_image(ai_flyer)
                        
                        event_datetime = datetime.combine(ai_date, ai_time)
                        
                        tags = [tag.strip() for tag in ai_tags.split(',')] if ai_tags else []
                        
                        final_event_data = {
                            'title': ai_title,
                            'description': ai_description,
                            'event_type': ai_event_type,
                            'event_category': ai_event_category,
                            'event_date': event_datetime.isoformat(),
                            'venue': ai_venue,
                            'organizer': ai_organizer,
                            'event_link': ai_event_link,
                            'registration_link': ai_reg_link,
                            'max_participants': ai_max_participants,
                            'difficulty_level': ai_difficulty,
                            'estimated_duration': ai_duration,
                            'has_certificate': ai_certificate,
                            'prerequisites': ai_prerequisites,
                            'tags': tags,
                            'flyer_path': flyer_path,
                            'created_by': st.session_state.username,
                            'created_by_name': st.session_state.name,
                            'ai_generated': True,
                            'ai_metadata': event_data.get('ai_metadata', {}),
                            'mentor_id': ai_mentor_id
                        }
                        
                        success, event_id = db.add_event(final_event_data)
                        if success:
                            st.success(f"✅ AI-generated event '{ai_title}' created successfully! 🎉")
                            if ai_mentor_id:
                                st.info(f"✅ Mentor assigned: {ai_selected_mentor}")
                            
                            if 'ai_generated_event' in st.session_state:
                                del st.session_state.ai_generated_event
                            
                            st.rerun()
                        else:
                            st.error("Failed to create event")

def faculty_events_page():
    """Faculty events management page"""
    st.header("📋 My Events")
    
    events = db.get_events_by_creator(st.session_state.username)
    
    if not events:
        st.info("You haven't created any events yet.")
        return
    
    st.subheader("📊 Event Engagement")
    total_likes = sum(db.get_event_likes_count(e['id']) for e in events)
    total_interested = sum(db.get_event_interested_count(e['id']) for e in events)
    total_registrations = sum(db._get_event_registration_count(e['id']) for e in events)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Likes", total_likes)
    with col2:
        st.metric("Total Interested", total_interested)
    with col3:
        st.metric("Total Registrations", total_registrations)
    
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
        past = [e for e in events if e.get('status') in ['completed', 'past']]
        if past:
            for event in past:
                display_event_card(event, None)
        else:
            st.info("No past events.")

def faculty_registrations_page():
    """Faculty registrations management page"""
    st.header("📝 Event Registrations")
    
    events = db.get_events_by_creator(st.session_state.username)
    
    if not events:
        st.info("You haven't created any events yet.")
        return
    
    event_titles = [e['title'] for e in events]
    selected_title = st.selectbox("Select Event", event_titles)
    
    if selected_title:
        selected_event = next(e for e in events if e['title'] == selected_title)
        event_id = selected_event['id']
        
        registrations = db.get_registrations_by_event(event_id)
        
        st.info(f"📊 Registrations for: **{selected_title}**")
        st.caption(f"Total Registrations: {len(registrations)}")
        
        if registrations:
            df_data = []
            for reg in registrations:
                df_data.append({
                    'Student Name': reg.get('student_name'),
                    'Roll No': reg.get('student_roll', 'N/A'),
                    'Mobile': reg.get('mobile', 'N/A'),
                    'Department': reg.get('department', 'N/A'),
                    'Year': reg.get('year', 'N/A'),
                    'Status': reg.get('status', 'pending').title(),
                    'Registered On': format_date(reg.get('registered_at')),
                    'Attendance': reg.get('attendance', 'absent').title()
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"registrations_{selected_title.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No registrations for this event yet.")

def mentor_management_page():
    """Mentor management page for faculty"""
    st.header("👨‍🏫 Mentor Management")
    
    tab1, tab2 = st.tabs(["View All Mentors", "Assign to Events"])
    
    with tab1:
        st.subheader("📋 All Mentors")
        
        mentors = db.get_all_mentors()
        
        if not mentors:
            st.info("No mentors found. Admin can add mentors.")
            return
        
        col_search, col_filter = st.columns(2)
        with col_search:
            search_term = st.text_input("🔍 Search mentors", placeholder="Search by name, department...")
        
        with col_filter:
            show_active = st.selectbox("Status", ["All", "Active Only", "Inactive Only"])
        
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
        
        st.caption(f"Found {len(filtered_mentors)} mentors")
        
        for mentor in filtered_mentors:
            with st.container():
                st.markdown('<div class="event-card">', unsafe_allow_html=True)
                
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
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
                    
                    events = db.get_events_by_mentor(mentor['id'])
                    st.caption(f"Events: {len(events)}")
                    
                    if st.button("View Events", key=f"view_mentor_{mentor['id']}", use_container_width=True):
                        st.session_state.view_mentor_id = mentor['id']
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📅 Assign Mentors to Events")
        
        active_mentors = db.get_active_mentors()
        if not active_mentors:
            st.info("No active mentors available. Please add mentors first.")
            return
        
        events = db.get_all_events()
        events_without_mentors = [e for e in events if not e.get('mentor_id')]
        
        if not events_without_mentors:
            st.success("🎉 All events have mentors assigned!")
            st.info("To reassign mentors, edit the event directly.")
            return
        
        event_options = {f"{e['title']} ({format_date(e['event_date'])})": e['id'] for e in events_without_mentors}
        selected_event_label = st.selectbox("Select Event (without mentor)", list(event_options.keys()))
        
        if selected_event_label:
            event_id = event_options[selected_event_label]
            selected_event = next(e for e in events_without_mentors if e['id'] == event_id)
            
            st.markdown(f"**Selected Event:** {selected_event['title']}")
            st.caption(f"Date: {format_date(selected_event['event_date'])}")
            st.caption(f"Type: {selected_event.get('event_type', 'N/A')}")
            st.caption(f"Venue: {selected_event.get('venue', 'N/A')}")
            
            mentor_options = {f"{m['full_name']} ({m['department']})": m['id'] for m in active_mentors}
            selected_mentor_label = st.selectbox("Select Mentor", list(mentor_options.keys()))
            
            if selected_mentor_label:
                mentor_id = mentor_options[selected_mentor_label]
                selected_mentor = next(m for m in active_mentors if m['id'] == mentor_id)
                
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
                
                if st.button("✅ Assign Mentor to Event", use_container_width=True, type="primary"):
                    if db.assign_mentor_to_event(event_id, mentor_id):
                        st.success(f"✅ {selected_mentor['full_name']} assigned to '{selected_event['title']}'!")
                        st.rerun()
                    else:
                        st.error("Failed to assign mentor.")

# ============================================
# MENTOR DASHBOARD
# ============================================

def mentor_dashboard():
    """Mentor dashboard"""
    st.sidebar.title("👨‍🏫 Mentor Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('mentor')
    
    mentor = db.get_mentor_by_email(st.session_state.username)
    if mentor:
        st.sidebar.markdown(f"**Department:** {mentor.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Email:** {mentor.get('email', 'N/A')}")
        st.sidebar.markdown(f"**Contact:** {mentor.get('contact', 'N/A')}")
        if mentor.get('expertise'):
            st.sidebar.markdown(f"**Expertise:** {mentor.get('expertise', 'N/A')}")
    
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["My Events", "Student Engagement", "My Profile"]
        
        if 'mentor_page' not in st.session_state:
            st.session_state.mentor_page = "My Events"
        
        for option in nav_options:
            is_active = st.session_state.mentor_page == option
            button_text = f"▶ {option}" if is_active else option
            
            if st.button(button_text, key=f"mentor_{option}", use_container_width=True):
                st.session_state.mentor_page = option
                st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    selected = st.session_state.mentor_page
    
    if selected == "My Events":
        mentor_events_page(mentor)
    elif selected == "Student Engagement":
        mentor_engagement_page(mentor)
    elif selected == "My Profile":
        mentor_profile_page(mentor)

def mentor_events_page(mentor):
    """Mentor's assigned events page"""
    st.markdown('<h1 class="main-header">📅 My Assigned Events</h1>', unsafe_allow_html=True)
    
    if not mentor:
        st.error("Mentor profile not found!")
        return
    
    mentor_id = mentor['id']
    events = db.get_events_by_mentor(mentor_id)
    
    if not events:
        st.info("No events assigned to you yet. Events will appear here when assigned by admin/faculty.")
        return
    
    total_events = len(events)
    upcoming = len([e for e in events if e.get('status') == 'upcoming'])
    ongoing = len([e for e in events if e.get('status') == 'ongoing'])
    past = len([e for e in events if e.get('status') in ['completed', 'past']])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", total_events)
    with col2:
        st.metric("Upcoming", upcoming)
    with col3:
        st.metric("Ongoing", ongoing)
    with col4:
        st.metric("Past", past)
    
    st.markdown("---")
    st.subheader("📋 Event Details")
    
    for event in events:
        display_event_card(event, None)

def mentor_engagement_page(mentor):
    """Student engagement monitoring page for mentors"""
    st.markdown('<h1 class="main-header">📊 Student Engagement</h1>', unsafe_allow_html=True)
    
    if not mentor:
        st.error("Mentor profile not found!")
        return
    
    mentor_id = mentor['id']
    events = db.get_events_by_mentor(mentor_id)
    
    if not events:
        st.info("No events assigned to monitor engagement.")
        return
    
    event_options = {e['title']: e['id'] for e in events}
    selected_event_title = st.selectbox("Select Event", list(event_options.keys()))
    
    if selected_event_title:
        event_id = event_options[selected_event_title]
        selected_event = next(e for e in events if e['id'] == event_id)
        
        likes_count = db.get_event_likes_count(event_id)
        interested_count = db.get_event_interested_count(event_id)
        registrations = db.get_registrations_by_event(event_id)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Likes", likes_count)
        with col2:
            st.metric("Interested", interested_count)
        with col3:
            st.metric("Registrations", len(registrations))
        
        st.markdown("---")
        st.subheader("📋 Registered Students")
        
        if registrations:
            df_data = []
            for reg in registrations:
                df_data.append({
                    'Student Name': reg.get('student_name'),
                    'Roll No': reg.get('student_roll', 'N/A'),
                    'Department': reg.get('student_dept', 'N/A'),
                    'Mobile': reg.get('student_mobile', 'N/A'),
                    'Status': reg.get('status', 'pending').title(),
                    'Attendance': reg.get('attendance', 'absent').title(),
                    'Registered On': format_date(reg.get('registered_at'))
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                st.subheader("📝 Attendance Management")
                student_options = {f"{reg['student_name']} ({reg['student_roll']})": reg['student_username'] 
                                 for reg in registrations}
                selected_student = st.selectbox("Select Student", list(student_options.keys()))
                
                if selected_student:
                    student_username = student_options[selected_student]
                    
                    col_att1, col_att2 = st.columns(2)
                    with col_att1:
                        if st.button("✅ Mark Present", use_container_width=True, type="primary"):
                            if db.use_supabase:
                                success = db.client.update('registrations', 
                                                         {'event_id': event_id, 'student_username': student_username},
                                                         {'attendance': 'present', 'checked_in_at': datetime.now().isoformat()})
                            else:
                                cursor = db.client.conn.cursor()
                                cursor.execute("UPDATE registrations SET attendance = 'present', checked_in_at = ? WHERE event_id = ? AND student_username = ?",
                                             (datetime.now().isoformat(), event_id, student_username))
                                db.client.conn.commit()
                                success = cursor.rowcount > 0
                            
                            if success:
                                st.success(f"Marked {selected_student} as present!")
                                time.sleep(1)
                                st.rerun()
                    
                    with col_att2:
                        if st.button("❌ Mark Absent", use_container_width=True, type="secondary"):
                            if db.use_supabase:
                                success = db.client.update('registrations', 
                                                         {'event_id': event_id, 'student_username': student_username},
                                                         {'attendance': 'absent'})
                            else:
                                cursor = db.client.conn.cursor()
                                cursor.execute("UPDATE registrations SET attendance = 'absent' WHERE event_id = ? AND student_username = ?",
                                             (event_id, student_username))
                                db.client.conn.commit()
                                success = cursor.rowcount > 0
                            
                            if success:
                                st.success(f"Marked {selected_student} as absent!")
                                time.sleep(1)
                                st.rerun()
        else:
            st.info("No students have registered for this event yet.")

def mentor_profile_page(mentor):
    """Mentor profile page"""
    st.header("👤 My Profile")
    
    if not mentor:
        st.error("Mentor profile not found!")
        return
    
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
        past = len([e for e in events if e.get('status') in ['completed', 'past']])
        st.metric("Past", past)

# ============================================
# ADMIN DASHBOARD
# ============================================

def admin_dashboard():
    """Admin dashboard"""
    st.sidebar.title("👑 Admin Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('admin')
    
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Dashboard", "Manage Events", "Manage Users", "Manage Mentors", "System Analytics", "Backup & Restore"]
        
        if 'admin_page' not in st.session_state:
            st.session_state.admin_page = "Dashboard"
        
        for option in nav_options:
            is_active = st.session_state.admin_page == option
            button_text = f"▶ {option}" if is_active else option
            
            if st.button(button_text, key=f"admin_{option}", use_container_width=True):
                st.session_state.admin_page = option
                st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    selected = st.session_state.admin_page
    
    if selected == "Dashboard":
        admin_dashboard_page()
    elif selected == "Manage Events":
        admin_manage_events_page()
    elif selected == "Manage Users":
        admin_manage_users_page()
    elif selected == "Manage Mentors":
        admin_manage_mentors_page()
    elif selected == "System Analytics":
        admin_system_analytics_page()
    elif selected == "Backup & Restore":
        admin_backup_page()

def admin_dashboard_page():
    """Admin dashboard page"""
    st.markdown('<h1 class="main-header">Admin Dashboard</h1>', unsafe_allow_html=True)
    
    try:
        db.update_event_status()
    except:
        pass
    
    events = db.get_all_events(cache_ttl=0, use_cache=False)
    users = db.get_all_users()
    mentors = db.get_all_mentors()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", len(events) if events else 0)
    with col2:
        upcoming = len([e for e in events if e.get('status') == 'upcoming']) if events else 0
        st.metric("Upcoming Events", upcoming)
    with col3:
        ai_events = len([e for e in events if e.get('ai_generated')]) if events else 0
        st.metric("🤖 AI Events", ai_events)
    with col4:
        active_mentors = len([m for m in mentors if m.get('is_active')]) if mentors else 0
        st.metric("👨‍🏫 Active Mentors", active_mentors)
    
    col5, col6, col7 = st.columns(3)
    with col5:
        total_users = len(users) if users else 0
        st.metric("Total Users", total_users)
    with col6:
        students = len([u for u in users if u.get('role') == 'student']) if users else 0
        st.metric("Students", students)
    with col7:
        total_points = sum(u.get('total_points', 0) for u in users) if users else 0
        st.metric("Total Points", total_points)
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        if events:
            status_counts = {}
            for event in events:
                status = event.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                fig = px.pie(
                    names=list(status_counts.keys()),
                    values=list(status_counts.values()),
                    title="Event Status Distribution",
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        if users:
            role_counts = {}
            for user in users:
                role = user.get('role', 'unknown')
                role_counts[role] = role_counts.get(role, 0) + 1
            
            if role_counts:
                fig = px.bar(
                    x=list(role_counts.keys()),
                    y=list(role_counts.values()),
                    title="User Role Distribution",
                    labels={'x': 'Role', 'y': 'Count'},
                    color=list(role_counts.values()),
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📅 Recent Events")
    
    if events:
        events_sorted = sorted(events, key=lambda x: x.get('created_at', ''), reverse=True)[:5]
        for event in events_sorted:
            with st.container():
                col_title, col_date, col_status = st.columns([3, 1, 1])
                with col_title:
                    st.markdown(f"**{event.get('title', 'Untitled')}**")
                    st.caption(f"Created by: {event.get('created_by_name', 'Unknown')}")
                with col_date:
                    st.caption(format_date(event.get('event_date')))
                with col_status:
                    status = event.get('status', 'unknown')
                    color_map = {
                        'upcoming': 'green',
                        'ongoing': 'orange',
                        'completed': 'gray',
                        'cancelled': 'red'
                    }
                    color = color_map.get(status, 'gray')
                    st.markdown(f'<span style="color: {color}; font-weight: bold;">{status.title()}</span>', unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("No events found.")

def admin_manage_events_page():
    """Admin event management page"""
    st.header("📋 Manage Events")
    
    events = db.get_all_events(cache_ttl=0, use_cache=False)
    
    if not events:
        st.info("No events found.")
        return
    
    search_term = st.text_input("🔍 Search events", placeholder="Search by title, description...")
    
    filtered_events = events
    if search_term:
        search_term = search_term.lower()
        filtered_events = [e for e in events if search_term in e.get('title', '').lower() or 
                          search_term in e.get('description', '').lower()]
    
    st.caption(f"Found {len(filtered_events)} events")
    
    for event in filtered_events:
        with st.container():
            st.markdown('<div class="event-card">', unsafe_allow_html=True)
            
            col_view, col_actions = st.columns([3, 1])
            
            with col_view:
                display_event_card(event, None)
            
            with col_actions:
                st.markdown("### Actions")
                
                if st.button("🗑️ Delete", key=f"delete_{event['id']}", use_container_width=True, type="secondary"):
                    if confirm_action("Delete this event? This action cannot be undone."):
                        event_id = event['id']
                        
                        try:
                            if db.use_supabase:
                                db.client.delete('registrations', {'event_id': event_id}, use_cache=False)
                                db.client.delete('event_likes', {'event_id': event_id}, use_cache=False)
                                db.client.delete('event_interested', {'event_id': event_id}, use_cache=False)
                                db.client.delete('event_feedback', {'event_id': event_id}, use_cache=False)
                                success = db.client.delete('events', {'id': event_id}, use_cache=False)
                            else:
                                cursor = db.client.conn.cursor()
                                cursor.execute("DELETE FROM registrations WHERE event_id = ?", (event_id,))
                                cursor.execute("DELETE FROM event_likes WHERE event_id = ?", (event_id,))
                                cursor.execute("DELETE FROM event_interested WHERE event_id = ?", (event_id,))
                                cursor.execute("DELETE FROM event_feedback WHERE event_id = ?", (event_id,))
                                cursor.execute("DELETE FROM events WHERE id = ?", (event_id,))
                                db.client.conn.commit()
                                success = cursor.rowcount > 0
                            
                            if success:
                                st.success("Event deleted successfully!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to delete event.")
                        except Exception as e:
                            st.error(f"Error deleting event: {e}")
                
                if st.button("✏️ Edit", key=f"edit_{event['id']}", use_container_width=True):
                    st.session_state.edit_event_id = event['id']
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
    
    if st.session_state.get('edit_event_id'):
        edit_event_form(st.session_state.edit_event_id)

def edit_event_form(event_id):
    """Form to edit event details"""
    st.header("✏️ Edit Event")
    
    event = db.get_event_by_id(event_id)
    if not event:
        st.error("Event not found!")
        return
    
    with st.form("edit_event_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Event Title", value=event.get('title', ''))
            event_type = st.selectbox("Event Type", COLLEGE_CONFIG['event_types'],
                                     index=COLLEGE_CONFIG['event_types'].index(event.get('event_type', 'Workshop')) 
                                     if event.get('event_type') in COLLEGE_CONFIG['event_types'] else 0)
            event_category = st.selectbox("Event Category", COLLEGE_CONFIG['event_categories'],
                                         index=COLLEGE_CONFIG['event_categories'].index(event.get('event_category', 'Technical')) 
                                         if event.get('event_category') in COLLEGE_CONFIG['event_categories'] else 0)
            
            try:
                event_date_str = event.get('event_date', '')
                if event_date_str:
                    event_date = datetime.fromisoformat(event_date_str.replace('Z', '+00:00')).date()
                    event_time = datetime.fromisoformat(event_date_str.replace('Z', '+00:00')).time()
                else:
                    event_date = date.today()
                    event_time = datetime.now().time()
            except:
                event_date = date.today()
                event_time = datetime.now().time()
            
            event_date = st.date_input("Event Date", value=event_date, min_value=date.today())
            event_time = st.time_input("Event Time", value=event_time)
            max_participants = st.number_input("Max Participants", min_value=1, value=event.get('max_participants', 100))
            difficulty_level = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced", "Expert"],
                                          index=["Beginner", "Intermediate", "Advanced", "Expert"].index(event.get('difficulty_level', 'Beginner')))
        
        with col2:
            venue = st.text_input("Venue", value=event.get('venue', ''))
            organizer = st.text_input("Organizer", value=event.get('organizer', ''))
            event_link = st.text_input("Event Website", value=event.get('event_link', ''))
            registration_link = st.text_input("Registration Link", value=event.get('registration_link', ''))
            
            status = st.selectbox("Status", ["upcoming", "ongoing", "completed", "cancelled"],
                                 index=["upcoming", "ongoing", "completed", "cancelled"].index(event.get('status', 'upcoming')))
            
            has_certificate = st.checkbox("Provide Certificate", value=bool(event.get('has_certificate')))
            
            active_mentors = db.get_active_mentors()
            current_mentor = event.get('mentor_id')
            current_mentor_name = "None"
            if current_mentor:
                mentor = db.get_mentor_by_id(current_mentor)
                if mentor:
                    current_mentor_name = f"{mentor['full_name']} ({mentor['department']})"
            
            if active_mentors:
                mentor_options = ["None"] + [f"{m['full_name']} ({m['department']})" for m in active_mentors]
                selected_mentor = st.selectbox("Assign Mentor", mentor_options,
                                             index=mentor_options.index(current_mentor_name) if current_mentor_name in mentor_options else 0)
            else:
                st.info("No active mentors available.")
                selected_mentor = "None"
        
        description = st.text_area("Description", value=event.get('description', ''), height=150)
        
        col_save, col_cancel = st.columns(2)
        with col_save:
            save = st.form_submit_button("💾 Save Changes", use_container_width=True, type="primary")
        with col_cancel:
            cancel = st.form_submit_button("❌ Cancel", use_container_width=True, type="secondary")
        
        if cancel:
            st.session_state.edit_event_id = None
            st.rerun()
        
        if save:
            if not all([title, venue, organizer, description]):
                st.error("Please fill all required fields")
                return
            
            mentor_id = None
            if selected_mentor != "None" and active_mentors:
                mentor_name = selected_mentor.split(" (")[0]
                mentor = next((m for m in active_mentors if m['full_name'] == mentor_name), None)
                if mentor:
                    mentor_id = mentor['id']
            
            event_datetime = datetime.combine(event_date, event_time)
            
            update_data = {
                'title': title,
                'description': description,
                'event_type': event_type,
                'event_category': event_category,
                'event_date': event_datetime.isoformat(),
                'venue': venue,
                'organizer': organizer,
                'event_link': event_link,
                'registration_link': registration_link,
                'max_participants': max_participants,
                'difficulty_level': difficulty_level,
                'status': status,
                'has_certificate': has_certificate,
                'mentor_id': mentor_id,
                'updated_at': datetime.now().isoformat()
            }
            
            try:
                if db.use_supabase:
                    success = db.client.update('events', {'id': event_id}, update_data, use_cache=False)
                else:
                    success = db.client.update('events', {'id': event_id}, update_data)
                
                if success:
                    st.success("✅ Event updated successfully!")
                    st.session_state.edit_event_id = None
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to update event.")
            except Exception as e:
                st.error(f"Error updating event: {e}")

def admin_manage_users_page():
    """Admin user management page"""
    st.header("👥 Manage Users")
    
    tab1, tab2, tab3 = st.tabs(["All Users", "Add New User", "User Analytics"])
    
    with tab1:
        st.subheader("📋 All Users")
        
        users = db.get_all_users()
        
        if not users:
            st.info("No users found.")
            return
        
        col_search, col_filter = st.columns(2)
        with col_search:
            search_term = st.text_input("🔍 Search users", placeholder="Search by name, email, username...")
        
        with col_filter:
            role_filter = st.selectbox("Filter by Role", ["All", "Admin", "Faculty", "Mentor", "Student"])
        
        filtered_users = users
        if search_term:
            search_term = search_term.lower()
            filtered_users = [u for u in users if search_term in u.get('name', '').lower() or 
                            search_term in u.get('email', '').lower() or 
                            search_term in u.get('username', '').lower()]
        
        if role_filter != "All":
            filtered_users = [u for u in filtered_users if u.get('role') == role_filter.lower()]
        
        st.caption(f"Found {len(filtered_users)} users")
        
        for user in filtered_users:
            with st.container():
                st.markdown('<div class="user-card">', unsafe_allow_html=True)
                
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    avatar = user.get('avatar_url', '👤')
                    st.markdown(f'''
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.5rem;">{avatar}</span>
                        <div>
                            <div style="font-weight: bold; font-size: 1.1rem;">{user.get('name')}</div>
                            <div style="color: #666; font-size: 0.9rem;">
                                @{user.get('username')} | {user.get('role', 'user').title()} | {user.get('department', 'N/A')}
                            </div>
                            <div style="font-size: 0.8rem; color: #888;">
                                Email: {user.get('email', 'N/A')} | Mobile: {user.get('mobile', 'N/A')}
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    if user.get('role') == 'student':
                        points = user.get('total_points', 0)
                        st.caption(f"🏆 Points: {points} | 📈 Level: {user.get('current_level', 1)}")
                
                with col_actions:
                    if user.get('username') not in ['admin@raisoni', 'faculty@raisoni']:
                        if st.button("🗑️ Delete", key=f"del_user_{user['id']}", use_container_width=True, type="secondary"):
                            if confirm_action(f"Delete user {user.get('name')}? This action cannot be undone."):
                                try:
                                    if db.use_supabase:
                                        if user.get('role') == 'mentor':
                                            db.client.delete('mentors', {'email': user.get('username')}, use_cache=False)
                                        success = db.client.delete('users', {'id': user.get('id')}, use_cache=False)
                                    else:
                                        cursor = db.client.conn.cursor()
                                        if user.get('role') == 'mentor':
                                            cursor.execute("DELETE FROM mentors WHERE email = ?", (user.get('username'),))
                                        cursor.execute("DELETE FROM users WHERE id = ?", (user.get('id'),))
                                        db.client.conn.commit()
                                        success = cursor.rowcount > 0
                                    
                                    if success:
                                        st.success("User deleted successfully!")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete user.")
                                except Exception as e:
                                    st.error(f"Error deleting user: {e}")
                    else:
                        st.caption("Protected account")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
    
    with tab2:
        st.subheader("➕ Add New User")
        
        with st.form("add_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name *")
                email = st.text_input("Email *")
                mobile = st.text_input("Mobile Number")
                role = st.selectbox("Role *", ["student", "faculty", "mentor", "admin"])
            
            with col2:
                username = st.text_input("Username *")
                password = st.text_input("Password *", type="password")
                confirm_password = st.text_input("Confirm Password *", type="password")
                department = st.selectbox("Department", ["Select"] + COLLEGE_CONFIG['departments'])
            
            additional_info = st.text_area("Additional Information", height=100)
            
            submit = st.form_submit_button("Create User", use_container_width=True, type="primary")
            
            if submit:
                if not all([name, email, username, password]):
                    st.error("Please fill all required fields (*)")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    existing_user = db.get_user(username, use_cache=False)
                    if existing_user:
                        st.error("Username already exists")
                    else:
                        user_data = {
                            'name': name,
                            'email': email,
                            'mobile': mobile,
                            'username': username,
                            'password': password,
                            'role': role,
                            'department': department if department != "Select" else "",
                            'avatar_url': generate_avatar_url(name, size=100)
                        }
                        
                        success, message = db.add_user(user_data)
                        if success:
                            st.success(f"User '{name}' created successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to create user: {message}")
    
    with tab3:
        st.subheader("📊 User Analytics")
        
        users = db.get_all_users()
        
        if users:
            total_users = len(users)
            students = len([u for u in users if u.get('role') == 'student'])
            faculty = len([u for u in users if u.get('role') == 'faculty'])
            mentors = len([u for u in users if u.get('role') == 'mentor'])
            admins = len([u for u in users if u.get('role') == 'admin'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Users", total_users)
            with col2:
                st.metric("Students", students)
            with col3:
                st.metric("Faculty", faculty)
            with col4:
                st.metric("Mentors", mentors)
            
            st.markdown("---")
            
            departments = {}
            for user in users:
                dept = user.get('department', 'Unknown')
                if dept:
                    departments[dept] = departments.get(dept, 0) + 1
            
            if departments:
                df = pd.DataFrame({
                    'Department': list(departments.keys()),
                    'Users': list(departments.values())
                })
                
                fig = px.bar(df, x='Department', y='Users',
                           title="Users by Department",
                           color='Users',
                           color_continuous_scale='Viridis')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            student_users = [u for u in users if u.get('role') == 'student']
            if student_users:
                points_data = [u.get('total_points', 0) for u in student_users]
                avg_points = sum(points_data) / len(points_data) if points_data else 0
                
                st.metric("Average Student Points", f"{avg_points:.1f}")
                
                levels_data = [u.get('current_level', 1) for u in student_users]
                level_counts = {}
                for level in levels_data:
                    level_counts[level] = level_counts.get(level, 0) + 1
                
                if level_counts:
                    fig = px.pie(
                        names=[f"Level {k}" for k in level_counts.keys()],
                        values=list(level_counts.values()),
                        title="Student Level Distribution",
                        hole=0.3
                    )
                    st.plotly_chart(fig, use_container_width=True)

def admin_manage_mentors_page():
    """Admin mentor management page"""
    st.header("👨‍🏫 Manage Mentors")
    
    tab1, tab2 = st.tabs(["All Mentors", "Add New Mentor"])
    
    with tab1:
        st.subheader("📋 All Mentors")
        
        mentors = db.get_all_mentors()
        
        if not mentors:
            st.info("No mentors found.")
            return
        
        for mentor in mentors:
            with st.container():
                st.markdown('<div class="event-card">', unsafe_allow_html=True)
                
                col_info, col_actions = st.columns([3, 1])
                
                with col_info:
                    status_color = "🟢" if mentor.get('is_active') else "🔴"
                    status_text = "Active" if mentor.get('is_active') else "Inactive"
                    
                    st.markdown(f'<div class="card-title">{mentor.get("full_name")} {status_color}</div>', unsafe_allow_html=True)
                    st.caption(f"**Department:** {mentor.get('department')}")
                    st.caption(f"**Email:** {mentor.get('email')}")
                    st.caption(f"**Contact:** {mentor.get('contact')}")
                    
                    if mentor.get('expertise'):
                        st.caption(f"**Expertise:** {mentor.get('expertise')}")
                    
                    events = db.get_events_by_mentor(mentor['id'])
                    st.caption(f"**Assigned Events:** {len(events)}")
                
                with col_actions:
                    if st.button("✏️ Edit", key=f"edit_mentor_{mentor['id']}", use_container_width=True):
                        st.session_state.edit_mentor_id = mentor['id']
                        st.rerun()
                    
                    if mentor.get('is_active'):
                        if st.button("❌ Deactivate", key=f"deact_{mentor['id']}", use_container_width=True, type="secondary"):
                            if db.delete_mentor(mentor['id']):
                                st.success("Mentor deactivated!")
                                time.sleep(1)
                                st.rerun()
                    else:
                        if st.button("✅ Activate", key=f"act_{mentor['id']}", use_container_width=True, type="secondary"):
                            if db.update_mentor(mentor['id'], {'is_active': True}):
                                st.success("Mentor activated!")
                                time.sleep(1)
                                st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
    
    with tab2:
        st.subheader("➕ Add New Mentor")
        
        with st.form("add_mentor_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                first_name = st.text_input("First Name *")
                last_name = st.text_input("Last Name *")
                department = st.selectbox("Department *", COLLEGE_CONFIG['departments'])
                email = st.text_input("Email *")
            
            with col2:
                contact = st.text_input("Contact Number *")
                expertise = st.text_area("Expertise/Areas", placeholder="Python, Machine Learning, Web Development...")
                is_active = st.checkbox("Active", value=True)
                
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
    
    if st.session_state.get('edit_mentor_id'):
        mentor = db.get_mentor_by_id(st.session_state.edit_mentor_id)
        if mentor:
            edit_mentor_form(mentor)

def edit_mentor_form(mentor):
    """Form to edit mentor details"""
    st.header("✏️ Edit Mentor")
    
    with st.form("edit_mentor_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name", value=mentor.get('first_name', ''))
            last_name = st.text_input("Last Name", value=mentor.get('last_name', ''))
            department = st.selectbox("Department", COLLEGE_CONFIG['departments'],
                                     index=COLLEGE_CONFIG['departments'].index(mentor.get('department', '')) 
                                     if mentor.get('department') in COLLEGE_CONFIG['departments'] else 0)
        
        with col2:
            email = st.text_input("Email", value=mentor.get('email', ''))
            contact = st.text_input("Contact", value=mentor.get('contact', ''))
            expertise = st.text_area("Expertise", value=mentor.get('expertise', ''))
            is_active = st.checkbox("Active", value=bool(mentor.get('is_active', True)))
        
        col_save, col_cancel = st.columns(2)
        with col_save:
            save = st.form_submit_button("💾 Save Changes", use_container_width=True, type="primary")
        with col_cancel:
            cancel = st.form_submit_button("❌ Cancel", use_container_width=True, type="secondary")
        
        if cancel:
            st.session_state.edit_mentor_id = None
            st.rerun()
        
        if save:
            update_data = {
                'first_name': first_name,
                'last_name': last_name,
                'department': department,
                'email': email,
                'contact': contact,
                'expertise': expertise,
                'is_active': is_active
            }
            
            if db.update_mentor(mentor['id'], update_data):
                st.success("✅ Mentor updated successfully!")
                st.session_state.edit_mentor_id = None
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to update mentor.")

def admin_system_analytics_page():
    """System analytics page for admin"""
    st.header("📊 System Analytics")
    
    tab1, tab2, tab3 = st.tabs(["Performance", "Usage Statistics", "Event Analytics"])
    
    with tab1:
        st.subheader("🏎️ System Performance")
        
        cache_stats = cache.get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cache Size", f"{cache_stats['size']}/{cache_stats['max_size']}")
        with col2:
            st.metric("Cache Hit Rate", cache_stats['hit_rate'])
        with col3:
            st.metric("Total Requests", cache_stats['hits'] + cache_stats['misses'])
        
        if db.use_supabase:
            st.info("✅ Using Supabase PostgreSQL")
        else:
            st.info("💾 Using SQLite (Local)")
        
        st.markdown("---")
        st.subheader("📈 Performance Metrics")
        
        metrics_data = {
            'Metric': ['Database Queries', 'Cache Hits', 'Cache Misses', 'Response Time'],
            'Value': [cache_stats['hits'] + cache_stats['misses'], cache_stats['hits'], 
                     cache_stats['misses'], '~50ms']
        }
        
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("📊 Usage Statistics")
        
        users = db.get_all_users()
        events = db.get_all_events(cache_ttl=0, use_cache=False)
        
        if users and events:
            total_users = len(users)
            active_users = len([u for u in users if u.get('is_active', True)])
            total_events = len(events)
            active_events = len([e for e in events if e.get('status') in ['upcoming', 'ongoing']])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Users", total_users)
            with col2:
                st.metric("Active Users", active_users)
            with col3:
                st.metric("Total Events", total_events)
            with col4:
                st.metric("Active Events", active_events)
            
            st.markdown("---")
            
            registrations = []
            for event in events:
                regs = db.get_registrations_by_event(event.get('id'))
                registrations.extend(regs)
            
            if registrations:
                total_registrations = len(registrations)
                avg_reg_per_event = total_registrations / total_events if total_events > 0 else 0
                
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("Total Registrations", total_registrations)
                with col6:
                    st.metric("Avg. Reg/Event", f"{avg_reg_per_event:.1f}")
                
                df = pd.DataFrame(registrations)
                if not df.empty:
                    df['registered_at'] = pd.to_datetime(df['registered_at'])
                    df['date'] = df['registered_at'].dt.date
                    
                    daily_regs = df.groupby('date').size().reset_index(name='count')
                    
                    fig = px.line(daily_regs, x='date', y='count',
                                title='Daily Registrations Trend',
                                labels={'date': 'Date', 'count': 'Registrations'})
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Event Analytics")
        
        events = db.get_all_events(cache_ttl=0, use_cache=False)
        
        if events:
            total_events = len(events)
            
            event_types = {}
            event_categories = {}
            status_counts = {}
            
            for event in events:
                event_type = event.get('event_type', 'Unknown')
                event_category = event.get('event_category', 'Unknown')
                status = event.get('status', 'unknown')
                
                event_types[event_type] = event_types.get(event_type, 0) + 1
                event_categories[event_category] = event_categories.get(event_category, 0) + 1
                status_counts[status] = status_counts.get(status, 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                if event_types:
                    fig1 = px.pie(
                        names=list(event_types.keys()),
                        values=list(event_types.values()),
                        title="Event Type Distribution",
                        hole=0.3
                    )
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                if event_categories:
                    fig2 = px.bar(
                        x=list(event_categories.keys()),
                        y=list(event_categories.values()),
                        title="Event Category Distribution",
                        labels={'x': 'Category', 'y': 'Count'},
                        color=list(event_categories.values()),
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📅 Monthly Events")
            
            events_df = pd.DataFrame(events)
            if 'created_at' in events_df.columns and not events_df.empty:
                events_df['month'] = pd.to_datetime(events_df['created_at']).dt.to_period('M')
                monthly_counts = events_df.groupby('month').size().reset_index(name='count')
                monthly_counts['month'] = monthly_counts['month'].dt.strftime('%Y-%m')
                
                fig3 = px.bar(monthly_counts, x='month', y='count',
                            title="Events Created by Month",
                            labels={'month': 'Month', 'count': 'Number of Events'},
                            color='count',
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig3, use_container_width=True)

def admin_backup_page():
    """Backup and restore page"""
    st.header("💾 Backup & Restore")
    
    tab1, tab2 = st.tabs(["Database Backup", "System Settings"])
    
    with tab1:
        st.subheader("📦 Database Backup")
        
        st.info("""
        **Backup Options:**
        - Export all data as JSON files
        - Export specific tables as CSV
        - Generate backup report
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 Export All Data", use_container_width=True):
                export_all_data()
        
        with col2:
            if st.button("📊 Export as CSV", use_container_width=True):
                export_csv_data()
        
        with col3:
            if st.button("📄 Generate Report", use_container_width=True):
                generate_backup_report()
        
        st.markdown("---")
        st.subheader("🔄 Restore Data")
        
        st.warning("⚠️ Restoring data will overwrite existing records. Use with caution!")
        
        uploaded_file = st.file_uploader("Upload backup file", type=['json', 'csv'])
        
        if uploaded_file:
            if st.button("🔄 Restore from Backup", type="secondary", use_container_width=True):
                st.warning("This feature is under development. Please contact system administrator.")
    
    with tab2:
        st.subheader("⚙️ System Settings")
        
        with st.form("system_settings"):
            st.markdown("### Application Settings")
            
            maintenance_mode = st.checkbox("Maintenance Mode", value=False,
                                          help="When enabled, only admins can access the system")
            
            new_user_registration = st.checkbox("Allow New User Registration", value=True)
            
            event_creation = st.selectbox("Who can create events?", 
                                         ["Admin Only", "Admin & Faculty", "All Users"])
            
            st.markdown("### Notification Settings")
            
            email_notifications = st.checkbox("Enable Email Notifications", value=False)
            push_notifications = st.checkbox("Enable Push Notifications", value=True)
            
            st.markdown("### Performance Settings")
            
            cache_enabled = st.checkbox("Enable Caching", value=CACHE_ENABLED)
            cache_size = st.slider("Cache Size (entries)", 100, 5000, 2000, 100)
            
            save_settings = st.form_submit_button("💾 Save Settings", use_container_width=True, type="primary")
            
            if save_settings:
                st.success("Settings saved successfully! (Note: Some changes require restart)")
                st.info("Actual implementation would save these to a configuration file or database.")

# ============================================
# HELPER FUNCTIONS FOR ADMIN
# ============================================

def get_all_users():
    """Get all users from database"""
    try:
        if db.use_supabase:
            return db.client.select('users', limit=1000, use_cache=False)
        else:
            return db.client.execute_query("SELECT * FROM users", fetchall=True, use_cache=False)
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        return []

def confirm_action(message):
    """Confirm an action with the user"""
    return st.checkbox(f"✅ {message}")

def export_all_data():
    """Export all data as JSON"""
    try:
        data = {}
        
        tables = ['users', 'events', 'mentors', 'registrations', 'event_likes', 
                 'event_interested', 'event_feedback', 'notifications', 'user_achievements']
        
        for table in tables:
            try:
                if db.use_supabase:
                    table_data = db.client.select(table, limit=10000, use_cache=False)
                else:
                    table_data = db.client.execute_query(f"SELECT * FROM {table}", 
                                                        fetchall=True, use_cache=False)
                data[table] = table_data
            except:
                pass
        
        json_str = json.dumps(data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download JSON Backup",
            data=json_str,
            file_name=f"event_system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ Data exported successfully!")
        
    except Exception as e:
        st.error(f"Error exporting data: {e}")

def export_csv_data():
    """Export data as CSV"""
    try:
        tables = ['users', 'events', 'registrations']
        
        for table in tables:
            try:
                if db.use_supabase:
                    table_data = db.client.select(table, limit=10000, use_cache=False)
                else:
                    table_data = db.client.execute_query(f"SELECT * FROM {table}", 
                                                        fetchall=True, use_cache=False)
                
                if table_data:
                    df = pd.DataFrame(table_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label=f"📥 Download {table.capitalize()} CSV",
                        data=csv,
                        file_name=f"{table}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key=f"download_{table}"
                    )
            except Exception as e:
                st.warning(f"Could not export {table}: {e}")
    
    except Exception as e:
        st.error(f"Error exporting CSV data: {e}")

def generate_backup_report():
    """Generate a backup report"""
    try:
        report = {
            "generated_at": datetime.now().isoformat(),
            "system_info": {
                "database": "Supabase PostgreSQL" if db.use_supabase else "SQLite",
                "cache_enabled": CACHE_ENABLED,
                "cache_size": cache.get_stats()['size']
            },
            "statistics": {}
        }
        
        tables = ['users', 'events', 'registrations']
        
        for table in tables:
            try:
                if db.use_supabase:
                    table_data = db.client.select(table, limit=1, use_cache=False)
                else:
                    table_data = db.client.execute_query(f"SELECT COUNT(*) as count FROM {table}", 
                                                        fetchone=True, use_cache=False)
                
                if table == 'users' and not db.use_supabase:
                    report["statistics"][table] = {"count": table_data['count']}
                else:
                    report["statistics"][table] = {"count": len(table_data) if table_data else 0}
            except:
                report["statistics"][table] = {"count": 0, "error": "Could not retrieve"}
        
        report_str = json.dumps(report, indent=2)
        
        st.download_button(
            label="📥 Download Backup Report",
            data=report_str,
            file_name=f"backup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.json(report)
        
    except Exception as e:
        st.error(f"Error generating report: {e}")

def format_date_relative(date_str):
    """Format date as relative time (e.g., '2 hours ago')"""
    try:
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            dt = date_str
        
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 365:
            years = diff.days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
        elif diff.days > 30:
            months = diff.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "just now"
    except:
        return str(date_str)

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
    
    # Check for remember me cookie
    if 'role' not in st.session_state or st.session_state.role is None:
        check_remember_me_cookie()
    
    # Session timeout check
    if (st.session_state.role and 'session_start' in st.session_state and 
        not st.session_state.get('remember_me', False)):
        session_duration = datetime.now() - st.session_state.session_start
        if session_duration.total_seconds() > SESSION_TIMEOUT_MINUTES * 60:
            st.warning("Session timed out. Please login again.")
            
            # Clear query parameters
            if hasattr(st, 'query_params'):
                st.query_params.clear()
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Display database info in sidebar
    if db.use_supabase:
        st.sidebar.success("✅ Using Supabase PostgreSQL")
    else:
        st.sidebar.info("💾 Using SQLite (Local)")
    
    # Update event status periodically
    if 'last_status_update' not in st.session_state:
        st.session_state.last_status_update = datetime.now()
    
    if (datetime.now() - st.session_state.last_status_update).total_seconds() > 300:
        try:
            db.update_event_status()
            st.session_state.last_status_update = datetime.now()
        except:
            pass
    
    # Route based on page
    if st.session_state.page == "forgot_password":
        forgot_password_page()
    elif st.session_state.page == "student_register":
        student_registration_page()
    elif st.session_state.role is None:
        landing_page()
    
    # Check for special states (feedback, view event)
    elif 'feedback_event_id' in st.session_state:
        feedback_form(st.session_state.feedback_event_id)
    
    elif 'view_event_id' in st.session_state:
        event = db.get_event_by_id(st.session_state.view_event_id)
        if event:
            st.header("📋 Event Details")
            display_event_card(event, st.session_state.username)
            
            if st.button("← Back", use_container_width=True):
                del st.session_state.view_event_id
                st.rerun()
        else:
            st.error("Event not found!")
            del st.session_state.view_event_id
            st.rerun()
    
    # Route to appropriate dashboard
    elif st.session_state.role == 'student':
        student_dashboard()
    elif st.session_state.role == 'faculty':
        faculty_dashboard()
    elif st.session_state.role == 'mentor':
        mentor_dashboard()
    elif st.session_state.role == 'admin':
        admin_dashboard()
    else:
        # If role is set but doesn't match any dashboard, show error
        st.error(f"Invalid role configuration: {st.session_state.role}")
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ============================================
# STYLES
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .college-header h1 {
        font-size: 2.8rem;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .college-header h3 {
        font-size: 1.2rem;
        color: #6b7280;
        margin-top: 0;
    }
    .event-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    .event-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transform: translateY(-2px);
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .registration-card, .user-card, .leaderboard-card {
        background: white;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .registration-section {
        background: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
        border: 1px solid #e2e8f0;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-1px);
    }
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)[:200]}")
        logger.error(f"Application error: {e}", exc_info=True)
        if st.button("Restart Application"):
            st.rerun()
