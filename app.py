"""
G H Raisoni College - Advanced Event Management System
Complete solution with AI, Image Uploads, Social Features
Deployable on Streamlit Cloud
"""

import streamlit as st
from datetime import datetime, date, timedelta
import json
import os
import hashlib
import uuid
from pathlib import Path
import pandas as pd
from PIL import Image
import io
import re
import sqlite3
from streamlit_option_menu import option_menu
import base64
import tempfile
import traceback
import logging
from functools import lru_cache
import bcrypt
import time

# ============================================
# CONFIGURATION
# ============================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="G H Raisoni Event Manager",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    
    .college-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .event-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .event-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border-color: #3B82F6;
    }
    
    .event-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #3B82F6, #1E3A8A);
    }
    
    .role-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .admin-badge { background: #FEE2E2; color: #DC2626; }
    .faculty-badge { background: #DBEAFE; color: #1D4ED8; }
    .student-badge { background: #D1FAE5; color: #065F46; }
    
    .ai-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .social-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .social-btn {
        flex: 1;
        min-width: 80px;
        text-align: center;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        background: white;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .social-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .social-btn.active {
        border-color: #3B82F6;
        background: #DBEAFE;
        color: #1E40AF;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #6B7280;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .flyer-container {
        border: 2px dashed #3B82F6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
        background: #F8FAFC;
    }
    
    .registration-section {
        background: #F0F9FF;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #3B82F6;
    }
    
    .stSpinner > div {
        border-color: #3B82F6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# ENHANCED SQLite DATABASE MANAGER
# ============================================
class EnhancedDatabaseManager:
    """Manage all data using SQLite database with enhanced features"""
    
    def __init__(self, db_path="event_management.db"):
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()
        
        # Default credentials
        self.default_creds = {
            'admin': {'username': 'admin@raisoni', 'password': 'admin123'},
            'faculty': {'username': 'faculty@raisoni', 'password': 'faculty123'}
        }
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info("Database connected successfully")
        except Exception as e:
            st.error(f"Database connection error: {e}")
            logger.error(f"Database connection error: {e}")
    
    def create_tables(self):
        """Create all necessary tables if they don't exist"""
        try:
            cursor = self.conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    roll_no TEXT,
                    department TEXT,
                    year TEXT,
                    email TEXT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Events table
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
                    flyer_path TEXT,
                    created_by TEXT,
                    created_by_name TEXT,
                    ai_generated BOOLEAN DEFAULT 0,
                    ai_prompt TEXT,
                    likes_count INTEGER DEFAULT 0,
                    favorites_count INTEGER DEFAULT 0,
                    interested_count INTEGER DEFAULT 0,
                    shares_count INTEGER DEFAULT 0,
                    views_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Event likes table (separate for better querying)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS event_likes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE,
                    FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE,
                    UNIQUE(event_id, username)
                )
            ''')
            
            # Event favorites table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS event_favorites (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE,
                    FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE,
                    UNIQUE(event_id, username)
                )
            ''')
            
            # Event interested table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS event_interested (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE,
                    FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE,
                    UNIQUE(event_id, username)
                )
            ''')
            
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
                    via_link BOOLEAN DEFAULT 0,
                    via_app BOOLEAN DEFAULT 1,
                    status TEXT DEFAULT 'pending',
                    attendance TEXT DEFAULT 'absent',
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE,
                    FOREIGN KEY (student_username) REFERENCES users (username) ON DELETE CASCADE
                )
            ''')
            
            # Social interactions log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS social_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES events (id) ON DELETE CASCADE,
                    FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_creator ON events(created_by)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_registrations_event ON registrations(event_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_registrations_student ON registrations(student_username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_likes_event ON event_likes(event_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_likes_user ON event_likes(username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_favorites_event ON event_favorites(event_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_favorites_user ON event_favorites(username)')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
            
            # Add default admin and faculty if not exists
            self._add_default_users()
            
        except Exception as e:
            st.error(f"Error creating tables: {e}")
            logger.error(f"Error creating tables: {e}")
            traceback.print_exc()
    
    def _add_default_users(self):
        """Add default admin and faculty users"""
        try:
            cursor = self.conn.cursor()
            
            # Check if admin exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ('admin@raisoni',))
            if cursor.fetchone()[0] == 0:
                admin_id = str(uuid.uuid4())
                hashed_pass = self._hash_password('admin123')
                cursor.execute('''
                    INSERT INTO users (id, name, username, password, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (admin_id, 'Administrator', 'admin@raisoni', hashed_pass, 'admin', datetime.now().isoformat()))
                logger.info("Default admin user created")
            
            # Check if faculty exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ('faculty@raisoni',))
            if cursor.fetchone()[0] == 0:
                faculty_id = str(uuid.uuid4())
                hashed_pass = self._hash_password('faculty123')
                cursor.execute('''
                    INSERT INTO users (id, name, username, password, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (faculty_id, 'Faculty Coordinator', 'faculty@raisoni', hashed_pass, 'faculty', datetime.now().isoformat()))
                logger.info("Default faculty user created")
            
            self.conn.commit()
        except Exception as e:
            st.error(f"Error adding default users: {e}")
            logger.error(f"Error adding default users: {e}")
    
    def _hash_password(self, password):
        """Hash password using bcrypt"""
        try:
            # For new passwords
            salt = bcrypt.gensalt()
            return bcrypt.hashpw(password.encode(), salt).decode()
        except:
            # Fallback for existing passwords
            return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, hashed):
        """Verify password against hash"""
        try:
            # Try bcrypt first
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except:
            # Fallback to SHA-256 for existing passwords
            return hashed == hashlib.sha256(password.encode()).hexdigest()
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials"""
        try:
            if role in ['admin', 'faculty']:
                creds = self.default_creds[role]
                if username == creds['username']:
                    return self.verify_password(password, self._hash_password(creds['password']))
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
                    return self.verify_password(password, stored_hash)
                return False
        except Exception as e:
            st.error(f"Login error: {e}")
            logger.error(f"Login error: {e}")
            return False
    
    def execute_query(self, query, params=(), fetch_one=False, fetch_all=False, commit=True):
        """Execute SQL query with error handling"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            
            if fetch_one:
                result = cursor.fetchone()
                return dict(result) if result else None
            elif fetch_all:
                results = cursor.fetchall()
                return [dict(row) for row in results]
            else:
                if commit:
                    self.conn.commit()
                return cursor.rowcount
        except Exception as e:
            st.error(f"Database error: {e}")
            logger.error(f"Database error in query '{query}': {e}")
            return None
    
    # ========== VALIDATION METHODS ==========
    
    def validate_user_data(self, user_data):
        """Validate user data before saving"""
        required_fields = ['name', 'username', 'password', 'email']
        for field in required_fields:
            if not user_data.get(field):
                return False, f"Missing required field: {field}"
        
        # Validate email format
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', user_data.get('email', '')):
            return False, "Invalid email format"
        
        # Validate username uniqueness
        existing = self.get_user(user_data['username'])
        if existing:
            return False, "Username already exists"
        
        return True, "Valid"
    
    def validate_event_data(self, event_data):
        """Validate event data before saving"""
        required_fields = ['title', 'description', 'event_date', 'venue', 'organizer']
        for field in required_fields:
            if not event_data.get(field):
                return False, f"Missing required field: {field}"
        
        # Validate date format
        try:
            if isinstance(event_data['event_date'], str):
                datetime.fromisoformat(event_data['event_date'].replace('Z', '+00:00'))
        except:
            return False, "Invalid date format"
        
        return True, "Valid"
    
    # ========== USERS ==========
    def add_user(self, user_data):
        """Add new user"""
        is_valid, message = self.validate_user_data(user_data)
        if not is_valid:
            st.error(f"Validation error: {message}")
            return None
        
        query = '''
            INSERT INTO users (id, name, roll_no, department, year, email, username, password, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            user_data.get('id', str(uuid.uuid4())),
            user_data.get('name'),
            user_data.get('roll_no'),
            user_data.get('department'),
            user_data.get('year'),
            user_data.get('email'),
            user_data.get('username'),
            self._hash_password(user_data.get('password')),
            user_data.get('role', 'student'),
            user_data.get('created_at', datetime.now().isoformat())
        )
        return self.execute_query(query, params)
    
    @lru_cache(maxsize=128)
    def get_user(self, username):
        """Get user by username (cached)"""
        query = "SELECT * FROM users WHERE username = ?"
        return self.execute_query(query, (username,), fetch_one=True)
    
    def get_all_users(self, limit=100):
        """Get all users with limit"""
        query = "SELECT * FROM users ORDER BY created_at DESC LIMIT ?"
        return self.execute_query(query, (limit,), fetch_all=True)
    
    # ========== EVENTS ==========
    def add_event(self, event_data):
        """Add new event"""
        is_valid, message = self.validate_event_data(event_data)
        if not is_valid:
            st.error(f"Validation error: {message}")
            return None
        
        query = '''
            INSERT INTO events (
                id, title, description, event_type, event_date, venue, organizer,
                registration_link, max_participants, flyer_path, created_by,
                created_by_name, ai_generated, ai_prompt, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            event_data.get('id', str(uuid.uuid4())),
            event_data.get('title'),
            event_data.get('description'),
            event_data.get('event_type'),
            event_data.get('event_date'),
            event_data.get('venue'),
            event_data.get('organizer'),
            event_data.get('registration_link', ''),
            event_data.get('max_participants', 100),
            event_data.get('flyer_path'),
            event_data.get('created_by'),
            event_data.get('created_by_name'),
            event_data.get('ai_generated', False),
            event_data.get('ai_prompt'),
            event_data.get('created_at', datetime.now().isoformat()),
            event_data.get('updated_at', datetime.now().isoformat())
        )
        return self.execute_query(query, params)
    
    @lru_cache(maxsize=128)
    def get_event(self, event_id):
        """Get event by ID (cached)"""
        query = "SELECT * FROM events WHERE id = ?"
        return self.execute_query(query, (event_id,), fetch_one=True)
    
    def get_all_events(self, limit=50):
        """Get all events with limit"""
        query = "SELECT * FROM events ORDER BY event_date DESC LIMIT ?"
        return self.execute_query(query, (limit,), fetch_all=True)
    
    def get_events_paginated(self, page=1, per_page=10):
        """Get events with pagination"""
        offset = (page - 1) * per_page
        query = "SELECT * FROM events ORDER BY event_date DESC LIMIT ? OFFSET ?"
        return self.execute_query(query, (per_page, offset), fetch_all=True)
    
    def get_events_by_creator(self, username, limit=50):
        """Get events created by specific user"""
        query = "SELECT * FROM events WHERE created_by = ? ORDER BY event_date DESC LIMIT ?"
        return self.execute_query(query, (username, limit), fetch_all=True)
    
    # ========== SOCIAL INTERACTIONS ==========
    
    def update_event_like(self, event_id, username, add=True):
        """Add or remove like efficiently"""
        try:
            if add:
                # Try to insert like
                query = '''
                    INSERT OR IGNORE INTO event_likes (event_id, username)
                    VALUES (?, ?)
                '''
                self.execute_query(query, (event_id, username), commit=False)
                # Update counter
                self.conn.execute(
                    "UPDATE events SET likes_count = likes_count + 1, updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), event_id)
                )
            else:
                # Remove like
                query = "DELETE FROM event_likes WHERE event_id = ? AND username = ?"
                self.execute_query(query, (event_id, username), commit=False)
                # Update counter
                self.conn.execute(
                    "UPDATE events SET likes_count = likes_count - 1, updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), event_id)
                )
            
            self.conn.commit()
            
            # Log the action
            action = 'like' if add else 'unlike'
            self.log_social_action(event_id, username, action)
            
            return True
        except Exception as e:
            self.conn.rollback()
            st.error(f"Error updating like: {e}")
            logger.error(f"Error updating like: {e}")
            return False
    
    def update_event_favorite(self, event_id, username, add=True):
        """Add or remove favorite efficiently"""
        try:
            if add:
                query = '''
                    INSERT OR IGNORE INTO event_favorites (event_id, username)
                    VALUES (?, ?)
                '''
                self.execute_query(query, (event_id, username), commit=False)
                self.conn.execute(
                    "UPDATE events SET favorites_count = favorites_count + 1, updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), event_id)
                )
            else:
                query = "DELETE FROM event_favorites WHERE event_id = ? AND username = ?"
                self.execute_query(query, (event_id, username), commit=False)
                self.conn.execute(
                    "UPDATE events SET favorites_count = favorites_count - 1, updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), event_id)
                )
            
            self.conn.commit()
            action = 'favorite' if add else 'unfavorite'
            self.log_social_action(event_id, username, action)
            return True
        except Exception as e:
            self.conn.rollback()
            st.error(f"Error updating favorite: {e}")
            logger.error(f"Error updating favorite: {e}")
            return False
    
    def update_event_interested(self, event_id, username, add=True):
        """Add or remove interested status efficiently"""
        try:
            if add:
                query = '''
                    INSERT OR IGNORE INTO event_interested (event_id, username)
                    VALUES (?, ?)
                '''
                self.execute_query(query, (event_id, username), commit=False)
                self.conn.execute(
                    "UPDATE events SET interested_count = interested_count + 1, updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), event_id)
                )
            else:
                query = "DELETE FROM event_interested WHERE event_id = ? AND username = ?"
                self.execute_query(query, (event_id, username), commit=False)
                self.conn.execute(
                    "UPDATE events SET interested_count = interested_count - 1, updated_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), event_id)
                )
            
            self.conn.commit()
            action = 'interested' if add else 'uninterested'
            self.log_social_action(event_id, username, action)
            return True
        except Exception as e:
            self.conn.rollback()
            st.error(f"Error updating interested: {e}")
            logger.error(f"Error updating interested: {e}")
            return False
    
    def increment_event_shares(self, event_id, username):
        """Increment shares count"""
        try:
            self.conn.execute(
                "UPDATE events SET shares_count = shares_count + 1, updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), event_id)
            )
            self.conn.commit()
            self.log_social_action(event_id, username, 'share')
            return True
        except Exception as e:
            st.error(f"Error incrementing shares: {e}")
            logger.error(f"Error incrementing shares: {e}")
            return False
    
    def increment_event_views(self, event_id, username):
        """Increment views count"""
        try:
            self.conn.execute(
                "UPDATE events SET views_count = views_count + 1, updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), event_id)
            )
            self.conn.commit()
            self.log_social_action(event_id, username, 'view')
            return True
        except Exception as e:
            st.error(f"Error incrementing views: {e}")
            logger.error(f"Error incrementing views: {e}")
            return False
    
    def get_user_likes(self, username):
        """Get events liked by user"""
        query = '''
            SELECT e.* FROM events e
            JOIN event_likes l ON e.id = l.event_id
            WHERE l.username = ?
            ORDER BY l.timestamp DESC
        '''
        return self.execute_query(query, (username,), fetch_all=True)
    
    def get_user_favorites(self, username):
        """Get events favorited by user"""
        query = '''
            SELECT e.* FROM events e
            JOIN event_favorites f ON e.id = f.event_id
            WHERE f.username = ?
            ORDER BY f.timestamp DESC
        '''
        return self.execute_query(query, (username,), fetch_all=True)
    
    def get_user_interested(self, username):
        """Get events user is interested in"""
        query = '''
            SELECT e.* FROM events e
            JOIN event_interested i ON e.id = i.event_id
            WHERE i.username = ?
            ORDER BY i.timestamp DESC
        '''
        return self.execute_query(query, (username,), fetch_all=True)
    
    def check_user_like(self, event_id, username):
        """Check if user liked event"""
        query = "SELECT 1 FROM event_likes WHERE event_id = ? AND username = ?"
        result = self.execute_query(query, (event_id, username), fetch_one=True)
        return result is not None
    
    def check_user_favorite(self, event_id, username):
        """Check if user favorited event"""
        query = "SELECT 1 FROM event_favorites WHERE event_id = ? AND username = ?"
        result = self.execute_query(query, (event_id, username), fetch_one=True)
        return result is not None
    
    def check_user_interested(self, event_id, username):
        """Check if user is interested in event"""
        query = "SELECT 1 FROM event_interested WHERE event_id = ? AND username = ?"
        result = self.execute_query(query, (event_id, username), fetch_one=True)
        return result is not None
    
    # ========== REGISTRATIONS ==========
    def add_registration(self, reg_data):
        """Add new registration"""
        query = '''
            INSERT INTO registrations (
                id, event_id, event_title, student_username, student_name,
                student_roll, student_dept, via_link, via_app, status,
                attendance, registered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            reg_data.get('id', str(uuid.uuid4())),
            reg_data.get('event_id'),
            reg_data.get('event_title'),
            reg_data.get('student_username'),
            reg_data.get('student_name'),
            reg_data.get('student_roll'),
            reg_data.get('student_dept'),
            reg_data.get('via_link', False),
            reg_data.get('via_app', True),
            reg_data.get('status', 'pending'),
            reg_data.get('attendance', 'absent'),
            reg_data.get('registered_at', datetime.now().isoformat())
        )
        return self.execute_query(query, params)
    
    def get_registrations_by_event(self, event_id, limit=100):
        """Get all registrations for an event"""
        query = "SELECT * FROM registrations WHERE event_id = ? ORDER BY registered_at DESC LIMIT ?"
        return self.execute_query(query, (event_id, limit), fetch_all=True)
    
    def get_registrations_by_student(self, username, limit=100):
        """Get all registrations for a student"""
        query = "SELECT * FROM registrations WHERE student_username = ? ORDER BY registered_at DESC LIMIT ?"
        return self.execute_query(query, (username, limit), fetch_all=True)
    
    def update_registration_status(self, reg_id, status, attendance):
        """Update registration status"""
        query = "UPDATE registrations SET status = ?, attendance = ? WHERE id = ?"
        return self.execute_query(query, (status, attendance, reg_id))
    
    # ========== SOCIAL LOGS ==========
    def log_social_action(self, event_id, username, action):
        """Log social interaction"""
        query = '''
            INSERT INTO social_logs (event_id, username, action, timestamp)
            VALUES (?, ?, ?, ?)
        '''
        return self.execute_query(query, (event_id, username, action, datetime.now().isoformat()))
    
    # ========== IMAGE HANDLING ==========
    def save_image_simple(self, uploaded_file):
        """Simple method to save image as base64 string - SAFE VERSION"""
        if uploaded_file is None:
            return None
        
        try:
            # Get file info first
            file_size = len(uploaded_file.getvalue())
            
            # Limit file size to prevent memory issues
            MAX_SIZE = 5 * 1024 * 1024  # 5MB
            if file_size > MAX_SIZE:
                st.warning(f"Image is too large ({file_size/1024/1024:.1f}MB). Max size is 5MB.")
                return None
            
            # Check file type
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            
            if file_ext not in allowed_extensions:
                st.warning(f"Unsupported file type: {file_ext}. Use: {', '.join(allowed_extensions)}")
                return None
            
            # Simple base64 conversion without PIL processing
            try:
                uploaded_file.seek(0)
                image_bytes = uploaded_file.getvalue()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                # Determine mime type
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
                st.error(f"Error converting image: {e}")
                logger.error(f"Error converting image: {e}")
                return None
                
        except Exception as e:
            st.error(f"Error saving image: {str(e)}")
            logger.error(f"Error saving image: {str(e)}")
            return None

# Initialize database manager
db_manager = EnhancedDatabaseManager()

# ============================================
# AI EVENT GENERATOR
# ============================================
class AIEventGenerator:
    """Generate events from WhatsApp/email messages"""
    
    def __init__(self):
        # Try to get OpenAI API key from secrets
        try:
            self.api_key = st.secrets.get("OPENAI_API_KEY", "")
        except:
            self.api_key = ""
    
    def extract_event_info(self, text):
        """Extract event info using AI or regex fallback"""
        
        # Try AI if API key is available
        if self.api_key:
            try:
                return self._extract_with_ai(text)
            except Exception as e:
                st.warning(f"AI extraction failed: {e}. Using regex fallback.")
                logger.warning(f"AI extraction failed: {e}")
        
        # Fallback to regex extraction
        return self._extract_with_regex(text)
    
    def _extract_with_ai(self, text):
        """Use OpenAI API to extract event info"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            prompt = f"""
            Extract event information from this message and return as valid JSON:
            
            {text}
            
            Return ONLY a JSON object with these exact field names (no other text):
            - title: Event title (string)
            - description: Event description (string)
            - event_type: workshop/hackathon/competition/bootcamp/seminar/conference/webinar (string)
            - event_date: YYYY-MM-DD format (string)
            - venue: Event location (string)
            - organizer: Who is organizing (string)
            - registration_link: URL if mentioned (string)
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You extract event information and return ONLY JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean response
            result = result.replace('```json', '').replace('```', '').strip()
            
            try:
                event_data = json.loads(result)
                event_data['ai_generated'] = True
                event_data['ai_prompt'] = text
                return event_data
            except json.JSONDecodeError:
                st.warning("AI returned invalid JSON. Using regex fallback.")
                return self._extract_with_regex(text)
                
        except Exception as e:
            st.warning(f"OpenAI API error: {e}. Using regex fallback.")
            logger.warning(f"OpenAI API error: {e}")
            return self._extract_with_regex(text)
    
    def _extract_with_regex(self, text):
        """Regex-based event extraction"""
        event_data = {
            'title': 'New Event',
            'description': text[:200] + '...' if len(text) > 200 else text,
            'event_type': 'workshop',
            'event_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'venue': 'G H Raisoni College',
            'organizer': 'College Department',
            'registration_link': '',
            'ai_generated': False
        }
        
        # Extract title (first line)
        lines = text.split('\n')
        if lines and lines[0].strip():
            event_data['title'] = lines[0].strip()[:100]
        
        # Extract date
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['event_date'] = match.group(1)
                break
        
        # Extract venue
        venue_keywords = ['at', 'venue', 'location', 'place']
        for keyword in venue_keywords:
            pattern = rf'{keyword}[:\s]*([^\n.,;]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['venue'] = match.group(1).strip()
                break
        
        # Extract organizer
        organizer_keywords = ['by', 'organizer', 'organized by', 'conducted by']
        for keyword in organizer_keywords:
            pattern = rf'{keyword}[:\s]*([^\n.,;]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['organizer'] = match.group(1).strip()
                break
        
        # Extract URL
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        if urls:
            event_data['registration_link'] = urls[0]
        
        return event_data

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

def is_upcoming(event_date):
    """Check if event is upcoming"""
    try:
        if isinstance(event_date, str):
            dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        else:
            dt = event_date
        return dt > datetime.now()
    except:
        return True

def safe_get(data, key, default=None):
    """Safely get value from dictionary"""
    return data.get(key, default) if data else default

# ============================================
# ENHANCED EVENT CARD WITH SOCIAL FEATURES
# ============================================
# ============================================
# ENHANCED EVENT CARD WITH SOCIAL FEATURES - FIXED VERSION
# ============================================
def display_event_card_social(event, current_user=None):
    unique_suffix = str(int(time.time() * 1000))  # Millisecond timestamp
    """Display event card with social features - FIXED VERSION"""
    event_id = event.get('id')
    
    # Generate a unique key for this card instance
    card_key = f"event_card_{event_id}_{int(time.time())}"
    
    # Use session state to track interactions for this specific card
    if f"{event_id}_liked" not in st.session_state:
        st.session_state[f"{event_id}_liked"] = db_manager.check_user_like(event_id, current_user) if current_user else False
    
    if f"{event_id}_favorited" not in st.session_state:
        st.session_state[f"{event_id}_favorited"] = db_manager.check_user_favorite(event_id, current_user) if current_user else False
    
    if f"{event_id}_interested" not in st.session_state:
        st.session_state[f"{event_id}_interested"] = db_manager.check_user_interested(event_id, current_user) if current_user else False
    
    if f"{event_id}_shares" not in st.session_state:
        st.session_state[f"{event_id}_shares"] = event.get('shares_count', 0) or 0
    
    if f"{event_id}_views" not in st.session_state:
        st.session_state[f"{event_id}_views"] = event.get('views_count', 0) or 0
    
    # Get counts
    likes_count = event.get('likes_count', 0) or 0
    favorites_count = event.get('favorites_count', 0) or 0
    interested_count = event.get('interested_count', 0) or 0
    shares_count = event.get('shares_count', 0) or 0
    views_count = event.get('views_count', 0) or 0
    
    # Use local variables that can be updated
    user_liked = st.session_state.get(f"{event_id}_liked", False)
    user_favorited = st.session_state.get(f"{event_id}_favorited", False)
    user_interested = st.session_state.get(f"{event_id}_interested", False)
    
    # Use a container with a unique key
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        # Header with AI badge
        col_header = st.columns([4, 1])
        with col_header[0]:
            st.subheader(event.get('title', 'Untitled Event'))
            if event.get('ai_generated'):
                st.markdown('<span class="ai-badge">🤖 AI Generated</span>', 
                           unsafe_allow_html=True)
        with col_header[1]:
            if is_upcoming(event.get('event_date')):
                st.success("🟢 Upcoming")
            else:
                st.error("🔴 Completed")
        
        # Event flyer
        if event.get('flyer_path'):
            flyer_path = event.get('flyer_path')
            try:
                # Check if it's a base64 data URL
                if flyer_path.startswith('data:image/'):
                    # Display base64 image
                    st.image(flyer_path, width=300, caption="Event Flyer")
                else:
                    # Try to load as file path
                    if os.path.exists(flyer_path):
                        image = Image.open(flyer_path)
                        st.image(image, width=300, caption="Event Flyer")
            except Exception as e:
                st.warning("Could not display event flyer")
        
        # Description
        desc = event.get('description', 'No description')
        if len(desc) > 300:
            desc = desc[:300] + "..."
        st.write(desc)
        
        # Details
        col_details = st.columns(4)
        with col_details[0]:
            st.caption(f"**📅 Date:** {format_date(event.get('event_date'))}")
        with col_details[1]:
            st.caption(f"**📍 Venue:** {event.get('venue', 'N/A')}")
        with col_details[2]:
            st.caption(f"**🏷️ Type:** {event.get('event_type', 'N/A')}")
        with col_details[3]:
            st.caption(f"**👨‍🏫 Organizer:** {event.get('organizer', 'N/A')}")
        
        # Social buttons (only for logged-in users)
        if current_user and current_user != "None":
            st.markdown("---")
            st.markdown("**Social Interactions**")
    
            col_social = st.columns(5)
    
            # Get current states from session state (these were updated earlier)
            user_liked = st.session_state.get(f"{event_id}_liked", False)
            user_favorited = st.session_state.get(f"{event_id}_favorited", False)
            user_interested = st.session_state.get(f"{event_id}_interested", False)
    
            # LIKE button - FIXED: Dynamic text based on current state
            with col_social[0]:
                # Determine button text based on current state
                # like_icon = "❤️" if user_liked else "🤍"
                # like_text = f"{like_icon} Unlike" if user_liked else f"{like_icon} Like"
        
                # Create button with appropriate text
                if st.button(f"{'❤️' if user_liked else '🤍'} Like", 
                     key=f"like_{event_id}_{current_user}_{unique_suffix}", 
                     use_container_width=True):
                    with st.spinner("Updating..."):
                        # Toggle like status
                        new_like_status = not user_liked
                        success = db_manager.update_event_like(
                            event_id, 
                            current_user, 
                            add=new_like_status
                        )
                
                        if success:
                            # Update session state
                            st.session_state[f"{event_id}_liked"] = new_like_status
                            # Force immediate UI update
                            st.rerun()
                        else:
                            st.error("Failed to update like")
        
                # Display count - update based on session state
                current_likes = likes_count
                if user_liked:
                    current_likes = likes_count
                st.caption(f"{current_likes} likes")
    
            # FAVORITE button - FIXED: Dynamic text based on current state
            with col_social[1]:
                # Determine button text based on current state
                fav_icon = "⭐" if user_favorited else "☆"
                fav_text = f"{fav_icon} Unfavorite" if user_favorited else f"{fav_icon} Favorite"
        
                if st.button(fav_text, key=f"fav_btn_{event_id}_{unique_suffix}", use_container_width=True):
                    with st.spinner("Updating..."):
                        # Toggle favorite status
                        new_fav_status = not user_favorited
                        success = db_manager.update_event_favorite(
                            event_id, 
                            current_user, 
                            add=new_fav_status
                        )
                
                        if success:
                            # Update session state
                            st.session_state[f"{event_id}_favorited"] = new_fav_status
                            # Force immediate UI update
                            st.rerun()
                        else:
                            st.error("Failed to update favorite")
        
                # Display count
                current_favs = favorites_count
                if user_favorited:
                    current_favs = favorites_count
                st.caption(f"{current_favs} favorites")
    
            # INTERESTED button - FIXED: Dynamic text based on current state
            with col_social[2]:
                # Determine button text based on current state
                int_icon = "✅" if user_interested else "🤔"
                int_text = f"{int_icon} Not Interested" if user_interested else f"{int_icon} Interested"
        
                if st.button(int_text, key=f"int_btn_{event_id}_{unique_suffix}", use_container_width=True):
                    with st.spinner("Updating..."):
                        # Toggle interested status
                        new_int_status = not user_interested
                        success = db_manager.update_event_interested(
                            event_id, 
                            current_user, 
                            add=new_int_status
                        )
                
                        if success:
                            # Update session state
                            st.session_state[f"{event_id}_interested"] = new_int_status
                            # Force immediate UI update
                            st.rerun()
                        else:
                            st.error("Failed to update interest")
        
                # Display count
                current_int = interested_count
                if user_interested:
                    current_int = interested_count
                st.caption(f"{current_int} interested")
    
            # SHARE button
            with col_social[3]:
                share_text = "📤 Share"
        
                if st.button(share_text, key=f"share_btn_{event_id}_{unique_suffix}", use_container_width=True):
                    with st.spinner("Sharing..."):
                        success = db_manager.increment_event_shares(event_id, current_user)
                
                        if success:
                            # Update session state
                            st.session_state[f"{event_id}_shares"] = shares_count + 1
                    
                            # Generate share text
                            share_content = f"Check out this event: {event['title']}"
                            if event.get('registration_link'):
                                share_content += f"\nRegister here: {event['registration_link']}"
                    
                            # Show success message
                            st.success(f"Event shared! Total shares: {st.session_state[f'{event_id}_shares']}")
                            # Show shareable text
                            st.code(share_content)
                    
                            # Force immediate UI update
                            st.rerun()
                        else:
                            st.error("Failed to share event")
        
                # Display count from session state
                display_shares = st.session_state.get(f"{event_id}_shares", shares_count)
                st.caption(f"{display_shares} shares")
    
            # VIEW button
            with col_social[4]:
                view_text = "👁️ View"
        
                if st.button(view_text, key=f"view_btn_{event_id}_{unique_suffix}", use_container_width=True):
                    with st.spinner("Recording view..."):
                        success = db_manager.increment_event_views(event_id, current_user)
                
                        if success:
                            # Update session state
                            st.session_state[f"{event_id}_views"] = views_count + 1
                    
                            st.success(f"View recorded! Total views: {st.session_state[f'{event_id}_views']}")
                    
                            # Force immediate UI update
                            st.rerun()
                        else:
                            st.error("Failed to record view")
        
                # Display count from session state
                display_views = st.session_state.get(f"{event_id}_views", views_count)
                st.caption(f"{display_views} views")
        
        else:
            # Show social stats without interactive buttons
            st.markdown("---")
            st.markdown("**Social Stats**")
            
            col_social = st.columns(5)
            with col_social[0]:
                st.caption(f"❤️ {likes_count} likes")
            with col_social[1]:
                st.caption(f"⭐ {favorites_count} favorites")
            with col_social[2]:
                st.caption(f"🤔 {interested_count} interested")
            with col_social[3]:
                st.caption(f"📤 {shares_count} shares")
            with col_social[4]:
                st.caption(f"👁️ {views_count} views")
        
        # Registration section - only show for logged-in users
        if current_user and current_user != "None":
            st.markdown("---")
            st.markdown('<div class="registration-section">', unsafe_allow_html=True)
            st.subheader("📝 Registration")
            
            # Check if user is registered
            registrations = db_manager.get_registrations_by_student(current_user)
            is_registered = any(r.get('event_id') == event_id for r in registrations) if registrations else False
            
            if is_registered:
                st.success("✅ You are registered for this event")
                
                # Show registration details
                registration = next((r for r in registrations if r.get('event_id') == event_id), None)
                
                if registration:
                    col_reg = st.columns(3)
                    with col_reg[0]:
                        st.info(f"Status: {registration.get('status', 'pending').title()}")
                    with col_reg[1]:
                        st.info(f"Via: {'Official Link' if registration.get('via_link') else 'App'}")
                    with col_reg[2]:
                        if registration.get('attendance') == 'present':
                            st.success("Attended ✅")
                        else:
                            st.warning("Not Attended")
            else:
                col_reg_actions = st.columns([2, 1])
                
                with col_reg_actions[0]:
                    if event.get('registration_link'):
                        st.markdown(f"[🔗 **Register via Official Link**]({event['registration_link']})", 
                                   unsafe_allow_html=True)
                        st.caption("Click the link above to register on the official platform")
                
                with col_reg_actions[1]:
                    reg_text = "✅ **I Have Registered**"
                    
                    if st.button(reg_text, key=f"reg_btn_{event_id}_{current_user}_{unique_suffix}", use_container_width=True, type="primary"):
                        with st.spinner("Recording registration..."):
                            # Create registration record
                            student = db_manager.get_user(current_user)
                            
                            reg_data = {
                                'id': str(uuid.uuid4()),
                                'event_id': event_id,
                                'event_title': event.get('title', 'Untitled Event'),
                                'student_username': current_user,
                                'student_name': student.get('name', current_user) if student else current_user,
                                'student_roll': student.get('roll_no', 'N/A') if student else 'N/A',
                                'student_dept': student.get('department', 'N/A') if student else 'N/A',
                                'via_link': False,
                                'via_app': True,
                                'status': 'pending',
                                'attendance': 'absent'
                            }
                            
                            if db_manager.add_registration(reg_data):
                                st.success("Registration recorded! Waiting for verification.")
                                st.rerun()
                            else:
                                st.error("Failed to record registration")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
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
        st.subheader("Admin Login")
        admin_user = st.text_input("Username", key="admin_user")
        admin_pass = st.text_input("Password", type="password", key="admin_pass")
        
        if st.button("Admin Login", use_container_width=True):
            with st.spinner("Verifying..."):
                if db_manager.verify_credentials(admin_user, admin_pass, 'admin'):
                    st.session_state.role = 'admin'
                    st.session_state.username = admin_user
                    st.session_state.name = "Administrator"
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with col2:
        st.subheader("Faculty Login")
        faculty_user = st.text_input("Username", key="faculty_user")
        faculty_pass = st.text_input("Password", type="password", key="faculty_pass")
        
        if st.button("Faculty Login", use_container_width=True):
            with st.spinner("Verifying..."):
                if db_manager.verify_credentials(faculty_user, faculty_pass, 'faculty'):
                    st.session_state.role = 'faculty'
                    st.session_state.username = faculty_user
                    st.session_state.name = "Faculty Coordinator"
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with col3:
        st.subheader("Student Portal")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            student_user = st.text_input("Username", key="student_user_login")
            student_pass = st.text_input("Password", type="password", key="student_pass_login")
            
            if st.button("Student Login", use_container_width=True):
                with st.spinner("Verifying..."):
                    if db_manager.verify_credentials(student_user, student_pass, 'student'):
                        # Get student info
                        student = db_manager.get_user(student_user)
                        
                        if student:
                            st.session_state.role = 'student'
                            st.session_state.username = student_user
                            st.session_state.name = student.get('name', student_user)
                            st.session_state.user_id = student.get('id')
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("User not found")
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("student_registration"):
                st.markdown("**Create Student Account**")
                
                name = st.text_input("Full Name *")
                roll_no = st.text_input("Roll Number *")
                department = st.selectbox("Department *", 
                                         ["CSE", "AIML", "ECE", "EEE", "MECH", "CIVIL", "IT", "DS"])
                year = st.selectbox("Year *", ["I", "II", "III", "IV"])
                email = st.text_input("Email *")
                username = st.text_input("Username *")
                password = st.text_input("Password *", type="password")
                confirm_pass = st.text_input("Confirm Password *", type="password")
                
                if st.form_submit_button("Register", use_container_width=True):
                    with st.spinner("Creating account..."):
                        if password != confirm_pass:
                            st.error("Passwords don't match")
                        elif not all([name, roll_no, email, username, password]):
                            st.error("Please fill all required fields")
                        else:
                            # Check if username exists
                            existing_user = db_manager.get_user(username)
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
                                    'role': 'student',
                                    'created_at': datetime.now().isoformat()
                                }
                                
                                if db_manager.add_user(user_data):
                                    st.success("Registration successful! Please login.")
                                    st.rerun()
                                else:
                                    st.error("Registration failed")

# ============================================
# AI EVENT CREATION
# ============================================
def ai_event_creation():
    """Create event using AI"""
    st.header("🤖 AI-Powered Event Creation")
    
    ai_gen = AIEventGenerator()
    
    tab1, tab2 = st.tabs(["From Text", "Upload File"])
    
    with tab1:
        st.subheader("Paste Event Details")
        event_text = st.text_area("Paste WhatsApp message, email, or event details:", 
                                 height=200,
                                 placeholder="""Example:
🎯 Hackathon Alert!
Join our AI Hackathon on Dec 20-21, 2024 at Seminar Hall.
Organized by CSE Department.
Register: https://forms.gle/example
Prizes: ₹50,000""")
        
        generate_button = st.button("Generate Event", use_container_width=True, key="generate_ai_event")
        
        # Store generated event data in session state
        if 'ai_generated_data' not in st.session_state:
            st.session_state.ai_generated_data = None
        
        if generate_button and event_text:
            with st.spinner("Extracting event details..."):
                event_data = ai_gen.extract_event_info(event_text)
                st.session_state.ai_generated_data = event_data
                st.session_state.ai_event_text = event_text
                st.rerun()
        
        # Show form if we have generated data
        if st.session_state.ai_generated_data:
            event_data = st.session_state.ai_generated_data
            event_text = st.session_state.get('ai_event_text', '')
            
            st.subheader("📋 Extracted Event")
            
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Title", value=event_data.get('title', ''), 
                             key="ai_title", disabled=True)
                st.text_area("Description", value=event_data.get('description', ''), 
                            height=100, key="ai_desc", disabled=True)
                st.text_input("Type", value=event_data.get('event_type', ''), 
                             key="ai_type", disabled=True)
            
            with col2:
                st.text_input("Date", value=event_data.get('event_date', ''), 
                             key="ai_date", disabled=True)
                st.text_input("Venue", value=event_data.get('venue', ''), 
                             key="ai_venue", disabled=True)
                st.text_input("Organizer", value=event_data.get('organizer', ''), 
                             key="ai_org", disabled=True)
                st.text_input("Registration Link", 
                             value=event_data.get('registration_link', ''), 
                             key="ai_link", disabled=True)
            
            # Allow editing
            st.subheader("✏️ Edit & Finalize")
            
            with st.form("finalize_event_form"):
                title = st.text_input("Event Title *", 
                                     value=event_data.get('title', ''))
                description = st.text_area("Description *", 
                                          value=event_data.get('description', ''),
                                          height=150)
                
                # Handle the case where event_type might not be in the list
                extracted_type = event_data.get('event_type', 'Workshop')
                event_type_options = ["Workshop", "Hackathon", "Competition", 
                                     "Bootcamp", "Seminar", "Conference", "Webinar"]
                
                # Safely get index
                try:
                    default_index = event_type_options.index(extracted_type)
                except ValueError:
                    default_index = 0  # Default to Workshop if not found
                
                event_type = st.selectbox("Event Type *", 
                                         event_type_options,
                                         index=default_index)
                
                col_date, col_time = st.columns(2)
                with col_date:
                    try:
                        # Try to parse the date
                        date_str = event_data.get('event_date', date.today().isoformat())
                        parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        event_date_input = st.date_input("Date *", value=parsed_date)
                    except:
                        event_date_input = st.date_input("Date *", min_value=date.today())
                with col_time:
                    event_time_input = st.time_input("Time *", value=datetime.now().time())
                
                venue = st.text_input("Venue *", value=event_data.get('venue', ''))
                organizer = st.text_input("Organizer *", value=event_data.get('organizer', ''))
                reg_link = st.text_input("Registration Link", 
                                        value=event_data.get('registration_link', ''))
                
                # Flyer upload
                st.subheader("📸 Event Flyer (Optional)")
                flyer = st.file_uploader("Upload flyer image", 
                                        type=['jpg', 'jpeg', 'png', 'gif', 'webp'],
                                        key="ai_flyer_uploader")
                
                submit_button = st.form_submit_button("Create Event", use_container_width=True)
                
                if submit_button:
                    if not all([title, description, venue, organizer]):
                        st.error("Please fill all required fields (*)")
                    else:
                        # Combine date and time
                        event_datetime = datetime.combine(event_date_input, event_time_input)
                        
                        # Save flyer
                        flyer_path = None
                        if flyer:
                            flyer_path = db_manager.save_image_simple(flyer)
                        
                        event_to_save = {
                            'id': str(uuid.uuid4()),
                            'title': title,
                            'description': description,
                            'event_type': event_type,
                            'event_date': event_datetime.isoformat(),
                            'venue': venue,
                            'organizer': organizer,
                            'registration_link': reg_link,
                            'flyer_path': flyer_path,
                            'created_by': st.session_state.username,
                            'created_by_name': st.session_state.name,
                            'ai_generated': event_data.get('ai_generated', False),
                            'ai_prompt': event_text if event_data.get('ai_generated') else None
                        }
                        
                        if db_manager.add_event(event_to_save):
                            st.success("Event created successfully! 🎉")
                            st.balloons()
                            # Clear session state
                            st.session_state.ai_generated_data = None
                            st.session_state.ai_event_text = None
                            st.rerun()
                        else:
                            st.error("Failed to save event")
    
    with tab2:
        st.subheader("Upload File")
        uploaded_file = st.file_uploader("Upload text file", type=['txt'], key="ai_file_upload")
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            st.text_area("File Content", content, height=200, key="ai_file_content")
            
            if st.button("Extract from File", use_container_width=True, key="extract_from_file"):
                with st.spinner("Extracting event details..."):
                    event_data = ai_gen.extract_event_info(content)
                    # Store in session state
                    st.session_state.ai_generated_data = event_data
                    st.session_state.ai_event_text = content
                    st.rerun()

# ============================================
# FACULTY DASHBOARD
# ============================================
def faculty_dashboard():
    """Faculty coordinator dashboard"""
    st.sidebar.title("👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('faculty')
    
    # Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Create Event", "AI Event Creator", "My Events", 
                    "Registrations", "Analytics"],
            icons=["house", "plus-circle", "robot", "calendar-event", "list-check", "graph-up"],
            default_index=0
        )
    
    if selected == "Dashboard":
        st.markdown('<h1 class="main-header">Faculty Dashboard</h1>', unsafe_allow_html=True)
        
        # Statistics
        events = db_manager.get_events_by_creator(st.session_state.username)
        all_registrations = db_manager.execute_query("SELECT * FROM registrations LIMIT 1000", fetch_all=True) or []
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("My Events", len(events))
        with col2:
            upcoming = len([e for e in events if is_upcoming(e.get('event_date'))])
            st.metric("Upcoming", upcoming)
        with col3:
            event_ids = [e['id'] for e in events]
            total_reg = len([r for r in all_registrations if r['event_id'] in event_ids])
            st.metric("Total Registrations", total_reg)
        with col4:
            attended = len([r for r in all_registrations if r['event_id'] in event_ids and r['attendance'] == 'present'])
            st.metric("Attended", attended)
        
        # Recent events
        st.subheader("📅 My Recent Events")
        if events:
            for event in events[-3:]:  # Last 3 events
                display_event_card_social(event, None)
        else:
            st.info("No events created yet. Create your first event!")
    
    elif selected == "Create Event":
        st.header("➕ Create New Event")
        
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
                registration_link = st.text_input("Registration Link")
                
                # Flyer upload
                st.subheader("Event Flyer (Optional)")
                flyer = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png', 'gif', 'webp'])
                if flyer:
                    st.image(flyer, width=200)
            
            description = st.text_area("Event Description *", height=150)
            
            submit_button = st.form_submit_button("Create Event", use_container_width=True)
            
            if submit_button:
                if not all([title, event_type, venue, organizer, description]):
                    st.error("Please fill all required fields (*)")
                else:
                    # Save flyer
                    flyer_path = None
                    if flyer:
                        flyer_path = db_manager.save_image_simple(flyer)
                    
                    # Combine date and time
                    event_datetime = datetime.combine(event_date, event_time)
                    
                    event_data = {
                        'id': str(uuid.uuid4()),
                        'title': title,
                        'description': description,
                        'event_type': event_type,
                        'event_date': event_datetime.isoformat(),
                        'venue': venue,
                        'organizer': organizer,
                        'registration_link': registration_link,
                        'max_participants': max_participants,
                        'flyer_path': flyer_path,
                        'created_by': st.session_state.username,
                        'created_by_name': st.session_state.name,
                        'ai_generated': False
                    }
                    
                    if db_manager.add_event(event_data):
                        st.success(f"Event '{title}' created successfully! 🎉")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Failed to create event")
    
    elif selected == "AI Event Creator":
        ai_event_creation()
    
    elif selected == "My Events":
        st.header("📋 My Events")
        
        events = db_manager.get_events_by_creator(st.session_state.username)
        
        if not events:
            st.info("You haven't created any events yet.")
            return
        
        # Filter tabs
        tab1, tab2 = st.tabs(["Upcoming Events", "Past Events"])
        
        with tab1:
            upcoming = [e for e in events if is_upcoming(e.get('event_date'))]
            if upcoming:
                for event in upcoming:
                    display_event_card_social(event, None)
            else:
                st.info("No upcoming events.")
        
        with tab2:
            past = [e for e in events if not is_upcoming(e.get('event_date'))]
            if past:
                for event in past:
                    display_event_card_social(event, None)
            else:
                st.info("No past events.")
    
    elif selected == "Registrations":
        st.header("📝 Event Registrations")
        
        events = db_manager.get_events_by_creator(st.session_state.username)
        
        if not events:
            st.info("You haven't created any events yet.")
            return
        
        # Select event
        event_titles = [e['title'] for e in events]
        selected_title = st.selectbox("Select Event", event_titles)
        
        if selected_title:
            selected_event = next(e for e in events if e['title'] == selected_title)
            event_id = selected_event['id']
            
            event_regs = db_manager.get_registrations_by_event(event_id)
            
            if event_regs:
                # Convert to DataFrame
                reg_data = []
                for reg in event_regs:
                    reg_data.append({
                        'Student Name': reg.get('student_name', 'N/A'),
                        'Roll No': reg.get('student_roll', 'N/A'),
                        'Department': reg.get('student_dept', 'N/A'),
                        'Registered Via': 'Official Link' if reg.get('via_link') else 'App',
                        'Status': reg.get('status', 'pending').title(),
                        'Attendance': reg.get('attendance', 'absent').title(),
                        'Registered On': format_date(reg.get('registered_at'))
                    })
                
                df = pd.DataFrame(reg_data)
                st.dataframe(df, use_container_width=True)
                
                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Registrations", len(event_regs))
                with col2:
                    via_link = len([r for r in event_regs if r.get('via_link')])
                    st.metric("Via Official Link", via_link)
                with col3:
                    via_app = len([r for r in event_regs if r.get('via_app')])
                    st.metric("Via App", via_app)
                
                # Export
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"registrations_{selected_title}.csv",
                    mime="text/csv"
                )
                
                # Update status
                st.subheader("Update Registration Status")
                selected_student = st.selectbox("Select Student", 
                                              [r['student_name'] for r in event_regs])
                
                if selected_student:
                    reg = next(r for r in event_regs if r['student_name'] == selected_student)
                    
                    col_status, col_att = st.columns(2)
                    with col_status:
                        new_status = st.selectbox("Registration Status", 
                                                 ["pending", "confirmed", "cancelled"],
                                                 index=["pending", "confirmed", "cancelled"]
                                                 .index(reg.get('status', 'pending')))
                    with col_att:
                        new_att = st.selectbox("Attendance", 
                                              ["absent", "present"],
                                              index=["absent", "present"]
                                              .index(reg.get('attendance', 'absent')))
                    
                    if st.button("Update Status"):
                        if db_manager.update_registration_status(reg['id'], new_status, new_att):
                            st.success("Status updated!")
                            st.rerun()
                        else:
                            st.error("Failed to update status")
            else:
                st.info(f"No registrations for '{selected_title}' yet.")
    
    elif selected == "Analytics":
        st.header("📊 Event Analytics")
        
        events = db_manager.get_events_by_creator(st.session_state.username)
        
        if not events:
            st.info("No events to analyze.")
            return
        
        # Overall statistics
        st.subheader("Overall Statistics")
        
        total_likes = sum(e.get('likes_count', 0) or 0 for e in events)
        total_favs = sum(e.get('favorites_count', 0) or 0 for e in events)
        total_int = sum(e.get('interested_count', 0) or 0 for e in events)
        total_views = sum(e.get('views_count', 0) or 0 for e in events)
        total_shares = sum(e.get('shares_count', 0) or 0 for e in events)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Likes", total_likes)
        with col2:
            st.metric("Total Favorites", total_favs)
        with col3:
            st.metric("Total Interested", total_int)
        with col4:
            st.metric("Total Views", total_views)
        with col5:
            st.metric("Total Shares", total_shares)
        
        # Event-wise analytics
        st.subheader("Event-wise Analytics")
        
        analytics_data = []
        for event in events:
            analytics_data.append({
                'Event': event['title'],
                'Likes': event.get('likes_count', 0) or 0,
                'Favorites': event.get('favorites_count', 0) or 0,
                'Interested': event.get('interested_count', 0) or 0,
                'Views': event.get('views_count', 0) or 0,
                'Shares': event.get('shares_count', 0) or 0,
                'Status': 'Upcoming' if is_upcoming(event.get('event_date')) else 'Past'
            })
        
        df = pd.DataFrame(analytics_data)
        st.dataframe(df, use_container_width=True)
        
        # Chart
        st.subheader("Engagement Chart")
        if len(df) > 0:
            chart_df = df.set_index('Event')[['Likes', 'Favorites', 'Interested']].head(5)
            st.bar_chart(chart_df)

# ============================================
# STUDENT DASHBOARD
# ============================================
def student_dashboard():
    """Student dashboard"""
    
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    
    # Get student info
    student = db_manager.get_user(st.session_state.username)
    
    if student:
        st.sidebar.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
        st.sidebar.markdown(f"**Department:** {student.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Year:** {student.get('year', 'N/A')}")
    
    display_role_badge('student')
    
    # Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Events Feed", "My Registrations", "My Interests", "Profile"],
            icons=["compass", "list-check", "heart", "person"],
            default_index=0
        )
    
    if selected == "Events Feed":
        st.markdown('<h1 class="main-header">🎯 Discover Events</h1>', unsafe_allow_html=True)
        
        # Filters
        col_filters = st.columns([2, 1, 1])
        with col_filters[0]:
            search = st.text_input("🔍 Search events", placeholder="Search...")
        with col_filters[1]:
            event_type = st.selectbox("Type", ["All", "Workshop", "Hackathon", "Competition", 
                                              "Bootcamp", "Seminar", "Conference"])
        with col_filters[2]:
            show_only = st.selectbox("Show", ["All", "Upcoming", "Past"])
        
        # Get events with pagination
        events = db_manager.get_all_events()
        
        # Apply filters
        filtered_events = events
        
        if search:
            filtered_events = [e for e in filtered_events 
                             if search.lower() in e.get('title', '').lower() or 
                             search.lower() in e.get('description', '').lower()]
        
        if event_type != "All":
            filtered_events = [e for e in filtered_events if e.get('event_type') == event_type]
        
        if show_only == "Upcoming":
            filtered_events = [e for e in filtered_events if is_upcoming(e.get('event_date'))]
        elif show_only == "Past":
            filtered_events = [e for e in filtered_events if not is_upcoming(e.get('event_date'))]
        
        # Sort by date
        filtered_events.sort(key=lambda x: x.get('event_date', ''), reverse=True)
        
        # Display events count
        st.caption(f"Found {len(filtered_events)} events")
        
        # Display events
        if filtered_events:
            for event in filtered_events:
                display_event_card_social(event, st.session_state.username)
        else:
            st.info("No events found matching your criteria.")
    
    elif selected == "My Registrations":
        st.header("📋 My Registrations")
        
        registrations = db_manager.get_registrations_by_student(st.session_state.username)
        
        if not registrations:
            st.info("You haven't registered for any events yet.")
            return
        
        # Get event details
        all_events = db_manager.get_all_events()
        event_map = {e['id']: e for e in all_events}
        
        # Tabs for different statuses
        tab1, tab2, tab3 = st.tabs(["Upcoming", "Completed", "All"])
        
        with tab1:
            upcoming_regs = []
            for reg in registrations:
                event = event_map.get(reg.get('event_id'))
                if event and is_upcoming(event.get('event_date')):
                    upcoming_regs.append((reg, event))
            
            if upcoming_regs:
                for reg, event in upcoming_regs:
                    with st.container():
                        st.markdown('<div class="event-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(event.get('title'))
                            st.caption(f"Date: {format_date(event.get('event_date'))}")
                            st.caption(f"Venue: {event.get('venue')}")
                        with col2:
                            st.info(f"Status: {reg.get('status', 'pending').title()}")
                            st.info(f"Via: {'Official Link' if reg.get('via_link') else 'App'}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No upcoming registered events.")
        
        with tab2:
            completed_regs = []
            for reg in registrations:
                event = event_map.get(reg.get('event_id'))
                if event and not is_upcoming(event.get('event_date')):
                    completed_regs.append((reg, event))
            
            if completed_regs:
                for reg, event in completed_regs:
                    with st.container():
                        st.markdown('<div class="event-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(event.get('title'))
                            st.caption(f"Date: {format_date(event.get('event_date'))}")
                            st.caption(f"Venue: {event.get('venue')}")
                        with col2:
                            status_color = "✅" if reg.get('attendance') == 'present' else "❌"
                            st.info(f"Attendance: {status_color}")
                            st.info(f"Status: {reg.get('status', 'pending').title()}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No completed events.")
        
        with tab3:
            for reg in registrations:
                event = event_map.get(reg.get('event_id'))
                if event:
                    with st.container():
                        st.markdown('<div class="event-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(event.get('title'))
                            st.caption(f"Date: {format_date(event.get('event_date'))}")
                            st.caption(f"Venue: {event.get('venue')}")
                        with col2:
                            if is_upcoming(event.get('event_date')):
                                st.success("🟢 Upcoming")
                            else:
                                st.error("🔴 Completed")
                            st.info(f"Status: {reg.get('status', 'pending').title()}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected == "My Interests":
        st.header("⭐ My Interests")
        
        # Get events user has interacted with using optimized queries
        liked_events = db_manager.get_user_likes(st.session_state.username) or []
        fav_events = db_manager.get_user_favorites(st.session_state.username) or []
        int_events = db_manager.get_user_interested(st.session_state.username) or []
        
        # Tabs
        tab1, tab2, tab3 = st.tabs([f"❤️ Liked ({len(liked_events)})", 
                                   f"⭐ Favorites ({len(fav_events)})", 
                                   f"🤔 Interested ({len(int_events)})"])
        
        with tab1:
            if liked_events:
                for event in liked_events:
                    display_event_card_social(event, st.session_state.username)
            else:
                st.info("You haven't liked any events yet.")
        
        with tab2:
            if fav_events:
                for event in fav_events:
                    display_event_card_social(event, st.session_state.username)
            else:
                st.info("You haven't favorited any events yet.")
        
        with tab3:
            if int_events:
                for event in int_events:
                    display_event_card_social(event, st.session_state.username)
            else:
                st.info("You haven't marked any events as interested.")
    
    elif selected == "Profile":
        st.header("👤 My Profile")
        
        student = db_manager.get_user(st.session_state.username)
        
        if student:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Full Name:** {student.get('name', 'N/A')}")
                st.markdown(f"**Roll Number:** {student.get('roll_no', 'N/A')}")
                st.markdown(f"**Department:** {student.get('department', 'N/A')}")
                st.markdown(f"**Year:** {student.get('year', 'N/A')}")
            
            with col2:
                st.markdown(f"**Email:** {student.get('email', 'N/A')}")
                st.markdown(f"**Username:** {student.get('username', 'N/A')}")
                st.markdown(f"**Member Since:** {format_date(student.get('created_at'))}")
            
            # Statistics
            st.markdown("---")
            st.subheader("📊 My Statistics")
            
            registrations = db_manager.get_registrations_by_student(st.session_state.username)
            my_regs = registrations if registrations else []
            
            liked_events = db_manager.get_user_likes(st.session_state.username) or []
            fav_events = db_manager.get_user_favorites(st.session_state.username) or []
            int_events = db_manager.get_user_interested(st.session_state.username) or []
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Events Registered", len(my_regs))
            with col_stat2:
                attended = len([r for r in my_regs if r.get('attendance') == 'present'])
                st.metric("Events Attended", attended)
            with col_stat3:
                st.metric("Events Liked", len(liked_events))
            with col_stat4:
                st.metric("Events Favorited", len(fav_events))

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
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    # Route based on login status
    if st.session_state.role is None:
        login_page()
    elif st.session_state.role == 'admin':
        # Admin dashboard
        st.sidebar.title("👑 Admin Panel")
        st.sidebar.markdown(f"**User:** {st.session_state.name}")
        display_role_badge('admin')
        
        st.markdown('<h1 class="main-header">Admin Dashboard</h1>', unsafe_allow_html=True)
        
        # Quick stats
        events = db_manager.get_all_events()
        users = db_manager.get_all_users()
        registrations = db_manager.execute_query("SELECT * FROM registrations LIMIT 1000", fetch_all=True) or []
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", len(events))
        with col2:
            st.metric("Total Users", len(users))
        with col3:
            st.metric("Total Registrations", len(registrations))
        with col4:
            upcoming = len([e for e in events if is_upcoming(e.get('event_date'))])
            st.metric("Upcoming Events", upcoming)
        
        # Database management
        st.markdown("---")
        st.subheader("Database Management")
        
        col_db1, col_db2, col_db3 = st.columns(3)
        
        with col_db1:
            if st.button("View All Users", use_container_width=True):
                users_df = pd.DataFrame(users)
                st.dataframe(users_df[['name', 'username', 'role', 'department', 'created_at']])
        
        with col_db2:
            if st.button("View All Events", use_container_width=True):
                events_df = pd.DataFrame(events)
                st.dataframe(events_df[['title', 'event_type', 'event_date', 'venue', 'created_by_name']])
        
        with col_db3:
            if st.button("System Health Check", use_container_width=True):
                with st.spinner("Checking system..."):
                    # Check database
                    cursor = db_manager.conn.cursor()
                    
                    tables = ['users', 'events', 'registrations', 'event_likes', 'event_favorites']
                    health_data = []
                    
                    for table in tables:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        health_data.append({
                            'Table': table,
                            'Records': count,
                            'Status': '✅ OK' if count >= 0 else '❌ Error'
                        })
                    
                    health_df = pd.DataFrame(health_data)
                    st.dataframe(health_df)
                    
                    # Check recent errors
                    st.subheader("Recent Logs")
                    st.info("Check server logs for detailed error information")
        
        # Logout button
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    elif st.session_state.role == 'faculty':
        faculty_dashboard()
        
        # Logout button
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    elif st.session_state.role == 'student':
        student_dashboard()
        
        # Logout button
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    main()
