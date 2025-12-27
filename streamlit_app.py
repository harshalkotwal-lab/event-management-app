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
# ENHANCED CUSTOM CSS
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
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .event-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        border-left: 4px solid #3B82F6;
    }
    
    .event-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
        border-color: #2563EB;
    }
    
    .role-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-left: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .admin-badge { background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%); color: #DC2626; }
    .faculty-badge { background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); color: #1D4ED8; }
    .student-badge { background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); color: #065F46; }
    
    .ai-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
    
    .social-btn {
        flex: 1;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        background: white;
        transition: all 0.2s;
        cursor: pointer;
        font-size: 0.9rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.25rem;
        min-height: 40px;
    }
    
    .social-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .social-btn.active {
        border-color: #3B82F6;
        background: #DBEAFE;
        color: #1E40AF;
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
    }
    
    .social-count {
        font-size: 0.75rem;
        color: #64748b;
        text-align: center;
        margin-top: 0.25rem;
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
    
    .flyer-container {
        border: 2px dashed #3B82F6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
    }
    
    .registration-section {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #3B82F6;
        font-size: 0.95rem;
    }
    
    .compact-details {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #475569;
    }
    
    .detail-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        background: #f8fafc;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
    }
    
    .status-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .status-upcoming { background: #D1FAE5; color: #065F46; }
    .status-ongoing { background: #FEF3C7; color: #92400E; }
    .status-past { background: #FEE2E2; color: #DC2626; }
    
    .action-button {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button {
        font-size: 0.95rem;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        transition: all 0.2s;
    }
    
    .stAlert {
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.95rem;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
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
</style>
""", unsafe_allow_html=True)

# ============================================
# ENHANCED SQLite DATABASE MANAGER - FIXED VERSION
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
            'admin': {'username': 'admin@raisoni', 'password': 'admin123', 'name': 'Administrator'},
            'faculty': {'username': 'faculty@raisoni', 'password': 'faculty123', 'name': 'Faculty Coordinator'}
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
                    roll_no TEXT UNIQUE,
                    department TEXT,
                    year TEXT,
                    email TEXT UNIQUE,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    current_participants INTEGER DEFAULT 0,
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
                    status TEXT DEFAULT 'upcoming',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Event likes table
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
                    FOREIGN KEY (student_username) REFERENCES users (username) ON DELETE CASCADE,
                    UNIQUE(event_id, student_username)
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
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_registrations_event ON registrations(event_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_registrations_student ON registrations(student_username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_likes_event ON event_likes(event_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_likes_user ON event_likes(username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_favorites_event ON event_favorites(event_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_favorites_user ON event_favorites(username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interested_event ON event_interested(event_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_interested_user ON event_interested(username)')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
            
            # Add default admin and faculty if not exists
            self._add_default_users()
            
        except Exception as e:
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
            logger.error(f"Error adding default users: {e}")
    
    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, hashed):
        """Verify password against hash"""
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
            logger.error(f"Database error in query '{query}': {e}")
            return None
    
    # ========== VALIDATION METHODS ==========
    
    def validate_user_data(self, user_data):
        """Validate user data before saving"""
        required_fields = ['name', 'username', 'password', 'email', 'roll_no']
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
        
        # Validate roll number uniqueness
        existing_roll = self.execute_query(
            "SELECT username FROM users WHERE roll_no = ?",
            (user_data['roll_no'],),
            fetch_one=True
        )
        if existing_roll:
            return False, "Roll number already exists"
        
        return True, "Valid"
    
    def validate_event_data(self, event_data):
        """Validate event data before saving"""
        required_fields = ['title', 'description', 'event_date', 'venue', 'organizer']
        for field in required_fields:
            if not event_data.get(field):
                return False, f"Missing required field: {field}"
        
        # Validate date
        try:
            if isinstance(event_data['event_date'], str):
                event_date = datetime.fromisoformat(event_data['event_date'].replace('Z', '+00:00'))
                if event_date < datetime.now():
                    return False, "Event date must be in the future"
        except:
            return False, "Invalid date format"
        
        return True, "Valid"
    
    # ========== USERS CRUD ==========
    def add_user(self, user_data):
        """Add new user"""
        is_valid, message = self.validate_user_data(user_data)
        if not is_valid:
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
    
    def update_user(self, username, user_data):
        """Update user information"""
        query = '''
            UPDATE users 
            SET name = ?, roll_no = ?, department = ?, year = ?, email = ?, updated_at = ?
            WHERE username = ?
        '''
        params = (
            user_data.get('name'),
            user_data.get('roll_no'),
            user_data.get('department'),
            user_data.get('year'),
            user_data.get('email'),
            datetime.now().isoformat(),
            username
        )
        return self.execute_query(query, params)
    
    def delete_user(self, username):
        """Delete user"""
        # First, delete user's registrations
        self.execute_query("DELETE FROM registrations WHERE student_username = ?", (username,))
        # Delete user
        return self.execute_query("DELETE FROM users WHERE username = ?", (username,))
    
    @lru_cache(maxsize=128)
    def get_user(self, username):
        """Get user by username (cached)"""
        query = "SELECT * FROM users WHERE username = ?"
        return self.execute_query(query, (username,), fetch_one=True)
    
    def get_all_users(self, limit=100):
        """Get all users with limit"""
        query = "SELECT * FROM users ORDER BY created_at DESC LIMIT ?"
        return self.execute_query(query, (limit,), fetch_all=True)
    
    def get_users_by_role(self, role):
        """Get users by role"""
        query = "SELECT * FROM users WHERE role = ? ORDER BY name"
        return self.execute_query(query, (role,), fetch_all=True)
    
    # ========== EVENTS CRUD ==========
    def add_event(self, event_data):
        """Add new event"""
        is_valid, message = self.validate_event_data(event_data)
        if not is_valid:
            return None
        
        # Determine event status based on date
        event_date = datetime.fromisoformat(event_data['event_date'].replace('Z', '+00:00'))
        status = 'upcoming' if event_date > datetime.now() else 'ongoing'
        
        query = '''
            INSERT INTO events (
                id, title, description, event_type, event_date, venue, organizer,
                registration_link, max_participants, flyer_path, created_by,
                created_by_name, ai_generated, ai_prompt, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            status,
            event_data.get('created_at', datetime.now().isoformat()),
            event_data.get('updated_at', datetime.now().isoformat())
        )
        return self.execute_query(query, params)
    
    def update_event(self, event_id, event_data):
        """Update event information"""
        # Determine event status based on date
        event_date = datetime.fromisoformat(event_data['event_date'].replace('Z', '+00:00'))
        status = 'upcoming' if event_date > datetime.now() else 'ongoing'
        
        query = '''
            UPDATE events 
            SET title = ?, description = ?, event_type = ?, event_date = ?, venue = ?,
                organizer = ?, registration_link = ?, max_participants = ?, flyer_path = ?,
                status = ?, updated_at = ?
            WHERE id = ?
        '''
        params = (
            event_data.get('title'),
            event_data.get('description'),
            event_data.get('event_type'),
            event_data.get('event_date'),
            event_data.get('venue'),
            event_data.get('organizer'),
            event_data.get('registration_link', ''),
            event_data.get('max_participants', 100),
            event_data.get('flyer_path'),
            status,
            datetime.now().isoformat(),
            event_id
        )
        return self.execute_query(query, params)
    
    def delete_event(self, event_id):
        """Delete event and all related data"""
        # Delete will cascade to related tables due to foreign key constraints
        return self.execute_query("DELETE FROM events WHERE id = ?", (event_id,))
    
    @lru_cache(maxsize=128)
    def get_event(self, event_id):
        """Get event by ID (cached)"""
        query = "SELECT * FROM events WHERE id = ?"
        return self.execute_query(query, (event_id,), fetch_one=True)
    
    def get_all_events(self, limit=100):
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
    
    def get_events_by_status(self, status):
        """Get events by status"""
        query = "SELECT * FROM events WHERE status = ? ORDER BY event_date"
        return self.execute_query(query, (status,), fetch_all=True)
    
    def update_event_status(self):
        """Update event status based on current time"""
        try:
            now = datetime.now().isoformat()
            # Update ongoing events that have passed
            self.execute_query(
                "UPDATE events SET status = 'past', updated_at = ? WHERE event_date <= ? AND status != 'past'",
                (now, now)
            )
            # Update upcoming events that are happening now
            self.execute_query(
                "UPDATE events SET status = 'ongoing', updated_at = ? WHERE event_date > ? AND status != 'ongoing'",
                (now, now)
            )
            return True
        except Exception as e:
            logger.error(f"Error updating event status: {e}")
            return False
    
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
            logger.error(f"Error incrementing views: {e}")
            return False
    
    def get_user_likes(self, username):
        """Get events liked by user - FIXED"""
        query = '''
            SELECT e.* FROM events e
            JOIN event_likes l ON e.id = l.event_id
            WHERE l.username = ?
            ORDER BY l.timestamp DESC
        '''
        return self.execute_query(query, (username,), fetch_all=True) or []
    
    def get_user_favorites(self, username):
        """Get events favorited by user - FIXED"""
        query = '''
            SELECT e.* FROM events e
            JOIN event_favorites f ON e.id = f.event_id
            WHERE f.username = ?
            ORDER BY f.timestamp DESC
        '''
        return self.execute_query(query, (username,), fetch_all=True) or []
    
    def get_user_interested(self, username):
        """Get events user is interested in - FIXED"""
        query = '''
            SELECT e.* FROM events e
            JOIN event_interested i ON e.id = i.event_id
            WHERE i.username = ?
            ORDER BY i.timestamp DESC
        '''
        return self.execute_query(query, (username,), fetch_all=True) or []
    
    def check_user_like(self, event_id, username):
        """Check if user liked event"""
        if not username:
            return False
        query = "SELECT 1 FROM event_likes WHERE event_id = ? AND username = ?"
        result = self.execute_query(query, (event_id, username), fetch_one=True)
        return result is not None
    
    def check_user_favorite(self, event_id, username):
        """Check if user favorited event"""
        if not username:
            return False
        query = "SELECT 1 FROM event_favorites WHERE event_id = ? AND username = ?"
        result = self.execute_query(query, (event_id, username), fetch_one=True)
        return result is not None
    
    def check_user_interested(self, event_id, username):
        """Check if user is interested in event"""
        if not username:
            return False
        query = "SELECT 1 FROM event_interested WHERE event_id = ? AND username = ?"
        result = self.execute_query(query, (event_id, username), fetch_one=True)
        return result is not None
    
    # ========== REGISTRATIONS ==========
    def add_registration(self, reg_data):
        """Add new registration - FIXED"""
        # Check if already registered
        existing = self.execute_query(
            "SELECT id FROM registrations WHERE event_id = ? AND student_username = ?",
            (reg_data['event_id'], reg_data['student_username']),
            fetch_one=True
        )
        
        if existing:
            return None  # Already registered
        
        # Get event details
        event = self.get_event(reg_data['event_id'])
        if not event:
            return None
        
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
            event.get('title', 'Untitled Event'),
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
        
        # Execute and return the registration ID
        if self.execute_query(query, params):
            # Update event participant count
            self.execute_query(
                "UPDATE events SET current_participants = current_participants + 1 WHERE id = ?",
                (reg_data['event_id'],)
            )
            return reg_data['id']
        return None
    
    def get_registrations_by_event(self, event_id, limit=100):
        """Get all registrations for an event"""
        query = "SELECT * FROM registrations WHERE event_id = ? ORDER BY registered_at DESC LIMIT ?"
        return self.execute_query(query, (event_id, limit), fetch_all=True) or []
    
    def get_registrations_by_student(self, username):
        """Get all registrations for a student - FIXED"""
        query = """
            SELECT r.*, e.event_date, e.venue, e.status as event_status 
            FROM registrations r
            LEFT JOIN events e ON r.event_id = e.id
            WHERE r.student_username = ?
            ORDER BY r.registered_at DESC
        """
        return self.execute_query(query, (username,), fetch_all=True) or []
    
    def update_registration_status(self, reg_id, status, attendance):
        """Update registration status"""
        query = "UPDATE registrations SET status = ?, attendance = ? WHERE id = ?"
        return self.execute_query(query, (status, attendance, reg_id))
    
    def delete_registration(self, reg_id):
        """Delete registration"""
        return self.execute_query("DELETE FROM registrations WHERE id = ?", (reg_id,))
    
    def is_student_registered(self, event_id, username):
        """Check if student is registered for event"""
        query = "SELECT id FROM registrations WHERE event_id = ? AND student_username = ?"
        result = self.execute_query(query, (event_id, username), fetch_one=True)
        return result is not None
    
    def get_registration_details(self, event_id, username):
        """Get specific registration details"""
        query = "SELECT * FROM registrations WHERE event_id = ? AND student_username = ?"
        return self.execute_query(query, (event_id, username), fetch_one=True)
    
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
                return None
            
            # Check file type
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            
            if file_ext not in allowed_extensions:
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
                logger.error(f"Error converting image: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return None

# Initialize database manager
db_manager = EnhancedDatabaseManager()

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

def get_event_status(event_date):
    """Get event status badge"""
    try:
        if isinstance(event_date, str):
            dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        else:
            dt = event_date
        
        if dt > datetime.now():
            return '<span class="status-badge status-upcoming">🟢 Upcoming</span>'
        elif dt.date() == datetime.now().date():
            return '<span class="status-badge status-ongoing">🟡 Ongoing</span>'
        else:
            return '<span class="status-badge status-past">🔴 Past</span>'
    except:
        return '<span class="status-badge">Unknown</span>'

# ============================================
# WORKING EVENT CARD WITH SOCIAL FEATURES
# ============================================
def display_event_card(event, current_user=None):
    """Display event card with WORKING social features"""
    if not event or not event.get('id'):
        return
    
    event_id = event.get('id')
    unique_key = f"{event_id}_{int(time.time() * 1000)}"
    
    # Check user interactions
    user_liked = False
    user_favorited = False
    user_interested = False
    
    if current_user:
        user_liked = db_manager.check_user_like(event_id, current_user)
        user_favorited = db_manager.check_user_favorite(event_id, current_user)
        user_interested = db_manager.check_user_interested(event_id, current_user)
    
    # Check registration
    is_registered = False
    if current_user:
        is_registered = db_manager.is_student_registered(event_id, current_user)
    
    # Get counts
    likes = event.get('likes_count', 0) or 0
    favorites = event.get('favorites_count', 0) or 0
    interested = event.get('interested_count', 0) or 0
    shares = event.get('shares_count', 0) or 0
    views = event.get('views_count', 0) or 0
    
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        # Header
        col_title, col_badge = st.columns([3, 1])
        with col_title:
            title = event.get('title', 'Untitled Event')
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
        with col_badge:
            if event.get('ai_generated'):
                st.markdown('<span class="ai-badge">🤖 AI Generated</span>', unsafe_allow_html=True)
        
        # Status and date
        event_date = event.get('event_date')
        st.markdown(get_event_status(event_date), unsafe_allow_html=True)
        
        # Details
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption(f"📅 {format_date(event_date)}")
        with col2:
            st.caption(f"📍 {event.get('venue', 'N/A')}")
        with col3:
            st.caption(f"🏷️ {event.get('event_type', 'N/A')}")
        with col4:
            st.caption(f"👨‍🏫 {event.get('organizer', 'N/A')}")
        
        # Description
        desc = event.get('description', '')
        if desc:
            if len(desc) > 150:
                with st.expander("📝 Description"):
                    st.write(desc)
                st.caption(f"{desc[:150]}...")
            else:
                st.caption(desc)
        
        # Flyer
        flyer = event.get('flyer_path')
        if flyer:
            try:
                if flyer.startswith('data:image'):
                    st.image(flyer, width=200, use_column_width=True)
            except:
                pass
        
        # SOCIAL INTERACTIONS - WORKING VERSION
        st.markdown('<div class="social-container">', unsafe_allow_html=True)
        
        # Like button
        col_like, col_fav, col_int, col_share, col_view = st.columns(5)
        
        with col_like:
            like_btn = st.button(
                f"{'❤️' if user_liked else '🤍'}",
                key=f"like_{unique_key}",
                help="Like this event",
                use_container_width=True
            )
            if like_btn and current_user:
                success = db_manager.update_event_like(
                    event_id, 
                    current_user, 
                    add=not user_liked
                )
                if success:
                    st.rerun()
            st.caption(f"{likes}")
        
        with col_fav:
            fav_btn = st.button(
                f"{'⭐' if user_favorited else '☆'}",
                key=f"fav_{unique_key}",
                help="Favorite this event",
                use_container_width=True
            )
            if fav_btn and current_user:
                success = db_manager.update_event_favorite(
                    event_id, 
                    current_user, 
                    add=not user_favorited
                )
                if success:
                    st.rerun()
            st.caption(f"{favorites}")
        
        with col_int:
            int_btn = st.button(
                f"{'✓' if user_interested else '?'}",
                key=f"int_{unique_key}",
                help="Mark as interested",
                use_container_width=True
            )
            if int_btn and current_user:
                success = db_manager.update_event_interested(
                    event_id, 
                    current_user, 
                    add=not user_interested
                )
                if success:
                    st.rerun()
            st.caption(f"{interested}")
        
        with col_share:
            share_btn = st.button(
                "📤",
                key=f"share_{unique_key}",
                help="Share this event",
                use_container_width=True
            )
            if share_btn and current_user:
                success = db_manager.increment_event_shares(event_id, current_user)
                if success:
                    st.success("Event shared!")
                    st.rerun()
            st.caption(f"{shares}")
        
        with col_view:
            view_btn = st.button(
                "👁️",
                key=f"view_{unique_key}",
                help="View event",
                use_container_width=True
            )
            if view_btn and current_user:
                db_manager.increment_event_views(event_id, current_user)
                st.rerun()
            st.caption(f"{views}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # REGISTRATION SECTION - WORKING
        if current_user:
            st.markdown('<div class="registration-section">', unsafe_allow_html=True)
            
            if is_registered:
                st.success("✅ You are registered for this event")
                
                # Get registration details
                reg = db_manager.get_registration_details(event_id, current_user)
                if reg:
                    col_status, col_via = st.columns(2)
                    with col_status:
                        st.info(f"**Status:** {reg.get('status', 'pending').title()}")
                    with col_via:
                        via = "Official Link" if reg.get('via_link') else "App"
                        st.info(f"**Via:** {via}")
            else:
                # Registration options
                reg_link = event.get('registration_link', '')
                
                if reg_link:
                    # Two registration methods available
                    col_link, col_app = st.columns(2)
                    
                    with col_link:
                        st.markdown(f"[🔗 **Register via Official Link**]({reg_link})", 
                                  unsafe_allow_html=True)
                        st.caption("Click to register on external platform")
                        
                        if st.button("✅ I've Registered via Link", 
                                   key=f"link_reg_{unique_key}",
                                   use_container_width=True):
                            record_registration(event_id, current_user, via_link=True)
                    
                    with col_app:
                        st.markdown("**Register via App**")
                        st.caption("Register directly in our system")
                        
                        if st.button("📱 Register via App", 
                                   key=f"app_reg_{unique_key}",
                                   use_container_width=True,
                                   type="primary"):
                            record_registration(event_id, current_user, via_link=False)
                else:
                    # Only app registration
                    st.markdown("**Register via App**")
                    st.caption("Register directly in our system")
                    
                    if st.button("📱 Register for Event", 
                               key=f"reg_{unique_key}",
                               use_container_width=True,
                               type="primary"):
                        record_registration(event_id, current_user, via_link=False)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Creator info
        created_by = event.get('created_by_name', 'Unknown')
        st.caption(f"Created by: {created_by}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def record_registration(event_id, username, via_link=False):
    """Helper function to record registration"""
    student = db_manager.get_user(username)
    
    if not student:
        st.error("Student information not found!")
        return
    
    reg_data = {
        'id': str(uuid.uuid4()),
        'event_id': event_id,
        'student_username': username,
        'student_name': student.get('name', username),
        'student_roll': student.get('roll_no', 'N/A'),
        'student_dept': student.get('department', 'N/A'),
        'via_link': via_link,
        'via_app': not via_link,
        'status': 'confirmed' if via_link else 'pending',
        'attendance': 'absent'
    }
    
    result = db_manager.add_registration(reg_data)
    if result is not None:
        st.success("✅ Registration recorded successfully!")
        st.rerun()
    else:
        st.error("Already registered or failed to record")

# ============================================
# LOGIN PAGE
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
            if db_manager.verify_credentials(admin_user, admin_pass, 'admin'):
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
            if db_manager.verify_credentials(faculty_user, faculty_pass, 'faculty'):
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
                st.markdown("### Create Student Account")
                
                name = st.text_input("Full Name *")
                roll_no = st.text_input("Roll Number *")
                department = st.selectbox("Department *", 
                                         ["CSE", "AIML", "IT", "EE", "BCA", "MCA", "BBA", "MBA", "EXTC", "MECH", "CIVIL", "DS", "Other"])
                year = st.selectbox("Year *", ["I", "II", "III", "IV"])
                email = st.text_input("Email *")
                username = st.text_input("Username *")
                password = st.text_input("Password *", type="password")
                confirm_pass = st.text_input("Confirm Password *", type="password")
                
                if st.form_submit_button("Register", use_container_width=True, type="primary"):
                    if password != confirm_pass:
                        st.error("Passwords don't match")
                    elif not all([name, roll_no, email, username, password]):
                        st.error("Please fill all required fields (*)")
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
                                st.success("✅ Registration successful! Please login.")
                                st.rerun()
                            else:
                                st.error("Registration failed")

# ============================================
# STUDENT DASHBOARD - WORKING VERSION
# ============================================
def student_dashboard():
    """Student dashboard - WORKING VERSION"""
    
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
        
        # Update event status
        db_manager.update_event_status()
        
        # Filters
        col_filters = st.columns([2, 1, 1])
        with col_filters[0]:
            search = st.text_input("🔍 Search events", placeholder="Search by title, description...")
        with col_filters[1]:
            event_type = st.selectbox("Type", ["All", "Workshop", "Hackathon", "Competition", 
                                              "Bootcamp", "Seminar", "Conference", "Webinar"])
        with col_filters[2]:
            show_only = st.selectbox("Show", ["All", "Upcoming", "Ongoing", "Past"])
        
        # Get events
        events = db_manager.get_all_events(limit=100)
        
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
        
        # Sort by date
        filtered_events.sort(key=lambda x: x.get('event_date', ''), reverse=True)
        
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
        
        registrations = db_manager.get_registrations_by_student(st.session_state.username)
        
        if not registrations:
            st.info("You haven't registered for any events yet.")
            # Show a link to events feed
            if st.button("Browse Events", use_container_width=True, type="primary"):
                st.rerun()
            return
        
        # Calculate statistics
        total = len(registrations)
        upcoming = len([r for r in registrations if r.get('event_status') == 'upcoming'])
        ongoing = len([r for r in registrations if r.get('event_status') == 'ongoing'])
        completed = len([r for r in registrations if r.get('event_status') == 'past'])
        
        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", total)
        with col2:
            st.metric("Upcoming", upcoming)
        with col3:
            st.metric("Ongoing", ongoing)
        with col4:
            st.metric("Completed", completed)
        
        # Create tabs
        tab_upcoming, tab_ongoing, tab_completed, tab_all = st.tabs([
            f"Upcoming ({upcoming})", 
            f"Ongoing ({ongoing})", 
            f"Completed ({completed})",
            f"All ({total})"
        ])
        
        with tab_upcoming:
            if upcoming > 0:
                for reg in registrations:
                    if reg.get('event_status') == 'upcoming':
                        with st.container():
                            st.markdown('<div class="event-card">', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([3, 1])
                            
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
                                reg_via = "Official Link" if reg.get('via_link') else "App"
                                st.caption(f"📝 Status: {reg_status} | Via: {reg_via}")
                            
                            with col2:
                                st.success("🟢 Upcoming")
                                
                                # Quick view
                                if st.button("View", key=f"view_up_{reg['id']}", use_container_width=True):
                                    # Get event and display
                                    event = db_manager.get_event(reg['event_id'])
                                    if event:
                                        display_event_card(event, st.session_state.username)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No upcoming registered events.")
        
        with tab_ongoing:
            if ongoing > 0:
                for reg in registrations:
                    if reg.get('event_status') == 'ongoing':
                        with st.container():
                            st.markdown('<div class="event-card">', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([3, 1])
                            
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
                                reg_via = "Official Link" if reg.get('via_link') else "App"
                                st.caption(f"📝 Status: {reg_status} | Via: {reg_via}")
                            
                            with col2:
                                st.warning("🟡 Ongoing")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No ongoing registered events.")
        
        with tab_completed:
            if completed > 0:
                for reg in registrations:
                    if reg.get('event_status') == 'past':
                        with st.container():
                            st.markdown('<div class="event-card">', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([3, 1])
                            
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
                                reg_via = "Official Link" if reg.get('via_link') else "App"
                                attendance = reg.get('attendance', 'absent')
                                attendance_text = "✅ Present" if attendance == 'present' else "❌ Absent"
                                st.caption(f"📝 Status: {reg_status} | Via: {reg_via} | {attendance_text}")
                            
                            with col2:
                                st.error("🔴 Completed")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No completed events.")
        
        with tab_all:
            for reg in registrations:
                with st.container():
                    st.markdown('<div class="event-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        event_title = reg.get('event_title', 'Unknown Event')
                        st.markdown(f'<div class="card-title">{event_title}</div>', unsafe_allow_html=True)
                        
                        # Event details
                        event_date = reg.get('event_date')
                        if event_date:
                            st.caption(f"📅 {format_date(event_date)}")
                        
                        venue = reg.get('venue', 'N/A')
                        st.caption(f"📍 {venue}")
                        
                        # Registration info
                        reg_status = reg.get('status', 'pending').title()
                        reg_via = "Official Link" if reg.get('via_link') else "App"
                        st.caption(f"📝 Status: {reg_status} | Via: {reg_via}")
                    
                    with col2:
                        # Event status
                        event_status = reg.get('event_status', 'unknown')
                        if event_status == 'upcoming':
                            st.success("🟢 Upcoming")
                        elif event_status == 'ongoing':
                            st.warning("🟡 Ongoing")
                        else:
                            st.error("🔴 Completed")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected == "My Interests":
        st.header("⭐ My Interests")
        
        # Get events user has interacted with
        liked_events = db_manager.get_user_likes(st.session_state.username)
        fav_events = db_manager.get_user_favorites(st.session_state.username)
        int_events = db_manager.get_user_interested(st.session_state.username)
        
        # Create tabs with counts
        tab_liked, tab_favorites, tab_interested = st.tabs([
            f"❤️ Liked ({len(liked_events)})",
            f"⭐ Favorites ({len(fav_events)})",
            f"🤔 Interested ({len(int_events)})"
        ])
        
        with tab_liked:
            if liked_events:
                for event in liked_events:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("You haven't liked any events yet.")
                st.caption("Like events by clicking the ❤️ button on event cards")
        
        with tab_favorites:
            if fav_events:
                for event in fav_events:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("You haven't favorited any events yet.")
                st.caption("Favorite events by clicking the ⭐ button on event cards")
        
        with tab_interested:
            if int_events:
                for event in int_events:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("You haven't marked any events as interested.")
                st.caption("Mark interest by clicking the ? button on event cards")
    
    elif selected == "Profile":
        st.header("👤 My Profile")
        
        student = db_manager.get_user(st.session_state.username)
        
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
            st.markdown("### Account Information")
            st.markdown(f"**Email:** {student.get('email', 'N/A')}")
            st.markdown(f"**Username:** {student.get('username', 'N/A')}")
            st.markdown(f"**Member Since:** {format_date(student.get('created_at'))}")
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 My Statistics")
        
        # Get actual data
        registrations = db_manager.get_registrations_by_student(st.session_state.username) or []
        liked_events = db_manager.get_user_likes(st.session_state.username)
        fav_events = db_manager.get_user_favorites(st.session_state.username)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Events Registered", len(registrations))
        with col_stat2:
            attended = len([r for r in registrations if r.get('attendance') == 'present'])
            st.metric("Events Attended", attended)
        with col_stat3:
            st.metric("Events Liked", len(liked_events))
        with col_stat4:
            st.metric("Events Favorited", len(fav_events))
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True, type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

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
                return self._extract_with_regex(text)
                
        except Exception as e:
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
        venue_keywords = ['at', 'venue', 'location', 'place', 'hall']
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
# FACULTY DASHBOARD
# ============================================
# ============================================
# FACULTY DASHBOARD - FIXED VERSION
# ============================================
def faculty_dashboard():
    """Faculty coordinator dashboard"""
    st.sidebar.title("👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('faculty')
    
    # Navigation - FIXED: Use simpler navigation to avoid option_menu issues
    with st.sidebar:
        st.markdown("### Navigation")
        
        # Create buttons for navigation
        if st.button("📊 Dashboard", use_container_width=True, key="faculty_dash"):
            st.session_state.faculty_page = "Dashboard"
            st.rerun()
        
        if st.button("➕ Create Event", use_container_width=True, key="faculty_create"):
            st.session_state.faculty_page = "Create Event"
            st.rerun()
        
        if st.button("🤖 AI Event Creator", use_container_width=True, key="faculty_ai"):
            st.session_state.faculty_page = "AI Event Creator"
            st.rerun()
        
        if st.button("📋 My Events", use_container_width=True, key="faculty_events"):
            st.session_state.faculty_page = "My Events"
            st.rerun()
        
        if st.button("📝 Registrations", use_container_width=True, key="faculty_reg"):
            st.session_state.faculty_page = "Registrations"
            st.rerun()
        
        if st.button("📊 Analytics", use_container_width=True, key="faculty_analytics"):
            st.session_state.faculty_page = "Analytics"
            st.rerun()
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Initialize page state
    if 'faculty_page' not in st.session_state:
        st.session_state.faculty_page = "Dashboard"
    
    selected = st.session_state.faculty_page
    
    # Page routing
    if selected == "Dashboard":
        st.markdown('<h1 class="main-header">Faculty Dashboard</h1>', unsafe_allow_html=True)
        
        # Statistics
        events = db_manager.get_events_by_creator(st.session_state.username)
        all_registrations = db_manager.execute_query("SELECT * FROM registrations", fetch_all=True) or []
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("My Events", len(events))
        with col2:
            upcoming = len([e for e in events if e.get('status') == 'upcoming'])
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
            for event in events[-3:]:
                display_event_card(event, None)
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
                flyer = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png', 'gif', 'webp'], key="faculty_flyer")
                if flyer:
                    st.image(flyer, width=200)
            
            description = st.text_area("Event Description *", height=150)
            
            submit_button = st.form_submit_button("Create Event", use_container_width=True, type="primary")
            
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
                        st.rerun()
                    else:
                        st.error("Failed to create event")
    
    elif selected == "AI Event Creator":
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
            
            generate_button = st.button("Generate Event", use_container_width=True, key="generate_ai_event", type="primary")
            
            # Store generated event data in session state
            if 'ai_generated_data' not in st.session_state:
                st.session_state.ai_generated_data = None
            
            if generate_button and event_text:
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
                        default_index = 0
                    
                    event_type = st.selectbox("Event Type *", 
                                             event_type_options,
                                             index=default_index)
                    
                    col_date, col_time = st.columns(2)
                    with col_date:
                        try:
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
                    
                    submit_button = st.form_submit_button("Create Event", use_container_width=True, type="primary")
                    
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
                
                if st.button("Extract from File", use_container_width=True, key="extract_from_file", type="primary"):
                    event_data = ai_gen.extract_event_info(content)
                    # Store in session state
                    st.session_state.ai_generated_data = event_data
                    st.session_state.ai_event_text = content
                    st.rerun()
    
    elif selected == "My Events":
        st.header("📋 My Events")
        
        events = db_manager.get_events_by_creator(st.session_state.username)
        
        if not events:
            st.info("You haven't created any events yet.")
            return
        
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
                    mime="text/csv",
                    use_container_width=True
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
                    
                    if st.button("Update Status", use_container_width=True, type="primary"):
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
                'Status': event.get('status', 'unknown').title()
            })
        
        df = pd.DataFrame(analytics_data)
        st.dataframe(df, use_container_width=True)
        
        # Export option
        csv = export_events_to_csv(events)
        if csv:
            st.download_button(
                label="📥 Export Events Data",
                data=csv,
                file_name="faculty_events_analytics.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Chart
        if len(events) > 0:
            st.subheader("Engagement Chart")
            chart_df = pd.DataFrame(analytics_data)
            if not chart_df.empty:
                chart_df = chart_df.set_index('Event')[['Likes', 'Favorites', 'Interested']].head(5)
                st.bar_chart(chart_df)

# ============================================
# ADMIN DASHBOARD
# ============================================
def admin_dashboard():
    """Admin dashboard with full CRUD operations"""
    st.sidebar.title("👑 Admin Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('admin')
    
    # Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Manage Events", "Manage Users", "System Analytics"],
            icons=["house", "calendar-event", "people", "graph-up"],
            default_index=0
        )
    
    if selected == "Dashboard":
        st.markdown('<h1 class="main-header">Admin Dashboard</h1>', unsafe_allow_html=True)
        
        # Update event status
        db_manager.update_event_status()
        
        # Quick stats
        events = db_manager.get_all_events(limit=1000)
        users = db_manager.get_all_users(limit=1000)
        registrations = db_manager.execute_query("SELECT * FROM registrations", fetch_all=True) or []
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", len(events))
        with col2:
            st.metric("Total Users", len(users))
        with col3:
            st.metric("Total Registrations", len(registrations))
        with col4:
            upcoming = len([e for e in events if e.get('status') == 'upcoming'])
            st.metric("Upcoming Events", upcoming)
        
        # Recent events
        st.subheader("📅 Recent Events")
        if events:
            for event in events[:5]:
                display_event_card(event, None)
        else:
            st.info("No events found.")
    
    elif selected == "Manage Events":
        st.header("📋 Manage Events")
        
        tab1, tab2 = st.tabs(["View All Events", "Create Event"])
        
        with tab1:
            events = db_manager.get_all_events()
            
            if events:
                for event in events:
                    with st.container():
                        col_view, col_actions = st.columns([3, 1])
                        
                        with col_view:
                            display_event_card(event, None)
                        
                        with col_actions:
                            st.markdown("### Actions")
                            if st.button("Delete", key=f"delete_{event['id']}", use_container_width=True, type="secondary"):
                                if db_manager.delete_event(event['id']):
                                    st.success("Event deleted successfully!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete event")
            else:
                st.info("No events found.")
        
        with tab2:
            st.subheader("➕ Create New Event")
            
            with st.form("admin_create_event"):
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
                
                submit_button = st.form_submit_button("Create Event", use_container_width=True, type="primary")
                
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
                            st.rerun()
                        else:
                            st.error("Failed to create event")
    
    elif selected == "Manage Users":
        st.header("👥 Manage Users")
        
        users = db_manager.get_all_users()
        
        if users:
            # Display users in a table
            user_data = []
            for user in users:
                user_data.append({
                    'Name': user.get('name'),
                    'Username': user.get('username'),
                    'Role': user.get('role'),
                    'Roll No': user.get('roll_no', 'N/A'),
                    'Department': user.get('department', 'N/A'),
                    'Year': user.get('year', 'N/A'),
                    'Joined': format_date(user.get('created_at'))
                })
            
            df = pd.DataFrame(user_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No users found.")
    
    elif selected == "System Analytics":
        st.header("📊 System Analytics")
        
        # Get data
        events = db_manager.get_all_events(limit=1000)
        users = db_manager.get_all_users(limit=1000)
        registrations = db_manager.execute_query("SELECT * FROM registrations", fetch_all=True) or []
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", len(events))
        with col2:
            st.metric("Total Users", len(users))
        with col3:
            st.metric("Total Registrations", len(registrations))
        with col4:
            active_events = len([e for e in events if e.get('status') in ['upcoming', 'ongoing']])
            st.metric("Active Events", active_events)
        
        # User distribution
        st.subheader("👥 User Distribution")
        user_roles = {}
        for user in users:
            role = user.get('role', 'unknown')
            user_roles[role] = user_roles.get(role, 0) + 1
        
        role_df = pd.DataFrame({
            'Role': list(user_roles.keys()),
            'Count': list(user_roles.values())
        })
        st.bar_chart(role_df.set_index('Role'))
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", type="secondary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

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
    
    # Update event status
    db_manager.update_event_status()
    
    # Route based on login status
    if st.session_state.role is None:
        login_page()
    elif st.session_state.role == 'admin':
        admin_dashboard()
    elif st.session_state.role == 'faculty':
        faculty_dashboard()
    elif st.session_state.role == 'student':
        student_dashboard()

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    main()
