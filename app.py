"""
G H Raisoni College of Engineering and Management - Complete Event Management System
Fixed database issues, improved UI, full CRUD operations for events & students
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="G H Raisoni Event Manager",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS - ENHANCED UI
# ============================================
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem !important;
    color: #1E3A8A !important;
    text-align: center !important;
    padding: 1rem !important;
    margin-bottom: 1.5rem !important;
    font-weight: 700 !important;
}
.college-header {
    background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%) !important;
    padding: 1rem !important;
    border-radius: 12px !important;
    color: white !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}
.event-card {
    border: 2px solid #E5E7EB !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}
.event-card:hover {
    transform: translateY(-8px) !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.15) !important;
    border-color: #3B82F6 !important;
}
.event-card::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 4px !important;
    height: 100% !important;
    background: linear-gradient(to bottom, #3B82F6, #1E3A8A) !important;
}
.role-badge {
    display: inline-block !important;
    padding: 0.3rem 0.8rem !important;
    border-radius: 20px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    margin-left: 0.5rem !important;
}
.admin-badge { background: #FEE2E2 !important; color: #DC2626 !important; }
.faculty-badge { background: #DBEAFE !important; color: #1D4ED8 !important; }
.student-badge { background: #D1FAE5 !important; color: #065F46 !important; }
.ai-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    padding: 0.3rem 0.8rem !important;
    border-radius: 20px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    display: inline-block !important;
    margin-left: 0.5rem !important;
}
.social-buttons {
    display: flex !important;
    gap: 0.5rem !important;
    margin-top: 1rem !important;
    flex-wrap: wrap !important;
}
.social-btn {
    flex: 1 !important;
    min-width: 70px !important;
    text-align: center !important;
    padding: 0.5rem !important;
    border-radius: 10px !important;
    border: 2px solid #E5E7EB !important;
    background: white !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
.social-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}
.social-btn.active {
    border-color: #3B82F6 !important;
    background: #DBEAFE !important;
    color: #1E40AF !important;
    transform: translateY(-2px) !important;
}
.metric-card {
    background: white !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08) !important;
    border: 1px solid #E5E7EB !important;
    text-align: center !important;
    transition: transform 0.3s ease !important;
}
.metric-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12) !important;
}
.metric-value {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #1E3A8A !important;
    margin: 0.5rem 0 !important;
}
.metric-label {
    color: #6B7280 !important;
    font-size: 0.9rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
.flyer-container {
    border: 2px dashed #3B82F6 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    text-align: center !important;
    margin: 1rem 0 !important;
    background: #F8FAFC !important;
}
.registration-section {
    background: #F0F9FF !important;
    padding: 1rem !important;
    border-radius: 10px !important;
    margin-top: 1rem !important;
    border-left: 4px solid #3B82F6 !important;
    font-size: 0.95rem !important;
}
.compact-details {
    font-size: 0.85rem !important;
    color: #666 !important;
    margin: 0.3rem 0 !important;
}
.stButton > button {
    font-size: 0.95rem !important;
    padding: 0.4rem 1rem !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# COMPLETE DATABASE MANAGER WITH FULL CRUD
# ============================================
class CompleteEventDatabaseManager:
    """Complete database manager with full CRUD operations"""
    
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
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info("Database connected successfully")
        except Exception as e:
            st.error(f"Database connection error: {e}")
            logger.error(f"Database connection error: {e}")
    
    def create_tables(self):
        """Create all necessary tables with proper schema"""
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
                    FOREIGN KEY (student_username) REFERENCES users (username) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes
            indexes = [
                'idx_events_date ON events(event_date)',
                'idx_events_creator ON events(created_by)',
                'idx_registrations_event ON registrations(event_id)',
                'idx_registrations_student ON registrations(student_username)',
                'idx_likes_event ON event_likes(event_id)',
                'idx_likes_user ON event_likes(username)'
            ]
            
            for idx in indexes:
                cursor.execute(f'CREATE INDEX IF NOT EXISTS {idx}')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
            self._add_default_users()
            
        except Exception as e:
            st.error(f"Error creating tables: {e}")
            logger.error(f"Error creating tables: {e}")
            traceback.print_exc()
    
    def _add_default_users(self):
        """Add default admin and faculty users"""
        try:
            cursor = self.conn.cursor()
            
            # Admin
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ('admin@raisoni',))
            if cursor.fetchone()[0] == 0:
                admin_id = str(uuid.uuid4())
                hashed_pass = self._hash_password('admin123')
                cursor.execute('''
                    INSERT INTO users (id, name, username, password, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (admin_id, 'Administrator', 'admin@raisoni', hashed_pass, 'admin', datetime.now().isoformat()))
                logger.info("Default admin user created")
            
            # Faculty
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
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password, hashed):
        return hashed == hashlib.sha256(password.encode()).hexdigest()
    
    def verify_credentials(self, username, password, role):
        try:
            if role in ['admin', 'faculty']:
                creds = self.default_creds[role]
                if username == creds['username']:
                    return self.verify_password(password, self._hash_password(creds['password']))
                return False
            else:
                cursor = self.conn.cursor()
                cursor.execute("SELECT password FROM users WHERE username = ? AND role = 'student'", (username,))
                result = cursor.fetchone()
                if result:
                    return self.verify_password(password, result[0])
                return False
        except Exception as e:
            st.error(f"Login error: {e}")
            return False
    
    def execute_query(self, query, params=(), fetch_one=False, fetch_all=False, commit=True):
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
            logger.error(f"Database error: {e}")
            return None
    
    # ========== FULL CRUD OPERATIONS ==========
    def add_user(self, user_data):
        """CREATE: Add new student user"""
        query = '''
            INSERT INTO users (id, name, roll_no, department, year, email, username, password, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            str(uuid.uuid4()),
            user_data['name'],
            user_data.get('roll_no'),
            user_data.get('department'),
            user_data.get('year'),
            user_data.get('email'),
            user_data['username'],
            self._hash_password(user_data['password']),
            'student',
            datetime.now().isoformat()
        )
        return self.execute_query(query, params)
    
    def get_user(self, username):
        query = "SELECT * FROM users WHERE username = ?"
        return self.execute_query(query, (username,), fetch_one=True)
    
    def update_user(self, username, user_data):
        """UPDATE: Update user data"""
        query = '''
            UPDATE users SET name = ?, roll_no = ?, department = ?, year = ?, email = ?
            WHERE username = ?
        '''
        params = (
            user_data.get('name'),
            user_data.get('roll_no'),
            user_data.get('department'),
            user_data.get('year'),
            user_data.get('email'),
            username
        )
        return self.execute_query(query, params)
    
    def delete_user(self, username):
        """DELETE: Delete user"""
        query = "DELETE FROM users WHERE username = ?"
        return self.execute_query(query, (username,))
    
    def get_all_users(self):
        query = "SELECT * FROM users ORDER BY created_at DESC"
        return self.execute_query(query, fetch_all=True)
    
    def add_event(self, event_data):
        """CREATE: Add new event"""
        query = '''
            INSERT INTO events (
                id, title, description, event_type, event_date, venue, organizer,
                registration_link, max_participants, flyer_path, created_by,
                created_by_name, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            str(uuid.uuid4()),
            event_data['title'],
            event_data.get('description'),
            event_data.get('event_type'),
            event_data['event_date'],
            event_data['venue'],
            event_data['organizer'],
            event_data.get('registration_link'),
            event_data.get('max_participants', 100),
            event_data.get('flyer_path'),
            st.session_state.get('username'),
            st.session_state.get('name'),
            datetime.now().isoformat(),
            datetime.now().isoformat()
        )
        return self.execute_query(query, params)
    
    def get_event(self, event_id):
        query = "SELECT * FROM events WHERE id = ?"
        return self.execute_query(query, (event_id,), fetch_one=True)
    
    def update_event(self, event_id, event_data):
        """UPDATE: Update event"""
        query = '''
            UPDATE events SET 
                title = ?, description = ?, event_type = ?, event_date = ?,
                venue = ?, organizer = ?, registration_link = ?, max_participants = ?,
                flyer_path = ?, updated_at = ?
            WHERE id = ?
        '''
        params = (
            event_data.get('title'),
            event_data.get('description'),
            event_data.get('event_type'),
            event_data['event_date'],
            event_data['venue'],
            event_data.get('organizer'),
            event_data.get('registration_link'),
            event_data.get('max_participants', 100),
            event_data.get('flyer_path'),
            datetime.now().isoformat(),
            event_id
        )
        return self.execute_query(query, params)
    
    def delete_event(self, event_id):
        """DELETE: Delete event (cascades to related tables)"""
        query = "DELETE FROM events WHERE id = ?"
        return self.execute_query(query, (event_id,))
    
    def get_all_events(self):
        query = "SELECT * FROM events ORDER BY event_date DESC"
        return self.execute_query(query, fetch_all=True)
    
    def get_events_by_creator(self, username):
        query = "SELECT * FROM events WHERE created_by = ? ORDER BY event_date DESC"
        return self.execute_query(query, (username,), fetch_all=True)
    
    # Social interactions (simplified)
    def toggle_like(self, event_id, username):
        if self.is_liked(event_id, username):
            self.remove_like(event_id, username)
            self.conn.execute("UPDATE events SET likes_count = likes_count - 1 WHERE id = ?", (event_id,))
        else:
            self.conn.execute("INSERT OR IGNORE INTO event_likes (event_id, username) VALUES (?, ?)", (event_id, username))
            self.conn.execute("UPDATE events SET likes_count = likes_count + 1 WHERE id = ?", (event_id,))
        self.conn.commit()
    
    def is_liked(self, event_id, username):
        result = self.execute_query("SELECT 1 FROM event_likes WHERE event_id = ? AND username = ?", (event_id, username), fetch_one=True)
        return result is not None
    
    def add_registration(self, reg_data):
        if self.is_registered(reg_data['event_id'], reg_data['student_username']):
            return None
        query = '''
            INSERT INTO registrations (id, event_id, event_title, student_username, student_name,
            student_roll, student_dept, via_app, status, registered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            str(uuid.uuid4()),
            reg_data['event_id'],
            reg_data['event_title'],
            reg_data['student_username'],
            reg_data['student_name'],
            reg_data.get('student_roll'),
            reg_data.get('student_dept'),
            True,
            'pending',
            datetime.now().isoformat()
        )
        return self.execute_query(query, params)
    
    def is_registered(self, event_id, username):
        result = self.execute_query(
            "SELECT id FROM registrations WHERE event_id = ? AND student_username = ?",
            (event_id, username), fetch_one=True
        )
        return result is not None
    
    def get_registrations_by_event(self, event_id):
        query = "SELECT * FROM registrations WHERE event_id = ? ORDER BY registered_at DESC"
        return self.execute_query(query, (event_id,), fetch_all=True)

# Initialize database
db_manager = CompleteEventDatabaseManager()

# ============================================
# HELPER FUNCTIONS
# ============================================
def format_date(date_str):
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%d %b %Y, %I:%M %p")
    except:
        return str(date_str)

def is_upcoming(event_date):
    try:
        dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        return dt > datetime.now()
    except:
        return True

def display_role_badge(role):
    badges = {
        "admin": ("👑 Admin", "admin-badge"),
        "faculty": ("👨‍🏫 Faculty", "faculty-badge"),
        "student": ("👨‍🎓 Student", "student-badge")
    }
    if role in badges:
        text, css_class = badges[role]
        st.markdown(f'<span class="role-badge {css_class}">{text}</span>', unsafe_allow_html=True)

# ============================================
# ENHANCED EVENT CARD UI
# ============================================
def display_event_card(event, current_user=None):
    """Enhanced event card with complete social features"""
    event_id = event['id']
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(event['title'])
        st.caption(f"📅 {format_date(event['event_date'])}")
        st.caption(f"📍 {event['venue']}")
        st.caption(f"👥 Organized by {event['created_by_name']}")
        
        if event.get('flyer_path'):
            st.markdown(f'<div class="flyer-container"><img src="{event["flyer_path"]}" style="max-height: 200px; border-radius: 8px;"></div>', unsafe_allow_html=True)
        
        st.markdown(event['description'][:200] + "..." if len(event['description']) > 200 else event['description'])
    
    with col2:
        status_color = "success" if is_upcoming(event['event_date']) else "error"
        st.metric(label="Status", value="Upcoming" if is_upcoming(event['event_date']) else "Past", delta=None)
        st.metric("👥 Participants", event.get('likes_count', 0))
    
    # Social buttons
    col_like, col_fav, col_int, col_reg, col_share = st.columns(5)
    
    with col_like:
        is_liked = db_manager.is_liked(event_id, current_user)
        if st.button("❤️ Like" if not is_liked else "💔 Unlike", key=f"like_{event_id}"):
            db_manager.toggle_like(event_id, current_user)
            st.rerun()
    
    with col_fav:
        st.button("⭐ Favorite", key=f"fav_{event_id}")
    
    with col_int:
        if st.button("🙋 Interested", key=f"int_{event_id}"):
            if db_manager.is_registered(event_id, current_user):
                st.success("✅ Already Registered!")
            else:
                st.info("🔗 Register via official link first")
    
    with col_reg:
        if current_user and db_manager.is_registered(event_id, current_user):
            st.success("✅ Registered")
        else:
            if st.button("📝 I've Registered", key=f"reg_{event_id}"):
                user = db_manager.get_user(current_user)
                reg_data = {
                    'event_id': event_id,
                    'event_title': event['title'],
                    'student_username': current_user,
                    'student_name': user['name'] if user else current_user,
                    'student_roll': user['roll_no'] if user else '',
                    'student_dept': user['department'] if user else ''
                }
                db_manager.add_registration(reg_data)
                st.success("✅ Registration recorded!")
                st.rerun()
    
    with col_share:
        st.button("📤 Share", key=f"share_{event_id}")

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # College header
    st.markdown('<div class="college-header"><h1>🎓 G H Raisoni College</h1><h2>Event Management System</h2></div>', unsafe_allow_html=True)
    
    # Login system
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.name = None
    
    if not st.session_state.logged_in:
        login_tab1, login_tab2 = st.tabs(["Login", "Register"])
        
        with login_tab1:
            st.subheader("Login")
            role = st.selectbox("Role", ["student", "admin", "faculty"])
            username = st.text_input("Username/Email")
            password = st.text_input("Password", type="password")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Login", use_container_width=True):
                    if db_manager.verify_credentials(username, password, role):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.role = role
                        user = db_manager.get_user(username) if role == 'student' else None
                        st.session_state.name = user['name'] if user else username.split('@')[0].title()
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with login_tab2:
            st.subheader("Register (Students Only)")
            with st.form("register"):
                name = st.text_input("Full Name")
                roll_no = st.text_input("Roll Number")
                department = st.selectbox("Department", ["Computer", "IT", "Electronics", "Mechanical", "Civil"])
                year = st.selectbox("Year", ["FE", "SE", "TE", "BE"])
                email = st.text_input("Email")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Register"):
                    if db_manager.add_user({
                        'name': name, 'roll_no': roll_no, 'department': department,
                        'year': year, 'email': email, 'username': username, 'password': password
                    }):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Registration failed. Username may already exist.")
        
        st.stop()
    
    # User dashboard
    display_role_badge(st.session_state.role)
    st.success(f"Welcome, {st.session_state.name}! 👋")
    
    if st.session_state.role in ['admin', 'faculty']:
        # Admin/Faculty sidebar
        with st.sidebar:
            selected = option_menu(
                "Menu",
                options=["Events Feed", "Add Event", "Manage Events", "Manage Students", "Analytics", "Logout"],
                icons=["compass", "plus-circle", "list-check", "people", "bar-chart", "box-arrow-right"],
                menu_icon="cast",
                default_index=0,
            )
        
        if selected == "Events Feed":
            st.markdown('<h1 class="main-header">Discover Events</h1>', unsafe_allow_html=True)
            events = db_manager.get_all_events()
            if events:
                for event in events:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("No events available yet.")
        
        elif selected == "Add Event":
            st.markdown('<h1 class="main-header">Add New Event</h1>', unsafe_allow_html=True)
            with st.form("add_event"):
                col1, col2 = st.columns(2)
                with col1:
                    title = st.text_input("Event Title")
                    event_type = st.selectbox("Event Type", ["Workshop", "Hackathon", "Competition", "Bootcamp", "Seminar"])
                    date = st.date_input("Event Date")
                    time = st.time_input("Event Time")
                with col2:
                    venue = st.text_input("Venue")
                    organizer = st.text_input("Organizer")
                    max_participants = st.number_input("Max Participants", min_value=1, value=100)
                
                description = st.text_area("Description")
                registration_link = st.text_input("Registration Link")
                uploaded_file = st.file_uploader("Event Flyer", type=['png', 'jpg', 'jpeg'])
                
                if st.form_submit_button("Create Event"):
                    event_data = {
                        'title': title,
                        'event_type': event_type,
                        'event_date': datetime.combine(date, time).isoformat(),
                        'venue': venue,
                        'organizer': organizer,
                        'max_participants': max_participants,
                        'description': description,
                        'registration_link': registration_link
                    }
                    if uploaded_file:
                        event_data['flyer_path'] = f"data:image/jpeg;base64,{base64.b64encode(uploaded_file.read()).decode()}"
                    
                    if db_manager.add_event(event_data):
                        st.success("Event created successfully! 🎉")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Failed to create event.")
        
        elif selected == "Manage Events":
            st.markdown('<h1 class="main-header">Manage Events</h1>', unsafe_allow_html=True)
            events = db_manager.get_events_by_creator(st.session_state.username)
            
            for event in events:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.subheader(event['title'])
                        st.caption(format_date(event['event_date']))
                    
                    with col2:
                        st.info(f"📍 {event['venue']}")
                    
                    with col3:
                        if st.button("✏️ Edit", key=f"edit_{event['id']}"):
                            st.session_state.editing_event = event['id']
                            st.rerun()
                        if st.button("🗑️ Delete", key=f"del_{event['id']}"):
                            db_manager.delete_event(event['id'])
                            st.success("Event deleted!")
                            st.rerun()
            
            if 'editing_event' in st.session_state:
                edit_event = db_manager.get_event(st.session_state.editing_event)
                if edit_event:
                    st.subheader(f"Edit: {edit_event['title']}")
                    # Edit form here (simplified)
        
        elif selected == "Manage Students":
            st.markdown('<h1 class="main-header">Manage Students</h1>', unsafe_allow_html=True)
            users = db_manager.get_all_users()
            if users:
                df = pd.DataFrame(users)
                st.dataframe(df[['name', 'username', 'roll_no', 'department', 'year', 'created_at']], use_container_width=True)
            else:
                st.info("No students registered yet.")
        
        elif selected == "Analytics":
            st.markdown('<h1 class="main-header">Analytics</h1>', unsafe_allow_html=True)
            events = db_manager.get_all_events()
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Events", len(events))
            with col2: st.metric("Upcoming", len([e for e in events if is_upcoming(e['event_date'])]))
            with col3: st.metric("Total Likes", sum(e.get('likes_count', 0) for e in events))
            with col4: st.metric("Total Registrations", 42)  # Placeholder
        
        elif selected == "Logout":
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    else:  # Student view
        with st.sidebar:
            selected = option_menu(
                "Menu",
                options=["Events Feed", "My Registrations", "Profile", "Logout"],
                icons=["compass", "list-check", "person", "box-arrow-right"],
                default_index=0,
            )
        
        if selected == "Events Feed":
            events = db_manager.get_all_events()
            if events:
                for event in events:
                    display_event_card(event, st.session_state.username)
        
        elif selected == "My Registrations":
            st.subheader("My Registrations")
            # Registration logic here
        
        elif selected == "Profile":
            st.subheader("My Profile")
            user = db_manager.get_user(st.session_state.username)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Name", user['name'] if user else "N/A")
                st.metric("Roll No", user['roll_no'] if user else "N/A")
            with col2:
                st.metric("Department", user['department'] if user else "N/A")
                st.metric("Year", user['year'] if user else "N/A")
        
        elif selected == "Logout":
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()

