"""
G H Raisoni College - Advanced Event Management System
Complete solution with AI, Image Uploads, Social Features
Deployable on Streamlit Cloud - Streamlit Native Version
"""

import streamlit as st
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
    
    .registration-section {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #3B82F6;
        font-size: 0.95rem;
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
        
        if dt > datetime.now():
            return '<span class="status-badge status-upcoming">🟢 Upcoming</span>'
        elif dt.date() == datetime.now().date():
            return '<span class="status-badge status-ongoing">🟡 Ongoing</span>'
        else:
            return '<span class="status-badge status-past">🔴 Past</span>'
    except:
        return '<span class="status-badge">Unknown</span>'

# ============================================
# DATABASE MANAGER
# ============================================
class DatabaseManager:
    """Simple database manager"""
    
    def __init__(self, db_path="event_management.db"):
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
    
    def create_tables(self):
        """Create all necessary tables"""
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
                    current_participants INTEGER DEFAULT 0,
                    flyer_path TEXT,
                    created_by TEXT,
                    created_by_name TEXT,
                    ai_generated BOOLEAN DEFAULT 0,
                    status TEXT DEFAULT 'upcoming',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    status TEXT DEFAULT 'pending',
                    attendance TEXT DEFAULT 'absent',
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(event_id, student_username)
                )
            ''')
            
            self.conn.commit()
            
            # Add default users if not exist
            self._add_default_users()
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    def _add_default_users(self):
        """Add default admin and faculty users"""
        try:
            cursor = self.conn.cursor()
            
            # Check if admin exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ('admin@raisoni',))
            if cursor.fetchone()[0] == 0:
                admin_id = str(uuid.uuid4())
                hashed_pass = hashlib.sha256('admin123'.encode()).hexdigest()
                cursor.execute('''
                    INSERT INTO users (id, name, username, password, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (admin_id, 'Administrator', 'admin@raisoni', hashed_pass, 'admin', datetime.now().isoformat()))
            
            # Check if faculty exists
            cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", ('faculty@raisoni',))
            if cursor.fetchone()[0] == 0:
                faculty_id = str(uuid.uuid4())
                hashed_pass = hashlib.sha256('faculty123'.encode()).hexdigest()
                cursor.execute('''
                    INSERT INTO users (id, name, username, password, role, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (faculty_id, 'Faculty Coordinator', 'faculty@raisoni', hashed_pass, 'faculty', datetime.now().isoformat()))
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error adding default users: {e}")
    
    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials"""
        try:
            if role in ['admin', 'faculty']:
                default_creds = {
                    'admin': {'username': 'admin@raisoni', 'password': 'admin123'},
                    'faculty': {'username': 'faculty@raisoni', 'password': 'faculty123'}
                }
                creds = default_creds[role]
                if username == creds['username']:
                    return hashlib.sha256(password.encode()).hexdigest() == hashlib.sha256(creds['password'].encode()).hexdigest()
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
                    return hashlib.sha256(password.encode()).hexdigest() == stored_hash
                return False
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    def get_user(self, username):
        """Get user by username"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        return dict(result) if result else None
    
    def add_user(self, user_data):
        """Add new user"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO users (id, name, roll_no, department, year, email, username, password, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_data.get('id', str(uuid.uuid4())),
                user_data.get('name'),
                user_data.get('roll_no'),
                user_data.get('department'),
                user_data.get('year'),
                user_data.get('email'),
                user_data.get('username'),
                self._hash_password(user_data.get('password')),
                user_data.get('role', 'student'),
                datetime.now().isoformat()
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return False
    
    def add_event(self, event_data):
        """Add new event"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO events (
                    id, title, description, event_type, event_date, venue, organizer,
                    registration_link, max_participants, flyer_path, created_by,
                    created_by_name, ai_generated, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
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
                'upcoming',
                datetime.now().isoformat()
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding event: {e}")
            return False
    
    def get_all_events(self):
        """Get all events"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM events ORDER BY event_date DESC")
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def get_events_by_creator(self, username):
        """Get events created by specific user"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM events WHERE created_by = ? ORDER BY event_date DESC", (username,))
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def add_registration(self, reg_data):
        """Add new registration"""
        try:
            cursor = self.conn.cursor()
            
            # Check if already registered
            cursor.execute("SELECT id FROM registrations WHERE event_id = ? AND student_username = ?", 
                          (reg_data['event_id'], reg_data['student_username']))
            if cursor.fetchone():
                return None
            
            cursor.execute('''
                INSERT INTO registrations (
                    id, event_id, event_title, student_username, student_name,
                    student_roll, student_dept, status, attendance, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                reg_data.get('id', str(uuid.uuid4())),
                reg_data.get('event_id'),
                reg_data.get('event_title'),
                reg_data.get('student_username'),
                reg_data.get('student_name'),
                reg_data.get('student_roll'),
                reg_data.get('student_dept'),
                reg_data.get('status', 'pending'),
                reg_data.get('attendance', 'absent'),
                datetime.now().isoformat()
            ))
            
            # Update event participant count
            cursor.execute("UPDATE events SET current_participants = current_participants + 1 WHERE id = ?", 
                          (reg_data['event_id'],))
            
            self.conn.commit()
            return reg_data['id']
        except Exception as e:
            logger.error(f"Error adding registration: {e}")
            return None
    
    def get_registrations_by_student(self, username):
        """Get all registrations for a student"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT r.*, e.event_date, e.venue, e.status as event_status 
            FROM registrations r
            LEFT JOIN events e ON r.event_id = e.id
            WHERE r.student_username = ?
            ORDER BY r.registered_at DESC
        ''', (username,))
        results = cursor.fetchall()
        return [dict(row) for row in results]
    
    def is_student_registered(self, event_id, username):
        """Check if student is registered for event"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM registrations WHERE event_id = ? AND student_username = ?", 
                      (event_id, username))
        return cursor.fetchone() is not None
    
    def update_event_status(self):
        """Update event status based on current time"""
        try:
            now = datetime.now().isoformat()
            cursor = self.conn.cursor()
            cursor.execute("UPDATE events SET status = 'past' WHERE event_date <= ? AND status != 'past'", (now,))
            cursor.execute("UPDATE events SET status = 'ongoing' WHERE event_date > ? AND status != 'ongoing'", (now,))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating event status: {e}")
            return False

# Initialize database
db = DatabaseManager()

# ============================================
# EVENT CARD DISPLAY
# ============================================
def display_event_card(event, current_user=None):
    """Display event card"""
    if not event or not event.get('id'):
        return
    
    event_id = event.get('id')
    
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
        if flyer and flyer.startswith('data:image'):
            try:
                st.image(flyer, width=200, use_column_width=True)
            except:
                pass
        
        # Registration Section
        if current_user:
            st.markdown('<div class="registration-section">', unsafe_allow_html=True)
            
            is_registered = db.is_student_registered(event_id, current_user)
            
            if is_registered:
                st.success("✅ You are registered for this event")
            else:
                reg_link = event.get('registration_link', '')
                
                if reg_link:
                    col_link, col_app = st.columns(2)
                    
                    with col_link:
                        st.markdown(f"[🔗 **Register via Official Link**]({reg_link})", 
                                  unsafe_allow_html=True)
                        st.caption("Click to register on external platform")
                        
                        if st.button("✅ I've Registered via Link", 
                                   key=f"link_reg_{event_id}",
                                   use_container_width=True):
                            student = db.get_user(current_user)
                            if student:
                                reg_data = {
                                    'id': str(uuid.uuid4()),
                                    'event_id': event_id,
                                    'event_title': event.get('title'),
                                    'student_username': current_user,
                                    'student_name': student.get('name', current_user),
                                    'student_roll': student.get('roll_no', 'N/A'),
                                    'student_dept': student.get('department', 'N/A'),
                                    'status': 'confirmed'
                                }
                                if db.add_registration(reg_data):
                                    st.success("✅ Registration recorded successfully!")
                                    st.rerun()
                    
                    with col_app:
                        st.markdown("**Register via App**")
                        st.caption("Register directly in our system")
                        
                        if st.button("📱 Register via App", 
                                   key=f"app_reg_{event_id}",
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
                                if db.add_registration(reg_data):
                                    st.success("✅ Registration recorded successfully!")
                                    st.rerun()
                else:
                    st.markdown("**Register via App**")
                    st.caption("Register directly in our system")
                    
                    if st.button("📱 Register for Event", 
                               key=f"reg_{event_id}",
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
                            if db.add_registration(reg_data):
                                st.success("✅ Registration recorded successfully!")
                                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Creator info
        created_by = event.get('created_by_name', 'Unknown')
        st.caption(f"Created by: {created_by}")
        
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
                                'username': username,
                                'password': password,
                                'role': 'student'
                            }
                            
                            if db.add_user(user_data):
                                st.success("✅ Registration successful! Please login.")
                                st.rerun()
                            else:
                                st.error("Registration failed")

# ============================================
# STUDENT DASHBOARD
# ============================================
def student_dashboard():
    """Student dashboard"""
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    
    # Get student info
    student = db.get_user(st.session_state.username)
    if student:
        st.sidebar.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
        st.sidebar.markdown(f"**Department:** {student.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Year:** {student.get('year', 'N/A')}")
    
    display_role_badge('student')
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Events Feed", "My Registrations", "My Profile"]
        
        if 'student_page' not in st.session_state:
            st.session_state.student_page = "Events Feed"
        
        for option in nav_options:
            is_active = st.session_state.student_page == option
            button_class = "active" if is_active else ""
            button_text = f"▶ {option}" if is_active else option
            
            if st.button(button_text, key=f"student_{option}", use_container_width=True):
                st.session_state.student_page = option
                st.rerun()
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Page content
    selected = st.session_state.student_page
    
    if selected == "Events Feed":
        st.markdown('<h1 class="main-header">🎯 Discover Events</h1>', unsafe_allow_html=True)
        
        # Update event status
        db.update_event_status()
        
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
        
        # Display registrations
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
                    
                    # Registration details
                    reg_status = reg.get('status', 'pending').title()
                    st.caption(f"📝 Status: {reg_status}")
                
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
            st.markdown("### Account Information")
            st.markdown(f"**Email:** {student.get('email', 'N/A')}")
            st.markdown(f"**Username:** {student.get('username', 'N/A')}")
            st.markdown(f"**Member Since:** {format_date(student.get('created_at'))}")
        
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
# FACULTY DASHBOARD
# ============================================
def faculty_dashboard():
    """Faculty coordinator dashboard"""
    st.sidebar.title("👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('faculty')
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Dashboard", "Create Event", "My Events", "Registrations"]
        
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
        if st.button("Logout", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Page content
    selected = st.session_state.faculty_page
    
    if selected == "Dashboard":
        st.markdown('<h1 class="main-header">Faculty Dashboard</h1>', unsafe_allow_html=True)
        
        # Statistics
        events = db.get_events_by_creator(st.session_state.username)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("My Events", len(events))
        with col2:
            upcoming = len([e for e in events if e.get('status') == 'upcoming'])
            st.metric("Upcoming", upcoming)
        
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
                        try:
                            flyer.seek(0)
                            image_bytes = flyer.getvalue()
                            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                            file_ext = os.path.splitext(flyer.name)[1].lower()
                            mime_types = {
                                '.jpg': 'image/jpeg',
                                '.jpeg': 'image/jpeg',
                                '.png': 'image/png',
                                '.gif': 'image/gif'
                            }
                            mime_type = mime_types.get(file_ext, 'image/jpeg')
                            flyer_path = f"data:{mime_type};base64,{image_base64}"
                        except Exception as e:
                            logger.error(f"Error processing image: {e}")
                    
                    # Combine date and time
                    event_datetime = datetime.combine(event_date, event_time)
                    
                    event_data = {
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
                    
                    if db.add_event(event_data):
                        st.success(f"Event '{title}' created successfully! 🎉")
                        st.rerun()
                    else:
                        st.error("Failed to create event")
    
    elif selected == "My Events":
        st.header("📋 My Events")
        
        events = db.get_events_by_creator(st.session_state.username)
        
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
            
            # Get registrations (simplified - in real app, you'd need a method to get registrations by event)
            st.info(f"Viewing registrations for: {selected_title}")
            st.caption("Registration tracking feature will be available in the next update.")

# ============================================
# ADMIN DASHBOARD
# ============================================
def admin_dashboard():
    """Admin dashboard"""
    st.sidebar.title("👑 Admin Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('admin')
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Dashboard", "Manage Events", "Manage Users"]
        
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
        if st.button("Logout", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Events", len(events))
        with col2:
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
                            # Note: In real app, implement delete functionality
                            st.warning("Delete functionality will be implemented in next update")
        else:
            st.info("No events found.")
    
    elif selected == "Manage Users":
        st.header("👥 Manage Users")
        st.info("User management feature will be available in the next update.")

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

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    main()
