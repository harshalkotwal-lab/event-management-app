"""
G H Raisoni College - Event Management System
SQLite Version with Like/Interested Features
"""

import streamlit as st
from datetime import datetime, date, timedelta
import sqlite3
import hashlib
import uuid
import pandas as pd
import json
import os
from pathlib import Path

# ============================================
# PAGE CONFIGURATION - MUST BE FIRST
# ============================================

st.set_page_config(
    page_title="Raisoni Event Manager",
    page_icon="🎓",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .event-card {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .event-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    
    .social-btn {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        background: white;
        font-size: 0.9rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .role-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-left: 0.5rem;
    }
    
    .admin-badge { background: #FEE2E2; color: #DC2626; }
    .faculty-badge { background: #DBEAFE; color: #1D4ED8; }
    .student-badge { background: #D1FAE5; color: #065F46; }
</style>
""", unsafe_allow_html=True)

# ============================================
# SQLITE DATABASE MANAGER
# ============================================

class EventDatabase:
    """Simple SQLite database manager"""
    
    def __init__(self, db_path="event_management.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT,
                role TEXT NOT NULL,
                department TEXT,
                roll_number TEXT,
                year TEXT,
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
                created_by TEXT,
                created_by_name TEXT,
                likes_count INTEGER DEFAULT 0,
                interested_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'upcoming',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Likes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS likes (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                username TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_id, username)
            )
        ''')
        
        # Interested table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interested (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                username TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_id, username)
            )
        ''')
        
        # Registrations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS registrations (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                username TEXT NOT NULL,
                status TEXT DEFAULT 'confirmed',
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_id, username)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # Add default users if not exist
        self.add_default_users()
    
    def add_default_users(self):
        """Add default admin and faculty users"""
        default_users = [
            {
                'id': 'admin-001',
                'username': 'admin@raisoni',
                'password_hash': hashlib.sha256('admin123'.encode()).hexdigest(),
                'name': 'Administrator',
                'email': 'admin@raisoni.edu',
                'role': 'admin'
            },
            {
                'id': 'faculty-001',
                'username': 'faculty@raisoni',
                'password_hash': hashlib.sha256('faculty123'.encode()).hexdigest(),
                'name': 'Faculty Coordinator',
                'email': 'faculty@raisoni.edu',
                'role': 'faculty',
                'department': 'CSE'
            }
        ]
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        for user in default_users:
            cursor.execute('''
                INSERT OR IGNORE INTO users (id, username, password_hash, name, email, role, department)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user['id'], user['username'], user['password_hash'], 
                  user['name'], user['email'], user['role'], user.get('department', '')))
        
        conn.commit()
        conn.close()
    
    # ========== USER OPERATIONS ==========
    
    def add_user(self, user_data):
        """Add new user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        user_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO users (id, username, password_hash, name, email, role, department, roll_number, year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, 
              user_data['username'],
              hashlib.sha256(user_data['password'].encode()).hexdigest(),
              user_data['name'],
              user_data['email'],
              user_data['role'],
              user_data.get('department', ''),
              user_data.get('roll_number', ''),
              user_data.get('year', '')))
        
        conn.commit()
        conn.close()
        return user_id
    
    def get_user(self, username):
        """Get user by username"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        conn.close()
        return dict(user) if user else None
    
    def verify_user(self, username, password):
        """Verify user credentials"""
        user = self.get_user(username)
        if user:
            return user['password_hash'] == hashlib.sha256(password.encode()).hexdigest()
        return False
    
    # ========== EVENT OPERATIONS ==========
    
    def add_event(self, event_data):
        """Add new event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        event_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO events (id, title, description, event_type, event_date, venue, 
                              organizer, registration_link, max_participants, created_by, created_by_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (event_id,
              event_data['title'],
              event_data['description'],
              event_data['event_type'],
              event_data['event_date'].isoformat() if isinstance(event_data['event_date'], datetime) else event_data['event_date'],
              event_data['venue'],
              event_data['organizer'],
              event_data.get('registration_link', ''),
              event_data.get('max_participants', 100),
              event_data['created_by'],
              event_data['created_by_name']))
        
        conn.commit()
        conn.close()
        return event_id
    
    def get_all_events(self):
        """Get all events"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT e.*, 
                   (SELECT COUNT(*) FROM likes WHERE event_id = e.id) as likes_count,
                   (SELECT COUNT(*) FROM interested WHERE event_id = e.id) as interested_count
            FROM events e
            ORDER BY e.event_date DESC
        ''')
        events = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return events
    
    def get_event(self, event_id):
        """Get specific event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT e.*, 
                   (SELECT COUNT(*) FROM likes WHERE event_id = e.id) as likes_count,
                   (SELECT COUNT(*) FROM interested WHERE event_id = e.id) as interested_count
            FROM events e
            WHERE e.id = ?
        ''', (event_id,))
        
        event = cursor.fetchone()
        conn.close()
        return dict(event) if event else None
    
    # ========== SOCIAL INTERACTIONS ==========
    
    def toggle_like(self, event_id, username):
        """Toggle like for an event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if already liked
        cursor.execute('SELECT id FROM likes WHERE event_id = ? AND username = ?', 
                      (event_id, username))
        existing = cursor.fetchone()
        
        if existing:
            # Remove like
            cursor.execute('DELETE FROM likes WHERE event_id = ? AND username = ?', 
                          (event_id, username))
            action = 'removed'
        else:
            # Add like
            like_id = str(uuid.uuid4())
            cursor.execute('INSERT INTO likes (id, event_id, username) VALUES (?, ?, ?)',
                          (like_id, event_id, username))
            action = 'added'
        
        # Update event likes count
        cursor.execute('''
            UPDATE events 
            SET likes_count = (SELECT COUNT(*) FROM likes WHERE event_id = ?)
            WHERE id = ?
        ''', (event_id, event_id))
        
        conn.commit()
        conn.close()
        return action
    
    def toggle_interested(self, event_id, username):
        """Toggle interested status for an event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if already interested
        cursor.execute('SELECT id FROM interested WHERE event_id = ? AND username = ?', 
                      (event_id, username))
        existing = cursor.fetchone()
        
        if existing:
            # Remove interest
            cursor.execute('DELETE FROM interested WHERE event_id = ? AND username = ?', 
                          (event_id, username))
            action = 'removed'
        else:
            # Add interest
            interest_id = str(uuid.uuid4())
            cursor.execute('INSERT INTO interested (id, event_id, username) VALUES (?, ?, ?)',
                          (interest_id, event_id, username))
            action = 'added'
        
        # Update event interested count
        cursor.execute('''
            UPDATE events 
            SET interested_count = (SELECT COUNT(*) FROM interested WHERE event_id = ?)
            WHERE id = ?
        ''', (event_id, event_id))
        
        conn.commit()
        conn.close()
        return action
    
    def check_user_like(self, event_id, username):
        """Check if user liked an event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM likes WHERE event_id = ? AND username = ?', 
                      (event_id, username))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def check_user_interested(self, event_id, username):
        """Check if user is interested in an event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM interested WHERE event_id = ? AND username = ?', 
                      (event_id, username))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def get_user_likes(self, username):
        """Get events liked by user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT e.* FROM events e
            JOIN likes l ON e.id = l.event_id
            WHERE l.username = ?
            ORDER BY l.created_at DESC
        ''', (username,))
        
        events = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return events
    
    def get_user_interested(self, username):
        """Get events user is interested in"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT e.* FROM events e
            JOIN interested i ON e.id = i.event_id
            WHERE i.username = ?
            ORDER BY i.created_at DESC
        ''', (username,))
        
        events = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return events
    
    # ========== REGISTRATION OPERATIONS ==========
    
    def register_for_event(self, event_id, username):
        """Register user for an event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if already registered
        cursor.execute('SELECT id FROM registrations WHERE event_id = ? AND username = ?', 
                      (event_id, username))
        if cursor.fetchone():
            conn.close()
            return False
        
        # Add registration
        reg_id = str(uuid.uuid4())
        cursor.execute('INSERT INTO registrations (id, event_id, username) VALUES (?, ?, ?)',
                      (reg_id, event_id, username))
        
        conn.commit()
        conn.close()
        return True
    
    def check_registration(self, event_id, username):
        """Check if user is registered for event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM registrations WHERE event_id = ? AND username = ?', 
                      (event_id, username))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def get_user_registrations(self, username):
        """Get all registrations for a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT r.*, e.title, e.event_date, e.venue, e.event_type
            FROM registrations r
            JOIN events e ON r.event_id = e.id
            WHERE r.username = ?
            ORDER BY r.registered_at DESC
        ''', (username,))
        
        registrations = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return registrations
    
    # ========== STATISTICS ==========
    
    def get_statistics(self):
        """Get system statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Count events
        cursor.execute('SELECT COUNT(*) FROM events')
        stats['total_events'] = cursor.fetchone()[0]
        
        # Count users by role
        cursor.execute("SELECT role, COUNT(*) FROM users GROUP BY role")
        role_counts = cursor.fetchall()
        for role, count in role_counts:
            stats[f'total_{role}s'] = count
        
        # Count registrations
        cursor.execute('SELECT COUNT(*) FROM registrations')
        stats['total_registrations'] = cursor.fetchone()[0]
        
        # Count upcoming events
        cursor.execute("SELECT COUNT(*) FROM events WHERE event_date > datetime('now')")
        stats['upcoming_events'] = cursor.fetchone()[0]
        
        conn.close()
        return stats

# Initialize database
db = EventDatabase()

# ============================================
# HELPER FUNCTIONS
# ============================================

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

# ============================================
# EVENT CARD WITH SOCIAL FEATURES
# ============================================

def display_event_card(event, current_user=None):
    """Display event card with Like/Interested buttons"""
    if not event:
        return
    
    event_id = event['id']
    
    # Check user interactions
    user_liked = False
    user_interested = False
    is_registered = False
    
    if current_user:
        user_liked = db.check_user_like(event_id, current_user)
        user_interested = db.check_user_interested(event_id, current_user)
        is_registered = db.check_registration(event_id, current_user)
    
    # Get counts
    likes_count = event.get('likes_count', 0)
    interested_count = event.get('interested_count', 0)
    
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        # Title
        st.markdown(f'<div class="card-title">{event["title"]}</div>', unsafe_allow_html=True)
        
        # Details
        st.caption(f"📅 {format_date(event['event_date'])} | 📍 {event.get('venue', 'N/A')}")
        st.caption(f"🏷️ {event.get('event_type', 'Event')} | 👨‍🏫 {event.get('organizer', 'N/A')}")
        
        # Description
        desc = event.get('description', '')
        if desc:
            if len(desc) > 150:
                with st.expander("📝 Description"):
                    st.write(desc)
                st.caption(f"{desc[:150]}...")
            else:
                st.write(desc)
        
        # SOCIAL BUTTONS - LIKE & INTERESTED
        col1, col2 = st.columns(2)
        
        with col1:
            like_icon = "❤️" if user_liked else "🤍"
            like_text = "Liked" if user_liked else "Like"
            
            if st.button(f"{like_icon} {like_text} ({likes_count})", 
                        key=f"like_{event_id}_{current_user}",
                        use_container_width=True):
                if current_user:
                    db.toggle_like(event_id, current_user)
                    st.rerun()
        
        with col2:
            int_icon = "👍" if user_interested else "🤔"
            int_text = "Interested" if user_interested else "Interested?"
            
            if st.button(f"{int_icon} {int_text} ({interested_count})", 
                        key=f"int_{event_id}_{current_user}",
                        use_container_width=True):
                if current_user:
                    db.toggle_interested(event_id, current_user)
                    st.rerun()
        
        # REGISTRATION BUTTON
        if current_user:
            if not is_registered:
                if st.button("📝 Register for Event", 
                           key=f"reg_{event_id}_{current_user}",
                           use_container_width=True,
                           type="primary"):
                    if db.register_for_event(event_id, current_user):
                        st.success("✅ Registered successfully!")
                        st.rerun()
            else:
                st.success("✅ Already registered")
        
        # Creator info
        if event.get('created_by_name'):
            st.caption(f"Created by: {event['created_by_name']}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# LOGIN PAGE
# ============================================

def login_page():
    """Login page"""
    st.markdown('<h1 class="main-header">🎓 G H Raisoni College</h1>', unsafe_allow_html=True)
    st.subheader("Event Management System")
    
    tab1, tab2, tab3 = st.tabs(["Admin", "Faculty", "Student"])
    
    with tab1:
        st.subheader("Admin Login")
        admin_user = st.text_input("Username", value="admin@raisoni", key="admin_user")
        admin_pass = st.text_input("Password", type="password", value="admin123", key="admin_pass")
        
        if st.button("Admin Login", type="primary", use_container_width=True):
            if db.verify_user(admin_user, admin_pass):
                user = db.get_user(admin_user)
                st.session_state.user = user
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with tab2:
        st.subheader("Faculty Login")
        faculty_user = st.text_input("Username", value="faculty@raisoni", key="faculty_user")
        faculty_pass = st.text_input("Password", type="password", value="faculty123", key="faculty_pass")
        
        if st.button("Faculty Login", type="primary", use_container_width=True):
            if db.verify_user(faculty_user, faculty_pass):
                user = db.get_user(faculty_user)
                st.session_state.user = user
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with tab3:
        st.subheader("Student Portal")
        
        login_tab, register_tab = st.tabs(["Login", "Register"])
        
        with login_tab:
            student_user = st.text_input("Username", key="student_login_user")
            student_pass = st.text_input("Password", type="password", key="student_login_pass")
            
            if st.button("Student Login", type="primary", use_container_width=True):
                if db.verify_user(student_user, student_pass):
                    user = db.get_user(student_user)
                    st.session_state.user = user
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        with register_tab:
            with st.form("student_registration"):
                st.markdown("### Create Student Account")
                
                name = st.text_input("Full Name *")
                roll_no = st.text_input("Roll Number *")
                department = st.selectbox("Department *", 
                                         ["CSE", "AIML", "IT", "EE", "BCA", "MCA", "BBA", "MBA"])
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
                        existing = db.get_user(username)
                        if existing:
                            st.error("Username already exists")
                        else:
                            user_data = {
                                'username': username,
                                'password': password,
                                'name': name,
                                'email': email,
                                'role': 'student',
                                'department': department,
                                'roll_number': roll_no,
                                'year': year
                            }
                            db.add_user(user_data)
                            st.success("✅ Registration successful! Please login.")
                            st.rerun()

# ============================================
# STUDENT DASHBOARD
# ============================================

def student_dashboard():
    """Student dashboard"""
    user = st.session_state.user
    
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**Name:** {user['name']}")
    if user.get('roll_number'):
        st.sidebar.markdown(f"**Roll No:** {user['roll_number']}")
    if user.get('department'):
        st.sidebar.markdown(f"**Department:** {user['department']}")
    
    display_role_badge('student')
    
    # Navigation
    menu = st.sidebar.selectbox("Navigation", 
                               ["📅 Browse Events", "📋 My Registrations", "⭐ My Interests", "👤 Profile"])
    
    if menu == "📅 Browse Events":
        st.header("🎯 Discover Events")
        
        # Get all events
        events = db.get_all_events()
        
        if events:
            for event in events:
                display_event_card(event, user['username'])
        else:
            st.info("No events available yet.")
            # Add sample events button for testing
            if st.button("Add Sample Events", type="secondary"):
                add_sample_events()
                st.rerun()
    
    elif menu == "📋 My Registrations":
        st.header("📋 My Registrations")
        
        registrations = db.get_user_registrations(user['username'])
        
        if registrations:
            for reg in registrations:
                with st.container():
                    st.markdown('<div class="event-card">', unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="card-title">{reg["title"]}</div>', unsafe_allow_html=True)
                    st.caption(f"📅 {format_date(reg['event_date'])} | 📍 {reg.get('venue', 'N/A')}")
                    st.caption(f"Type: {reg.get('event_type', 'Event')} | Status: {reg.get('status', 'confirmed').title()}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Statistics
            st.metric("Total Registrations", len(registrations))
        else:
            st.info("No registrations yet. Browse events to register!")
    
    elif menu == "⭐ My Interests":
        st.header("⭐ My Interests")
        
        # Get liked and interested events
        liked_events = db.get_user_likes(user['username'])
        interested_events = db.get_user_interested(user['username'])
        
        tab1, tab2 = st.tabs([f"❤️ Liked ({len(liked_events)})", 
                             f"🤔 Interested ({len(interested_events)})"])
        
        with tab1:
            if liked_events:
                for event in liked_events:
                    display_event_card(event, user['username'])
            else:
                st.info("No liked events yet.")
                st.caption("Like events by clicking the ❤️ button")
        
        with tab2:
            if interested_events:
                for event in interested_events:
                    display_event_card(event, user['username'])
            else:
                st.info("No interested events yet.")
                st.caption("Mark interest by clicking the 👍 button")
    
    elif menu == "👤 Profile":
        st.header("👤 My Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Personal Information")
            st.markdown(f"**Full Name:** {user['name']}")
            st.markdown(f"**Roll Number:** {user.get('roll_number', 'N/A')}")
            st.markdown(f"**Department:** {user.get('department', 'N/A')}")
            st.markdown(f"**Year:** {user.get('year', 'N/A')}")
        
        with col2:
            st.markdown("### Account Information")
            st.markdown(f"**Email:** {user.get('email', 'N/A')}")
            st.markdown(f"**Username:** {user['username']}")
            st.markdown(f"**Role:** {user['role'].title()}")
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 My Statistics")
        
        # Get counts
        registrations = db.get_user_registrations(user['username'])
        liked_events = db.get_user_likes(user['username'])
        interested_events = db.get_user_interested(user['username'])
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Events Registered", len(registrations))
        with col_stat2:
            st.metric("Events Liked", len(liked_events))
        with col_stat3:
            st.metric("Events Interested", len(interested_events))
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True, type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================
# FACULTY DASHBOARD
# ============================================

def faculty_dashboard():
    """Faculty dashboard"""
    user = st.session_state.user
    
    st.sidebar.title("👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**Name:** {user['name']}")
    if user.get('department'):
        st.sidebar.markdown(f"**Department:** {user['department']}")
    
    display_role_badge('faculty')
    
    # Navigation
    menu = st.sidebar.selectbox("Navigation", 
                               ["📊 Dashboard", "➕ Create Event", "📋 My Events", "👥 Registrations"])
    
    if menu == "📊 Dashboard":
        st.header("Faculty Dashboard")
        
        stats = db.get_statistics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Events", stats.get('total_events', 0))
        with col2:
            st.metric("Total Students", stats.get('total_students', 0))
        with col3:
            st.metric("Upcoming Events", stats.get('upcoming_events', 0))
        
        # Recent events
        st.subheader("📅 Recent Events")
        events = db.get_all_events()
        if events:
            for event in events[:3]:
                display_event_card(event, None)
    
    elif menu == "➕ Create Event":
        st.header("➕ Create New Event")
        
        with st.form("create_event_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Event Title *")
                event_type = st.selectbox("Event Type *", 
                                        ["Workshop", "Hackathon", "Seminar", "Conference", "Webinar"])
                event_date = st.date_input("Event Date *", min_value=date.today())
                event_time = st.time_input("Event Time *")
            
            with col2:
                venue = st.text_input("Venue *")
                organizer = st.text_input("Organizer *", value="G H Raisoni College")
                registration_link = st.text_input("Registration Link")
                max_participants = st.number_input("Max Participants", min_value=1, value=100)
            
            description = st.text_area("Event Description *", height=150)
            
            if st.form_submit_button("Create Event", use_container_width=True, type="primary"):
                if not all([title, event_type, venue, organizer, description]):
                    st.error("Please fill all required fields (*)")
                else:
                    # Combine date and time
                    event_datetime = datetime.combine(event_date, event_time)
                    
                    event_data = {
                        'title': title,
                        'description': description,
                        'event_type': event_type,
                        'event_date': event_datetime,
                        'venue': venue,
                        'organizer': organizer,
                        'registration_link': registration_link or '',
                        'max_participants': max_participants,
                        'created_by': user['username'],
                        'created_by_name': user['name']
                    }
                    
                    event_id = db.add_event(event_data)
                    if event_id:
                        st.success(f"Event '{title}' created successfully! 🎉")
                        st.rerun()
                    else:
                        st.error("Failed to create event")
    
    elif menu == "📋 My Events":
        st.header("📋 All Events")
        
        events = db.get_all_events()
        
        if events:
            for event in events:
                display_event_card(event, None)
        else:
            st.info("No events created yet.")
    
    elif menu == "👥 Registrations":
        st.header("📝 Event Registrations")
        
        events = db.get_all_events()
        
        if events:
            # Get unique event titles
            event_titles = list(set([e['title'] for e in events]))
            selected_title = st.selectbox("Select Event", event_titles)
            
            if selected_title:
                # Find the event
                event = next((e for e in events if e['title'] == selected_title), None)
                if event:
                    # Get registrations for this event
                    conn = db.get_connection()
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT u.name, u.roll_number, u.department, r.registered_at
                        FROM registrations r
                        JOIN users u ON r.username = u.username
                        WHERE r.event_id = ?
                    ''', (event['id'],))
                    
                    registrations = cursor.fetchall()
                    conn.close()
                    
                    if registrations:
                        # Display in table
                        reg_data = []
                        for reg in registrations:
                            reg_data.append({
                                'Student Name': reg[0],
                                'Roll No': reg[1],
                                'Department': reg[2],
                                'Registered On': format_date(reg[3])
                            })
                        
                        df = pd.DataFrame(reg_data)
                        st.dataframe(df, use_container_width=True)
                        
                        st.metric("Total Registrations", len(registrations))
                    else:
                        st.info(f"No registrations for '{selected_title}' yet.")
        else:
            st.info("No events to show registrations for.")
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True, type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================
# ADMIN DASHBOARD
# ============================================

def admin_dashboard():
    """Admin dashboard"""
    user = st.session_state.user
    
    st.sidebar.title("👑 Admin Panel")
    st.sidebar.markdown(f"**Name:** {user['name']}")
    
    display_role_badge('admin')
    
    # Navigation
    menu = st.sidebar.selectbox("Navigation", 
                               ["📊 Dashboard", "📅 Manage Events", "👥 Manage Users"])
    
    if menu == "📊 Dashboard":
        st.header("Admin Dashboard")
        
        stats = db.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", stats.get('total_events', 0))
        with col2:
            st.metric("Total Students", stats.get('total_students', 0))
        with col3:
            st.metric("Total Faculty", stats.get('total_faculty', 0))
        with col4:
            st.metric("Total Registrations", stats.get('total_registrations', 0))
    
    elif menu == "📅 Manage Events":
        st.header("Manage Events")
        
        events = db.get_all_events()
        
        if events:
            for event in events:
                with st.container():
                    col_view, col_actions = st.columns([3, 1])
                    
                    with col_view:
                        display_event_card(event, None)
                    
                    with col_actions:
                        if st.button("Delete", key=f"delete_{event['id']}", use_container_width=True):
                            conn = db.get_connection()
                            cursor = conn.cursor()
                            
                            # Delete from all related tables
                            cursor.execute('DELETE FROM events WHERE id = ?', (event['id'],))
                            cursor.execute('DELETE FROM likes WHERE event_id = ?', (event['id'],))
                            cursor.execute('DELETE FROM interested WHERE event_id = ?', (event['id'],))
                            cursor.execute('DELETE FROM registrations WHERE event_id = ?', (event['id'],))
                            
                            conn.commit()
                            conn.close()
                            
                            st.success(f"Event '{event['title']}' deleted!")
                            st.rerun()
        else:
            st.info("No events found.")
            if st.button("Add Sample Events", type="secondary"):
                add_sample_events()
                st.rerun()
    
    elif menu == "👥 Manage Users":
        st.header("Manage Users")
        
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users ORDER BY created_at DESC')
        users = cursor.fetchall()
        
        if users:
            user_data = []
            for user in users:
                user_data.append({
                    'Name': user['name'],
                    'Username': user['username'],
                    'Role': user['role'],
                    'Department': user.get('department', 'N/A'),
                    'Roll No': user.get('roll_number', 'N/A'),
                    'Joined': format_date(user['created_at'])
                })
            
            df = pd.DataFrame(user_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No users found.")
        
        conn.close()
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True, type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================
# SAMPLE EVENTS FUNCTION
# ============================================

def add_sample_events():
    """Add sample events for testing"""
    sample_events = [
        {
            'title': 'AI & Machine Learning Workshop',
            'description': 'Learn the fundamentals of AI and ML with hands-on projects.',
            'event_type': 'Workshop',
            'event_date': datetime.now() + timedelta(days=7),
            'venue': 'Seminar Hall',
            'organizer': 'CSE Department',
            'created_by': 'admin@raisoni',
            'created_by_name': 'Administrator'
        },
        {
            'title': 'Annual Coding Hackathon 2024',
            'description': '24-hour coding competition with cash prizes.',
            'event_type': 'Hackathon',
            'event_date': datetime.now() + timedelta(days=14),
            'venue': 'Computer Lab Block',
            'organizer': 'IT Department',
            'created_by': 'faculty@raisoni',
            'created_by_name': 'Faculty Coordinator'
        },
        {
            'title': 'Career Guidance Seminar',
            'description': 'Get expert advice on career paths and job opportunities.',
            'event_type': 'Seminar',
            'event_date': datetime.now() + timedelta(days=3),
            'venue': 'Auditorium',
            'organizer': 'Placement Cell',
            'created_by': 'admin@raisoni',
            'created_by_name': 'Administrator'
        }
    ]
    
    for event_data in sample_events:
        db.add_event(event_data)

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application"""
    
    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Route based on login status
    if not st.session_state.user:
        login_page()
    elif st.session_state.user['role'] == 'admin':
        admin_dashboard()
    elif st.session_state.user['role'] == 'faculty':
        faculty_dashboard()
    elif st.session_state.user['role'] == 'student':
        student_dashboard()

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()
