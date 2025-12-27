"""
G H Raisoni College - Event Management System
Simple ORM Version
"""

import streamlit as st

# ============================================
# PAGE CONFIGURATION - MUST BE FIRST
# ============================================

st.set_page_config(
    page_title="Raisoni Event Manager",
    page_icon="🎓",
    layout="wide"
)

# ============================================
# IMPORTS (SAFE AFTER set_page_config)
# ============================================

from datetime import datetime, date
import pandas as pd
import hashlib
import sys
import os
from pathlib import Path

# ============================================
# DATABASE SETUP
# ============================================

# Add current directory to path for database imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Initialize ORM availability flag
ORM_AVAILABLE = False
db_operations = None

try:
    # Check if database folder exists
    db_folder = current_dir / "database"
    if db_folder.exists():
        # Try to import database operations
        from database.operations import DatabaseOperations
        from database.connection import get_session, init_database
        
        ORM_AVAILABLE = True
        
        # Initialize database on first run
        try:
            # Check if we can connect to database
            with DatabaseOperations() as test_db:
                test_db.get_all_users()
            st.success("✅ Database connection successful!")
        except Exception as db_error:
            st.warning(f"⚠️ Database connection issue: {str(db_error)[:100]}")
    else:
        st.warning("⚠️ Database folder not found. Running in limited mode.")
except ImportError as e:
    st.warning(f"⚠️ ORM modules not available: {str(e)[:100]}")
except Exception as e:
    st.warning(f"⚠️ Database setup error: {str(e)[:100]}")

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
    
    .card-description {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0.75rem 0;
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
    
    .social-btn {
        padding: 0.5rem 1rem;
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
        gap: 0.5rem;
    }
    
    .social-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()

def format_date(dt):
    """Format datetime for display"""
    if isinstance(dt, datetime):
        return dt.strftime("%d %b %Y, %I:%M %p")
    elif isinstance(dt, str):
        try:
            return datetime.fromisoformat(dt.replace('Z', '+00:00')).strftime("%d %b %Y, %I:%M %p")
        except:
            return dt
    return str(dt)

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

def get_event_status(event_date):
    """Get event status badge"""
    try:
        if isinstance(event_date, datetime):
            dt = event_date
        elif isinstance(event_date, str):
            dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        else:
            return '<span class="status-badge">Unknown</span>'
        
        if dt > datetime.now():
            return '<span class="status-badge status-upcoming">🟢 Upcoming</span>'
        elif dt.date() == datetime.now().date():
            return '<span class="status-badge status-ongoing">🟡 Ongoing</span>'
        else:
            return '<span class="status-badge status-past">🔴 Past</span>'
    except:
        return '<span class="status-badge">Unknown</span>'

# ============================================
# EVENT DISPLAY FUNCTIONS
# ============================================

def display_event_card(event, current_user=None):
    """Display event card with social interactions"""
    if not event:
        return
    
    # Get user interactions if logged in
    user_liked = False
    user_interested = False
    
    if current_user and ORM_AVAILABLE:
        try:
            with DatabaseOperations() as db:
                user = db.get_user_by_username(current_user)
                if user:
                    interactions = db.get_user_interactions(event.id, user.id)
                    user_liked = interactions.get('like', False)
                    user_interested = interactions.get('interested', False)
        except:
            pass
    
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        # Event title and status
        col_title, col_status = st.columns([3, 1])
        with col_title:
            st.markdown(f'<div class="card-title">{event.title}</div>', unsafe_allow_html=True)
        with col_status:
            st.markdown(get_event_status(event.event_date), unsafe_allow_html=True)
        
        # Event details
        st.caption(f"📅 {format_date(event.event_date)} | 📍 {event.venue}")
        st.caption(f"🏷️ {event.event_type} | 👨‍🏫 {event.organizer}")
        
        # Description
        if event.description:
            desc = event.description
            if len(desc) > 150:
                with st.expander("📝 Description"):
                    st.write(desc)
                st.caption(f"{desc[:150]}...")
            else:
                st.write(desc)
        
        # SOCIAL INTERACTIONS
        col_like, col_int = st.columns(2)
        
        with col_like:
            like_icon = "❤️" if user_liked else "🤍"
            like_text = "Liked" if user_liked else "Like"
            like_count = event.like_count if hasattr(event, 'like_count') else 0
            
            if st.button(f"{like_icon} {like_text} ({like_count})", 
                        key=f"like_{event.id}_{current_user}",
                        use_container_width=True):
                if current_user and ORM_AVAILABLE:
                    try:
                        with DatabaseOperations() as db:
                            user = db.get_user_by_username(current_user)
                            if user:
                                db.add_social_interaction(event.id, user.id, 'like')
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)[:50]}")
        
        with col_int:
            int_icon = "👍" if user_interested else "🤔"
            int_text = "Interested" if user_interested else "Interested?"
            int_count = event.interested_count if hasattr(event, 'interested_count') else 0
            
            if st.button(f"{int_icon} {int_text} ({int_count})", 
                        key=f"int_{event.id}_{current_user}",
                        use_container_width=True):
                if current_user and ORM_AVAILABLE:
                    try:
                        with DatabaseOperations() as db:
                            user = db.get_user_by_username(current_user)
                            if user:
                                db.add_social_interaction(event.id, user.id, 'interested')
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)[:50]}")
        
        # Registration button
        if current_user and ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    user = db.get_user_by_username(current_user)
                    if user:
                        registrations = db.get_student_registrations(user.id)
                        is_registered = any(reg.event_id == event.id for reg in registrations)
                        
                        if not is_registered:
                            if st.button("📝 Register for Event", 
                                       key=f"reg_{event.id}_{current_user}",
                                       use_container_width=True,
                                       type="primary"):
                                reg_data = {
                                    'event_id': event.id,
                                    'student_id': user.id,
                                    'registration_status': 'pending',
                                    'registered_in_app': True
                                }
                                db.create_registration(reg_data)
                                st.success("✅ Registered successfully!")
                                st.rerun()
                        else:
                            st.success("✅ Already registered for this event")
            except Exception as e:
                st.error(f"Registration error: {str(e)[:50]}")
        
        # Creator info
        if hasattr(event, 'creator') and event.creator:
            st.caption(f"Created by: {event.creator.name}")
        
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
        
        admin_user = st.text_input("Username", key="admin_user", value="admin@raisoni")
        admin_pass = st.text_input("Password", type="password", key="admin_pass", value="admin123")
        
        if st.button("Admin Login", type="primary", use_container_width=True):
            # Default admin credentials
            if admin_user == "admin@raisoni" and admin_pass == "admin123":
                st.session_state.user = {"username": "admin@raisoni", "name": "Administrator", "role": "admin"}
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid admin credentials")
    
    with tab2:
        st.subheader("Faculty Login")
        
        faculty_user = st.text_input("Username", key="faculty_user", value="faculty@raisoni")
        faculty_pass = st.text_input("Password", type="password", key="faculty_pass", value="faculty123")
        
        if st.button("Faculty Login", type="primary", use_container_width=True):
            # Default faculty credentials
            if faculty_user == "faculty@raisoni" and faculty_pass == "faculty123":
                st.session_state.user = {"username": "faculty@raisoni", "name": "Faculty Coordinator", "role": "faculty"}
                st.success("Login successful!")
                st.rerun()
            elif ORM_AVAILABLE:
                try:
                    with DatabaseOperations() as db:
                        user = db.get_user_by_username(faculty_user)
                        if user and user.role == 'faculty':
                            # Simple password check (in production, use proper hashing)
                            if user.password_hash == hash_password(faculty_pass):
                                st.session_state.user = {
                                    "id": user.id,
                                    "username": user.username,
                                    "name": user.name,
                                    "role": user.role,
                                    "department": user.department
                                }
                                st.success("Login successful!")
                                st.rerun()
                            else:
                                st.error("Invalid password")
                        else:
                            st.error("Faculty not found")
                except:
                    st.error("Database error. Please try again.")
            else:
                st.error("Faculty not found")
    
    with tab3:
        st.subheader("Student Portal")
        
        login_tab, register_tab = st.tabs(["Login", "Register"])
        
        with login_tab:
            student_user = st.text_input("Username", key="student_login_user")
            student_pass = st.text_input("Password", type="password", key="student_login_pass")
            
            if st.button("Student Login", type="primary", use_container_width=True):
                if ORM_AVAILABLE:
                    try:
                        with DatabaseOperations() as db:
                            user = db.get_user_by_username(student_user)
                            if user and user.role == 'student':
                                if user.password_hash == hash_password(student_pass):
                                    st.session_state.user = {
                                        "id": user.id,
                                        "username": user.username,
                                        "name": user.name,
                                        "role": user.role,
                                        "department": user.department,
                                        "roll_number": user.roll_number,
                                        "year": user.year
                                    }
                                    st.success("Login successful!")
                                    st.rerun()
                                else:
                                    st.error("Invalid password")
                            else:
                                st.error("Student not found")
                    except Exception as e:
                        st.error(f"Login error: {str(e)[:100]}")
                else:
                    st.error("Database not available. Please contact administrator.")
        
        with register_tab:
            with st.form("student_registration_form"):
                st.markdown("### Create Student Account")
                
                name = st.text_input("Full Name *")
                roll_no = st.text_input("Roll Number *")
                department = st.selectbox("Department *", 
                                         ["CSE", "AIML", "IT", "EE", "BCA", "MCA", "BBA", "MBA", "EXTC", "MECH", "CIVIL", "Other"])
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
                    elif ORM_AVAILABLE:
                        try:
                            with DatabaseOperations() as db:
                                # Check if username exists
                                existing_user = db.get_user_by_username(username)
                                if existing_user:
                                    st.error("Username already exists")
                                else:
                                    user_data = {
                                        'username': username,
                                        'password_hash': hash_password(password),
                                        'name': name,
                                        'email': email,
                                        'role': 'student',
                                        'department': department,
                                        'roll_number': roll_no,
                                        'year': year,
                                        'is_active': True
                                    }
                                    user = db.create_user(user_data)
                                    if user:
                                        st.success("✅ Registration successful! Please login.")
                                        st.rerun()
                                    else:
                                        st.error("Registration failed")
                        except Exception as e:
                            st.error(f"Registration error: {str(e)[:100]}")
                    else:
                        st.error("Database not available. Please try again later.")

# ============================================
# STUDENT DASHBOARD
# ============================================

def student_dashboard():
    """Student dashboard"""
    user = st.session_state.user
    
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**Name:** {user.get('name', 'N/A')}")
    if 'roll_number' in user:
        st.sidebar.markdown(f"**Roll No:** {user['roll_number']}")
    if 'department' in user:
        st.sidebar.markdown(f"**Department:** {user['department']}")
    
    display_role_badge('student')
    
    # Navigation
    menu = st.sidebar.selectbox("Navigation", 
                               ["📅 Browse Events", "📋 My Registrations", "⭐ My Interests", "👤 Profile"])
    
    if menu == "📅 Browse Events":
        st.header("🎯 Discover Events")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            event_type = st.selectbox("Event Type", ["All", "Workshop", "Hackathon", "Competition", 
                                                    "Seminar", "Conference", "Webinar"])
        with col2:
            status_filter = st.selectbox("Status", ["All", "Upcoming", "Ongoing", "Past"])
        
        # Get events
        if ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    # Apply filters
                    filters = {}
                    if event_type != "All":
                        filters['event_type'] = event_type
                    if status_filter == "Upcoming":
                        filters['status'] = 'upcoming'
                    elif status_filter == "Ongoing":
                        filters['status'] = 'ongoing'
                    elif status_filter == "Past":
                        filters['status'] = 'past'
                    
                    events = db.get_all_events(filters)
                    
                    if events:
                        for event in events:
                            display_event_card(event, user['username'])
                    else:
                        st.info("No events found matching your criteria.")
            except Exception as e:
                st.error(f"Error loading events: {str(e)[:100]}")
        else:
            st.error("Database not available. Cannot load events.")
    
    elif menu == "📋 My Registrations":
        st.header("📋 My Registrations")
        
        if ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    user_obj = db.get_user_by_username(user['username'])
                    if user_obj:
                        registrations = db.get_student_registrations(user_obj.id)
                        
                        if registrations:
                            for reg in registrations:
                                event = db.get_event(reg.event_id)
                                if event:
                                    with st.container():
                                        st.markdown('<div class="event-card">', unsafe_allow_html=True)
                                        
                                        col1, col2 = st.columns([3, 1])
                                        
                                        with col1:
                                            st.markdown(f'<div class="card-title">{event.title}</div>', unsafe_allow_html=True)
                                            st.caption(f"📅 {format_date(event.event_date)} | 📍 {event.venue}")
                                            st.caption(f"Status: {reg.registration_status.title()} | Attendance: {reg.attendance_status.title()}")
                                        
                                        with col2:
                                            st.markdown(get_event_status(event.event_date), unsafe_allow_html=True)
                                        
                                        st.markdown('</div>', unsafe_allow_html=True)
                            # Show statistics
                            total = len(registrations)
                            upcoming = len([r for r in registrations 
                                          if hasattr(r, 'event') and r.event and r.event.event_date > datetime.now()])
                            
                            col_stat1, col_stat2 = st.columns(2)
                            with col_stat1:
                                st.metric("Total Registrations", total)
                            with col_stat2:
                                st.metric("Upcoming Events", upcoming)
                        else:
                            st.info("No registrations yet.")
                            if st.button("Browse Events", use_container_width=True):
                                st.session_state.menu = "📅 Browse Events"
                                st.rerun()
            except Exception as e:
                st.error(f"Error loading registrations: {str(e)[:100]}")
        else:
            st.error("Database not available.")
    
    elif menu == "⭐ My Interests":
        st.header("⭐ My Interests")
        
        if ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    user_obj = db.get_user_by_username(user['username'])
                    if user_obj:
                        # Get all events
                        all_events = db.get_all_events()
                        
                        # Separate liked and interested events
                        liked_events = []
                        interested_events = []
                        
                        for event in all_events:
                            interactions = db.get_user_interactions(event.id, user_obj.id)
                            if interactions.get('like'):
                                liked_events.append(event)
                            if interactions.get('interested'):
                                interested_events.append(event)
                        
                        # Create tabs
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
            except Exception as e:
                st.error(f"Error loading interests: {str(e)[:100]}")
        else:
            st.error("Database not available.")
    
    elif menu == "👤 Profile":
        st.header("👤 My Profile")
        
        if ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    user_obj = db.get_user_by_username(user['username'])
                    if user_obj:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Personal Information")
                            st.markdown(f"**Full Name:** {user_obj.name}")
                            st.markdown(f"**Roll Number:** {user_obj.roll_number}")
                            st.markdown(f"**Department:** {user_obj.department}")
                            st.markdown(f"**Year:** {user_obj.year}")
                        
                        with col2:
                            st.markdown("### Account Information")
                            st.markdown(f"**Email:** {user_obj.email}")
                            st.markdown(f"**Username:** {user_obj.username}")
                            st.markdown(f"**Member Since:** {format_date(user_obj.created_at)}")
                        
                        # Statistics
                        st.markdown("---")
                        st.subheader("📊 My Statistics")
                        
                        # Get actual data
                        registrations = db.get_student_registrations(user_obj.id)
                        all_events = db.get_all_events()
                        
                        liked_count = 0
                        interested_count = 0
                        for event in all_events:
                            interactions = db.get_user_interactions(event.id, user_obj.id)
                            if interactions.get('like'):
                                liked_count += 1
                            if interactions.get('interested'):
                                interested_count += 1
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("Events Registered", len(registrations))
                        with col_stat2:
                            st.metric("Events Liked", liked_count)
                        with col_stat3:
                            st.metric("Events Interested", interested_count)
            except Exception as e:
                st.error(f"Error loading profile: {str(e)[:100]}")
        else:
            # Show basic profile from session
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Personal Information")
                st.markdown(f"**Full Name:** {user.get('name', 'N/A')}")
                st.markdown(f"**Roll Number:** {user.get('roll_number', 'N/A')}")
                st.markdown(f"**Department:** {user.get('department', 'N/A')}")
                st.markdown(f"**Year:** {user.get('year', 'N/A')}")
            
            with col2:
                st.markdown("### Account Information")
                st.markdown(f"**Username:** {user.get('username', 'N/A')}")
    
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
    st.sidebar.markdown(f"**Name:** {user.get('name', 'N/A')}")
    if 'department' in user:
        st.sidebar.markdown(f"**Department:** {user['department']}")
    
    display_role_badge('faculty')
    
    if not ORM_AVAILABLE:
        st.error("⚠️ Database not available. Faculty features require database access.")
        st.info("Please check your database configuration.")
        return
    
    # Navigation
    menu = st.sidebar.selectbox("Navigation", 
                               ["📊 Dashboard", "➕ Create Event", "📋 My Events", "👥 Registrations"])
    
    if menu == "📊 Dashboard":
        st.header("Faculty Dashboard")
        
        if ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    # Get faculty's events
                    all_events = db.get_all_events({'created_by': user['username']})
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Events", len(all_events))
                    with col2:
                        upcoming = len([e for e in all_events if e.event_date > datetime.now()])
                        st.metric("Upcoming Events", upcoming)
                    with col3:
                        total_reg = 0
                        for event in all_events:
                            regs = db.get_event_registrations(event.id)
                            total_reg += len(regs)
                        st.metric("Total Registrations", total_reg)
                    
                    # Recent events
                    st.subheader("📅 My Recent Events")
                    if all_events:
                        for event in all_events[-3:]:
                            display_event_card(event, None)
                    else:
                        st.info("No events created yet.")
            except Exception as e:
                st.error(f"Error loading dashboard: {str(e)[:100]}")
    
    elif menu == "➕ Create Event":
        st.header("➕ Create New Event")
        
        with st.form("create_event_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Event Title *")
                event_type = st.selectbox("Event Type *", 
                                        ["Workshop", "Hackathon", "Competition", 
                                         "Seminar", "Conference", "Webinar"])
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
                elif ORM_AVAILABLE:
                    try:
                        # Combine date and time
                        event_datetime = datetime.combine(event_date, event_time)
                        
                        with DatabaseOperations() as db:
                            event_data = {
                                'title': title,
                                'description': description,
                                'event_type': event_type,
                                'event_date': event_datetime,
                                'venue': venue,
                                'organizer': organizer,
                                'registration_link': registration_link or None,
                                'max_participants': max_participants,
                                'created_by': user['username'],
                                'status': 'upcoming' if event_datetime > datetime.now() else 'ongoing',
                                'ai_generated': False
                            }
                            
                            event = db.create_event(event_data)
                            if event:
                                st.success(f"Event '{title}' created successfully! 🎉")
                                st.rerun()
                            else:
                                st.error("Failed to create event")
                    except Exception as e:
                        st.error(f"Error creating event: {str(e)[:100]}")
                else:
                    st.error("Database not available.")
    
    elif menu == "📋 My Events":
        st.header("📋 My Events")
        
        if ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    events = db.get_all_events({'created_by': user['username']})
                    
                    if events:
                        tab1, tab2, tab3 = st.tabs(["Upcoming", "Ongoing", "Past"])
                        
                        with tab1:
                            upcoming = [e for e in events if e.event_date > datetime.now()]
                            if upcoming:
                                for event in upcoming:
                                    display_event_card(event, None)
                            else:
                                st.info("No upcoming events.")
                        
                        with tab2:
                            ongoing = [e for e in events if e.event_date.date() == datetime.now().date()]
                            if ongoing:
                                for event in ongoing:
                                    display_event_card(event, None)
                            else:
                                st.info("No ongoing events.")
                        
                        with tab3:
                            past = [e for e in events if e.event_date < datetime.now()]
                            if past:
                                for event in past:
                                    display_event_card(event, None)
                            else:
                                st.info("No past events.")
                    else:
                        st.info("No events created yet.")
            except Exception as e:
                st.error(f"Error loading events: {str(e)[:100]}")
    
    elif menu == "👥 Registrations":
        st.header("📝 Event Registrations")
        
        if ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    events = db.get_all_events({'created_by': user['username']})
                    
                    if events:
                        event_titles = [e.title for e in events]
                        selected_title = st.selectbox("Select Event", event_titles)
                        
                        if selected_title:
                            selected_event = next(e for e in events if e.title == selected_title)
                            registrations = db.get_event_registrations(selected_event.id)
                            
                            if registrations:
                                # Display in table
                                reg_data = []
                                for reg in registrations:
                                    student = db.get_user_by_id(reg.student_id)
                                    reg_data.append({
                                        'Student Name': student.name if student else 'N/A',
                                        'Roll No': student.roll_number if student else 'N/A',
                                        'Department': student.department if student else 'N/A',
                                        'Status': reg.registration_status,
                                        'Attendance': reg.attendance_status,
                                        'Registered On': format_date(reg.registration_date)
                                    })
                                
                                df = pd.DataFrame(reg_data)
                                st.dataframe(df, use_container_width=True)
                                
                                # Summary
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Registrations", len(registrations))
                                with col2:
                                    present = len([r for r in registrations if r.attendance_status == 'present'])
                                    st.metric("Attended", present)
                            else:
                                st.info(f"No registrations for '{selected_title}' yet.")
                    else:
                        st.info("No events to show registrations for.")
            except Exception as e:
                st.error(f"Error loading registrations: {str(e)[:100]}")
    
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
    st.sidebar.title("👑 Admin Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.user.get('name', 'Administrator')}")
    
    display_role_badge('admin')
    
    # Navigation
    menu = st.sidebar.selectbox("Navigation", 
                               ["📊 Dashboard", "📅 Manage Events", "👥 Manage Users", "📈 Analytics"])
    
    if menu == "📊 Dashboard":
        st.header("Admin Dashboard")
        
        if ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    stats = db.get_system_statistics()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Events", stats.get('total_events', 0))
                    with col2:
                        st.metric("Total Students", stats.get('total_students', 0))
                    with col3:
                        st.metric("Total Faculty", stats.get('total_faculty', 0))
                    with col4:
                        st.metric("Upcoming Events", stats.get('upcoming_events', 0))
                    
                    # Recent activity
                    st.subheader("📈 System Overview")
                    
                    # User distribution
                    user_data = {
                        'Role': ['Students', 'Faculty'],
                        'Count': [stats.get('total_students', 0), stats.get('total_faculty', 0)]
                    }
                    user_df = pd.DataFrame(user_data)
                    st.bar_chart(user_df.set_index('Role'))
            except Exception as e:
                st.error(f"Error loading statistics: {str(e)[:100]}")
        else:
            st.warning("Database not available for statistics.")
    
    elif menu == "📅 Manage Events":
        st.header("Manage Events")
        
        if ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    events = db.get_all_events()
                    
                    if events:
                        for event in events:
                            with st.container():
                                col_view, col_actions = st.columns([3, 1])
                                
                                with col_view:
                                    display_event_card(event, None)
                                
                                with col_actions:
                                    if st.button("Delete", key=f"delete_{event.id}", use_container_width=True):
                                        st.warning(f"Delete functionality for '{event.title}' would go here")
                    else:
                        st.info("No events found.")
            except Exception as e:
                st.error(f"Error loading events: {str(e)[:100]}")
    
    elif menu == "👥 Manage Users":
        st.header("Manage Users")
        
        if ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    users = db.get_all_users()
                    
                    if users:
                        user_data = []
                        for user in users:
                            user_data.append({
                                'Name': user.name,
                                'Username': user.username,
                                'Role': user.role,
                                'Department': user.department or 'N/A',
                                'Active': 'Yes' if user.is_active else 'No',
                                'Joined': format_date(user.created_at)
                            })
                        
                        df = pd.DataFrame(user_data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No users found.")
            except Exception as e:
                st.error(f"Error loading users: {str(e)[:100]}")
    
    elif menu == "📈 Analytics":
        st.header("System Analytics")
        
        if ORM_AVAILABLE:
            try:
                with DatabaseOperations() as db:
                    stats = db.get_system_statistics()
                    
                    # Overall metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Events", stats.get('total_events', 0))
                    with col2:
                        st.metric("Total Users", stats.get('total_students', 0) + stats.get('total_faculty', 0))
                    with col3:
                        st.metric("Total Registrations", stats.get('total_registrations', 0))
                    
                    # Popular events
                    st.subheader("🎯 Popular Events")
                    popular = stats.get('popular_events', [])
                    if popular:
                        for event in popular:
                            st.write(f"**{event['title']}** - {event['count']} registrations")
                    else:
                        st.info("No popular events data available.")
            except Exception as e:
                st.error(f"Error loading analytics: {str(e)[:100]}")
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True, type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Route based on login status
    if not st.session_state.user:
        login_page()
    elif st.session_state.user.get('role') == 'admin':
        admin_dashboard()
    elif st.session_state.user.get('role') == 'faculty':
        faculty_dashboard()
    elif st.session_state.user.get('role') == 'student':
        student_dashboard()
    else:
        st.error("Unknown user role. Please logout and login again.")
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    main()
