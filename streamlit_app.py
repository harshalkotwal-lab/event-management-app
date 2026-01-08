# streamlit_app.py
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

import logging
import time
from datetime import datetime

# Import modules
from config import *
from utils import get_custom_css, display_role_badge, format_date, get_event_status, save_flyer_image
from database import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize database
@st.cache_resource
def get_database():
    return DatabaseManager(use_supabase=USE_SUPABASE)

db = get_database()

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================
def display_event_card(event, current_user=None):
    """Display event card"""
    if not event:
        return
    
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            title = event.get('title', 'Untitled Event')
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
            
            # Event details
            event_date = event.get('event_date')
            st.markdown(get_event_status(event_date), unsafe_allow_html=True)
            st.caption(f"📅 {format_date(event_date)}")
            
            venue = event.get('venue', 'TBD')
            event_type = event.get('event_type', 'Event')
            st.caption(f"📍 {venue} | 🏷️ {event_type}")
            
            # Description
            desc = event.get('description', '')
            if desc:
                if len(desc) > 150:
                    with st.expander("📝 Description"):
                        st.write(desc)
                else:
                    st.caption(desc)
        
        with col2:
            # Flyer image
            flyer = event.get('flyer_path')
            if flyer and flyer.startswith('data:image'):
                st.image(flyer, use_column_width=True)
            
            # Engagement stats
            likes_count = db.get_event_likes_count(event['id'])
            st.caption(f"❤️ {likes_count} Likes")
            
            # Like button
            if current_user:
                is_liked = db.is_event_liked(event['id'], current_user)
                like_text = "❤️ Liked" if is_liked else "🤍 Like"
                
                if st.button(like_text, key=f"like_{event['id']}", use_container_width=True):
                    if is_liked:
                        db.remove_like(event['id'], current_user)
                    else:
                        db.add_like(event['id'], current_user)
                    st.rerun()
        
        # Registration for students
        if current_user and st.session_state.role == 'student':
            is_registered = db.get_registration(event['id'], current_user)
            
            if is_registered:
                st.success("✅ Registered")
            else:
                if st.button("📝 Register", key=f"reg_{event['id']}", use_container_width=True):
                    user = db.get_user(current_user)
                    reg_data = {
                        'event_id': event['id'],
                        'student_username': current_user,
                        'student_name': user.get('name', current_user)
                    }
                    success, message = db.add_registration(reg_data)
                    if success:
                        st.success("✅ Registered!")
                        st.rerun()
                    else:
                        st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PAGE FUNCTIONS
# ============================================
def login_page():
    """Login page"""
    st.markdown(f'<div class="college-header"><h2>{COLLEGE_CONFIG["name"]}</h2><p>Advanced Event Management System</p></div>', 
                unsafe_allow_html=True)
    
    st.subheader("🔐 Login")
    
    role = st.selectbox(
        "Select Role",
        ["Select Role", "Admin", "Faculty", "Student"],
        key="login_role"
    )
    
    if role != "Select Role":
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", use_container_width=True, type="primary"):
            if not username or not password:
                st.error("Please enter username and password")
            else:
                role_map = {
                    "Admin": "admin",
                    "Faculty": "faculty",
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
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error("Invalid credentials")
        
        # Student registration
        if role == "Student":
            st.markdown("---")
            if st.button("Create New Student Account", use_container_width=True, type="secondary"):
                st.session_state.page = "register"
                st.rerun()

def register_page():
    """Student registration page"""
    st.markdown('<div class="college-header"><h2>👨‍🎓 Student Registration</h2></div>', 
                unsafe_allow_html=True)
    
    with st.form("registration_form"):
        st.subheader("Create Account")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *")
            roll_no = st.text_input("Roll Number *")
            department = st.selectbox("Department *", COLLEGE_CONFIG['departments'])
            year = st.selectbox("Year *", COLLEGE_CONFIG['academic_years'])
        
        with col2:
            email = st.text_input("Email *")
            mobile = st.text_input("Mobile *")
            username = st.text_input("Username *")
            password = st.text_input("Password *", type="password")
        
        submit = st.form_submit_button("Register", use_container_width=True, type="primary")
        
        if submit:
            if not all([name, roll_no, email, mobile, username, password]):
                st.error("Please fill all required fields (*)")
            else:
                # Check if username exists
                existing = db.get_user(username)
                if existing:
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
                        'password': hashlib.sha256(password.encode()).hexdigest(),
                        'role': 'student'
                    }
                    
                    if db.add_user(user_data):
                        st.success("✅ Registration successful!")
                        st.info("Please login with your new credentials")
                        time.sleep(2)
                        st.session_state.page = "login"
                        st.rerun()
    
    if st.button("← Back to Login", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()

def student_dashboard():
    """Student dashboard"""
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('student')
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Events Feed", "My Registrations", "My Profile"])
    
    if page == "Events Feed":
        st.markdown('<h1 class="main-header">🎯 Discover Events</h1>', unsafe_allow_html=True)
        
        # Update event status
        db.update_event_status()
        
        # Get events
        events = db.get_all_events()
        
        if events:
            for event in events:
                display_event_card(event, st.session_state.username)
        else:
            st.info("No events found.")
    
    elif page == "My Registrations":
        st.header("📋 My Registrations")
        
        registrations = db.get_registrations_by_student(st.session_state.username)
        
        if registrations:
            for reg in registrations:
                with st.container():
                    st.markdown(f"**{reg.get('event_title', 'Event')}**")
                    st.caption(f"📅 {format_date(reg.get('event_date'))}")
                    st.caption(f"Status: {reg.get('status', 'pending').title()}")
        else:
            st.info("No registrations yet.")
    
    elif page == "My Profile":
        st.header("👤 My Profile")
        
        student = db.get_user(st.session_state.username)
        if student:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Name:** {student.get('name')}")
                st.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
                st.markdown(f"**Department:** {student.get('department', 'N/A')}")
            with col2:
                st.markdown(f"**Year:** {student.get('year', 'N/A')}")
                st.markdown(f"**Email:** {student.get('email', 'N/A')}")
                st.markdown(f"**Mobile:** {student.get('mobile', 'N/A')}")
    
    # Logout
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def faculty_dashboard():
    """Faculty dashboard"""
    st.sidebar.title("👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('faculty')
    
    page = st.sidebar.radio("Navigation", ["Dashboard", "Create Event", "My Events"])
    
    if page == "Dashboard":
        st.markdown('<h1 class="main-header">Faculty Dashboard</h1>', unsafe_allow_html=True)
        
        # Statistics
        events = db.get_events_by_creator(st.session_state.username)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("My Events", len(events))
        with col2:
            upcoming = len([e for e in events if e.get('status') == 'upcoming'])
            st.metric("Upcoming", upcoming)
    
    elif page == "Create Event":
        st.header("➕ Create Event")
        
        with st.form("create_event"):
            title = st.text_input("Event Title *")
            description = st.text_area("Description *")
            event_type = st.selectbox("Event Type *", COLLEGE_CONFIG['event_types'])
            event_date = st.date_input("Event Date *")
            venue = st.text_input("Venue *")
            
            col1, col2 = st.columns(2)
            with col1:
                organizer = st.text_input("Organizer *", value="G H Raisoni College")
                max_participants = st.number_input("Max Participants", value=100, min_value=1)
            with col2:
                flyer = st.file_uploader("Event Flyer", type=['jpg', 'png', 'jpeg'])
            
            submit = st.form_submit_button("Create Event", use_container_width=True, type="primary")
            
            if submit:
                if not all([title, description, venue, organizer]):
                    st.error("Please fill all required fields (*)")
                else:
                    flyer_path = save_flyer_image(flyer)
                    
                    event_data = {
                        'title': title,
                        'description': description,
                        'event_type': event_type,
                        'event_date': event_date.isoformat(),
                        'venue': venue,
                        'organizer': organizer,
                        'max_participants': max_participants,
                        'created_by': st.session_state.username,
                        'flyer_path': flyer_path
                    }
                    
                    if db.add_event(event_data):
                        st.success("✅ Event created successfully!")
                        st.rerun()
    
    elif page == "My Events":
        st.header("📋 My Events")
        
        events = db.get_events_by_creator(st.session_state.username)
        
        if events:
            for event in events:
                display_event_card(event, None)
        else:
            st.info("No events created yet.")
    
    # Logout
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def admin_dashboard():
    """Admin dashboard"""
    st.sidebar.title("👑 Admin Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('admin')
    
    page = st.sidebar.radio("Navigation", ["Dashboard", "Manage Events", "Manage Users", "System Stats"])
    
    if page == "Dashboard":
        st.markdown('<h1 class="main-header">Admin Dashboard</h1>', unsafe_allow_html=True)
        
        # Update event status
        db.update_event_status()
        
        # Get data
        events = db.get_all_events()
        users = db.get_all_users()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", len(events))
        with col2:
            st.metric("Total Users", len(users))
        with col3:
            admin_count = len([u for u in users if u['role'] == 'admin'])
            st.metric("Admins", admin_count)
        with col4:
            student_count = len([u for u in users if u['role'] == 'student'])
            st.metric("Students", student_count)
    
    elif page == "Manage Events":
        st.header("📋 Manage Events")
        
        events = db.get_all_events()
        
        if events:
            st.dataframe([
                {
                    'Title': e['title'],
                    'Type': e.get('event_type', 'N/A'),
                    'Date': format_date(e.get('event_date')),
                    'Venue': e.get('venue', 'N/A'),
                    'Status': e.get('status', 'N/A'),
                    'Created By': e.get('created_by', 'N/A')
                }
                for e in events
            ], use_container_width=True)
        else:
            st.info("No events found.")
    
    elif page == "Manage Users":
        st.header("👥 Manage Users")
        
        users = db.get_all_users()
        
        if users:
            st.dataframe([
                {
                    'Name': u['name'],
                    'Username': u['username'],
                    'Role': u['role'],
                    'Department': u.get('department', 'N/A'),
                    'Email': u.get('email', 'N/A')
                }
                for u in users
            ], use_container_width=True)
        else:
            st.info("No users found.")
    
    elif page == "System Stats":
        st.header("📊 System Statistics")
        
        stats = db.get_system_stats()
        
        if stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("User Statistics")
                if 'users' in stats:
                    for role, count in stats['users'].items():
                        st.metric(role.title(), count)
            
            with col2:
                st.subheader("Event Statistics")
                if 'total_events' in stats:
                    st.metric("Total Events", stats['total_events'])
                if 'ai_events' in stats:
                    st.metric("AI Events", stats['ai_events'])
                if 'registrations' in stats:
                    st.metric("Registrations", stats['registrations'])
    
    # Logout
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

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
    
    # Database info in sidebar
    if db.use_supabase:
        st.sidebar.success("✅ Using Supabase PostgreSQL")
    else:
        st.sidebar.info("💾 Using SQLite (Local)")
    
    # Show setup guide if Supabase is not configured
    if USE_SUPABASE and not db.use_supabase:
        with st.sidebar.expander("🚀 Setup Supabase", expanded=False):
            st.markdown("""
            ### Get Free PostgreSQL:
            1. Go to [supabase.com](https://supabase.com)
            2. Create free account and project
            3. Get URL and anon key from Settings > API
            
            ### Set Streamlit Secrets:
            ```toml
            [SUPABASE]
            url = "https://your-project.supabase.co"
            key = "your-anon-key"
            ```
            """)
    
    # Route based on page/role
    if st.session_state.page == "register":
        register_page()
    elif st.session_state.role is None:
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
