"""
G H Raisoni College - Event Management System
Simple ORM Version
"""

import streamlit as st
from datetime import datetime, date
import pandas as pd
import hashlib
import sys
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
# REST OF YOUR IMPORTS AND CODE
# ============================================

# Add database path
sys.path.append(str(Path(__file__).parent))

# Try to import ORM
try:
    from database.operations import DatabaseOperations
    ORM_AVAILABLE = True
except:
    ORM_AVAILABLE = False
    st.error("Database modules not found!")

# Custom CSS - This comes AFTER set_page_config
st.markdown("""
<style>
    .event-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: white;
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()

def format_date(dt):
    """Format datetime"""
    if isinstance(dt, datetime):
        return dt.strftime("%d %b %Y, %I:%M %p")
    return str(dt)

# ============================================
# LOGIN PAGE
# ============================================

def login_page():
    """Simple login page"""
    st.title("🎓 G H Raisoni College")
    st.subheader("Event Management System")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", type="primary"):
            if ORM_AVAILABLE:
                with DatabaseOperations() as db:
                    user = db.get_user_by_username(username)
                    if user and user.password_hash == hash_password(password):
                        st.session_state.user = user
                        st.session_state.role = user.role
                        st.success(f"Welcome {user.name}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
    
    with tab2:
        with st.form("register_form"):
            name = st.text_input("Full Name")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["student", "faculty"])
            department = st.selectbox("Department", ["CSE", "AIML", "IT", "EE"])
            
            if st.form_submit_button("Register"):
                if ORM_AVAILABLE:
                    with DatabaseOperations() as db:
                        # Check if user exists
                        existing = db.get_user_by_username(username)
                        if existing:
                            st.error("Username already exists")
                        else:
                            user_data = {
                                'username': username,
                                'password_hash': hash_password(password),
                                'name': name,
                                'email': email,
                                'role': role,
                                'department': department,
                                'is_active': True
                            }
                            db.create_user(user_data)
                            st.success("Registration successful! Please login.")

# ============================================
# EVENT DISPLAY
# ============================================

def display_event_simple(event):
    """Simple event display"""
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">{event.title}</div>', unsafe_allow_html=True)
        st.caption(f"📅 {format_date(event.event_date)} | 📍 {event.venue}")
        st.write(event.description[:100] + "..." if len(event.description) > 100 else event.description)
        
        # Like and Interested buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"❤️ Like ({event.like_count})", key=f"like_{event.id}"):
                if 'user' in st.session_state:
                    with DatabaseOperations() as db:
                        db.add_social_interaction(
                            event.id, 
                            st.session_state.user.id, 
                            'like'
                        )
                        st.rerun()
        
        with col2:
            if st.button(f"👍 Interested ({event.interested_count})", key=f"int_{event.id}"):
                if 'user' in st.session_state:
                    with DatabaseOperations() as db:
                        db.add_social_interaction(
                            event.id, 
                            st.session_state.user.id, 
                            'interested'
                        )
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# STUDENT DASHBOARD
# ============================================

def student_dashboard():
    """Simple student dashboard"""
    user = st.session_state.user
    
    st.sidebar.title(f"👨‍🎓 {user.name}")
    st.sidebar.write(f"Department: {user.department}")
    
    menu = st.sidebar.selectbox("Menu", ["Browse Events", "My Registrations", "My Interests"])
    
    if menu == "Browse Events":
        st.header("📅 All Events")
        
        if ORM_AVAILABLE:
            with DatabaseOperations() as db:
                events = db.get_all_events()
                for event in events:
                    display_event_simple(event)
    
    elif menu == "My Registrations":
        st.header("📋 My Registrations")
        
        if ORM_AVAILABLE:
            with DatabaseOperations() as db:
                registrations = db.get_student_registrations(user.id)
                if registrations:
                    for reg in registrations:
                        event = db.get_event(reg.event_id)
                        if event:
                            st.write(f"**{event.title}**")
                            st.caption(f"Status: {reg.registration_status}")
                            st.divider()
                else:
                    st.info("No registrations yet.")
    
    elif menu == "My Interests":
        st.header("⭐ My Interests")
        
        if ORM_AVAILABLE:
            with DatabaseOperations() as db:
                # Get all events
                all_events = db.get_all_events()
                
                tab1, tab2 = st.tabs(["Liked Events", "Interested Events"])
                
                with tab1:
                    liked_events = []
                    for event in all_events:
                        interactions = db.get_user_interactions(event.id, user.id)
                        if interactions.get('like'):
                            liked_events.append(event)
                    
                    if liked_events:
                        for event in liked_events:
                            display_event_simple(event)
                    else:
                        st.info("No liked events yet.")
                
                with tab2:
                    interested_events = []
                    for event in all_events:
                        interactions = db.get_user_interactions(event.id, user.id)
                        if interactions.get('interested'):
                            interested_events.append(event)
                    
                    if interested_events:
                        for event in interested_events:
                            display_event_simple(event)
                    else:
                        st.info("No interested events yet.")
    
    # Logout
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================
# MAIN APP
# ============================================

def main():
    """Main app"""
    
    # Initialize session
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Check login
    if not st.session_state.user:
        login_page()
    else:
        if st.session_state.user.role == 'student':
            student_dashboard()
        elif st.session_state.user.role == 'faculty':
            st.title("Faculty Dashboard")
            st.write("Faculty features coming soon...")
        elif st.session_state.user.role == 'admin':
            st.title("Admin Dashboard")
            st.write("Admin features coming soon...")

if __name__ == "__main__":
    main()
