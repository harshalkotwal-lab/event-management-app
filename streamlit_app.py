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
import hashlib
import uuid
import base64
import os
from datetime import datetime, date, timedelta
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from config import *
from utils import get_custom_css, display_role_badge, format_date, get_event_status, save_flyer_image
from database import DatabaseManager

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
    """Display improved event card with flyer, mentor info, and registration links"""
    if not event or not event.get('id'):
        return
    
    event_id = event.get('id')
    
    # Create a container for the entire card
    card_container = st.container()
    
    with card_container:
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        # Create two-column layout: image on left, details on right
        col_img, col_info = st.columns([1, 3], gap="medium")
        
        with col_img:
            # Display event flyer if available
            flyer = event.get('flyer_path')
            if flyer and flyer.startswith('data:image'):
                try:
                    st.image(flyer, use_column_width=True)
                except:
                    # Fallback if image fails to load
                    st.markdown('<div style="width: 100%; height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; border-radius: 8px;"><span style="font-size: 32px; color: white;">🎯</span></div>', 
                               unsafe_allow_html=True)
            else:
                # Default placeholder
                st.markdown('<div style="width: 100%; height: 150px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; border-radius: 8px;"><span style="font-size: 32px; color: white;">🎯</span></div>', 
                           unsafe_allow_html=True)
        
        with col_info:
            # Header with title and badges - use HTML for layout
            title = event.get('title', 'Untitled Event')
            if len(title) > 60:
                title = title[:57] + "..."
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
            
            # Status and date row - using HTML layout instead of nested columns
            event_date = event.get('event_date')
            status_html = get_event_status(event_date)
            formatted_date = format_date(event_date)
            
            # Use markdown for status and date side by side
            st.markdown(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div>{status_html}</div>
                <div style="color: #666; font-size: 0.9rem;">📅 {formatted_date}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Event details - use single row
            venue = event.get('venue', 'TBD')
            if len(venue) > 25:
                venue = venue[:22] + "..."
            
            event_type = event.get('event_type', 'Event')
            max_participants = event.get('max_participants', 100)
            current_participants = event.get('current_participants', 0)
            
            st.caption(f"📍 {venue} | 🏷️ {event_type} | 👥 {current_participants}/{max_participants}")
            
            # Mentor information (if assigned)
            if event.get('mentor_id'):
                mentor = db.get_mentor_by_id(event['mentor_id'])
                if mentor:
                    st.markdown('<div style="background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%); padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid #E5E7EB; border-left: 3px solid #8B5CF6; font-size: 0.9rem;">', unsafe_allow_html=True)
                    st.markdown(f"**Mentor:** {mentor['full_name']} | **Contact:** {mentor['contact']}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Engagement metrics
            likes_count = db.get_event_likes_count(event_id)
            interested_count = db.get_event_interested_count(event_id)
            
            # Engagement row - use a separate container for buttons
            if current_user:
                # Create a container for engagement buttons
                engagement_container = st.container()
                with engagement_container:
                    # Use horizontal layout for buttons without creating nested columns
                    button_col1, button_col2, button_col3 = st.columns(3)
                    
                    with button_col1:
                        is_liked = db.is_event_liked(event_id, current_user)
                        like_text = "❤️ Liked" if is_liked else "🤍 Like"
                        like_type = "secondary" if is_liked else "primary"
                        
                        unique_key = f"like_{event_id}_{int(time.time() * 1000) % 10000}"
                        if st.button(like_text, key=unique_key, 
                                   use_container_width=True, type=like_type, 
                                   help="Like this event"):
                            if is_liked:
                                if db.remove_like(event_id, current_user):
                                    st.rerun()
                            else:
                                if db.add_like(event_id, current_user):
                                    st.rerun()
                    
                    with button_col2:
                        is_interested = db.is_event_interested(event_id, current_user)
                        interested_text = "⭐ Interested" if is_interested else "☆ Interested"
                        interested_type = "secondary" if is_interested else "primary"
                        
                        unique_key_interested = f"interested_{event_id}_{int(time.time() * 1000) % 10000}"
                        if st.button(interested_text, key=unique_key_interested, 
                                    use_container_width=True, type=interested_type,
                                    help="Mark as interested"):
                            if is_interested:
                                if db.remove_interested(event_id, current_user):
                                    st.rerun()
                            else:
                                if db.add_interested(event_id, current_user):
                                    st.rerun()
                    
                    with button_col3:
                        # Share button to promote the app
                        unique_key_share = f"share_{event_id}_{int(time.time() * 1000) % 10000}"
                        if st.button("📤 Share", key=unique_key_share,
                                   use_container_width=True, type="secondary",
                                   help="Share this event with friends"):
                            # Create a share message
                            event_title = event.get('title', 'Cool Event')
                            share_text = f"Check out '{event_title}' at G H Raisoni College Event Manager! 🎓\n\nJoin the platform to discover more events: [Event Manager App]"
                            
                            # Copy to clipboard
                            st.code(share_text)
                            st.success("📋 Share message copied! Share with your friends.")
            
            # Show engagement counts
            st.caption(f"❤️ {likes_count} Likes | ⭐ {interested_count} Interested")
            
            # Event links (if available)
            event_link = event.get('event_link', '')
            registration_link = event.get('registration_link', '')
            
            if event_link or registration_link:
                with st.expander("🔗 Event Links", expanded=False):
                    if event_link:
                        st.markdown(f"**🌐 Event Page:** [Click here]({event_link})")
                    if registration_link:
                        st.markdown(f"**📝 Registration:** [Click here]({registration_link})")
            
            # Description (collapsible)
            desc = event.get('description', '')
            if desc:
                if len(desc) > 150:
                    with st.expander("📝 Description", expanded=False):
                        st.write(desc)
                else:
                    st.caption(desc[:150] + "..." if len(desc) > 150 else desc)
        
        # ============================================
        # REGISTRATION SECTION (For students only)
        # ============================================
        if current_user and st.session_state.role == 'student':
            st.markdown('<div style="background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); padding: 8px; border-radius: 6px; margin-top: 8px; border-left: 3px solid #3B82F6; font-size: 0.9rem;">', unsafe_allow_html=True)
            
            is_registered = db.is_student_registered(event_id, current_user)
            
            if is_registered:
                st.success("✅ You are already registered for this event")
                
                # Show "I Have Registered Externally" button
                if registration_link:
                    unique_key_ext_reg = f"ext_reg_{event_id}_{int(time.time() * 1000) % 10000}"
                    if st.button("✅ I Have Registered Externally", 
                               key=unique_key_ext_reg,
                               use_container_width=True,
                               type="secondary",
                               help="Mark that you have registered externally"):
                        # Update registration status
                        if db.use_supabase:
                            success = db.client.update('registrations', 
                                                     {'event_id': event_id, 'student_username': current_user},
                                                     {'status': 'confirmed', 'updated_at': datetime.now().isoformat()})
                        else:
                            cursor = db.client.conn.cursor()
                            cursor.execute("UPDATE registrations SET status = 'confirmed', updated_at = ? WHERE event_id = ? AND student_username = ?",
                                         (datetime.now().isoformat(), event_id, current_user))
                            db.client.conn.commit()
                            success = cursor.rowcount > 0
                        
                        if success:
                            st.success("✅ External registration recorded!")
                            st.rerun()
            else:
                # Registration options
                reg_col1, reg_col2 = st.columns([1, 1])
                
                with reg_col1:
                    # Register in App button
                    unique_key_app_reg = f"app_reg_{event_id}_{int(time.time() * 1000) % 10000}"
                    if st.button("📱 Register in App", 
                                key=unique_key_app_reg,
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
                            reg_id, message = db.add_registration(reg_data)
                            if reg_id:
                                st.success("✅ Registered in college system!")
                                st.rerun()
                            else:
                                st.error(message)
                
                with reg_col2:
                    # External registration link button (if available)
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
    """Enhanced Student dashboard matching streamlit_app (10).py"""
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    
    # Get student info
    student = db.get_user(st.session_state.username)
    if student:
        st.sidebar.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
        st.sidebar.markdown(f"**Department:** {student.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Year:** {student.get('year', 'N/A')}")
        mobile = student.get('mobile', 'Not provided')
        st.sidebar.markdown(f"**Mobile:** {mobile}")
    
    display_role_badge('student')
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Events Feed", "My Registrations", "Liked Events", "Interested Events", "My Profile"]
        
        if 'student_page' not in st.session_state:
            st.session_state.student_page = "Events Feed"
        
        for option in nav_options:
            is_active = st.session_state.student_page == option
            button_class = "active" if is_active else ""
            button_text = f"▶ {option}" if is_active else option
            
            if st.button(button_text, key=f"student_{option}", use_container_width=True):
                st.session_state.student_page = option
                st.rerun()
        
        # Engagement statistics
        st.markdown("---")
        st.markdown("### My Engagement")
        
        # Get counts
        liked_events = db.get_student_liked_events(st.session_state.username)
        interested_events = db.get_student_interested_events(st.session_state.username)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("❤️ Liked", len(liked_events))
        with col_stat2:
            st.metric("⭐ Interested", len(interested_events))
        
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
        col_filters = st.columns([2, 1, 1, 1])
        with col_filters[0]:
            search = st.text_input("🔍 Search events", placeholder="Search by title, description...")
        with col_filters[1]:
            event_type = st.selectbox("Type", ["All", "Workshop", "Hackathon", "Competition", 
                                              "Bootcamp", "Seminar", "Conference", "Webinar"])
        with col_filters[2]:
            show_only = st.selectbox("Show", ["All", "Upcoming", "Ongoing", "Past"])
        with col_filters[3]:
            ai_only = st.checkbox("🤖 AI-Generated")
        
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
        
        if ai_only:
            filtered_events = [e for e in filtered_events if e.get('ai_generated')]
        
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

    elif selected == "Liked Events":
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
            if st.button("Browse Events", use_container_width=True, type="primary"):
                st.session_state.student_page = "Events Feed"
                st.rerun()
            return
        
        # Filter tabs
        tab1, tab2, tab3 = st.tabs(["All Liked", "Upcoming", "Past"])
        
        with tab1:
            st.caption(f"Total liked events: {len(liked_events)}")
            for event in liked_events:
                display_event_card(event, st.session_state.username)
        
        with tab2:
            upcoming = [e for e in liked_events if e.get('status') == 'upcoming']
            if upcoming:
                st.caption(f"Upcoming liked events: {len(upcoming)}")
                for event in upcoming:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("No upcoming liked events.")
        
        with tab3:
            past = [e for e in liked_events if e.get('status') == 'past']
            if past:
                st.caption(f"Past liked events: {len(past)}")
                for event in past:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("No past liked events.")
    
    elif selected == "Interested Events":
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
            if st.button("Browse Events", use_container_width=True, type="primary"):
                st.session_state.student_page = "Events Feed"
                st.rerun()
            return
        
        # Filter tabs
        tab1, tab2, tab3 = st.tabs(["All Interested", "Upcoming", "Past"])
        
        with tab1:
            st.caption(f"Total interested events: {len(interested_events)}")
            for event in interested_events:
                display_event_card(event, st.session_state.username)
        
        with tab2:
            upcoming = [e for e in interested_events if e.get('status') == 'upcoming']
            if upcoming:
                st.caption(f"Upcoming interested events: {len(upcoming)}")
                for event in upcoming:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("No upcoming interested events.")
        
        with tab3:
            past = [e for e in interested_events if e.get('status') == 'past']
            if past:
                st.caption(f"Past interested events: {len(past)}")
                for event in past:
                    display_event_card(event, st.session_state.username)
            else:
                st.info("No past interested events.")
    
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
            st.markdown("### Contact Information")
            st.markdown(f"**Email:** {student.get('email', 'N/A')}")
            mobile = student.get('mobile', 'Not provided')
            st.markdown(f"**Mobile:** {mobile}")
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

def faculty_dashboard():
    """Enhanced Faculty dashboard matching streamlit_app (10).py"""
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
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
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
        
        tab1, tab2 = st.tabs(["📝 Manual Entry", "🤖 AI Generator"])
        
        with tab1:
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
                    event_link = st.text_input("Event Website/URL", 
                                             placeholder="https://example.com/event-details")
                    registration_link = st.text_input("Registration Link", 
                                                    placeholder="https://forms.google.com/registration")
                    
                    # Mentor selection
                    st.subheader("👨‍🏫 Assign Mentor (Optional)")
                    active_mentors = db.get_active_mentors()
                    if active_mentors:
                        mentor_options = ["None"] + [f"{m['full_name']} ({m['department']})" for m in active_mentors]
                        selected_mentor = st.selectbox("Select Mentor", mentor_options)
                    else:
                        st.info("No active mentors available. Admin can add mentors.")
                        selected_mentor = "None"
                    
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
                        # Get mentor ID if selected
                        mentor_id = None
                        if selected_mentor != "None" and active_mentors:
                            mentor_name = selected_mentor.split(" (")[0]
                            mentor = next((m for m in active_mentors if m['full_name'] == mentor_name), None)
                            if mentor:
                                mentor_id = mentor['id']
                        
                        # Save flyer
                        flyer_path = save_flyer_image(flyer)
                        
                        # Combine date and time
                        event_datetime = datetime.combine(event_date, event_time)
                        
                        event_data = {
                            'title': title,
                            'description': description,
                            'event_type': event_type,
                            'event_date': event_datetime.isoformat(),
                            'venue': venue,
                            'organizer': organizer,
                            'event_link': event_link,
                            'registration_link': registration_link,
                            'max_participants': max_participants,
                            'flyer_path': flyer_path,
                            'created_by': st.session_state.username,
                            'created_by_name': st.session_state.name,
                            'ai_generated': False,
                            'mentor_id': mentor_id
                        }
                        
                        if db.add_event(event_data):
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
            
            # Text input for AI processing
            event_text = st.text_area("Paste event text here:", 
                             placeholder="Example: Join us for a Python Workshop on 15th Dec 2023 at Seminar Hall. Organized by CSE Department...",
                             height=200,
                             key="ai_text_input")
            
            if st.button("🤖 Generate Event with AI", use_container_width=True, type="primary", key="ai_generate_btn"):
                if event_text:
                    # Note: AI Event Generator needs to be imported/implemented in config or utils
                    st.warning("AI Event Generator feature requires OpenAI API key setup.")
                    st.info("Please use manual entry for now, or setup OpenAI API in config.")
            
            # Display and edit AI-generated event (placeholder)
            if 'ai_generated_event' in st.session_state:
                event_data = st.session_state.ai_generated_event
                
                st.markdown("---")
                st.subheader("✏️ Review & Edit AI-Generated Event")
                
                with st.form("ai_event_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        ai_title = st.text_input("Event Title", value=event_data.get('title', ''))
                        ai_event_type = st.selectbox("Event Type", 
                                                   ["Workshop", "Hackathon", "Competition", 
                                                    "Bootcamp", "Seminar", "Conference", "Webinar"],
                                                   index=0)
                        ai_date = st.date_input("Event Date", value=date.today(), min_value=date.today())
                        ai_time = st.time_input("Event Time", value=datetime.now().time())
                        ai_max_participants = st.number_input("Max Participants", min_value=1, value=100)
                    
                    with col2:
                        ai_venue = st.text_input("Venue", value="G H Raisoni College")
                        ai_organizer = st.text_input("Organizer", value="G H Raisoni College")
                        ai_event_link = st.text_input("Event Website", value="")
                        ai_reg_link = st.text_input("Registration Link", value="")
                        
                        # Mentor selection for AI-generated events
                        st.subheader("👨‍🏫 Assign Mentor (Optional)")
                        active_mentors = db.get_active_mentors()
                        if active_mentors:
                            mentor_options = ["None"] + [f"{m['full_name']} ({m['department']})" for m in active_mentors]
                            ai_selected_mentor = st.selectbox("Select Mentor", mentor_options, key="ai_mentor_select")
                        else:
                            st.info("No active mentors available. Admin can add mentors.")
                            ai_selected_mentor = "None"
                    
                    ai_description = st.text_area("Event Description", 
                                                value="Event description will appear here...",
                                                height=150)
                    
                    # Flyer upload for AI events
                    st.subheader("Event Flyer (Optional)")
                    ai_flyer = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png', 'gif', 'webp'], key="ai_flyer")
                    if ai_flyer:
                        st.image(ai_flyer, width=200)
                    
                    ai_submit = st.form_submit_button("✅ Create AI-Generated Event", use_container_width=True)
                    
                    if ai_submit:
                        if not all([ai_title, ai_venue, ai_organizer, ai_description]):
                            st.error("Please fill all required fields (*)")
                        else:
                            # Get mentor ID if selected
                            ai_mentor_id = None
                            if ai_selected_mentor != "None" and active_mentors:
                                mentor_name = ai_selected_mentor.split(" (")[0]
                                mentor = next((m for m in active_mentors if m['full_name'] == mentor_name), None)
                                if mentor:
                                    ai_mentor_id = mentor['id']
                            
                            # Save flyer
                            flyer_path = save_flyer_image(ai_flyer)
                            
                            # Combine date and time
                            event_datetime = datetime.combine(ai_date, ai_time)
                            
                            final_event_data = {
                                'title': ai_title,
                                'description': ai_description,
                                'event_type': ai_event_type,
                                'event_date': event_datetime.isoformat(),
                                'venue': ai_venue,
                                'organizer': ai_organizer,
                                'event_link': ai_event_link,
                                'registration_link': ai_reg_link,
                                'max_participants': ai_max_participants,
                                'flyer_path': flyer_path,
                                'created_by': st.session_state.username,
                                'created_by_name': st.session_state.name,
                                'ai_generated': True,
                                'mentor_id': ai_mentor_id
                            }
                            
                            if db.add_event(final_event_data):
                                st.success(f"✅ AI-generated event '{ai_title}' created successfully! 🎉")
                                if ai_mentor_id:
                                    st.info(f"✅ Mentor assigned: {ai_selected_mentor}")
                                
                                # Clear session state
                                if 'ai_generated_event' in st.session_state:
                                    del st.session_state.ai_generated_event
                                
                                st.rerun()
                            else:
                                st.error("Failed to create event")
    
    elif selected == "My Events":
        st.header("📋 My Events")
        
        events = db.get_events_by_creator(st.session_state.username)
        
        if not events:
            st.info("You haven't created any events yet.")
            return
        
        # Show engagement statistics
        st.subheader("📊 Event Engagement")
        total_likes = sum(db.get_event_likes_count(e['id']) for e in events)
        total_interested = sum(db.get_event_interested_count(e['id']) for e in events)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Likes", total_likes)
        with col2:
            st.metric("Total Interested", total_interested)
            
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
            
            # Get registrations for the selected event
            registrations = db.get_registrations_by_event(event_id)
            
            st.info(f"📊 Registrations for: **{selected_title}**")
            st.caption(f"Total Registrations: {len(registrations)}")
            
            if registrations:
                # Display registrations in a table with mobile numbers
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
                    
                    # Export option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"registrations_{selected_title.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No registrations for this event yet.")

def admin_dashboard():
    """Enhanced Admin dashboard matching streamlit_app (10).py"""
    st.sidebar.title("👑 Admin Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('admin')
    
    # Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        nav_options = ["Dashboard", "Manage Events", "Manage Users", "Manage Mentors"]
        
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
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
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
        users = db.get_all_users()
        mentors = db.get_all_mentors()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", len(events))
        with col2:
            upcoming = len([e for e in events if e.get('status') == 'upcoming'])
            st.metric("Upcoming Events", upcoming)
        with col3:
            ai_events = len([e for e in events if e.get('ai_generated')])
            st.metric("🤖 AI Events", ai_events)
        with col4:
            active_mentors = len([m for m in mentors if m.get('is_active')])
            st.metric("👨‍🏫 Active Mentors", active_mentors)
        
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
                # Create a container for each event
                with st.container():
                    # Use columns for layout
                    col_view, col_actions = st.columns([3, 1])
                    
                    with col_view:
                        # Display event card in the view column
                        display_event_card(event, None)
                    
                    with col_actions:
                        st.markdown("### Actions")
                        if st.button("Delete", key=f"delete_{event['id']}", use_container_width=True, type="secondary"):
                            # Delete event from database
                            if db.use_supabase:
                                # Delete registrations first
                                db.client.delete('registrations', {'event_id': event['id']})
                                # Delete likes
                                db.client.delete('event_likes', {'event_id': event['id']})
                                # Delete interested
                                db.client.delete('event_interested', {'event_id': event['id']})
                                # Delete event
                                success = db.client.delete('events', {'id': event['id']})
                            else:
                                cursor = db.client.conn.cursor()
                                try:
                                    # First delete registrations
                                    cursor.execute("DELETE FROM registrations WHERE event_id = ?", (event['id'],))
                                    # Delete likes
                                    cursor.execute("DELETE FROM event_likes WHERE event_id = ?", (event['id'],))
                                    # Delete interested
                                    cursor.execute("DELETE FROM event_interested WHERE event_id = ?", (event['id'],))
                                    # Then delete event
                                    cursor.execute("DELETE FROM events WHERE id = ?", (event['id'],))
                                    db.client.conn.commit()
                                    success = cursor.rowcount > 0
                                except Exception as e:
                                    st.error(f"Error deleting event: {e}")
                                    success = False
                            
                            if success:
                                st.success("Event deleted successfully!")
                                st.rerun()
        else:
            st.info("No events found.")
    
    elif selected == "Manage Users":
        st.header("👥 Manage Users")
        
        # Get all users
        if db.use_supabase:
            users = db.client.select('users')
        else:
            cursor = db.client.conn.cursor()
            cursor.execute("SELECT * FROM users ORDER BY created_at DESC")
            users = [dict(row) for row in cursor.fetchall()]
        
        if users:
            # Display user statistics
            admin_count = len([u for u in users if u['role'] == 'admin'])
            faculty_count = len([u for u in users if u['role'] == 'faculty'])
            student_count = len([u for u in users if u['role'] == 'student'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Admins", admin_count)
            with col2:
                st.metric("Faculty", faculty_count)
            with col3:
                st.metric("Students", student_count)
            with col4:
                total_users = len(users)
                st.metric("Total Users", total_users)
            
            # User table with mobile numbers
            df_data = []
            for user in users:
                df_data.append({
                    'Name': user.get('name'),
                    'Username': user.get('username'),
                    'Role': user.get('role').title(),
                    'Department': user.get('department', 'N/A'),
                    'Roll No': user.get('roll_no', 'N/A'),
                    'Mobile': user.get('mobile', 'N/A'),
                    'Created': format_date(user.get('created_at')),
                    'Status': 'Active' if user.get('is_active', True) else 'Inactive'
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                # User actions
                st.subheader("User Actions")
                user_options = {f"{user['name']} ({user['username']})": user['id'] for user in users}
                selected_user = st.selectbox("Select User", list(user_options.keys()))
                
                if selected_user:
                    user_id = user_options[selected_user]
                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        if st.button("Reset Password", use_container_width=True):
                            default_pass = hashlib.sha256('password123'.encode()).hexdigest()
                            if db.use_supabase:
                                success = db.client.update('users', {'id': user_id}, {'password': default_pass})
                            else:
                                cursor = db.client.conn.cursor()
                                cursor.execute("UPDATE users SET password = ? WHERE id = ?", (default_pass, user_id))
                                db.client.conn.commit()
                                success = cursor.rowcount > 0
                            
                            if success:
                                st.success("Password reset to 'password123'")
                    with col_act2:
                        if st.button("Delete User", use_container_width=True, type="secondary"):
                            # Get user info
                            selected_user_data = next(u for u in users if u['id'] == user_id)
                            
                            # Don't allow deleting default admin and faculty
                            if selected_user_data['username'] in ['admin@raisoni', 'faculty@raisoni']:
                                st.error("Cannot delete default admin/faculty accounts")
                            else:
                                if db.use_supabase:
                                    # Delete user
                                    success = db.client.delete('users', {'id': user_id})
                                else:
                                    cursor = db.client.conn.cursor()
                                    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
                                    db.client.conn.commit()
                                    success = cursor.rowcount > 0
                                
                                if success:
                                    st.success("User deleted successfully!")
                                    st.rerun()
        else:
            st.info("No users found.")
    
    elif selected == "Manage Mentors":
        st.header("👨‍🏫 Manage Mentors")
        
        tab1, tab2, tab3 = st.tabs(["Add New Mentor", "View All Mentors", "Assign to Events"])
        
        with tab1:
            st.subheader("➕ Add New Mentor")
            
            with st.form("add_mentor_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    first_name = st.text_input("First Name *")
                    last_name = st.text_input("Last Name *")
                    department = st.selectbox("Department *", COLLEGE_CONFIG['departments'])
                    email = st.text_input("Email *", help="This will be the username for login")
                
                with col2:
                    contact = st.text_input("Contact Number *")
                    expertise = st.text_area("Expertise/Areas", placeholder="Python, Machine Learning, Web Development...")
                    is_active = st.checkbox("Active", value=True)
                    
                    # Password options
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
                        
                        # Add custom password if provided
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
        
        with tab2:
            st.subheader("📋 All Mentors")
            
            mentors = db.get_all_mentors()
            
            if not mentors:
                st.info("No mentors found. Add your first mentor!")
                return
            
            # Search and filter
            col_search, col_filter = st.columns(2)
            with col_search:
                search_term = st.text_input("🔍 Search mentors", placeholder="Search by name, department...")
            
            with col_filter:
                show_active = st.selectbox("Status", ["All", "Active Only", "Inactive Only"])
            
            # Filter mentors
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
            
            # Display mentors
            st.caption(f"Found {len(filtered_mentors)} mentors")
            
            for mentor in filtered_mentors:
                with st.container():
                    st.markdown('<div class="event-card">', unsafe_allow_html=True)
                    
                    col_info, col_actions = st.columns([3, 1])
                    
                    with col_info:
                        # Mentor status badge
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
                        
                        # Edit button
                        if st.button("✏️ Edit", key=f"edit_{mentor['id']}", use_container_width=True):
                            st.session_state.editing_mentor = mentor['id']
                            st.rerun()
                        
                        # Delete/Activate button
                        if mentor.get('is_active'):
                            if st.button("❌ Deactivate", key=f"deact_{mentor['id']}", use_container_width=True, type="secondary"):
                                if db.delete_mentor(mentor['id']):
                                    st.success("Mentor deactivated!")
                                    st.rerun()
                        else:
                            if st.button("✅ Activate", key=f"act_{mentor['id']}", use_container_width=True, type="secondary"):
                                if db.update_mentor(mentor['id'], {'is_active': True}):
                                    st.success("Mentor activated!")
                                    st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Edit mentor form (appears when editing)
            if 'editing_mentor' in st.session_state:
                mentor_id = st.session_state.editing_mentor
                mentor = db.get_mentor_by_id(mentor_id)
                
                if mentor:
                    st.markdown("---")
                    st.subheader("✏️ Edit Mentor")
                    
                    with st.form("edit_mentor_form"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            edit_first_name = st.text_input("First Name", value=mentor.get('first_name', ''))
                            edit_last_name = st.text_input("Last Name", value=mentor.get('last_name', ''))
                            edit_department = st.selectbox("Department", COLLEGE_CONFIG['departments'], 
                                                         index=COLLEGE_CONFIG['departments'].index(mentor.get('department', '')) 
                                                         if mentor.get('department') in COLLEGE_CONFIG['departments'] else 0)
                        
                        with col2:
                            edit_email = st.text_input("Email", value=mentor.get('email', ''))
                            edit_contact = st.text_input("Contact", value=mentor.get('contact', ''))
                            edit_expertise = st.text_area("Expertise", value=mentor.get('expertise', ''))
                            edit_active = st.checkbox("Active", value=bool(mentor.get('is_active', True)))
                        
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            save = st.form_submit_button("💾 Save Changes", use_container_width=True, type="primary")
                        with col_cancel:
                            cancel = st.form_submit_button("❌ Cancel", use_container_width=True, type="secondary")
                        
                        if save:
                            update_data = {
                                'first_name': edit_first_name,
                                'last_name': edit_last_name,
                                'department': edit_department,
                                'email': edit_email,
                                'contact': edit_contact,
                                'expertise': edit_expertise,
                                'is_active': edit_active
                            }
                            
                            if db.update_mentor(mentor_id, update_data):
                                st.success("✅ Mentor updated successfully!")
                                del st.session_state.editing_mentor
                                st.rerun()
                            else:
                                st.error("Failed to update mentor.")
                        
                        if cancel:
                            del st.session_state.editing_mentor
                            st.rerun()
        
        with tab3:
            st.subheader("📅 Assign Mentors to Events")
            
            # Get all active mentors
            active_mentors = db.get_active_mentors()
            if not active_mentors:
                st.info("No active mentors available. Please add mentors first.")
                return
            
            # Get all events without mentors
            events = db.get_all_events()
            events_without_mentors = [e for e in events if not e.get('mentor_id')]
            
            if not events_without_mentors:
                st.success("🎉 All events have mentors assigned!")
                st.info("To reassign mentors, go to Faculty dashboard.")
                return
            
            # Select event to assign mentor
            event_options = {f"{e['title']} ({format_date(e['event_date'])})": e['id'] for e in events_without_mentors}
            selected_event_label = st.selectbox("Select Event (without mentor)", list(event_options.keys()))
            
            if selected_event_label:
                event_id = event_options[selected_event_label]
                selected_event = next(e for e in events_without_mentors if e['id'] == event_id)
                
                # Display event details
                st.markdown(f"**Selected Event:** {selected_event['title']}")
                st.caption(f"Date: {format_date(selected_event['event_date'])}")
                st.caption(f"Type: {selected_event.get('event_type', 'N/A')}")
                st.caption(f"Venue: {selected_event.get('venue', 'N/A')}")
                
                # Select mentor
                mentor_options = {f"{m['full_name']} ({m['department']})": m['id'] for m in active_mentors}
                selected_mentor_label = st.selectbox("Select Mentor", list(mentor_options.keys()))
                
                if selected_mentor_label:
                    mentor_id = mentor_options[selected_mentor_label]
                    selected_mentor = next(m for m in active_mentors if m['id'] == mentor_id)
                    
                    # Display mentor details
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
                    
                    # Assign button
                    if st.button("✅ Assign Mentor to Event", use_container_width=True, type="primary"):
                        if db.assign_mentor_to_event(event_id, mentor_id):
                            st.success(f"✅ {selected_mentor['full_name']} assigned to '{selected_event['title']}'!")
                            st.rerun()
                        else:
                            st.error("Failed to assign mentor.")

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
    
    # Update event status
    if 'last_status_update' not in st.session_state:
        st.session_state.last_status_update = datetime.now()
    
    if (datetime.now() - st.session_state.last_status_update).total_seconds() > 300:
        db.update_event_status()
        st.session_state.last_status_update = datetime.now()
    
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
