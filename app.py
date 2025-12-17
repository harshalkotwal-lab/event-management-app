"""
G H Raisoni College Event Management System
Complete web application for managing college events
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime, date
import hashlib
from pathlib import Path
import uuid
import os
from typing import Dict, List, Optional

# ============================================
# PAGE CONFIGURATION
# ============================================
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
    
    .stButton > button {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA MANAGER CLASS
# ============================================
class EventDataManager:
    """Manages all data operations for the application"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        self.users_file = self.data_dir / "users.json"
        self.events_file = self.data_dir / "events.json"
        self.registrations_file = self.data_dir / "registrations.json"
        
        self._initialize_files()
        
        # Default credentials (Change these in production!)
        self.default_admin = {
            "username": "admin@raisoni",
            "password": self._hash_password("admin123"),
            "name": "Administrator",
            "role": "admin",
            "department": "Administration",
            "email": "admin@ghraisoni.edu",
            "created_at": datetime.now().isoformat()
        }
        
        self.default_faculty = {
            "username": "faculty@raisoni",
            "password": self._hash_password("faculty123"),
            "name": "Faculty Coordinator",
            "role": "faculty",
            "department": "Faculty",
            "email": "faculty@ghraisoni.edu",
            "created_at": datetime.now().isoformat()
        }
    
    def _initialize_files(self):
        """Initialize JSON files if they don't exist"""
        default_data = {
            self.users_file: [],
            self.events_file: [],
            self.registrations_file: []
        }
        
        for file_path, default_content in default_data.items():
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump(default_content, f, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_credentials(self, username: str, password: str, role: str) -> bool:
        """Verify user credentials"""
        hashed_pw = self._hash_password(password)
        
        if role == "admin":
            return (username == self.default_admin["username"] and 
                    hashed_pw == self.default_admin["password"])
        
        elif role == "faculty":
            return (username == self.default_faculty["username"] and 
                    hashed_pw == self.default_faculty["password"])
        
        else:  # Student
            users = self.load_users()
            for user in users:
                if user.get("username") == username and user.get("password") == hashed_pw:
                    return True
            return False
    
    def load_users(self) -> List[Dict]:
        """Load all users from JSON file"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_user(self, user_data: Dict) -> bool:
        """Save a new user"""
        users = self.load_users()
        
        # Check if username already exists
        if any(u.get("username") == user_data.get("username") for u in users):
            return False
        
        # Add metadata
        user_data["user_id"] = str(uuid.uuid4())
        user_data["created_at"] = datetime.now().isoformat()
        user_data["is_active"] = True
        
        users.append(user_data)
        
        try:
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            return True
        except:
            return False
    
    def load_events(self) -> List[Dict]:
        """Load all events from JSON file"""
        try:
            with open(self.events_file, 'r') as f:
                events = json.load(f)
                # Sort by date (newest first)
                return sorted(events, key=lambda x: x.get('event_date', ''), reverse=True)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_event(self, event_data: Dict) -> str:
        """Save a new event"""
        events = self.load_events()
        
        # Generate event ID and add metadata
        event_id = str(uuid.uuid4())
        event_data["event_id"] = event_id
        event_data["created_at"] = datetime.now().isoformat()
        event_data["updated_at"] = datetime.now().isoformat()
        
        # Initialize engagement metrics
        event_data.setdefault("likes", 0)
        event_data.setdefault("favorites", [])
        event_data.setdefault("interested", [])
        event_data.setdefault("views", 0)
        
        events.append(event_data)
        
        try:
            with open(self.events_file, 'w') as f:
                json.dump(events, f, indent=2)
            return event_id
        except:
            return ""
    
    def update_event(self, event_id: str, updates: Dict) -> bool:
        """Update an existing event"""
        events = self.load_events()
        
        for i, event in enumerate(events):
            if event.get("event_id") == event_id:
                updates["updated_at"] = datetime.now().isoformat()
                events[i].update(updates)
                
                try:
                    with open(self.events_file, 'w') as f:
                        json.dump(events, f, indent=2)
                    return True
                except:
                    return False
        
        return False
    
    def load_registrations(self) -> List[Dict]:
        """Load all registrations"""
        try:
            with open(self.registrations_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_registration(self, registration_data: Dict) -> bool:
        """Save a new registration"""
        registrations = self.load_registrations()
        
        # Check if already registered
        event_id = registration_data.get("event_id")
        student_username = registration_data.get("student_username")
        
        if any(r.get("event_id") == event_id and 
               r.get("student_username") == student_username 
               for r in registrations):
            return False
        
        # Add metadata
        registration_data["registration_id"] = str(uuid.uuid4())
        registration_data["registered_at"] = datetime.now().isoformat()
        registration_data.setdefault("status", "pending")
        registration_data.setdefault("attendance", "absent")
        
        registrations.append(registration_data)
        
        try:
            with open(self.registrations_file, 'w') as f:
                json.dump(registrations, f, indent=2)
            return True
        except:
            return False
    
    def get_statistics(self) -> Dict:
        """Get application statistics"""
        events = self.load_events()
        users = self.load_users()
        registrations = self.load_registrations()
        
        # Count students
        students = [u for u in users if u.get("role") == "student"]
        
        # Count upcoming events
        today = datetime.now().date()
        upcoming_events = 0
        for event in events:
            try:
                event_date = datetime.fromisoformat(event.get("event_date", "")).date()
                if event_date >= today:
                    upcoming_events += 1
            except:
                continue
        
        return {
            "total_events": len(events),
            "upcoming_events": upcoming_events,
            "total_students": len(students),
            "total_registrations": len(registrations),
            "unique_participants": len(set(r.get("student_username") for r in registrations))
        }

# ============================================
# INITIALIZE DATA MANAGER
# ============================================
data_manager = EventDataManager()

# ============================================
# HELPER FUNCTIONS
# ============================================
def display_role_badge(role: str):
    """Display a badge for the user's role"""
    badges = {
        "admin": ("👑 Admin", "admin-badge"),
        "faculty": ("👨‍🏫 Faculty", "faculty-badge"),
        "student": ("👨‍🎓 Student", "student-badge")
    }
    
    if role in badges:
        text, css_class = badges[role]
        st.markdown(f'<span class="role-badge {css_class}">{text}</span>', 
                   unsafe_allow_html=True)

def format_date(date_string: str) -> str:
    """Format date string for display"""
    try:
        date_obj = datetime.fromisoformat(date_string)
        return date_obj.strftime("%d %b %Y, %I:%M %p")
    except:
        return date_string

def is_event_upcoming(event_date: str) -> bool:
    """Check if event date is in the future"""
    try:
        event_dt = datetime.fromisoformat(event_date)
        return event_dt > datetime.now()
    except:
        return False

# ============================================
# LOGIN PAGE
# ============================================
def login_page():
    """Display login page"""
    st.markdown('<div class="college-header"><h2>G H Raisoni College of Engineering and Management</h2><p>Event Management System</p></div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Admin Login
    with col1:
        st.subheader("Admin Login")
        st.markdown("**Default Credentials:**")
        st.code("Username: admin@raisoni\nPassword: admin123")
        
        if st.button("Login as Admin", key="admin_login", use_container_width=True):
            st.session_state.role = "admin"
            st.session_state.username = "admin@raisoni"
            st.session_state.name = "Administrator"
            st.rerun()
    
    # Faculty Login
    with col2:
        st.subheader("Faculty Login")
        st.markdown("**Default Credentials:**")
        st.code("Username: faculty@raisoni\nPassword: faculty123")
        
        if st.button("Login as Faculty", key="faculty_login", use_container_width=True):
            st.session_state.role = "faculty"
            st.session_state.username = "faculty@raisoni"
            st.session_state.name = "Faculty Coordinator"
            st.rerun()
    
    # Student Login/Register
    with col3:
        st.subheader("Student Portal")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        # Student Login Tab
        with tab1:
            student_username = st.text_input("Username", key="student_username")
            student_password = st.text_input("Password", type="password", key="student_password")
            
            if st.button("Student Login", key="student_login", use_container_width=True):
                if data_manager.verify_credentials(student_username, student_password, "student"):
                    st.session_state.role = "student"
                    st.session_state.username = student_username
                    
                    # Get student name from database
                    users = data_manager.load_users()
                    student = next((u for u in users if u.get("username") == student_username), None)
                    if student:
                        st.session_state.name = student.get("name", student_username)
                    
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        # Student Registration Tab
        with tab2:
            with st.form("student_registration"):
                st.markdown("**Create Student Account**")
                
                full_name = st.text_input("Full Name")
                roll_number = st.text_input("Roll Number")
                department = st.selectbox("Department", 
                                         ["CSE", "AIML", "ECE", "EEE", "MECH", "CIVIL", "IT", "DS"])
                academic_year = st.selectbox("Year", ["I", "II", "III", "IV"])
                email = st.text_input("Email Address")
                new_username = st.text_input("Choose Username")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                if st.form_submit_button("Register", use_container_width=True):
                    if not all([full_name, roll_number, email, new_username, new_password]):
                        st.error("Please fill all required fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        # Check if username exists
                        users = data_manager.load_users()
                        if any(u.get("username") == new_username for u in users):
                            st.error("Username already exists")
                        else:
                            user_data = {
                                "name": full_name,
                                "roll_number": roll_number,
                                "department": department,
                                "year": academic_year,
                                "email": email,
                                "username": new_username,
                                "password": data_manager._hash_password(new_password),
                                "role": "student"
                            }
                            
                            if data_manager.save_user(user_data):
                                st.success("Registration successful! Please login.")
                                st.rerun()
                            else:
                                st.error("Registration failed. Please try again.")

# ============================================
# ADMIN DASHBOARD
# ============================================
def admin_dashboard():
    """Admin dashboard with full control"""
    st.sidebar.title("👑 Admin Panel")
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.get('name', 'Admin')}")
    display_role_badge("admin")
    
    menu_options = ["Dashboard", "View All Events", "User Management", "Reports", "System Logs"]
    selected_menu = st.sidebar.selectbox("Navigation", menu_options)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Dashboard
    if selected_menu == "Dashboard":
        st.markdown('<h1 class="main-header">Administrator Dashboard</h1>', unsafe_allow_html=True)
        
        # Display statistics
        stats = data_manager.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{stats["total_events"]}</div>
                    <div class="metric-label">Total Events</div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{stats["upcoming_events"]}</div>
                    <div class="metric-label">Upcoming Events</div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{stats["total_students"]}</div>
                    <div class="metric-label">Total Students</div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{stats["total_registrations"]}</div>
                    <div class="metric-label">Total Registrations</div>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recent Events
        st.subheader("📅 Recent Events")
        events = data_manager.load_events()
        
        if events:
            recent_events = events[:5]  # Show 5 most recent
            for event in recent_events:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{event.get('title', 'Untitled Event')}**")
                        st.caption(f"Date: {format_date(event.get('event_date', ''))}")
                        st.caption(f"Type: {event.get('event_type', 'N/A')}")
                    with col2:
                        status = "✅ Upcoming" if is_event_upcoming(event.get('event_date', '')) else "❌ Completed"
                        st.markdown(f"**{status}**")
                    st.markdown("---")
        else:
            st.info("No events created yet.")
    
    # View All Events
    elif selected_menu == "View All Events":
        st.header("📋 All Events")
        
        events = data_manager.load_events()
        
        if not events:
            st.info("No events available.")
            return
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.selectbox("Filter by Type", 
                                      ["All", "Workshop", "Hackathon", "Competition", 
                                       "Bootcamp", "Seminar", "Conference"])
        with col2:
            filter_status = st.selectbox("Filter by Status", ["All", "Upcoming", "Completed"])
        with col3:
            search_query = st.text_input("Search Events")
        
        # Apply filters
        filtered_events = events
        
        if filter_type != "All":
            filtered_events = [e for e in filtered_events if e.get("event_type") == filter_type]
        
        if filter_status == "Upcoming":
            filtered_events = [e for e in filtered_events if is_event_upcoming(e.get("event_date", ""))]
        elif filter_status == "Completed":
            filtered_events = [e for e in filtered_events if not is_event_upcoming(e.get("event_date", ""))]
        
        if search_query:
            filtered_events = [e for e in filtered_events 
                             if search_query.lower() in e.get("title", "").lower() 
                             or search_query.lower() in e.get("description", "").lower()]
        
        # Display events
        for event in filtered_events:
            with st.container():
                st.markdown('<div class="event-card">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(event.get("title", "Untitled Event"))
                    st.write(event.get("description", "No description available."))
                    
                    col_info = st.columns(4)
                    with col_info[0]:
                        st.caption(f"**Date:** {format_date(event.get('event_date', ''))}")
                    with col_info[1]:
                        st.caption(f"**Type:** {event.get('event_type', 'N/A')}")
                    with col_info[2]:
                        st.caption(f"**Venue:** {event.get('venue', 'N/A')}")
                    with col_info[3]:
                        st.caption(f"**Organizer:** {event.get('organizer', 'N/A')}")
                
                with col2:
                    status = "🟢 Upcoming" if is_event_upcoming(event.get('event_date', '')) else "🔴 Completed"
                    st.markdown(f"**{status}**")
                    
                    # Engagement metrics
                    col_metrics = st.columns(2)
                    with col_metrics[0]:
                        st.metric("Likes", event.get("likes", 0))
                    with col_metrics[1]:
                        st.metric("Interested", len(event.get("interested", [])))
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # User Management
    elif selected_menu == "User Management":
        st.header("👥 User Management")
        
        users = data_manager.load_users()
        
        if not users:
            st.info("No users registered yet.")
            return
        
        # Convert to DataFrame for better display
        user_data = []
        for user in users:
            user_data.append({
                "Name": user.get("name", "N/A"),
                "Username": user.get("username", "N/A"),
                "Role": user.get("role", "N/A"),
                "Department": user.get("department", "N/A"),
                "Year": user.get("year", "N/A"),
                "Email": user.get("email", "N/A"),
                "Registered": user.get("created_at", "N/A")[:10] if user.get("created_at") else "N/A"
            })
        
        df = pd.DataFrame(user_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export option
        if st.button("Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="users.csv",
                mime="text/csv"
            )
    
    # Reports
    elif selected_menu == "Reports":
        st.header("📊 Reports & Analytics")
        
        stats = data_manager.get_statistics()
        registrations = data_manager.load_registrations()
        events = data_manager.load_events()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Event type distribution
            event_types = {}
            for event in events:
                etype = event.get("event_type", "Other")
                event_types[etype] = event_types.get(etype, 0) + 1
            
            if event_types:
                st.subheader("Events by Type")
                type_df = pd.DataFrame(list(event_types.items()), columns=["Event Type", "Count"])
                st.bar_chart(type_df.set_index("Event Type"))
        
        with col2:
            # Registration trends
            if registrations:
                st.subheader("Recent Registrations")
                
                # Get last 7 days registrations
                recent_regs = []
                for reg in registrations:
                    try:
                        reg_date = datetime.fromisoformat(reg.get("registered_at", ""))
                        if (datetime.now() - reg_date).days <= 7:
                            recent_regs.append(reg)
                    except:
                        continue
                
                if recent_regs:
                    st.metric("Last 7 Days", len(recent_regs))
                else:
                    st.info("No registrations in the last 7 days")
        
        # Detailed report
        st.subheader("Detailed Report")
        report_data = [
            {"Metric": "Total Events", "Value": stats["total_events"]},
            {"Metric": "Upcoming Events", "Value": stats["upcoming_events"]},
            {"Metric": "Total Students", "Value": stats["total_students"]},
            {"Metric": "Total Registrations", "Value": stats["total_registrations"]},
            {"Metric": "Unique Participants", "Value": stats["unique_participants"]},
        ]
        
        report_df = pd.DataFrame(report_data)
        st.table(report_df)
    
    # System Logs
    elif selected_menu == "System Logs":
        st.header("📝 System Logs")
        st.info("System logs feature will be available in the next update.")

# ============================================
# FACULTY DASHBOARD
# ============================================
def faculty_dashboard():
    """Faculty coordinator dashboard"""
    st.sidebar.title("👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.get('name', 'Faculty')}")
    display_role_badge("faculty")
    
    menu_options = ["Dashboard", "Create Event", "My Events", "View Registrations"]
    selected_menu = st.sidebar.selectbox("Navigation", menu_options)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Dashboard
    if selected_menu == "Dashboard":
        st.markdown('<h1 class="main-header">Faculty Coordinator Dashboard</h1>', unsafe_allow_html=True)
        
        # Get faculty-specific statistics
        events = data_manager.load_events()
        my_events = [e for e in events if e.get("created_by") == st.session_state.username]
        registrations = data_manager.load_registrations()
        
        # Count my event registrations
        my_event_ids = [e.get("event_id") for e in my_events]
        my_registrations = [r for r in registrations if r.get("event_id") in my_event_ids]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{len(my_events)}</div>
                    <div class="metric-label">My Events</div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{len(my_registrations)}</div>
                    <div class="metric-label">Total Registrations</div>
                </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            upcoming_count = len([e for e in my_events if is_event_upcoming(e.get("event_date", ""))])
            st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{upcoming_count}</div>
                    <div class="metric-label">Upcoming Events</div>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # My Upcoming Events
        st.subheader("📅 My Upcoming Events")
        upcoming_events = [e for e in my_events if is_event_upcoming(e.get("event_date", ""))]
        
        if upcoming_events:
            for event in upcoming_events[:3]:  # Show only 3
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{event.get('title', 'Untitled Event')}**")
                        st.caption(f"Date: {format_date(event.get('event_date', ''))}")
                        st.caption(f"Venue: {event.get('venue', 'N/A')}")
                    with col2:
                        registrations_count = len([r for r in my_registrations 
                                                  if r.get("event_id") == event.get("event_id")])
                        st.metric("Registrations", registrations_count)
                    st.markdown("---")
        else:
            st.info("No upcoming events. Create your first event!")
    
    # Create Event
    elif selected_menu == "Create Event":
        st.header("➕ Create New Event")
        
        with st.form("create_event_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                event_title = st.text_input("Event Title *", placeholder="e.g., AI Workshop 2024")
                event_type = st.selectbox("Event Type *", 
                                         ["Workshop", "Hackathon", "Competition", 
                                          "Bootcamp", "Seminar", "Conference", "Webinar"])
                event_date = st.date_input("Event Date *", min_value=date.today())
                event_time = st.time_input("Event Time *")
                
            with col2:
                venue = st.text_input("Venue *", placeholder="e.g., Seminar Hall, Block A")
                organizer = st.text_input("Organizer *", value="G H Raisoni College")
                registration_link = st.text_input("Registration Link (Optional)", 
                                                 placeholder="https://forms.google.com/...")
            
            description = st.text_area("Event Description *", 
                                      placeholder="Describe the event details, agenda, speakers, etc.",
                                      height=150)
            
            required_fields = [event_title, event_type, venue, organizer, description]
            
            if st.form_submit_button("Create Event", use_container_width=True):
                if not all(required_fields):
                    st.error("Please fill all required fields (*)")
                else:
                    # Combine date and time
                    event_datetime = datetime.combine(event_date, event_time)
                    
                    event_data = {
                        "title": event_title,
                        "description": description,
                        "event_type": event_type,
                        "event_date": event_datetime.isoformat(),
                        "venue": venue,
                        "organizer": organizer,
                        "registration_link": registration_link,
                        "created_by": st.session_state.username,
                        "created_by_name": st.session_state.name
                    }
                    
                    event_id = data_manager.save_event(event_data)
                    if event_id:
                        st.success(f"Event '{event_title}' created successfully!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Failed to create event. Please try again.")
    
    # My Events
    elif selected_menu == "My Events":
        st.header("📋 My Events")
        
        events = data_manager.load_events()
        my_events = [e for e in events if e.get("created_by") == st.session_state.username]
        
        if not my_events:
            st.info("You haven't created any events yet.")
            return
        
        # Filter options
        tab1, tab2 = st.tabs(["Upcoming Events", "Past Events"])
        
        with tab1:
            upcoming_events = [e for e in my_events if is_event_upcoming(e.get("event_date", ""))]
            
            if upcoming_events:
                for event in upcoming_events:
                    display_event_card(event, show_actions=False)
            else:
                st.info("No upcoming events.")
        
        with tab2:
            past_events = [e for e in my_events if not is_event_upcoming(e.get("event_date", ""))]
            
            if past_events:
                for event in past_events:
                    display_event_card(event, show_actions=False)
            else:
                st.info("No past events.")
    
    # View Registrations
    elif selected_menu == "View Registrations":
        st.header("📝 Event Registrations")
        
        events = data_manager.load_events()
        my_events = [e for e in events if e.get("created_by") == st.session_state.username]
        
        if not my_events:
            st.info("You haven't created any events yet.")
            return
        
        # Select event to view registrations
        event_titles = [e.get("title") for e in my_events]
        selected_event_title = st.selectbox("Select Event", event_titles)
        
        if selected_event_title:
            selected_event = next(e for e in my_events if e.get("title") == selected_event_title)
            event_id = selected_event.get("event_id")
            
            registrations = data_manager.load_registrations()
            event_registrations = [r for r in registrations if r.get("event_id") == event_id]
            
            if event_registrations:
                st.subheader(f"Registrations for: {selected_event_title}")
                
                # Convert to DataFrame
                reg_data = []
                for reg in event_registrations:
                    reg_data.append({
                        "Student Name": reg.get("student_name", "N/A"),
                        "Roll Number": reg.get("roll_number", "N/A"),
                        "Department": reg.get("department", "N/A"),
                        "Registration Date": format_date(reg.get("registered_at", "")),
                        "Status": reg.get("status", "pending").title(),
                        "Attendance": reg.get("attendance", "absent").title()
                    })
                
                df = pd.DataFrame(reg_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Export option
                if st.button("Export Registrations to CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"registrations_{selected_event_title.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                
                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Registrations", len(event_registrations))
                with col2:
                    attended = len([r for r in event_registrations if r.get("attendance") == "present"])
                    st.metric("Attended", attended)
                with col3:
                    pending = len([r for r in event_registrations if r.get("status") == "pending"])
                    st.metric("Pending Approval", pending)
            else:
                st.info(f"No registrations yet for '{selected_event_title}'")

# ============================================
# STUDENT DASHBOARD
# ============================================
def student_dashboard():
    """Student dashboard"""
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.get('name', 'Student')}")
    display_role_badge("student")
    
    # Get student info
    users = data_manager.load_users()
    student = next((u for u in users if u.get("username") == st.session_state.username), {})
    
    if student:
        st.sidebar.markdown(f"**Roll No:** {student.get('roll_number', 'N/A')}")
        st.sidebar.markdown(f"**Department:** {student.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Year:** {student.get('year', 'N/A')}")
    
    menu_options = ["Events Feed", "My Registrations", "My Profile"]
    selected_menu = st.sidebar.selectbox("Navigation", menu_options)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Events Feed
    if selected_menu == "Events Feed":
        st.markdown('<h1 class="main-header">🎯 Upcoming Events</h1>', unsafe_allow_html=True)
        
        events = data_manager.load_events()
        registrations = data_manager.load_registrations()
        
        # Filter upcoming events
        upcoming_events = [e for e in events if is_event_upcoming(e.get("event_date", ""))]
        
        if not upcoming_events:
            st.info("No upcoming events at the moment. Check back later!")
            return
        
        # Search and filter
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("Search events", placeholder="Type to search...")
        with col2:
            event_type_filter = st.selectbox("Filter by type", 
                                            ["All", "Workshop", "Hackathon", "Competition", 
                                             "Bootcamp", "Seminar", "Conference"])
        
        # Apply filters
        filtered_events = upcoming_events
        
        if search_query:
            filtered_events = [e for e in filtered_events 
                             if search_query.lower() in e.get("title", "").lower() 
                             or search_query.lower() in e.get("description", "").lower()]
        
        if event_type_filter != "All":
            filtered_events = [e for e in filtered_events if e.get("event_type") == event_type_filter]
        
        # Display events
        for event in filtered_events:
            display_event_card(event, show_actions=True)
    
    # My Registrations
    elif selected_menu == "My Registrations":
        st.header("📋 My Registered Events")
        
        registrations = data_manager.load_registrations()
        my_registrations = [r for r in registrations 
                           if r.get("student_username") == st.session_state.username]
        
        if not my_registrations:
            st.info("You haven't registered for any events yet.")
            return
        
        # Get event details for each registration
        events = data_manager.load_events()
        event_map = {e.get("event_id"): e for e in events}
        
        # Tabs for different statuses
        tab1, tab2, tab3 = st.tabs(["Upcoming", "Completed", "All"])
        
        with tab1:
            upcoming_regs = []
            for reg in my_registrations:
                event = event_map.get(reg.get("event_id"))
                if event and is_event_upcoming(event.get("event_date", "")):
                    upcoming_regs.append((reg, event))
            
            if upcoming_regs:
                for reg, event in upcoming_regs:
                    display_registration_card(reg, event)
            else:
                st.info("No upcoming registered events.")
        
        with tab2:
            completed_regs = []
            for reg in my_registrations:
                event = event_map.get(reg.get("event_id"))
                if event and not is_event_upcoming(event.get("event_date", "")):
                    completed_regs.append((reg, event))
            
            if completed_regs:
                for reg, event in completed_regs:
                    display_registration_card(reg, event)
            else:
                st.info("No completed events.")
        
        with tab3:
            for reg in my_registrations:
                event = event_map.get(reg.get("event_id"))
                if event:
                    display_registration_card(reg, event)
    
    # My Profile
    elif selected_menu == "My Profile":
        st.header("👤 My Profile")
        
        users = data_manager.load_users()
        student = next((u for u in users if u.get("username") == st.session_state.username), None)
        
        if not student:
            st.error("Profile not found")
            return
        
        # Display profile info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Full Name:** {student.get('name', 'N/A')}")
            st.markdown(f"**Roll Number:** {student.get('roll_number', 'N/A')}")
            st.markdown(f"**Department:** {student.get('department', 'N/A')}")
            st.markdown(f"**Year:** {student.get('year', 'N/A')}")
        
        with col2:
            st.markdown(f"**Email:** {student.get('email', 'N/A')}")
            st.markdown(f"**Username:** {student.get('username', 'N/A')}")
            st.markdown(f"**Member Since:** {student.get('created_at', 'N/A')[:10]}")
        
        st.markdown("---")
        
        # Statistics
        registrations = data_manager.load_registrations()
        my_registrations = [r for r in registrations 
                           if r.get("student_username") == st.session_state.username]
        
        events = data_manager.load_events()
        event_map = {e.get("event_id"): e for e in events}
        
        # Calculate stats
        total_registrations = len(my_registrations)
        attended_events = len([r for r in my_registrations if r.get("attendance") == "present"])
        upcoming_events = len([r for r in my_registrations 
                              if event_map.get(r.get("event_id")) and 
                              is_event_upcoming(event_map.get(r.get("event_id"), {}).get("event_date", ""))])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Registrations", total_registrations)
        with col2:
            st.metric("Events Attended", attended_events)
        with col3:
            st.metric("Upcoming Events", upcoming_events)

# ============================================
# EVENT CARD DISPLAY
# ============================================
def display_event_card(event: Dict, show_actions: bool = True):
    """Display an event card"""
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(event.get("title", "Untitled Event"))
            st.write(event.get("description", "No description available."))
            
            # Event details in columns
            col_details = st.columns(4)
            with col_details[0]:
                st.caption(f"**📅 Date:** {format_date(event.get('event_date', ''))}")
            with col_details[1]:
                st.caption(f"**📍 Venue:** {event.get('venue', 'N/A')}")
            with col_details[2]:
                st.caption(f"**🏷️ Type:** {event.get('event_type', 'N/A')}")
            with col_details[3]:
                st.caption(f"**👨‍🏫 Organizer:** {event.get('organizer', 'N/A')}")
        
        with col2:
            # Status indicator
            if is_event_upcoming(event.get("event_date", "")):
                st.success("🟢 Upcoming")
            else:
                st.error("🔴 Completed")
            
            # Engagement metrics
            col_metrics = st.columns(2)
            with col_metrics[0]:
                st.metric("👍", event.get("likes", 0))
            with col_metrics[1]:
                st.metric("🤔", len(event.get("interested", [])))
        
        # Action buttons (for students)
        if show_actions and st.session_state.get("role") == "student":
            st.markdown("---")
            
            # Check if already registered
            registrations = data_manager.load_registrations()
            already_registered = any(
                r.get("event_id") == event.get("event_id") and 
                r.get("student_username") == st.session_state.username 
                for r in registrations
            )
            
            col_actions = st.columns(4)
            
            with col_actions[0]:
                if st.button("👍 Like", key=f"like_{event.get('event_id')}"):
                    current_likes = event.get("likes", 0)
                    data_manager.update_event(event.get("event_id"), {"likes": current_likes + 1})
                    st.success("Liked!")
                    st.rerun()
            
            with col_actions[1]:
                interested_users = event.get("interested", [])
                is_interested = st.session_state.username in interested_users
                button_text = "✅ Interested" if is_interested else "🤔 Interested"
                
                if st.button(button_text, key=f"interest_{event.get('event_id')}"):
                    if is_interested:
                        interested_users.remove(st.session_state.username)
                    else:
                        interested_users.append(st.session_state.username)
                    
                    data_manager.update_event(event.get("event_id"), {"interested": interested_users})
                    st.rerun()
            
            with col_actions[2]:
                if event.get("registration_link"):
                    st.markdown(f"[🔗 Register Here]({event.get('registration_link')})", 
                               unsafe_allow_html=True)
            
            with col_actions[3]:
                if already_registered:
                    st.success("✅ Registered")
                else:
                    if st.button("📝 Mark as Registered", key=f"register_{event.get('event_id')}"):
                        # Get student info
                        users = data_manager.load_users()
                        student = next((u for u in users 
                                       if u.get("username") == st.session_state.username), {})
                        
                        registration_data = {
                            "event_id": event.get("event_id"),
                            "event_title": event.get("title"),
                            "student_username": st.session_state.username,
                            "student_name": student.get("name", st.session_state.username),
                            "roll_number": student.get("roll_number", "N/A"),
                            "department": student.get("department", "N/A"),
                            "status": "pending"
                        }
                        
                        if data_manager.save_registration(registration_data):
                            st.success("Registration marked successfully!")
                            st.rerun()
                        else:
                            st.error("Already registered for this event")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# REGISTRATION CARD DISPLAY
# ============================================
def display_registration_card(registration: Dict, event: Dict):
    """Display a registration card"""
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(event.get("title", "Untitled Event"))
            
            col_details = st.columns(3)
            with col_details[0]:
                st.caption(f"**📅 Event Date:** {format_date(event.get('event_date', ''))}")
            with col_details[1]:
                st.caption(f"**📍 Venue:** {event.get('venue', 'N/A')}")
            with col_details[2]:
                st.caption(f"**🏷️ Type:** {event.get('event_type', 'N/A')}")
        
        with col2:
            # Status indicators
            status = registration.get("status", "pending").title()
            attendance = registration.get("attendance", "absent").title()
            
            if status == "Approved":
                st.success(f"✅ {status}")
            elif status == "Rejected":
                st.error(f"❌ {status}")
            else:
                st.warning(f"⏳ {status}")
            
            st.caption(f"Attendance: {attendance}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application entry point"""
    
    # Initialize session state
    if "role" not in st.session_state:
        st.session_state.role = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "name" not in st.session_state:
        st.session_state.name = None
    
    # Route to appropriate dashboard
    if st.session_state.role is None:
        login_page()
    elif st.session_state.role == "admin":
        admin_dashboard()
    elif st.session_state.role == "faculty":
        faculty_dashboard()
    elif st.session_state.role == "student":
        student_dashboard()

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    main()
