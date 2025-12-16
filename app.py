import streamlit as st
import json
import pandas as pd
from datetime import datetime
import hashlib
from pathlib import Path
import uuid

# Rest of your app code...

# Page configuration
st.set_page_config(
    page_title="G H Raisoni Event Manager",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .event-card {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .event-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .role-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .admin-badge { background: #FEE2E2; color: #DC2626; }
    .faculty-badge { background: #DBEAFE; color: #1D4ED8; }
    .student-badge { background: #D1FAE5; color: #065F46; }
    .action-button {
        margin: 0.25rem;
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Data storage functions
class EventManager:
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.users_file = self.data_dir / "users.json"
        self.events_file = self.data_dir / "events.json"
        self.registrations_file = self.data_dir / "registrations.json"
        
        # Initialize files if they don't exist
        self._init_files()
        
        # Default credentials
        self.default_admin = {"username": "admin@raisoni", "password": "admin123", "role": "admin", "name": "Administrator"}
        self.default_faculty = {"username": "faculty@raisoni", "password": "faculty123", "role": "faculty", "name": "Faculty Coordinator"}
    
    def _init_files(self):
        # Initialize users.json
        if not self.users_file.exists():
            with open(self.users_file, 'w') as f:
                json.dump([], f)
        
        # Initialize events.json
        if not self.events_file.exists():
            with open(self.events_file, 'w') as f:
                json.dump([], f)
        
        # Initialize registrations.json
        if not self.registrations_file.exists():
            with open(self.registrations_file, 'w') as f:
                json.dump([], f)
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_login(self, username, password, role):
        hashed_pw = self.hash_password(password)
        
        if role == "admin":
            return username == self.default_admin["username"] and hashed_pw == self.hash_password(self.default_admin["password"])
        elif role == "faculty":
            return username == self.default_faculty["username"] and hashed_pw == self.hash_password(self.default_faculty["password"])
        else:
            users = self.load_users()
            for user in users:
                if user["username"] == username and user["password"] == hashed_pw:
                    return True
            return False
    
    def load_users(self):
        with open(self.users_file, 'r') as f:
            return json.load(f)
    
    def save_user(self, user_data):
        users = self.load_users()
        users.append(user_data)
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def load_events(self):
        with open(self.events_file, 'r') as f:
            return json.load(f)
    
    def save_event(self, event_data):
        events = self.load_events()
        event_data["event_id"] = str(uuid.uuid4())
        event_data["created_at"] = datetime.now().isoformat()
        events.append(event_data)
        with open(self.events_file, 'w') as f:
            json.dump(events, f, indent=2)
        return event_data["event_id"]
    
    def update_event(self, event_id, updated_data):
        events = self.load_events()
        for i, event in enumerate(events):
            if event["event_id"] == event_id:
                events[i].update(updated_data)
                break
        with open(self.events_file, 'w') as f:
            json.dump(events, f, indent=2)
    
    def load_registrations(self):
        with open(self.registrations_file, 'r') as f:
            return json.load(f)
    
    def save_registration(self, registration_data):
        registrations = self.load_registrations()
        registrations.append(registration_data)
        with open(self.registrations_file, 'w') as f:
            json.dump(registrations, f, indent=2)

# Initialize Event Manager
manager = EventManager()

def login_page():
    st.title("🎓 G H Raisoni College - Event Management System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Admin Login")
        st.markdown("Default Credentials:")
        st.code("Username: admin@raisoni\nPassword: admin123")
        if st.button("Login as Admin", key="admin_btn"):
            st.session_state["role"] = "admin"
            st.session_state["username"] = "admin@raisoni"
            st.rerun()
    
    with col2:
        st.subheader("Faculty Login")
        st.markdown("Default Credentials:")
        st.code("Username: faculty@raisoni\nPassword: faculty123")
        if st.button("Login as Faculty", key="faculty_btn"):
            st.session_state["role"] = "faculty"
            st.session_state["username"] = "faculty@raisoni"
            st.rerun()
    
    with col3:
        st.subheader("Student Login/Register")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            student_user = st.text_input("Username")
            student_pass = st.text_input("Password", type="password")
            if st.button("Student Login"):
                if manager.validate_login(student_user, student_pass, "student"):
                    st.session_state["role"] = "student"
                    st.session_state["username"] = student_user
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        with tab2:
            with st.form("register_form"):
                new_name = st.text_input("Full Name")
                new_roll = st.text_input("Roll Number")
                new_dept = st.selectbox("Department", ["CSE", "AIML", "ECE", "EEE", "MECH", "CIVIL"])
                new_year = st.selectbox("Year", ["I", "II", "III", "IV"])
                new_email = st.text_input("Email")
                new_user = st.text_input("Username")
                new_pass = st.text_input("Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                
                if st.form_submit_button("Register"):
                    if new_pass != confirm_pass:
                        st.error("Passwords don't match!")
                    else:
                        users = manager.load_users()
                        if any(u["username"] == new_user for u in users):
                            st.error("Username already exists!")
                        else:
                            user_data = {
                                "name": new_name,
                                "roll_no": new_roll,
                                "department": new_dept,
                                "year": new_year,
                                "email": new_email,
                                "username": new_user,
                                "password": manager.hash_password(new_pass),
                                "role": "student",
                                "registered_at": datetime.now().isoformat()
                            }
                            manager.save_user(user_data)
                            st.success("Registration successful! Please login.")
                            st.rerun()

def admin_dashboard():
    st.sidebar.title(f"👑 Admin Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.get('username', '')}")
    
    menu = ["Dashboard", "All Events", "User Management", "Analytics", "System Settings"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    if choice == "Dashboard":
        st.markdown('<h1 class="main-header">Administrator Dashboard</h1>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        events = manager.load_events()
        users = manager.load_users()
        registrations = manager.load_registrations()
        
        with col1:
            st.metric("Total Events", len(events))
        with col2:
            st.metric("Total Students", len([u for u in users if u["role"] == "student"]))
        with col3:
            st.metric("Total Registrations", len(registrations))
        with col4:
            st.metric("Active Events", len([e for e in events if datetime.fromisoformat(e.get('event_date', '2000-01-01')) > datetime.now()]))
        
        st.markdown("---")
        
        # Recent Events
        st.subheader("📅 Recent Events")
        if events:
            df_events = pd.DataFrame(events)
            if 'event_date' in df_events.columns:
                df_events['event_date'] = pd.to_datetime(df_events['event_date'])
                df_events = df_events.sort_values('event_date', ascending=False)
            st.dataframe(df_events.head(5), use_container_width=True)
        else:
            st.info("No events created yet.")
    
    elif choice == "All Events":
        st.header("📋 All Events")
        events = manager.load_events()
        
        for event in events:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(event["title"])
                    st.write(f"**Date:** {event.get('event_date', 'N/A')}")
                    st.write(f"**Type:** {event.get('event_type', 'N/A')}")
                    st.write(f"**Organizer:** {event.get('organizer', 'N/A')}")
                with col2:
                    st.button("Edit", key=f"edit_{event['event_id']}", on_click=lambda e=event: edit_event(e))
                    st.button("Delete", key=f"del_{event['event_id']}", type="secondary")
                st.markdown("---")
    
    elif choice == "User Management":
        st.header("👥 User Management")
        users = manager.load_users()
        df_users = pd.DataFrame(users)
        st.dataframe(df_users, use_container_width=True)
    
    elif choice == "Analytics":
        st.header("📊 Analytics")
        
        events = manager.load_events()
        registrations = manager.load_registrations()
        
        if events:
            # Event type distribution
            event_types = {}
            for event in events:
                etype = event.get('event_type', 'Other')
                event_types[etype] = event_types.get(etype, 0) + 1
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Events by Type")
                st.bar_chart(event_types)
            
            with col2:
                st.subheader("Registration Status")
                status_counts = {
                    'Total Events': len(events),
                    'With Registrations': len(set(r['event_id'] for r in registrations)),
                    'Total Registrations': len(registrations)
                }
                st.dataframe(pd.DataFrame(list(status_counts.items()), columns=['Metric', 'Count']))

def faculty_dashboard():
    st.sidebar.title(f"👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.get('username', '')}")
    
    menu = ["Dashboard", "Create Event", "My Events", "Student Registrations"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    if choice == "Dashboard":
        st.markdown('<h1 class="main-header">Faculty Coordinator Dashboard</h1>', unsafe_allow_html=True)
        
        events = manager.load_events()
        my_events = [e for e in events if e.get('created_by') == st.session_state.get('username')]
        registrations = manager.load_registrations()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("My Events", len(my_events))
        with col2:
            st.metric("Total Registrations", len([r for r in registrations if r['event_id'] in [e['event_id'] for e in my_events]]))
        with col3:
            upcoming = len([e for e in my_events if datetime.fromisoformat(e.get('event_date', '2000-01-01')) > datetime.now()])
            st.metric("Upcoming Events", upcoming)
    
    elif choice == "Create Event":
        st.header("➕ Create New Event")
        
        with st.form("create_event"):
            title = st.text_input("Event Title*")
            description = st.text_area("Description*")
            col1, col2 = st.columns(2)
            with col1:
                event_date = st.date_input("Event Date*")
                event_time = st.time_input("Event Time")
                venue = st.text_input("Venue*")
            with col2:
                event_type = st.selectbox("Event Type*", ["Workshop", "Hackathon", "Competition", "Bootcamp", "Seminar", "Conference"])
                organizer = st.text_input("Organizer*", value="G H Raisoni College")
                official_link = st.text_input("Registration Link")
            
            submit = st.form_submit_button("Create Event")
            
            if submit:
                if not all([title, description, venue, organizer]):
                    st.error("Please fill all required fields (*)")
                else:
                    event_data = {
                        "title": title,
                        "description": description,
                        "event_date": event_date.isoformat(),
                        "event_time": str(event_time),
                        "venue": venue,
                        "event_type": event_type,
                        "organizer": organizer,
                        "official_link": official_link,
                        "created_by": st.session_state.get('username'),
                        "likes": 0,
                        "favorites": [],
                        "interested": []
                    }
                    manager.save_event(event_data)
                    st.success("Event created successfully!")
                    st.rerun()
    
    elif choice == "My Events":
        st.header("📋 My Events")
        events = manager.load_events()
        my_events = [e for e in events if e.get('created_by') == st.session_state.get('username')]
        
        for event in my_events:
            display_event_card(event, show_actions=False, show_stats=True)
    
    elif choice == "Student Registrations":
        st.header("📝 Student Registrations")
        events = manager.load_events()
        my_events = [e for e in events if e.get('created_by') == st.session_state.get('username')]
        registrations = manager.load_registrations()
        
        selected_event = st.selectbox("Select Event", [e['title'] for e in my_events])
        
        if selected_event:
            event_id = next(e['event_id'] for e in my_events if e['title'] == selected_event)
            event_registrations = [r for r in registrations if r['event_id'] == event_id]
            
            if event_registrations:
                df = pd.DataFrame(event_registrations)
                st.dataframe(df[['student_name', 'roll_no', 'department', 'registration_date', 'status']], use_container_width=True)
            else:
                st.info("No registrations yet for this event.")

def student_dashboard():
    st.sidebar.title(f"👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.get('username', '')}")
    
    menu = ["Events Feed", "My Registrations", "Profile"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    if choice == "Events Feed":
        st.markdown('<h1 class="main-header">🎯 Upcoming Events</h1>', unsafe_allow_html=True)
        
        events = manager.load_events()
        current_user = st.session_state.get('username')
        
        # Filter upcoming events
        upcoming_events = []
        for event in events:
            try:
                event_date = datetime.fromisoformat(event.get('event_date', '2000-01-01'))
                if event_date > datetime.now():
                    upcoming_events.append(event)
            except:
                continue
        
        if not upcoming_events:
            st.info("No upcoming events at the moment.")
        
        for event in upcoming_events:
            display_event_card(event, show_actions=True, current_user=current_user)
    
    elif choice == "My Registrations":
        st.header("📋 My Registered Events")
        
        registrations = manager.load_registrations()
        current_user = st.session_state.get('username')
        my_registrations = [r for r in registrations if r['student_username'] == current_user]
        
        if not my_registrations:
            st.info("You haven't registered for any events yet.")
        else:
            for reg in my_registrations:
                events = manager.load_events()
                event = next((e for e in events if e['event_id'] == reg['event_id']), None)
                if event:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(event['title'])
                            st.write(f"**Date:** {event.get('event_date', 'N/A')}")
                            st.write(f"**Venue:** {event.get('venue', 'N/A')}")
                            st.write(f"**Registration Status:** {reg.get('status', 'Pending')}")
                        with col2:
                            if st.button("View Certificate", key=f"cert_{reg['registration_id']}"):
                                st.info("Certificate generation feature coming soon!")
                        st.markdown("---")
    
    elif choice == "Profile":
        st.header("👤 My Profile")
        users = manager.load_users()
        current_user = st.session_state.get('username')
        user_info = next((u for u in users if u['username'] == current_user), None)
        
        if user_info:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Name:** {user_info.get('name', 'N/A')}")
                st.write(f"**Roll No:** {user_info.get('roll_no', 'N/A')}")
                st.write(f"**Department:** {user_info.get('department', 'N/A')}")
            with col2:
                st.write(f"**Year:** {user_info.get('year', 'N/A')}")
                st.write(f"**Email:** {user_info.get('email', 'N/A')}")
                st.write(f"**Member Since:** {user_info.get('registered_at', 'N/A')[:10]}")

def display_event_card(event, show_actions=True, current_user=None, show_stats=False):
    """Display event card with interactive elements"""
    with st.container():
        st.markdown(f'<div class="event-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(event["title"])
            st.write(event["description"])
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.write(f"**📅 Date:** {event.get('event_date', 'N/A')}")
 
