# app.py
import streamlit as st
import json
import pandas as pd
from datetime import datetime
import hashlib
from pathlib import Path
import uuid

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="G H Raisoni Event Manager",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
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
    }
</style>
""", unsafe_allow_html=True)

# ---------------- DATA LAYER ----------------
class EventManager:
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.users_file = self.data_dir / "users.json"
        self.events_file = self.data_dir / "events.json"
        self.registrations_file = self.data_dir / "registrations.json"
        self._init_files()

        self.default_admin = {
            "username": "admin@raisoni",
            "password": "admin123",
            "role": "admin"
        }
        self.default_faculty = {
            "username": "faculty@raisoni",
            "password": "faculty123",
            "role": "faculty"
        }

    def _init_files(self):
        for f in [self.users_file, self.events_file, self.registrations_file]:
            if not f.exists():
                f.write_text("[]")

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def validate_login(self, username, password, role):
        hashed = self.hash_password(password)
        if role == "admin":
            return username == self.default_admin["username"] and hashed == self.hash_password(self.default_admin["password"])
        if role == "faculty":
            return username == self.default_faculty["username"] and hashed == self.hash_password(self.default_faculty["password"])
        for u in self.load_users():
            if u["username"] == username and u["password"] == hashed:
                return True
        return False

    def load_users(self):
        return json.loads(self.users_file.read_text())

    def save_user(self, data):
        users = self.load_users()
        users.append(data)
        self.users_file.write_text(json.dumps(users, indent=2))

    def load_events(self):
        return json.loads(self.events_file.read_text())

    def save_event(self, data):
        events = self.load_events()
        data["event_id"] = str(uuid.uuid4())
        events.append(data)
        self.events_file.write_text(json.dumps(events, indent=2))

    def load_registrations(self):
        return json.loads(self.registrations_file.read_text())

manager = EventManager()

# ---------------- PAGES ----------------
def login_page():
    st.title("🎓 G H Raisoni Event Management System")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Admin")
        if st.button("Login as Admin"):
            st.session_state.role = "admin"
            st.session_state.username = "admin@raisoni"
            st.rerun()

    with col2:
        st.subheader("Faculty")
        if st.button("Login as Faculty"):
            st.session_state.role = "faculty"
            st.session_state.username = "faculty@raisoni"
            st.rerun()

    with col3:
        st.subheader("Student Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if manager.validate_login(u, p, "student"):
                st.session_state.role = "student"
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid credentials")


def admin_dashboard():
    st.sidebar.title("👑 Admin")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()
    st.markdown('<h1 class="main-header">Admin Dashboard</h1>', unsafe_allow_html=True)
    st.metric("Total Events", len(manager.load_events()))


def faculty_dashboard():
    st.sidebar.title("👨‍🏫 Faculty")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()
    st.markdown('<h1 class="main-header">Faculty Dashboard</h1>', unsafe_allow_html=True)


def student_dashboard():
    st.sidebar.title("👨‍🎓 Student")
    if st.sidebar.button("Logout"):
        st.session_state.clear(); st.rerun()
    st.markdown('<h1 class="main-header">Student Dashboard</h1>', unsafe_allow_html=True)

# ---------------- MAIN ROUTER (FIX) ----------------
if "role" not in st.session_state:
    login_page()
else:
    if st.session_state.role == "admin":
        admin_dashboard()
    elif st.session_state.role == "faculty":
        faculty_dashboard()
    elif st.session_state.role == "student":
        student_dashboard()
    else:
        st.error("Unknown role")
