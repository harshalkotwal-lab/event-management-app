"""
Authentication and authorization utilities
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
import streamlit as st

from config import DEFAULT_CREDENTIALS
from utils.database import DatabaseManager


class AuthManager:
    """Handles user authentication and session management"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.session_timeout = timedelta(hours=24)
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_default_credentials(self, username: str, password: str, role: str) -> bool:
        """Verify default admin/faculty credentials"""
        if role == 'admin':
            default = DEFAULT_CREDENTIALS['admin']
        elif role == 'faculty':
            default = DEFAULT_CREDENTIALS['faculty']
        else:
            return False
        
        return (username == default['username'] and 
                self.hash_password(password) == self.hash_password(default['password']))
    
    def verify_student_credentials(self, username: str, password: str) -> bool:
        """Verify student credentials from database"""
        user = self.db.get_user_by_username(username)
        if not user:
            return False
        
        hashed_input = self.hash_password(password)
        return user.get('password') == hashed_input and user.get('is_active', True)
    
    def create_session(self, username: str, role: str, user_data: Optional[Dict] = None):
        """Create user session"""
        st.session_state['authenticated'] = True
        st.session_state['username'] = username
        st.session_state['role'] = role
        st.session_state['login_time'] = datetime.now()
        
        if user_data:
            st.session_state['user_data'] = user_data
    
    def logout(self):
        """Clear user session"""
        for key in ['authenticated', 'username', 'role', 'login_time', 'user_data']:
            if key in st.session_state:
                del st.session_state[key]
    
    def is_session_valid(self) -> bool:
        """Check if session is still valid"""
        if not st.session_state.get('authenticated', False):
            return False
        
        login_time = st.session_state.get('login_time')
        if not login_time:
            return False
        
        if isinstance(login_time, str):
            login_time = datetime.fromisoformat(login_time)
        
        return datetime.now() - login_time < self.session_timeout
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current user data"""
        if not self.is_session_valid():
            return None
        
        username = st.session_state.get('username')
        if not username:
            return None
        
        # For admin/faculty, return default data
        role = st.session_state.get('role')
        if role == 'admin':
            return DEFAULT_CREDENTIALS['admin']
        elif role == 'faculty':
            return DEFAULT_CREDENTIALS['faculty']
        else:
            # For students, get from database
            return self.db.get_user_by_username(username)
    
    def require_auth(self, roles: list = None):
        """Decorator to require authentication for specific roles"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.is_session_valid():
                    st.error("Session expired. Please login again.")
                    st.stop()
                
                user_role = st.session_state.get('role')
                if roles and user_role not in roles:
                    st.error(f"Access denied. Required roles: {', '.join(roles)}")
                    st.stop()
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
