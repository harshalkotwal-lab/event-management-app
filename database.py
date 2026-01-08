# database.py
"""
Database management for the Event Management System
"""
import os
import uuid
import hashlib
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import streamlit as st

logger = logging.getLogger(__name__)

class SupabaseClient:
    """Supabase PostgreSQL client"""
    def __init__(self):
        self.url = None
        self.key = None
        self.headers = None
        self.is_configured = False
        self._initialize()
    
    def _initialize(self):
        """Initialize Supabase connection"""
        try:
            if hasattr(st, 'secrets') and 'SUPABASE' in st.secrets:
                self.url = st.secrets.SUPABASE.get('url', '').rstrip('/')
                self.key = st.secrets.SUPABASE.get('key', '')
                
                if self.url and self.key:
                    if not self.url.startswith(('http://', 'https://')):
                        self.url = f'https://{self.url}'
                    
                    self.headers = {
                        'apikey': self.key,
                        'Authorization': f'Bearer {self.key}',
                        'Content-Type': 'application/json',
                        'Prefer': 'return=minimal'
                    }
                    self.is_configured = True
                    logger.info("✅ Supabase configured successfully")
            else:
                logger.warning("⚠️ Supabase credentials not found")
        except Exception as e:
            logger.error(f"Supabase init error: {e}")
    
    def execute_query(self, table, method='GET', data=None, filters=None):
        """Execute REST API query"""
        if not self.is_configured:
            return None
        
        try:
            import requests
            url = f"{self.url}/rest/v1/{table}"
            
            # Add filters
            params = []
            if filters:
                for k, v in filters.items():
                    if v is not None:
                        params.append(f"{k}=eq.{v}")
            
            if params:
                url = f"{url}?{'&'.join(params)}"
            
            # Make request
            timeout = 10
            if method == 'GET':
                response = requests.get(url, headers=self.headers, timeout=timeout)
                return response.json() if response.text else []
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=data, timeout=timeout)
                return response.status_code in [200, 201]
            elif method == 'PATCH':
                response = requests.patch(url, headers=self.headers, json=data, timeout=timeout)
                return response.status_code in [200, 204]
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers, timeout=timeout)
                return response.status_code in [200, 204]
        except Exception as e:
            logger.error(f"Supabase error: {e}")
            return None

class SQLiteClient:
    """SQLite client for local development"""
    def __init__(self, db_path="data/event_management.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize()
    
    def _initialize(self):
        """Initialize SQLite database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info("✅ SQLite database initialized")
        except Exception as e:
            logger.error(f"SQLite initialization error: {e}")
            raise
    
    def execute(self, query, params=None, fetchone=False, fetchall=False, commit=False):
        """Execute SQL query"""
        try:
            cursor = self.conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if commit:
                self.conn.commit()
            
            if fetchone:
                result = cursor.fetchone()
                return dict(result) if result else None
            
            if fetchall:
                results = cursor.fetchall()
                return [dict(row) for row in results]
            
            return cursor
        except Exception as e:
            logger.error(f"SQLite error: {e}")
            return None

class DatabaseManager:
    """Unified database manager"""
    def __init__(self, use_supabase=True):
        self.use_supabase = use_supabase
        
        if self.use_supabase:
            self.client = SupabaseClient()
            if not self.client.is_configured:
                logger.warning("Falling back to SQLite")
                self.use_supabase = False
                self.client = SQLiteClient()
        else:
            self.client = SQLiteClient()
        
        self._initialize_tables()
    
    def _initialize_tables(self):
        """Initialize database tables"""
        if not self.use_supabase:
            self._create_sqlite_tables()
        
        # Add default users
        self._add_default_users()
    
    def _create_sqlite_tables(self):
        """Create SQLite tables"""
        tables = [
            """CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                department TEXT,
                email TEXT,
                mobile TEXT,
                roll_no TEXT,
                year TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            
            """CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                event_type TEXT,
                event_date TIMESTAMP,
                venue TEXT,
                organizer TEXT,
                max_participants INTEGER DEFAULT 100,
                created_by TEXT,
                status TEXT DEFAULT 'upcoming',
                flyer_path TEXT,
                ai_generated BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""",
            
            """CREATE TABLE IF NOT EXISTS registrations (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                student_username TEXT NOT NULL,
                student_name TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_id, student_username)
            )""",
            
            """CREATE TABLE IF NOT EXISTS event_likes (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                student_username TEXT NOT NULL,
                liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(event_id, student_username)
            )""",
            
            """CREATE TABLE IF NOT EXISTS mentors (
                id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                department TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                contact TEXT NOT NULL,
                expertise TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        ]
        
        for table_sql in tables:
            self.client.execute(table_sql, commit=True)
    
    def _add_default_users(self):
        """Add default users"""
        default_users = [
            {
                'username': 'admin@raisoni',
                'password': hashlib.sha256('Admin@12345'.encode()).hexdigest(),
                'name': 'Administrator',
                'role': 'admin',
                'email': 'admin@ghraisoni.edu'
            },
            {
                'username': 'faculty@raisoni',
                'password': hashlib.sha256('Faculty@12345'.encode()).hexdigest(),
                'name': 'Faculty Coordinator',
                'role': 'faculty',
                'email': 'faculty@ghraisoni.edu'
            }
        ]
        
        for user in default_users:
            self.add_user(user)
    
    # User methods
    def add_user(self, user_data):
        """Add new user"""
        try:
            user_id = str(uuid.uuid4())
            user_record = {
                'id': user_id,
                'username': user_data['username'],
                'password': user_data['password'],
                'name': user_data['name'],
                'role': user_data.get('role', 'student'),
                'department': user_data.get('department', ''),
                'email': user_data.get('email', ''),
                'mobile': user_data.get('mobile', ''),
                'roll_no': user_data.get('roll_no', ''),
                'year': user_data.get('year', '')
            }
            
            if self.use_supabase:
                return self.client.execute_query('users', 'POST', user_record)
            else:
                query = """INSERT INTO users (id, username, password, name, role, department, email, mobile, roll_no, year) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                return self.client.execute(query, tuple(user_record.values()), commit=True)
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return False
    
    def get_user(self, username):
        """Get user by username"""
        try:
            if self.use_supabase:
                result = self.client.execute_query('users', filters={'username': username})
                return result[0] if result else None
            else:
                query = "SELECT * FROM users WHERE username = ?"
                return self.client.execute(query, (username,), fetchone=True)
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials"""
        try:
            user = self.get_user(username)
            if not user:
                return False
            
            if user['role'] != role:
                return False
            
            # Check password
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            return user['password'] == hashed_password or user['password'] == password
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    # Event methods
    def add_event(self, event_data):
        """Add new event"""
        try:
            event_id = str(uuid.uuid4())
            event_record = {
                'id': event_id,
                'title': event_data['title'],
                'description': event_data.get('description', ''),
                'event_type': event_data.get('event_type', 'Workshop'),
                'event_date': event_data.get('event_date', datetime.now().isoformat()),
                'venue': event_data.get('venue', ''),
                'organizer': event_data.get('organizer', ''),
                'max_participants': event_data.get('max_participants', 100),
                'created_by': event_data.get('created_by', ''),
                'flyer_path': event_data.get('flyer_path', ''),
                'ai_generated': event_data.get('ai_generated', False)
            }
            
            if self.use_supabase:
                return self.client.execute_query('events', 'POST', event_record)
            else:
                query = """INSERT INTO events (id, title, description, event_type, event_date, venue, 
                          organizer, max_participants, created_by, flyer_path, ai_generated) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                return self.client.execute(query, tuple(event_record.values()), commit=True)
        except Exception as e:
            logger.error(f"Error adding event: {e}")
            return False
    
    def get_all_events(self):
        """Get all events"""
        try:
            if self.use_supabase:
                return self.client.execute_query('events')
            else:
                query = "SELECT * FROM events ORDER BY event_date DESC"
                return self.client.execute(query, fetchall=True)
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []
    
    def get_events_by_creator(self, username):
        """Get events created by user"""
        try:
            if self.use_supabase:
                return self.client.execute_query('events', filters={'created_by': username})
            else:
                query = "SELECT * FROM events WHERE created_by = ? ORDER BY event_date DESC"
                return self.client.execute(query, (username,), fetchall=True)
        except Exception as e:
            logger.error(f"Error getting creator events: {e}")
            return []
    
    # Registration methods
    def add_registration(self, reg_data):
        """Add new registration"""
        try:
            # Check if already registered
            existing = self.get_registration(reg_data['event_id'], reg_data['student_username'])
            if existing:
                return False, "Already registered"
            
            reg_id = str(uuid.uuid4())
            reg_record = {
                'id': reg_id,
                'event_id': reg_data['event_id'],
                'student_username': reg_data['student_username'],
                'student_name': reg_data['student_name']
            }
            
            if self.use_supabase:
                success = self.client.execute_query('registrations', 'POST', reg_record)
            else:
                query = "INSERT INTO registrations (id, event_id, student_username, student_name) VALUES (?, ?, ?, ?)"
                success = self.client.execute(query, tuple(reg_record.values()), commit=True)
            
            return success, "Registration successful" if success else "Registration failed"
        except Exception as e:
            logger.error(f"Error adding registration: {e}")
            return False, "Registration failed"
    
    def get_registration(self, event_id, username):
        """Get registration"""
        try:
            if self.use_supabase:
                result = self.client.execute_query('registrations', filters={
                    'event_id': event_id,
                    'student_username': username
                })
                return result[0] if result else None
            else:
                query = "SELECT * FROM registrations WHERE event_id = ? AND student_username = ?"
                return self.client.execute(query, (event_id, username), fetchone=True)
        except Exception as e:
            logger.error(f"Error getting registration: {e}")
            return None
    
    def get_registrations_by_student(self, username):
        """Get all registrations for a student"""
        try:
            if self.use_supabase:
                return self.client.execute_query('registrations', filters={'student_username': username})
            else:
                query = "SELECT r.*, e.title as event_title, e.event_date FROM registrations r LEFT JOIN events e ON r.event_id = e.id WHERE r.student_username = ? ORDER BY r.registered_at DESC"
                return self.client.execute(query, (username,), fetchall=True)
        except Exception as e:
            logger.error(f"Error getting student registrations: {e}")
            return []
    
    # Like methods
    def add_like(self, event_id, username):
        """Add like to event"""
        try:
            like_record = {
                'id': str(uuid.uuid4()),
                'event_id': event_id,
                'student_username': username
            }
            
            if self.use_supabase:
                return self.client.execute_query('event_likes', 'POST', like_record)
            else:
                query = "INSERT INTO event_likes (id, event_id, student_username) VALUES (?, ?, ?)"
                return self.client.execute(query, tuple(like_record.values()), commit=True)
        except Exception as e:
            logger.error(f"Error adding like: {e}")
            return False
    
    def remove_like(self, event_id, username):
        """Remove like from event"""
        try:
            if self.use_supabase:
                return self.client.execute_query('event_likes', 'DELETE', filters={
                    'event_id': event_id,
                    'student_username': username
                })
            else:
                query = "DELETE FROM event_likes WHERE event_id = ? AND student_username = ?"
                return self.client.execute(query, (event_id, username), commit=True)
        except Exception as e:
            logger.error(f"Error removing like: {e}")
            return False
    
    def is_event_liked(self, event_id, username):
        """Check if event is liked by user"""
        try:
            if self.use_supabase:
                result = self.client.execute_query('event_likes', filters={
                    'event_id': event_id,
                    'student_username': username
                })
                return bool(result)
            else:
                query = "SELECT id FROM event_likes WHERE event_id = ? AND student_username = ?"
                return bool(self.client.execute(query, (event_id, username), fetchone=True))
        except Exception as e:
            logger.error(f"Error checking like: {e}")
            return False
    
    def get_event_likes_count(self, event_id):
        """Get like count for event"""
        try:
            if self.use_supabase:
                likes = self.client.execute_query('event_likes', filters={'event_id': event_id})
                return len(likes) if likes else 0
            else:
                query = "SELECT COUNT(*) as count FROM event_likes WHERE event_id = ?"
                result = self.client.execute(query, (event_id,), fetchone=True)
                return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting likes count: {e}")
            return 0
    
    # System stats
    def get_system_stats(self):
        """Get system statistics"""
        try:
            stats = {}
            
            # User counts
            users = self.get_all_users()
            if users:
                role_counts = {}
                for user in users:
                    role = user.get('role', 'unknown')
                    role_counts[role] = role_counts.get(role, 0) + 1
                stats['users'] = role_counts
            
            # Event counts
            events = self.get_all_events()
            if events:
                stats['total_events'] = len(events)
                stats['ai_events'] = len([e for e in events if e.get('ai_generated')])
            
            # Registration count
            if self.use_supabase:
                regs = self.client.execute_query('registrations')
                stats['registrations'] = len(regs) if regs else 0
            else:
                result = self.client.execute("SELECT COUNT(*) as count FROM registrations", fetchone=True)
                stats['registrations'] = result['count'] if result else 0
            
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def get_all_users(self):
        """Get all users"""
        try:
            if self.use_supabase:
                return self.client.execute_query('users')
            else:
                return self.client.execute("SELECT * FROM users", fetchall=True)
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return []
    
    def update_event_status(self):
        """Update event status based on date"""
        try:
            now = datetime.now()
            
            if self.use_supabase:
                # Update past events
                self.client.execute_query('events', 'PATCH', {'status': 'past'}, 
                                        {'event_date': {'lt': now.isoformat()}, 'status': {'neq': 'past'}})
                # Update ongoing events (today)
                today_start = now.replace(hour=0, minute=0, second=0).isoformat()
                today_end = now.replace(hour=23, minute=59, second=59).isoformat()
                self.client.execute_query('events', 'PATCH', {'status': 'ongoing'}, 
                                        {'event_date': {'gte': today_start, 'lte': today_end}, 'status': 'upcoming'})
            else:
                # Update past events
                self.client.execute(
                    "UPDATE events SET status = 'past' WHERE event_date <= ? AND status != 'past'",
                    (now.isoformat(),), commit=True
                )
                # Update ongoing events
                today_start = now.replace(hour=0, minute=0, second=0)
                today_end = now.replace(hour=23, minute=59, second=59)
                self.client.execute(
                    "UPDATE events SET status = 'ongoing' WHERE event_date BETWEEN ? AND ? AND status = 'upcoming'",
                    (today_start.isoformat(), today_end.isoformat()), commit=True
                )
            
            return True
        except Exception as e:
            logger.error(f"Error updating event status: {e}")
            return False
