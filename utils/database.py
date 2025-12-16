"""
Database operations for Event Management System
"""

import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from config import DATA_DIR, USERS_FILE, EVENTS_FILE, REGISTRATIONS_FILE


class DatabaseManager:
    """Manages all database operations for the application"""
    
    def __init__(self):
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """Create data directory if it doesn't exist"""
        DATA_DIR.mkdir(exist_ok=True)
    
    def _load_json(self, file_path: Path) -> List[Dict]:
        """Load JSON data from file"""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
            return []
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_json(self, file_path: Path, data: List[Dict]):
        """Save data to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    # User operations
    def get_all_users(self) -> List[Dict]:
        return self._load_json(USERS_FILE)
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        users = self.get_all_users()
        for user in users:
            if user.get('username') == username:
                return user
        return None
    
    def save_user(self, user_data: Dict) -> bool:
        users = self.get_all_users()
        
        # Check if username already exists
        if any(u.get('username') == user_data.get('username') for u in users):
            return False
        
        # Add metadata
        user_data['user_id'] = str(uuid.uuid4())
        user_data['created_at'] = datetime.now().isoformat()
        user_data['is_active'] = True
        
        users.append(user_data)
        self._save_json(USERS_FILE, users)
        return True
    
    def update_user(self, username: str, updates: Dict) -> bool:
        users = self.get_all_users()
        for i, user in enumerate(users):
            if user.get('username') == username:
                users[i].update(updates)
                self._save_json(USERS_FILE, users)
                return True
        return False
    
    # Event operations
    def get_all_events(self) -> List[Dict]:
        return self._load_json(EVENTS_FILE)
    
    def get_event_by_id(self, event_id: str) -> Optional[Dict]:
        events = self.get_all_events()
        for event in events:
            if event.get('event_id') == event_id:
                return event
        return None
    
    def save_event(self, event_data: Dict) -> str:
        events = self.get_all_events()
        
        # Generate event ID
        event_id = str(uuid.uuid4())
        event_data['event_id'] = event_id
        event_data['created_at'] = datetime.now().isoformat()
        event_data['updated_at'] = datetime.now().isoformat()
        
        # Initialize engagement metrics
        event_data.setdefault('likes', [])
        event_data.setdefault('favorites', [])
        event_data.setdefault('interested', [])
        event_data.setdefault('views', 0)
        
        events.append(event_data)
        self._save_json(EVENTS_FILE, events)
        return event_id
    
    def update_event(self, event_id: str, updates: Dict) -> bool:
        events = self.get_all_events()
        for i, event in enumerate(events):
            if event.get('event_id') == event_id:
                updates['updated_at'] = datetime.now().isoformat()
                events[i].update(updates)
                self._save_json(EVENTS_FILE, events)
                return True
        return False
    
    def delete_event(self, event_id: str) -> bool:
        events = self.get_all_events()
        new_events = [e for e in events if e.get('event_id') != event_id]
        
        if len(new_events) != len(events):
            self._save_json(EVENTS_FILE, new_events)
            return True
        return False
    
    # Registration operations
    def get_all_registrations(self) -> List[Dict]:
        return self._load_json(REGISTRATIONS_FILE)
    
    def get_registrations_by_event(self, event_id: str) -> List[Dict]:
        registrations = self.get_all_registrations()
        return [r for r in registrations if r.get('event_id') == event_id]
    
    def get_registrations_by_student(self, username: str) -> List[Dict]:
        registrations = self.get_all_registrations()
        return [r for r in registrations if r.get('student_username') == username]
    
    def save_registration(self, registration_data: Dict) -> str:
        registrations = self.get_all_registrations()
        
        # Generate registration ID
        reg_id = str(uuid.uuid4())
        registration_data['registration_id'] = reg_id
        registration_data['registered_at'] = datetime.now().isoformat()
        registration_data.setdefault('status', 'pending')
        registration_data.setdefault('attendance', 'absent')
        
        registrations.append(registration_data)
        self._save_json(REGISTRATIONS_FILE, registrations)
        return reg_id
    
    def update_registration_status(self, registration_id: str, status: str) -> bool:
        registrations = self.get_all_registrations()
        for i, reg in enumerate(registrations):
            if reg.get('registration_id') == registration_id:
                registrations[i]['status'] = status
                registrations[i]['verified_at'] = datetime.now().isoformat()
                self._save_json(REGISTRATIONS_FILE, registrations)
                return True
        return False
    
    # Analytics functions
    def get_event_statistics(self) -> Dict:
        events = self.get_all_events()
        registrations = self.get_all_registrations()
        
        return {
            'total_events': len(events),
            'active_events': len([e for e in events if datetime.fromisoformat(e.get('event_date', '2000-01-01')) > datetime.now()]),
            'total_registrations': len(registrations),
            'unique_students': len(set(r.get('student_username') for r in registrations)),
            'event_types': pd.Series([e.get('event_type', 'Other') for e in events]).value_counts().to_dict()
        }
    
    def get_student_participation(self, username: str) -> Dict:
        registrations = self.get_registrations_by_student(username)
        events = self.get_all_events()
        
        student_events = []
        for reg in registrations:
            event = self.get_event_by_id(reg['event_id'])
            if event:
                student_events.append({
                    'event_title': event['title'],
                    'event_date': event.get('event_date'),
                    'registration_status': reg.get('status'),
                    'attendance': reg.get('attendance')
                })
        
        return {
            'total_registered': len(registrations),
            'events_attended': len([r for r in registrations if r.get('attendance') == 'present']),
            'events_list': student_events
      }
