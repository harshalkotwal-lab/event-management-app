---

## **4. config.py**
```python
"""
Configuration settings for G H Raisoni Event Management System
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / "data"
USERS_FILE = DATA_DIR / "users.json"
EVENTS_FILE = DATA_DIR / "events.json"
REGISTRATIONS_FILE = DATA_DIR / "registrations.json"

# Default credentials (CHANGE THESE IN PRODUCTION)
DEFAULT_CREDENTIALS = {
    "admin": {
        "username": "admin@raisoni",
        "password": "admin123",
        "name": "Administrator",
        "role": "admin",
        "department": "Administration",
        "email": "admin@ghraisoni.edu"
    },
    "faculty": {
        "username": "faculty@raisoni",
        "password": "faculty123",
        "name": "Faculty Coordinator",
        "role": "faculty",
        "department": "Faculty",
        "email": "faculty@ghraisoni.edu"
    }
}

# College configuration
COLLEGE_CONFIG = {
    "name": "G H Raisoni College of Engineering and Management",
    "departments": [
        "Computer Science & Engineering",
        "Artificial Intelligence & Machine Learning",
        "Electronics & Communication",
        "Electrical & Electronics",
        "Mechanical Engineering",
        "Civil Engineering",
        "Information Technology",
        "Data Science"
    ],
    "academic_years": ["I", "II", "III", "IV"],
    "event_types": [
        "Workshop",
        "Hackathon",
        "Competition",
        "Bootcamp",
        "Seminar",
        "Conference",
        "Webinar",
        "Training",
        "Symposium",
        "Cultural Event"
    ]
}

# Application settings
APP_CONFIG = {
    "title": "G H Raisoni Event Manager",
    "page_icon": "🎓",
    "layout": "wide",
    "initial_sidebar_state": "auto",
    "session_state_expiry": 86400,  # 24 hours in seconds
    "items_per_page": 10
}

# Feature flags
FEATURES = {
    "student_registration": True,
    "email_verification": False,
    "qr_checkin": False,
    "certificate_generation": False,
    "whatsapp_integration": False
}

# Email configuration (if enabled)
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "events@ghraisoni.edu",
    "use_tls": True
}
