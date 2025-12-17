"""
G H Raisoni College - Advanced Event Management System
Complete solution with AI, Image Uploads, Social Features
Deployable on Streamlit Cloud
"""

import streamlit as st
from datetime import datetime, date, timedelta
import json
import os
import hashlib
import uuid
from pathlib import Path
import pandas as pd
from PIL import Image
import io
import re
import requests
from streamlit_option_menu import option_menu
import base64

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
    
    .ai-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .social-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .social-btn {
        flex: 1;
        min-width: 80px;
        text-align: center;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        background: white;
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .social-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .social-btn.active {
        border-color: #3B82F6;
        background: #DBEAFE;
        color: #1E40AF;
    }
    
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
    
    .flyer-container {
        border: 2px dashed #3B82F6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
        background: #F8FAFC;
    }
    
    .registration-section {
        background: #F0F9FF;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA MANAGER (JSON Storage)
# ============================================
class JSONDataManager:
    """Manage all data using JSON files"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Uploads directory for event flyers
        self.uploads_dir = Path("uploads")
        self.uploads_dir.mkdir(exist_ok=True)
        
        self.files = {
            'users': self.data_dir / "users.json",
            'events': self.data_dir / "events.json",
            'registrations': self.data_dir / "registrations.json",
            'social': self.data_dir / "social_interactions.json",
            'analytics': self.data_dir / "analytics.json"
        }
        
        # Initialize files
        self._init_files()
        
        # Default credentials
        self.default_creds = {
            'admin': {'username': 'admin@raisoni', 'password': 'admin123'},
            'faculty': {'username': 'faculty@raisoni', 'password': 'faculty123'}
        }
    
    def _init_files(self):
        """Initialize JSON files if they don't exist"""
        for file_path in self.files.values():
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump([], f, indent=2)
    
    def _hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_credentials(self, username, password, role):
        """Verify user credentials"""
        if role in ['admin', 'faculty']:
            creds = self.default_creds[role]
            return (username == creds['username'] and 
                    self._hash_password(password) == self._hash_password(creds['password']))
        else:
            users = self.load('users')
            for user in users:
                if user.get('username') == username and user.get('password') == self._hash_password(password):
                    return True
            return False
    
    def load(self, data_type):
        """Load data from JSON file"""
        try:
            with open(self.files[data_type], 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save(self, data_type, data):
        """Save data to JSON file"""
        try:
            with open(self.files[data_type], 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except:
            return False
    
    def add_item(self, data_type, item_data):
        """Add new item to data"""
        data = self.load(data_type)
        
        # Generate unique ID
        if 'id' not in item_data:
            item_data['id'] = str(uuid.uuid4())
        
        # Add timestamps
        item_data['created_at'] = datetime.now().isoformat()
        item_data['updated_at'] = datetime.now().isoformat()
        
        data.append(item_data)
        return self.save(data_type, data)
    
    def update_item(self, data_type, item_id, updates):
        """Update existing item"""
        data = self.load(data_type)
        for i, item in enumerate(data):
            if item.get('id') == item_id:
                updates['updated_at'] = datetime.now().isoformat()
                data[i].update(updates)
                return self.save(data_type, data)
        return False
    
    def get_item(self, data_type, item_id):
        """Get item by ID"""
        data = self.load(data_type)
        for item in data:
            if item.get('id') == item_id:
                return item
        return None
    
    def get_items_by_field(self, data_type, field, value):
        """Get items by field value"""
        data = self.load(data_type)
        return [item for item in data if item.get(field) == value]
    
    def save_image(self, uploaded_file, event_id=None):
        """Save uploaded image and return path"""
        if uploaded_file is None:
            return None
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        if event_id:
            filename = f"event_{event_id}_{timestamp}{file_ext}"
        else:
            filename = f"{timestamp}{file_ext}"
        
        file_path = self.uploads_dir / filename
        
        try:
            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Optimize image
            self._optimize_image(file_path)
            
            return str(file_path)
        except Exception as e:
            st.error(f"Error saving image: {e}")
            return None
    
    def _optimize_image(self, image_path, max_size=(1200, 1200)):
        """Optimize image size"""
        try:
            with Image.open(image_path) as img:
                # Resize if too large
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    img.save(image_path, optimize=True, quality=85)
        except:
            pass

# Initialize data manager
data_manager = JSONDataManager()

# ============================================
# AI EVENT GENERATOR
# ============================================
class AIEventGenerator:
    """Generate events from WhatsApp/email messages"""
    
    def __init__(self):
        # Try to get OpenAI API key from secrets
        try:
            self.api_key = st.secrets.get("OPENAI_API_KEY", "")
        except:
            self.api_key = ""
    
    def extract_event_info(self, text):
        """Extract event info using AI or regex fallback"""
        
        # Try AI if API key is available
        if self.api_key:
            try:
                return self._extract_with_ai(text)
            except:
                pass
        
        # Fallback to regex extraction
        return self._extract_with_regex(text)
    
    def _extract_with_ai(self, text):
        """Use OpenAI API to extract event info"""
        import openai
        
        client = openai.OpenAI(api_key=self.api_key)
        
        prompt = f"""
        Extract event information from this message and return as JSON:
        
        {text}
        
        Return JSON with these fields:
        - title: Event title
        - description: Event description
        - event_type: workshop/hackathon/competition/bootcamp/seminar/conference/webinar
        - event_date: YYYY-MM-DD (extract from text or use reasonable future date)
        - venue: Event location
        - organizer: Who is organizing
        - registration_link: URL if mentioned
        
        Return ONLY JSON, no other text.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract event information and return JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean response
        result = result.replace('```json', '').replace('```', '').strip()
        
        try:
            event_data = json.loads(result)
            event_data['ai_generated'] = True
            event_data['ai_prompt'] = text
            return event_data
        except:
            return self._extract_with_regex(text)
    
    def _extract_with_regex(self, text):
        """Regex-based event extraction"""
        event_data = {
            'title': 'New Event',
            'description': text[:200] + '...' if len(text) > 200 else text,
            'event_type': 'workshop',
            'event_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'venue': 'G H Raisoni College',
            'organizer': 'College Department',
            'registration_link': '',
            'ai_generated': False
        }
        
        # Extract title (first line)
        lines = text.split('\n')
        if lines and lines[0].strip():
            event_data['title'] = lines[0].strip()[:100]
        
        # Extract date
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['event_date'] = match.group(1)
                break
        
        # Extract venue
        venue_keywords = ['at', 'venue', 'location', 'place']
        for keyword in venue_keywords:
            pattern = rf'{keyword}[:\s]*([^\n.,;]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['venue'] = match.group(1).strip()
                break
        
        # Extract organizer
        organizer_keywords = ['by', 'organizer', 'organized by', 'conducted by']
        for keyword in organizer_keywords:
            pattern = rf'{keyword}[:\s]*([^\n.,;]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['organizer'] = match.group(1).strip()
                break
        
        # Extract URL
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        if urls:
            event_data['registration_link'] = urls[0]
        
        return event_data

# ============================================
# HELPER FUNCTIONS
# ============================================
def display_role_badge(role):
    """Display role badge"""
    badges = {
        "admin": ("👑 Admin", "admin-badge"),
        "faculty": ("👨‍🏫 Faculty", "faculty-badge"),
        "student": ("👨‍🎓 Student", "student-badge")
    }
    
    if role in badges:
        text, css_class = badges[role]
        st.markdown(f'<span class="role-badge {css_class}">{text}</span>', 
                   unsafe_allow_html=True)

def format_date(date_str):
    """Format date for display"""
    try:
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            dt = date_str
        return dt.strftime("%d %b %Y, %I:%M %p")
    except:
        return str(date_str)

def is_upcoming(event_date):
    """Check if event is upcoming"""
    try:
        if isinstance(event_date, str):
            dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
        else:
            dt = event_date
        return dt > datetime.now()
    except:
        return True

# ============================================
# LOGIN PAGE
# ============================================
def login_page():
    """Display login page"""
    st.markdown('<div class="college-header"><h2>G H Raisoni College of Engineering and Management</h2><p>Advanced Event Management System</p></div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Admin Login")
        admin_user = st.text_input("Username", key="admin_user")
        admin_pass = st.text_input("Password", type="password", key="admin_pass")
        
        if st.button("Admin Login", use_container_width=True):
            if data_manager.verify_credentials(admin_user, admin_pass, 'admin'):
                st.session_state.role = 'admin'
                st.session_state.username = admin_user
                st.session_state.name = "Administrator"
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with col2:
        st.subheader("Faculty Login")
        faculty_user = st.text_input("Username", key="faculty_user")
        faculty_pass = st.text_input("Password", type="password", key="faculty_pass")
        
        if st.button("Faculty Login", use_container_width=True):
            if data_manager.verify_credentials(faculty_user, faculty_pass, 'faculty'):
                st.session_state.role = 'faculty'
                st.session_state.username = faculty_user
                st.session_state.name = "Faculty Coordinator"
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with col3:
        st.subheader("Student Portal")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            student_user = st.text_input("Username", key="student_user_login")
            student_pass = st.text_input("Password", type="password", key="student_pass_login")
            
            if st.button("Student Login", use_container_width=True):
                if data_manager.verify_credentials(student_user, student_pass, 'student'):
                    # Get student info
                    users = data_manager.load('users')
                    student = next((u for u in users if u.get('username') == student_user), None)
                    
                    if student:
                        st.session_state.role = 'student'
                        st.session_state.username = student_user
                        st.session_state.name = student.get('name', student_user)
                        st.session_state.user_id = student.get('id')
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("User not found")
                else:
                    st.error("Invalid credentials")
        
        with tab2:
            with st.form("student_registration"):
                st.markdown("**Create Student Account**")
                
                name = st.text_input("Full Name *")
                roll_no = st.text_input("Roll Number *")
                department = st.selectbox("Department *", 
                                         ["CSE", "AIML", "ECE", "EEE", "MECH", "CIVIL", "IT", "DS"])
                year = st.selectbox("Year *", ["I", "II", "III", "IV"])
                email = st.text_input("Email *")
                username = st.text_input("Username *")
                password = st.text_input("Password *", type="password")
                confirm_pass = st.text_input("Confirm Password *", type="password")
                
                if st.form_submit_button("Register", use_container_width=True):
                    if password != confirm_pass:
                        st.error("Passwords don't match")
                    elif not all([name, roll_no, email, username, password]):
                        st.error("Please fill all required fields")
                    else:
                        users = data_manager.load('users')
                        
                        # Check if username exists
                        if any(u.get('username') == username for u in users):
                            st.error("Username already exists")
                        else:
                            user_data = {
                                'id': str(uuid.uuid4()),
                                'name': name,
                                'roll_no': roll_no,
                                'department': department,
                                'year': year,
                                'email': email,
                                'username': username,
                                'password': data_manager._hash_password(password),
                                'role': 'student',
                                'created_at': datetime.now().isoformat()
                            }
                            
                            if data_manager.add_item('users', user_data):
                                st.success("Registration successful! Please login.")
                                st.rerun()
                            else:
                                st.error("Registration failed")

# ============================================
# AI EVENT CREATION
# ============================================
def ai_event_creation():
    """Create event using AI"""
    st.header("🤖 AI-Powered Event Creation")
    
    ai_gen = AIEventGenerator()
    
    tab1, tab2 = st.tabs(["From Text", "Upload File"])
    
    with tab1:
        st.subheader("Paste Event Details")
        event_text = st.text_area("Paste WhatsApp message, email, or event details:", 
                                 height=200,
                                 placeholder="""Example:
🎯 Hackathon Alert!
Join our AI Hackathon on Dec 20-21, 2024 at Seminar Hall.
Organized by CSE Department.
Register: https://forms.gle/example
Prizes: ₹50,000""")
        
        generate_button = st.button("Generate Event", use_container_width=True, key="generate_ai_event")
        
        # Store generated event data in session state
        if 'ai_generated_data' not in st.session_state:
            st.session_state.ai_generated_data = None
        
        if generate_button and event_text:
            with st.spinner("Extracting event details..."):
                event_data = ai_gen.extract_event_info(event_text)
                st.session_state.ai_generated_data = event_data
                st.session_state.ai_event_text = event_text
                st.rerun()
        
        # Show form if we have generated data
        if st.session_state.ai_generated_data:
            event_data = st.session_state.ai_generated_data
            event_text = st.session_state.get('ai_event_text', '')
            
            st.subheader("📋 Extracted Event")
            
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Title", value=event_data.get('title', ''), 
                             key="ai_title", disabled=True)
                st.text_area("Description", value=event_data.get('description', ''), 
                            height=100, key="ai_desc", disabled=True)
                st.text_input("Type", value=event_data.get('event_type', ''), 
                             key="ai_type", disabled=True)
            
            with col2:
                st.text_input("Date", value=event_data.get('event_date', ''), 
                             key="ai_date", disabled=True)
                st.text_input("Venue", value=event_data.get('venue', ''), 
                             key="ai_venue", disabled=True)
                st.text_input("Organizer", value=event_data.get('organizer', ''), 
                             key="ai_org", disabled=True)
                st.text_input("Registration Link", 
                             value=event_data.get('registration_link', ''), 
                             key="ai_link", disabled=True)
            
            # Allow editing
            st.subheader("✏️ Edit & Finalize")
            
            with st.form("finalize_event_form"):
                title = st.text_input("Event Title *", 
                                     value=event_data.get('title', ''))
                description = st.text_area("Description *", 
                                          value=event_data.get('description', ''),
                                          height=150)
                
                # Handle the case where event_type might not be in the list
                extracted_type = event_data.get('event_type', 'Workshop')
                event_type_options = ["Workshop", "Hackathon", "Competition", 
                                     "Bootcamp", "Seminar", "Conference", "Webinar"]
                
                # Safely get index
                try:
                    default_index = event_type_options.index(extracted_type)
                except ValueError:
                    default_index = 0  # Default to Workshop if not found
                
                event_type = st.selectbox("Event Type *", 
                                         event_type_options,
                                         index=default_index)
                
                col_date, col_time = st.columns(2)
                with col_date:
                    try:
                        # Try to parse the date
                        date_str = event_data.get('event_date', date.today().isoformat())
                        parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        event_date_input = st.date_input("Date *", value=parsed_date)
                    except:
                        event_date_input = st.date_input("Date *", min_value=date.today())
                with col_time:
                    event_time_input = st.time_input("Time *", value=datetime.now().time())
                
                venue = st.text_input("Venue *", value=event_data.get('venue', ''))
                organizer = st.text_input("Organizer *", value=event_data.get('organizer', ''))
                reg_link = st.text_input("Registration Link", 
                                        value=event_data.get('registration_link', ''))
                
                # Flyer upload
                st.subheader("📸 Event Flyer (Optional)")
                flyer = st.file_uploader("Upload flyer image", 
                                        type=['jpg', 'jpeg', 'png', 'gif'],
                                        key="ai_flyer_uploader")
                
                submit_button = st.form_submit_button("Create Event", use_container_width=True)
                
                if submit_button:
                    if not all([title, description, venue, organizer]):
                        st.error("Please fill all required fields (*)")
                    else:
                        # Combine date and time
                        event_datetime = datetime.combine(event_date_input, event_time_input)
                        
                        # Save flyer
                        flyer_path = None
                        if flyer:
                            flyer_path = data_manager.save_image(flyer)
                        
                        event_to_save = {
                            'id': str(uuid.uuid4()),
                            'title': title,
                            'description': description,
                            'event_type': event_type,
                            'event_date': event_datetime.isoformat(),
                            'venue': venue,
                            'organizer': organizer,
                            'registration_link': reg_link,
                            'flyer_path': flyer_path,
                            'created_by': st.session_state.username,
                            'created_by_name': st.session_state.name,
                            'ai_generated': event_data.get('ai_generated', False),
                            'ai_prompt': event_text if event_data.get('ai_generated') else None,
                            'social_stats': {
                                'likes': [],
                                'favorites': [],
                                'interested': [],
                                'shares': 0,
                                'views': 0
                            }
                        }
                        
                        if data_manager.add_item('events', event_to_save):
                            st.success("Event created successfully! 🎉")
                            st.balloons()
                            # Clear session state
                            st.session_state.ai_generated_data = None
                            st.session_state.ai_event_text = None
                            st.rerun()
                        else:
                            st.error("Failed to save event")
    
    with tab2:
        st.subheader("Upload File")
        uploaded_file = st.file_uploader("Upload text file", type=['txt'], key="ai_file_upload")
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            st.text_area("File Content", content, height=200, key="ai_file_content")
            
            if st.button("Extract from File", use_container_width=True, key="extract_from_file"):
                with st.spinner("Extracting event details..."):
                    event_data = ai_gen.extract_event_info(content)
                    # Store in session state
                    st.session_state.ai_generated_data = event_data
                    st.session_state.ai_event_text = content
                    st.rerun()
# ============================================
# EVENT CARD WITH SOCIAL FEATURES
# ============================================
def display_event_card_social(event, current_user=None):
    """Display event card with social features"""
    event_id = event.get('id')
    
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        # Header with AI badge
        col_header = st.columns([4, 1])
        with col_header[0]:
            st.subheader(event.get('title', 'Untitled Event'))
            if event.get('ai_generated'):
                st.markdown('<span class="ai-badge">🤖 AI Generated</span>', 
                           unsafe_allow_html=True)
        with col_header[1]:
            if is_upcoming(event.get('event_date')):
                st.success("🟢 Upcoming")
            else:
                st.error("🔴 Completed")
        
        # Event flyer
        if event.get('flyer_path') and os.path.exists(event.get('flyer_path')):
            try:
                image = Image.open(event.get('flyer_path'))
                st.image(image, width=300, caption="Event Flyer")
            except:
                pass
        
        # Description
        desc = event.get('description', 'No description')
        if len(desc) > 300:
            desc = desc[:300] + "..."
        st.write(desc)
        
        # Details
        col_details = st.columns(4)
        with col_details[0]:
            st.caption(f"**📅 Date:** {format_date(event.get('event_date'))}")
        with col_details[1]:
            st.caption(f"**📍 Venue:** {event.get('venue', 'N/A')}")
        with col_details[2]:
            st.caption(f"**🏷️ Type:** {event.get('event_type', 'N/A')}")
        with col_details[3]:
            st.caption(f"**👨‍🏫 Organizer:** {event.get('organizer', 'N/A')}")
        
        # Social buttons (only for logged-in users)
        if current_user:
            # Load fresh data for social stats
            events = data_manager.load('events')
            current_event = next((e for e in events if e.get('id') == event_id), event)
            social_stats = current_event.get('social_stats', {})
            
            # Get user's interactions
            user_interactions = {
                'liked': current_user in social_stats.get('likes', []),
                'favorited': current_user in social_stats.get('favorites', []),
                'interested': current_user in social_stats.get('interested', [])
            }
            
            st.markdown("---")
            st.markdown("**Social Interactions**")
            
            col_social = st.columns(5)
            
            with col_social[0]:
                like_icon = "❤️" if user_interactions['liked'] else "🤍"
                like_button = st.button(f"{like_icon} Like", 
                                      key=f"like_{event_id}_{current_user}",
                                      use_container_width=True)
                
                if like_button:
                    # Load fresh events data
                    events = data_manager.load('events')
                    event_to_update = None
                    event_index = -1
                    
                    # Find the event
                    for i, e in enumerate(events):
                        if e.get('id') == event_id:
                            event_to_update = e
                            event_index = i
                            break
                    
                    if event_to_update:
                        # Initialize social_stats if not present
                        if 'social_stats' not in event_to_update:
                            event_to_update['social_stats'] = {
                                'likes': [],
                                'favorites': [],
                                'interested': [],
                                'shares': 0,
                                'views': 0
                            }
                        
                        # Toggle like
                        likes = event_to_update['social_stats'].get('likes', [])
                        if current_user in likes:
                            likes.remove(current_user)
                        else:
                            likes.append(current_user)
                        
                        # Update the event in the list
                        events[event_index]['social_stats']['likes'] = likes
                        
                        # Save back to file
                        if data_manager.save('events', events):
                            st.success("Like updated!")
                            st.rerun()
                
                st.caption(f"{len(social_stats.get('likes', []))} likes")
            
            with col_social[1]:
                fav_icon = "⭐" if user_interactions['favorited'] else "☆"
                fav_button = st.button(f"{fav_icon} Favorite", 
                                     key=f"fav_{event_id}_{current_user}",
                                     use_container_width=True)
                
                if fav_button:
                    # Load fresh events data
                    events = data_manager.load('events')
                    event_to_update = None
                    event_index = -1
                    
                    # Find the event
                    for i, e in enumerate(events):
                        if e.get('id') == event_id:
                            event_to_update = e
                            event_index = i
                            break
                    
                    if event_to_update:
                        # Initialize social_stats if not present
                        if 'social_stats' not in event_to_update:
                            event_to_update['social_stats'] = {
                                'likes': [],
                                'favorites': [],
                                'interested': [],
                                'shares': 0,
                                'views': 0
                            }
                        
                        # Toggle favorite
                        favorites = event_to_update['social_stats'].get('favorites', [])
                        if current_user in favorites:
                            favorites.remove(current_user)
                        else:
                            favorites.append(current_user)
                        
                        # Update the event in the list
                        events[event_index]['social_stats']['favorites'] = favorites
                        
                        # Save back to file
                        if data_manager.save('events', events):
                            st.success("Favorite updated!")
                            st.rerun()
                
                st.caption(f"{len(social_stats.get('favorites', []))} favorites")
            
            with col_social[2]:
                int_icon = "✅" if user_interactions['interested'] else "🤔"
                int_button = st.button(f"{int_icon} Interested", 
                                     key=f"int_{event_id}_{current_user}",
                                     use_container_width=True)
                
                if int_button:
                    # Load fresh events data
                    events = data_manager.load('events')
                    event_to_update = None
                    event_index = -1
                    
                    # Find the event
                    for i, e in enumerate(events):
                        if e.get('id') == event_id:
                            event_to_update = e
                            event_index = i
                            break
                    
                    if event_to_update:
                        # Initialize social_stats if not present
                        if 'social_stats' not in event_to_update:
                            event_to_update['social_stats'] = {
                                'likes': [],
                                'favorites': [],
                                'interested': [],
                                'shares': 0,
                                'views': 0
                            }
                        
                        # Toggle interested
                        interested = event_to_update['social_stats'].get('interested', [])
                        if current_user in interested:
                            interested.remove(current_user)
                        else:
                            interested.append(current_user)
                        
                        # Update the event in the list
                        events[event_index]['social_stats']['interested'] = interested
                        
                        # Save back to file
                        if data_manager.save('events', events):
                            st.success("Interest updated!")
                            st.rerun()
                
                st.caption(f"{len(social_stats.get('interested', []))} interested")
            
            with col_social[3]:
                share_button = st.button("📤 Share", 
                                       key=f"share_{event_id}_{current_user}",
                                       use_container_width=True)
                
                if share_button:
                    # Load fresh events data
                    events = data_manager.load('events')
                    event_to_update = None
                    event_index = -1
                    
                    # Find the event
                    for i, e in enumerate(events):
                        if e.get('id') == event_id:
                            event_to_update = e
                            event_index = i
                            break
                    
                    if event_to_update:
                        # Initialize social_stats if not present
                        if 'social_stats' not in event_to_update:
                            event_to_update['social_stats'] = {
                                'likes': [],
                                'favorites': [],
                                'interested': [],
                                'shares': 0,
                                'views': 0
                            }
                        
                        # Increment share count
                        shares = event_to_update['social_stats'].get('shares', 0) + 1
                        
                        # Update the event in the list
                        events[event_index]['social_stats']['shares'] = shares
                        
                        # Save back to file
                        if data_manager.save('events', events):
                            # Generate share text
                            share_text = f"Check out this event: {event['title']}"
                            if event.get('registration_link'):
                                share_text += f"\nRegister here: {event['registration_link']}"
                            
                            # Show share options
                            st.toast("Event shared! 📤")
                            st.info(f"Copy this text to share:\n\n`{share_text}`")
                            st.rerun()
                
                st.caption(f"{social_stats.get('shares', 0)} shares")
            
            with col_social[4]:
                view_button = st.button("👁️ View", 
                                      key=f"view_{event_id}_{current_user}",
                                      use_container_width=True)
                
                if view_button:
                    # Load fresh events data
                    events = data_manager.load('events')
                    event_to_update = None
                    event_index = -1
                    
                    # Find the event
                    for i, e in enumerate(events):
                        if e.get('id') == event_id:
                            event_to_update = e
                            event_index = i
                            break
                    
                    if event_to_update:
                        # Initialize social_stats if not present
                        if 'social_stats' not in event_to_update:
                            event_to_update['social_stats'] = {
                                'likes': [],
                                'favorites': [],
                                'interested': [],
                                'shares': 0,
                                'views': 0
                            }
                        
                        # Increment view count
                        views = event_to_update['social_stats'].get('views', 0) + 1
                        
                        # Update the event in the list
                        events[event_index]['social_stats']['views'] = views
                        
                        # Save back to file
                        if data_manager.save('events', events):
                            st.success("View recorded!")
                            st.rerun()
                
                st.caption(f"{social_stats.get('views', 0)} views")
        
        # Registration section
        st.markdown("---")
        st.markdown('<div class="registration-section">', unsafe_allow_html=True)
        st.subheader("📝 Registration")
        
        # Check if user is registered
        registrations = data_manager.load('registrations')
        is_registered = any(r.get('event_id') == event_id and 
                           r.get('student_username') == current_user 
                           for r in registrations)
        
        if is_registered:
            st.success("✅ You are registered for this event")
            
            # Show registration details
            registration = next(r for r in registrations 
                              if r.get('event_id') == event_id and 
                              r.get('student_username') == current_user)
            
            col_reg = st.columns(3)
            with col_reg[0]:
                st.info(f"Status: {registration.get('status', 'pending').title()}")
            with col_reg[1]:
                st.info(f"Via: {'Official Link' if registration.get('via_link') else 'App'}")
            with col_reg[2]:
                if registration.get('attendance') == 'present':
                    st.success("Attended ✅")
                else:
                    st.warning("Not Attended")
        else:
            col_reg_actions = st.columns([2, 1])
            
            with col_reg_actions[0]:
                if event.get('registration_link'):
                    st.markdown(f"[🔗 **Register via Official Link**]({event['registration_link']})", 
                               unsafe_allow_html=True)
                    st.caption("Click the link above to register on the official platform")
            
            with col_reg_actions[1]:
                reg_button = st.button("✅ **I Have Registered**", 
                                     key=f"reg_{event_id}_{current_user}",
                                     use_container_width=True, 
                                     type="primary")
                
                if reg_button:
                    # Create registration record
                    users = data_manager.load('users')
                    student = next((u for u in users if u.get('username') == current_user), {})
                    
                    reg_data = {
                        'id': str(uuid.uuid4()),
                        'event_id': event_id,
                        'event_title': event.get('title', 'Untitled Event'),
                        'student_username': current_user,
                        'student_name': student.get('name', current_user),
                        'student_roll': student.get('roll_no', 'N/A'),
                        'student_dept': student.get('department', 'N/A'),
                        'via_link': False,
                        'via_app': True,
                        'status': 'pending',
                        'attendance': 'absent',
                        'registered_at': datetime.now().isoformat()
                    }
                    
                    if data_manager.add_item('registrations', reg_data):
                        st.success("Registration recorded! Waiting for verification.")
                        st.rerun()
                    else:
                        st.error("Failed to record registration")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# FACULTY DASHBOARD
# ============================================
def faculty_dashboard():
    """Faculty coordinator dashboard"""
    st.sidebar.title("👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    display_role_badge('faculty')
    
    # Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Create Event", "AI Event Creator", "My Events", 
                    "Registrations", "Analytics"],
            icons=["house", "plus-circle", "robot", "calendar-event", "list-check", "graph-up"],
            default_index=0
        )
    
    if selected == "Dashboard":
        st.markdown('<h1 class="main-header">Faculty Dashboard</h1>', unsafe_allow_html=True)
        
        # Statistics
        events = data_manager.load('events')
        my_events = [e for e in events if e.get('created_by') == st.session_state.username]
        registrations = data_manager.load('registrations')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("My Events", len(my_events))
        with col2:
            upcoming = len([e for e in my_events if is_upcoming(e.get('event_date'))])
            st.metric("Upcoming", upcoming)
        with col3:
            total_reg = len([r for r in registrations 
                           if any(e.get('id') == r.get('event_id') for e in my_events)])
            st.metric("Total Registrations", total_reg)
        with col4:
            attended = len([r for r in registrations 
                          if r.get('attendance') == 'present' and 
                          any(e.get('id') == r.get('event_id') for e in my_events)])
            st.metric("Attended", attended)
        
        # Recent events
        st.subheader("📅 My Recent Events")
        if my_events:
            for event in my_events[-3:]:  # Last 3 events
                display_event_card_social(event, None)
        else:
            st.info("No events created yet. Create your first event!")
    
    elif selected == "Create Event":
        st.header("➕ Create New Event")
    
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
                registration_link = st.text_input("Registration Link")
            
                # Flyer upload
                st.subheader("Event Flyer (Optional)")
                flyer = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png', 'gif'])
                if flyer:
                    st.image(flyer, width=200)
        
            description = st.text_area("Event Description *", height=150)
        
            submit_button = st.form_submit_button("Create Event", use_container_width=True)
        
            if submit_button:
                if not all([title, event_type, venue, organizer, description]):
                    st.error("Please fill all required fields (*)")
                else:
                    # Save flyer
                    flyer_path = None
                    if flyer:
                        flyer_path = data_manager.save_image(flyer)
                
                    # Combine date and time
                    event_datetime = datetime.combine(event_date, event_time)
                
                    event_data = {
                        'id': str(uuid.uuid4()),
                        'title': title,
                        'description': description,
                        'event_type': event_type,
                        'event_date': event_datetime.isoformat(),
                        'venue': venue,
                        'organizer': organizer,
                        'registration_link': registration_link,
                        'max_participants': max_participants,
                        'flyer_path': flyer_path,
                        'created_by': st.session_state.username,
                        'created_by_name': st.session_state.name,
                        'ai_generated': False,
                        'social_stats': {
                            'likes': [],
                            'favorites': [],
                            'interested': [],
                            'shares': 0,
                            'views': 0
                        }
                    }
                
                    if data_manager.add_item('events', event_data):
                        st.success(f"Event '{title}' created successfully! 🎉")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Failed to create event")
    
    elif selected == "AI Event Creator":
        ai_event_creation()
    
    elif selected == "My Events":
        st.header("📋 My Events")
        
        events = data_manager.load('events')
        my_events = [e for e in events if e.get('created_by') == st.session_state.username]
        
        if not my_events:
            st.info("You haven't created any events yet.")
            return
        
        # Filter tabs
        tab1, tab2 = st.tabs(["Upcoming Events", "Past Events"])
        
        with tab1:
            upcoming = [e for e in my_events if is_upcoming(e.get('event_date'))]
            if upcoming:
                for event in upcoming:
                    display_event_card_social(event, None)
            else:
                st.info("No upcoming events.")
        
        with tab2:
            past = [e for e in my_events if not is_upcoming(e.get('event_date'))]
            if past:
                for event in past:
                    display_event_card_social(event, None)
            else:
                st.info("No past events.")
    
    elif selected == "Registrations":
        st.header("📝 Event Registrations")
        
        events = data_manager.load('events')
        my_events = [e for e in events if e.get('created_by') == st.session_state.username]
        
        if not my_events:
            st.info("You haven't created any events yet.")
            return
        
        # Select event
        event_titles = [e['title'] for e in my_events]
        selected_title = st.selectbox("Select Event", event_titles)
        
        if selected_title:
            selected_event = next(e for e in my_events if e['title'] == selected_title)
            event_id = selected_event['id']
            
            registrations = data_manager.load('registrations')
            event_regs = [r for r in registrations if r.get('event_id') == event_id]
            
            if event_regs:
                # Convert to DataFrame
                reg_data = []
                for reg in event_regs:
                    reg_data.append({
                        'Student Name': reg.get('student_name', 'N/A'),
                        'Roll No': reg.get('student_roll', 'N/A'),
                        'Department': reg.get('student_dept', 'N/A'),
                        'Registered Via': 'Official Link' if reg.get('via_link') else 'App',
                        'Status': reg.get('status', 'pending').title(),
                        'Attendance': reg.get('attendance', 'absent').title(),
                        'Registered On': format_date(reg.get('registered_at'))
                    })
                
                df = pd.DataFrame(reg_data)
                st.dataframe(df, use_container_width=True)
                
                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Registrations", len(event_regs))
                with col2:
                    via_link = len([r for r in event_regs if r.get('via_link')])
                    st.metric("Via Official Link", via_link)
                with col3:
                    via_app = len([r for r in event_regs if r.get('via_app')])
                    st.metric("Via App", via_app)
                
                # Export
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"registrations_{selected_title}.csv",
                    mime="text/csv"
                )
                
                # Update status
                st.subheader("Update Registration Status")
                selected_student = st.selectbox("Select Student", 
                                               [r['student_name'] for r in event_regs])
                
                if selected_student:
                    reg = next(r for r in event_regs if r['student_name'] == selected_student)
                    
                    col_status, col_att = st.columns(2)
                    with col_status:
                        new_status = st.selectbox("Registration Status", 
                                                 ["pending", "confirmed", "cancelled"],
                                                 index=["pending", "confirmed", "cancelled"]
                                                 .index(reg.get('status', 'pending')))
                    with col_att:
                        new_att = st.selectbox("Attendance", 
                                              ["absent", "present"],
                                              index=["absent", "present"]
                                              .index(reg.get('attendance', 'absent')))
                    
                    if st.button("Update Status"):
                        # Find and update registration
                        all_regs = data_manager.load('registrations')
                        for i, r in enumerate(all_regs):
                            if (r.get('event_id') == event_id and 
                                r.get('student_name') == selected_student):
                                all_regs[i]['status'] = new_status
                                all_regs[i]['attendance'] = new_att
                                break
                        
                        data_manager.save('registrations', all_regs)
                        st.success("Status updated!")
                        st.rerun()
            else:
                st.info(f"No registrations for '{selected_title}' yet.")
    
    elif selected == "Analytics":
        st.header("📊 Event Analytics")
        
        events = data_manager.load('events')
        my_events = [e for e in events if e.get('created_by') == st.session_state.username]
        
        if not my_events:
            st.info("No events to analyze.")
            return
        
        # Overall statistics
        st.subheader("Overall Statistics")
        
        total_likes = sum(len(e.get('social_stats', {}).get('likes', [])) for e in my_events)
        total_favs = sum(len(e.get('social_stats', {}).get('favorites', [])) for e in my_events)
        total_int = sum(len(e.get('social_stats', {}).get('interested', [])) for e in my_events)
        total_views = sum(e.get('social_stats', {}).get('views', 0) for e in my_events)
        total_shares = sum(e.get('social_stats', {}).get('shares', 0) for e in my_events)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Likes", total_likes)
        with col2:
            st.metric("Total Favorites", total_favs)
        with col3:
            st.metric("Total Interested", total_int)
        with col4:
            st.metric("Total Views", total_views)
        with col5:
            st.metric("Total Shares", total_shares)
        
        # Event-wise analytics
        st.subheader("Event-wise Analytics")
        
        analytics_data = []
        for event in my_events:
            social = event.get('social_stats', {})
            analytics_data.append({
                'Event': event['title'],
                'Likes': len(social.get('likes', [])),
                'Favorites': len(social.get('favorites', [])),
                'Interested': len(social.get('interested', [])),
                'Views': social.get('views', 0),
                'Shares': social.get('shares', 0),
                'Status': 'Upcoming' if is_upcoming(event.get('event_date')) else 'Past'
            })
        
        df = pd.DataFrame(analytics_data)
        st.dataframe(df, use_container_width=True)
        
        # Chart
        st.subheader("Engagement Chart")
        chart_df = df.set_index('Event')[['Likes', 'Favorites', 'Interested']].head(5)
        st.bar_chart(chart_df)

# ============================================
# STUDENT DASHBOARD
# ============================================
def student_dashboard():
    """Student dashboard"""
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**User:** {st.session_state.name}")
    
    # Get student info
    users = data_manager.load('users')
    student = next((u for u in users if u.get('username') == st.session_state.username), {})
    
    if student:
        st.sidebar.markdown(f"**Roll No:** {student.get('roll_no', 'N/A')}")
        st.sidebar.markdown(f"**Department:** {student.get('department', 'N/A')}")
        st.sidebar.markdown(f"**Year:** {student.get('year', 'N/A')}")
    
    display_role_badge('student')
    
    # Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Events Feed", "My Registrations", "My Interests", "Profile"],
            icons=["compass", "list-check", "heart", "person"],
            default_index=0
        )
    
    if selected == "Events Feed":
        st.markdown('<h1 class="main-header">🎯 Discover Events</h1>', unsafe_allow_html=True)
        
        # Filters
        col_filters = st.columns([2, 1, 1])
        with col_filters[0]:
            search = st.text_input("🔍 Search events", placeholder="Search...")
        with col_filters[1]:
            event_type = st.selectbox("Type", ["All", "Workshop", "Hackathon", "Competition", 
                                              "Bootcamp", "Seminar", "Conference"])
        with col_filters[2]:
            show_only = st.selectbox("Show", ["All", "Upcoming", "Past"])
        
        # Get events
        events = data_manager.load('events')
        
        # Apply filters
        filtered_events = events
        
        if search:
            filtered_events = [e for e in filtered_events 
                             if search.lower() in e.get('title', '').lower() or 
                             search.lower() in e.get('description', '').lower()]
        
        if event_type != "All":
            filtered_events = [e for e in filtered_events if e.get('event_type') == event_type]
        
        if show_only == "Upcoming":
            filtered_events = [e for e in filtered_events if is_upcoming(e.get('event_date'))]
        elif show_only == "Past":
            filtered_events = [e for e in filtered_events if not is_upcoming(e.get('event_date'))]
        
        # Sort by date
        filtered_events.sort(key=lambda x: x.get('event_date', ''), reverse=True)
        
        # Display events
        if filtered_events:
            for event in filtered_events:
                display_event_card_social(event, st.session_state.username)
        else:
            st.info("No events found matching your criteria.")
    
    elif selected == "My Registrations":
        st.header("📋 My Registrations")
        
        registrations = data_manager.load('registrations')
        my_regs = [r for r in registrations 
                  if r.get('student_username') == st.session_state.username]
        
        if not my_regs:
            st.info("You haven't registered for any events yet.")
            return
        
        # Get event details
        events = data_manager.load('events')
        event_map = {e['id']: e for e in events}
        
        # Tabs for different statuses
        tab1, tab2, tab3 = st.tabs(["Upcoming", "Completed", "All"])
        
        with tab1:
            upcoming_regs = []
            for reg in my_regs:
                event = event_map.get(reg.get('event_id'))
                if event and is_upcoming(event.get('event_date')):
                    upcoming_regs.append((reg, event))
            
            if upcoming_regs:
                for reg, event in upcoming_regs:
                    with st.container():
                        st.markdown('<div class="event-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(event.get('title'))
                            st.caption(f"Date: {format_date(event.get('event_date'))}")
                            st.caption(f"Venue: {event.get('venue')}")
                        with col2:
                            st.info(f"Status: {reg.get('status', 'pending').title()}")
                            st.info(f"Via: {'Official Link' if reg.get('via_link') else 'App'}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No upcoming registered events.")
        
        with tab2:
            completed_regs = []
            for reg in my_regs:
                event = event_map.get(reg.get('event_id'))
                if event and not is_upcoming(event.get('event_date')):
                    completed_regs.append((reg, event))
            
            if completed_regs:
                for reg, event in completed_regs:
                    with st.container():
                        st.markdown('<div class="event-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(event.get('title'))
                            st.caption(f"Date: {format_date(event.get('event_date'))}")
                            st.caption(f"Venue: {event.get('venue')}")
                        with col2:
                            status_color = "✅" if reg.get('attendance') == 'present' else "❌"
                            st.info(f"Attendance: {status_color}")
                            st.info(f"Status: {reg.get('status', 'pending').title()}")
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No completed events.")
        
        with tab3:
            for reg in my_regs:
                event = event_map.get(reg.get('event_id'))
                if event:
                    with st.container():
                        st.markdown('<div class="event-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(event.get('title'))
                            st.caption(f"Date: {format_date(event.get('event_date'))}")
                            st.caption(f"Venue: {event.get('venue')}")
                        with col2:
                            if is_upcoming(event.get('event_date')):
                                st.success("🟢 Upcoming")
                            else:
                                st.error("🔴 Completed")
                            st.info(f"Status: {reg.get('status', 'pending').title()}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected == "My Interests":
        st.header("⭐ My Interests")
        
        events = data_manager.load('events')
        
        # Get events user has interacted with
        liked_events = []
        fav_events = []
        int_events = []
        
        for event in events:
            social = event.get('social_stats', {})
            if st.session_state.username in social.get('likes', []):
                liked_events.append(event)
            if st.session_state.username in social.get('favorites', []):
                fav_events.append(event)
            if st.session_state.username in social.get('interested', []):
                int_events.append(event)
        
        # Tabs
        tab1, tab2, tab3 = st.tabs([f"❤️ Liked ({len(liked_events)})", 
                                   f"⭐ Favorites ({len(fav_events)})", 
                                   f"🤔 Interested ({len(int_events)})"])
        
        with tab1:
            if liked_events:
                for event in liked_events:
                    display_event_card_social(event, st.session_state.username)
            else:
                st.info("You haven't liked any events yet.")
        
        with tab2:
            if fav_events:
                for event in fav_events:
                    display_event_card_social(event, st.session_state.username)
            else:
                st.info("You haven't favorited any events yet.")
        
        with tab3:
            if int_events:
                for event in int_events:
                    display_event_card_social(event, st.session_state.username)
            else:
                st.info("You haven't marked any events as interested.")
    
    elif selected == "Profile":
        st.header("👤 My Profile")
        
        users = data_manager.load('users')
        student = next((u for u in users if u.get('username') == st.session_state.username), None)
        
        if student:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Full Name:** {student.get('name', 'N/A')}")
                st.markdown(f"**Roll Number:** {student.get('roll_no', 'N/A')}")
                st.markdown(f"**Department:** {student.get('department', 'N/A')}")
                st.markdown(f"**Year:** {student.get('year', 'N/A')}")
            
            with col2:
                st.markdown(f"**Email:** {student.get('email', 'N/A')}")
                st.markdown(f"**Username:** {student.get('username', 'N/A')}")
                st.markdown(f"**Member Since:** {student.get('created_at', 'N/A')[:10]}")
            
            # Statistics
            st.markdown("---")
            st.subheader("📊 My Statistics")
            
            registrations = data_manager.load('registrations')
            my_regs = [r for r in registrations 
                      if r.get('student_username') == st.session_state.username]
            
            events = data_manager.load('events')
            total_likes = sum(1 for e in events 
                            if st.session_state.username in e.get('social_stats', {}).get('likes', []))
            total_favs = sum(1 for e in events 
                           if st.session_state.username in e.get('social_stats', {}).get('favorites', []))
            total_int = sum(1 for e in events 
                          if st.session_state.username in e.get('social_stats', {}).get('interested', []))
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Events Registered", len(my_regs))
            with col_stat2:
                attended = len([r for r in my_regs if r.get('attendance') == 'present'])
                st.metric("Events Attended", attended)
            with col_stat3:
                st.metric("Events Liked", total_likes)
            with col_stat4:
                st.metric("Events Favorited", total_favs)

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application"""
    
    # Initialize session state
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'name' not in st.session_state:
        st.session_state.name = None
    
    # Route based on login status
    if st.session_state.role is None:
        login_page()
    elif st.session_state.role == 'admin':
        # Admin dashboard (simplified version)
        st.sidebar.title("👑 Admin Panel")
        st.sidebar.markdown(f"**User:** {st.session_state.name}")
        display_role_badge('admin')
        
        st.markdown('<h1 class="main-header">Admin Dashboard</h1>', unsafe_allow_html=True)
        
        # Quick stats
        events = data_manager.load('events')
        users = data_manager.load('users')
        registrations = data_manager.load('registrations')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Events", len(events))
        with col2:
            st.metric("Total Users", len(users))
        with col3:
            st.metric("Total Registrations", len(registrations))
        with col4:
            upcoming = len([e for e in events if is_upcoming(e.get('event_date'))])
            st.metric("Upcoming Events", upcoming)
        
        # Logout button
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    elif st.session_state.role == 'faculty':
        faculty_dashboard()
        
        # Logout button
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    elif st.session_state.role == 'student':
        student_dashboard()
        
        # Logout button
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ============================================
# RUN APPLICATION
# ============================================
if __name__ == "__main__":
    main()
