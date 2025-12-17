"""
G H Raisoni College - Advanced Event Management System
With AI, Database, Image Uploads, and Social Features
"""

import streamlit as st
from datetime import datetime, date
import pandas as pd
from pathlib import Path
import os

# Import custom modules
from database.operations import DatabaseOperations
from utils.ai_event_generator import AIEventGenerator
from utils.image_processor import ImageProcessor
from utils.social_features import SocialFeatures

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
    /* ... (keep all previous CSS styles) ... */
    
    .ai-generated-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 0.5rem;
    }
    
    .flyer-preview {
        border: 2px dashed #3B82F6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
        background: #F8FAFC;
    }
    
    .social-buttons-container {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .social-button {
        flex: 1;
        min-width: 80px;
        text-align: center;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        background: white;
        transition: all 0.3s;
    }
    
    .social-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .social-button.active {
        border-color: #3B82F6;
        background: #DBEAFE;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE MODULES
# ============================================
@st.cache_resource
def get_ai_generator():
    return AIEventGenerator()

@st.cache_resource
def get_image_processor():
    return ImageProcessor(upload_dir="static/uploads")

# ============================================
# HELPER FUNCTIONS
# ============================================
def hash_password(password):
    """Simple password hashing (use bcrypt in production)"""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def format_datetime(dt):
    """Format datetime for display"""
    if isinstance(dt, str):
        return dt
    return dt.strftime("%d %b %Y, %I:%M %p")

# ============================================
# AUTHENTICATION
# ============================================
def login_page():
    """Enhanced login page with database"""
    st.markdown('<div class="college-header"><h2>G H Raisoni College of Engineering and Management</h2><p>Advanced Event Management System</p></div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Admin Login")
        admin_user = st.text_input("Username", key="admin_user")
        admin_pass = st.text_input("Password", type="password", key="admin_pass")
        
        if st.button("Admin Login", key="admin_login", use_container_width=True):
            with DatabaseOperations() as db:
                user = db.get_user_by_username(admin_user)
                if user and user.role == 'admin' and user.password_hash == hash_password(admin_pass):
                    st.session_state.user = user
                    st.session_state.role = user.role
                    st.success("Admin login successful!")
                    st.rerun()
                else:
                    st.error("Invalid admin credentials")
    
    with col2:
        st.subheader("Faculty Login")
        faculty_user = st.text_input("Username", key="faculty_user")
        faculty_pass = st.text_input("Password", type="password", key="faculty_pass")
        
        if st.button("Faculty Login", key="faculty_login", use_container_width=True):
            with DatabaseOperations() as db:
                user = db.get_user_by_username(faculty_user)
                if user and user.role == 'faculty' and user.password_hash == hash_password(faculty_pass):
                    st.session_state.user = user
                    st.session_state.role = user.role
                    st.success("Faculty login successful!")
                    st.rerun()
                else:
                    st.error("Invalid faculty credentials")
    
    with col3:
        st.subheader("Student Portal")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            student_user = st.text_input("Username", key="student_user_login")
            student_pass = st.text_input("Password", type="password", key="student_pass_login")
            
            if st.button("Student Login", key="student_login", use_container_width=True):
                with DatabaseOperations() as db:
                    user = db.get_user_by_username(student_user)
                    if user and user.role == 'student' and user.password_hash == hash_password(student_pass):
                        st.session_state.user = user
                        st.session_state.role = user.role
                        st.success("Student login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid student credentials")
        
        with tab2:
            with st.form("student_registration"):
                st.markdown("**Create Student Account**")
                
                full_name = st.text_input("Full Name *")
                roll_number = st.text_input("Roll Number *")
                department = st.selectbox("Department *", 
                                         ["CSE", "AIML", "ECE", "EEE", "MECH", "CIVIL", "IT", "DS"])
                year = st.selectbox("Year *", ["I", "II", "III", "IV"])
                email = st.text_input("Email *")
                username = st.text_input("Username *")
                password = st.text_input("Password *", type="password")
                
                if st.form_submit_button("Register", use_container_width=True):
                    if not all([full_name, roll_number, email, username, password]):
                        st.error("Please fill all required fields (*)")
                    else:
                        with DatabaseOperations() as db:
                            # Check if username exists
                            if db.get_user_by_username(username):
                                st.error("Username already exists")
                            else:
                                user_data = {
                                    'username': username,
                                    'password_hash': hash_password(password),
                                    'name': full_name,
                                    'email': email,
                                    'role': 'student',
                                    'department': department,
                                    'roll_number': roll_number,
                                    'year': year
                                }
                                
                                user = db.create_user(user_data)
                                if user:
                                    st.success("Registration successful! Please login.")
                                    st.rerun()
                                else:
                                    st.error("Registration failed")

# ============================================
# AI EVENT CREATION
# ============================================
def ai_event_creation():
    """Create event using AI from WhatsApp/email text"""
    st.header("🤖 AI-Powered Event Creation")
    
    ai_gen = get_ai_generator()
    
    tab1, tab2 = st.tabs(["From Text", "From Email/WhatsApp"])
    
    with tab1:
        st.subheader("Paste Event Text")
        event_text = st.text_area("Paste event details from WhatsApp, email, or any source:", 
                                 height=200,
                                 placeholder="""Example:
Hackathon Alert! 🚀
Join us for the annual AI Hackathon on December 20-21, 2024 at Seminar Hall, Block C.
Organized by Computer Science Department.
Register at: https://forms.gle/example
Prizes worth ₹50,000!""")
        
        if st.button("Generate Event", key="generate_from_text"):
            if event_text:
                with st.spinner("AI is extracting event details..."):
                    event_data = ai_gen.extract_event_info(event_text)
                    
                    # Show extracted data
                    st.subheader("📋 Extracted Event Details")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_input("Title", value=event_data.get('title', ''), key="ai_title")
                        st.text_area("Description", value=event_data.get('description', ''), 
                                    height=150, key="ai_description")
                        st.selectbox("Event Type", 
                                    ["Workshop", "Hackathon", "Competition", "Bootcamp", "Seminar", "Conference", "Webinar"],
                                    index=["Workshop", "Hackathon", "Competition", "Bootcamp", "Seminar", "Conference", "Webinar"]
                                    .index(event_data.get('event_type', 'Workshop')),
                                    key="ai_type")
                    
                    with col2:
                        st.date_input("Event Date", 
                                     value=datetime.strptime(event_data.get('event_date', date.today().isoformat()), '%Y-%m-%d').date(),
                                     key="ai_date")
                        st.text_input("Venue", value=event_data.get('venue', ''), key="ai_venue")
                        st.text_input("Organizer", value=event_data.get('organizer', ''), key="ai_organizer")
                        st.text_input("Registration Link", value=event_data.get('registration_link', ''), 
                                     key="ai_link")
                    
                    # Enhance with AI
                    if st.button("✨ Enhance with AI", key="enhance_ai"):
                        with st.spinner("Enhancing event details..."):
                            enhanced = ai_gen.enhance_event_description(event_data)
                            st.success("Event enhanced!")
                            st.rerun()
                    
                    # Save to database
                    if st.button("Save Event", key="save_ai_event"):
                        # Get values from session state
                        event_to_save = {
                            'title': st.session_state.ai_title,
                            'description': st.session_state.ai_description,
                            'event_type': st.session_state.ai_type,
                            'event_date': datetime.combine(st.session_state.ai_date, datetime.min.time()),
                            'venue': st.session_state.ai_venue,
                            'organizer': st.session_state.ai_organizer,
                            'registration_link': st.session_state.ai_link,
                            'created_by': st.session_state.user.id,
                            'ai_generated': True,
                            'ai_prompt': event_text,
                            'ai_metadata': event_data.get('ai_metadata', {})
                        }
                        
                        with DatabaseOperations() as db:
                            event = db.create_event(event_to_save)
                            if event:
                                st.success(f"Event '{event.title}' created successfully!")
                                st.balloons()
                                st.rerun()
    
    with tab2:
        st.subheader("Upload Email/WhatsApp Export")
        
        uploaded_file = st.file_uploader("Upload text file", type=['txt', 'pdf', 'docx'])
        
        if uploaded_file:
            content = uploaded_file.getvalue().decode('utf-8')
            st.text_area("File Content", content, height=200)
            
            if st.button("Extract Events", key="extract_from_file"):
                # Simple extraction - in real app, you'd parse the file properly
                events = content.split('\n\n')  # Simple split by blank lines
                
                for i, event_text in enumerate(events[:3]):  # Process first 3 events
                    if len(event_text.strip()) > 50:
                        with st.expander(f"Event {i+1}"):
                            event_data = ai_gen.extract_event_info(event_text)
                            st.json(event_data)

# ============================================
# EVENT CARD WITH SOCIAL FEATURES
# ============================================
def display_advanced_event_card(event, user_id, show_actions=True):
    """Display event card with social features"""
    with st.container():
        st.markdown('<div class="event-card">', unsafe_allow_html=True)
        
        # Header with AI badge
        col_header = st.columns([4, 1])
        with col_header[0]:
            st.subheader(event.title)
            if event.ai_generated:
                st.markdown('<span class="ai-generated-badge">🤖 AI Generated</span>', 
                           unsafe_allow_html=True)
        with col_header[1]:
            if event.is_upcoming():
                st.success("🟢 Upcoming")
            else:
                st.error("🔴 Completed")
        
        # Event flyer
        if event.flyer_image_path:
            img_processor = get_image_processor()
            img_processor.display_image(event.flyer_image_path, width=300)
        
        # Description
        st.write(event.description[:300] + "..." if len(event.description) > 300 else event.description)
        
        # Details
        col_details = st.columns(4)
        with col_details[0]:
            st.caption(f"**📅 Date:** {format_datetime(event.event_date)}")
        with col_details[1]:
            st.caption(f"**📍 Venue:** {event.venue}")
        with col_details[2]:
            st.caption(f"**🏷️ Type:** {event.event_type}")
        with col_details[3]:
            st.caption(f"**👨‍🏫 Organizer:** {event.organizer}")
        
        # Social buttons
        if show_actions and user_id:
            social = SocialFeatures(DatabaseOperations())
            
            # Get base URL for sharing (adjust for your deployment)
            base_url = st.secrets.get("APP_URL", "https://yourapp.streamlit.app")
            
            # Display social buttons
            social.display_social_buttons(event, user_id, base_url)
            
            # Registration section
            st.markdown("---")
            st.subheader("Registration")
            
            # Check if already registered
            with DatabaseOperations() as db:
                registration = db.session.query(Registration).filter(
                    Registration.event_id == event.id,
                    Registration.student_id == user_id
                ).first()
            
            if registration:
                st.success("✅ You are registered for this event")
                
                col_reg = st.columns(3)
                with col_reg[0]:
                    st.info(f"Status: {registration.registration_status.title()}")
                with col_reg[1]:
                    st.info(f"Attendance: {registration.attendance_status.title()}")
                with col_reg[2]:
                    if registration.registered_via_link:
                        st.success("Registered via official link")
                    if registration.registered_in_app:
                        st.success("Marked 'I Have Registered'")
            else:
                col_reg_actions = st.columns(2)
                
                with col_reg_actions[0]:
                    if event.registration_link:
                        st.markdown(f"[🔗 Register via Official Link]({event.registration_link})", 
                                   unsafe_allow_html=True)
                
                with col_reg_actions[1]:
                    if st.button("✅ I Have Registered", key=f"registered_{event.id}"):
                        with DatabaseOperations() as db:
                            reg_data = {
                                'event_id': event.id,
                                'student_id': user_id,
                                'registered_in_app': True,
                                'registration_status': 'pending'
                            }
                            db.create_registration(reg_data)
                            st.success("Registration marked! Waiting for verification.")
                            st.rerun()
        
        # Comments section (collapsible)
        with st.expander("💬 Comments"):
            with DatabaseOperations() as db:
                comments = db.get_event_comments(event.id)
                
                if comments:
                    for comment in comments:
                        st.markdown(f"**{comment.user.name}** ({format_datetime(comment.created_at)})")
                        st.write(comment.comment_text)
                        st.markdown("---")
                else:
                    st.info("No comments yet. Be the first to comment!")
            
            # Add comment
            if user_id:
                new_comment = st.text_area("Add a comment", key=f"comment_{event.id}")
                if st.button("Post Comment", key=f"post_comment_{event.id}"):
                    if new_comment:
                        with DatabaseOperations() as db:
                            db.add_comment({
                                'event_id': event.id,
                                'user_id': user_id,
                                'comment_text': new_comment
                            })
                            st.success("Comment posted!")
                            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# FACULTY DASHBOARD - ENHANCED
# ============================================
def faculty_dashboard_enhanced():
    """Enhanced faculty dashboard with AI and image uploads"""
    st.sidebar.title("👨‍🏫 Faculty Panel")
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.user.name}")
    
    menu = st.sidebar.selectbox("Navigation", [
        "Dashboard", "Create Event", "AI Event Creation", "My Events", 
        "View Registrations", "Event Analytics"
    ])
    
    if menu == "Dashboard":
        # ... (similar to previous dashboard but with database)
        pass
    
    elif menu == "Create Event":
        st.header("➕ Create New Event (Manual)")
        
        with st.form("manual_event_form"):
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
                st.subheader("Event Flyer")
                flyer_file = st.file_uploader("Upload flyer image", 
                                             type=['jpg', 'jpeg', 'png', 'gif'])
                if flyer_file:
                    img_processor = get_image_processor()
                    temp_path = img_processor.save_uploaded_file(flyer_file)
                    if temp_path:
                        st.image(flyer_file, width=200)
                        st.session_state.temp_flyer_path = temp_path
            
            description = st.text_area("Event Description *", height=150)
            
            if st.form_submit_button("Create Event"):
                if not all([title, event_type, venue, organizer, description]):
                    st.error("Please fill all required fields (*)")
                else:
                    event_datetime = datetime.combine(event_date, event_time)
                    
                    event_data = {
                        'title': title,
                        'description': description,
                        'event_type': event_type,
                        'event_date': event_datetime,
                        'venue': venue,
                        'organizer': organizer,
                        'registration_link': registration_link,
                        'max_participants': max_participants,
                        'created_by': st.session_state.user.id,
                        'ai_generated': False
                    }
                    
                    with DatabaseOperations() as db:
                        event = db.create_event(event_data)
                        
                        # Save flyer if uploaded
                        if hasattr(st.session_state, 'temp_flyer_path'):
                            img_processor = get_image_processor()
                            final_path = img_processor.save_uploaded_file(
                                None, event_id=event.id
                            )
                            # In real app, you'd move the temp file to final location
                            
                            # Update event with flyer path
                            db.update_event(event.id, {'flyer_image_path': final_path})
                        
                        st.success(f"Event '{event.title}' created successfully!")
                        st.balloons()
                        st.rerun()
    
    elif menu == "AI Event Creation":
        ai_event_creation()
    
    elif menu == "Event Analytics":
        st.header("📊 Event Analytics")
        
        with DatabaseOperations() as db:
            # Get faculty's events
            events = db.get_all_events({'created_by': st.session_state.user.id})
            
            if events:
                # Select event for detailed analytics
                event_titles = [e.title for e in events]
                selected_title = st.selectbox("Select Event", event_titles)
                
                selected_event = next(e for e in events if e.title == selected_title)
                
                # Display analytics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Views", selected_event.view_count)
                with col2:
                    st.metric("Likes", selected_event.like_count)
                with col3:
                    st.metric("Interested", selected_event.interested_count)
                with col4:
                    st.metric("Shares", selected_event.share_count)
                
                # Registrations
                registrations = db.get_event_registrations(selected_event.id)
                
                st.subheader("Registrations")
                if registrations:
                    reg_data = []
                    for reg in registrations:
                        student = db.get_user_by_id(reg.student_id)
                        reg_data.append({
                            'Student': student.name,
                            'Roll No': student.roll_number,
                            'Department': student.department,
                            'Status': reg.registration_status,
                            'Attendance': reg.attendance_status,
                            'Registered Via': 'Official Link' if reg.registered_via_link else 'App',
                            'Date': format_datetime(reg.registration_date)
                        })
                    
                    df = pd.DataFrame(reg_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Export
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Registrations",
                        data=csv,
                        file_name=f"registrations_{selected_title}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No registrations yet")
            else:
                st.info("You haven't created any events yet")

# ============================================
# STUDENT DASHBOARD - ENHANCED
# ============================================
def student_dashboard_enhanced():
    """Enhanced student dashboard with social features"""
    st.sidebar.title("👨‍🎓 Student Panel")
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.user.name}")
    
    menu = st.sidebar.selectbox("Navigation", [
        "Events Feed", "My Registrations", "My Interests", "My Profile"
    ])
    
    if menu == "Events Feed":
        st.markdown('<h1 class="main-header">🎯 Discover Events</h1>', unsafe_allow_html=True)
        
        # Filters
        col_filters = st.columns([2, 1, 1, 1])
        with col_filters[0]:
            search = st.text_input("🔍 Search events", placeholder="Search by title, description...")
        with col_filters[1]:
            event_type = st.selectbox("Type", ["All", "Workshop", "Hackathon", "Competition", 
                                              "Bootcamp", "Seminar", "Conference"])
        with col_filters[2]:
            sort_by = st.selectbox("Sort by", ["Date", "Popularity", "Newest"])
        with col_filters[3]:
            show_only = st.selectbox("Show", ["All", "Upcoming", "Past"])
        
        # Get events
        with DatabaseOperations() as db:
            filters = {}
            if search:
                filters['search'] = search
            if event_type != "All":
                filters['event_type'] = event_type
            
            events = db.get_all_events(filters)
            
            # Apply additional filters
            if show_only == "Upcoming":
                events = [e for e in events if e.is_upcoming()]
            elif show_only == "Past":
                events = [e for e in events if not e.is_upcoming()]
            
            # Sort
            if sort_by == "Popularity":
                events.sort(key=lambda x: x.like_count + x.interested_count, reverse=True)
            elif sort_by == "Newest":
                events.sort(key=lambda x: x.created_at, reverse=True)
            else:  # Date
                events.sort(key=lambda x: x.event_date)
            
            # Display events
            for event in events:
                display_advanced_event_card(event, st.session_state.user.id, show_actions=True)
    
    elif menu == "My Interests":
        st.header("⭐ My Interests")
        
        with DatabaseOperations() as db:
            # Get events user has interacted with
            interactions = db.session.query(SocialInteraction).filter(
                SocialInteraction.user_id == st.session_state.user.id
            ).all()
            
            if interactions:
                tab_fav, tab_int, tab_like = st.tabs(["Favorites", "Interested", "Liked"])
                
                with tab_fav:
                    fav_events = [db.get_event(i.event_id) for i in interactions 
                                 if i.interaction_type == 'favorite']
                    for event in fav_events:
                        if event:
                            display_advanced_event_card(event, st.session_state.user.id, show_actions=False)
                
                with tab_int:
                    int_events = [db.get_event(i.event_id) for i in interactions 
                                 if i.interaction_type == 'interested']
                    for event in int_events:
                        if event:
                            display_advanced_event_card(event, st.session_state.user.id, show_actions=False)
                
                with tab_like:
                    like_events = [db.get_event(i.event_id) for i in interactions 
                                  if i.interaction_type == 'like']
                    for event in like_events:
                        if event:
                            display_advanced_event_card(event, st.session_state.user.id, show_actions=False)
            else:
                st.info("You haven't interacted with any events yet. Start exploring!")

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application"""
    
    # Check if user is logged in
    if 'user' not in st.session_state:
        login_page()
        return
    
    # Route based on role
    role = st.session_state.role
    
    if role == 'admin':
        # ... (admin dashboard - similar to previous but with database)
        pass
    elif role == 'faculty':
        faculty_dashboard_enhanced()
    elif role == 'student':
        student_dashboard_enhanced()
    
    # Logout button in sidebar
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
