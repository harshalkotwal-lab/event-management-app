# G H Raisoni College - Event Management System

A comprehensive web application for managing college events, workshops, hackathons, and competitions with three-tier user roles.

## 🎯 Features

### **User Roles**
1. **Administrator** (Dean/Director)
   - Default credentials: `admin@raisoni` / `admin123`
   - View all analytics and reports
   - Manage faculty coordinators
   - System configuration

2. **Faculty Coordinator**
   - Default credentials: `faculty@raisoni` / `faculty123`
   - Create and manage events
   - Track student registrations
   - Verify participation

3. **Students**
   - Self-registration with college email
   - Browse and filter events
   - Register for events
   - Track participation history

### **Event Features**
- Social media style event cards
- Like, Favorite, Interested buttons
- Official registration link integration
- "I Have Registered" confirmation
- Real-time participation tracking
- Event analytics and reports

## 🚀 Quick Start

### **Local Development**
```bash
# 1. Clone repository
git clone https://github.com/your-org/raisoni-event-manager.git
cd raisoni-event-manager

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
streamlit run app.py
