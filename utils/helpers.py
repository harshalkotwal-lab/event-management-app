"""
Helper functions for the application
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd


def display_role_badge(role: str):
    """Display role badge with appropriate color"""
    badges = {
        'admin': ('👑 Admin', '#DC2626', '#FEE2E2'),
        'faculty': ('👨‍🏫 Faculty', '#1D4ED8', '#DBEAFE'),
        'student': ('👨‍🎓 Student', '#065F46', '#D1FAE5')
    }
    
    if role in badges:
        text, text_color, bg_color = badges[role]
        st.markdown(
            f'<span style="background-color: {bg_color}; color: {text_color}; '
            f'padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; '
            f'font-weight: 600;">{text}</span>',
            unsafe_allow_html=True
        )


def format_date(date_string: str, format_str: str = "%d %b %Y") -> str:
    """Format date string to readable format"""
    try:
        date_obj = datetime.fromisoformat(date_string)
        return date_obj.strftime(format_str)
    except (ValueError, TypeError):
        return date_string


def calculate_time_remaining(event_date: str) -> str:
    """Calculate time remaining until event"""
    try:
        event_dt = datetime.fromisoformat(event_date)
        now = datetime.now()
        
        if event_dt < now:
            return "Event passed"
        
        delta = event_dt - now
        days = delta.days
        
        if days > 30:
            return f"{days//30} months remaining"
        elif days > 0:
            return f"{days} days remaining"
        else:
            hours = delta.seconds // 3600
            if hours > 0:
                return f"{hours} hours remaining"
            return "Today"
    except ValueError:
        return "Invalid date"


def create_dataframe_from_events(events: List[Dict]) -> pd.DataFrame:
    """Convert events list to pandas DataFrame"""
    if not events:
        return pd.DataFrame()
    
    df_data = []
    for event in events:
        df_data.append({
            'Title': event.get('title', ''),
            'Type': event.get('event_type', ''),
            'Date': format_date(event.get('event_date', '')),
            'Venue': event.get('venue', ''),
            'Organizer': event.get('organizer', ''),
            'Likes': len(event.get('likes', [])),
            'Interested': len(event.get('interested', [])),
            'Status': 'Upcoming' if datetime.fromisoformat(event.get('event_date', '2000-01-01')) > datetime.now() else 'Completed'
        })
    
    return pd.DataFrame(df_data)


def export_to_csv(data: List[Dict], filename: str = "export.csv"):
    """Export data to CSV and provide download button"""
    if not data:
        st.warning("No data to export")
        return
    
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def paginate_items(items: List[Any], items_per_page: int = 10) -> List[Any]:
    """Paginate list of items"""
    if not st.session_state.get('page'):
        st.session_state.page = 1
    
    total_pages = max(1, (len(items) + items_per_page - 1) // items_per_page)
    page = st.session_state.page
    
    # Page navigation
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    with col2:
        if st.button("◀ Previous", disabled=page <= 1):
            st.session_state.page -= 1
            st.rerun()
    with col3:
        if st.button("Next ▶", disabled=page >= total_pages):
            st.session_state.page += 1
            st.rerun()
    
    # Display page info
    st.caption(f"Page {page} of {total_pages} ({len(items)} total items)")
    
    # Return items for current page
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(items))
    
    return items[start_idx:end_idx]


def show_loading_spinner(text: str = "Loading..."):
    """Display loading spinner"""
    return st.spinner(text)


def show_success_message(message: str):
    """Display success message"""
    st.success(message)
    st.balloons()


def show_error_message(message: str):
    """Display error message"""
    st.error(message)
