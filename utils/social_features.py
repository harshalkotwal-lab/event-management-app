"""
Social media interactions and sharing functionality
"""

import streamlit as st
from datetime import datetime
import urllib.parse

class SocialFeatures:
    """Handle social interactions and sharing"""
    
    def __init__(self, db_operations):
        self.db = db_operations
    
    def handle_like(self, event_id, user_id):
        """Handle like/unlike action"""
        return self.db.add_social_interaction(event_id, user_id, 'like')
    
    def handle_favorite(self, event_id, user_id):
        """Handle favorite/unfavorite action"""
        return self.db.add_social_interaction(event_id, user_id, 'favorite')
    
    def handle_interested(self, event_id, user_id):
        """Handle interested/not interested action"""
        return self.db.add_social_interaction(event_id, user_id, 'interested')
    
    def handle_share(self, event_id, user_id, platform=None):
        """Handle share action"""
        metadata = {'platform': platform, 'shared_at': datetime.utcnow().isoformat()}
        self.db.add_social_interaction(event_id, user_id, 'share', metadata)
        
        # Increment share count
        event = self.db.get_event(event_id)
        if event:
            event.share_count += 1
            self.db.session.commit()
        
        return True
    
    def get_share_links(self, event, base_url):
        """Generate social media share links"""
        title = urllib.parse.quote(event.title)
        description = urllib.parse.quote(event.description[:100])
        url = urllib.parse.quote(f"{base_url}/event/{event.id}")
        
        share_links = {
            'twitter': f"https://twitter.com/intent/tweet?text={title}&url={url}",
            'facebook': f"https://www.facebook.com/sharer/sharer.php?u={url}",
            'linkedin': f"https://www.linkedin.com/sharing/share-offsite/?url={url}",
            'whatsapp': f"https://api.whatsapp.com/send?text={title}%20{url}",
            'telegram': f"https://t.me/share/url?url={url}&text={title}"
        }
        
        return share_links
    
    def get_user_interactions(self, event_id, user_id):
        """Get all user interactions for an event"""
        return self.db.get_user_interactions(event_id, user_id)
    
    def display_social_buttons(self, event, user_id, base_url):
        """Display social interaction buttons"""
        if not user_id:
            return
        
        # Get user's current interactions
        interactions = self.get_user_interactions(event.id, user_id)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            like_icon = "❤️" if interactions.get('like') else "🤍"
            if st.button(f"{like_icon} {event.like_count}", key=f"like_{event.id}"):
                action = self.handle_like(event.id, user_id)
                st.rerun()
        
        with col2:
            fav_icon = "⭐" if interactions.get('favorite') else "☆"
            if st.button(f"{fav_icon} {event.favorite_count}", key=f"fav_{event.id}"):
                action = self.handle_favorite(event.id, user_id)
                st.rerun()
        
        with col3:
            interest_icon = "✅" if interactions.get('interested') else "🤔"
            if st.button(f"{interest_icon} {event.interested_count}", key=f"int_{event.id}"):
                action = self.handle_interested(event.id, user_id)
                st.rerun()
        
        with col4:
            if st.button(f"📤 {event.share_count}", key=f"share_{event.id}"):
                # Show share options
                with st.popover("Share Event"):
                    st.write("Share on:")
                    share_links = self.get_share_links(event, base_url)
                    
                    platforms = {
                        'Twitter': '🐦',
                        'Facebook': '📘',
                        'LinkedIn': '💼',
                        'WhatsApp': '💚',
                        'Telegram': '📡'
                    }
                    
                    for platform, icon in platforms.items():
                        platform_key = platform.lower()
                        share_url = share_links.get(platform_key)
                        if share_url:
                            st.markdown(f"[{icon} {platform}]({share_url})")
                
                # Record share
                self.handle_share(event.id, user_id, 'popup')
                st.rerun()
        
        with col5:
            if st.button("👁️", key=f"view_{event.id}"):
                # Increment view count
                event.view_count += 1
                self.db.session.commit()
                st.rerun()
