"""
Database operations for Event Management System
"""

from sqlalchemy import func, desc, and_, or_
from datetime import datetime, timedelta
from database.models import User, Event, Registration, SocialInteraction, Comment, Analytics
import streamlit as st

class DatabaseOperations:
    """Database operations wrapper"""
    
    def __init__(self):
        self.session = None
    
    def __enter__(self):
        from database.connection import get_session
        self.session = get_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    # User operations
    def create_user(self, user_data):
        """Create a new user"""
        user = User(**user_data)
        self.session.add(user)
        self.session.commit()
        return user
    
    def get_user_by_username(self, username):
        """Get user by username"""
        return self.session.query(User).filter(
            User.username == username,
            User.is_active == True
        ).first()
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        return self.session.query(User).filter(User.id == user_id).first()
    
    def get_all_users(self, role=None):
        """Get all users, optionally filtered by role"""
        query = self.session.query(User).filter(User.is_active == True)
        if role:
            query = query.filter(User.role == role)
        return query.order_by(User.created_at.desc()).all()
    
    # Event operations
    def create_event(self, event_data):
        """Create a new event"""
        event = Event(**event_data)
        self.session.add(event)
        self.session.commit()
        return event
    
    def update_event(self, event_id, update_data):
        """Update event"""
        event = self.session.query(Event).filter(Event.id == event_id).first()
        if event:
            for key, value in update_data.items():
                setattr(event, key, value)
            event.updated_at = datetime.utcnow()
            self.session.commit()
        return event
    
    def get_event(self, event_id):
        """Get event by ID"""
        return self.session.query(Event).filter(Event.id == event_id).first()
    
    def get_all_events(self, filters=None):
        """Get all events with optional filters"""
        query = self.session.query(Event)
        
        if filters:
            # Apply filters
            if filters.get('status') == 'upcoming':
                query = query.filter(Event.event_date > datetime.utcnow())
            elif filters.get('status') == 'past':
                query = query.filter(Event.event_date <= datetime.utcnow())
            
            if filters.get('event_type'):
                query = query.filter(Event.event_type == filters['event_type'])
            
            if filters.get('created_by'):
                query = query.filter(Event.created_by == filters['created_by'])
            
            if filters.get('search'):
                search_term = f"%{filters['search']}%"
                query = query.filter(
                    or_(
                        Event.title.ilike(search_term),
                        Event.description.ilike(search_term),
                        Event.venue.ilike(search_term)
                    )
                )
        
        # Order by event date
        return query.order_by(Event.event_date).all()
    
    def get_upcoming_events(self, limit=None):
        """Get upcoming events"""
        query = self.session.query(Event).filter(
            Event.event_date > datetime.utcnow(),
            Event.status == 'upcoming'
        ).order_by(Event.event_date)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    # Registration operations
    def create_registration(self, registration_data):
        """Create a new registration"""
        # Check if already registered
        existing = self.session.query(Registration).filter(
            Registration.event_id == registration_data['event_id'],
            Registration.student_id == registration_data['student_id']
        ).first()
        
        if existing:
            return existing
        
        registration = Registration(**registration_data)
        self.session.add(registration)
        self.session.commit()
        return registration
    
    def get_event_registrations(self, event_id):
        """Get all registrations for an event"""
        return self.session.query(Registration).filter(
            Registration.event_id == event_id
        ).all()
    
    def get_student_registrations(self, student_id):
        """Get all registrations for a student"""
        return self.session.query(Registration).filter(
            Registration.student_id == student_id
        ).all()
    
    def update_registration_status(self, registration_id, status_data):
        """Update registration status"""
        registration = self.session.query(Registration).filter(
            Registration.id == registration_id
        ).first()
        
        if registration:
            for key, value in status_data.items():
                setattr(registration, key, value)
            self.session.commit()
        
        return registration
    
    # Social interaction operations
    def add_social_interaction(self, event_id, user_id, interaction_type, metadata=None):
        """Add social interaction (like, favorite, etc.)"""
        # Check if interaction already exists
        existing = self.session.query(SocialInteraction).filter(
            SocialInteraction.event_id == event_id,
            SocialInteraction.user_id == user_id,
            SocialInteraction.interaction_type == interaction_type
        ).first()
        
        if existing:
            # Remove if already exists (toggle)
            self.session.delete(existing)
            action = 'removed'
        else:
            # Add new interaction
            interaction = SocialInteraction(
                event_id=event_id,
                user_id=user_id,
                interaction_type=interaction_type,
                metadata=metadata
            )
            self.session.add(interaction)
            action = 'added'
        
        # Update event counters
        event = self.session.query(Event).filter(Event.id == event_id).first()
        if event:
            if interaction_type == 'like':
                event.like_count = self.session.query(func.count(SocialInteraction.id)).filter(
                    SocialInteraction.event_id == event_id,
                    SocialInteraction.interaction_type == 'like'
                ).scalar()
            elif interaction_type == 'favorite':
                event.favorite_count = self.session.query(func.count(SocialInteraction.id)).filter(
                    SocialInteraction.event_id == event_id,
                    SocialInteraction.interaction_type == 'favorite'
                ).scalar()
            elif interaction_type == 'interested':
                event.interested_count = self.session.query(func.count(SocialInteraction.id)).filter(
                    SocialInteraction.event_id == event_id,
                    SocialInteraction.interaction_type == 'interested'
                ).scalar()
        
        self.session.commit()
        return action
    
    def get_user_interactions(self, event_id, user_id):
        """Get all interactions of a user for an event"""
        interactions = self.session.query(SocialInteraction).filter(
            SocialInteraction.event_id == event_id,
            SocialInteraction.user_id == user_id
        ).all()
        
        return {i.interaction_type: True for i in interactions}
    
    # Analytics operations
    def record_analytics(self, event_id, metric_name, metric_value):
        """Record analytics data"""
        analytics = Analytics(
            event_id=event_id,
            metric_name=metric_name,
            metric_value=metric_value
        )
        self.session.add(analytics)
        self.session.commit()
        return analytics
    
    def get_event_analytics(self, event_id):
        """Get analytics for an event"""
        return self.session.query(Analytics).filter(
            Analytics.event_id == event_id
        ).order_by(Analytics.recorded_at.desc()).all()
    
    # Comment operations
    def add_comment(self, comment_data):
        """Add a comment to an event"""
        comment = Comment(**comment_data)
        self.session.add(comment)
        self.session.commit()
        return comment
    
    def get_event_comments(self, event_id):
        """Get all comments for an event"""
        return self.session.query(Comment).filter(
            Comment.event_id == event_id,
            Comment.is_approved == True,
            Comment.parent_comment_id == None  # Only top-level comments
        ).order_by(Comment.created_at.desc()).all()
    
    # Statistics
    def get_system_statistics(self):
        """Get system-wide statistics"""
        stats = {}
        
        # Count events
        stats['total_events'] = self.session.query(func.count(Event.id)).scalar()
        stats['upcoming_events'] = self.session.query(func.count(Event.id)).filter(
            Event.event_date > datetime.utcnow()
        ).scalar()
        
        # Count users by role
        stats['total_students'] = self.session.query(func.count(User.id)).filter(
            User.role == 'student',
            User.is_active == True
        ).scalar()
        
        stats['total_faculty'] = self.session.query(func.count(User.id)).filter(
            User.role == 'faculty',
            User.is_active == True
        ).scalar()
        
        # Count registrations
        stats['total_registrations'] = self.session.query(func.count(Registration.id)).scalar()
        
        # Popular events (by registrations)
        popular_events = self.session.query(
            Event.title,
            func.count(Registration.id).label('registration_count')
        ).join(Registration, Event.id == Registration.event_id)\
         .group_by(Event.id)\
         .order_by(desc('registration_count'))\
         .limit(5).all()
        
        stats['popular_events'] = [
            {'title': title, 'count': count} 
            for title, count in popular_events
        ]
        
        # Recent activity
        week_ago = datetime.utcnow() - timedelta(days=7)
        stats['recent_registrations'] = self.session.query(func.count(Registration.id)).filter(
            Registration.registration_date >= week_ago
        ).scalar()
        
        stats['recent_events'] = self.session.query(func.count(Event.id)).filter(
            Event.created_at >= week_ago
        ).scalar()
        
        return stats
