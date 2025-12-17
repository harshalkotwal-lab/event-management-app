"""
Database models for Event Management System
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import pytz

Base = declarative_base()

class User(Base):
    """User model for all roles"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(200), nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    role = Column(String(50), nullable=False)  # admin, faculty, student
    department = Column(String(100))
    roll_number = Column(String(50))
    year = Column(String(10))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    events_created = relationship("Event", back_populates="creator")
    registrations = relationship("Registration", back_populates="student")
    social_interactions = relationship("SocialInteraction", back_populates="user")
    comments = relationship("Comment", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"

class Event(Base):
    """Event model"""
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    event_type = Column(String(50), nullable=False)  # workshop, hackathon, etc.
    event_date = Column(DateTime, nullable=False)
    venue = Column(String(200), nullable=False)
    organizer = Column(String(200), nullable=False)
    registration_link = Column(String(500))
    flyer_image_url = Column(String(500))
    flyer_image_path = Column(String(500))
    status = Column(String(50), default='upcoming')  # upcoming, ongoing, completed, cancelled
    max_participants = Column(Integer)
    created_by = Column(Integer, ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # AI-generated fields
    ai_generated = Column(Boolean, default=False)
    ai_prompt = Column(Text)
    ai_metadata = Column(JSON)  # Store AI response metadata
    
    # Social metrics (cached for performance)
    like_count = Column(Integer, default=0)
    favorite_count = Column(Integer, default=0)
    interested_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    
    # Relationships
    creator = relationship("User", back_populates="events_created")
    registrations = relationship("Registration", back_populates="event")
    social_interactions = relationship("SocialInteraction", back_populates="event")
    comments = relationship("Comment", back_populates="event")
    
    def is_upcoming(self):
        """Check if event is upcoming"""
        return self.event_date > datetime.utcnow()
    
    def __repr__(self):
        return f"<Event(id={self.id}, title='{self.title}', type='{self.event_type}')>"

class Registration(Base):
    """Event registration model"""
    __tablename__ = 'registrations'
    
    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False)
    student_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    registration_status = Column(String(50), default='pending')  # pending, confirmed, cancelled
    attendance_status = Column(String(50), default='absent')  # absent, present
    registered_via_link = Column(Boolean, default=False)  # True if registered via official link
    registered_in_app = Column(Boolean, default=False)  # True if marked "I have registered"
    registration_date = Column(DateTime, default=datetime.utcnow)
    attended_at = Column(DateTime)
    verified_by = Column(Integer, ForeignKey('users.id'))  # Faculty who verified
    verification_notes = Column(Text)
    
    # Relationships
    event = relationship("Event", back_populates="registrations")
    student = relationship("User", back_populates="registrations")
    verifier = relationship("User", foreign_keys=[verified_by])
    
    def __repr__(self):
        return f"<Registration(id={self.id}, event={self.event_id}, student={self.student_id})>"

class SocialInteraction(Base):
    """Social interactions with events"""
    __tablename__ = 'social_interactions'
    
    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    interaction_type = Column(String(50), nullable=False)  # like, favorite, interested, share
    interacted_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)  # Additional data like share platform, etc.
    
    # Relationships
    event = relationship("Event", back_populates="social_interactions")
    user = relationship("User", back_populates="social_interactions")
    
    __table_args__ = (UniqueConstraint('event_id', 'user_id', 'interaction_type', 
                                       name='unique_interaction'),)
    
    def __repr__(self):
        return f"<SocialInteraction(event={self.event_id}, user={self.user_id}, type='{self.interaction_type}')>"

class Comment(Base):
    """Comments on events"""
    __tablename__ = 'comments'
    
    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey('events.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    comment_text = Column(Text, nullable=False)
    parent_comment_id = Column(Integer, ForeignKey('comments.id'))  # For replies
    is_approved = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    event = relationship("Event", back_populates="comments")
    user = relationship("User", back_populates="comments")
    parent = relationship("Comment", remote_side=[id], backref='replies')
    
    def __repr__(self):
        return f"<Comment(id={self.id}, event={self.event_id}, user={self.user_id})>"

class Analytics(Base):
    """Analytics tracking"""
    __tablename__ = 'analytics'
    
    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, ForeignKey('events.id'))
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(JSON, nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    
    event = relationship("Event")
    
    def __repr__(self):
        return f"<Analytics(id={self.id}, metric='{self.metric_name}')>"
