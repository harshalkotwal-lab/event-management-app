"""
Database connection and session management
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from database.models import Base
import streamlit as st

def get_database_url():
    """Get database URL from environment or Streamlit secrets"""
    # Try Streamlit secrets first
    if 'DATABASE_URL' in st.secrets:
        return st.secrets['DATABASE_URL']
    
    # Try environment variable
    if 'DATABASE_URL' in os.environ:
        return os.environ['DATABASE_URL']
    
    # Default local PostgreSQL (for development)
    return "postgresql://postgres:password@localhost:5432/event_management"

def init_database():
    """Initialize database connection and create tables"""
    database_url = get_database_url()
    
    # Create engine with connection pooling
    engine = create_engine(
        database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False  # Set to True for SQL debugging
    )
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    return engine

def get_session():
    """Get database session with proper cleanup"""
    if 'engine' not in st.session_state:
        st.session_state.engine = init_database()
    
    # Create scoped session
    Session = scoped_session(sessionmaker(bind=st.session_state.engine))
    
    return Session()

def close_session(session):
    """Close database session"""
    session.close()
