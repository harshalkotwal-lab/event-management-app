# database/clients.py
"""
Database clients for Supabase and SQLite
"""
import os
import sqlite3
import logging
from typing import Dict, Any, Optional
import streamlit as st

logger = logging.getLogger(__name__)

class SupabaseClient:
    """Supabase PostgreSQL client"""
    def __init__(self):
        self.url = None
        self.key = None
        self.headers = None
        self.is_configured = False
        self._initialize()
    
    def _initialize(self):
        """Initialize Supabase connection"""
        try:
            if hasattr(st, 'secrets') and 'SUPABASE' in st.secrets:
                self.url = st.secrets.SUPABASE.get('url', '').rstrip('/')
                self.key = st.secrets.SUPABASE.get('key', '')
                
                if self.url and self.key:
                    if not self.url.startswith(('http://', 'https://')):
                        self.url = f'https://{self.url}'
                    
                    self.headers = {
                        'apikey': self.key,
                        'Authorization': f'Bearer {self.key}',
                        'Content-Type': 'application/json',
                        'Prefer': 'return=minimal'
                    }
                    self.is_configured = True
                    logger.info("✅ Supabase configured successfully")
            else:
                logger.warning("⚠️ Supabase credentials not found")
        except Exception as e:
            logger.error(f"Supabase init error: {e}")
    
    def execute_query(self, table, method='GET', data=None, filters=None):
        """Execute REST API query"""
        if not self.is_configured:
            return None
        
        try:
            import requests
            url = f"{self.url}/rest/v1/{table}"
            
            # Add filters
            params = []
            if filters:
                for k, v in filters.items():
                    if v is not None:
                        params.append(f"{k}=eq.{v}")
            
            if params:
                url = f"{url}?{'&'.join(params)}"
            
            # Make request
            timeout = 10
            if method == 'GET':
                response = requests.get(url, headers=self.headers, timeout=timeout)
                return response.json() if response.text else []
            elif method == 'POST':
                response = requests.post(url, headers=self.headers, json=data, timeout=timeout)
                return response.status_code in [200, 201]
            elif method == 'PATCH':
                response = requests.patch(url, headers=self.headers, json=data, timeout=timeout)
                return response.status_code in [200, 204]
            elif method == 'DELETE':
                response = requests.delete(url, headers=self.headers, timeout=timeout)
                return response.status_code in [200, 204]
        except Exception as e:
            logger.error(f"Supabase error: {e}")
            return None

class SQLiteClient:
    """SQLite client for local development"""
    def __init__(self, db_path="data/event_management.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize()
    
    def _initialize(self):
        """Initialize SQLite database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info("✅ SQLite database initialized")
        except Exception as e:
            logger.error(f"SQLite initialization error: {e}")
            raise
    
    def execute(self, query, params=None, fetchone=False, fetchall=False, commit=False):
        """Execute SQL query"""
        try:
            cursor = self.conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if commit:
                self.conn.commit()
            
            if fetchone:
                result = cursor.fetchone()
                return dict(result) if result else None
            
            if fetchall:
                results = cursor.fetchall()
                return [dict(row) for row in results]
            
            return cursor
        except Exception as e:
            logger.error(f"SQLite error: {e}")
            return None
