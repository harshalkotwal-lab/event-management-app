"""
AI-powered event generator from WhatsApp/email messages
"""

import openai
import json
import re
from datetime import datetime, timedelta
import streamlit as st

class AIEventGenerator:
    """Generate structured event data from unstructured text"""
    
    def __init__(self):
        # Initialize OpenAI client (use Streamlit secrets for API key)
        self.api_key = st.secrets.get("OPENAI_API_KEY", "")
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None
            st.warning("OpenAI API key not configured. AI features will be limited.")
    
    def extract_event_info(self, text):
        """Extract event information from text using AI or regex fallback"""
        
        # Try OpenAI first if available
        if self.client:
            try:
                return self._extract_with_openai(text)
            except Exception as e:
                st.warning(f"OpenAI extraction failed: {e}. Using regex fallback.")
        
        # Fallback to regex extraction
        return self._extract_with_regex(text)
    
    def _extract_with_openai(self, text):
        """Use OpenAI to extract structured event data"""
        prompt = f"""
        Extract event information from the following text and return as JSON with these fields:
        - title: Event title (string)
        - description: Detailed event description (string)
        - event_type: Type of event (workshop, hackathon, competition, bootcamp, seminar, conference, webinar)
        - event_date: Event date in YYYY-MM-DD format (extract from text or use reasonable default)
        - venue: Event venue/location (string)
        - organizer: Event organizer (string)
        - registration_link: Registration URL if mentioned (string or null)
        
        Text: {text}
        
        Return only valid JSON, no other text.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at extracting event information from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        # Parse response
        result_text = response.choices[0].message.content.strip()
        
        # Clean response (remove markdown code blocks if present)
        result_text = re.sub(r'```json\s*', '', result_text)
        result_text = re.sub(r'\s*```', '', result_text)
        
        try:
            event_data = json.loads(result_text)
            
            # Add AI metadata
            event_data['ai_generated'] = True
            event_data['ai_prompt'] = text
            event_data['ai_metadata'] = {
                'model': 'gpt-3.5-turbo',
                'extracted_at': datetime.utcnow().isoformat()
            }
            
            return event_data
        except json.JSONDecodeError:
            st.error("Failed to parse AI response")
            return self._extract_with_regex(text)
    
    def _extract_with_regex(self, text):
        """Fallback regex-based extraction"""
        event_data = {
            'title': 'New Event',
            'description': text[:200] + '...' if len(text) > 200 else text,
            'event_type': 'workshop',
            'event_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'venue': 'G H Raisoni College',
            'organizer': 'College Department',
            'registration_link': None,
            'ai_generated': False,
            'ai_prompt': text
        }
        
        # Try to extract title (first line or sentence)
        lines = text.split('\n')
        if lines and lines[0].strip():
            event_data['title'] = lines[0].strip()[:100]
        
        # Try to extract date patterns
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # DD-MM-YYYY
            r'(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})',  # YYYY-MM-DD
            r'(?:on|date|Date)[:\s]*(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['event_date'] = match.group(1)
                break
        
        # Try to extract venue
        venue_keywords = ['at', 'venue', 'location', 'place']
        for keyword in venue_keywords:
            pattern = rf'{keyword}[:\s]*([^\n.,;]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['venue'] = match.group(1).strip()
                break
        
        # Try to extract organizer
        organizer_keywords = ['by', 'organizer', 'organized by', 'conducted by']
        for keyword in organizer_keywords:
            pattern = rf'{keyword}[:\s]*([^\n.,;]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                event_data['organizer'] = match.group(1).strip()
                break
        
        # Try to extract URLs (registration links)
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        if urls:
            event_data['registration_link'] = urls[0]
        
        return event_data
    
    def enhance_event_description(self, basic_info):
        """Use AI to enhance event description"""
        if not self.client:
            return basic_info
        
        prompt = f"""
        Enhance this event description to make it more engaging and professional:
        
        Title: {basic_info.get('title', 'Event')}
        Basic Description: {basic_info.get('description', '')}
        
        Please provide:
        1. A catchy, engaging title (max 10 words)
        2. A professional description (3-4 paragraphs)
        3. Key highlights/bullet points
        4. Who should attend
        5. What participants will learn/gain
        
        Return as JSON with fields: enhanced_title, enhanced_description, key_highlights, target_audience, learning_outcomes
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at creating engaging event descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'\s*```', '', result_text)
            
            enhanced_data = json.loads(result_text)
            
            # Merge enhanced data with basic info
            if 'enhanced_title' in enhanced_data:
                basic_info['title'] = enhanced_data['enhanced_title']
            if 'enhanced_description' in enhanced_data:
                basic_info['description'] = enhanced_data['enhanced_description']
            
            basic_info['ai_enhanced'] = True
            basic_info['ai_metadata']['enhancement'] = enhanced_data
            
            return basic_info
            
        except Exception as e:
            st.warning(f"AI enhancement failed: {e}")
            return basic_info
