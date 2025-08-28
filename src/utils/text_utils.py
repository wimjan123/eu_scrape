"""Text processing utilities for EU Parliament scraper."""

import re
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
import html
import unicodedata

from ..core.logging import get_logger

logger = get_logger(__name__)


def clean_html_text(html_content: str) -> str:
    """
    Clean HTML content and extract readable text.
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Cleaned text content
    """
    try:
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and handle whitespace
        text = soup.get_text()
        
        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        return text.strip()
        
    except Exception as e:
        logger.error("Failed to clean HTML text", error=str(e))
        return html_content  # Return original if cleaning fails


def normalize_speaker_name(name: str) -> str:
    """
    Normalize speaker name for consistent matching.
    
    Args:
        name: Raw speaker name
        
    Returns:
        Normalized speaker name
    """
    if not name:
        return ""
    
    # Remove HTML tags if present
    name = clean_html_text(name)
    
    # Remove titles and honorifics
    titles_pattern = r'\b(Mr|Mrs|Ms|Miss|Dr|Prof|President|Vice-President|Commissioner|Minister)\b\.?\s*'
    name = re.sub(titles_pattern, '', name, flags=re.IGNORECASE)
    
    # Remove role indicators in parentheses
    name = re.sub(r'\([^)]*\)', '', name)
    
    # Remove special characters and normalize unicode
    name = unicodedata.normalize('NFKD', name)
    
    # Clean punctuation and extra whitespace
    name = re.sub(r'[^\w\s\'-]', ' ', name)
    name = ' '.join(name.split())  # Normalize whitespace
    
    return name.strip()


def extract_speaker_info(text: str) -> Optional[Dict[str, str]]:
    """
    Extract speaker name and role from text using patterns.
    
    Args:
        text: Text containing speaker information
        
    Returns:
        Dictionary with speaker name and role, or None
    """
    # Common speaker patterns in EU Parliament documents
    speaker_patterns = [
        # "President. –" or "President (PPE). –"
        r'^([^(]+?)(?:\s*\(([^)]+)\))?\s*[.–-]\s*',
        # "Mr Smith (PPE-DE):"
        r'^((?:Mr|Mrs|Ms|Dr|Prof)\s+[^(]+?)\s*(?:\(([^)]+)\))?\s*[:\-–]',
        # "President:"
        r'^([A-Za-z\s]+?):\s*',
        # Bold or emphasis markers
        r'<(?:b|strong)>\s*([^<]+?)\s*</(?:b|strong)>\s*[:\-–]',
    ]
    
    for pattern in speaker_patterns:
        match = re.match(pattern, text.strip(), re.MULTILINE | re.IGNORECASE)
        if match:
            speaker_name = match.group(1).strip()
            role = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else None
            
            if speaker_name:
                return {
                    'name': normalize_speaker_name(speaker_name),
                    'role': role
                }
    
    return None


def extract_speech_text(element) -> str:
    """
    Extract clean speech text from HTML element.
    
    Args:
        element: BeautifulSoup element containing speech
        
    Returns:
        Cleaned speech text
    """
    if not element:
        return ""
    
    # Get text content
    text = element.get_text() if hasattr(element, 'get_text') else str(element)
    
    # Remove speaker name from beginning if present
    speaker_info = extract_speaker_info(text)
    if speaker_info and speaker_info['name']:
        # Remove speaker name and delimiter from start
        pattern = re.escape(speaker_info['name']) + r'\s*[:\-–]?\s*'
        text = re.sub(pattern, '', text, count=1, flags=re.IGNORECASE)
    
    # Clean and normalize text
    text = clean_html_text(text)
    
    # Remove procedural markers
    procedural_patterns = [
        r'\(applause\)',
        r'\(laughter\)',
        r'\(interruption\)',
        r'\(mixed reactions\)',
        r'\([^)]*applause[^)]*\)',
    ]
    
    for pattern in procedural_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # Final cleanup
    text = ' '.join(text.split())  # Normalize whitespace
    
    return text.strip()


def is_procedural_text(text: str) -> bool:
    """
    Check if text appears to be procedural rather than speech content.
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears procedural
    """
    if not text or len(text.strip()) < 10:
        return True
    
    procedural_indicators = [
        'sitting is opened',
        'sitting is closed',
        'sitting is suspended',
        'sitting is resumed', 
        'the debate is closed',
        'next item',
        'voting time',
        'written statements',
        'procedures without debate',
        'i call upon',
        'the session is adjourned'
    ]
    
    text_lower = text.lower()
    
    for indicator in procedural_indicators:
        if indicator in text_lower:
            return True
    
    return False


def extract_timestamp_from_text(text: str) -> Optional[str]:
    """
    Extract timestamp from text using common patterns.
    
    Args:
        text: Text that may contain timestamp
        
    Returns:
        Extracted timestamp string or None
    """
    timestamp_patterns = [
        r'(\d{1,2}[:.]\d{2})\s*[-–]\s*',  # 14:30 -
        r'\((\d{1,2}[:.]\d{2})\)',        # (14:30)
        r'at\s+(\d{1,2}[:.]\d{2})',      # at 14:30
        r'(\d{1,2}[:.]\d{2})\s+[AP]M',   # 2:30 PM
        r'(\d{1,2}h\d{2})',               # 14h30 (European format)
    ]
    
    for pattern in timestamp_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            timestamp = match.group(1)
            # Normalize format
            timestamp = timestamp.replace('h', ':')
            return timestamp
    
    return None


def classify_announcement_type(text: str, speaker_role: str = None) -> Optional[str]:
    """
    Classify the type of announcement based on content.
    
    Args:
        text: Announcement text
        speaker_role: Role of speaker (optional)
        
    Returns:
        Announcement type or None
    """
    text_lower = text.lower()
    
    # Session management
    session_patterns = [
        'sitting is opened',
        'sitting is closed', 
        'sitting is suspended',
        'sitting is resumed',
        'session begins',
        'session ends',
        'break',
        'adjournment'
    ]
    
    for pattern in session_patterns:
        if pattern in text_lower:
            return 'session_management'
    
    # Voting procedures
    voting_patterns = [
        'voting time',
        'time to vote',
        'vote is taken',
        'vote on',
        'voting list',
        'voting results',
        'unanimously adopted',
        'rejected'
    ]
    
    for pattern in voting_patterns:
        if pattern in text_lower:
            return 'voting_procedure'
    
    # Agenda items
    agenda_patterns = [
        'next item',
        'debate on',
        'discussion on',
        'report by',
        'question by',
        'statement by',
        'commission statement'
    ]
    
    for pattern in agenda_patterns:
        if pattern in text_lower:
            return 'agenda_item'
    
    # Procedural notices
    procedural_patterns = [
        'procedures without debate',
        'written statements',
        'i call upon',
        'order of business',
        'rule',
        'regulation'
    ]
    
    for pattern in procedural_patterns:
        if pattern in text_lower:
            return 'procedural_notice'
    
    # Role-based classification
    if speaker_role:
        role_lower = speaker_role.lower()
        if 'president' in role_lower:
            return 'session_management'
    
    # Default for short announcements
    if len(text.split()) < 20:
        return 'general_announcement'
    
    return None


def validate_speech_text_quality(text: str) -> Dict[str, Any]:
    """
    Validate speech text quality and return metrics.
    
    Args:
        text: Speech text to validate
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        'length': len(text),
        'word_count': len(text.split()) if text else 0,
        'is_valid': False,
        'quality_score': 0.0,
        'issues': []
    }
    
    if not text or not text.strip():
        metrics['issues'].append('empty_text')
        return metrics
    
    # Check minimum length
    if metrics['word_count'] < 3:
        metrics['issues'].append('too_short')
    
    # Check for excessive HTML remnants
    html_tags = len(re.findall(r'<[^>]+>', text))
    if html_tags > metrics['word_count'] * 0.1:
        metrics['issues'].append('html_remnants')
    
    # Check character diversity (not just repeated chars)
    unique_chars = len(set(text.lower().replace(' ', '')))
    if unique_chars < 5:
        metrics['issues'].append('low_diversity')
    
    # Check for reasonable sentence structure
    sentences = text.count('.') + text.count('!') + text.count('?')
    if metrics['word_count'] > 50 and sentences == 0:
        metrics['issues'].append('no_sentences')
    
    # Calculate quality score
    score = 1.0
    
    if metrics['word_count'] < 10:
        score *= 0.5
    
    if 'html_remnants' in metrics['issues']:
        score *= 0.3
        
    if 'low_diversity' in metrics['issues']:
        score *= 0.2
    
    metrics['quality_score'] = score
    metrics['is_valid'] = score > 0.5 and not metrics['issues']
    
    return metrics