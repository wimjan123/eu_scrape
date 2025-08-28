"""Time and timestamp utilities for EU Parliament scraper."""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pytz

from ..core.logging import get_logger

logger = get_logger(__name__)


def parse_timestamp(timestamp_str: str, session_date: datetime = None) -> Optional[datetime]:
    """
    Parse timestamp string to datetime object.
    
    Args:
        timestamp_str: Timestamp string in various formats
        session_date: Session date for context (optional)
        
    Returns:
        Parsed datetime object or None
    """
    if not timestamp_str:
        return None
    
    # Common timestamp formats
    time_patterns = [
        r'^(\d{1,2})[:.:](\d{2})$',  # 14:30 or 14.30
        r'^(\d{1,2})[:.:](\d{2})\s*([AP]M)$',  # 2:30 PM
        r'^(\d{1,2})h(\d{2})$',  # 14h30 (European format)
    ]
    
    for pattern in time_patterns:
        match = re.match(pattern, timestamp_str.strip(), re.IGNORECASE)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            # Handle AM/PM
            if len(match.groups()) > 2 and match.group(3):
                ampm = match.group(3).upper()
                if ampm == 'PM' and hour != 12:
                    hour += 12
                elif ampm == 'AM' and hour == 12:
                    hour = 0
            
            # Validate time
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                if session_date:
                    # Combine with session date
                    return session_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    # Return today with this time
                    today = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                    return today
    
    logger.warning("Failed to parse timestamp", timestamp_str=timestamp_str)
    return None


def format_iso8601_utc(dt: datetime) -> str:
    """
    Format datetime as ISO 8601 UTC timestamp.
    
    Args:
        dt: Datetime object
        
    Returns:
        ISO 8601 UTC formatted string
    """
    # Ensure UTC timezone
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    elif dt.tzinfo != pytz.UTC:
        dt = dt.astimezone(pytz.UTC)
    
    return dt.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'


def estimate_speech_duration(text: str, words_per_minute: int = 150) -> timedelta:
    """
    Estimate speech duration based on word count.
    
    Args:
        text: Speech text
        words_per_minute: Average speaking rate
        
    Returns:
        Estimated duration as timedelta
    """
    if not text:
        return timedelta(seconds=0)
    
    word_count = len(text.split())
    
    # Base duration calculation
    duration_minutes = word_count / words_per_minute
    
    # Adjust for different content types
    text_lower = text.lower()
    
    # Applause and interruptions add time
    interruptions = (
        text_lower.count('applause') +
        text_lower.count('laughter') + 
        text_lower.count('interruption') +
        text_lower.count('mixed reactions')
    )
    
    if interruptions > 0:
        duration_minutes *= 1.2  # 20% longer for interruptions
    
    # Question and answer sessions may be slower
    if any(word in text_lower for word in ['question', 'answer', 'mr president']):
        duration_minutes *= 1.1  # 10% longer for Q&A
    
    # Minimum duration
    duration_minutes = max(0.5, duration_minutes)  # At least 30 seconds
    
    return timedelta(minutes=duration_minutes)


def interpolate_timestamps(segments: list, session_start: datetime, 
                         session_duration: timedelta = None) -> list:
    """
    Interpolate timestamps for speech segments.
    
    Args:
        segments: List of speech segments
        session_start: Session start time
        session_duration: Total session duration (optional)
        
    Returns:
        Segments with interpolated timestamps
    """
    if not segments:
        return segments
    
    logger.info("Interpolating timestamps for segments", count=len(segments))
    
    current_time = session_start
    enhanced_segments = []
    
    for i, segment in enumerate(segments):
        # Check for explicit timestamp
        explicit_time = None
        if 'timestamp' in segment and segment['timestamp']:
            parsed_time = parse_timestamp(segment['timestamp'], session_start)
            if parsed_time:
                explicit_time = parsed_time
                current_time = explicit_time
        
        # Estimate segment duration
        speech_text = segment.get('speech_text', '')
        duration = estimate_speech_duration(speech_text)
        
        # Create enhanced segment
        enhanced_segment = segment.copy()
        enhanced_segment.update({
            'segment_start_ts': format_iso8601_utc(current_time),
            'segment_end_ts': format_iso8601_utc(current_time + duration),
            'timestamp_method': 'explicit' if explicit_time else 'estimated'
        })
        
        enhanced_segments.append(enhanced_segment)
        
        # Move to next segment start time
        current_time += duration + timedelta(seconds=5)  # Small gap between segments
    
    logger.info("Completed timestamp interpolation")
    return enhanced_segments


def parse_session_date(date_str: str) -> Optional[datetime]:
    """
    Parse session date string to datetime.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Parsed datetime or None
    """
    if not date_str:
        return None
    
    date_patterns = [
        '%Y-%m-%d',          # 2024-01-15
        '%d/%m/%Y',          # 15/01/2024
        '%d.%m.%Y',          # 15.01.2024
        '%d %B %Y',          # 15 January 2024
        '%B %d, %Y',         # January 15, 2024
    ]
    
    for pattern in date_patterns:
        try:
            return datetime.strptime(date_str.strip(), pattern)
        except ValueError:
            continue
    
    logger.warning("Failed to parse session date", date_str=date_str)
    return None


def validate_timestamp_sequence(segments: list) -> Tuple[bool, list]:
    """
    Validate that timestamps are in chronological order.
    
    Args:
        segments: List of segments with timestamps
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not segments:
        return True, issues
    
    prev_end_time = None
    
    for i, segment in enumerate(segments):
        start_ts = segment.get('segment_start_ts')
        end_ts = segment.get('segment_end_ts')
        
        if not start_ts or not end_ts:
            issues.append(f"Segment {i}: Missing timestamps")
            continue
        
        try:
            start_time = datetime.fromisoformat(start_ts.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end_ts.replace('Z', '+00:00'))
        except ValueError as e:
            issues.append(f"Segment {i}: Invalid timestamp format - {e}")
            continue
        
        # Check segment duration
        duration = end_time - start_time
        if duration.total_seconds() <= 0:
            issues.append(f"Segment {i}: End time before start time")
        elif duration.total_seconds() > 3600:  # More than 1 hour
            issues.append(f"Segment {i}: Suspiciously long duration ({duration})")
        
        # Check chronological order
        if prev_end_time and start_time < prev_end_time:
            issues.append(f"Segment {i}: Overlaps with previous segment")
        
        prev_end_time = end_time
    
    is_valid = len(issues) == 0
    return is_valid, issues


def get_session_time_bounds(segments: list) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Get the earliest and latest timestamps from segments.
    
    Args:
        segments: List of segments with timestamps
        
    Returns:
        Tuple of (earliest_time, latest_time)
    """
    if not segments:
        return None, None
    
    earliest = None
    latest = None
    
    for segment in segments:
        start_ts = segment.get('segment_start_ts')
        end_ts = segment.get('segment_end_ts')
        
        if start_ts:
            try:
                start_time = datetime.fromisoformat(start_ts.replace('Z', '+00:00'))
                if earliest is None or start_time < earliest:
                    earliest = start_time
            except ValueError:
                continue
        
        if end_ts:
            try:
                end_time = datetime.fromisoformat(end_ts.replace('Z', '+00:00'))
                if latest is None or end_time > latest:
                    latest = end_time
            except ValueError:
                continue
    
    return earliest, latest