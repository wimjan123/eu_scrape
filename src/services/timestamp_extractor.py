#!/usr/bin/env python3
"""
Advanced Timestamp Extraction and Temporal Analysis System

Comprehensive timestamp extraction with speech-timestamp association,
temporal sequence validation, duration calculations, and timeline reconstruction
for EU Parliament session analysis.

Key Features:
- Multiple timestamp format recognition (HH:MM, HH:MM:SS, contextual times)
- Speech-to-timestamp association with confidence scoring
- Temporal sequence validation and gap detection
- Duration analysis and speech timing metrics
- Timeline reconstruction with session flow analysis
- Temporal quality assessment and validation
- Integration with content extraction and quality validation systems
"""

import asyncio
import re
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Set, Union, NamedTuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time as datetime_time
from enum import Enum
import logging
import statistics
from collections import defaultdict, OrderedDict

import pandas as pd
import numpy as np

from ..models.speech import RawSpeechSegment
from ..models.session import SessionMetadata
from ..core.logging import get_logger
from ..core.metrics import MetricsCollector

logger = get_logger(__name__)


class TimestampFormat(Enum):
    """Recognized timestamp formats"""
    HH_MM = "HH:MM"                    # 14:30
    HH_MM_SS = "HH:MM:SS"              # 14:30:45
    HH_MM_AMPM = "HH:MM AM/PM"         # 2:30 PM
    HH_MM_SS_AMPM = "HH:MM:SS AM/PM"   # 2:30:45 PM
    CONTEXTUAL = "contextual"           # "at fourteen thirty"
    SESSION_RELATIVE = "session_relative" # "45 minutes into session"
    UNKNOWN = "unknown"


class TimestampConfidence(Enum):
    """Confidence levels for timestamp extraction"""
    HIGH = "high"           # 90-100% - Clearly formatted timestamp
    MEDIUM = "medium"       # 70-89% - Contextual or inferred timestamp
    LOW = "low"             # 50-69% - Uncertain or estimated timestamp
    ESTIMATED = "estimated" # 0-49% - System-generated estimate


class TemporalEventType(Enum):
    """Types of temporal events in parliament sessions"""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    BREAK_START = "break_start"
    BREAK_END = "break_end"
    VOTE_START = "vote_start"
    VOTE_END = "vote_end"
    INTERRUPTION = "interruption"
    APPLAUSE = "applause"
    PROCEDURAL = "procedural"


@dataclass
class ExtractedTimestamp:
    """Individual extracted timestamp with metadata"""
    timestamp_id: str
    original_text: str
    extracted_time: datetime_time
    format_type: TimestampFormat
    confidence: TimestampConfidence
    
    # Position information
    position_start: int
    position_end: int
    context_before: str = ""
    context_after: str = ""
    
    # Association information
    associated_speech_id: Optional[str] = None
    event_type: TemporalEventType = TemporalEventType.SPEECH_START
    
    # Validation metadata
    is_valid_time: bool = True
    validation_notes: List[str] = field(default_factory=list)
    extraction_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SpeechTimestamp:
    """Timestamp information for a speech segment"""
    speech_id: str
    start_time: Optional[datetime_time] = None
    end_time: Optional[datetime_time] = None
    duration_seconds: Optional[float] = None
    
    # Timestamp sources
    start_timestamp: Optional[ExtractedTimestamp] = None
    end_timestamp: Optional[ExtractedTimestamp] = None
    
    # Quality metrics
    timing_confidence: float = 0.0
    has_explicit_timestamps: bool = False
    is_estimated: bool = False
    
    # Temporal context
    sequence_number: int = 0
    gap_before_seconds: Optional[float] = None
    gap_after_seconds: Optional[float] = None


@dataclass
class SessionTimeline:
    """Complete temporal timeline for a parliament session"""
    session_id: str
    session_date: datetime
    
    # Timeline boundaries
    session_start_time: Optional[datetime_time] = None
    session_end_time: Optional[datetime_time] = None
    total_duration_minutes: Optional[float] = None
    
    # Timeline events
    speech_timestamps: List[SpeechTimestamp] = field(default_factory=list)
    temporal_events: List[ExtractedTimestamp] = field(default_factory=list)
    
    # Timeline quality
    timeline_completeness: float = 0.0
    timestamp_coverage: float = 0.0
    temporal_consistency: float = 0.0
    
    # Timeline statistics
    total_speeches: int = 0
    timestamped_speeches: int = 0
    average_speech_duration: Optional[float] = None
    longest_speech_duration: Optional[float] = None
    total_break_time: Optional[float] = None
    
    # Timeline metadata
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    validation_issues: List[str] = field(default_factory=list)


class TimestampPatternMatcher:
    """Advanced pattern matching for various timestamp formats"""
    
    def __init__(self):
        # Comprehensive timestamp patterns
        self.timestamp_patterns = {
            # Standard time formats
            TimestampFormat.HH_MM_SS: [
                r'\b([01]?\d|2[0-3]):([0-5]\d):([0-5]\d)\b',
                r'\b(\d{1,2})\.([0-5]\d)\.([0-5]\d)\b',  # European format
            ],
            TimestampFormat.HH_MM: [
                r'\b([01]?\d|2[0-3]):([0-5]\d)\b',
                r'\b(\d{1,2})\.([0-5]\d)\b',  # European format
                r'\b(\d{1,2})h([0-5]\d)\b',   # French format
            ],
            TimestampFormat.HH_MM_SS_AMPM: [
                r'\b(\d{1,2}):([0-5]\d):([0-5]\d)\s*(AM|PM|am|pm)\b',
                r'\b(\d{1,2}):([0-5]\d):([0-5]\d)\s*(A\.M\.|P\.M\.)\b',
            ],
            TimestampFormat.HH_MM_AMPM: [
                r'\b(\d{1,2}):([0-5]\d)\s*(AM|PM|am|pm)\b',
                r'\b(\d{1,2}):([0-5]\d)\s*(A\.M\.|P\.M\.)\b',
            ],
            TimestampFormat.CONTEXTUAL: [
                r'\bat\s+(\d{1,2})\s*(?:hours?\s*)?(?:and\s*)?(\d{1,2})?\s*(?:minutes?)?\b',
                r'\b(?:around|about|approximately)\s+(\d{1,2})(?::(\d{2}))?\b',
                r'\b(?:quarter\s+past|half\s+past|\d+\s+minutes?\s+past)\s+(\d{1,2})\b',
            ],
        }
        
        # Context indicators for timestamp identification
        self.timestamp_contexts = {
            'session_times': [
                'sitting.*?(?:opens?|begins?|starts?)',
                'sitting.*?(?:closes?|ends?|adjourns?)',
                'session.*?(?:opens?|begins?|starts?)',
                'session.*?(?:closes?|ends?|adjourns?)',
            ],
            'speech_times': [
                'at.*?(?:says?|speaks?|states?)',
                'speaker.*?(?:begins?|continues?)',
                '(?:mr|mrs|ms|madam).*?president.*?at',
            ],
            'procedural_times': [
                'vote.*?(?:at|begins?|starts?)',
                'break.*?(?:at|until)',
                'recess.*?(?:at|until)',
                'applause.*?(?:at|lasting)',
            ],
        }
        
        # European Parliament specific time indicators
        self.parliament_time_indicators = [
            r'sitting\s+(?:opens?|begins?|starts?)\s+at\s+',
            r'president\s*\.\s*–\s*(?:good\s+morning|the\s+sitting)',
            r'(?:madam|mr)\s+president\s*\.\s*–',
            r'commissioner\s+\w+\s*\.\s*–',
            r'(?:vote|voting)\s+(?:at|begins?)\s+',
        ]
    
    def extract_timestamps_from_content(self, content: str) -> List[ExtractedTimestamp]:
        """Extract all timestamps from content with confidence scoring"""
        extracted_timestamps = []
        
        # Process each timestamp format
        for format_type, patterns in self.timestamp_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                
                for match in matches:
                    timestamp = self._create_timestamp_from_match(
                        match, format_type, content
                    )
                    if timestamp:
                        extracted_timestamps.append(timestamp)
        
        # Sort by position in content
        extracted_timestamps.sort(key=lambda x: x.position_start)
        
        # Remove duplicates and assign unique IDs
        unique_timestamps = self._deduplicate_timestamps(extracted_timestamps)
        
        return unique_timestamps
    
    def _create_timestamp_from_match(
        self, 
        match: re.Match, 
        format_type: TimestampFormat, 
        content: str
    ) -> Optional[ExtractedTimestamp]:
        """Create timestamp object from regex match"""
        try:
            groups = match.groups()
            original_text = match.group(0)
            
            # Parse time components based on format
            time_obj = self._parse_time_components(groups, format_type)
            if not time_obj:
                return None
            
            # Extract context
            start_pos = max(0, match.start() - 50)
            end_pos = min(len(content), match.end() + 50)
            context_before = content[start_pos:match.start()]
            context_after = content[match.end():end_pos]
            
            # Determine confidence and event type
            confidence = self._calculate_timestamp_confidence(
                original_text, context_before, context_after, format_type
            )
            
            event_type = self._determine_event_type(
                original_text, context_before, context_after
            )
            
            # Generate unique ID
            timestamp_id = f"ts_{match.start()}_{int(time.time() * 1000) % 100000}"
            
            return ExtractedTimestamp(
                timestamp_id=timestamp_id,
                original_text=original_text,
                extracted_time=time_obj,
                format_type=format_type,
                confidence=confidence,
                position_start=match.start(),
                position_end=match.end(),
                context_before=context_before.strip(),
                context_after=context_after.strip(),
                event_type=event_type
            )
            
        except Exception as e:
            logger.warning(f"Failed to create timestamp from match: {e}")
            return None
    
    def _parse_time_components(self, groups: Tuple, format_type: TimestampFormat) -> Optional[datetime_time]:
        """Parse time components into datetime.time object"""
        try:
            if format_type in [TimestampFormat.HH_MM_SS, TimestampFormat.HH_MM_SS_AMPM]:
                hour = int(groups[0])
                minute = int(groups[1])
                second = int(groups[2]) if len(groups) > 2 and groups[2] else 0
                
                # Handle AM/PM
                if format_type == TimestampFormat.HH_MM_SS_AMPM and len(groups) > 3:
                    ampm = groups[3].upper()
                    if ampm in ['PM', 'P.M.'] and hour < 12:
                        hour += 12
                    elif ampm in ['AM', 'A.M.'] and hour == 12:
                        hour = 0
                
                if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                    return datetime_time(hour, minute, second)
                    
            elif format_type in [TimestampFormat.HH_MM, TimestampFormat.HH_MM_AMPM]:
                hour = int(groups[0])
                minute = int(groups[1])
                
                # Handle AM/PM
                if format_type == TimestampFormat.HH_MM_AMPM and len(groups) > 2:
                    ampm = groups[2].upper()
                    if ampm in ['PM', 'P.M.'] and hour < 12:
                        hour += 12
                    elif ampm in ['AM', 'A.M.'] and hour == 12:
                        hour = 0
                
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return datetime_time(hour, minute, 0)
                    
            elif format_type == TimestampFormat.CONTEXTUAL:
                # Handle contextual times like "at fourteen thirty"
                hour = int(groups[0]) if groups[0] else 0
                minute = int(groups[1]) if len(groups) > 1 and groups[1] else 0
                
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return datetime_time(hour, minute, 0)
            
            return None
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse time components: {e}")
            return None
    
    def _calculate_timestamp_confidence(
        self, 
        original_text: str, 
        context_before: str, 
        context_after: str, 
        format_type: TimestampFormat
    ) -> TimestampConfidence:
        """Calculate confidence level for extracted timestamp"""
        
        # Base confidence by format type
        format_confidence = {
            TimestampFormat.HH_MM_SS: 0.9,
            TimestampFormat.HH_MM: 0.8,
            TimestampFormat.HH_MM_SS_AMPM: 0.85,
            TimestampFormat.HH_MM_AMPM: 0.75,
            TimestampFormat.CONTEXTUAL: 0.6,
            TimestampFormat.SESSION_RELATIVE: 0.5,
        }.get(format_type, 0.3)
        
        # Context boost
        context_text = (context_before + " " + context_after).lower()
        context_boost = 0.0
        
        # Check for parliament-specific context
        for indicator in self.parliament_time_indicators:
            if re.search(indicator, context_text, re.IGNORECASE):
                context_boost += 0.15
                break
        
        # Check for general time context
        for context_type, patterns in self.timestamp_contexts.items():
            for pattern in patterns:
                if re.search(pattern, context_text, re.IGNORECASE):
                    context_boost += 0.1
                    break
        
        # Calculate final confidence
        final_confidence = min(1.0, format_confidence + context_boost)
        
        # Map to confidence enum
        if final_confidence >= 0.9:
            return TimestampConfidence.HIGH
        elif final_confidence >= 0.7:
            return TimestampConfidence.MEDIUM
        elif final_confidence >= 0.5:
            return TimestampConfidence.LOW
        else:
            return TimestampConfidence.ESTIMATED
    
    def _determine_event_type(
        self, 
        original_text: str, 
        context_before: str, 
        context_after: str
    ) -> TemporalEventType:
        """Determine the type of temporal event"""
        
        context_text = (context_before + " " + original_text + " " + context_after).lower()
        
        # Session events
        if any(phrase in context_text for phrase in ['sitting opens', 'sitting begins', 'session starts']):
            return TemporalEventType.SESSION_START
        elif any(phrase in context_text for phrase in ['sitting closes', 'sitting ends', 'session ends']):
            return TemporalEventType.SESSION_END
        
        # Vote events
        elif any(phrase in context_text for phrase in ['vote', 'voting', 'ballot']):
            if any(phrase in context_text for phrase in ['begins', 'starts', 'at']):
                return TemporalEventType.VOTE_START
            else:
                return TemporalEventType.VOTE_END
        
        # Break events
        elif any(phrase in context_text for phrase in ['break', 'recess', 'adjourn']):
            if any(phrase in context_text for phrase in ['until', 'resume']):
                return TemporalEventType.BREAK_START
            else:
                return TemporalEventType.BREAK_END
        
        # Speech events (default for most parliament content)
        else:
            return TemporalEventType.SPEECH_START
    
    def _deduplicate_timestamps(self, timestamps: List[ExtractedTimestamp]) -> List[ExtractedTimestamp]:
        """Remove duplicate timestamps and assign unique IDs"""
        unique_timestamps = []
        seen_times = set()
        
        for i, timestamp in enumerate(timestamps):
            # Create a key based on time and position
            time_key = (
                timestamp.extracted_time.hour,
                timestamp.extracted_time.minute,
                timestamp.extracted_time.second,
                timestamp.position_start // 100  # Group by approximate position
            )
            
            if time_key not in seen_times:
                # Assign sequential ID
                timestamp.timestamp_id = f"ts_{i:04d}_{timestamp.extracted_time.strftime('%H%M%S')}"
                unique_timestamps.append(timestamp)
                seen_times.add(time_key)
        
        return unique_timestamps


class SpeechTimestampAssociator:
    """Associates extracted timestamps with speech segments"""
    
    def __init__(self):
        self.association_threshold = 200  # Maximum character distance for association
        
    def associate_timestamps_with_speeches(
        self,
        speeches: List[RawSpeechSegment],
        timestamps: List[ExtractedTimestamp]
    ) -> List[SpeechTimestamp]:
        """Associate timestamps with speech segments"""
        
        speech_timestamps = []
        
        for speech in speeches:
            speech_timestamp = SpeechTimestamp(
                speech_id=getattr(speech, 'segment_id', f'speech_{speech.sequence_number}'),
                sequence_number=speech.sequence_number
            )
            
            # Find the best timestamp association
            best_timestamp = self._find_best_timestamp_for_speech(speech, timestamps)
            
            if best_timestamp:
                speech_timestamp.start_timestamp = best_timestamp
                speech_timestamp.start_time = best_timestamp.extracted_time
                speech_timestamp.has_explicit_timestamps = True
                speech_timestamp.timing_confidence = self._confidence_to_float(best_timestamp.confidence)
                
                # Mark timestamp as associated
                best_timestamp.associated_speech_id = speech_timestamp.speech_id
            
            # Estimate duration if possible
            speech_timestamp.duration_seconds = self._estimate_speech_duration(speech)
            
            # Calculate end time if we have start time and duration
            if speech_timestamp.start_time and speech_timestamp.duration_seconds:
                start_datetime = datetime.combine(datetime.today(), speech_timestamp.start_time)
                end_datetime = start_datetime + timedelta(seconds=speech_timestamp.duration_seconds)
                speech_timestamp.end_time = end_datetime.time()
            
            speech_timestamps.append(speech_timestamp)
        
        # Calculate gaps between speeches
        self._calculate_temporal_gaps(speech_timestamps)
        
        return speech_timestamps
    
    def _find_best_timestamp_for_speech(
        self, 
        speech: RawSpeechSegment, 
        timestamps: List[ExtractedTimestamp]
    ) -> Optional[ExtractedTimestamp]:
        """Find the best timestamp match for a speech segment"""
        
        if not hasattr(speech, 'start_position'):
            return None
        
        speech_start = speech.start_position
        candidates = []
        
        # Find timestamps near this speech
        for timestamp in timestamps:
            distance = abs(timestamp.position_start - speech_start)
            
            if distance <= self.association_threshold:
                # Calculate association score
                score = self._calculate_association_score(speech, timestamp, distance)
                candidates.append((timestamp, score))
        
        # Return the best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def _calculate_association_score(
        self, 
        speech: RawSpeechSegment, 
        timestamp: ExtractedTimestamp, 
        distance: int
    ) -> float:
        """Calculate association score between speech and timestamp"""
        
        score = 0.0
        
        # Distance factor (closer is better)
        max_distance = self.association_threshold
        distance_score = 1.0 - (distance / max_distance)
        score += distance_score * 0.4
        
        # Confidence factor
        confidence_score = self._confidence_to_float(timestamp.confidence)
        score += confidence_score * 0.3
        
        # Context relevance
        if hasattr(speech, 'speaker_raw') and speech.speaker_raw:
            speaker_text = speech.speaker_raw.lower()
            context_text = (timestamp.context_before + " " + timestamp.context_after).lower()
            
            # Check if speaker mentioned near timestamp
            if any(word in context_text for word in speaker_text.split()[:3]):
                score += 0.2
        
        # Event type appropriateness
        if timestamp.event_type == TemporalEventType.SPEECH_START:
            score += 0.1
        
        return score
    
    def _confidence_to_float(self, confidence: TimestampConfidence) -> float:
        """Convert confidence enum to float value"""
        mapping = {
            TimestampConfidence.HIGH: 0.95,
            TimestampConfidence.MEDIUM: 0.8,
            TimestampConfidence.LOW: 0.6,
            TimestampConfidence.ESTIMATED: 0.3,
        }
        return mapping.get(confidence, 0.0)
    
    def _estimate_speech_duration(self, speech: RawSpeechSegment) -> Optional[float]:
        """Estimate speech duration based on text length"""
        if not hasattr(speech, 'speech_text') or not speech.speech_text:
            return None
        
        # Rough estimate: 150 words per minute, average 5 characters per word
        text_length = len(speech.speech_text)
        estimated_words = text_length / 5  # Average characters per word
        estimated_minutes = estimated_words / 150  # Words per minute
        estimated_seconds = estimated_minutes * 60
        
        # Minimum duration of 10 seconds, maximum of 20 minutes
        return max(10.0, min(1200.0, estimated_seconds))
    
    def _calculate_temporal_gaps(self, speech_timestamps: List[SpeechTimestamp]):
        """Calculate gaps between consecutive speeches"""
        
        # Sort by sequence number
        speech_timestamps.sort(key=lambda x: x.sequence_number)
        
        for i in range(len(speech_timestamps)):
            current = speech_timestamps[i]
            
            # Calculate gap before
            if i > 0:
                previous = speech_timestamps[i - 1]
                if (current.start_time and previous.end_time and 
                    current.start_time > previous.end_time):
                    
                    prev_end = datetime.combine(datetime.today(), previous.end_time)
                    curr_start = datetime.combine(datetime.today(), current.start_time)
                    gap = (curr_start - prev_end).total_seconds()
                    current.gap_before_seconds = gap
            
            # Calculate gap after
            if i < len(speech_timestamps) - 1:
                next_speech = speech_timestamps[i + 1]
                if (current.end_time and next_speech.start_time and 
                    next_speech.start_time > current.end_time):
                    
                    curr_end = datetime.combine(datetime.today(), current.end_time)
                    next_start = datetime.combine(datetime.today(), next_speech.start_time)
                    gap = (next_start - curr_end).total_seconds()
                    current.gap_after_seconds = gap


class TimelineAnalyzer:
    """Analyzes complete session timeline for quality and consistency"""
    
    def __init__(self):
        self.quality_thresholds = {
            'minimum_timestamp_coverage': 0.3,  # 30% of speeches should have timestamps
            'timeline_completeness': 0.5,       # 50% timeline information present
            'temporal_consistency': 0.7,        # 70% temporal consistency required
        }
    
    def build_session_timeline(
        self,
        session_metadata: SessionMetadata,
        speech_timestamps: List[SpeechTimestamp],
        temporal_events: List[ExtractedTimestamp]
    ) -> SessionTimeline:
        """Build complete session timeline with quality analysis"""
        
        timeline = SessionTimeline(
            session_id=session_metadata.session_id,
            session_date=session_metadata.date,
            speech_timestamps=speech_timestamps,
            temporal_events=temporal_events
        )
        
        # Analyze timeline boundaries
        self._analyze_session_boundaries(timeline, temporal_events)
        
        # Calculate timeline statistics
        self._calculate_timeline_statistics(timeline)
        
        # Assess timeline quality
        self._assess_timeline_quality(timeline)
        
        # Validate temporal consistency
        self._validate_temporal_consistency(timeline)
        
        return timeline
    
    def _analyze_session_boundaries(
        self, 
        timeline: SessionTimeline, 
        temporal_events: List[ExtractedTimestamp]
    ):
        """Analyze session start and end times"""
        
        # Find session start events
        start_events = [
            event for event in temporal_events 
            if event.event_type == TemporalEventType.SESSION_START
        ]
        
        if start_events:
            # Use the earliest high-confidence start event
            start_events.sort(key=lambda x: (x.extracted_time, -self._confidence_to_float(x.confidence)))
            timeline.session_start_time = start_events[0].extracted_time
        
        # Find session end events
        end_events = [
            event for event in temporal_events 
            if event.event_type == TemporalEventType.SESSION_END
        ]
        
        if end_events:
            # Use the latest high-confidence end event
            end_events.sort(key=lambda x: (x.extracted_time, -self._confidence_to_float(x.confidence)), reverse=True)
            timeline.session_end_time = end_events[0].extracted_time
        
        # Calculate total duration
        if timeline.session_start_time and timeline.session_end_time:
            start_dt = datetime.combine(timeline.session_date, timeline.session_start_time)
            end_dt = datetime.combine(timeline.session_date, timeline.session_end_time)
            
            # Handle day boundary crossing
            if end_dt < start_dt:
                end_dt += timedelta(days=1)
            
            duration = (end_dt - start_dt).total_seconds() / 60  # Convert to minutes
            timeline.total_duration_minutes = duration
    
    def _calculate_timeline_statistics(self, timeline: SessionTimeline):
        """Calculate comprehensive timeline statistics"""
        
        speech_timestamps = timeline.speech_timestamps
        
        # Basic counts
        timeline.total_speeches = len(speech_timestamps)
        timeline.timestamped_speeches = len([
            st for st in speech_timestamps if st.has_explicit_timestamps
        ])
        
        # Duration statistics
        durations = [
            st.duration_seconds for st in speech_timestamps 
            if st.duration_seconds is not None
        ]
        
        if durations:
            timeline.average_speech_duration = statistics.mean(durations)
            timeline.longest_speech_duration = max(durations)
        
        # Calculate break time
        gaps = [
            st.gap_after_seconds for st in speech_timestamps 
            if st.gap_after_seconds is not None and st.gap_after_seconds > 60  # Gaps > 1 minute
        ]
        
        if gaps:
            timeline.total_break_time = sum(gaps) / 60  # Convert to minutes
        
        # Timeline coverage metrics
        if timeline.total_speeches > 0:
            timeline.timestamp_coverage = timeline.timestamped_speeches / timeline.total_speeches
        
        # Timeline completeness (how much timeline information we have)
        completeness_factors = [
            1.0 if timeline.session_start_time else 0.0,
            1.0 if timeline.session_end_time else 0.0,
            1.0 if timeline.total_duration_minutes else 0.0,
            1.0 if timeline.timestamped_speeches > 0 else 0.0,
            1.0 if timeline.average_speech_duration else 0.0,
        ]
        
        timeline.timeline_completeness = statistics.mean(completeness_factors)
    
    def _assess_timeline_quality(self, timeline: SessionTimeline):
        """Assess overall timeline quality"""
        
        quality_factors = []
        
        # Timestamp coverage quality
        coverage_score = min(1.0, timeline.timestamp_coverage / self.quality_thresholds['minimum_timestamp_coverage'])
        quality_factors.append(coverage_score * 0.4)
        
        # Timeline completeness quality
        completeness_score = timeline.timeline_completeness
        quality_factors.append(completeness_score * 0.3)
        
        # Temporal consistency (will be calculated in validation)
        consistency_score = timeline.temporal_consistency if timeline.temporal_consistency > 0 else 0.5
        quality_factors.append(consistency_score * 0.3)
        
        # Overall timeline quality
        if quality_factors:
            overall_quality = sum(quality_factors)
            # Normalize to 0-1 range
            timeline.extraction_metadata['timeline_quality_score'] = min(1.0, overall_quality)
    
    def _validate_temporal_consistency(self, timeline: SessionTimeline):
        """Validate temporal consistency and detect anomalies"""
        
        issues = []
        consistency_score = 1.0
        
        # Check session boundary consistency
        if timeline.session_start_time and timeline.session_end_time:
            if timeline.session_end_time <= timeline.session_start_time:
                issues.append("Session end time is before or equal to start time")
                consistency_score -= 0.3
        
        # Check speech sequence consistency
        speech_timestamps = [st for st in timeline.speech_timestamps if st.start_time]
        speech_timestamps.sort(key=lambda x: x.sequence_number)
        
        time_inconsistencies = 0
        for i in range(1, len(speech_timestamps)):
            prev_speech = speech_timestamps[i - 1]
            curr_speech = speech_timestamps[i]
            
            if prev_speech.start_time and curr_speech.start_time:
                if curr_speech.start_time < prev_speech.start_time:
                    time_inconsistencies += 1
        
        if speech_timestamps and time_inconsistencies > 0:
            inconsistency_rate = time_inconsistencies / len(speech_timestamps)
            if inconsistency_rate > 0.1:  # More than 10% inconsistent
                issues.append(f"High temporal inconsistency rate: {inconsistency_rate:.1%}")
                consistency_score -= inconsistency_rate * 0.5
        
        # Check for unrealistic durations
        long_speeches = [
            st for st in timeline.speech_timestamps 
            if st.duration_seconds and st.duration_seconds > 1800  # 30 minutes
        ]
        
        if long_speeches:
            issues.append(f"Found {len(long_speeches)} unusually long speeches (>30 minutes)")
            consistency_score -= len(long_speeches) * 0.05
        
        # Check for unrealistic gaps
        long_gaps = [
            st for st in timeline.speech_timestamps 
            if st.gap_after_seconds and st.gap_after_seconds > 3600  # 1 hour
        ]
        
        if long_gaps:
            issues.append(f"Found {len(long_gaps)} unusually long gaps (>1 hour)")
            consistency_score -= len(long_gaps) * 0.05
        
        # Set final consistency score
        timeline.temporal_consistency = max(0.0, consistency_score)
        timeline.validation_issues.extend(issues)
    
    def _confidence_to_float(self, confidence: TimestampConfidence) -> float:
        """Convert confidence enum to float value"""
        mapping = {
            TimestampConfidence.HIGH: 0.95,
            TimestampConfidence.MEDIUM: 0.8,
            TimestampConfidence.LOW: 0.6,
            TimestampConfidence.ESTIMATED: 0.3,
        }
        return mapping.get(confidence, 0.0)


class AdvancedTimestampExtractor:
    """Main comprehensive timestamp extraction system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.pattern_matcher = TimestampPatternMatcher()
        self.speech_associator = SpeechTimestampAssociator()
        self.timeline_analyzer = TimelineAnalyzer()
        
        # Performance optimization
        self.metrics = MetricsCollector()
        
        # Processing statistics
        self.extraction_stats = {
            'sessions_processed': 0,
            'timestamps_extracted': 0,
            'speeches_timestamped': 0,
            'average_extraction_time': 0.0,
            'timeline_completeness_avg': 0.0
        }
        
        logger.info("Advanced Timestamp Extractor initialized")
    
    async def extract_comprehensive_timestamps(
        self,
        content: str,
        speeches: List[RawSpeechSegment],
        session_metadata: SessionMetadata
    ) -> SessionTimeline:
        """
        Extract comprehensive timestamp information for a session
        
        Args:
            content: Raw session content
            speeches: List of speech segments
            session_metadata: Session metadata
            
        Returns:
            Complete session timeline with timestamps and analysis
        """
        start_time = time.time()
        
        logger.info(f"Starting comprehensive timestamp extraction for session: {session_metadata.session_id}")
        
        try:
            # Step 1: Extract all timestamps from content
            extracted_timestamps = self.pattern_matcher.extract_timestamps_from_content(content)
            logger.info(f"Extracted {len(extracted_timestamps)} timestamps from content")
            
            # Step 2: Associate timestamps with speech segments
            speech_timestamps = self.speech_associator.associate_timestamps_with_speeches(
                speeches, extracted_timestamps
            )
            logger.info(f"Associated timestamps with {len(speech_timestamps)} speech segments")
            
            # Step 3: Filter temporal events (non-speech timestamps)
            temporal_events = [
                ts for ts in extracted_timestamps 
                if ts.event_type in [
                    TemporalEventType.SESSION_START,
                    TemporalEventType.SESSION_END,
                    TemporalEventType.VOTE_START,
                    TemporalEventType.VOTE_END,
                    TemporalEventType.BREAK_START,
                    TemporalEventType.BREAK_END
                ]
            ]
            
            # Step 4: Build comprehensive timeline
            timeline = self.timeline_analyzer.build_session_timeline(
                session_metadata, speech_timestamps, temporal_events
            )
            
            # Step 5: Add extraction metadata
            extraction_duration = (time.time() - start_time) * 1000
            timeline.extraction_metadata.update({
                'extraction_duration_ms': extraction_duration,
                'total_timestamps_found': len(extracted_timestamps),
                'temporal_events_count': len(temporal_events),
                'extraction_version': '2.0',
                'extraction_timestamp': datetime.now().isoformat()
            })
            
            # Update statistics
            self._update_extraction_statistics(timeline, extraction_duration)
            
            # Update metrics
            self.metrics.increment('timestamp_extractions_completed')
            self.metrics.histogram('timestamp_extraction_duration', extraction_duration)
            self.metrics.gauge('timestamps_per_session', len(extracted_timestamps))
            self.metrics.gauge('timeline_quality', timeline.extraction_metadata.get('timeline_quality_score', 0.0))
            
            logger.info(f"Timestamp extraction completed for session: {session_metadata.session_id} - "
                       f"Timeline quality: {timeline.extraction_metadata.get('timeline_quality_score', 0.0):.3f}, "
                       f"Coverage: {timeline.timestamp_coverage:.1%}")
            
            return timeline
            
        except Exception as e:
            logger.error(f"Timestamp extraction failed for session {session_metadata.session_id}: {e}")
            
            # Create minimal timeline with error information
            error_timeline = SessionTimeline(
                session_id=session_metadata.session_id,
                session_date=session_metadata.date
            )
            error_timeline.validation_issues.append(f"Timestamp extraction failed: {str(e)}")
            error_timeline.extraction_metadata['extraction_error'] = str(e)
            
            return error_timeline
    
    async def extract_batch_timestamps(
        self,
        session_data: List[Tuple[str, List[RawSpeechSegment], SessionMetadata]],
        max_concurrent: int = 3
    ) -> List[SessionTimeline]:
        """
        Extract timestamps for multiple sessions concurrently
        
        Args:
            session_data: List of (content, speeches, session_metadata) tuples
            max_concurrent: Maximum concurrent extractions
            
        Returns:
            List of session timelines
        """
        logger.info(f"Starting batch timestamp extraction: {len(session_data)} sessions")
        
        # Create extraction tasks
        tasks = []
        for content, speeches, session_metadata in session_data:
            task = self.extract_comprehensive_timestamps(content, speeches, session_metadata)
            tasks.append(task)
        
        # Process in batches
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch timestamp extraction failed: {result}")
                    # Create error timeline
                    error_timeline = SessionTimeline(
                        session_id=f"error_{int(time.time())}",
                        session_date=datetime.now()
                    )
                    error_timeline.validation_issues.append(f"Batch extraction failed: {str(result)}")
                    results.append(error_timeline)
                else:
                    results.append(result)
        
        logger.info(f"Batch timestamp extraction completed: {len(results)} timelines")
        return results
    
    def _update_extraction_statistics(self, timeline: SessionTimeline, extraction_duration: float):
        """Update extraction statistics"""
        self.extraction_stats['sessions_processed'] += 1
        self.extraction_stats['timestamps_extracted'] += len(timeline.temporal_events)
        self.extraction_stats['speeches_timestamped'] += timeline.timestamped_speeches
        
        # Update average extraction time
        prev_avg = self.extraction_stats['average_extraction_time']
        count = self.extraction_stats['sessions_processed']
        new_avg = ((prev_avg * (count - 1)) + extraction_duration) / count
        self.extraction_stats['average_extraction_time'] = new_avg
        
        # Update average timeline completeness
        prev_completeness_avg = self.extraction_stats['timeline_completeness_avg']
        new_completeness_avg = ((prev_completeness_avg * (count - 1)) + timeline.timeline_completeness) / count
        self.extraction_stats['timeline_completeness_avg'] = new_completeness_avg
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics"""
        total_sessions = self.extraction_stats['sessions_processed']
        
        return {
            **self.extraction_stats,
            'timestamps_per_session_avg': (
                self.extraction_stats['timestamps_extracted'] / max(1, total_sessions)
            ),
            'timestamped_speeches_rate': (
                self.extraction_stats['speeches_timestamped'] / 
                max(1, self.extraction_stats['sessions_processed'])
            ),
            'performance_metrics': {
                'avg_extraction_time_ms': self.extraction_stats['average_extraction_time'],
                'avg_timeline_completeness': self.extraction_stats['timeline_completeness_avg']
            }
        }
    
    def generate_timeline_report(self, timeline: SessionTimeline) -> Dict[str, Any]:
        """Generate comprehensive timeline report"""
        return {
            'session_info': {
                'session_id': timeline.session_id,
                'session_date': timeline.session_date.isoformat(),
                'start_time': timeline.session_start_time.isoformat() if timeline.session_start_time else None,
                'end_time': timeline.session_end_time.isoformat() if timeline.session_end_time else None,
                'duration_minutes': timeline.total_duration_minutes
            },
            'timeline_quality': {
                'completeness': timeline.timeline_completeness,
                'timestamp_coverage': timeline.timestamp_coverage,
                'temporal_consistency': timeline.temporal_consistency,
                'overall_quality': timeline.extraction_metadata.get('timeline_quality_score', 0.0)
            },
            'speech_analysis': {
                'total_speeches': timeline.total_speeches,
                'timestamped_speeches': timeline.timestamped_speeches,
                'average_duration_seconds': timeline.average_speech_duration,
                'longest_speech_seconds': timeline.longest_speech_duration
            },
            'temporal_events': {
                'total_events': len(timeline.temporal_events),
                'event_types': list(set(event.event_type.value for event in timeline.temporal_events)),
                'break_time_minutes': timeline.total_break_time
            },
            'validation_results': {
                'issues_found': len(timeline.validation_issues),
                'validation_issues': timeline.validation_issues
            },
            'extraction_metadata': timeline.extraction_metadata
        }


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        extractor = AdvancedTimestampExtractor()
        
        # Example content with timestamps
        test_content = """
        The sitting opens at 9:00.
        
        President. – Good morning. The sitting is open at 9:01.
        
        García Pérez (PPE). – Madam President, at 9:05 I rise to speak about human rights.
        The situation requires immediate attention. This speech continues until 9:08.
        
        Commissioner Vestager. – Thank you. At 9:10, I would like to present our assessment.
        
        The vote begins at 9:15.
        
        President. – The sitting is closed at 10:00.
        """
        
        # Example speeches
        class MockSpeech:
            def __init__(self, seq, speaker, text, start_pos):
                self.sequence_number = seq
                self.speaker_raw = speaker
                self.speech_text = text
                self.start_position = start_pos
                self.segment_id = f"speech_{seq}"
        
        speeches = [
            MockSpeech(1, "President", "Good morning. The sitting is open.", 100),
            MockSpeech(2, "García Pérez (PPE)", "I rise to speak about human rights. The situation requires immediate attention.", 200),
            MockSpeech(3, "Commissioner Vestager", "Thank you. I would like to present our assessment.", 400),
        ]
        
        # Example session metadata
        class MockSession:
            def __init__(self):
                self.session_id = "test_session_001"
                self.date = datetime(2025, 1, 15)
        
        session_metadata = MockSession()
        
        # Extract timestamps
        timeline = await extractor.extract_comprehensive_timestamps(
            content=test_content,
            speeches=speeches,
            session_metadata=session_metadata
        )
        
        # Generate report
        report = extractor.generate_timeline_report(timeline)
        
        print("Timestamp Extraction Results:")
        print(json.dumps(report, indent=2, default=str))
        
        # Get statistics
        stats = extractor.get_extraction_statistics()
        print(f"\nExtraction Statistics:")
        print(json.dumps(stats, indent=2))
    
    asyncio.run(main())