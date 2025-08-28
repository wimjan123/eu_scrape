"""Speech segment data models for EU Parliament scraper."""

import re
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class AnnouncementType(str, Enum):
    """Types of procedural announcements."""
    SESSION_MANAGEMENT = "session_management"
    AGENDA_ITEM = "agenda_item"
    VOTING_PROCEDURE = "voting_procedure"
    PROCEDURAL_NOTICE = "procedural_notice"
    GENERAL_ANNOUNCEMENT = "general_announcement"


class SpeechSegment(BaseModel):
    """Complete speech segment with all required fields."""
    speaker_name: str = Field(..., min_length=1)
    speaker_country: str = Field(..., min_length=1)
    speaker_party_or_group: str = Field(..., min_length=1)
    segment_start_ts: str = Field(..., description="ISO 8601 UTC timestamp")
    segment_end_ts: str = Field(..., description="ISO 8601 UTC timestamp")
    speech_text: str = Field(..., min_length=10)
    is_announcement: bool = Field(default=False)
    announcement_label: str = Field(default="")
    
    # Metadata fields (optional, for quality tracking)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    resolution_method: Optional[str] = None
    timestamp_method: Optional[str] = None
    classification_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    @validator('segment_start_ts', 'segment_end_ts')
    def validate_iso8601_format(cls, v):
        """Ensure timestamps are in ISO 8601 format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError(f'Invalid ISO 8601 timestamp: {v}')
        return v
    
    @validator('announcement_label')
    def validate_announcement_label(cls, v, values):
        """Ensure announcement_label is set when is_announcement is True."""
        is_announcement = values.get('is_announcement', False)
        if is_announcement and not v:
            raise ValueError('announcement_label must be set when is_announcement is True')
        if not is_announcement and v:
            # Clear label if not an announcement
            return ""
        return v


class RawSpeechSegment(BaseModel):
    """Raw speech segment extracted from verbatim reports."""
    session_id: str = Field(..., description="Session identifier")
    sequence_number: int = Field(..., ge=1, description="Sequential position in session")
    speaker_raw: str = Field(..., description="Raw speaker name from document")
    speech_text: str = Field(..., min_length=10, description="Raw speech content")
    timestamp_hint: Optional[str] = Field(None, description="Timestamp hint from document")
    is_procedural: bool = Field(default=False, description="Whether segment is procedural")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Parsing confidence")
    parsing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Parser metadata")
    
    # Optional enhancement fields for backward compatibility
    speaker_role: Optional[str] = Field(None, description="Extracted speaker role/party")
    session_date: Optional[datetime] = Field(None, description="Session date for context")
    raw_html: Optional[str] = Field(None, description="Original HTML element")
    
    @validator('timestamp_hint')
    def validate_timestamp_hint(cls, v):
        """Basic timestamp format validation."""
        if v is None:
            return v
        # Accept common time formats: HH:MM, H:MM, HH.MM
        if not re.match(r'^\d{1,2}[:.]\d{2}$', v):
            raise ValueError(f'Invalid timestamp format: {v}')
        return v