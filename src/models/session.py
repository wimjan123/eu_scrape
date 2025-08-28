"""Session data models for EU Parliament scraper."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


@dataclass
class SessionMetadata:
    """Metadata for a plenary session."""
    session_id: str
    date: datetime
    title: str
    session_type: str
    language: str
    verbatim_url: Optional[str]
    agenda_url: Optional[str]
    status: str


class SessionConfig(BaseModel):
    """Configuration for session processing."""
    session_id: str
    processing_priority: int = Field(default=1, ge=1, le=10)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=1)
    
    @property
    def should_retry(self) -> bool:
        """Check if session processing should be retried."""
        return self.retry_count < self.max_retries