"""Speaker data models for EU Parliament scraper."""

from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from enum import Enum


class SpeakerType(str, Enum):
    """Types of speakers in Parliament."""
    MEP = "mep"
    INSTITUTIONAL = "institutional"
    GUEST = "guest"
    UNKNOWN = "unknown"


class MEPData(BaseModel):
    """Member of European Parliament data."""
    mep_id: str
    full_name: str
    first_name: str
    family_name: str
    country: str
    political_group: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    @property
    def is_current_member(self) -> bool:
        """Check if MEP is currently serving."""
        if self.end_date is None:
            return True
        try:
            end_date = datetime.fromisoformat(self.end_date)
            return datetime.now() < end_date
        except (ValueError, TypeError):
            return True


class InstitutionalRole(BaseModel):
    """Institutional speaker role information."""
    role_title: str
    institution: str  # EP_PRESIDENT, EUROPEAN_COMMISSION, etc.
    country: str = "N/A"
    priority: int = Field(default=5, ge=1, le=10)


class SpeakerResolution(BaseModel):
    """Result of speaker resolution process."""
    speaker_name: str
    speaker_country: str
    speaker_party_or_group: str
    speaker_type: SpeakerType
    
    # Resolution metadata
    mep_id: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    resolution_method: str
    alternative_matches: Optional[List[Dict]] = None
    
    # Quality indicators
    name_similarity_score: Optional[float] = None
    context_validation_passed: bool = True
    manual_review_required: bool = False


class SpeakerDatabase(BaseModel):
    """Database of known speakers."""
    meps: List[MEPData] = Field(default_factory=list)
    institutional_roles: Dict[str, InstitutionalRole] = Field(default_factory=dict)
    
    last_updated: Optional[datetime] = None
    version: str = "1.0"
    
    def get_mep_by_name(self, name: str) -> Optional[MEPData]:
        """Find MEP by name."""
        name_lower = name.lower()
        for mep in self.meps:
            if name_lower in mep.full_name.lower():
                return mep
        return None
    
    def get_institutional_role(self, role_hint: str) -> Optional[InstitutionalRole]:
        """Find institutional role by hint."""
        role_hint_lower = role_hint.lower()
        for role_key, role_data in self.institutional_roles.items():
            if role_key in role_hint_lower:
                return role_data
        return None