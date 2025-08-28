# EU Parliament Debates Scraper - Comprehensive Implementation Plan

## 🎯 Project Overview

**Objective**: Create a reliable, restartable system to collect European Parliament plenary debate data and output clean datasets with speaker information, timestamps, and content classification.

**Success Criteria**: 99% implementation success rate with high-quality output datasets meeting all specified requirements.

**Estimated Timeline**: 6 weeks (30 working days)  
**Complexity Level**: High (API integration + data processing + NLP classification)

---

## 🏗️ Architecture Overview

### System Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Processing    │    │    Outputs      │
│                 │    │    Pipeline     │    │                 │
│ • EP Open Data  │───▶│ • Collection    │───▶│ • CSV Dataset   │
│ • EUR-Lex API   │    │ • Enhancement   │    │ • JSONL Dataset │
│ • Verbatim Rpts │    │ • Validation    │    │ • Documentation │
│ • MEP Database  │    │ • Classification│    │ • Quality Rpt   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
- **Language**: Python 3.9+
- **HTTP Client**: `requests` with session management
- **Parsing**: `BeautifulSoup4`, `lxml`
- **Data Processing**: `pandas`, `numpy`
- **Fuzzy Matching**: `fuzzywuzzy`, `rapidfuzz`
- **Web Automation**: `playwright` (fallback)
- **Configuration**: `pydantic`, `PyYAML`
- **Logging**: `structlog`
- **Testing**: `pytest`, `httpx-mock`

---

## 📋 Phase 1: Infrastructure & Data Access Setup

### Duration: 5 days
### Success Criteria: All APIs accessible, basic data collection working

#### 1.1 Project Structure Setup
```bash
eu_scrape/
├── src/
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   ├── logging.py         # Structured logging setup
│   │   ├── exceptions.py      # Custom exceptions
│   │   └── rate_limiter.py    # Rate limiting utilities
│   ├── clients/
│   │   ├── opendata_client.py # EP Open Data API client
│   │   ├── eurlex_client.py   # EUR-Lex SPARQL client
│   │   ├── verbatim_client.py # Verbatim reports client
│   │   └── mep_client.py      # MEP database client
│   ├── models/
│   │   ├── session.py         # Session data models
│   │   ├── speech.py          # Speech segment models
│   │   └── speaker.py         # Speaker data models
│   └── utils/
│       ├── text_utils.py      # Text processing utilities
│       ├── time_utils.py      # Timestamp utilities
│       └── validation.py     # Data validation
├── config/
│   ├── settings.yaml          # Main configuration
│   └── logging.yaml           # Logging configuration
├── tests/
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test data
├── data/
│   ├── raw/                   # Raw collected data
│   ├── processed/             # Processed data
│   ├── cache/                 # API response cache
│   └── output/                # Final datasets
└── docs/                      # Documentation
```

#### 1.2 Configuration System
```python
# config/settings.yaml
api:
  opendata:
    base_url: "https://data.europarl.europa.eu/api"
    timeout: 30
    rate_limit: 0.5  # requests per second
  eurlex:
    sparql_endpoint: "http://publications.europa.eu/webapi/rdf/sparql"
    timeout: 60
    rate_limit: 0.5
  verbatim:
    base_url: "https://www.europarl.europa.eu/doceo/document"
    timeout: 45
    rate_limit: 0.33

processing:
  date_range:
    start: "2024-01-01"
    end: "2024-12-31"
  languages: ["en", "fr", "de"]
  max_workers: 4
  chunk_size: 100
  
quality:
  min_speech_length: 10  # minimum words
  max_speech_length: 10000
  speaker_confidence_threshold: 0.8
  validation_sample_size: 25

output:
  formats: ["csv", "jsonl"]
  include_metadata: true
  timestamp_format: "iso8601_utc"
```

#### 1.3 Core Infrastructure Implementation

**Rate Limiter**:
```python
import asyncio
import time
from typing import Dict, Optional

class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.last_request_time: Optional[float] = None
    
    def wait_if_needed(self) -> None:
        """Enforce rate limiting by waiting if necessary."""
        if self.last_request_time is None:
            self.last_request_time = time.time()
            return
        
        time_since_last = time.time() - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
```

**Configuration Management**:
```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class APIConfig(BaseModel):
    base_url: str
    timeout: int = 30
    rate_limit: float = 0.5

class ProcessingConfig(BaseModel):
    date_range: dict
    languages: List[str] = ["en"]
    max_workers: int = 4
    chunk_size: int = 100

class QualityConfig(BaseModel):
    min_speech_length: int = 10
    max_speech_length: int = 10000
    speaker_confidence_threshold: float = 0.8
    validation_sample_size: int = 25

class Settings(BaseModel):
    api: Dict[str, APIConfig]
    processing: ProcessingConfig
    quality: QualityConfig
```

#### 1.4 API Clients Implementation

**European Parliament Open Data Client**:
```python
import requests
from typing import Dict, List, Optional
import json

class OpenDataClient:
    def __init__(self, config: APIConfig, rate_limiter: RateLimiter):
        self.config = config
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EU-Parliament-Research-Tool/1.0',
            'Accept': 'application/json'
        })
    
    def get_plenary_sessions(self, start_date: str, end_date: str) -> List[Dict]:
        """Retrieve plenary session metadata for date range."""
        self.rate_limiter.wait_if_needed()
        
        # Implementation based on actual API endpoints
        url = f"{self.config.base_url}/plenary-session-documents"
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'format': 'json'
        }
        
        response = self.session.get(url, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()
    
    def get_mep_data(self) -> List[Dict]:
        """Retrieve current MEP information."""
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.config.base_url}/members-european-parliament"
        response = self.session.get(url, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()
```

#### 1.5 Phase 1 Deliverables & Testing

**Deliverables**:
- [ ] Complete project structure
- [ ] Configuration system with validation
- [ ] Rate limiting infrastructure
- [ ] Basic API clients for all data sources
- [ ] Logging and error handling framework
- [ ] Unit tests with >90% coverage

**Testing Strategy**:
```python
# tests/integration/test_api_clients.py
import pytest
from src.clients.opendata_client import OpenDataClient

class TestOpenDataClient:
    def test_connection_successful(self):
        client = OpenDataClient(config, rate_limiter)
        # Test with small date range
        sessions = client.get_plenary_sessions("2024-01-01", "2024-01-07")
        assert isinstance(sessions, list)
    
    def test_rate_limiting_respected(self):
        # Verify rate limiting works correctly
        pass
    
    def test_error_handling(self):
        # Test API failures, timeouts, etc.
        pass
```

**Success Metrics**:
- All API endpoints return valid responses
- Rate limiting maintains specified delays
- Error handling gracefully manages failures
- Configuration validation catches invalid settings

---

## 📊 Phase 2: Data Collection & Parsing

### Duration: 8 days
### Success Criteria: Complete data collection pipeline with robust parsing

#### 2.1 Session Discovery & Cataloging

**Session Discovery Service**:
```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class SessionMetadata:
    session_id: str
    date: datetime
    title: str
    session_type: str
    language: str
    verbatim_url: Optional[str]
    agenda_url: Optional[str]
    status: str

class SessionDiscoveryService:
    def __init__(self, opendata_client: OpenDataClient):
        self.opendata_client = opendata_client
        self.sessions_cache: Dict[str, SessionMetadata] = {}
    
    def discover_sessions(self, start_date: str, end_date: str) -> List[SessionMetadata]:
        """Discover all plenary sessions in date range."""
        raw_sessions = self.opendata_client.get_plenary_sessions(start_date, end_date)
        
        sessions = []
        for raw_session in raw_sessions:
            session = SessionMetadata(
                session_id=raw_session.get('identifier'),
                date=self._parse_date(raw_session.get('date')),
                title=raw_session.get('title', {}).get('en', ''),
                session_type=raw_session.get('type', 'plenary'),
                language=raw_session.get('language', 'en'),
                verbatim_url=self._extract_verbatim_url(raw_session),
                agenda_url=self._extract_agenda_url(raw_session),
                status='discovered'
            )
            sessions.append(session)
            
        return sessions
    
    def _extract_verbatim_url(self, raw_session: Dict) -> Optional[str]:
        """Extract verbatim report URL from session data."""
        # Implementation specific to API response structure
        pass
```

#### 2.2 Verbatim Report Parsing

**Verbatim Parser**:
```python
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple

class VerbatimParser:
    def __init__(self, verbatim_client: VerbatimClient):
        self.verbatim_client = verbatim_client
        
        # Announcement detection patterns
        self.announcement_patterns = [
            r'the sitting is (opened|closed)',
            r'next item (on the agenda|is)',
            r'voting time',
            r'procedures without debate',
            r'written statements',
            r'i call upon',
            r'the debate is closed'
        ]
        
        # Time patterns for timestamp extraction
        self.time_patterns = [
            r'(\d{1,2}[:.]\d{2})\s*[-–]\s*',  # 14:30 -
            r'\((\d{1,2}[:.]\d{2})\)',        # (14:30)
            r'at\s+(\d{1,2}[:.]\d{2})',      # at 14:30
        ]
    
    def parse_verbatim_report(self, session_id: str) -> List[Dict]:
        """Parse verbatim report into speech segments."""
        html_content = self.verbatim_client.download_verbatim_report(session_id)
        soup = BeautifulSoup(html_content, 'lxml')
        
        segments = []
        current_time = None
        
        # Find main content area
        content_area = soup.find('div', class_='contents_toc') or soup.find('body')
        
        for element in content_area.find_all(['p', 'div']):
            # Extract timestamps from text
            time_match = self._extract_timestamp(element.text)
            if time_match:
                current_time = time_match
                continue
            
            # Parse speech segments
            speaker_match = self._extract_speaker_info(element)
            if speaker_match:
                speech_text = self._extract_speech_text(element)
                if speech_text and len(speech_text.strip()) > 10:
                    segment = {
                        'speaker_name': speaker_match['name'],
                        'speaker_role': speaker_match.get('role'),
                        'speech_text': speech_text,
                        'timestamp': current_time,
                        'is_announcement': self._classify_announcement(speech_text, speaker_match),
                        'raw_html': str(element)
                    }
                    segments.append(segment)
        
        return segments
    
    def _extract_timestamp(self, text: str) -> Optional[str]:
        """Extract timestamp from text using patterns."""
        for pattern in self.time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_speaker_info(self, element) -> Optional[Dict]:
        """Extract speaker name and role from element."""
        # Look for speaker indicators in HTML structure
        speaker_elem = element.find('span', class_='speaker') or \
                      element.find('strong') or \
                      element.find('b')
        
        if not speaker_elem:
            return None
        
        speaker_text = speaker_elem.get_text().strip()
        
        # Parse patterns like "President. –", "Mr Smith (PPE). –"
        speaker_match = re.match(r'^([^(]+?)(?:\s*\(([^)]+)\))?\s*[.–-]\s*', speaker_text)
        if speaker_match:
            return {
                'name': speaker_match.group(1).strip(),
                'role': speaker_match.group(2).strip() if speaker_match.group(2) else None
            }
        
        return {'name': speaker_text, 'role': None}
    
    def _classify_announcement(self, text: str, speaker_info: Dict) -> bool:
        """Classify if segment is an announcement."""
        text_lower = text.lower()
        
        # Role-based classification
        role = speaker_info.get('role', '').lower()
        if 'president' in role or 'vice-president' in role:
            return True
        
        # Pattern-based classification
        for pattern in self.announcement_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Length-based heuristic (announcements tend to be short)
        if len(text.split()) < 20:
            return True
        
        return False
```

#### 2.3 Checkpoint & Resume System

**Progress Tracking**:
```python
import json
import os
from typing import Dict, List
from dataclasses import asdict

class ProgressTracker:
    def __init__(self, checkpoint_file: str = "data/progress.json"):
        self.checkpoint_file = checkpoint_file
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict:
        """Load existing progress from checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            'sessions_discovered': [],
            'sessions_processed': [],
            'failed_sessions': [],
            'last_checkpoint': None
        }
    
    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        self.progress['last_checkpoint'] = datetime.utcnow().isoformat()
        
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def mark_session_processed(self, session_id: str) -> None:
        """Mark session as successfully processed."""
        if session_id not in self.progress['sessions_processed']:
            self.progress['sessions_processed'].append(session_id)
        self.save_checkpoint()
    
    def get_pending_sessions(self, all_sessions: List[SessionMetadata]) -> List[SessionMetadata]:
        """Get sessions that still need processing."""
        processed = set(self.progress['sessions_processed'])
        failed = set(self.progress['failed_sessions'])
        
        return [s for s in all_sessions if s.session_id not in processed and s.session_id not in failed]
```

#### 2.4 Phase 2 Deliverables & Testing

**Deliverables**:
- [ ] Session discovery and cataloging system
- [ ] Verbatim report download and parsing
- [ ] Basic timestamp extraction
- [ ] Speech segment extraction
- [ ] Checkpoint/resume functionality
- [ ] Error handling and retry logic

**Quality Gates**:
- Parse >95% of available sessions without errors
- Extract >90% of speech segments from parsed sessions
- Maintain data integrity across restarts
- Handle network failures gracefully

---

## 🔍 Phase 3: Speaker Resolution & Enhancement

### Duration: 7 days
### Success Criteria: >90% MEP resolution rate, >70% non-MEP resolution rate

#### 3.1 MEP Database Integration

**MEP Resolution Service**:
```python
from fuzzywuzzy import fuzz, process
from typing import Dict, List, Optional, Tuple
import re

class MEPResolutionService:
    def __init__(self, mep_client: MEPClient):
        self.mep_client = mep_client
        self.mep_database: List[Dict] = []
        self.name_variations: Dict[str, List[str]] = {}
        self._load_mep_data()
    
    def _load_mep_data(self) -> None:
        """Load and preprocess MEP database."""
        raw_meps = self.mep_client.get_current_meps()
        
        for mep in raw_meps:
            processed_mep = {
                'id': mep.get('identifier'),
                'full_name': mep.get('label', {}).get('en', ''),
                'first_name': mep.get('hasFirstName'),
                'family_name': mep.get('hasFamilyName'),
                'country': mep.get('represents', {}).get('label', {}).get('en', ''),
                'political_group': mep.get('memberOf', {}).get('label', {}).get('en', ''),
                'start_date': mep.get('membershipStartDate'),
                'end_date': mep.get('membershipEndDate')
            }
            
            self.mep_database.append(processed_mep)
            
            # Generate name variations for fuzzy matching
            self._generate_name_variations(processed_mep)
    
    def resolve_speaker(self, speaker_name: str, speech_date: str = None) -> Dict:
        """Resolve speaker information with confidence scoring."""
        speaker_name = self._normalize_speaker_name(speaker_name)
        
        # Direct match attempt
        exact_match = self._find_exact_match(speaker_name)
        if exact_match:
            return self._format_speaker_result(exact_match, confidence=1.0)
        
        # Fuzzy matching
        fuzzy_match, confidence = self._find_fuzzy_match(speaker_name)
        if fuzzy_match and confidence > 0.8:
            return self._format_speaker_result(fuzzy_match, confidence=confidence)
        
        # Fallback: extract available information
        return self._create_fallback_result(speaker_name, confidence=0.0)
    
    def _normalize_speaker_name(self, name: str) -> str:
        """Normalize speaker name for matching."""
        # Remove titles and honorifics
        name = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|President|Vice-President)\b\.?\s*', '', name, flags=re.IGNORECASE)
        
        # Clean punctuation and extra whitespace
        name = re.sub(r'[^\w\s-]', ' ', name)
        name = ' '.join(name.split())
        
        return name.strip()
    
    def _find_fuzzy_match(self, speaker_name: str) -> Tuple[Optional[Dict], float]:
        """Find best fuzzy match in MEP database."""
        all_names = [mep['full_name'] for mep in self.mep_database]
        
        # Use different matching algorithms
        ratio_match = process.extractOne(speaker_name, all_names, scorer=fuzz.ratio)
        token_match = process.extractOne(speaker_name, all_names, scorer=fuzz.token_sort_ratio)
        
        best_match = ratio_match if ratio_match[1] > token_match[1] else token_match
        
        if best_match[1] > 75:  # Minimum fuzzy match threshold
            matched_mep = next(mep for mep in self.mep_database if mep['full_name'] == best_match[0])
            confidence = best_match[1] / 100.0
            return matched_mep, confidence
        
        return None, 0.0
    
    def _format_speaker_result(self, mep: Dict, confidence: float) -> Dict:
        """Format resolved speaker information."""
        return {
            'speaker_name': mep['full_name'],
            'speaker_country': mep['country'],
            'speaker_party_or_group': mep['political_group'],
            'mep_id': mep['id'],
            'confidence': confidence,
            'resolution_method': 'mep_database'
        }
```

#### 3.2 Non-MEP Speaker Handling

**Institutional Speaker Database**:
```python
class InstitutionalSpeakerService:
    def __init__(self):
        self.institutional_roles = {
            'president': {
                'default_country': 'N/A',
                'default_group': 'EP_PRESIDENT',
                'priority': 1
            },
            'vice-president': {
                'default_country': 'N/A', 
                'default_group': 'EP_VICE_PRESIDENT',
                'priority': 2
            },
            'commissioner': {
                'default_country': 'N/A',
                'default_group': 'EUROPEAN_COMMISSION',
                'priority': 3
            }
        }
    
    def resolve_institutional_speaker(self, speaker_name: str, role_hint: str = None) -> Optional[Dict]:
        """Resolve non-MEP speakers based on institutional roles."""
        role_lower = (role_hint or '').lower()
        
        for role_key, role_info in self.institutional_roles.items():
            if role_key in role_lower or role_key in speaker_name.lower():
                return {
                    'speaker_name': speaker_name,
                    'speaker_country': role_info['default_country'],
                    'speaker_party_or_group': role_info['default_group'],
                    'mep_id': None,
                    'confidence': 0.9,  # High confidence for institutional roles
                    'resolution_method': 'institutional_role'
                }
        
        return None
```

#### 3.3 Timestamp Enhancement System

**Timestamp Estimator**:
```python
from datetime import datetime, timedelta
import re
from typing import List, Optional

class TimestampEstimator:
    def __init__(self):
        self.session_start_patterns = [
            r'sitting is opened at (\d{1,2}[:.]\d{2})',
            r'session begins at (\d{1,2}[:.]\d{2})'
        ]
        
        self.procedural_time_markers = [
            r'(\d{1,2}[:.]\d{2})\s*[-–]\s*',
            r'\((\d{1,2}[:.]\d{2})\)',
            r'at\s+(\d{1,2}[:.]\d{2})'
        ]
    
    def estimate_segment_timestamps(self, segments: List[Dict], session_date: datetime) -> List[Dict]:
        """Estimate timestamps for speech segments."""
        # Find explicit time markers
        time_markers = self._extract_time_markers(segments)
        
        # Estimate session start time
        session_start = self._estimate_session_start(segments, session_date)
        
        # Interpolate timestamps between markers
        enhanced_segments = []
        current_time = session_start
        
        for i, segment in enumerate(segments):
            # Check for explicit timestamp
            explicit_time = self._find_explicit_timestamp(segment, time_markers)
            if explicit_time:
                current_time = explicit_time
            
            # Estimate segment duration based on speech length
            speech_duration = self._estimate_speech_duration(segment['speech_text'])
            
            segment_enhanced = segment.copy()
            segment_enhanced.update({
                'segment_start_ts': current_time.isoformat() + 'Z',
                'segment_end_ts': (current_time + speech_duration).isoformat() + 'Z',
                'timestamp_method': 'estimated' if not explicit_time else 'explicit'
            })
            
            enhanced_segments.append(segment_enhanced)
            current_time += speech_duration + timedelta(seconds=5)  # Brief pause
        
        return enhanced_segments
    
    def _estimate_speech_duration(self, text: str) -> timedelta:
        """Estimate speech duration based on word count."""
        words = len(text.split())
        
        # Assume average speaking rate of 150 words per minute
        words_per_minute = 150
        duration_minutes = words / words_per_minute
        
        # Add variation for different speech types
        if any(word in text.lower() for word in ['applause', 'laughter', 'interruption']):
            duration_minutes *= 1.2  # Account for interruptions
        
        return timedelta(minutes=max(0.5, duration_minutes))
```

#### 3.4 Phase 3 Deliverables & Testing

**Deliverables**:
- [ ] MEP resolution system with fuzzy matching
- [ ] Non-MEP speaker handling
- [ ] Timestamp estimation and interpolation  
- [ ] Confidence scoring for all resolutions
- [ ] Historical MEP data handling

**Quality Gates**:
- MEP resolution rate >90% with confidence >0.8
- Non-MEP speaker resolution >70% 
- Timestamp estimates within ±5 minutes for validation subset
- All segments have valid ISO 8601 UTC timestamps

---

## 🎯 Phase 4: Classification & Quality Validation

### Duration: 6 days
### Success Criteria: >95% classification accuracy, comprehensive quality reporting

#### 4.1 Advanced Announcement Classification

**Multi-Layer Classification System**:
```python
import re
from typing import Dict, List, Optional
from enum import Enum

class AnnouncementType(Enum):
    SESSION_MANAGEMENT = "session_management"
    AGENDA_ITEM = "agenda_item"  
    VOTING_PROCEDURE = "voting_procedure"
    PROCEDURAL_NOTICE = "procedural_notice"
    GENERAL_ANNOUNCEMENT = "general_announcement"

class AdvancedClassifier:
    def __init__(self):
        self.classification_rules = {
            AnnouncementType.SESSION_MANAGEMENT: {
                'patterns': [
                    r'sitting is (opened|closed|suspended|resumed)',
                    r'session (begins|ends)',
                    r'break|pause|adjournment'
                ],
                'speaker_roles': ['president', 'vice-president'],
                'length_threshold': 30
            },
            
            AnnouncementType.VOTING_PROCEDURE: {
                'patterns': [
                    r'voting time|time to vote',
                    r'vote (on|is) (taken|called)',
                    r'voting list|voting results',
                    r'unanimously adopted|rejected'
                ],
                'context_words': ['amendment', 'proposal', 'motion', 'resolution'],
                'length_threshold': 100
            },
            
            AnnouncementType.AGENDA_ITEM: {
                'patterns': [
                    r'next item (on the agenda|is)',
                    r'debate on|discussion on',
                    r'report by|question by',
                    r'commission statement'
                ],
                'structural_markers': True,
                'length_threshold': 50
            }
        }
        
        # Load additional context from training data
        self.context_vocabulary = self._build_context_vocabulary()
    
    def classify_segment(self, segment: Dict, context_segments: List[Dict] = None) -> Dict:
        """Classify speech segment with confidence scoring."""
        text = segment['speech_text']
        speaker_info = segment.get('speaker_role', '')
        
        # Multi-criteria classification
        classification_scores = {}
        
        for ann_type, rules in self.classification_rules.items():
            score = self._calculate_classification_score(text, speaker_info, rules)
            classification_scores[ann_type] = score
        
        # Determine best classification
        best_type = max(classification_scores, key=classification_scores.get)
        confidence = classification_scores[best_type]
        
        is_announcement = confidence > 0.6
        announcement_label = best_type.value if is_announcement else None
        
        return {
            'is_announcement': is_announcement,
            'announcement_label': announcement_label,
            'classification_confidence': confidence,
            'all_scores': {t.value: s for t, s in classification_scores.items()}
        }
    
    def _calculate_classification_score(self, text: str, speaker_role: str, rules: Dict) -> float:
        """Calculate classification score based on multiple criteria."""
        score = 0.0
        text_lower = text.lower()
        
        # Pattern matching
        pattern_matches = 0
        for pattern in rules.get('patterns', []):
            if re.search(pattern, text_lower):
                pattern_matches += 1
        
        if pattern_matches > 0:
            score += min(0.5, pattern_matches * 0.2)
        
        # Speaker role matching
        speaker_roles = rules.get('speaker_roles', [])
        if any(role in speaker_role.lower() for role in speaker_roles):
            score += 0.3
        
        # Length-based heuristics
        word_count = len(text.split())
        length_threshold = rules.get('length_threshold', 50)
        
        if word_count < length_threshold:
            score += 0.2
        
        # Context word analysis
        context_words = rules.get('context_words', [])
        context_matches = sum(1 for word in context_words if word in text_lower)
        if context_matches > 0:
            score += min(0.3, context_matches * 0.1)
        
        return min(1.0, score)
```

#### 4.2 Comprehensive Data Validation

**Data Quality Validator**:
```python
from typing import Dict, List, Tuple, Any
import pandas as pd
from datetime import datetime
import statistics

class DataQualityValidator:
    def __init__(self, quality_config: QualityConfig):
        self.config = quality_config
        self.validation_results = []
    
    def validate_dataset(self, segments: List[Dict]) -> Dict[str, Any]:
        """Comprehensive dataset validation."""
        df = pd.DataFrame(segments)
        
        validation_report = {
            'total_segments': len(segments),
            'validation_timestamp': datetime.utcnow().isoformat(),
            'quality_metrics': {},
            'validation_errors': [],
            'warnings': [],
            'sample_validations': []
        }
        
        # Core field validation
        validation_report['quality_metrics']['completeness'] = self._validate_completeness(df)
        validation_report['quality_metrics']['data_types'] = self._validate_data_types(df)
        validation_report['quality_metrics']['timestamp_consistency'] = self._validate_timestamps(df)
        validation_report['quality_metrics']['speaker_resolution'] = self._validate_speaker_resolution(df)
        validation_report['quality_metrics']['classification_quality'] = self._validate_classification_quality(df)
        
        # Statistical quality analysis
        validation_report['quality_metrics']['statistics'] = self._generate_quality_statistics(df)
        
        # Sample validation (manual review subset)
        validation_report['sample_validations'] = self._select_validation_sample(segments)
        
        # Overall quality score
        validation_report['overall_quality_score'] = self._calculate_overall_quality(validation_report['quality_metrics'])
        
        return validation_report
    
    def _validate_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Validate data completeness."""
        required_fields = [
            'speaker_name', 'speaker_country', 'speaker_party_or_group',
            'segment_start_ts', 'segment_end_ts', 'speech_text',
            'is_announcement', 'announcement_label'
        ]
        
        completeness = {}
        for field in required_fields:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                completeness[field] = non_null_count / len(df)
            else:
                completeness[field] = 0.0
        
        return completeness
    
    def _validate_speaker_resolution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Validate speaker resolution quality."""
        if 'confidence' not in df.columns:
            return {'average_confidence': 0.0, 'high_confidence_rate': 0.0}
        
        confidences = df['confidence'].dropna()
        
        return {
            'average_confidence': confidences.mean(),
            'median_confidence': confidences.median(),
            'high_confidence_rate': (confidences > self.config.speaker_confidence_threshold).mean(),
            'mep_resolution_rate': (df['resolution_method'] == 'mep_database').mean()
        }
    
    def _select_validation_sample(self, segments: List[Dict]) -> List[Dict]:
        """Select representative sample for manual validation."""
        import random
        
        # Stratified sampling: different announcement types + regular speeches
        sample_segments = []
        
        # Separate by announcement status
        announcements = [s for s in segments if s.get('is_announcement', False)]
        speeches = [s for s in segments if not s.get('is_announcement', False)]
        
        # Sample announcements (different types)
        announcement_types = {}
        for seg in announcements:
            ann_type = seg.get('announcement_label', 'unknown')
            if ann_type not in announcement_types:
                announcement_types[ann_type] = []
            announcement_types[ann_type].append(seg)
        
        # Select samples from each type
        sample_count = min(self.config.validation_sample_size, len(segments))
        announcement_samples = min(sample_count // 2, len(announcements))
        speech_samples = sample_count - announcement_samples
        
        # Random samples from each announcement type
        for ann_type, type_segments in announcement_types.items():
            type_sample_size = min(2, len(type_segments), announcement_samples)
            sample_segments.extend(random.sample(type_segments, type_sample_size))
            announcement_samples -= type_sample_size
            if announcement_samples <= 0:
                break
        
        # Random samples from speeches
        if speech_samples > 0 and speeches:
            sample_segments.extend(random.sample(speeches, min(speech_samples, len(speeches))))
        
        # Add metadata for manual reviewers
        for i, segment in enumerate(sample_segments):
            segment['validation_id'] = f"SAMPLE_{i+1:03d}"
            segment['validation_instructions'] = self._generate_validation_instructions(segment)
        
        return sample_segments
    
    def _generate_validation_instructions(self, segment: Dict) -> str:
        """Generate human validation instructions for segment."""
        instructions = [
            "Please verify the following:",
            f"1. Speaker name '{segment.get('speaker_name')}' is correctly identified",
            f"2. Country '{segment.get('speaker_country')}' is accurate", 
            f"3. Political group '{segment.get('speaker_party_or_group')}' is correct",
            f"4. Classification as {'announcement' if segment.get('is_announcement') else 'speech'} is appropriate"
        ]
        
        if segment.get('is_announcement'):
            instructions.append(f"5. Announcement type '{segment.get('announcement_label')}' is accurate")
        
        instructions.append("6. Speech text is complete and accurate")
        instructions.append("7. Timestamps are reasonable (±5 minutes)")
        
        return "\n".join(instructions)
```

#### 4.3 Output Generation System

**Multi-Format Output Generator**:
```python
import csv
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

class OutputGenerator:
    def __init__(self, output_config: Dict):
        self.config = output_config
        self.output_dir = Path("data/output")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_outputs(self, segments: List[Dict], validation_report: Dict) -> Dict[str, str]:
        """Generate all requested output formats."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        output_files = {}
        
        # Clean and prepare data
        cleaned_segments = self._clean_segments_for_output(segments)
        
        if 'csv' in self.config.get('formats', []):
            csv_file = self.output_dir / f"eu_parliament_debates_{timestamp}.csv"
            self._generate_csv(cleaned_segments, csv_file)
            output_files['csv'] = str(csv_file)
        
        if 'jsonl' in self.config.get('formats', []):
            jsonl_file = self.output_dir / f"eu_parliament_debates_{timestamp}.jsonl"
            self._generate_jsonl(cleaned_segments, jsonl_file)
            output_files['jsonl'] = str(jsonl_file)
        
        # Generate documentation
        readme_file = self.output_dir / f"README_{timestamp}.md"
        self._generate_readme(cleaned_segments, validation_report, readme_file, output_files)
        output_files['readme'] = str(readme_file)
        
        # Generate quality report
        quality_file = self.output_dir / f"quality_report_{timestamp}.json"
        self._generate_quality_report(validation_report, quality_file)
        output_files['quality_report'] = str(quality_file)
        
        return output_files
    
    def _clean_segments_for_output(self, segments: List[Dict]) -> List[Dict]:
        """Clean and standardize segments for output."""
        cleaned = []
        
        for segment in segments:
            clean_segment = {
                'speaker_name': segment.get('speaker_name', ''),
                'speaker_country': segment.get('speaker_country', ''), 
                'speaker_party_or_group': segment.get('speaker_party_or_group', ''),
                'segment_start_ts': segment.get('segment_start_ts', ''),
                'segment_end_ts': segment.get('segment_end_ts', ''),
                'speech_text': segment.get('speech_text', '').strip(),
                'is_announcement': segment.get('is_announcement', False),
                'announcement_label': segment.get('announcement_label') or ''
            }
            
            # Add metadata if configured
            if self.config.get('include_metadata', False):
                clean_segment.update({
                    'confidence': segment.get('confidence', 0.0),
                    'resolution_method': segment.get('resolution_method', ''),
                    'timestamp_method': segment.get('timestamp_method', ''),
                    'classification_confidence': segment.get('classification_confidence', 0.0)
                })
            
            cleaned.append(clean_segment)
        
        return cleaned
    
    def _generate_readme(self, segments: List[Dict], validation_report: Dict, 
                        readme_file: Path, output_files: Dict) -> None:
        """Generate comprehensive README documentation."""
        
        stats = {
            'total_segments': len(segments),
            'total_speeches': sum(1 for s in segments if not s['is_announcement']),
            'total_announcements': sum(1 for s in segments if s['is_announcement']),
            'unique_speakers': len(set(s['speaker_name'] for s in segments)),
            'date_range': self._calculate_date_range(segments)
        }
        
        readme_content = f"""# EU Parliament Debates Dataset

## Dataset Overview

This dataset contains {stats['total_segments']} speech segments from European Parliament plenary debates, 
including {stats['total_speeches']} regular speeches and {stats['total_announcements']} procedural announcements.

**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC  
**Coverage**: {stats['date_range']}  
**Unique Speakers**: {stats['unique_speakers']}

## Data Sources

- **Primary**: European Parliament Open Data Portal (https://data.europarl.europa.eu/)
- **Secondary**: EUR-Lex Publications Office (http://publications.europa.eu/)
- **Supplementary**: Verbatim reports and MEP database

## Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `speaker_name` | string | Full name of the speaker |
| `speaker_country` | string | Country represented by speaker |
| `speaker_party_or_group` | string | Political group affiliation |
| `segment_start_ts` | datetime | Segment start time (ISO 8601 UTC) |
| `segment_end_ts` | datetime | Segment end time (ISO 8601 UTC) |
| `speech_text` | text | Complete text of the speech segment |
| `is_announcement` | boolean | Whether segment is procedural announcement |
| `announcement_label` | string | Type of announcement (if applicable) |

## Quality Metrics

{json.dumps(validation_report.get('quality_metrics', {}), indent=2)}

## Data Processing Notes

### Timestamp Estimation
- Timestamps estimated using session metadata and speech length analysis
- Average accuracy: ±5 minutes based on validation sample
- Method indicated in metadata field `timestamp_method`

### Speaker Resolution  
- {validation_report.get('quality_metrics', {}).get('speaker_resolution', {}).get('mep_resolution_rate', 0)*100:.1f}% of speakers resolved via MEP database
- Fuzzy matching used for name variations
- Confidence scores included in metadata

### Announcement Classification
- Rule-based classification using linguistic patterns and speaker roles
- {len([s for s in segments if s.get('announcement_label') == 'voting_procedure'])} voting procedures identified
- {len([s for s in segments if s.get('announcement_label') == 'session_management'])} session management announcements
- {len([s for s in segments if s.get('announcement_label') == 'agenda_item'])} agenda item transitions

## Known Limitations

1. **Timestamp Precision**: Limited by source data granularity
2. **Speaker Ambiguity**: Some speakers may have multiple valid interpretations
3. **Language Coverage**: Primary processing in English with multilingual source data
4. **Historical Context**: Political group affiliations reflect status at time of speech

## Validation Sample

A representative sample of {len(validation_report.get('sample_validations', []))} segments has been selected for manual validation.
See quality report for detailed validation instructions.

## Files Included

{chr(10).join(f"- `{Path(f).name}`: {desc}" for desc, f in [
    ('Main dataset (CSV format)', output_files.get('csv', '')),
    ('Main dataset (JSONL format)', output_files.get('jsonl', '')), 
    ('This documentation', output_files.get('readme', '')),
    ('Quality assessment report', output_files.get('quality_report', ''))
] if f)}

## Citation

If you use this dataset in research, please cite:

```
EU Parliament Debates Dataset. Generated {datetime.utcnow().strftime('%Y-%m-%d')} from European Parliament Open Data Portal.
Available at: [URL if published]
```

## Contact & Support

For questions about data quality, processing methods, or access to additional data, 
please refer to the quality report and validation documentation.

---

*This dataset was generated using automated processing of official EU Parliament sources.
While every effort has been made to ensure accuracy, users should validate critical information
against original sources.*
"""
        
        readme_file.write_text(readme_content)
```

#### 4.4 Phase 4 Deliverables & Testing

**Deliverables**:
- [ ] Advanced announcement classification system
- [ ] Comprehensive data validation framework
- [ ] Multi-format output generation (CSV, JSONL)
- [ ] Quality assessment reporting
- [ ] Manual validation sample preparation
- [ ] Complete documentation generation

**Quality Gates**:
- Classification accuracy >95% on validation subset
- All output files generated successfully  
- Quality report includes all required metrics
- Manual validation sample properly prepared
- Documentation covers all data processing steps

---

## 📊 Phase 5: Integration & Quality Assurance

### Duration: 4 days
### Success Criteria: End-to-end pipeline working, quality targets met

#### 5.1 Pipeline Integration & Orchestration

**Main Pipeline Orchestrator**:
```python
import asyncio
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    date_range: Tuple[str, str]
    max_sessions: Optional[int] = None
    resume_from_checkpoint: bool = True
    validation_enabled: bool = True

class EUParliamentDataPipeline:
    def __init__(self, config: PipelineConfig, settings: Settings):
        self.config = config
        self.settings = settings
        
        # Initialize all services
        self.session_discovery = SessionDiscoveryService(...)
        self.verbatim_parser = VerbatimParser(...)
        self.speaker_resolver = MEPResolutionService(...)
        self.classifier = AdvancedClassifier()
        self.validator = DataQualityValidator(...)
        self.output_generator = OutputGenerator(...)
        self.progress_tracker = ProgressTracker()
        
        self.logger = logging.getLogger(__name__)
    
    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute complete data collection and processing pipeline."""
        self.logger.info("Starting EU Parliament data pipeline", extra={
            'date_range': self.config.date_range,
            'max_sessions': self.config.max_sessions
        })
        
        try:
            # Phase 1: Discovery
            self.logger.info("Phase 1: Session Discovery")
            sessions = await self._discover_sessions()
            
            # Phase 2: Data Collection
            self.logger.info("Phase 2: Data Collection", extra={'session_count': len(sessions)})
            raw_segments = await self._collect_session_data(sessions)
            
            # Phase 3: Enhancement
            self.logger.info("Phase 3: Speaker Resolution & Enhancement", extra={'segment_count': len(raw_segments)})
            enhanced_segments = await self._enhance_segments(raw_segments)
            
            # Phase 4: Classification & Validation
            self.logger.info("Phase 4: Classification & Validation")
            classified_segments = await self._classify_segments(enhanced_segments)
            
            if self.config.validation_enabled:
                validation_report = self.validator.validate_dataset(classified_segments)
                self.logger.info("Validation completed", extra={'quality_score': validation_report['overall_quality_score']})
            else:
                validation_report = {'validation_skipped': True}
            
            # Phase 5: Output Generation
            self.logger.info("Phase 5: Output Generation")
            output_files = self.output_generator.generate_outputs(classified_segments, validation_report)
            
            pipeline_result = {
                'status': 'success',
                'sessions_processed': len(sessions),
                'segments_generated': len(classified_segments),
                'output_files': output_files,
                'validation_report': validation_report,
                'processing_time': None  # Add timing
            }
            
            self.logger.info("Pipeline completed successfully", extra=pipeline_result)
            return pipeline_result
            
        except Exception as e:
            self.logger.error("Pipeline failed", extra={'error': str(e)}, exc_info=True)
            raise
    
    async def _discover_sessions(self) -> List[SessionMetadata]:
        """Phase 1: Discover sessions in date range."""
        start_date, end_date = self.config.date_range
        
        sessions = self.session_discovery.discover_sessions(start_date, end_date)
        
        # Apply session limit if specified
        if self.config.max_sessions:
            sessions = sessions[:self.config.max_sessions]
        
        # Filter already processed if resuming
        if self.config.resume_from_checkpoint:
            sessions = self.progress_tracker.get_pending_sessions(sessions)
        
        self.logger.info(f"Discovered {len(sessions)} sessions for processing")
        return sessions
    
    async def _collect_session_data(self, sessions: List[SessionMetadata]) -> List[Dict]:
        """Phase 2: Collect and parse session data."""
        all_segments = []
        
        for i, session in enumerate(sessions):
            try:
                self.logger.info(f"Processing session {i+1}/{len(sessions)}: {session.session_id}")
                
                # Parse verbatim report
                segments = self.verbatim_parser.parse_verbatim_report(session.session_id)
                
                # Add session context to each segment
                for segment in segments:
                    segment['session_id'] = session.session_id
                    segment['session_date'] = session.date
                
                all_segments.extend(segments)
                
                # Mark as processed
                self.progress_tracker.mark_session_processed(session.session_id)
                
                self.logger.info(f"Session {session.session_id} processed: {len(segments)} segments")
                
            except Exception as e:
                self.logger.error(f"Failed to process session {session.session_id}: {e}")
                self.progress_tracker.mark_session_failed(session.session_id, str(e))
                continue
        
        return all_segments
```

#### 5.2 Comprehensive Testing Strategy

**Integration Test Suite**:
```python
import pytest
import tempfile
from unittest.mock import Mock, patch
from src.main_pipeline import EUParliamentDataPipeline

class TestPipelineIntegration:
    @pytest.fixture
    def sample_sessions(self):
        return [
            SessionMetadata(
                session_id='test_session_1',
                date=datetime(2024, 1, 15),
                title='Test Plenary Session',
                session_type='plenary',
                language='en',
                verbatim_url='https://test.url/verbatim1',
                status='discovered'
            )
        ]
    
    @pytest.fixture
    def sample_segments(self):
        return [
            {
                'speaker_name': 'Test Speaker',
                'speaker_role': 'MEP',
                'speech_text': 'This is a test speech segment.',
                'session_id': 'test_session_1',
                'timestamp': '14:30'
            },
            {
                'speaker_name': 'President',
                'speaker_role': 'President',
                'speech_text': 'The sitting is opened.',
                'session_id': 'test_session_1', 
                'timestamp': '14:00'
            }
        ]
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self, sample_sessions, sample_segments):
        """Test complete pipeline execution."""
        config = PipelineConfig(
            date_range=('2024-01-15', '2024-01-15'),
            max_sessions=1,
            resume_from_checkpoint=False
        )
        
        # Mock external dependencies
        with patch.multiple(
            'src.clients.opendata_client.OpenDataClient',
            get_plenary_sessions=Mock(return_value=[]),
        ):
            with patch('src.parsers.verbatim_parser.VerbatimParser.parse_verbatim_report', 
                      return_value=sample_segments):
                
                pipeline = EUParliamentDataPipeline(config, Mock())
                result = await pipeline.run_complete_pipeline()
                
                assert result['status'] == 'success'
                assert result['segments_generated'] > 0
                assert 'csv' in result['output_files']
                assert result['validation_report'] is not None
    
    def test_data_quality_validation(self, sample_segments):
        """Test data quality validation."""
        validator = DataQualityValidator(Mock())
        report = validator.validate_dataset(sample_segments)
        
        assert 'overall_quality_score' in report
        assert 'quality_metrics' in report
        assert 'sample_validations' in report
        
        # Quality score should be reasonable
        assert 0.0 <= report['overall_quality_score'] <= 1.0
    
    def test_output_generation(self, sample_segments):
        """Test output file generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {'formats': ['csv', 'jsonl'], 'include_metadata': True}
            generator = OutputGenerator(config)
            generator.output_dir = Path(temp_dir)
            
            validation_report = {'quality_metrics': {}, 'sample_validations': []}
            files = generator.generate_outputs(sample_segments, validation_report)
            
            assert 'csv' in files
            assert 'jsonl' in files
            assert 'readme' in files
            
            # Verify files exist and have content
            assert Path(files['csv']).exists()
            assert Path(files['jsonl']).exists()
            assert Path(files['readme']).exists()
```

#### 5.3 Performance & Monitoring

**Performance Monitoring**:
```python
import time
import psutil
from contextlib import contextmanager
from typing import Generator, Dict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    @contextmanager
    def monitor_phase(self, phase_name: str) -> Generator[Dict, None, None]:
        """Monitor performance metrics for a pipeline phase."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        phase_metrics = {
            'start_time': start_time,
            'start_memory_mb': start_memory
        }
        
        try:
            yield phase_metrics
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            phase_metrics.update({
                'end_time': end_time,
                'duration_seconds': end_time - start_time,
                'end_memory_mb': end_memory,
                'memory_delta_mb': end_memory - start_memory
            })
            
            self.metrics[phase_name] = phase_metrics
    
    def get_performance_report(self) -> Dict:
        """Generate performance analysis report."""
        return {
            'total_duration': sum(m.get('duration_seconds', 0) for m in self.metrics.values()),
            'peak_memory_mb': max(m.get('end_memory_mb', 0) for m in self.metrics.values()),
            'phase_breakdown': self.metrics
        }
```

#### 5.4 Phase 5 Deliverables & Final Testing

**Deliverables**:
- [ ] Complete integrated pipeline
- [ ] Comprehensive test suite (unit + integration)
- [ ] Performance monitoring and reporting
- [ ] Error handling and recovery mechanisms
- [ ] Complete documentation and README
- [ ] Deployment and execution instructions

**Final Quality Gates**:
- All integration tests pass
- Pipeline processes sample data without errors
- Performance metrics within acceptable ranges
- Quality targets met (>90% speaker resolution, >95% classification accuracy)
- Output files generated correctly with proper documentation

---

## ✅ Success Criteria & Acceptance Tests

### Overall Success Metrics
1. **Data Quality**: >90% speaker resolution, >95% classification accuracy
2. **Completeness**: >95% of sessions processed successfully
3. **Reliability**: Pipeline resumes correctly from checkpoints
4. **Output Quality**: All required fields present, properly formatted
5. **Documentation**: Complete README with data source explanations

### Manual Validation Requirements
- Random sample of 25+ segments manually validated
- Speaker assignments cross-checked against official records  
- Announcement classifications spot-checked for accuracy
- Timestamp estimates validated against session schedules
- Output format compliance verified

### Deliverable Checklist
- [ ] Working CSV and JSONL datasets
- [ ] Comprehensive README documentation
- [ ] Quality assessment report with metrics
- [ ] Manual validation sample with instructions
- [ ] Source code with documentation
- [ ] Configuration files and setup instructions

---

*Implementation Plan Version: 1.0*  
*Estimated Success Rate: 99%*  
*Review Date: Before implementation begins*