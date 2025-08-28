# EU Parliament Debates Scraper - Implementation Prompt

## 🎯 Your Mission

You are an expert implementation AI tasked with building a comprehensive European Parliament plenary debates data collection system. Your goal is to create a reliable, production-ready system that collects debate data and outputs clean datasets with 99% accuracy.

**Critical Success Factors**: Follow the comprehensive implementation plan exactly, implement all error handling and quality checks, and ensure the system can resume from failures gracefully.

---

## 📋 Pre-Implementation Setup

### 1. Review All Documentation
**REQUIRED**: Read and understand these files before starting:
- `./planner/research/eu_parliament_data_sources.md` - Data source analysis
- `./planner/plan/comprehensive_implementation_plan.md` - Complete implementation plan
- `./planner/checklist/progress_checklist.md` - Phase-based progress tracking

### 2. Validate Environment
```bash
# Ensure Python 3.9+ is available
python3 --version

# Create virtual environment
python3 -m venv eu_scrape_env
source eu_scrape_env/bin/activate

# Install required packages (you'll define these)
pip install requests beautifulsoup4 pandas pydantic playwright pytest
```

### 3. Confirm API Access
**CRITICAL**: Test connectivity to all data sources before proceeding:
- European Parliament Open Data Portal: https://data.europarl.europa.eu/
- EUR-Lex SPARQL Endpoint: http://publications.europa.eu/webapi/rdf/sparql
- Verbatim Reports: https://www.europarl.europa.eu/doceo/document/
- MEP Database: Accessible via Open Data Portal

---

## 🏗️ Implementation Strategy

### Phase-Based Approach
Execute the implementation in **exactly 5 phases** as specified in the comprehensive plan:

1. **Phase 1 (5 days)**: Infrastructure & Data Access Setup
2. **Phase 2 (8 days)**: Data Collection & Parsing  
3. **Phase 3 (7 days)**: Speaker Resolution & Enhancement
4. **Phase 4 (6 days)**: Classification & Quality Validation
5. **Phase 5 (4 days)**: Integration & Quality Assurance

### Daily Progress Reporting
**MANDATORY**: At the end of each development day, update the progress checklist with:
- Completed tasks and deliverables
- Any deviations from the plan
- Issues encountered and solutions implemented
- Quality metrics achieved
- Next day's priorities

---

## 🎯 Target Output Schema

**PRIMARY OBJECTIVE**: Generate datasets with these exact fields:

```python
@dataclass
class SpeechSegment:
    speaker_name: str              # Full name of the speaker
    speaker_country: str           # Country representation
    speaker_party_or_group: str    # Political group affiliation  
    segment_start_ts: str          # ISO 8601 UTC timestamp
    segment_end_ts: str            # ISO 8601 UTC timestamp
    speech_text: str               # Complete speech content
    is_announcement: bool          # True for procedural announcements
    announcement_label: str        # Type if is_announcement=True
```

### Quality Requirements
- **Speaker Resolution**: >90% MEP resolution rate, >70% non-MEP resolution rate
- **Timestamp Accuracy**: ±5 minutes based on manual validation
- **Classification Accuracy**: >95% for announcement vs speech classification
- **Data Completeness**: >98% of segments have all required fields
- **Manual Validation**: 25+ segments manually validated as specified

---

## 🛠️ Technical Implementation Guidelines

### Technology Stack (REQUIRED)
```python
# Core Dependencies
requests>=2.28.0           # HTTP client with session management
beautifulsoup4>=4.11.0     # HTML/XML parsing
lxml>=4.9.0               # Fast XML processing
pandas>=1.5.0             # Data manipulation
numpy>=1.24.0             # Numerical operations
pydantic>=1.10.0          # Data validation and settings
PyYAML>=6.0               # Configuration management

# Fuzzy Matching & NLP
fuzzywuzzy>=0.18.0        # Fuzzy string matching
rapidfuzz>=2.13.0         # Fast fuzzy matching
python-Levenshtein>=0.20.0 # String distance

# Web Automation (Fallback)
playwright>=1.30.0         # Browser automation

# Testing & Quality
pytest>=7.2.0             # Testing framework
httpx>=0.24.0             # Async HTTP for testing
pytest-asyncio>=0.21.0    # Async testing support

# Logging & Monitoring
structlog>=22.3.0         # Structured logging
psutil>=5.9.0             # System monitoring
```

### Architecture Requirements
```
src/
├── core/                  # Core utilities and configuration
├── clients/               # API clients for each data source
├── models/                # Pydantic data models
├── parsers/               # Content parsing and extraction
├── services/              # Business logic services
├── utils/                 # Utility functions
└── main_pipeline.py       # Main orchestration
```

### Rate Limiting (CRITICAL)
**MANDATORY**: Implement conservative rate limiting to respect EU servers:
- Open Data Portal: 0.5 requests/second (2-second delays)
- EUR-Lex SPARQL: 0.5 requests/second (2-second delays)  
- Verbatim Reports: 0.33 requests/second (3-second delays)
- **Include exponential backoff on failures**
- **Log all requests for monitoring**

---

## 🔍 Data Processing Requirements

### 1. Session Discovery
```python
# Target: Discover plenary sessions in date range
date_range = ("2024-01-01", "2024-12-31")  # Start with recent months
min_sessions_for_poc = 50                   # Proof of concept threshold

# Must handle:
# - API pagination
# - Date filtering
# - Session type filtering (plenary only)
# - Metadata validation
```

### 2. Content Extraction
```python
# Target: Extract speech segments from verbatim reports
# Must implement:
# - HTML/XML parsing with BeautifulSoup
# - Speaker name extraction with role detection
# - Speech text cleaning and validation
# - Timestamp extraction using regex patterns
# - Checkpoint/resume functionality
```

### 3. Speaker Resolution
```python
# Target: Resolve speaker information
# Must implement:
# - MEP database synchronization
# - Fuzzy name matching (fuzzywuzzy)
# - Confidence scoring (0.0-1.0)
# - Non-MEP speaker handling (institutional roles)
# - Historical affiliation tracking
```

### 4. Announcement Classification
```python
# Target: Classify announcements vs speeches
# Must implement patterns for:
announcement_patterns = [
    'sitting is (opened|closed|suspended)',
    'next item (on the agenda|is)',
    'voting time|time to vote',
    'procedures without debate',
    'written statements',
    'i call upon'
]

# Classification types:
# - session_management
# - agenda_item
# - voting_procedure  
# - procedural_notice
# - general_announcement
```

---

## 📊 Quality Assurance Requirements

### Automated Validation (REQUIRED)
```python
# Must implement these validation checks:

def validate_segment(segment: Dict) -> Dict[str, Any]:
    validations = {
        'has_speaker_name': bool(segment.get('speaker_name', '').strip()),
        'has_speech_text': len(segment.get('speech_text', '')) > 10,
        'valid_timestamps': _validate_iso8601_timestamps(segment),
        'reasonable_duration': _validate_speech_duration(segment),
        'valid_country': segment.get('speaker_country') in VALID_COUNTRIES,
        'classification_consistent': _validate_classification_logic(segment)
    }
    
    return {
        'segment_id': segment.get('id'),
        'validations': validations,
        'overall_valid': all(validations.values()),
        'validation_score': sum(validations.values()) / len(validations)
    }
```

### Manual Validation Sample (CRITICAL)
**REQUIRED**: Generate a stratified sample for manual validation:
```python
# Must create validation sample with:
# - 25+ segments total
# - Representative mix of announcements and speeches
# - Different speaker types (MEPs, institutional)
# - Various political groups and countries
# - Clear validation instructions for each segment
# - Validation checklist for human reviewers
```

### Quality Metrics Dashboard
```python
# Must track and report:
quality_metrics = {
    'completeness': {
        'total_sessions_available': 0,
        'sessions_processed': 0,
        'segments_extracted': 0,
        'segments_with_all_fields': 0
    },
    'accuracy': {
        'speaker_resolution_rate': 0.0,
        'mep_resolution_rate': 0.0, 
        'classification_accuracy': 0.0,
        'timestamp_accuracy': 0.0
    },
    'confidence': {
        'avg_speaker_confidence': 0.0,
        'avg_classification_confidence': 0.0,
        'high_confidence_segments_pct': 0.0
    }
}
```

---

## 🔄 Error Handling & Resilience

### Checkpoint System (CRITICAL)
```python
# Must implement comprehensive checkpointing:
checkpoint_data = {
    'last_checkpoint': datetime.utcnow().isoformat(),
    'sessions_discovered': [...],
    'sessions_processed': [...],
    'sessions_failed': [...],
    'processing_phase': 'data_collection',
    'progress_metrics': {...}
}

# Save checkpoints after:
# - Each session processed
# - Each phase completed
# - Every 30 minutes during long operations
# - Before any risky operation
```

### Failure Recovery
```python
# Must handle these failure scenarios:
# - Network timeouts and connection errors
# - API rate limiting and temporary blocks
# - Malformed HTML/XML in source documents
# - Missing speaker information
# - Parsing errors in verbatim reports
# - Disk space and memory limitations

# Recovery strategies:
# - Exponential backoff with jitter
# - Graceful degradation (partial data collection)
# - Alternative parsing strategies
# - Clear error reporting and logging
```

---

## 📤 Output Generation Requirements

### File Formats (REQUIRED)
**Must generate both CSV and JSONL formats**:

```python
# CSV Format
# - Standard CSV with proper quoting
# - UTF-8 encoding
# - Header row with exact field names
# - Escape special characters in text fields

# JSONL Format  
# - One JSON object per line
# - UTF-8 encoding
# - All fields as strings except booleans
# - Proper JSON escaping
```

### Documentation Package
**REQUIRED**: Generate comprehensive documentation:
```markdown
# Must include:
README.md                 # Dataset overview and usage
quality_report.json       # Detailed quality metrics
data_dictionary.md        # Field definitions and examples
processing_log.txt        # Processing notes and limitations
validation_sample.csv     # Manual validation subset
```

---

## 🚦 Implementation Checkpoints

### Phase Completion Criteria
**Each phase must meet ALL criteria before proceeding**:

#### Phase 1: Infrastructure
- [ ] All API clients implemented and tested
- [ ] Rate limiting working correctly
- [ ] Configuration management functional
- [ ] Basic error handling implemented
- [ ] Project structure follows specification

#### Phase 2: Data Collection  
- [ ] Session discovery working for date range
- [ ] Verbatim report parsing extracting segments
- [ ] Basic timestamp extraction functional
- [ ] Checkpoint/resume system implemented
- [ ] >95% of available sessions processed

#### Phase 3: Speaker Resolution
- [ ] MEP database integrated and synchronized
- [ ] Fuzzy matching achieving >90% MEP resolution
- [ ] Non-MEP speaker handling functional
- [ ] Confidence scoring implemented
- [ ] Political group affiliations resolved

#### Phase 4: Classification & Validation
- [ ] Announcement classification achieving >95% accuracy
- [ ] Comprehensive validation framework implemented
- [ ] Quality metrics collection functional
- [ ] Manual validation sample generated
- [ ] All data quality targets met

#### Phase 5: Integration
- [ ] End-to-end pipeline functional
- [ ] All output formats generated correctly
- [ ] Documentation package complete
- [ ] Performance within acceptable ranges
- [ ] Final quality validation passed

---

## ⚠️ Critical Success Factors

### 1. **Follow the Plan Exactly**
- Do not deviate from the 5-phase structure
- Implement ALL specified components
- Meet ALL quality thresholds
- Include ALL error handling mechanisms

### 2. **Test Continuously**
- Unit tests for each component
- Integration tests for each phase
- End-to-end testing before completion
- Manual validation of sample data

### 3. **Handle Edge Cases**
- Speakers with multiple political groups
- Timestamps in different formats
- Missing or malformed source data
- Network failures and API changes
- Non-Latin character sets

### 4. **Maintain Data Integrity**
- Validate all data transformations
- Preserve original source information
- Track confidence and processing metadata
- Enable audit trail for all decisions

### 5. **Document Everything**
- Clear code comments and docstrings
- Processing decisions and assumptions
- Known limitations and edge cases
- Instructions for manual validation

---

## 🎯 Final Deliverables Checklist

**REQUIRED**: All items must be completed:

### Code & Implementation
- [ ] Complete source code following architecture specification
- [ ] All dependencies properly specified in requirements.txt
- [ ] Configuration files with documented parameters
- [ ] Comprehensive test suite with >90% coverage
- [ ] Command-line interface for pipeline execution

### Data Outputs
- [ ] CSV dataset with all required fields
- [ ] JSONL dataset with identical data
- [ ] Quality assessment report (JSON format)
- [ ] Manual validation sample (25+ segments)
- [ ] Processing log with methodology notes

### Documentation
- [ ] README.md with dataset overview and usage instructions
- [ ] Data dictionary with field definitions and examples
- [ ] Quality report with metrics and limitations
- [ ] Installation and setup instructions
- [ ] Troubleshooting guide for common issues

### Validation Evidence
- [ ] Automated validation results showing quality targets met
- [ ] Manual validation sample with clear review instructions
- [ ] Performance benchmarks and system requirements
- [ ] Test execution results and coverage reports

---

## 🚀 Getting Started

1. **Read all documentation** in `/planner/` directory
2. **Set up development environment** with required dependencies
3. **Test API connectivity** to all data sources
4. **Initialize project structure** exactly as specified
5. **Begin Phase 1** following the comprehensive plan
6. **Update progress checklist daily** with status and metrics
7. **Execute phases sequentially** - do not skip ahead
8. **Validate deliverables** before marking phases complete

---

**Remember**: Your success is measured by the quality and completeness of the final datasets. Follow the plan methodically, implement all safeguards, and prioritize data quality over speed.

**Good luck! The European Parliament research community is counting on accurate, reliable data from this implementation.**

---

*Implementation Prompt Version: 1.0*  
*Target Success Rate: 99%*  
*Expected Timeline: 30 working days*