# EU Parliament Data Sources Research Report

## 📋 Executive Summary

This research identifies the optimal data sources and collection strategies for extracting European Parliament plenary debate data with the required fields: speaker_name, speaker_country, speaker_party_or_group, segment_start_ts, segment_end_ts, speech_text, is_announcement, and announcement_label.

**Primary Finding**: A hybrid approach combining official APIs with structured web scraping will be required to obtain all target data fields with sufficient quality and granularity.

---

## 🎯 Primary Data Sources Identified

### 1. European Parliament Open Data Portal ⭐⭐⭐⭐⭐
**URL**: `https://data.europarl.europa.eu/`  
**Status**: Production-ready since January 2023  
**API Access**: Available through developer corner  
**Rate Limiting**: Not explicitly documented - requires conservative approach

**Available Datasets**:
- Plenary session documents (agendas, minutes, verbatim reports)
- MEP information with political groups and countries
- Parliamentary activities and votes

**Data Formats**: RDF/Turtle, JSON-LD, CSV (planned)
**Languages**: Multilingual support (24 EU languages)

**Strengths**:
- Official, authoritative source
- RESTful API with OpenAPI documentation
- Structured metadata
- Regular updates

**Limitations**:
- Limited timestamp granularity (session-level, not speech-segment level)
- API documentation access issues encountered
- May require supplementary sources for fine-grained timing

### 2. EUR-Lex Publications Office ⭐⭐⭐⭐
**URL**: `http://publications.europa.eu/webapi/rdf/sparql`  
**Status**: Mature, well-documented  
**Access**: SPARQL endpoint + REST API  
**Rate Limiting**: Conservative 1-2 second delays recommended

**Capabilities**:
- SPARQL queries for complex data relationships
- Document metadata and full-text access
- Historical data going back decades
- Linked data approach with rich semantic relationships

**Strengths**:
- Powerful query capabilities
- Comprehensive historical coverage
- Well-documented with extensive examples
- Supports complex filtering and joins

**Limitations**:
- Primarily document-oriented, not speech-segment oriented
- Timestamps may be document-level, not speech-level
- Requires SPARQL expertise

### 3. Verbatim Reports (CRE Documents) ⭐⭐⭐⭐
**URL**: `https://www.europarl.europa.eu/doceo/document/CRE-*`  
**Format**: Structured XML/HTML  
**Coverage**: Complete speech transcripts  
**Update Frequency**: Published after each plenary session

**Structure**:
- Contains full speech text in original languages
- Speaker identification and roles
- Procedural annotations and timestamps
- Session structure with agenda items

**Strengths**:
- Complete speech content
- Speaker identification
- Procedural context
- Multiple language versions

**Limitations**:
- May require parsing for segment-level timestamps
- Announcement classification may need custom logic
- Political group affiliations may need separate resolution

### 4. MEP Database ⭐⭐⭐⭐
**URL**: `https://www.europarl.europa.eu/meps/en/home`  
**API Access**: Through Open Data Portal  
**Coverage**: Current and historical MEP information

**Data Available**:
- Full MEP profiles with countries
- Political group affiliations
- Historical changes in group membership
- Contact information and roles

**Strengths**:
- Authoritative source for MEP data
- Historical tracking of group changes
- Complete coverage of all MEPs

**Limitations**:
- Non-MEP speakers require alternative resolution
- Group affiliations change over time
- May need temporal matching for historical accuracy

---

## 🔍 Timestamp Analysis

### Current State
**Challenge**: No single source provides speech-segment level timestamps in a structured format.

**Available Timing Information**:
1. **Session-level**: Start/end times for entire plenary sessions
2. **Agenda-item level**: Timing for major debate topics
3. **Video metadata**: Potential timing information from multimedia content
4. **Procedural markers**: Time references in verbatim reports (e.g., "14:30 - The sitting opened")

### Recommended Approach
**Hybrid Timing Strategy**:
1. Extract session and agenda-item timestamps from official sources
2. Parse time references from verbatim report text
3. Use video metadata where available as validation
4. Implement fallback estimation based on speech length and session duration

**Implementation Priority**:
- Phase 1: Session and agenda-item level timing
- Phase 2: Speech-segment estimation from text analysis
- Phase 3: Video metadata integration (if feasible)

---

## 📢 Announcement Detection Strategy

### Announcement Types Identified
1. **Session Management**: Opening/closing, breaks, procedural issues
2. **Agenda Items**: Introduction of new topics, transitions
3. **Voting Procedures**: Vote announcements, results
4. **Procedural Notices**: Rule clarifications, technical issues
5. **General Announcements**: Administrative communications

### Detection Patterns
**Role-based Detection**:
- Speaker roles: President, Vice-President, Secretary-General
- Procedural titles and formal language patterns

**Content-based Detection**:
```
Linguistic Patterns:
- "The sitting is opened/closed"
- "Next item on the agenda"
- "Voting time"
- "I call upon..."
- "The debate is closed"
- "Procedures without debate"
- "Written statements"

Structural Patterns:
- Short segments (< 50 words)
- Formal language register
- Procedural vocabulary
- Time references
```

**Context-based Detection**:
- Position in session structure
- Surrounding content analysis
- Speaker role correlation

---

## 👥 Speaker Resolution Strategy

### MEP Speaker Resolution
**Primary Method**: Direct matching against MEP database
- Full name matching with fuzzy logic for variations
- Temporal validation for group affiliations
- Country assignment based on representation

**Fallback Methods**:
- Partial name matching with confidence scoring
- Role-based inference (e.g., "President" → current EP President)
- Context clues from speech content

### Non-MEP Speaker Handling
**Categories**:
- European Council representatives
- European Commission officials
- Guest speakers and experts
- National government representatives

**Resolution Strategy**:
- Maintain separate database of institutional roles
- Extract title/role information from speech context
- Manual verification for high-profile speakers
- Clear flagging of unresolved speakers

---

## ⚡ Technical Implementation Approach

### Data Collection Pipeline
```
Phase 1: Discovery
├── Session catalog from Open Data Portal
├── Date range filtering and prioritization
└── Metadata collection and validation

Phase 2: Content Collection
├── Verbatim report download (XML/HTML)
├── MEP database synchronization
└── Session structure parsing

Phase 3: Processing
├── Speech segment extraction
├── Speaker resolution and enrichment
├── Timestamp estimation and validation
├── Announcement classification
└── Data quality validation

Phase 4: Output
├── CSV/JSONL generation
├── Quality report creation
└── Documentation generation
```

### Rate Limiting Strategy
**Conservative Approach**:
- Open Data Portal: 1 request per 2 seconds
- EUR-Lex SPARQL: 1 query per 2 seconds
- Verbatim reports: 1 document per 3 seconds
- MEP database: 1 request per 1 second

**Error Handling**:
- Exponential backoff on failures
- Checkpoint/resume capability
- Request logging for debugging
- Graceful degradation on partial failures

---

## 🎯 Recommended Implementation Priority

### Phase 1: Foundation (Weeks 1-2)
1. Set up Open Data Portal API client
2. Implement MEP database synchronization
3. Create basic session discovery and cataloging
4. Build verbatim report download system

### Phase 2: Core Processing (Weeks 3-4)
1. Develop speech segment extraction
2. Implement speaker resolution system
3. Create announcement classification rules
4. Build timestamp estimation logic

### Phase 3: Quality & Output (Weeks 5-6)
1. Implement data validation pipeline
2. Create output format generators
3. Build quality assessment tools
4. Generate comprehensive documentation

---

## 📊 Expected Data Quality Metrics

### Completeness Targets
- **Session Coverage**: 95%+ of available sessions
- **Speaker Resolution**: 90%+ MEPs resolved, 70%+ non-MEPs
- **Speech Content**: 98%+ segments with complete text
- **Timestamp Accuracy**: ±5 minutes for segment timing

### Quality Assurance Plan
- Manual validation of 20+ randomly selected segments
- Cross-reference speaker assignments with official records
- Validate announcement classifications against sample data
- Compare outputs with existing research datasets where available

---

## 🚨 Known Limitations & Risks

### Technical Risks
1. **API Changes**: Official APIs may change without notice
2. **Rate Limiting**: Undocumented limits may cause collection delays
3. **Data Format Changes**: XML/HTML structure changes in source documents
4. **Access Restrictions**: Potential IP blocking or access limitations

### Data Quality Risks
1. **Timestamp Precision**: Limited by source data granularity
2. **Speaker Ambiguity**: Multiple speakers with similar names
3. **Language Variations**: Name variations across languages
4. **Historical Changes**: Political group changes over time

### Mitigation Strategies
- Implement robust error handling and retry logic
- Create comprehensive logging and monitoring
- Build flexible parsers that adapt to format changes
- Maintain human validation workflows for edge cases

---

## 📚 Research Sources

1. **European Parliament Open Data Portal**: https://data.europarl.europa.eu/
2. **EUR-Lex Publications Office**: http://publications.europa.eu/webapi/rdf/sparql
3. **LinkedEP Research**: Academic research on EP debates as Linked Open Data
4. **DCEP Corpus**: Digital Corpus of the European Parliament
5. **OpenSanctions MEP Database**: https://www.opensanctions.org/datasets/eu_meps/

---

*Research conducted: 2025-08-28*  
*Confidence Level: High for data source identification, Medium for implementation complexity*  
*Recommended Review Date: Before implementation phase*