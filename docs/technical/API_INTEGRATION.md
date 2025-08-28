# EU Parliament API Integration Guide

## 📋 Overview

This document outlines the API integration strategy for collecting European Parliament plenary debate data. Based on research using the EUR-Lex framework and EU Publications Office endpoints, this guide provides comprehensive technical details for accessing official EU data sources.

---

## 🎯 Primary Data Sources

### 1. EU Publications Office SPARQL Endpoint
**Endpoint**: `http://publications.europa.eu/webapi/rdf/sparql`  
**Purpose**: Semantic queries for metadata, document discovery, and structured data retrieval  
**Rate Limits**: Not explicitly documented - use conservative approach  
**Authentication**: None required for public data  

### 2. EUR-Lex REST API
**Base URL**: `https://eur-lex.europa.eu/`  
**Purpose**: Document content retrieval, full-text access, metadata  
**Rate Limits**: Implement respectful delays (1-2 seconds between requests)  
**Authentication**: None required for public data  

### 3. European Parliament Open Data Portal
**Base URL**: `https://data.europarl.europa.eu/`  
**Purpose**: Structured datasets, member information, session metadata  
**Rate Limits**: Follow robots.txt and implement delays  
**Authentication**: None required for public access  

---

## 📊 Data Schema Mapping

### Target Output Schema
```json
{
  "speaker_name": "string",           // Full name of speaker
  "speaker_country": "string",        // ISO country code or full name  
  "speaker_party_or_group": "string", // Political group affiliation
  "segment_start_ts": "datetime",     // ISO 8601 UTC timestamp
  "segment_end_ts": "datetime",       // ISO 8601 UTC timestamp
  "speech_text": "text",             // Full speech content
  "is_announcement": "boolean",       // Classification flag
  "announcement_label": "string"      // Type if is_announcement=true
}
```

### Source Field Mapping
| Output Field | SPARQL Property | REST API Field | Notes |
|--------------|----------------|----------------|-------|
| speaker_name | `cdm:person_name` | `speaker.name` | Full name extraction |
| speaker_country | `cdm:country_of_origin` | `speaker.country` | ISO code preferred |
| speaker_party_or_group | `cdm:political_group` | `speaker.group` | EU group affiliation |
| segment_start_ts | `cdm:timestamp_start` | `timestamps.start` | Convert to ISO 8601 UTC |
| segment_end_ts | `cdm:timestamp_end` | `timestamps.end` | Convert to ISO 8601 UTC |
| speech_text | `cdm:expression` | `content.text` | Full text extraction |
| is_announcement | N/A | N/A | Derived classification |
| announcement_label | N/A | N/A | Derived from content analysis |

---

## 🔍 SPARQL Query Patterns

### Basic Plenary Session Query
```sparql
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?work ?type ?celex ?date WHERE {
  ?work cdm:resource_legal_id_sector ?sector .
  FILTER(str(?sector) = '3')  # Legal acts sector
  
  ?work cdm:resource_legal_is_about_concept_directory-code ?directory .
  FILTER(str(?directory) = '18')  # CFSP directory as example
  
  OPTIONAL { ?work cdm:resource_legal_id_celex ?celex }
  OPTIONAL { ?work cdm:work_date_document ?date }
  OPTIONAL { ?work cdm:resource_type ?type }
  
  FILTER NOT EXISTS { ?work cdm:do_not_index "true"^^xsd:boolean }
}
LIMIT 1000
```

### Speaker Information Query
```sparql
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

SELECT ?speaker ?name ?country ?group WHERE {
  ?speaker cdm:person_name ?name .
  
  OPTIONAL { ?speaker cdm:country_of_origin ?country }
  OPTIONAL { ?speaker cdm:political_group ?group }
  
  FILTER(LANG(?name) = 'en' || LANG(?name) = '')
}
```

### Parliamentary Debate Sessions Query
```sparql
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX dct: <http://purl.org/dc/terms/>

SELECT ?session ?date ?title ?participants WHERE {
  ?session a cdm:PlenarySession .
  ?session dct:date ?date .
  
  OPTIONAL { ?session dct:title ?title }
  OPTIONAL { ?session cdm:has_participants ?participants }
  
  FILTER(?date >= "2024-01-01"^^xsd:date)
  FILTER(?date <= "2024-12-31"^^xsd:date)
}
ORDER BY DESC(?date)
```

---

## 📡 API Implementation Strategy

### 1. Query Execution Pattern
```python
import requests
import time
from typing import Dict, List, Optional

class EUParliamentAPI:
    def __init__(self):
        self.sparql_endpoint = "http://publications.europa.eu/webapi/rdf/sparql"
        self.rest_base = "https://eur-lex.europa.eu"
        self.session = requests.Session()
        self.request_delay = 2.0  # Conservative rate limiting
    
    def execute_sparql_query(self, query: str) -> Dict:
        """Execute SPARQL query with rate limiting and error handling."""
        params = {
            'query': query,
            'format': 'application/sparql-results+json'
        }
        
        response = self.session.get(
            self.sparql_endpoint,
            params=params,
            timeout=30
        )
        
        time.sleep(self.request_delay)  # Rate limiting
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"SPARQL query failed: {response.status_code}")
    
    def fetch_document_content(self, celex_id: str, lang: str = 'en') -> str:
        """Fetch full document content via REST API."""
        url = f"{self.rest_base}/legal-content/{lang.upper()}/TXT/"
        
        response = self.session.get(url, timeout=30)
        time.sleep(self.request_delay)
        
        if response.status_code == 200:
            return response.text
        else:
            return None
```

### 2. Data Processing Pipeline
```python
class DataProcessor:
    def __init__(self):
        self.api = EUParliamentAPI()
    
    def extract_debate_segments(self, session_data: Dict) -> List[Dict]:
        """Extract individual speech segments from session data."""
        segments = []
        
        # Parse session structure
        for speech in session_data.get('speeches', []):
            segment = {
                'speaker_name': self.normalize_speaker_name(speech.get('speaker')),
                'speaker_country': self.resolve_country(speech.get('speaker')),
                'speaker_party_or_group': self.resolve_political_group(speech.get('speaker')),
                'segment_start_ts': self.parse_timestamp(speech.get('start_time')),
                'segment_end_ts': self.parse_timestamp(speech.get('end_time')),
                'speech_text': self.clean_speech_text(speech.get('content')),
                'is_announcement': self.classify_announcement(speech),
                'announcement_label': self.label_announcement_type(speech)
            }
            segments.append(segment)
        
        return segments
    
    def classify_announcement(self, speech: Dict) -> bool:
        """Classify whether a segment is an announcement."""
        content = speech.get('content', '').lower()
        speaker_role = speech.get('speaker', {}).get('role', '').lower()
        
        # Detection patterns for announcements
        announcement_patterns = [
            'the sitting',
            'order of the day',
            'next item',
            'voting time',
            'procedures without debate',
            'written statements'
        ]
        
        role_indicators = [
            'president',
            'vice-president',
            'secretary-general'
        ]
        
        # Check content patterns
        has_announcement_content = any(pattern in content for pattern in announcement_patterns)
        
        # Check speaker role
        has_announcement_role = any(role in speaker_role for role in role_indicators)
        
        return has_announcement_content or has_announcement_role
    
    def label_announcement_type(self, speech: Dict) -> Optional[str]:
        """Label the type of announcement if applicable."""
        if not self.classify_announcement(speech):
            return None
        
        content = speech.get('content', '').lower()
        
        if 'voting' in content:
            return 'voting_procedure'
        elif 'order of the day' in content:
            return 'agenda_item'
        elif 'sitting' in content:
            return 'session_management'
        elif 'written statements' in content:
            return 'procedural_notice'
        else:
            return 'general_announcement'
```

---

## ⚠️ Rate Limiting & Compliance

### Rate Limiting Strategy
- **SPARQL Queries**: Max 1 query per 2 seconds
- **REST API Calls**: Max 1 request per 2 seconds  
- **Bulk Operations**: Implement exponential backoff on errors
- **Daily Limits**: Monitor for any service-imposed daily limits

### Error Handling
```python
import time
import random
from requests.exceptions import RequestException

def execute_with_backoff(func, max_retries=3):
    """Execute function with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return func()
        except RequestException as e:
            if attempt == max_retries - 1:
                raise e
            
            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
    
    raise Exception("Max retries exceeded")
```

### Compliance Considerations
- **Robots.txt**: Check and respect robots.txt files
- **User-Agent**: Use descriptive User-Agent header
- **Data Attribution**: Include proper attribution in outputs
- **Terms of Service**: Comply with EU Publications Office terms

---

## 🔄 Data Quality Assurance

### Validation Rules
1. **Timestamp Validation**: Ensure all timestamps are valid ISO 8601 UTC
2. **Speaker Validation**: Cross-reference speakers against MEP database
3. **Content Completeness**: Verify speech_text is not truncated
4. **Announcement Classification**: Manual spot-check 20+ classified segments

### Quality Metrics
- **Completeness Rate**: % of segments with all required fields
- **Speaker Resolution Rate**: % of speakers successfully linked to MEP data
- **Timestamp Accuracy**: % of segments with valid start/end times
- **Classification Accuracy**: % of correctly classified announcements

---

## 🔗 Cross-References

### Related Documentation
- [Data Schema](../data/DATA_SCHEMA.md) - Detailed field definitions
- [Quality Assurance](../data/QUALITY_ASSURANCE.md) - Validation procedures  
- [Scraping Strategy](./SCRAPING_STRATEGY.md) - Web scraping fallback methods
- [Configuration](../operations/CONFIGURATION.md) - API credentials and settings

### External Resources
- [EU Publications Office SPARQL Guide](http://publications.europa.eu/webapi/rdf/sparql)
- [EUR-Lex Documentation](https://eur-lex.europa.eu/content/help/data-reuse/webservice.html)
- [European Parliament Open Data](https://data.europarl.europa.eu/en/home)

---

## 🚀 Implementation Checklist

- [ ] Set up API client with rate limiting
- [ ] Implement SPARQL query execution
- [ ] Create REST API integration
- [ ] Build speaker resolution system  
- [ ] Implement announcement classification
- [ ] Add timestamp normalization
- [ ] Create data validation pipeline
- [ ] Add error handling and retries
- [ ] Implement quality metrics collection
- [ ] Test with sample date range

---

*Last updated: 2025-08-28 | Next review: Implementation phase*