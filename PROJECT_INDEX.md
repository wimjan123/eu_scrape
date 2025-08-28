# EU Parliament Debates Scraper - Project Documentation Index

## 🎯 Project Overview

**Goal**: Collect European Parliament plenary debates and create a reliable dataset with speaker information, timestamps, and content classification.

**Status**: Planning Phase  
**Last Updated**: 2025-08-28

---

## 📋 Quick Navigation

### 🔬 Research & Planning
- [📊 Research Documentation](./docs/research/README.md) - EU Parliament API analysis and data source evaluation
- [🗺️ Project Plan](./docs/planning/COMPREHENSIVE_PLAN.md) - Multi-phase implementation strategy
- [✅ Implementation Checklist](./docs/planning/CHECKLIST.md) - Phase-based progress tracking
- [🎯 Implementation Prompt](./docs/planning/IMPLEMENTATION_PROMPT.md) - AI implementation guidance

### 🏗️ Technical Architecture
- [🖥️ System Architecture](./docs/technical/ARCHITECTURE.md) - Component design and data flow
- [📡 API Integration](./docs/technical/API_INTEGRATION.md) - EU Parliament endpoints and authentication
- [🗄️ Data Schema](./docs/technical/DATA_SCHEMA.md) - Output format and field definitions
- [🕷️ Scraping Strategy](./docs/technical/SCRAPING_STRATEGY.md) - Web scraping approach and rate limiting

### 📊 Data Management
- [📈 Data Pipeline](./docs/data/PIPELINE.md) - Processing workflow and transformations
- [🔍 Quality Assurance](./docs/data/QUALITY_ASSURANCE.md) - Validation procedures and testing
- [📤 Output Formats](./docs/data/OUTPUT_FORMATS.md) - CSV, JSONL, and export specifications
- [🏷️ Content Classification](./docs/data/CLASSIFICATION.md) - Announcement detection and labeling

### 🚀 Operations & Deployment
- [⚙️ Configuration](./docs/operations/CONFIGURATION.md) - Environment setup and parameters
- [📈 Monitoring](./docs/operations/MONITORING.md) - Logging, metrics, and health checks
- [🛠️ Troubleshooting](./docs/operations/TROUBLESHOOTING.md) - Common issues and solutions
- [🔒 Compliance](./docs/operations/COMPLIANCE.md) - Rate limiting and legal considerations

---

## 📂 Directory Structure

```
eu_scrape/
├── PROJECT_INDEX.md           # This master index
├── README.md                  # Project overview and quick start
├── INITIAL_PRE-PLAN.md        # Original planning document
├── docs/                      # Comprehensive documentation
│   ├── research/              # Research findings and analysis
│   ├── planning/              # Implementation strategy
│   ├── technical/             # System architecture and design
│   ├── data/                  # Data management and schema
│   └── operations/            # Deployment and maintenance
├── src/                       # Source code (planned)
├── tests/                     # Test suite (planned)
├── data/                      # Output datasets (planned)
└── config/                    # Configuration files (planned)
```

---

## 🎯 Target Data Schema

| Field | Type | Description | Source |
|-------|------|-------------|---------|
| `speaker_name` | string | Full name of the speaker | EU Parliament API |
| `speaker_country` | string | Country representation | Member database |
| `speaker_party_or_group` | string | Political group affiliation | Member database |
| `segment_start_ts` | datetime | Segment start timestamp (ISO 8601 UTC) | Debate transcripts |
| `segment_end_ts` | datetime | Segment end timestamp (ISO 8601 UTC) | Debate transcripts |
| `speech_text` | text | Full text of the speech segment | Transcript data |
| `is_announcement` | boolean | Whether segment is an announcement | Content analysis |
| `announcement_label` | string | Type of announcement (if applicable) | Classification rules |

---

## 🔄 Development Workflow

### Phase 1: Research & Discovery ✅
- [x] Initial requirements analysis
- [ ] EU Parliament API exploration
- [ ] Data source evaluation
- [ ] Technical feasibility assessment

### Phase 2: Architecture & Planning 🔄
- [ ] System design documentation
- [ ] API integration strategy
- [ ] Data pipeline architecture
- [ ] Quality assurance framework

### Phase 3: Implementation 📅
- [ ] Core scraping engine
- [ ] Data processing pipeline
- [ ] Output generation system
- [ ] Quality validation tools

### Phase 4: Validation & Testing 📅
- [ ] Data quality verification
- [ ] Performance testing
- [ ] Compliance validation
- [ ] Documentation review

---

## 🛠️ Tools & Technologies

### Primary Tools
- **Web Scraping**: Playwright, Puppeteer
- **Data Processing**: Python, pandas
- **API Integration**: requests, httpx
- **Output Formats**: CSV, JSONL
- **Quality Assurance**: pytest, data validation

### MCP Tools Available
- **Sequential Thinking**: Complex analysis and planning
- **Context7**: EU Parliament API documentation
- **Playwright**: Browser automation and testing

---

## 📚 Key Resources

### Official Sources
- [EU Parliament Open Data Portal](https://data.europarl.europa.eu/)
- [Daily Verbatim Reports](https://europarl.europa.eu/plenary/en/debates-video.html)
- [Members Database](https://europarl.europa.eu/meps/en/directory-result)

### Documentation References
- [API Documentation](./docs/technical/API_INTEGRATION.md#endpoints)
- [Data Quality Standards](./docs/data/QUALITY_ASSURANCE.md#validation-rules)
- [Rate Limiting Guidelines](./docs/operations/COMPLIANCE.md#rate-limits)

---

## 🤝 Contributing

1. **Research Phase**: Add findings to `./docs/research/`
2. **Planning Updates**: Update `./docs/planning/COMPREHENSIVE_PLAN.md`
3. **Technical Design**: Document in `./docs/technical/`
4. **Implementation**: Follow `./docs/planning/CHECKLIST.md`

---

## 📞 Support

For questions about:
- **Research**: See [Research Documentation](./docs/research/README.md)
- **Implementation**: Check [Implementation Prompt](./docs/planning/IMPLEMENTATION_PROMPT.md)
- **Technical Issues**: Refer to [Troubleshooting Guide](./docs/operations/TROUBLESHOOTING.md)

---

*Last updated: 2025-08-28 | Next review: TBD*