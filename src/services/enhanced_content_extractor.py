#!/usr/bin/env python3
"""
Enhanced Content Extraction and Preprocessing System

Advanced content extraction with intelligent preprocessing, quality assessment,
and multi-format document handling for EU Parliament data.

Key Features:
- Multi-format document processing (HTML, PDF, XML, JSON)
- Intelligent content preprocessing with noise reduction
- Quality-based content scoring and filtering
- Language detection and handling
- Speaker identification and normalization
- Content structure analysis and enhancement
- Performance optimization with caching and batching
"""

import asyncio
import re
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set, Union, NamedTuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import time

from bs4 import BeautifulSoup, Tag, NavigableString
import pandas as pd
from langdetect import detect, detect_langs
from fuzzywuzzy import fuzz, process
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from ..models.speech import RawSpeechSegment
from ..models.session import SessionMetadata
from ..core.logging import get_logger
from ..core.metrics import MetricsCollector
from ..core.cache import get_cache_manager

logger = get_logger(__name__)


class ContentType(Enum):
    """Document content types for processing"""
    VERBATIM_TRANSCRIPT = "verbatim_transcript"
    COMMITTEE_MINUTES = "committee_minutes"
    AMENDMENT_TEXT = "amendment_text"
    VOTING_RECORD = "voting_record"
    PRESS_RELEASE = "press_release"
    LEGISLATIVE_TEXT = "legislative_text"
    UNKNOWN = "unknown"


class ProcessingQuality(Enum):
    """Quality levels for processed content"""
    EXCELLENT = "excellent"    # >90% confidence, clean structure
    GOOD = "good"             # >70% confidence, minor issues
    FAIR = "fair"             # >50% confidence, some processing needed
    POOR = "poor"             # <50% confidence, significant issues
    UNUSABLE = "unusable"     # Cannot be processed reliably


class LanguageCode(Enum):
    """EU official languages for content detection"""
    EN = "en"  # English
    FR = "fr"  # French  
    DE = "de"  # German
    ES = "es"  # Spanish
    IT = "it"  # Italian
    PL = "pl"  # Polish
    RO = "ro"  # Romanian
    NL = "nl"  # Dutch
    EL = "el"  # Greek
    PT = "pt"  # Portuguese
    CS = "cs"  # Czech
    HU = "hu"  # Hungarian
    SV = "sv"  # Swedish
    FI = "fi"  # Finnish
    DA = "da"  # Danish
    SK = "sk"  # Slovak
    LT = "lt"  # Lithuanian
    LV = "lv"  # Latvian
    ET = "et"  # Estonian
    SL = "sl"  # Slovenian
    BG = "bg"  # Bulgarian
    HR = "hr"  # Croatian
    MT = "mt"  # Maltese
    GA = "ga"  # Irish
    UNKNOWN = "unknown"


@dataclass
class ContentExtractionResult:
    """Result of content extraction with quality metrics"""
    content_id: str
    original_content: str
    extracted_content: str
    content_type: ContentType
    processing_quality: ProcessingQuality
    language: LanguageCode
    
    # Structure analysis
    speaker_segments: List[Dict[str, Any]] = field(default_factory=list)
    structural_elements: Dict[str, int] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Processing metadata
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0
    preprocessing_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class SpeakerSegment:
    """Individual speaker segment with enhanced metadata"""
    segment_id: str
    speaker_raw: str
    speaker_normalized: str
    speech_text: str
    language: LanguageCode
    
    # Position and context
    sequence_number: int
    start_position: int
    end_position: int
    context_before: str = ""
    context_after: str = ""
    
    # Quality indicators
    confidence_score: float = 0.0
    quality_indicators: Dict[str, bool] = field(default_factory=dict)
    processing_notes: List[str] = field(default_factory=list)


@dataclass
class PreprocessingConfig:
    """Configuration for content preprocessing"""
    # Language processing
    enable_language_detection: bool = True
    target_languages: List[LanguageCode] = field(default_factory=lambda: [LanguageCode.EN, LanguageCode.FR])
    
    # Content cleaning
    remove_html_tags: bool = True
    normalize_whitespace: bool = True
    remove_empty_segments: bool = True
    min_segment_length: int = 10
    
    # Speaker processing
    normalize_speaker_names: bool = True
    merge_continuation_speeches: bool = True
    speaker_confidence_threshold: float = 0.6
    
    # Quality filtering
    quality_threshold: ProcessingQuality = ProcessingQuality.FAIR
    enable_quality_scoring: bool = True
    
    # Performance
    enable_caching: bool = True
    batch_processing: bool = True
    max_workers: int = 4


class LanguageDetector:
    """Enhanced language detection with EU Parliament context"""
    
    def __init__(self):
        self.eu_language_codes = {lang.value for lang in LanguageCode if lang != LanguageCode.UNKNOWN}
        self.common_eu_phrases = {
            'en': ['madam president', 'mr president', 'honourable members', 'european parliament'],
            'fr': ['madame la présidente', 'monsieur le président', 'députés', 'parlement européen'],
            'de': ['frau präsidentin', 'herr präsident', 'abgeordnete', 'europäisches parlament'],
            'es': ['señora presidenta', 'señor presidente', 'diputados', 'parlamento europeo'],
            'it': ['signora presidente', 'signor presidente', 'deputati', 'parlamento europeo']
        }
        
    def detect_language(self, text: str, use_context: bool = True) -> Tuple[LanguageCode, float]:
        """
        Detect language with EU Parliament context awareness
        
        Args:
            text: Text to analyze
            use_context: Use EU Parliament contextual hints
            
        Returns:
            Tuple of (language, confidence)
        """
        if not text or len(text.strip()) < 10:
            return LanguageCode.UNKNOWN, 0.0
        
        try:
            # Primary language detection
            lang_probs = detect_langs(text.lower())
            
            # Filter to EU languages only
            eu_lang_probs = [
                (lang.lang, lang.prob) for lang in lang_probs 
                if lang.lang in self.eu_language_codes
            ]
            
            if not eu_lang_probs:
                return LanguageCode.UNKNOWN, 0.0
                
            primary_lang, primary_confidence = eu_lang_probs[0]
            
            # Enhance with contextual analysis
            if use_context and primary_confidence < 0.9:
                context_boost = self._analyze_context(text.lower(), primary_lang)
                primary_confidence = min(1.0, primary_confidence + context_boost)
            
            try:
                return LanguageCode(primary_lang), primary_confidence
            except ValueError:
                return LanguageCode.UNKNOWN, 0.0
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return LanguageCode.UNKNOWN, 0.0
    
    def _analyze_context(self, text: str, detected_lang: str) -> float:
        """Analyze EU Parliament contextual clues for language confirmation"""
        if detected_lang not in self.common_eu_phrases:
            return 0.0
            
        phrases = self.common_eu_phrases[detected_lang]
        found_phrases = sum(1 for phrase in phrases if phrase in text)
        
        # Boost confidence based on contextual matches
        return min(0.3, found_phrases * 0.1)


class SpeakerNormalizer:
    """Advanced speaker name normalization with fuzzy matching"""
    
    def __init__(self):
        self.known_speakers: Dict[str, str] = {}  # raw -> normalized mapping
        self.speaker_patterns = {
            'president': r'(?:madam|mr|mrs|ms)?\s*president(?:e|a)?',
            'commissioner': r'commissioner\s+([a-zA-ZÀ-ÿ\s]+)',
            'mep': r'([a-zA-ZÀ-ÿ\s]+)\s*\([A-Z]{2,4}\)',
            'rapporteur': r'([a-zA-ZÀ-ÿ\s]+),?\s*rapporteur'
        }
        self.title_patterns = [
            r'^(madam|mr|mrs|ms|dr|prof)\s+',
            r'\s+(mp|mep|commissioner)$',
            r'\([A-Z]{2,4}\)$'  # Political group abbreviations
        ]
    
    def normalize_speaker(self, raw_speaker: str) -> Tuple[str, float]:
        """
        Normalize speaker name with confidence scoring
        
        Args:
            raw_speaker: Raw speaker string from document
            
        Returns:
            Tuple of (normalized_name, confidence_score)
        """
        if not raw_speaker or len(raw_speaker.strip()) < 2:
            return "", 0.0
            
        # Check cache first
        if raw_speaker in self.known_speakers:
            return self.known_speakers[raw_speaker], 1.0
        
        # Clean and normalize
        normalized = self._clean_speaker_name(raw_speaker)
        confidence = self._calculate_confidence(raw_speaker, normalized)
        
        # Cache if high confidence
        if confidence >= 0.8:
            self.known_speakers[raw_speaker] = normalized
            
        return normalized, confidence
    
    def _clean_speaker_name(self, raw_speaker: str) -> str:
        """Clean and standardize speaker name"""
        # Remove HTML tags and excessive whitespace
        cleaned = re.sub(r'<[^>]+>', '', raw_speaker)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Handle special characters and encoding
        cleaned = cleaned.replace('–', '-').replace('—', '-')
        
        # Extract name from common patterns
        for pattern_name, pattern in self.speaker_patterns.items():
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                if pattern_name == 'president':
                    return "President"
                elif pattern_name == 'commissioner':
                    return f"Commissioner {match.group(1).strip().title()}"
                elif pattern_name in ['mep', 'rapporteur']:
                    return match.group(1).strip().title()
        
        # Remove titles and suffixes
        for title_pattern in self.title_patterns:
            cleaned = re.sub(title_pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Capitalize properly
        if cleaned:
            return ' '.join(word.capitalize() for word in cleaned.split())
        
        return raw_speaker.strip()
    
    def _calculate_confidence(self, raw: str, normalized: str) -> float:
        """Calculate confidence score for normalization"""
        if not raw or not normalized:
            return 0.0
        
        # Base confidence from string similarity
        similarity = fuzz.ratio(raw.lower(), normalized.lower()) / 100.0
        
        # Boost for recognized patterns
        pattern_boost = 0.0
        for pattern in self.speaker_patterns.values():
            if re.search(pattern, raw, re.IGNORECASE):
                pattern_boost = 0.2
                break
        
        # Penalty for very short names
        length_penalty = 0.0
        if len(normalized) < 5:
            length_penalty = 0.2
        
        return max(0.0, min(1.0, similarity + pattern_boost - length_penalty))


class ContentStructureAnalyzer:
    """Analyze document structure and extract metadata"""
    
    def __init__(self):
        self.structural_patterns = {
            'headings': r'<h[1-6][^>]*>(.*?)</h[1-6]>',
            'paragraphs': r'<p[^>]*>(.*?)</p>',
            'lists': r'<[uo]l[^>]*>(.*?)</[uo]l>',
            'tables': r'<table[^>]*>(.*?)</table>',
            'speakers': r'<p[^>]*>([^.–\-<]*?)(?:\.|\s*–|\s*—)',
            'timestamps': r'(\d{1,2}:\d{2}(?::\d{2})?)',
            'votes': r'(voting|vote|ballot)',
            'amendments': r'(amendment|amend)'
        }
    
    def analyze_structure(self, content: str, content_type: ContentType) -> Dict[str, Any]:
        """
        Analyze document structure and extract metadata
        
        Args:
            content: HTML or text content
            content_type: Type of document being analyzed
            
        Returns:
            Structure analysis results
        """
        analysis = {
            'element_counts': {},
            'content_sections': [],
            'speaker_indicators': [],
            'quality_indicators': {},
            'structural_score': 0.0
        }
        
        try:
            # Count structural elements
            for element_type, pattern in self.structural_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                analysis['element_counts'][element_type] = len(matches)
                
                if element_type == 'speakers' and matches:
                    analysis['speaker_indicators'] = matches[:10]  # Sample first 10
            
            # Analyze content sections
            if content_type == ContentType.VERBATIM_TRANSCRIPT:
                analysis['content_sections'] = self._analyze_verbatim_structure(content)
            elif content_type == ContentType.COMMITTEE_MINUTES:
                analysis['content_sections'] = self._analyze_minutes_structure(content)
            
            # Calculate structural quality score
            analysis['structural_score'] = self._calculate_structural_score(analysis, content_type)
            
            # Quality indicators
            analysis['quality_indicators'] = {
                'has_speakers': analysis['element_counts'].get('speakers', 0) > 0,
                'has_structure': analysis['element_counts'].get('headings', 0) > 0,
                'reasonable_length': 100 < len(content) < 1000000,
                'has_paragraphs': analysis['element_counts'].get('paragraphs', 0) > 0
            }
            
        except Exception as e:
            logger.warning(f"Structure analysis failed: {e}")
            analysis['structural_score'] = 0.0
        
        return analysis
    
    def _analyze_verbatim_structure(self, content: str) -> List[Dict[str, Any]]:
        """Analyze verbatim transcript structure"""
        soup = BeautifulSoup(content, 'lxml')
        sections = []
        
        # Find major sections
        headings = soup.find_all(['h1', 'h2', 'h3'])
        for heading in headings:
            section = {
                'type': 'heading',
                'level': int(heading.name[1]),
                'text': heading.get_text().strip(),
                'position': str(heading)[:50]
            }
            sections.append(section)
        
        # Find speech blocks
        speech_paragraphs = soup.find_all('p', class_='contents')
        for i, para in enumerate(speech_paragraphs[:20]):  # Limit to first 20
            text = para.get_text().strip()
            if len(text) > 20 and ('.' in text or '–' in text):
                section = {
                    'type': 'speech',
                    'sequence': i,
                    'preview': text[:100],
                    'estimated_speaker': self._extract_speaker_from_text(text)
                }
                sections.append(section)
        
        return sections
    
    def _analyze_minutes_structure(self, content: str) -> List[Dict[str, Any]]:
        """Analyze committee minutes structure"""
        # Similar analysis adapted for minutes format
        return []  # Simplified for now
    
    def _extract_speaker_from_text(self, text: str) -> str:
        """Extract potential speaker from text"""
        # Look for speaker patterns at beginning of text
        speaker_match = re.match(r'^([^.–\-<]*?)(?:\.|\s*–|\s*—)', text)
        if speaker_match:
            return speaker_match.group(1).strip()
        return ""
    
    def _calculate_structural_score(self, analysis: Dict[str, Any], content_type: ContentType) -> float:
        """Calculate overall structural quality score"""
        scores = []
        
        # Base score from element presence
        if analysis['element_counts'].get('speakers', 0) > 0:
            scores.append(0.3)
        if analysis['element_counts'].get('paragraphs', 0) > 5:
            scores.append(0.2)
        if analysis['element_counts'].get('headings', 0) > 0:
            scores.append(0.2)
            
        # Quality indicators
        quality_count = sum(1 for v in analysis['quality_indicators'].values() if v)
        scores.append(quality_count * 0.1)
        
        return min(1.0, sum(scores))


class EnhancedContentExtractor:
    """Main enhanced content extraction system"""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        
        # Core components
        self.language_detector = LanguageDetector()
        self.speaker_normalizer = SpeakerNormalizer()
        self.structure_analyzer = ContentStructureAnalyzer()
        
        # Performance optimization
        self.cache_manager = get_cache_manager() if self.config.enable_caching else None
        self.metrics = MetricsCollector()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Processing statistics
        self.processing_stats = {
            'documents_processed': 0,
            'extraction_failures': 0,
            'quality_filtered': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0
        }
        
        logger.info("Enhanced Content Extractor initialized")
    
    async def extract_content(
        self,
        content: str,
        content_type: ContentType = ContentType.UNKNOWN,
        session_metadata: SessionMetadata = None
    ) -> ContentExtractionResult:
        """
        Extract and preprocess content with comprehensive analysis
        
        Args:
            content: Raw content to process
            content_type: Type of content for specialized processing
            session_metadata: Session context for enhanced processing
            
        Returns:
            Complete extraction result with quality metrics
        """
        start_time = time.time()
        
        # Generate content ID for caching
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        content_id = f"{content_type.value}_{content_hash[:12]}"
        
        # Check cache first
        if self.cache_manager:
            cached_result = await self.cache_manager.get(f"extraction_{content_id}")
            if cached_result:
                self.processing_stats['cache_hits'] += 1
                return ContentExtractionResult(**cached_result)
        
        try:
            # Initialize result
            result = ContentExtractionResult(
                content_id=content_id,
                original_content=content,
                extracted_content="",
                content_type=content_type,
                processing_quality=ProcessingQuality.POOR,
                language=LanguageCode.UNKNOWN
            )
            
            # Step 1: Content type detection if unknown
            if content_type == ContentType.UNKNOWN:
                content_type = self._detect_content_type(content)
                result.content_type = content_type
                result.preprocessing_steps.append("content_type_detection")
            
            # Step 2: Language detection
            if self.config.enable_language_detection:
                language, lang_confidence = self.language_detector.detect_language(content)
                result.language = language
                result.quality_metrics['language_confidence'] = lang_confidence
                result.preprocessing_steps.append("language_detection")
                
                if language == LanguageCode.UNKNOWN:
                    result.warnings.append("Language could not be detected reliably")
            
            # Step 3: Structure analysis
            structure_analysis = self.structure_analyzer.analyze_structure(content, content_type)
            result.structural_elements = structure_analysis['element_counts']
            result.quality_metrics['structural_score'] = structure_analysis['structural_score']
            result.preprocessing_steps.append("structure_analysis")
            
            # Step 4: Content cleaning and preprocessing
            cleaned_content = await self._preprocess_content(content, content_type, result)
            result.extracted_content = cleaned_content
            result.preprocessing_steps.append("content_cleaning")
            
            # Step 5: Speaker segment extraction
            if content_type in [ContentType.VERBATIM_TRANSCRIPT, ContentType.COMMITTEE_MINUTES]:
                speaker_segments = await self._extract_speaker_segments(
                    cleaned_content, result.language, session_metadata
                )
                result.speaker_segments = [asdict(segment) for segment in speaker_segments]
                result.preprocessing_steps.append("speaker_extraction")
            
            # Step 6: Quality assessment
            processing_quality = self._assess_processing_quality(result, structure_analysis)
            result.processing_quality = processing_quality
            result.preprocessing_steps.append("quality_assessment")
            
            # Step 7: Final validation
            if processing_quality.value in ['poor', 'unusable']:
                if len(result.extracted_content.strip()) < self.config.min_segment_length:
                    result.errors.append("Extracted content too short")
                if result.quality_metrics.get('structural_score', 0) < 0.1:
                    result.errors.append("Poor document structure")
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            
            # Update statistics
            self.processing_stats['documents_processed'] += 1
            self.processing_stats['avg_processing_time'] = (
                (self.processing_stats['avg_processing_time'] * (self.processing_stats['documents_processed'] - 1) + 
                 processing_time) / self.processing_stats['documents_processed']
            )
            
            # Cache result if enabled
            if self.cache_manager and processing_quality != ProcessingQuality.UNUSABLE:
                await self.cache_manager.set(
                    f"extraction_{content_id}",
                    asdict(result),
                    ttl_hours=24
                )
            
            # Update metrics
            self.metrics.increment('content_extractions_total')
            self.metrics.histogram('extraction_processing_time', processing_time)
            self.metrics.gauge('extraction_quality_score', result.quality_metrics.get('structural_score', 0))
            
            logger.info(f"Content extraction completed: {content_id} - "
                       f"Quality: {processing_quality.value}, Time: {processing_time:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Content extraction failed for {content_id}: {e}")
            self.processing_stats['extraction_failures'] += 1
            self.metrics.increment('content_extraction_failures')
            
            # Return minimal result with error
            result.processing_quality = ProcessingQuality.UNUSABLE
            result.errors.append(f"Extraction failed: {str(e)}")
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            return result
    
    async def extract_batch(
        self,
        content_items: List[Tuple[str, ContentType, Optional[SessionMetadata]]],
        max_concurrent: int = None
    ) -> List[ContentExtractionResult]:
        """
        Extract multiple content items concurrently
        
        Args:
            content_items: List of (content, content_type, session_metadata) tuples
            max_concurrent: Maximum concurrent extractions
            
        Returns:
            List of extraction results
        """
        max_concurrent = max_concurrent or self.config.max_workers
        
        # Create extraction tasks
        tasks = []
        for content, content_type, session_metadata in content_items:
            task = self.extract_content(content, content_type, session_metadata)
            tasks.append(task)
        
        # Process in batches to control concurrency
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch extraction failed: {result}")
                    # Create error result
                    error_result = ContentExtractionResult(
                        content_id=f"error_{int(time.time())}",
                        original_content="",
                        extracted_content="",
                        content_type=ContentType.UNKNOWN,
                        processing_quality=ProcessingQuality.UNUSABLE,
                        language=LanguageCode.UNKNOWN
                    )
                    error_result.errors.append(f"Batch processing failed: {str(result)}")
                    results.append(error_result)
                else:
                    results.append(result)
        
        logger.info(f"Batch extraction completed: {len(results)} items processed")
        return results
    
    def _detect_content_type(self, content: str) -> ContentType:
        """Detect content type from content analysis"""
        content_lower = content.lower()
        
        # Check for verbatim indicators
        verbatim_indicators = ['madam president', 'mr president', 'honourable members', 'sitting is open']
        if any(indicator in content_lower for indicator in verbatim_indicators):
            return ContentType.VERBATIM_TRANSCRIPT
        
        # Check for committee indicators
        committee_indicators = ['committee', 'rapporteur', 'opinion', 'draft report']
        if any(indicator in content_lower for indicator in committee_indicators):
            return ContentType.COMMITTEE_MINUTES
        
        # Check for amendment indicators
        amendment_indicators = ['amendment', 'amend', 'proposal for']
        if any(indicator in content_lower for indicator in amendment_indicators):
            return ContentType.AMENDMENT_TEXT
        
        # Check for voting indicators
        voting_indicators = ['voting', 'vote', 'ballot', 'in favour', 'against', 'abstentions']
        if any(indicator in content_lower for indicator in voting_indicators):
            return ContentType.VOTING_RECORD
        
        return ContentType.UNKNOWN
    
    async def _preprocess_content(
        self,
        content: str,
        content_type: ContentType,
        result: ContentExtractionResult
    ) -> str:
        """Preprocess content based on type and configuration"""
        processed = content
        
        # Remove HTML tags if configured
        if self.config.remove_html_tags:
            soup = BeautifulSoup(processed, 'html.parser')
            processed = soup.get_text()
            result.preprocessing_steps.append("html_removal")
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            processed = re.sub(r'\s+', ' ', processed)
            processed = processed.strip()
            result.preprocessing_steps.append("whitespace_normalization")
        
        # Content-type specific preprocessing
        if content_type == ContentType.VERBATIM_TRANSCRIPT:
            processed = self._preprocess_verbatim(processed)
        elif content_type == ContentType.COMMITTEE_MINUTES:
            processed = self._preprocess_minutes(processed)
        
        # Remove empty segments if configured
        if self.config.remove_empty_segments:
            lines = processed.split('\n')
            non_empty_lines = [line for line in lines if len(line.strip()) >= self.config.min_segment_length]
            processed = '\n'.join(non_empty_lines)
            result.preprocessing_steps.append("empty_segment_removal")
        
        return processed
    
    def _preprocess_verbatim(self, content: str) -> str:
        """Verbatim-specific preprocessing"""
        # Remove common verbatim artifacts
        content = re.sub(r'\(applause\)', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\(laughter\)', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\(interruption\)', '', content, flags=re.IGNORECASE)
        
        # Clean speaker indicators
        content = re.sub(r'^\s*[-–—]\s*', '', content, flags=re.MULTILINE)
        
        return content
    
    def _preprocess_minutes(self, content: str) -> str:
        """Committee minutes specific preprocessing"""
        # Remove administrative notes
        content = re.sub(r'\[.*?\]', '', content)
        
        # Clean numbering
        content = re.sub(r'^\s*\d+\.\s*', '', content, flags=re.MULTILINE)
        
        return content
    
    async def _extract_speaker_segments(
        self,
        content: str,
        language: LanguageCode,
        session_metadata: SessionMetadata = None
    ) -> List[SpeakerSegment]:
        """Extract individual speaker segments from content"""
        segments = []
        
        # Split content into potential speech segments
        # This is a simplified implementation - would need more sophisticated logic
        paragraphs = content.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < self.config.min_segment_length:
                continue
            
            # Try to extract speaker
            speaker_match = re.match(r'^([^.–\-<]*?)(?:\.|\s*–|\s*—)\s*(.*)', paragraph.strip())
            
            if speaker_match:
                raw_speaker = speaker_match.group(1).strip()
                speech_text = speaker_match.group(2).strip()
                
                if self.config.normalize_speaker_names:
                    normalized_speaker, confidence = self.speaker_normalizer.normalize_speaker(raw_speaker)
                else:
                    normalized_speaker = raw_speaker
                    confidence = 0.5
                
                segment = SpeakerSegment(
                    segment_id=f"seg_{i:04d}",
                    speaker_raw=raw_speaker,
                    speaker_normalized=normalized_speaker,
                    speech_text=speech_text,
                    language=language,
                    sequence_number=i,
                    start_position=0,  # Would calculate actual positions
                    end_position=len(speech_text),
                    confidence_score=confidence
                )
                
                # Quality indicators
                segment.quality_indicators = {
                    'has_speaker': bool(raw_speaker),
                    'sufficient_length': len(speech_text) >= self.config.min_segment_length,
                    'high_confidence': confidence >= self.config.speaker_confidence_threshold,
                    'proper_structure': '.' in speech_text or '?' in speech_text
                }
                
                segments.append(segment)
        
        # Merge continuation speeches if configured
        if self.config.merge_continuation_speeches:
            segments = self._merge_continuation_speeches(segments)
        
        return segments
    
    def _merge_continuation_speeches(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Merge speeches from the same speaker that appear consecutively"""
        if not segments:
            return segments
        
        merged = [segments[0]]
        
        for segment in segments[1:]:
            last_segment = merged[-1]
            
            # Check if same speaker and high confidence
            if (segment.speaker_normalized == last_segment.speaker_normalized and
                segment.confidence_score >= self.config.speaker_confidence_threshold and
                last_segment.confidence_score >= self.config.speaker_confidence_threshold):
                
                # Merge speeches
                last_segment.speech_text += " " + segment.speech_text
                last_segment.end_position = segment.end_position
                last_segment.processing_notes.append(f"Merged with segment {segment.segment_id}")
                
            else:
                merged.append(segment)
        
        return merged
    
    def _assess_processing_quality(
        self,
        result: ContentExtractionResult,
        structure_analysis: Dict[str, Any]
    ) -> ProcessingQuality:
        """Assess overall processing quality"""
        quality_score = 0.0
        
        # Content length factor
        content_length = len(result.extracted_content)
        if content_length >= 1000:
            quality_score += 0.2
        elif content_length >= 100:
            quality_score += 0.1
        
        # Language detection confidence
        lang_confidence = result.quality_metrics.get('language_confidence', 0.0)
        quality_score += lang_confidence * 0.2
        
        # Structural score
        structural_score = result.quality_metrics.get('structural_score', 0.0)
        quality_score += structural_score * 0.3
        
        # Speaker segments quality
        if result.speaker_segments:
            valid_segments = sum(
                1 for seg in result.speaker_segments 
                if seg.get('quality_indicators', {}).get('high_confidence', False)
            )
            speaker_quality = valid_segments / len(result.speaker_segments)
            quality_score += speaker_quality * 0.2
        
        # Error penalty
        if result.errors:
            quality_score -= len(result.errors) * 0.1
        
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Map to quality enum
        if quality_score >= 0.9:
            return ProcessingQuality.EXCELLENT
        elif quality_score >= 0.7:
            return ProcessingQuality.GOOD
        elif quality_score >= 0.5:
            return ProcessingQuality.FAIR
        elif quality_score >= 0.3:
            return ProcessingQuality.POOR
        else:
            return ProcessingQuality.UNUSABLE
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            **self.processing_stats,
            'cache_hit_rate': (
                self.processing_stats['cache_hits'] / 
                max(1, self.processing_stats['documents_processed'])
            ),
            'failure_rate': (
                self.processing_stats['extraction_failures'] / 
                max(1, self.processing_stats['documents_processed'])
            ),
            'current_config': asdict(self.config)
        }


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        config = PreprocessingConfig(
            enable_language_detection=True,
            normalize_speaker_names=True,
            quality_threshold=ProcessingQuality.FAIR
        )
        
        extractor = EnhancedContentExtractor(config)
        
        # Example content
        test_content = """
        <div class="contents">
            <h1>Tuesday, 15 January 2025 - Strasbourg</h1>
            <p class="contents">President. – Good morning. The sitting is open.</p>
            <p class="contents">García Pérez (PPE). – Madam President, I rise to speak about human rights.</p>
            <p class="contents">The situation in Myanmar remains concerning. We must take action.</p>
        </div>
        """
        
        # Extract content
        result = await extractor.extract_content(
            content=test_content,
            content_type=ContentType.VERBATIM_TRANSCRIPT
        )
        
        print(f"Extraction Result:")
        print(f"Quality: {result.processing_quality.value}")
        print(f"Language: {result.language.value}")
        print(f"Speaker segments: {len(result.speaker_segments)}")
        print(f"Processing time: {result.processing_time_ms:.1f}ms")
        
        # Get statistics
        stats = extractor.get_processing_statistics()
        print(f"Processing statistics: {json.dumps(stats, indent=2)}")
    
    asyncio.run(main())