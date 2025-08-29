#!/usr/bin/env python3
"""
Comprehensive Data Validation and Quality Scoring System

Advanced multi-dimensional quality assessment system for EU Parliament data
with intelligent scoring, validation rules, and quality improvement suggestions.

Key Features:
- Multi-dimensional quality scoring (structure, content, language, metadata)
- Intelligent validation rules with configurable thresholds
- Quality trend analysis and improvement recommendations
- Real-time quality monitoring with alerting
- Batch validation with performance optimization
- Integration with session management and error recovery
- Comprehensive quality reporting and analytics
"""

import asyncio
import json
import time
import hashlib
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set, Union, NamedTuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import logging
import re
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from langdetect import detect_langs
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from textstat import flesch_reading_ease, automated_readability_index

from ..models.speech import RawSpeechSegment
from ..models.session import SessionMetadata
from ..core.logging import get_logger
from ..core.metrics import MetricsCollector
from ..core.cache import get_cache_manager

logger = get_logger(__name__)


class QualityDimension(Enum):
    """Quality assessment dimensions"""
    STRUCTURAL = "structural"        # Document structure and formatting
    CONTENT = "content"             # Content completeness and coherence
    LINGUISTIC = "linguistic"       # Language quality and readability
    METADATA = "metadata"           # Metadata completeness and accuracy
    SPEAKER = "speaker"             # Speaker identification quality
    TEMPORAL = "temporal"           # Temporal consistency and ordering
    CONTEXTUAL = "contextual"       # Context appropriateness and relevance


class QualityLevel(Enum):
    """Overall quality levels"""
    EXCELLENT = "excellent"         # 90-100% - Production ready
    GOOD = "good"                  # 75-89% - Minor improvements needed
    ACCEPTABLE = "acceptable"       # 60-74% - Usable with some limitations
    POOR = "poor"                  # 40-59% - Significant issues present
    UNACCEPTABLE = "unacceptable"  # 0-39% - Not suitable for use


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "critical"           # Blocks processing
    HIGH = "high"                  # Major quality impact
    MEDIUM = "medium"              # Moderate quality impact
    LOW = "low"                    # Minor quality impact
    INFO = "info"                  # Informational only


class ValidationRuleType(Enum):
    """Types of validation rules"""
    MANDATORY = "mandatory"         # Must pass for acceptance
    RECOMMENDED = "recommended"     # Should pass for good quality
    OPTIONAL = "optional"          # Nice to have
    CONTEXTUAL = "contextual"      # Depends on context


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    issue_id: str
    rule_id: str
    severity: ValidationSeverity
    dimension: QualityDimension
    title: str
    description: str
    suggested_fix: Optional[str] = None
    
    # Location information
    segment_id: Optional[str] = None
    position: Optional[int] = None
    context: Optional[str] = None
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    auto_fixable: bool = False
    confidence: float = 1.0


@dataclass
class QualityScore:
    """Quality score for a specific dimension"""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    max_possible: float = 1.0
    weight: float = 1.0
    
    # Component scores
    component_scores: Dict[str, float] = field(default_factory=dict)
    
    # Quality indicators
    pass_threshold: float = 0.6
    issues_count: int = 0
    critical_issues: int = 0
    
    @property
    def weighted_score(self) -> float:
        return self.score * self.weight
    
    @property
    def passes_threshold(self) -> bool:
        return self.score >= self.pass_threshold


@dataclass
class ComprehensiveQualityAssessment:
    """Complete quality assessment result"""
    assessment_id: str
    target_type: str  # 'session', 'speech', 'document'
    target_id: str
    
    # Overall metrics
    overall_score: float
    quality_level: QualityLevel
    
    # Dimensional scores
    dimension_scores: Dict[QualityDimension, QualityScore] = field(default_factory=dict)
    
    # Validation results
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    critical_issues_count: int = 0
    total_issues_count: int = 0
    
    # Quality indicators
    passes_minimum_quality: bool = False
    recommended_for_production: bool = False
    requires_manual_review: bool = False
    
    # Improvement suggestions
    improvement_suggestions: List[str] = field(default_factory=list)
    estimated_improvement_potential: float = 0.0
    
    # Assessment metadata
    assessment_timestamp: datetime = field(default_factory=datetime.now)
    assessment_duration_ms: float = 0.0
    validator_version: str = "2.0"
    
    # Quality trends (if available)
    quality_trend: Optional[str] = None  # 'improving', 'stable', 'declining'
    historical_comparison: Optional[Dict[str, float]] = None


@dataclass
class ValidationRule:
    """Individual validation rule configuration"""
    rule_id: str
    name: str
    description: str
    dimension: QualityDimension
    severity: ValidationSeverity
    rule_type: ValidationRuleType
    
    # Rule configuration
    enabled: bool = True
    weight: float = 1.0
    threshold: Optional[float] = None
    
    # Rule logic
    validation_function: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Context filters
    applies_to_content_types: List[str] = field(default_factory=list)
    applies_to_languages: List[str] = field(default_factory=list)


class StructuralQualityAnalyzer:
    """Analyzes structural quality of documents and speeches"""
    
    def __init__(self):
        self.structural_patterns = {
            'speaker_indicators': r'(?:president|commissioner|mr|mrs|ms)\s*[.:]',
            'speech_separators': r'[.–—]\s*',
            'timestamps': r'\d{1,2}:\d{2}(?::\d{2})?',
            'interruptions': r'\([^)]*\)',
            'headings': r'^[A-Z\s]{10,}$',
            'numbering': r'^\d+\.',
        }
    
    async def analyze_structural_quality(
        self,
        content: str,
        content_type: str,
        metadata: Dict[str, Any] = None
    ) -> QualityScore:
        """Analyze structural quality of content"""
        
        component_scores = {}
        issues = []
        
        # Analyze overall structure
        structure_score = self._analyze_document_structure(content, content_type)
        component_scores['document_structure'] = structure_score
        
        # Analyze speaker structure
        speaker_score = self._analyze_speaker_structure(content)
        component_scores['speaker_structure'] = speaker_score
        
        # Analyze formatting consistency
        formatting_score = self._analyze_formatting_consistency(content)
        component_scores['formatting_consistency'] = formatting_score
        
        # Analyze content segmentation
        segmentation_score = self._analyze_content_segmentation(content)
        component_scores['content_segmentation'] = segmentation_score
        
        # Calculate overall structural score
        overall_score = statistics.mean(component_scores.values())
        
        return QualityScore(
            dimension=QualityDimension.STRUCTURAL,
            score=overall_score,
            component_scores=component_scores,
            issues_count=len(issues)
        )
    
    def _analyze_document_structure(self, content: str, content_type: str) -> float:
        """Analyze overall document structure"""
        score = 0.0
        
        # Check for basic structure elements
        if len(content.strip()) > 100:
            score += 0.2
        
        # Check for headings or sections
        headings = re.findall(self.structural_patterns['headings'], content, re.MULTILINE)
        if headings:
            score += 0.3
        
        # Check for proper paragraph structure
        paragraphs = content.split('\n\n')
        if len(paragraphs) >= 3:
            score += 0.2
        
        # Check for speaker indicators
        speakers = re.findall(self.structural_patterns['speaker_indicators'], content, re.IGNORECASE)
        if speakers:
            score += 0.3
        
        return min(1.0, score)
    
    def _analyze_speaker_structure(self, content: str) -> float:
        """Analyze speaker identification structure"""
        lines = content.split('\n')
        speaker_lines = 0
        total_content_lines = 0
        
        for line in lines:
            line = line.strip()
            if len(line) < 10:
                continue
                
            total_content_lines += 1
            
            # Check if line has speaker indicator
            if re.match(r'^[A-Za-zÀ-ÿ\s]+[.–—:]', line):
                speaker_lines += 1
        
        if total_content_lines == 0:
            return 0.0
            
        return min(1.0, speaker_lines / total_content_lines)
    
    def _analyze_formatting_consistency(self, content: str) -> float:
        """Analyze formatting consistency"""
        score = 0.5  # Base score
        
        # Check for consistent punctuation
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) > 5:
            score += 0.2
        
        # Check for consistent spacing
        excessive_spaces = len(re.findall(r'\s{3,}', content))
        if excessive_spaces < len(content) / 1000:  # Less than 1 per 1000 chars
            score += 0.2
        
        # Check for proper capitalization
        words = content.split()
        if len(words) > 10:
            proper_caps = sum(1 for word in words[:100] if word and word[0].isupper())
            if proper_caps / min(100, len(words)) > 0.1:  # At least 10% proper caps
                score += 0.1
        
        return min(1.0, score)
    
    def _analyze_content_segmentation(self, content: str) -> float:
        """Analyze how well content is segmented"""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return 0.0
        
        # Check paragraph length consistency
        lengths = [len(p) for p in paragraphs]
        if len(lengths) > 1:
            avg_length = statistics.mean(lengths)
            std_dev = statistics.stdev(lengths) if len(lengths) > 1 else 0
            
            # Good segmentation has reasonable variance
            coefficient_variation = std_dev / avg_length if avg_length > 0 else 0
            if 0.3 <= coefficient_variation <= 1.5:  # Reasonable variation
                return 0.8
        
        return 0.5


class ContentQualityAnalyzer:
    """Analyzes content quality and completeness"""
    
    def __init__(self):
        self.content_indicators = {
            'substantive_words': r'\b(?:europe|parliament|commission|council|member|states?|union|policy|law|regulation|directive)\b',
            'procedural_words': r'\b(?:amendment|vote|debate|committee|report|opinion|resolution)\b',
            'discourse_markers': r'\b(?:however|therefore|furthermore|moreover|nevertheless|consequently)\b',
        }
    
    async def analyze_content_quality(
        self,
        content: str,
        expected_content_type: str,
        metadata: Dict[str, Any] = None
    ) -> QualityScore:
        """Analyze content quality and completeness"""
        
        component_scores = {}
        
        # Analyze content completeness
        completeness_score = self._analyze_content_completeness(content, expected_content_type)
        component_scores['completeness'] = completeness_score
        
        # Analyze content coherence
        coherence_score = self._analyze_content_coherence(content)
        component_scores['coherence'] = coherence_score
        
        # Analyze content depth
        depth_score = self._analyze_content_depth(content)
        component_scores['depth'] = depth_score
        
        # Analyze content relevance
        relevance_score = self._analyze_content_relevance(content, expected_content_type)
        component_scores['relevance'] = relevance_score
        
        # Calculate overall content score
        overall_score = statistics.mean(component_scores.values())
        
        return QualityScore(
            dimension=QualityDimension.CONTENT,
            score=overall_score,
            component_scores=component_scores
        )
    
    def _analyze_content_completeness(self, content: str, content_type: str) -> float:
        """Analyze content completeness"""
        score = 0.0
        
        # Basic length check
        if len(content) > 500:
            score += 0.3
        elif len(content) > 100:
            score += 0.1
        
        # Check for expected content elements
        if content_type == 'verbatim':
            # Should have speakers and speeches
            speaker_matches = len(re.findall(r'^[A-Za-zÀ-ÿ\s]+[.–—:]', content, re.MULTILINE))
            if speaker_matches >= 3:
                score += 0.4
            elif speaker_matches >= 1:
                score += 0.2
        
        # Check for procedural content
        procedural_matches = len(re.findall(self.content_indicators['procedural_words'], content, re.IGNORECASE))
        if procedural_matches >= 3:
            score += 0.3
        elif procedural_matches >= 1:
            score += 0.1
        
        return min(1.0, score)
    
    def _analyze_content_coherence(self, content: str) -> float:
        """Analyze logical coherence of content"""
        sentences = sent_tokenize(content)
        
        if len(sentences) < 3:
            return 0.3
        
        score = 0.5  # Base score
        
        # Check for discourse markers
        discourse_markers = len(re.findall(self.content_indicators['discourse_markers'], content, re.IGNORECASE))
        if discourse_markers > 0:
            score += min(0.3, discourse_markers * 0.1)
        
        # Check sentence length variation (coherent text has varied sentence lengths)
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        if len(sentence_lengths) > 3:
            avg_length = statistics.mean(sentence_lengths)
            if 8 <= avg_length <= 25:  # Reasonable average sentence length
                score += 0.2
        
        return min(1.0, score)
    
    def _analyze_content_depth(self, content: str) -> float:
        """Analyze content depth and substance"""
        words = content.split()
        
        if len(words) < 50:
            return 0.2
        
        score = 0.3  # Base score
        
        # Check for substantive vocabulary
        substantive_matches = len(re.findall(self.content_indicators['substantive_words'], content, re.IGNORECASE))
        substantive_ratio = substantive_matches / len(words) if words else 0
        
        if substantive_ratio > 0.05:  # More than 5% substantive words
            score += 0.4
        elif substantive_ratio > 0.02:  # More than 2% substantive words
            score += 0.2
        
        # Check for complex sentences (depth indicator)
        complex_sentences = len(re.findall(r'[,;:]', content))
        if complex_sentences > len(words) / 20:  # At least 1 complex punctuation per 20 words
            score += 0.3
        
        return min(1.0, score)
    
    def _analyze_content_relevance(self, content: str, content_type: str) -> float:
        """Analyze relevance to expected content type"""
        content_lower = content.lower()
        
        relevance_indicators = {
            'verbatim': ['president', 'debate', 'amendment', 'vote', 'member states', 'european'],
            'committee': ['committee', 'rapporteur', 'opinion', 'draft report'],
            'general': ['european', 'parliament', 'commission', 'council']
        }
        
        indicators = relevance_indicators.get(content_type, relevance_indicators['general'])
        matches = sum(1 for indicator in indicators if indicator in content_lower)
        
        return min(1.0, matches / len(indicators))


class LinguisticQualityAnalyzer:
    """Analyzes linguistic quality including readability and language consistency"""
    
    def __init__(self):
        self.language_patterns = {
            'en': r'\b(?:the|and|of|to|a|in|is|it|you|that|he|was|for|on|are|as|with|his|they|i|at|be|this|have|from|or|one|had|by|word|but|not|what|all|were|we|when|your|can|said|there|each|which|she|do|how|their|if|will|up|other|about|out|many|then|them|these|so|some|her|would|make|like|into|him|has|two|more|go|no|way|could|my|than|first|been|call|who|oil|its|now|find|long|down|day|did|get|come|made|may|part)\b',
            'fr': r'\b(?:le|de|et|à|un|il|être|et|en|avoir|que|pour|dans|ce|son|une|sur|avec|ne|se|pas|tout|plus|par|grand|le|même|et|où|celui|elle|ainsi|faire|sans|power|donc|son|ces|mes|nos|vos|mais|quel|quelque|entre|chaque|pendant|depuis|selon|vers|chez|parmi|durant|contre|sous|après|avant|avec|sans|pour|dans|sur|vers|de|du|des|au|aux)\b',
            'de': r'\b(?:der|die|und|in|den|von|zu|das|mit|sich|des|auf|für|ist|im|dem|nicht|ein|eine|als|auch|es|an|werden|aus|er|hat|dass|sie|nach|wird|bei|einer|um|am|sind|noch|wie|einem|über|einen|so|zum|war|haben|nur|oder|aber|vor|zur|bis|unter|während|ohne|durch|gegen|zwischen|seit|trotz|wegen|außer|innerhalb|oberhalb|unterhalb|diesseits|jenseits|unweit|entlang|samt)\b'
        }
    
    async def analyze_linguistic_quality(
        self,
        content: str,
        expected_language: str = None,
        metadata: Dict[str, Any] = None
    ) -> QualityScore:
        """Analyze linguistic quality"""
        
        component_scores = {}
        
        # Analyze language consistency
        consistency_score = self._analyze_language_consistency(content, expected_language)
        component_scores['language_consistency'] = consistency_score
        
        # Analyze readability
        readability_score = self._analyze_readability(content)
        component_scores['readability'] = readability_score
        
        # Analyze grammar and syntax (basic)
        grammar_score = self._analyze_basic_grammar(content)
        component_scores['grammar'] = grammar_score
        
        # Analyze vocabulary quality
        vocabulary_score = self._analyze_vocabulary_quality(content)
        component_scores['vocabulary'] = vocabulary_score
        
        # Calculate overall linguistic score
        overall_score = statistics.mean(component_scores.values())
        
        return QualityScore(
            dimension=QualityDimension.LINGUISTIC,
            score=overall_score,
            component_scores=component_scores
        )
    
    def _analyze_language_consistency(self, content: str, expected_language: str = None) -> float:
        """Analyze language consistency"""
        try:
            # Detect languages in the content
            detected_langs = detect_langs(content)
            
            if not detected_langs:
                return 0.3
            
            # Get primary language
            primary_lang = detected_langs[0]
            
            # If expected language specified, check match
            if expected_language:
                if primary_lang.lang == expected_language:
                    return min(1.0, primary_lang.prob + 0.2)
                else:
                    return max(0.1, primary_lang.prob - 0.3)
            
            # Otherwise, score based on confidence
            return primary_lang.prob
            
        except Exception as e:
            logger.warning(f"Language consistency analysis failed: {e}")
            return 0.5
    
    def _analyze_readability(self, content: str) -> float:
        """Analyze text readability"""
        try:
            # Calculate readability scores
            flesch_score = flesch_reading_ease(content)
            ari_score = automated_readability_index(content)
            
            # Normalize Flesch score (0-100 to 0-1)
            flesch_normalized = flesch_score / 100.0
            
            # Normalize ARI score (assume reasonable range 5-15)
            ari_normalized = max(0.0, min(1.0, (15 - ari_score) / 10))
            
            # Average the scores
            return (flesch_normalized + ari_normalized) / 2
            
        except Exception as e:
            logger.warning(f"Readability analysis failed: {e}")
            return 0.5
    
    def _analyze_basic_grammar(self, content: str) -> float:
        """Basic grammar and syntax analysis"""
        score = 0.7  # Base score
        
        # Check for basic punctuation
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) > 2:
            score += 0.1
        
        # Check for proper capitalization at sentence start
        sentence_starts = re.findall(r'[.!?]+\s+([A-Za-z])', content)
        if sentence_starts:
            proper_caps = sum(1 for char in sentence_starts if char.isupper())
            if proper_caps / len(sentence_starts) > 0.8:
                score += 0.1
        
        # Penalty for excessive repetition
        words = content.split()
        if len(words) > 10:
            word_freq = Counter(word.lower() for word in words)
            max_freq = max(word_freq.values())
            if max_freq > len(words) / 4:  # Any word appears more than 25%
                score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    def _analyze_vocabulary_quality(self, content: str) -> float:
        """Analyze vocabulary quality and diversity"""
        words = [word.lower() for word in re.findall(r'\b[A-Za-zÀ-ÿ]+\b', content)]
        
        if len(words) < 10:
            return 0.3
        
        # Calculate vocabulary diversity
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        # Good diversity is around 0.4-0.7
        if 0.4 <= diversity_ratio <= 0.7:
            diversity_score = 1.0
        elif diversity_ratio > 0.7:
            diversity_score = 0.8  # Might be too diverse
        else:
            diversity_score = diversity_ratio / 0.4
        
        # Check for professional vocabulary (longer words)
        long_words = [word for word in words if len(word) >= 6]
        long_word_ratio = len(long_words) / len(words) if words else 0
        
        professional_score = min(1.0, long_word_ratio * 4)  # Up to 25% long words
        
        return (diversity_score + professional_score) / 2


class MetadataQualityAnalyzer:
    """Analyzes metadata quality and completeness"""
    
    def __init__(self):
        self.required_fields = {
            'session': ['session_id', 'date', 'session_type'],
            'speech': ['speaker', 'sequence_number', 'speech_text'],
            'document': ['title', 'document_type', 'content']
        }
        
        self.optional_fields = {
            'session': ['location', 'parliament_term', 'sitting_number'],
            'speech': ['language', 'duration', 'interruptions'],
            'document': ['author', 'publication_date', 'document_number']
        }
    
    async def analyze_metadata_quality(
        self,
        metadata: Dict[str, Any],
        target_type: str
    ) -> QualityScore:
        """Analyze metadata quality and completeness"""
        
        component_scores = {}
        
        # Analyze completeness
        completeness_score = self._analyze_metadata_completeness(metadata, target_type)
        component_scores['completeness'] = completeness_score
        
        # Analyze accuracy
        accuracy_score = self._analyze_metadata_accuracy(metadata, target_type)
        component_scores['accuracy'] = accuracy_score
        
        # Analyze consistency
        consistency_score = self._analyze_metadata_consistency(metadata)
        component_scores['consistency'] = consistency_score
        
        # Calculate overall metadata score
        overall_score = statistics.mean(component_scores.values())
        
        return QualityScore(
            dimension=QualityDimension.METADATA,
            score=overall_score,
            component_scores=component_scores
        )
    
    def _analyze_metadata_completeness(self, metadata: Dict[str, Any], target_type: str) -> float:
        """Analyze metadata completeness"""
        required = self.required_fields.get(target_type, [])
        optional = self.optional_fields.get(target_type, [])
        
        if not required:
            return 0.5  # Unknown type
        
        # Check required fields
        required_present = sum(1 for field in required if field in metadata and metadata[field])
        required_score = required_present / len(required) if required else 0
        
        # Check optional fields (bonus)
        optional_present = sum(1 for field in optional if field in metadata and metadata[field])
        optional_score = optional_present / len(optional) if optional else 0
        
        # Weighted combination (required is 80%, optional is 20%)
        return required_score * 0.8 + optional_score * 0.2
    
    def _analyze_metadata_accuracy(self, metadata: Dict[str, Any], target_type: str) -> float:
        """Analyze metadata accuracy"""
        score = 0.8  # Base score
        
        # Check date formats
        if 'date' in metadata:
            date_value = metadata['date']
            if isinstance(date_value, str):
                # Try to parse common date formats
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{1,2}\s+\w+\s+\d{4}'  # DD Month YYYY
                ]
                
                if any(re.match(pattern, date_value) for pattern in date_patterns):
                    score += 0.1
                else:
                    score -= 0.2
        
        # Check ID formats
        if 'session_id' in metadata:
            session_id = metadata['session_id']
            if isinstance(session_id, str) and len(session_id) > 3:
                score += 0.05
        
        # Check numeric fields
        for field in ['sequence_number', 'sitting_number']:
            if field in metadata:
                value = metadata[field]
                if isinstance(value, (int, float)) and value > 0:
                    score += 0.05
                elif isinstance(value, str) and value.isdigit():
                    score += 0.03
        
        return min(1.0, score)
    
    def _analyze_metadata_consistency(self, metadata: Dict[str, Any]) -> float:
        """Analyze internal metadata consistency"""
        score = 0.7  # Base score
        
        # Check for consistency between related fields
        inconsistencies = 0
        
        # Date consistency
        if 'date' in metadata and 'parliament_term' in metadata:
            # Simple consistency check - could be more sophisticated
            try:
                date_str = str(metadata['date'])
                year = int(date_str[:4]) if len(date_str) >= 4 else None
                term = metadata['parliament_term']
                
                if year and isinstance(term, (int, str)):
                    # EU Parliament terms are roughly 5 years
                    # This is a simplified check
                    if year < 2000 or year > 2030:
                        inconsistencies += 1
                        
            except (ValueError, TypeError):
                inconsistencies += 1
        
        # Apply penalty for inconsistencies
        score -= inconsistencies * 0.2
        
        return max(0.0, score)


class ComprehensiveQualityValidator:
    """Main comprehensive quality validation system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Quality analyzers
        self.structural_analyzer = StructuralQualityAnalyzer()
        self.content_analyzer = ContentQualityAnalyzer()
        self.linguistic_analyzer = LinguisticQualityAnalyzer()
        self.metadata_analyzer = MetadataQualityAnalyzer()
        
        # Performance optimization
        self.cache_manager = get_cache_manager()
        self.metrics = MetricsCollector()
        
        # Validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        # Quality thresholds
        self.quality_thresholds = {
            'minimum_acceptable': 0.4,
            'production_ready': 0.75,
            'excellent': 0.9,
            'critical_issue_threshold': 0.3
        }
        
        # Dimension weights
        self.dimension_weights = {
            QualityDimension.STRUCTURAL: 0.25,
            QualityDimension.CONTENT: 0.30,
            QualityDimension.LINGUISTIC: 0.20,
            QualityDimension.METADATA: 0.15,
            QualityDimension.SPEAKER: 0.10
        }
        
        # Processing statistics
        self.validation_stats = {
            'assessments_performed': 0,
            'critical_issues_found': 0,
            'avg_assessment_time': 0.0,
            'quality_distribution': defaultdict(int)
        }
        
        logger.info("Comprehensive Quality Validator initialized")
    
    async def assess_comprehensive_quality(
        self,
        target_data: Any,
        target_type: str,
        target_id: str,
        context: Dict[str, Any] = None
    ) -> ComprehensiveQualityAssessment:
        """
        Perform comprehensive quality assessment
        
        Args:
            target_data: Data to assess (speech, session, document)
            target_type: Type of target ('session', 'speech', 'document')
            target_id: Unique identifier for target
            context: Additional context for assessment
            
        Returns:
            Complete quality assessment with scores and recommendations
        """
        start_time = time.time()
        context = context or {}
        
        # Generate assessment ID
        assessment_id = f"qa_{target_type}_{hashlib.md5(f'{target_id}_{int(start_time)}'.encode()).hexdigest()[:12]}"
        
        logger.info(f"Starting comprehensive quality assessment: {assessment_id}")
        
        try:
            # Initialize assessment
            assessment = ComprehensiveQualityAssessment(
                assessment_id=assessment_id,
                target_type=target_type,
                target_id=target_id,
                overall_score=0.0,
                quality_level=QualityLevel.UNACCEPTABLE
            )
            
            # Extract content and metadata based on target type
            content, metadata = self._extract_content_and_metadata(target_data, target_type)
            
            # Perform dimensional analysis
            dimension_scores = await self._perform_dimensional_analysis(
                content, metadata, target_type, context
            )
            assessment.dimension_scores = dimension_scores
            
            # Perform validation checks
            validation_issues = await self._perform_validation_checks(
                target_data, content, metadata, target_type, context
            )
            assessment.validation_issues = validation_issues
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_quality_score(dimension_scores, validation_issues)
            assessment.overall_score = overall_score
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score, validation_issues)
            assessment.quality_level = quality_level
            
            # Generate quality indicators
            self._set_quality_indicators(assessment, dimension_scores, validation_issues)
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(
                dimension_scores, validation_issues, target_type
            )
            assessment.improvement_suggestions = improvement_suggestions
            
            # Calculate improvement potential
            assessment.estimated_improvement_potential = self._calculate_improvement_potential(
                dimension_scores, validation_issues
            )
            
            # Add quality trend if historical data available
            if context.get('enable_trend_analysis'):
                assessment.quality_trend = await self._analyze_quality_trend(target_id, overall_score)
            
            # Record processing time
            assessment.assessment_duration_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_validation_statistics(assessment)
            
            # Cache assessment if enabled
            if self.cache_manager:
                await self.cache_manager.set(
                    f"assessment_{assessment_id}",
                    asdict(assessment),
                    ttl_hours=24
                )
            
            logger.info(f"Quality assessment completed: {assessment_id} - "
                       f"Score: {overall_score:.3f}, Level: {quality_level.value}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Quality assessment failed for {assessment_id}: {e}")
            
            # Create error assessment
            error_assessment = ComprehensiveQualityAssessment(
                assessment_id=assessment_id,
                target_type=target_type,
                target_id=target_id,
                overall_score=0.0,
                quality_level=QualityLevel.UNACCEPTABLE,
                assessment_duration_ms=(time.time() - start_time) * 1000
            )
            
            error_assessment.validation_issues.append(
                ValidationIssue(
                    issue_id=f"error_{int(time.time())}",
                    rule_id="SYSTEM_ERROR",
                    severity=ValidationSeverity.CRITICAL,
                    dimension=QualityDimension.STRUCTURAL,
                    title="Assessment System Error",
                    description=f"Quality assessment failed: {str(e)}",
                    auto_fixable=False
                )
            )
            
            return error_assessment
    
    async def assess_batch(
        self,
        targets: List[Tuple[Any, str, str, Dict[str, Any]]],
        max_concurrent: int = 5
    ) -> List[ComprehensiveQualityAssessment]:
        """
        Assess multiple targets concurrently
        
        Args:
            targets: List of (target_data, target_type, target_id, context) tuples
            max_concurrent: Maximum concurrent assessments
            
        Returns:
            List of quality assessments
        """
        logger.info(f"Starting batch quality assessment: {len(targets)} targets")
        
        # Create assessment tasks
        tasks = []
        for target_data, target_type, target_id, context in targets:
            task = self.assess_comprehensive_quality(target_data, target_type, target_id, context)
            tasks.append(task)
        
        # Process in batches
        results = []
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch assessment failed: {result}")
                    # Create minimal error assessment
                    error_assessment = ComprehensiveQualityAssessment(
                        assessment_id=f"batch_error_{int(time.time())}",
                        target_type="unknown",
                        target_id="unknown",
                        overall_score=0.0,
                        quality_level=QualityLevel.UNACCEPTABLE
                    )
                    results.append(error_assessment)
                else:
                    results.append(result)
        
        logger.info(f"Batch assessment completed: {len(results)} assessments")
        return results
    
    def _extract_content_and_metadata(self, target_data: Any, target_type: str) -> Tuple[str, Dict[str, Any]]:
        """Extract content and metadata from target data"""
        if target_type == 'speech':
            if hasattr(target_data, 'speech_text'):
                content = target_data.speech_text
                metadata = {
                    'speaker': getattr(target_data, 'speaker_raw', ''),
                    'sequence_number': getattr(target_data, 'sequence_number', 0),
                    'language': getattr(target_data, 'language', ''),
                    'parsing_metadata': getattr(target_data, 'parsing_metadata', {})
                }
            else:
                content = str(target_data)
                metadata = {}
                
        elif target_type == 'session':
            if hasattr(target_data, 'speeches'):
                # Combine all speeches
                speeches = target_data.speeches
                content = '\n\n'.join(speech.speech_text for speech in speeches if hasattr(speech, 'speech_text'))
                metadata = {
                    'session_id': getattr(target_data, 'session_id', ''),
                    'date': getattr(target_data, 'date', ''),
                    'session_type': getattr(target_data, 'session_type', ''),
                    'speeches_count': len(speeches)
                }
            else:
                content = str(target_data)
                metadata = {}
                
        elif target_type == 'document':
            if hasattr(target_data, 'content'):
                content = target_data.content
                metadata = {
                    'title': getattr(target_data, 'title', ''),
                    'document_type': getattr(target_data, 'document_type', ''),
                    'author': getattr(target_data, 'author', ''),
                    'publication_date': getattr(target_data, 'publication_date', '')
                }
            else:
                content = str(target_data)
                metadata = {}
        else:
            content = str(target_data)
            metadata = {}
        
        return content, metadata
    
    async def _perform_dimensional_analysis(
        self,
        content: str,
        metadata: Dict[str, Any],
        target_type: str,
        context: Dict[str, Any]
    ) -> Dict[QualityDimension, QualityScore]:
        """Perform analysis across all quality dimensions"""
        dimension_scores = {}
        
        # Structural analysis
        structural_score = await self.structural_analyzer.analyze_structural_quality(
            content, target_type, metadata
        )
        dimension_scores[QualityDimension.STRUCTURAL] = structural_score
        
        # Content analysis
        content_score = await self.content_analyzer.analyze_content_quality(
            content, target_type, metadata
        )
        dimension_scores[QualityDimension.CONTENT] = content_score
        
        # Linguistic analysis
        linguistic_score = await self.linguistic_analyzer.analyze_linguistic_quality(
            content, context.get('expected_language'), metadata
        )
        dimension_scores[QualityDimension.LINGUISTIC] = linguistic_score
        
        # Metadata analysis
        metadata_score = await self.metadata_analyzer.analyze_metadata_quality(
            metadata, target_type
        )
        dimension_scores[QualityDimension.METADATA] = metadata_score
        
        return dimension_scores
    
    async def _perform_validation_checks(
        self,
        target_data: Any,
        content: str,
        metadata: Dict[str, Any],
        target_type: str,
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Perform comprehensive validation checks"""
        issues = []
        
        # Apply validation rules
        for rule in self.validation_rules:
            if not rule.enabled:
                continue
                
            # Check if rule applies to this target type
            if rule.applies_to_content_types and target_type not in rule.applies_to_content_types:
                continue
            
            # Perform validation check
            try:
                rule_issues = await self._apply_validation_rule(
                    rule, target_data, content, metadata, target_type, context
                )
                issues.extend(rule_issues)
                
            except Exception as e:
                logger.warning(f"Validation rule {rule.rule_id} failed: {e}")
        
        return issues
    
    async def _apply_validation_rule(
        self,
        rule: ValidationRule,
        target_data: Any,
        content: str,
        metadata: Dict[str, Any],
        target_type: str,
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Apply a specific validation rule"""
        issues = []
        
        # This is a simplified implementation
        # In practice, each rule would have specific validation logic
        
        if rule.rule_id == "CONTENT_MIN_LENGTH":
            min_length = rule.parameters.get('min_length', 50)
            if len(content) < min_length:
                issues.append(ValidationIssue(
                    issue_id=f"{rule.rule_id}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    dimension=rule.dimension,
                    title="Content too short",
                    description=f"Content length {len(content)} is below minimum {min_length}",
                    suggested_fix="Ensure content has sufficient detail and substance",
                    auto_fixable=False
                ))
        
        elif rule.rule_id == "SPEAKER_PRESENT":
            if target_type == 'speech':
                speaker = metadata.get('speaker', '').strip()
                if not speaker:
                    issues.append(ValidationIssue(
                        issue_id=f"{rule.rule_id}_{int(time.time())}",
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        dimension=rule.dimension,
                        title="Missing speaker information",
                        description="Speech segment does not have speaker identification",
                        suggested_fix="Add speaker identification to the segment",
                        auto_fixable=False
                    ))
        
        elif rule.rule_id == "METADATA_COMPLETENESS":
            required_fields = rule.parameters.get('required_fields', [])
            missing_fields = [field for field in required_fields if field not in metadata or not metadata[field]]
            
            if missing_fields:
                issues.append(ValidationIssue(
                    issue_id=f"{rule.rule_id}_{int(time.time())}",
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    dimension=rule.dimension,
                    title="Incomplete metadata",
                    description=f"Missing required fields: {', '.join(missing_fields)}",
                    suggested_fix=f"Add the missing metadata fields: {', '.join(missing_fields)}",
                    auto_fixable=False
                ))
        
        return issues
    
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize default validation rules"""
        rules = [
            ValidationRule(
                rule_id="CONTENT_MIN_LENGTH",
                name="Minimum Content Length",
                description="Content must meet minimum length requirements",
                dimension=QualityDimension.CONTENT,
                severity=ValidationSeverity.HIGH,
                rule_type=ValidationRuleType.MANDATORY,
                parameters={'min_length': 50}
            ),
            
            ValidationRule(
                rule_id="SPEAKER_PRESENT",
                name="Speaker Present",
                description="Speech segments must have speaker identification",
                dimension=QualityDimension.SPEAKER,
                severity=ValidationSeverity.HIGH,
                rule_type=ValidationRuleType.MANDATORY,
                applies_to_content_types=['speech']
            ),
            
            ValidationRule(
                rule_id="METADATA_COMPLETENESS",
                name="Metadata Completeness",
                description="Required metadata fields must be present",
                dimension=QualityDimension.METADATA,
                severity=ValidationSeverity.MEDIUM,
                rule_type=ValidationRuleType.RECOMMENDED,
                parameters={'required_fields': ['session_id', 'date']}
            ),
            
            ValidationRule(
                rule_id="LANGUAGE_CONSISTENCY",
                name="Language Consistency",
                description="Content should be in consistent language",
                dimension=QualityDimension.LINGUISTIC,
                severity=ValidationSeverity.MEDIUM,
                rule_type=ValidationRuleType.RECOMMENDED
            ),
            
            ValidationRule(
                rule_id="STRUCTURAL_INTEGRITY",
                name="Structural Integrity",
                description="Document structure should be well-formed",
                dimension=QualityDimension.STRUCTURAL,
                severity=ValidationSeverity.MEDIUM,
                rule_type=ValidationRuleType.RECOMMENDED
            )
        ]
        
        return rules
    
    def _calculate_overall_quality_score(
        self,
        dimension_scores: Dict[QualityDimension, QualityScore],
        validation_issues: List[ValidationIssue]
    ) -> float:
        """Calculate overall quality score from dimensional scores and issues"""
        
        # Calculate weighted dimensional score
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = self.dimension_weights.get(dimension, 1.0)
            total_weighted_score += score.score * weight
            total_weight += weight
        
        base_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Apply penalties for validation issues
        critical_issues = [issue for issue in validation_issues if issue.severity == ValidationSeverity.CRITICAL]
        high_issues = [issue for issue in validation_issues if issue.severity == ValidationSeverity.HIGH]
        medium_issues = [issue for issue in validation_issues if issue.severity == ValidationSeverity.MEDIUM]
        
        # Calculate penalties
        penalty = 0.0
        penalty += len(critical_issues) * 0.3    # Critical issues: -30% each
        penalty += len(high_issues) * 0.15       # High issues: -15% each
        penalty += len(medium_issues) * 0.05     # Medium issues: -5% each
        
        # Apply penalty
        final_score = max(0.0, base_score - penalty)
        
        return final_score
    
    def _determine_quality_level(
        self,
        overall_score: float,
        validation_issues: List[ValidationIssue]
    ) -> QualityLevel:
        """Determine quality level based on score and issues"""
        
        # Check for critical issues first
        critical_issues = [issue for issue in validation_issues if issue.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            return QualityLevel.UNACCEPTABLE
        
        # Determine level based on score
        if overall_score >= 0.90:
            return QualityLevel.EXCELLENT
        elif overall_score >= 0.75:
            return QualityLevel.GOOD
        elif overall_score >= 0.60:
            return QualityLevel.ACCEPTABLE
        elif overall_score >= 0.40:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE
    
    def _set_quality_indicators(
        self,
        assessment: ComprehensiveQualityAssessment,
        dimension_scores: Dict[QualityDimension, QualityScore],
        validation_issues: List[ValidationIssue]
    ):
        """Set quality indicators for the assessment"""
        
        # Count issues by severity
        assessment.critical_issues_count = len([i for i in validation_issues if i.severity == ValidationSeverity.CRITICAL])
        assessment.total_issues_count = len(validation_issues)
        
        # Set quality indicators
        assessment.passes_minimum_quality = assessment.overall_score >= self.quality_thresholds['minimum_acceptable']
        assessment.recommended_for_production = (
            assessment.overall_score >= self.quality_thresholds['production_ready'] and
            assessment.critical_issues_count == 0
        )
        assessment.requires_manual_review = (
            assessment.critical_issues_count > 0 or
            assessment.overall_score < self.quality_thresholds['minimum_acceptable'] or
            len([i for i in validation_issues if i.severity == ValidationSeverity.HIGH]) > 5
        )
    
    def _generate_improvement_suggestions(
        self,
        dimension_scores: Dict[QualityDimension, QualityScore],
        validation_issues: List[ValidationIssue],
        target_type: str
    ) -> List[str]:
        """Generate improvement suggestions based on analysis"""
        
        suggestions = []
        
        # Suggestions based on low dimensional scores
        for dimension, score in dimension_scores.items():
            if score.score < 0.6:
                if dimension == QualityDimension.STRUCTURAL:
                    suggestions.append("Improve document structure with clear headings and proper formatting")
                elif dimension == QualityDimension.CONTENT:
                    suggestions.append("Enhance content completeness and coherence")
                elif dimension == QualityDimension.LINGUISTIC:
                    suggestions.append("Improve language quality and readability")
                elif dimension == QualityDimension.METADATA:
                    suggestions.append("Complete missing metadata fields")
                elif dimension == QualityDimension.SPEAKER:
                    suggestions.append("Improve speaker identification and normalization")
        
        # Suggestions based on validation issues
        issue_types = Counter(issue.dimension for issue in validation_issues)
        
        for dimension, count in issue_types.items():
            if count > 2:  # Multiple issues in same dimension
                suggestions.append(f"Address multiple {dimension.value} issues ({count} found)")
        
        # Critical issue suggestions
        critical_issues = [issue for issue in validation_issues if issue.severity == ValidationSeverity.CRITICAL]
        for issue in critical_issues:
            if issue.suggested_fix:
                suggestions.append(f"CRITICAL: {issue.suggested_fix}")
        
        return suggestions[:10]  # Limit to top 10 suggestions
    
    def _calculate_improvement_potential(
        self,
        dimension_scores: Dict[QualityDimension, QualityScore],
        validation_issues: List[ValidationIssue]
    ) -> float:
        """Calculate estimated improvement potential"""
        
        # Calculate potential improvement from dimensional scores
        total_potential = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = self.dimension_weights.get(dimension, 1.0)
            # Potential improvement is the gap to perfect score
            potential = (1.0 - score.score) * weight
            total_potential += potential
            total_weight += weight
        
        dimensional_potential = total_potential / total_weight if total_weight > 0 else 0.0
        
        # Add potential from fixing validation issues
        fixable_issues = [issue for issue in validation_issues if issue.auto_fixable]
        issue_potential = min(0.3, len(fixable_issues) * 0.05)  # Up to 30% from fixing issues
        
        return min(1.0, dimensional_potential + issue_potential)
    
    async def _analyze_quality_trend(self, target_id: str, current_score: float) -> str:
        """Analyze quality trend for the target"""
        try:
            # This would typically query historical data
            # For now, return a placeholder
            return "stable"
        except Exception:
            return None
    
    def _update_validation_statistics(self, assessment: ComprehensiveQualityAssessment):
        """Update validation statistics"""
        self.validation_stats['assessments_performed'] += 1
        self.validation_stats['critical_issues_found'] += assessment.critical_issues_count
        
        # Update average assessment time
        prev_avg = self.validation_stats['avg_assessment_time']
        count = self.validation_stats['assessments_performed']
        new_avg = ((prev_avg * (count - 1)) + assessment.assessment_duration_ms) / count
        self.validation_stats['avg_assessment_time'] = new_avg
        
        # Update quality distribution
        self.validation_stats['quality_distribution'][assessment.quality_level.value] += 1
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        total_assessments = self.validation_stats['assessments_performed']
        
        return {
            **self.validation_stats,
            'critical_issue_rate': (
                self.validation_stats['critical_issues_found'] / max(1, total_assessments)
            ),
            'quality_level_distribution': dict(self.validation_stats['quality_distribution']),
            'current_thresholds': self.quality_thresholds,
            'dimension_weights': {dim.value: weight for dim, weight in self.dimension_weights.items()}
        }


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        validator = ComprehensiveQualityValidator()
        
        # Example speech data
        class MockSpeech:
            def __init__(self):
                self.speech_text = "Mr. President, I rise to speak about the important matter of European integration. The situation requires our immediate attention."
                self.speaker_raw = "García Pérez (PPE)"
                self.sequence_number = 1
                self.language = "en"
                self.parsing_metadata = {}
        
        # Assess quality
        speech = MockSpeech()
        assessment = await validator.assess_comprehensive_quality(
            target_data=speech,
            target_type="speech",
            target_id="test_speech_001",
            context={'expected_language': 'en'}
        )
        
        print(f"Quality Assessment Results:")
        print(f"Overall Score: {assessment.overall_score:.3f}")
        print(f"Quality Level: {assessment.quality_level.value}")
        print(f"Issues Found: {assessment.total_issues_count}")
        print(f"Critical Issues: {assessment.critical_issues_count}")
        print(f"Improvement Potential: {assessment.estimated_improvement_potential:.3f}")
        
        print(f"\nDimensional Scores:")
        for dimension, score in assessment.dimension_scores.items():
            print(f"  {dimension.value}: {score.score:.3f}")
        
        print(f"\nImprovement Suggestions:")
        for suggestion in assessment.improvement_suggestions[:5]:
            print(f"  - {suggestion}")
        
        # Get statistics
        stats = validator.get_validation_statistics()
        print(f"\nValidation Statistics:")
        print(json.dumps(stats, indent=2, default=str))
    
    asyncio.run(main())