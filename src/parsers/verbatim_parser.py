"""Parser for EU Parliament verbatim reports with speech segment extraction."""

import re
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass

from ..models.speech import RawSpeechSegment, SpeechSegment
from ..core.exceptions import ParsingError
from ..core.logging import get_logger
from ..utils.text_utils import clean_html_text, extract_speaker_info, normalize_speaker_name
from ..utils.time_utils import parse_timestamp, interpolate_timestamps

logger = get_logger(__name__)


@dataclass
class ParsedSegment:
    """Intermediate representation of a parsed speech segment."""
    speaker_text: str
    speech_content: str
    timestamp_hint: Optional[str] = None
    segment_position: int = 0
    confidence: float = 1.0
    is_procedural: bool = False


class VerbatimParser:
    """Enhanced parser for EU Parliament verbatim reports."""
    
    # Enhanced speaker patterns with better Unicode support and context
    SPEAKER_PATTERNS = [
        # Full name with title and party: "President García Pérez (PPE). –"
        r'^\s*([^\(]{3,50})\s*\(([^)]+)\)\s*\.\s*[-–−]\s*',
        # Name with party: "García Pérez (PPE) –" or "García Pérez (PPE). −"
        r'^\s*([^\(]{3,50})\s*\(([^)]+)\)\s*\.?\s*[-–−]\s*',
        # Simple name with period: "García Pérez. –"
        r'^\s*([^\.\(]{3,50})\s*\.\s*[-–−]\s*',
        # Name with dash: "García Pérez –"
        r'^\s*([^-–−\(]{3,50})\s*[-–−]\s*',
        # Name with colon: "García Pérez:" or "Name (Party):"
        r'^\s*([^:]{3,80})\s*:\s*',
        # Bold/strong HTML tags: "<strong>García Pérez</strong>:"
        r'^\s*<(?:strong|b)>\s*([^<]+?)\s*</(?:strong|b)>\s*[:\-–−]?\s*',
        # President specific patterns
        r'^\s*((?:The\s+)?President(?:\s+of\s+the\s+European\s+Parliament)?)\s*[.\-–−:]\s*',
        # Vice-President patterns
        r'^\s*(Vice-President\s+[A-Za-züäöüúéèàâêîôûç\-\'\s]+)\s*[.\-–−:]\s*',
    ]
    
    # Enhanced procedural patterns with more comprehensive coverage
    PROCEDURAL_PATTERNS = [
        # Session management
        r'^(?:The\s+)?(?:sitting|session|meeting)\s+(?:is\s+)?(?:opened|closed|suspended|resumed|adjourned)',
        r'^(?:The\s+)?debate\s+(?:is\s+)?(?:opened|closed|suspended)',
        r'^(?:We\s+)?(?:shall|will)\s+(?:now\s+)?(?:proceed|continue|move)',
        
        # Presidential announcements
        r'^(?:Madam|Mr)\s+President(?:\s+of\s+the\s+Commission)?',
        r'^(?:The\s+)?President\b',
        r'^Ladies\s+and\s+gentlemen',
        r'^Honourable\s+Members',
        r'^Dear\s+colleagues',
        
        # Voting and procedures
        r'^Voting\s+time',
        r'^(?:The\s+)?vote\s+(?:is\s+)?(?:taken|called|conducted)',
        r'^Question\s+Time',
        r'^Written\s+statements?',
        r'^Procedures?\s+without\s+debate',
        r'^Explanations?\s+of\s+votes?',
        r'^Corrections?\s+to\s+votes?',
        
        # Agenda and order
        r'^Next\s+item(?:\s+on\s+the\s+agenda)?',
        r'^Order\s+of\s+business',
        r'^(?:The\s+)?agenda\s+(?:is\s+)?(?:adopted|approved)',
        r'^(?:That\s+)?concludes\s+(?:the\s+)?(?:debate|item)',
        
        # Interruptions and reactions
        r'^\([^)]*(?:applause|laughter|protests?|shouts?|interruption)[^)]*\)',
        r'^\([^)]*(?:mixed\s+reactions|uproar|noise)[^)]*\)',
        r'^\((?:The\s+)?sitting\s+(?:was\s+)?(?:suspended|interrupted)',
        
        # Time announcements
        r'^(?:The\s+)?sitting\s+(?:opened|closed|suspended|resumed)\s+at\s+\d',
        r'^It\s+is\s+now\s+\d{1,2}[:.]\d{2}',
        r'^At\s+\d{1,2}[:.]\d{2}',
        
        # Parliamentary procedures
        r'^I\s+call\s+(?:upon|on)\s+(?:the\s+)?(?:Commission|Council|rapporteur)',
        r'^Point\s+of\s+order',
        r'^Blue\s+card\s+question',
        r'^(?:Catch-the-eye|Blue-card)\s+procedure',
    ]
    
    # Enhanced time patterns with European formats
    TIME_PATTERNS = [
        r'\b(\d{1,2}[:.]\d{2})\s*(?:hrs?|hours?)?\b',  # 14:30 hrs
        r'\b(\d{1,2}h\d{2})\b',  # 14h30 (European format)
        r'\((The\s+sitting\s+(?:opened|closed|suspended|resumed)\s+at\s+(\d{1,2}[:.]\d{2}))\)',
        r'at\s+(\d{1,2}[:.]\d{2})',
        r'\b(\d{1,2}[:.]\d{2})\s*([ap]\.?m\.?)\b',  # 2:30 p.m.
        r'(?:opened|closed|suspended|resumed)\s+at\s+(\d{1,2}[:.]\d{2})',
    ]
    
    def __init__(self):
        """Initialize enhanced verbatim parser."""
        self.current_session_id: Optional[str] = None
        self.session_date: Optional[datetime] = None
        self.parsed_segments: List[ParsedSegment] = []
        self.parsing_stats = {
            'total_elements': 0,
            'speaker_patterns_matched': 0,
            'procedural_detected': 0,
            'timestamps_found': 0,
            'validation_failures': 0
        }
        
        logger.info("Enhanced verbatim parser initialized")
    
    def parse_verbatim_report(self, html_content: str, session_id: str, 
                             session_date: datetime) -> List[RawSpeechSegment]:
        """
        Parse verbatim report HTML with enhanced pattern matching.
        
        Args:
            html_content: Raw HTML content of verbatim report
            session_id: Session identifier
            session_date: Session date for timestamp context
            
        Returns:
            List of raw speech segments with enhanced parsing
        """
        self.current_session_id = session_id
        self.session_date = session_date
        self.parsed_segments = []
        self.parsing_stats = {k: 0 for k in self.parsing_stats}
        
        logger.info(
            "Starting enhanced verbatim parsing",
            session_id=session_id,
            content_length=len(html_content)
        )
        
        try:
            # Parse HTML with enhanced error handling
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract main content with multiple fallback strategies
            content_elements = self._extract_content_elements_enhanced(soup)
            self.parsing_stats['total_elements'] = len(content_elements)
            
            # Parse speech segments with context awareness
            previous_speaker = None
            speech_context = {'in_procedural_block': False, 'last_timestamp': None}
            
            for i, element in enumerate(content_elements):
                segment = self._parse_speech_element_enhanced(element, previous_speaker, speech_context)
                if segment:
                    self.parsed_segments.append(segment)
                    if not segment.is_procedural:
                        previous_speaker = segment.speaker_text
            
            # Post-process segments for quality enhancement
            self._post_process_segments()
            
            # Convert to RawSpeechSegment objects
            raw_segments = self._convert_to_raw_segments_enhanced()
            
            logger.info(
                "Enhanced verbatim parsing completed",
                session_id=session_id,
                segments_extracted=len(raw_segments),
                stats=self.parsing_stats
            )
            
            return raw_segments
            
        except Exception as e:
            logger.error("Enhanced verbatim parsing failed", error=str(e), session_id=session_id)
            raise ParsingError(f"Failed to parse verbatim report: {e}")
    
    def _extract_content_elements_enhanced(self, soup: BeautifulSoup) -> List[Tag]:
        """Extract content elements with enhanced strategies."""
        content_elements = []
        
        # Multiple fallback strategies for finding content
        content_selectors = [
            ('div', {'class': 'contents'}),
            ('div', {'class': 'report-content'}),
            ('div', {'class': 'verbatim-content'}),
            ('div', {'class': 'doceo-content'}),  # New selector for EU docs
            ('div', {'id': 'content'}),
            ('main', {}),
            ('div', {'class': 'document-content'}),
            ('body', {})
        ]
        
        main_content = None
        for tag, attrs in content_selectors:
            main_content = soup.find(tag, attrs) if attrs else soup.find(tag)
            if main_content:
                logger.debug("Found main content", selector=f"{tag}({attrs})")
                break
        
        if not main_content:
            logger.warning("No main content container found, using body")
            main_content = soup.find('body') or soup
        
        # Extract elements with better filtering
        candidate_elements = main_content.find_all(['p', 'div', 'section', 'span'], recursive=True)
        
        for element in candidate_elements:
            text_content = element.get_text(strip=True)
            
            # Enhanced filtering criteria
            if self._is_content_element(element, text_content):
                content_elements.append(element)
        
        logger.debug("Content elements extracted", count=len(content_elements))
        return content_elements
    
    def _is_content_element(self, element: Tag, text_content: str) -> bool:
        """Determine if an element contains relevant content."""
        if not text_content or len(text_content) < 5:
            return False
        
        # Skip navigation, menu, and metadata elements
        skip_classes = {'nav', 'menu', 'header', 'footer', 'sidebar', 'metadata', 
                       'breadcrumb', 'pagination', 'share', 'social'}
        element_classes = set(element.get('class', []))
        if element_classes & skip_classes:
            return False
        
        # Skip elements that are clearly not content
        if element.name in ['script', 'style', 'nav', 'header', 'footer']:
            return False
        
        # Must have meaningful text length
        if len(text_content) < 10:
            return False
        
        # Skip elements with excessive punctuation (likely metadata)
        punct_ratio = sum(1 for c in text_content if c in '.,;:!?()[]{}') / len(text_content)
        if punct_ratio > 0.3:
            return False
        
        return True
    
    def _parse_speech_element_enhanced(self, element: Tag, previous_speaker: str, 
                                     context: Dict[str, Any]) -> Optional[ParsedSegment]:
        """Parse speech element with enhanced pattern matching."""
        text_content = element.get_text(strip=True)
        
        if not text_content:
            return None
        
        # Extract timestamp hints early
        timestamp_hint = self._extract_timestamp_hint_enhanced(text_content)
        if timestamp_hint:
            context['last_timestamp'] = timestamp_hint
            self.parsing_stats['timestamps_found'] += 1
        
        # Enhanced speaker extraction
        speaker_match = self._extract_speaker_enhanced(text_content, element)
        
        if speaker_match:
            speaker_name, remaining_text, speaker_info = speaker_match
            self.parsing_stats['speaker_patterns_matched'] += 1
            
            # Determine if this is procedural with enhanced logic
            is_procedural = self._is_procedural_content_enhanced(text_content, speaker_name)
            if is_procedural:
                self.parsing_stats['procedural_detected'] += 1
                context['in_procedural_block'] = True
            else:
                context['in_procedural_block'] = False
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_parsing_confidence(speaker_name, remaining_text, 
                                                          speaker_info, is_procedural)
            
            return ParsedSegment(
                speaker_text=speaker_name,
                speech_content=remaining_text,
                timestamp_hint=timestamp_hint or context.get('last_timestamp'),
                segment_position=len(self.parsed_segments),
                is_procedural=is_procedural,
                confidence=confidence
            )
        
        # Handle continuation of previous speech
        elif (not context.get('in_procedural_block', False) and 
              self.parsed_segments and 
              not self._looks_like_new_speaker(text_content)):
            
            # Add to last segment's content
            last_segment = self.parsed_segments[-1]
            last_segment.speech_content += f" {text_content}"
            last_segment.confidence = min(last_segment.confidence, 0.7)  # Reduce confidence for continuations
            
            # Update timestamp if we found one
            if timestamp_hint and not last_segment.timestamp_hint:
                last_segment.timestamp_hint = timestamp_hint
        
        return None
    
    def _extract_speaker_enhanced(self, text: str, element: Tag) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Enhanced speaker extraction with contextual information."""
        speaker_info = {'method': 'pattern_match', 'party': None, 'role': None, 'html_tag': None}
        
        # Check for HTML formatting clues
        if element.find(['strong', 'b']):
            speaker_info['html_tag'] = 'bold'
        
        for i, pattern in enumerate(self.SPEAKER_PATTERNS):
            match = re.match(pattern, text, re.MULTILINE | re.UNICODE)
            if match:
                speaker_name = match.group(1).strip()
                remaining_text = text[match.end():].strip()
                
                # Extract party/role information if available
                if len(match.groups()) > 1 and match.group(2):
                    speaker_info['party'] = match.group(2).strip()
                
                # Enhanced validation
                if self._validate_speaker_name(speaker_name, remaining_text):
                    speaker_info['pattern_index'] = i
                    return speaker_name, remaining_text, speaker_info
        
        return None
    
    def _validate_speaker_name(self, speaker_name: str, remaining_text: str) -> bool:
        """Enhanced speaker name validation."""
        # Must be reasonable length
        if len(speaker_name) < 2 or len(speaker_name) > 100:
            return False
        
        # Must have remaining speech text of reasonable length
        if len(remaining_text) < 5:
            return False
        
        # Must contain letters (not just punctuation/numbers)
        if not re.search(r'[A-Za-züäöüúéèàâêîôûç]', speaker_name):
            return False
        
        # Should not be mostly punctuation
        punct_count = sum(1 for c in speaker_name if c in '.,;:!?()[]{}"-')
        if punct_count > len(speaker_name) * 0.5:
            return False
        
        # Common false positives to exclude
        false_positives = [
            'written statements', 'voting time', 'next item',
            'the sitting', 'the session', 'order of business'
        ]
        
        speaker_lower = speaker_name.lower()
        if any(fp in speaker_lower for fp in false_positives):
            return False
        
        return True
    
    def _is_procedural_content_enhanced(self, text: str, speaker_name: str = None) -> bool:
        """Enhanced procedural content detection."""
        text_lower = text.lower()
        
        # Check against enhanced procedural patterns
        for pattern in self.PROCEDURAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                return True
        
        # Speaker-based detection
        if speaker_name:
            speaker_lower = speaker_name.lower()
            if 'president' in speaker_lower and len(text.split()) < 50:
                # Short statements from presidents are usually procedural
                return True
        
        # Content-based heuristics
        procedural_keywords = [
            'applause', 'laughter', 'interruption', 'mixed reactions',
            'uproar', 'protests', 'shouts', 'silence',
            'sitting opened', 'sitting closed', 'sitting suspended',
            'voting time', 'vote is taken', 'unanimously adopted',
            'next item', 'debate is closed', 'that concludes'
        ]
        
        keyword_matches = sum(1 for keyword in procedural_keywords if keyword in text_lower)
        if keyword_matches >= 2:  # Multiple procedural keywords
            return True
        
        # Format-based detection
        if text.startswith('(') and text.endswith(')'):
            return True
        
        # Very short statements are often procedural
        if len(text.split()) < 10 and not re.search(r'[.!?]', text):
            return True
        
        return False
    
    def _extract_timestamp_hint_enhanced(self, text: str) -> Optional[str]:
        """Enhanced timestamp extraction with European formats."""
        for pattern in self.TIME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Find the actual time string from the groups
                for i in range(1, len(match.groups()) + 1):
                    group = match.group(i)
                    if group and re.match(r'\d{1,2}[:.h]\d{2}', group):
                        # Normalize format
                        normalized = group.replace('h', ':').replace('.', ':')
                        return normalized
        
        return None
    
    def _calculate_parsing_confidence(self, speaker_name: str, speech_text: str, 
                                    speaker_info: Dict, is_procedural: bool) -> float:
        """Calculate parsing confidence based on multiple factors."""
        confidence = 0.5  # Base confidence
        
        # Speaker name quality
        if len(speaker_name) > 10:
            confidence += 0.1
        if speaker_info.get('party'):
            confidence += 0.1
        if speaker_info.get('html_tag'):
            confidence += 0.05
        
        # Speech text quality
        word_count = len(speech_text.split())
        if word_count > 20:
            confidence += 0.15
        elif word_count > 50:
            confidence += 0.25
        
        # Pattern match quality
        pattern_index = speaker_info.get('pattern_index', 99)
        if pattern_index < 3:  # Earlier patterns are more reliable
            confidence += 0.1
        
        # Procedural vs speech content
        if is_procedural and word_count < 20:
            confidence += 0.1  # Short procedural is usually clear
        elif not is_procedural and word_count > 30:
            confidence += 0.15  # Long speech is usually clear
        
        return min(1.0, confidence)
    
    def _looks_like_new_speaker(self, text: str) -> bool:
        """Check if text looks like it starts with a new speaker."""
        # Quick check for common speaker indicators
        return bool(re.match(r'^\s*[A-ZÜÄÖÚÉ][A-Za-z\s]+[:\-–]', text))
    
    def _post_process_segments(self) -> None:
        """Post-process parsed segments for quality improvement."""
        if not self.parsed_segments:
            return
        
        # Merge very short procedural segments that appear to be continuations
        merged_segments = []
        i = 0
        
        while i < len(self.parsed_segments):
            current = self.parsed_segments[i]
            
            # Check if we should merge with next segment
            if (i < len(self.parsed_segments) - 1 and 
                self._should_merge_segments(current, self.parsed_segments[i + 1])):
                
                next_segment = self.parsed_segments[i + 1]
                current.speech_content += " " + next_segment.speech_content
                current.confidence = min(current.confidence, next_segment.confidence)
                i += 1  # Skip the merged segment
            
            merged_segments.append(current)
            i += 1
        
        self.parsed_segments = merged_segments
        
        # Update sequence positions
        for i, segment in enumerate(self.parsed_segments):
            segment.segment_position = i
    
    def _should_merge_segments(self, seg1: ParsedSegment, seg2: ParsedSegment) -> bool:
        """Determine if two segments should be merged."""
        # Don't merge if speakers are different and both are confident
        if (seg1.speaker_text != seg2.speaker_text and 
            seg1.confidence > 0.7 and seg2.confidence > 0.7):
            return False
        
        # Merge very short procedural segments
        if (seg1.is_procedural and seg2.is_procedural and
            len(seg1.speech_content.split()) < 5 and 
            len(seg2.speech_content.split()) < 5):
            return True
        
        # Merge if second segment looks like continuation (no clear speaker)
        if (seg1.speaker_text == seg2.speaker_text and 
            seg2.confidence < 0.5):
            return True
        
        return False
    
    def _convert_to_raw_segments_enhanced(self) -> List[RawSpeechSegment]:
        """Convert parsed segments to RawSpeechSegment objects with enhanced validation."""
        raw_segments = []
        
        for i, segment in enumerate(self.parsed_segments):
            try:
                # Enhanced text cleaning
                speaker_name = normalize_speaker_name(segment.speaker_text)
                speech_text = clean_html_text(segment.speech_content)
                
                # Enhanced validation
                validation_result = self._validate_segment_quality(speaker_name, speech_text, segment)
                if not validation_result['is_valid']:
                    logger.debug(
                        "Segment failed validation",
                        segment_index=i,
                        issues=validation_result['issues']
                    )
                    self.parsing_stats['validation_failures'] += 1
                    continue
                
                # Create enhanced raw segment
                raw_segment = RawSpeechSegment(
                    session_id=self.current_session_id,
                    sequence_number=len(raw_segments) + 1,  # Resequence after filtering
                    speaker_raw=speaker_name,
                    speech_text=speech_text,
                    timestamp_hint=segment.timestamp_hint,
                    is_procedural=segment.is_procedural,
                    confidence_score=segment.confidence,
                    parsing_metadata={
                        'parser_version': '2.0',  # Enhanced version
                        'position': segment.segment_position,
                        'timestamp_extracted': segment.timestamp_hint is not None,
                        'procedural_detected': segment.is_procedural,
                        'validation_score': validation_result['quality_score'],
                        'parsing_method': 'enhanced_patterns'
                    }
                )
                
                raw_segments.append(raw_segment)
                
            except Exception as e:
                logger.warning(
                    "Failed to create enhanced raw segment",
                    error=str(e),
                    segment_index=i,
                    speaker=segment.speaker_text[:50]
                )
                self.parsing_stats['validation_failures'] += 1
                continue
        
        return raw_segments
    
    def _validate_segment_quality(self, speaker_name: str, speech_text: str, 
                                 segment: ParsedSegment) -> Dict[str, Any]:
        """Validate segment quality with detailed scoring."""
        quality_score = 0.0
        issues = []
        
        # Speaker validation
        if not speaker_name or len(speaker_name) < 2:
            issues.append('invalid_speaker')
        else:
            quality_score += 0.3
            
            if len(speaker_name) > 5:
                quality_score += 0.1
        
        # Speech text validation
        if not speech_text or len(speech_text) < 10:
            issues.append('insufficient_content')
        else:
            quality_score += 0.4
            
            word_count = len(speech_text.split())
            if word_count > 20:
                quality_score += 0.1
            if word_count > 50:
                quality_score += 0.1
        
        # Confidence validation
        if segment.confidence >= 0.8:
            quality_score += 0.1
        elif segment.confidence < 0.3:
            issues.append('low_confidence')
        
        # Final validation  
        is_valid = quality_score >= 0.3  # More lenient threshold
        
        return {
            'is_valid': is_valid,
            'quality_score': quality_score,
            'issues': issues
        }
    
    def get_enhanced_parsing_stats(self) -> Dict[str, Any]:
        """Get comprehensive parsing statistics."""
        if not self.parsed_segments:
            return {'error': 'no_segments_parsed'}
        
        total_segments = len(self.parsed_segments)
        procedural_count = sum(1 for s in self.parsed_segments if s.is_procedural)
        with_timestamps = sum(1 for s in self.parsed_segments if s.timestamp_hint)
        
        confidence_scores = [s.confidence for s in self.parsed_segments]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        high_confidence = sum(1 for c in confidence_scores if c >= 0.8)
        low_confidence = sum(1 for c in confidence_scores if c < 0.5)
        
        return {
            'parsing_method': 'enhanced_patterns_v2',
            'total_segments': total_segments,
            'procedural_segments': procedural_count,
            'speech_segments': total_segments - procedural_count,
            'segments_with_timestamps': with_timestamps,
            'timestamp_coverage_pct': round(with_timestamps / total_segments * 100, 1),
            'average_confidence': round(avg_confidence, 3),
            'high_confidence_segments': high_confidence,
            'low_confidence_segments': low_confidence,
            'raw_stats': self.parsing_stats,
            'quality_metrics': {
                'speaker_pattern_success_rate': round(
                    self.parsing_stats['speaker_patterns_matched'] / 
                    max(1, self.parsing_stats['total_elements']) * 100, 1
                ),
                'validation_failure_rate': round(
                    self.parsing_stats['validation_failures'] / 
                    max(1, total_segments) * 100, 1
                )
            }
        }


class VerbatimSegmentProcessor:
    """Post-processor for verbatim segments to enhance quality."""
    
    def __init__(self):
        """Initialize segment processor."""
        logger.info("Verbatim segment processor initialized")
    
    def process_segments(self, raw_segments: List[RawSpeechSegment], 
                        session_date: datetime) -> List[RawSpeechSegment]:
        """
        Process raw segments to enhance quality and consistency.
        
        Args:
            raw_segments: List of raw speech segments
            session_date: Session date for context
            
        Returns:
            List of processed segments
        """
        if not raw_segments:
            return []
        
        logger.info("Processing verbatim segments", count=len(raw_segments))
        
        processed_segments = []
        
        for segment in raw_segments:
            try:
                # Clean and enhance segment
                enhanced_segment = self._enhance_segment(segment, session_date)
                
                if enhanced_segment and self._validate_segment(enhanced_segment):
                    processed_segments.append(enhanced_segment)
                    
            except Exception as e:
                logger.warning(
                    "Failed to process segment",
                    error=str(e),
                    session_id=segment.session_id,
                    sequence=segment.sequence_number
                )
                continue
        
        # Post-process for time interpolation if needed
        processed_segments = self._interpolate_timestamps(processed_segments, session_date)
        
        logger.info(
            "Segment processing completed",
            input_count=len(raw_segments),
            output_count=len(processed_segments)
        )
        
        return processed_segments
    
    def _enhance_segment(self, segment: RawSpeechSegment, 
                        session_date: datetime) -> Optional[RawSpeechSegment]:
        """Enhance individual segment quality."""
        # Clean speaker name
        speaker_info = extract_speaker_info(segment.speaker_raw)
        enhanced_speaker = speaker_info.get('name') if speaker_info else segment.speaker_raw
        if not enhanced_speaker:
            return None
        
        # Clean speech text
        enhanced_text = clean_html_text(segment.speech_text)
        if len(enhanced_text) < 10:
            return None
        
        # Update segment
        segment.speaker_raw = enhanced_speaker
        segment.speech_text = enhanced_text
        
        # Enhance metadata
        if segment.parsing_metadata:
            segment.parsing_metadata['enhanced'] = True
        
        return segment
    
    def _validate_segment(self, segment: RawSpeechSegment) -> bool:
        """Validate segment quality."""
        if not segment.speaker_raw or len(segment.speaker_raw) < 2:
            return False
        
        if not segment.speech_text or len(segment.speech_text) < 10:
            return False
        
        if segment.confidence_score < 0.1:
            return False
        
        return True
    
    def _interpolate_timestamps(self, segments: List[RawSpeechSegment], 
                               session_date: datetime) -> List[RawSpeechSegment]:
        """Interpolate missing timestamps."""
        if not segments:
            return segments
        
        # Basic implementation - use time interpolation from utils
        try:
            return interpolate_timestamps(segments, session_date)
        except Exception as e:
            logger.warning("Timestamp interpolation failed", error=str(e))
            return segments