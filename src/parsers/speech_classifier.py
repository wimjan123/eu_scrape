"""Speech classification system for distinguishing speeches from announcements."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.speech import AnnouncementType, RawSpeechSegment
from ..core.logging import get_logger

logger = get_logger(__name__)


class ContentType(str, Enum):
    """Content classification types."""
    SPEECH = "speech"
    ANNOUNCEMENT = "announcement"
    PROCEDURAL = "procedural"
    INTERRUPTION = "interruption"


@dataclass
class ClassificationResult:
    """Result of speech classification."""
    content_type: ContentType
    announcement_type: Optional[AnnouncementType] = None
    confidence: float = 1.0
    features: List[str] = None
    reasoning: str = ""


class SpeechClassifier:
    """Advanced classifier for EU Parliament speech segments."""
    
    def __init__(self):
        """Initialize the speech classifier."""
        self.session_patterns = self._build_session_patterns()
        self.voting_patterns = self._build_voting_patterns()
        self.agenda_patterns = self._build_agenda_patterns()
        self.procedural_patterns = self._build_procedural_patterns()
        self.speech_indicators = self._build_speech_indicators()
        
        logger.info("Speech classifier initialized")
    
    def classify_segment(self, segment: RawSpeechSegment) -> ClassificationResult:
        """
        Classify a speech segment as speech or announcement.
        
        Args:
            segment: Raw speech segment to classify
            
        Returns:
            Classification result with type and confidence
        """
        text = segment.speech_text.lower()
        speaker = segment.speaker_raw.lower()
        
        # Extract features for classification
        features = self._extract_features(text, speaker)
        
        # Run classification logic
        result = self._classify_by_patterns(text, speaker, features)
        
        # Enhance with metadata
        result.features = features
        
        logger.debug(
            "Classified segment",
            session_id=segment.session_id,
            sequence=segment.sequence_number,
            content_type=result.content_type,
            confidence=result.confidence
        )
        
        return result
    
    def classify_batch(self, segments: List[RawSpeechSegment]) -> List[ClassificationResult]:
        """
        Classify multiple segments with context awareness.
        
        Args:
            segments: List of segments to classify
            
        Returns:
            List of classification results
        """
        results = []
        context = {'previous_type': None, 'session_phase': 'opening'}
        
        for i, segment in enumerate(segments):
            result = self.classify_segment(segment)
            
            # Apply contextual adjustments
            result = self._apply_context(result, context, i, len(segments))
            
            results.append(result)
            
            # Update context
            context['previous_type'] = result.content_type
            if i < len(segments) * 0.2:
                context['session_phase'] = 'opening'
            elif i > len(segments) * 0.8:
                context['session_phase'] = 'closing'
            else:
                context['session_phase'] = 'main'
        
        logger.info("Classified batch", total_segments=len(segments))
        return results
    
    def _build_session_patterns(self) -> List[Tuple[str, float]]:
        """Build session management patterns with confidence weights."""
        return [
            (r'\bsitting.*(?:opened|begins?|starts?)\b', 0.95),
            (r'\bsitting.*(?:closed|ends?|adjourned?)\b', 0.95),
            (r'\bsitting.*(?:suspended?|resumed?|interrupted?)\b', 0.90),
            (r'\bsession.*(?:begins?|starts?|opens?)\b', 0.85),
            (r'\bsession.*(?:ends?|closes?|adjourns?)\b', 0.85),
            (r'\b(?:good morning|good afternoon|good evening)\b.*colleagues', 0.80),
            (r'\bbreak.*(?:minutes?|lunch|coffee)\b', 0.75),
        ]
    
    def _build_voting_patterns(self) -> List[Tuple[str, float]]:
        """Build voting procedure patterns."""
        return [
            (r'\bvoting\s+time\b', 0.95),
            (r'\bvote\s+(?:on|is taken|now)\b', 0.90),
            (r'\bvoting\s+list\b', 0.90),
            (r'\bvoting\s+results?\b', 0.85),
            (r'\b(?:adopted|rejected).*(?:unanimously|majority)\b', 0.85),
            (r'\bexplanations?\s+of\s+votes?\b', 0.80),
            (r'\bcorrections?\s+to\s+votes?\b', 0.80),
            (r'\broll.*call.*vote\b', 0.75),
        ]
    
    def _build_agenda_patterns(self) -> List[Tuple[str, float]]:
        """Build agenda item patterns."""
        return [
            (r'\bnext\s+item.*agenda\b', 0.95),
            (r'\bdebate\s+on.*(?:report|proposal|question)\b', 0.90),
            (r'\breport\s+by.*(?:mr|mrs|ms)\s+\w+', 0.85),
            (r'\bquestion\s+(?:by|from).*(?:mr|mrs|ms)\s+\w+', 0.85),
            (r'\bstatement\s+by.*(?:commission|council)\b', 0.80),
            (r'\border\s+of\s+business\b', 0.80),
            (r'\bdiscussion\s+on.*', 0.70),
        ]
    
    def _build_procedural_patterns(self) -> List[Tuple[str, float]]:
        """Build procedural notice patterns."""
        return [
            (r'\bwritten\s+statements?\b', 0.90),
            (r'\bprocedures?\s+without\s+debate\b', 0.90),
            (r'\bi\s+call\s+(?:upon|on)\b', 0.85),
            (r'\brule\s+\d+.*(?:regulation|procedure)\b', 0.80),
            (r'\bmembers?\s+requests?\b', 0.75),
            (r'\bpoint\s+of\s+order\b', 0.85),
        ]
    
    def _build_speech_indicators(self) -> List[Tuple[str, float]]:
        """Build patterns that indicate actual speech content."""
        return [
            (r'\b(?:mr|madam)\s+president.*(?:thank you|i would like)\b', 0.90),
            (r'\bi\s+(?:believe|think|propose|suggest|urge)\b', 0.80),
            (r'\bwe\s+(?:must|should|need to|have to)\b', 0.80),
            (r'\bthis\s+(?:is|shows|demonstrates|proves)\b', 0.75),
            (r'\bin\s+my\s+(?:view|opinion|country)\b', 0.75),
            (r'\blet\s+me\s+(?:say|tell|explain|clarify)\b', 0.70),
            (r'\bi\s+(?:want|wish|hope)\s+to\b', 0.70),
        ]
    
    def _extract_features(self, text: str, speaker: str) -> List[str]:
        """Extract classification features from text and speaker."""
        features = []
        
        # Length features
        word_count = len(text.split())
        if word_count < 20:
            features.append('short_text')
        elif word_count > 200:
            features.append('long_text')
        else:
            features.append('medium_text')
        
        # Speaker features
        if 'president' in speaker:
            features.append('president_speaker')
        elif any(role in speaker for role in ['commissioner', 'minister', 'rapporteur']):
            features.append('official_speaker')
        else:
            features.append('member_speaker')
        
        # Content features
        if text.startswith('(') and text.endswith(')'):
            features.append('parenthetical')
        
        if any(word in text for word in ['applause', 'laughter', 'interruption']):
            features.append('has_interruptions')
        
        question_count = text.count('?')
        if question_count > 2:
            features.append('many_questions')
        elif question_count > 0:
            features.append('has_questions')
        
        # Time references
        if re.search(r'\b\d{1,2}[:.]\d{2}\b', text):
            features.append('has_timestamp')
        
        # Formal language indicators
        if any(phrase in text for phrase in ['ladies and gentlemen', 'honourable members', 'distinguished colleagues']):
            features.append('formal_address')
        
        # Policy content indicators
        policy_words = ['regulation', 'directive', 'amendment', 'proposal', 'budget', 'policy', 'legislation']
        if sum(1 for word in policy_words if word in text) >= 3:
            features.append('policy_heavy')
        
        return features
    
    def _classify_by_patterns(self, text: str, speaker: str, features: List[str]) -> ClassificationResult:
        """Run pattern-based classification."""
        scores = {
            ContentType.SPEECH: 0.0,
            ContentType.ANNOUNCEMENT: 0.0,
            ContentType.PROCEDURAL: 0.0,
            ContentType.INTERRUPTION: 0.0
        }
        
        matched_patterns = []
        
        # Check session management patterns
        for pattern, weight in self.session_patterns:
            if re.search(pattern, text):
                scores[ContentType.ANNOUNCEMENT] += weight
                matched_patterns.append(('session', pattern, weight))
        
        # Check voting patterns
        for pattern, weight in self.voting_patterns:
            if re.search(pattern, text):
                scores[ContentType.PROCEDURAL] += weight
                matched_patterns.append(('voting', pattern, weight))
        
        # Check agenda patterns
        for pattern, weight in self.agenda_patterns:
            if re.search(pattern, text):
                scores[ContentType.ANNOUNCEMENT] += weight
                matched_patterns.append(('agenda', pattern, weight))
        
        # Check procedural patterns
        for pattern, weight in self.procedural_patterns:
            if re.search(pattern, text):
                scores[ContentType.PROCEDURAL] += weight
                matched_patterns.append(('procedural', pattern, weight))
        
        # Check speech indicators
        for pattern, weight in self.speech_indicators:
            if re.search(pattern, text):
                scores[ContentType.SPEECH] += weight
                matched_patterns.append(('speech', pattern, weight))
        
        # Apply feature-based adjustments
        scores = self._apply_feature_adjustments(scores, features)
        
        # Determine winner
        max_score = max(scores.values())
        if max_score == 0:
            # Default classification based on features
            return self._default_classification(features, matched_patterns)
        
        content_type = max(scores, key=scores.get)
        confidence = min(1.0, max_score / 2.0)  # Normalize confidence
        
        # Determine announcement type if applicable
        announcement_type = None
        if content_type == ContentType.ANNOUNCEMENT:
            announcement_type = self._determine_announcement_type(matched_patterns)
        
        reasoning = self._build_reasoning(matched_patterns, features)
        
        return ClassificationResult(
            content_type=content_type,
            announcement_type=announcement_type,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _apply_feature_adjustments(self, scores: Dict[ContentType, float], features: List[str]) -> Dict[ContentType, float]:
        """Apply feature-based score adjustments."""
        # President speakers are more likely to make announcements
        if 'president_speaker' in features:
            scores[ContentType.ANNOUNCEMENT] += 0.3
            scores[ContentType.PROCEDURAL] += 0.2
        
        # Very short text is likely procedural
        if 'short_text' in features:
            scores[ContentType.SPEECH] -= 0.2
            scores[ContentType.PROCEDURAL] += 0.2
        
        # Long text is more likely to be speech
        if 'long_text' in features:
            scores[ContentType.SPEECH] += 0.3
        
        # Parenthetical content is usually interruptions or announcements
        if 'parenthetical' in features:
            scores[ContentType.INTERRUPTION] += 0.4
            scores[ContentType.SPEECH] -= 0.2
        
        # Interruptions indicate procedural content
        if 'has_interruptions' in features:
            scores[ContentType.INTERRUPTION] += 0.2
        
        # Policy-heavy content is usually speech
        if 'policy_heavy' in features:
            scores[ContentType.SPEECH] += 0.4
        
        return scores
    
    def _determine_announcement_type(self, matched_patterns: List[Tuple[str, str, float]]) -> AnnouncementType:
        """Determine the specific type of announcement."""
        pattern_types = [p[0] for p in matched_patterns]
        
        if 'session' in pattern_types:
            return AnnouncementType.SESSION_MANAGEMENT
        elif 'voting' in pattern_types:
            return AnnouncementType.VOTING_PROCEDURE
        elif 'agenda' in pattern_types:
            return AnnouncementType.AGENDA_ITEM
        elif 'procedural' in pattern_types:
            return AnnouncementType.PROCEDURAL_NOTICE
        else:
            return AnnouncementType.GENERAL_ANNOUNCEMENT
    
    def _default_classification(self, features: List[str], patterns: List) -> ClassificationResult:
        """Provide default classification when no patterns match."""
        # Use features to make educated guess
        if 'parenthetical' in features or 'has_interruptions' in features:
            return ClassificationResult(
                content_type=ContentType.INTERRUPTION,
                confidence=0.6,
                reasoning="No patterns matched, classified by parenthetical/interruption features"
            )
        
        if 'president_speaker' in features and 'short_text' in features:
            return ClassificationResult(
                content_type=ContentType.ANNOUNCEMENT,
                announcement_type=AnnouncementType.GENERAL_ANNOUNCEMENT,
                confidence=0.5,
                reasoning="President speaker with short text, likely announcement"
            )
        
        if 'long_text' in features or 'policy_heavy' in features:
            return ClassificationResult(
                content_type=ContentType.SPEECH,
                confidence=0.7,
                reasoning="Long or policy-heavy content, likely speech"
            )
        
        # Default to speech with low confidence
        return ClassificationResult(
            content_type=ContentType.SPEECH,
            confidence=0.3,
            reasoning="No clear indicators, defaulting to speech"
        )
    
    def _build_reasoning(self, patterns: List[Tuple], features: List[str]) -> str:
        """Build human-readable reasoning for classification."""
        reasons = []
        
        if patterns:
            pattern_summary = {}
            for ptype, pattern, weight in patterns:
                if ptype not in pattern_summary:
                    pattern_summary[ptype] = []
                pattern_summary[ptype].append(weight)
            
            for ptype, weights in pattern_summary.items():
                avg_weight = sum(weights) / len(weights)
                reasons.append(f"{ptype} patterns (avg weight: {avg_weight:.2f})")
        
        if 'president_speaker' in features:
            reasons.append("president speaker")
        
        if 'short_text' in features:
            reasons.append("short text")
        elif 'long_text' in features:
            reasons.append("long text")
        
        if 'policy_heavy' in features:
            reasons.append("policy-heavy content")
        
        if 'parenthetical' in features:
            reasons.append("parenthetical format")
        
        return "; ".join(reasons) if reasons else "no clear indicators"
    
    def _apply_context(self, result: ClassificationResult, context: Dict, 
                      position: int, total: int) -> ClassificationResult:
        """Apply contextual adjustments to classification result."""
        # Opening/closing sessions are more likely to have procedural content
        if context['session_phase'] in ['opening', 'closing'] and result.content_type == ContentType.SPEECH:
            if result.confidence < 0.8:  # Only adjust if we're not very confident
                result.confidence *= 0.9
        
        # Consecutive announcements are common
        if (context['previous_type'] == ContentType.ANNOUNCEMENT and 
            result.content_type == ContentType.SPEECH and 
            result.confidence < 0.6):
            result.confidence *= 0.8
        
        return result
    
    def get_classification_stats(self, results: List[ClassificationResult]) -> Dict[str, Any]:
        """Get statistics about classification results."""
        if not results:
            return {}
        
        content_counts = {}
        announcement_counts = {}
        total_confidence = 0
        
        for result in results:
            # Count content types
            content_type = result.content_type.value
            content_counts[content_type] = content_counts.get(content_type, 0) + 1
            
            # Count announcement types
            if result.announcement_type:
                ann_type = result.announcement_type.value
                announcement_counts[ann_type] = announcement_counts.get(ann_type, 0) + 1
            
            total_confidence += result.confidence
        
        avg_confidence = total_confidence / len(results)
        
        return {
            'total_segments': len(results),
            'content_type_distribution': content_counts,
            'announcement_type_distribution': announcement_counts,
            'average_confidence': round(avg_confidence, 3),
            'high_confidence_count': sum(1 for r in results if r.confidence >= 0.8),
            'low_confidence_count': sum(1 for r in results if r.confidence < 0.5)
        }