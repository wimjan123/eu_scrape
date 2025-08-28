"""Quality validation framework for parsed speech segments."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..models.speech import RawSpeechSegment, SpeechSegment
from ..core.logging import get_logger
from ..utils.text_utils import validate_speech_text_quality
from ..parsers.speech_classifier import SpeechClassifier, ClassificationResult

logger = get_logger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    CRITICAL = "critical"  # Must fix - segment unusable
    HIGH = "high"         # Should fix - quality issues
    MEDIUM = "medium"     # May fix - minor issues
    LOW = "low"          # Optional - style issues


class ValidationCategory(str, Enum):
    """Categories of validation issues."""
    SPEAKER_QUALITY = "speaker_quality"
    CONTENT_QUALITY = "content_quality"
    TIMESTAMP_VALIDITY = "timestamp_validity"
    CLASSIFICATION_ACCURACY = "classification_accuracy"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    METADATA_CONSISTENCY = "metadata_consistency"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    segment_id: str = ""
    sequence_number: int = 0
    field_name: str = ""
    suggested_fix: str = ""
    confidence: float = 1.0


@dataclass
class ValidationResult:
    """Result of validation process."""
    is_valid: bool
    quality_score: float
    issues: List[ValidationIssue]
    segment_count: int = 0
    passed_count: int = 0
    warnings_count: int = 0
    errors_count: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'is_valid': self.is_valid,
            'quality_score': round(self.quality_score, 3),
            'total_segments': self.segment_count,
            'passed_segments': self.passed_count,
            'segments_with_warnings': self.warnings_count,
            'segments_with_errors': self.errors_count,
            'issues_by_category': self._group_issues_by_category(),
            'issues_by_severity': self._group_issues_by_severity()
        }
    
    def _group_issues_by_category(self) -> Dict[str, int]:
        """Group issues by category."""
        categories = {}
        for issue in self.issues:
            cat = issue.category.value
            categories[cat] = categories.get(cat, 0) + 1
        return categories
    
    def _group_issues_by_severity(self) -> Dict[str, int]:
        """Group issues by severity."""
        severities = {}
        for issue in self.issues:
            sev = issue.severity.value
            severities[sev] = severities.get(sev, 0) + 1
        return severities


class ParsingValidator:
    """Comprehensive validator for parsed speech segments."""
    
    def __init__(self):
        """Initialize the parsing validator."""
        self.classifier = SpeechClassifier()
        
        # Quality thresholds
        self.thresholds = {
            'min_speaker_length': 2,
            'max_speaker_length': 100,
            'min_speech_length': 10,
            'max_speech_length': 10000,
            'min_confidence': 0.3,
            'min_quality_score': 0.5,
            'max_timestamp_gap_hours': 8,
            'min_procedural_confidence': 0.6
        }
        
        # Common validation patterns
        self.speaker_patterns = {
            'valid_name': r'^[A-ZÜÄÖÚÉ][A-Za-züäöüúéèàâêîôûç\'\-\s.]+$',
            'has_title': r'\b(Mr|Mrs|Ms|Dr|Prof|President|Minister|Commissioner)\b',
            'has_party': r'\([^)]+\)',
            'suspicious_chars': r'[<>{}[\]|\\@#$%^&*+=_~`]'
        }
        
        logger.info("Parsing validator initialized")
    
    def validate_segment(self, segment: RawSpeechSegment) -> ValidationResult:
        """
        Validate a single speech segment.
        
        Args:
            segment: Raw speech segment to validate
            
        Returns:
            Validation result with issues and quality score
        """
        issues = []
        
        # Validate speaker quality
        speaker_issues = self._validate_speaker_quality(segment)
        issues.extend(speaker_issues)
        
        # Validate content quality
        content_issues = self._validate_content_quality(segment)
        issues.extend(content_issues)
        
        # Validate timestamp
        timestamp_issues = self._validate_timestamp(segment)
        issues.extend(timestamp_issues)
        
        # Validate metadata consistency
        metadata_issues = self._validate_metadata_consistency(segment)
        issues.extend(metadata_issues)
        
        # Validate classification if procedural
        if segment.is_procedural:
            classification_issues = self._validate_classification(segment)
            issues.extend(classification_issues)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(segment, issues)
        
        # Determine if segment is valid
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        is_valid = len(critical_issues) == 0 and quality_score >= self.thresholds['min_quality_score']
        
        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            segment_count=1,
            passed_count=1 if is_valid else 0,
            warnings_count=1 if issues and is_valid else 0,
            errors_count=1 if not is_valid else 0
        )
    
    def validate_batch(self, segments: List[RawSpeechSegment]) -> ValidationResult:
        """
        Validate a batch of speech segments with cross-segment validation.
        
        Args:
            segments: List of segments to validate
            
        Returns:
            Comprehensive validation result
        """
        if not segments:
            return ValidationResult(
                is_valid=True,
                quality_score=1.0,
                issues=[],
                segment_count=0
            )
        
        logger.info("Starting batch validation", segment_count=len(segments))
        
        all_issues = []
        individual_scores = []
        passed_count = 0
        warnings_count = 0
        errors_count = 0
        
        # Individual segment validation
        for segment in segments:
            result = self.validate_segment(segment)
            all_issues.extend(result.issues)
            individual_scores.append(result.quality_score)
            
            if result.is_valid:
                passed_count += 1
                if result.issues:
                    warnings_count += 1
            else:
                errors_count += 1
        
        # Cross-segment validation
        cross_segment_issues = self._validate_cross_segment_consistency(segments)
        all_issues.extend(cross_segment_issues)
        
        # Sequence validation
        sequence_issues = self._validate_sequence_integrity(segments)
        all_issues.extend(sequence_issues)
        
        # Calculate overall batch quality
        avg_individual_score = sum(individual_scores) / len(individual_scores)
        cross_segment_penalty = len(cross_segment_issues) * 0.05
        overall_quality = max(0.0, avg_individual_score - cross_segment_penalty)
        
        # Batch is valid if most segments are valid and no critical cross-segment issues
        critical_cross_issues = [i for i in cross_segment_issues 
                               if i.severity == ValidationSeverity.CRITICAL]
        batch_is_valid = (passed_count / len(segments) >= 0.8 and 
                         len(critical_cross_issues) == 0)
        
        logger.info(
            "Batch validation completed",
            is_valid=batch_is_valid,
            quality_score=overall_quality,
            issues_count=len(all_issues)
        )
        
        return ValidationResult(
            is_valid=batch_is_valid,
            quality_score=overall_quality,
            issues=all_issues,
            segment_count=len(segments),
            passed_count=passed_count,
            warnings_count=warnings_count,
            errors_count=errors_count
        )
    
    def _validate_speaker_quality(self, segment: RawSpeechSegment) -> List[ValidationIssue]:
        """Validate speaker name and information quality."""
        issues = []
        speaker = segment.speaker_raw
        
        if not speaker:
            issues.append(ValidationIssue(
                category=ValidationCategory.SPEAKER_QUALITY,
                severity=ValidationSeverity.CRITICAL,
                message="Speaker name is empty",
                segment_id=segment.session_id,
                sequence_number=segment.sequence_number,
                field_name="speaker_raw",
                suggested_fix="Extract speaker name from content or mark as procedural"
            ))
            return issues
        
        # Length validation
        if len(speaker) < self.thresholds['min_speaker_length']:
            issues.append(ValidationIssue(
                category=ValidationCategory.SPEAKER_QUALITY,
                severity=ValidationSeverity.HIGH,
                message=f"Speaker name too short: '{speaker}'",
                segment_id=segment.session_id,
                sequence_number=segment.sequence_number,
                field_name="speaker_raw",
                suggested_fix="Validate speaker extraction or merge with adjacent segment"
            ))
        
        if len(speaker) > self.thresholds['max_speaker_length']:
            issues.append(ValidationIssue(
                category=ValidationCategory.SPEAKER_QUALITY,
                severity=ValidationSeverity.HIGH,
                message=f"Speaker name too long: {len(speaker)} chars",
                segment_id=segment.session_id,
                sequence_number=segment.sequence_number,
                field_name="speaker_raw",
                suggested_fix="Extract just the name portion, remove excess content"
            ))
        
        # Character validation
        if re.search(self.speaker_patterns['suspicious_chars'], speaker):
            issues.append(ValidationIssue(
                category=ValidationCategory.SPEAKER_QUALITY,
                severity=ValidationSeverity.MEDIUM,
                message="Speaker name contains suspicious characters",
                segment_id=segment.session_id,
                sequence_number=segment.sequence_number,
                field_name="speaker_raw",
                suggested_fix="Clean special characters from speaker name"
            ))
        
        # Format validation
        if not re.match(self.speaker_patterns['valid_name'], speaker):
            issues.append(ValidationIssue(
                category=ValidationCategory.SPEAKER_QUALITY,
                severity=ValidationSeverity.MEDIUM,
                message="Speaker name format appears invalid",
                segment_id=segment.session_id,
                sequence_number=segment.sequence_number,
                field_name="speaker_raw",
                suggested_fix="Verify speaker name extraction patterns"
            ))
        
        return issues
    
    def _validate_content_quality(self, segment: RawSpeechSegment) -> List[ValidationIssue]:
        """Validate speech content quality."""
        issues = []
        content = segment.speech_text
        
        if not content:
            issues.append(ValidationIssue(
                category=ValidationCategory.CONTENT_QUALITY,
                severity=ValidationSeverity.CRITICAL,
                message="Speech content is empty",
                segment_id=segment.session_id,
                sequence_number=segment.sequence_number,
                field_name="speech_text",
                suggested_fix="Extract content or remove segment"
            ))
            return issues
        
        # Use text utility for detailed quality validation
        quality_metrics = validate_speech_text_quality(content)
        
        if not quality_metrics['is_valid']:
            for issue_type in quality_metrics['issues']:
                severity = self._map_quality_issue_severity(issue_type)
                issues.append(ValidationIssue(
                    category=ValidationCategory.CONTENT_QUALITY,
                    severity=severity,
                    message=f"Content quality issue: {issue_type}",
                    segment_id=segment.session_id,
                    sequence_number=segment.sequence_number,
                    field_name="speech_text",
                    suggested_fix=self._get_quality_fix_suggestion(issue_type)
                ))
        
        # Length validation
        if len(content) < self.thresholds['min_speech_length']:
            issues.append(ValidationIssue(
                category=ValidationCategory.CONTENT_QUALITY,
                severity=ValidationSeverity.HIGH,
                message=f"Speech content too short: {len(content)} chars",
                segment_id=segment.session_id,
                sequence_number=segment.sequence_number,
                field_name="speech_text",
                suggested_fix="Merge with adjacent segments or remove if procedural"
            ))
        
        if len(content) > self.thresholds['max_speech_length']:
            issues.append(ValidationIssue(
                category=ValidationCategory.CONTENT_QUALITY,
                severity=ValidationSeverity.MEDIUM,
                message=f"Speech content very long: {len(content)} chars",
                segment_id=segment.session_id,
                sequence_number=segment.sequence_number,
                field_name="speech_text",
                suggested_fix="Consider splitting into multiple segments"
            ))
        
        return issues
    
    def _validate_timestamp(self, segment: RawSpeechSegment) -> List[ValidationIssue]:
        """Validate timestamp information."""
        issues = []
        
        if segment.timestamp_hint:
            # Validate timestamp format
            if not re.match(r'^\d{1,2}[:.]\d{2}$', segment.timestamp_hint):
                issues.append(ValidationIssue(
                    category=ValidationCategory.TIMESTAMP_VALIDITY,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Invalid timestamp format: '{segment.timestamp_hint}'",
                    segment_id=segment.session_id,
                    sequence_number=segment.sequence_number,
                    field_name="timestamp_hint",
                    suggested_fix="Normalize timestamp to HH:MM format"
                ))
            else:
                # Validate time values
                time_parts = segment.timestamp_hint.replace(':', '.').split('.')
                try:
                    hour = int(time_parts[0])
                    minute = int(time_parts[1])
                    
                    if not (0 <= hour <= 23):
                        issues.append(ValidationIssue(
                            category=ValidationCategory.TIMESTAMP_VALIDITY,
                            severity=ValidationSeverity.HIGH,
                            message=f"Invalid hour in timestamp: {hour}",
                            segment_id=segment.session_id,
                            sequence_number=segment.sequence_number,
                            field_name="timestamp_hint",
                            suggested_fix="Correct hour value or remove invalid timestamp"
                        ))
                    
                    if not (0 <= minute <= 59):
                        issues.append(ValidationIssue(
                            category=ValidationCategory.TIMESTAMP_VALIDITY,
                            severity=ValidationSeverity.HIGH,
                            message=f"Invalid minute in timestamp: {minute}",
                            segment_id=segment.session_id,
                            sequence_number=segment.sequence_number,
                            field_name="timestamp_hint",
                            suggested_fix="Correct minute value or remove invalid timestamp"
                        ))
                        
                except ValueError:
                    issues.append(ValidationIssue(
                        category=ValidationCategory.TIMESTAMP_VALIDITY,
                        severity=ValidationSeverity.HIGH,
                        message=f"Could not parse timestamp: '{segment.timestamp_hint}'",
                        segment_id=segment.session_id,
                        sequence_number=segment.sequence_number,
                        field_name="timestamp_hint",
                        suggested_fix="Fix timestamp format or remove"
                    ))
        
        return issues
    
    def _validate_classification(self, segment: RawSpeechSegment) -> List[ValidationIssue]:
        """Validate procedural classification accuracy."""
        issues = []
        
        if segment.is_procedural:
            # Validate with classifier
            classification = self.classifier.classify_segment(segment)
            
            # If classifier disagrees with strong confidence, flag it
            if (classification.content_type.value != 'procedural' and 
                classification.content_type.value != 'announcement' and
                classification.confidence > 0.7):
                
                issues.append(ValidationIssue(
                    category=ValidationCategory.CLASSIFICATION_ACCURACY,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Classifier suggests '{classification.content_type.value}' not procedural",
                    segment_id=segment.session_id,
                    sequence_number=segment.sequence_number,
                    field_name="is_procedural",
                    suggested_fix="Review procedural classification",
                    confidence=classification.confidence
                ))
        
        # Validate confidence score
        if segment.confidence_score < self.thresholds['min_confidence']:
            issues.append(ValidationIssue(
                category=ValidationCategory.CLASSIFICATION_ACCURACY,
                severity=ValidationSeverity.HIGH,
                message=f"Low parsing confidence: {segment.confidence_score}",
                segment_id=segment.session_id,
                sequence_number=segment.sequence_number,
                field_name="confidence_score",
                suggested_fix="Review parsing patterns or manual verification needed"
            ))
        
        return issues
    
    def _validate_metadata_consistency(self, segment: RawSpeechSegment) -> List[ValidationIssue]:
        """Validate metadata consistency and completeness."""
        issues = []
        metadata = segment.parsing_metadata or {}
        
        # Check required metadata fields
        required_fields = ['parser_version', 'position', 'parsing_method']
        for field in required_fields:
            if field not in metadata:
                issues.append(ValidationIssue(
                    category=ValidationCategory.METADATA_CONSISTENCY,
                    severity=ValidationSeverity.LOW,
                    message=f"Missing metadata field: {field}",
                    segment_id=segment.session_id,
                    sequence_number=segment.sequence_number,
                    field_name="parsing_metadata",
                    suggested_fix=f"Add {field} to parsing metadata"
                ))
        
        # Validate metadata consistency
        if segment.is_procedural and not metadata.get('procedural_detected', False):
            issues.append(ValidationIssue(
                category=ValidationCategory.METADATA_CONSISTENCY,
                severity=ValidationSeverity.LOW,
                message="is_procedural=True but metadata doesn't reflect detection",
                segment_id=segment.session_id,
                sequence_number=segment.sequence_number,
                field_name="parsing_metadata",
                suggested_fix="Update metadata to reflect procedural detection"
            ))
        
        if segment.timestamp_hint and not metadata.get('timestamp_extracted', False):
            issues.append(ValidationIssue(
                category=ValidationCategory.METADATA_CONSISTENCY,
                severity=ValidationSeverity.LOW,
                message="timestamp_hint present but metadata doesn't reflect extraction",
                segment_id=segment.session_id,
                sequence_number=segment.sequence_number,
                field_name="parsing_metadata",
                suggested_fix="Update metadata to reflect timestamp extraction"
            ))
        
        return issues
    
    def _validate_cross_segment_consistency(self, segments: List[RawSpeechSegment]) -> List[ValidationIssue]:
        """Validate consistency across multiple segments."""
        issues = []
        
        if len(segments) < 2:
            return issues
        
        # Check for duplicate sequence numbers
        sequence_numbers = [s.sequence_number for s in segments]
        if len(set(sequence_numbers)) != len(sequence_numbers):
            issues.append(ValidationIssue(
                category=ValidationCategory.STRUCTURAL_INTEGRITY,
                severity=ValidationSeverity.CRITICAL,
                message="Duplicate sequence numbers found in batch",
                suggested_fix="Renumber segments to ensure unique sequence"
            ))
        
        # Check for reasonable sequence ordering
        sorted_sequences = sorted(sequence_numbers)
        if sequence_numbers != sorted_sequences:
            issues.append(ValidationIssue(
                category=ValidationCategory.STRUCTURAL_INTEGRITY,
                severity=ValidationSeverity.HIGH,
                message="Segments not in sequence order",
                suggested_fix="Sort segments by sequence number"
            ))
        
        # Check for session consistency
        session_ids = set(s.session_id for s in segments)
        if len(session_ids) > 1:
            issues.append(ValidationIssue(
                category=ValidationCategory.STRUCTURAL_INTEGRITY,
                severity=ValidationSeverity.HIGH,
                message=f"Multiple session IDs in batch: {session_ids}",
                suggested_fix="Process segments from same session together"
            ))
        
        # Check for speaker name consistency (detect potential parsing errors)
        speaker_counts = {}
        for segment in segments:
            speaker = segment.speaker_raw.lower()
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        # Flag speakers that appear very infrequently (might be parsing errors)
        total_segments = len(segments)
        for speaker, count in speaker_counts.items():
            if count == 1 and total_segments > 10:  # Single appearance in large batch
                if not any(proc_word in speaker for proc_word in ['president', 'commissioner']):
                    # Find the segment for detailed issue
                    for segment in segments:
                        if segment.speaker_raw.lower() == speaker:
                            issues.append(ValidationIssue(
                                category=ValidationCategory.SPEAKER_QUALITY,
                                severity=ValidationSeverity.LOW,
                                message=f"Speaker '{speaker}' appears only once - possible parsing error",
                                segment_id=segment.session_id,
                                sequence_number=segment.sequence_number,
                                field_name="speaker_raw",
                                suggested_fix="Verify speaker name or merge with adjacent segment"
                            ))
                            break
        
        return issues
    
    def _validate_sequence_integrity(self, segments: List[RawSpeechSegment]) -> List[ValidationIssue]:
        """Validate sequence integrity and logical flow."""
        issues = []
        
        if len(segments) < 2:
            return issues
        
        # Check for timestamp consistency
        timestamped_segments = [(i, s) for i, s in enumerate(segments) if s.timestamp_hint]
        
        if len(timestamped_segments) > 1:
            for i in range(1, len(timestamped_segments)):
                prev_idx, prev_seg = timestamped_segments[i-1]
                curr_idx, curr_seg = timestamped_segments[i]
                
                prev_time = self._parse_time_hint(prev_seg.timestamp_hint)
                curr_time = self._parse_time_hint(curr_seg.timestamp_hint)
                
                if prev_time and curr_time:
                    # Check for time going backwards
                    if curr_time < prev_time:
                        issues.append(ValidationIssue(
                            category=ValidationCategory.TIMESTAMP_VALIDITY,
                            severity=ValidationSeverity.HIGH,
                            message=f"Timestamp goes backwards: {prev_seg.timestamp_hint} → {curr_seg.timestamp_hint}",
                            segment_id=curr_seg.session_id,
                            sequence_number=curr_seg.sequence_number,
                            field_name="timestamp_hint",
                            suggested_fix="Correct timestamp order or interpolate missing times"
                        ))
                    
                    # Check for unreasonably large gaps
                    time_diff = abs((curr_time.hour * 60 + curr_time.minute) - 
                                   (prev_time.hour * 60 + prev_time.minute))
                    
                    if time_diff > self.thresholds['max_timestamp_gap_hours'] * 60:
                        issues.append(ValidationIssue(
                            category=ValidationCategory.TIMESTAMP_VALIDITY,
                            severity=ValidationSeverity.MEDIUM,
                            message=f"Large time gap: {time_diff} minutes between segments",
                            segment_id=curr_seg.session_id,
                            sequence_number=curr_seg.sequence_number,
                            field_name="timestamp_hint",
                            suggested_fix="Verify timestamp accuracy or mark as session break"
                        ))
        
        return issues
    
    def _parse_time_hint(self, time_hint: str) -> Optional[datetime]:
        """Parse time hint to datetime for comparison."""
        if not time_hint:
            return None
        
        try:
            time_parts = time_hint.replace(':', '.').split('.')
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            return datetime(2024, 1, 1, hour, minute)  # Use dummy date for comparison
        except (ValueError, IndexError):
            return None
    
    def _calculate_quality_score(self, segment: RawSpeechSegment, 
                                issues: List[ValidationIssue]) -> float:
        """Calculate overall quality score for a segment."""
        base_score = segment.confidence_score or 0.5
        
        # Apply penalties for issues
        penalties = {
            ValidationSeverity.CRITICAL: 0.4,
            ValidationSeverity.HIGH: 0.2,
            ValidationSeverity.MEDIUM: 0.1,
            ValidationSeverity.LOW: 0.05
        }
        
        total_penalty = sum(penalties.get(issue.severity, 0) for issue in issues)
        
        # Quality bonuses
        bonuses = 0.0
        
        # Good speaker name
        if (segment.speaker_raw and 
            len(segment.speaker_raw) > 5 and 
            re.match(self.speaker_patterns['valid_name'], segment.speaker_raw)):
            bonuses += 0.1
        
        # Good content length
        if segment.speech_text and 20 <= len(segment.speech_text.split()) <= 500:
            bonuses += 0.1
        
        # Has timestamp
        if segment.timestamp_hint:
            bonuses += 0.05
        
        # Has metadata
        if segment.parsing_metadata:
            bonuses += 0.05
        
        final_score = max(0.0, min(1.0, base_score + bonuses - total_penalty))
        return final_score
    
    def _map_quality_issue_severity(self, issue_type: str) -> ValidationSeverity:
        """Map text quality issue types to validation severities."""
        severity_mapping = {
            'empty_text': ValidationSeverity.CRITICAL,
            'too_short': ValidationSeverity.HIGH,
            'html_remnants': ValidationSeverity.HIGH,
            'low_diversity': ValidationSeverity.MEDIUM,
            'no_sentences': ValidationSeverity.MEDIUM
        }
        return severity_mapping.get(issue_type, ValidationSeverity.LOW)
    
    def _get_quality_fix_suggestion(self, issue_type: str) -> str:
        """Get fix suggestions for text quality issues."""
        suggestions = {
            'empty_text': 'Remove segment or extract content from adjacent elements',
            'too_short': 'Merge with adjacent segments or mark as procedural',
            'html_remnants': 'Improve HTML cleaning in text processing',
            'low_diversity': 'Verify content extraction - may be corrupted',
            'no_sentences': 'Check if content is actually speech or procedural text'
        }
        return suggestions.get(issue_type, 'Review content extraction process')
    
    def generate_validation_report(self, result: ValidationResult, 
                                 include_details: bool = True) -> str:
        """Generate a human-readable validation report."""
        report = []
        
        # Header
        report.append("=" * 60)
        report.append("EU PARLIAMENT PARSING VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        summary = result.get_summary()
        report.append("SUMMARY:")
        report.append(f"  Overall Status: {'PASS' if result.is_valid else 'FAIL'}")
        report.append(f"  Quality Score: {summary['quality_score']:.3f}")
        report.append(f"  Total Segments: {summary['total_segments']}")
        report.append(f"  Passed: {summary['passed_segments']}")
        report.append(f"  With Warnings: {summary['segments_with_warnings']}")
        report.append(f"  With Errors: {summary['segments_with_errors']}")
        report.append("")
        
        # Issues by severity
        if summary['issues_by_severity']:
            report.append("ISSUES BY SEVERITY:")
            for severity, count in summary['issues_by_severity'].items():
                report.append(f"  {severity.upper()}: {count}")
            report.append("")
        
        # Issues by category
        if summary['issues_by_category']:
            report.append("ISSUES BY CATEGORY:")
            for category, count in summary['issues_by_category'].items():
                report.append(f"  {category.replace('_', ' ').title()}: {count}")
            report.append("")
        
        # Detailed issues
        if include_details and result.issues:
            report.append("DETAILED ISSUES:")
            report.append("-" * 40)
            
            for issue in sorted(result.issues, 
                              key=lambda x: (x.severity.value, x.category.value)):
                report.append(f"Segment {issue.sequence_number}: {issue.severity.upper()}")
                report.append(f"  Category: {issue.category.value.replace('_', ' ').title()}")
                report.append(f"  Message: {issue.message}")
                if issue.suggested_fix:
                    report.append(f"  Suggestion: {issue.suggested_fix}")
                if issue.field_name:
                    report.append(f"  Field: {issue.field_name}")
                report.append("")
        
        return "\n".join(report)