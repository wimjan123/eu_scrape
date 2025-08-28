"""Data validation utilities for EU Parliament scraper."""

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..core.logging import get_logger
from ..models.speech import SpeechSegment

logger = get_logger(__name__)


# Valid EU country codes and names
EU_COUNTRIES = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus',
    'CZ': 'Czech Republic', 'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia',
    'ES': 'Spain', 'FI': 'Finland', 'FR': 'France', 'GR': 'Greece',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
    'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MT': 'Malta',
    'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania',
    'SE': 'Sweden', 'SI': 'Slovenia', 'SK': 'Slovakia'
}

# Common political groups
POLITICAL_GROUPS = {
    'EPP': 'European People\'s Party',
    'S&D': 'Progressive Alliance of Socialists and Democrats',
    'RE': 'Renew Europe',
    'ECR': 'European Conservatives and Reformists',
    'ID': 'Identity and Democracy',
    'Greens/EFA': 'Greens/European Free Alliance',
    'GUE/NGL': 'The Left in the European Parliament',
    'NI': 'Non-attached Members',
    'EP_PRESIDENT': 'European Parliament President',
    'EP_VICE_PRESIDENT': 'European Parliament Vice-President',
    'EUROPEAN_COMMISSION': 'European Commission'
}


def validate_speech_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive validation of a speech segment.
    
    Args:
        segment: Speech segment dictionary
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'segment_id': segment.get('id', 'unknown'),
        'is_valid': True,
        'validation_score': 1.0,
        'issues': [],
        'warnings': [],
        'field_scores': {}
    }
    
    # Validate required fields
    required_fields = [
        'speaker_name', 'speaker_country', 'speaker_party_or_group',
        'segment_start_ts', 'segment_end_ts', 'speech_text',
        'is_announcement', 'announcement_label'
    ]
    
    for field in required_fields:
        if field not in segment:
            validation_result['issues'].append(f'missing_field_{field}')
            validation_result['field_scores'][field] = 0.0
        elif segment[field] is None:
            validation_result['issues'].append(f'null_field_{field}')
            validation_result['field_scores'][field] = 0.0
        else:
            validation_result['field_scores'][field] = 1.0
    
    # Validate speaker name
    speaker_name = segment.get('speaker_name', '')
    if not speaker_name or len(speaker_name.strip()) < 2:
        validation_result['issues'].append('invalid_speaker_name')
        validation_result['field_scores']['speaker_name'] = 0.0
    elif len(speaker_name) > 200:
        validation_result['warnings'].append('very_long_speaker_name')
        validation_result['field_scores']['speaker_name'] = 0.8
    
    # Validate country
    speaker_country = segment.get('speaker_country', '')
    if speaker_country:
        country_valid = (
            speaker_country in EU_COUNTRIES or
            speaker_country in EU_COUNTRIES.values() or
            speaker_country in ['N/A', 'European Commission', 'Council']
        )
        if not country_valid:
            validation_result['warnings'].append('unrecognized_country')
            validation_result['field_scores']['speaker_country'] = 0.7
    
    # Validate political group
    party_group = segment.get('speaker_party_or_group', '')
    if party_group and party_group not in POLITICAL_GROUPS and party_group not in POLITICAL_GROUPS.values():
        validation_result['warnings'].append('unrecognized_political_group')
    
    # Validate timestamps
    start_ts = segment.get('segment_start_ts')
    end_ts = segment.get('segment_end_ts')
    
    if start_ts and end_ts:
        start_valid, start_time = validate_iso8601_timestamp(start_ts)
        end_valid, end_time = validate_iso8601_timestamp(end_ts)
        
        if not start_valid:
            validation_result['issues'].append('invalid_start_timestamp')
            validation_result['field_scores']['segment_start_ts'] = 0.0
        
        if not end_valid:
            validation_result['issues'].append('invalid_end_timestamp')
            validation_result['field_scores']['segment_end_ts'] = 0.0
        
        if start_valid and end_valid:
            duration = (end_time - start_time).total_seconds()
            if duration <= 0:
                validation_result['issues'].append('negative_duration')
            elif duration > 3600:  # More than 1 hour
                validation_result['warnings'].append('very_long_speech')
    
    # Validate speech text
    speech_text = segment.get('speech_text', '')
    if not speech_text or len(speech_text.strip()) < 10:
        validation_result['issues'].append('insufficient_speech_text')
        validation_result['field_scores']['speech_text'] = 0.0
    elif len(speech_text) > 50000:  # Very long speech
        validation_result['warnings'].append('very_long_speech_text')
    
    # Validate announcement fields consistency
    is_announcement = segment.get('is_announcement', False)
    announcement_label = segment.get('announcement_label', '')
    
    if is_announcement and not announcement_label:
        validation_result['issues'].append('missing_announcement_label')
    elif not is_announcement and announcement_label:
        validation_result['warnings'].append('unexpected_announcement_label')
    
    # Calculate overall validation score
    field_scores = list(validation_result['field_scores'].values())
    if field_scores:
        base_score = sum(field_scores) / len(field_scores)
    else:
        base_score = 0.0
    
    # Apply penalties for issues and warnings
    issue_penalty = len(validation_result['issues']) * 0.1
    warning_penalty = len(validation_result['warnings']) * 0.05
    
    validation_result['validation_score'] = max(0.0, base_score - issue_penalty - warning_penalty)
    validation_result['is_valid'] = validation_result['validation_score'] > 0.8 and len(validation_result['issues']) == 0
    
    return validation_result


def validate_iso8601_timestamp(timestamp_str: str) -> Tuple[bool, Optional[datetime]]:
    """
    Validate ISO 8601 timestamp format.
    
    Args:
        timestamp_str: Timestamp string to validate
        
    Returns:
        Tuple of (is_valid, parsed_datetime)
    """
    if not timestamp_str:
        return False, None
    
    try:
        # Handle 'Z' suffix for UTC
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        
        dt = datetime.fromisoformat(timestamp_str)
        return True, dt
    except ValueError:
        return False, None


def validate_dataset_completeness(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate completeness of entire dataset.
    
    Args:
        segments: List of speech segments
        
    Returns:
        Completeness validation report
    """
    if not segments:
        return {
            'total_segments': 0,
            'completeness_score': 0.0,
            'field_completeness': {},
            'issues': ['empty_dataset']
        }
    
    required_fields = [
        'speaker_name', 'speaker_country', 'speaker_party_or_group',
        'segment_start_ts', 'segment_end_ts', 'speech_text',
        'is_announcement', 'announcement_label'
    ]
    
    field_completeness = {}
    
    for field in required_fields:
        complete_count = 0
        for segment in segments:
            value = segment.get(field)
            if value is not None and str(value).strip():
                complete_count += 1
        
        completeness_rate = complete_count / len(segments)
        field_completeness[field] = completeness_rate
    
    overall_completeness = sum(field_completeness.values()) / len(field_completeness)
    
    # Identify issues
    issues = []
    for field, rate in field_completeness.items():
        if rate < 0.9:  # Less than 90% complete
            issues.append(f'low_completeness_{field}')
    
    return {
        'total_segments': len(segments),
        'completeness_score': overall_completeness,
        'field_completeness': field_completeness,
        'issues': issues
    }


def validate_speaker_resolution_quality(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate quality of speaker resolution.
    
    Args:
        segments: List of speech segments
        
    Returns:
        Speaker resolution quality report
    """
    if not segments:
        return {'error': 'no_segments'}
    
    confidence_scores = []
    resolution_methods = {}
    mep_count = 0
    
    for segment in segments:
        # Collect confidence scores
        confidence = segment.get('confidence')
        if confidence is not None:
            confidence_scores.append(confidence)
        
        # Count resolution methods
        method = segment.get('resolution_method', 'unknown')
        resolution_methods[method] = resolution_methods.get(method, 0) + 1
        
        # Count MEP resolutions
        if method == 'mep_database':
            mep_count += 1
    
    # Calculate metrics
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    high_confidence_rate = sum(1 for c in confidence_scores if c > 0.8) / len(confidence_scores) if confidence_scores else 0.0
    mep_resolution_rate = mep_count / len(segments)
    
    return {
        'total_segments': len(segments),
        'average_confidence': avg_confidence,
        'high_confidence_rate': high_confidence_rate,
        'mep_resolution_rate': mep_resolution_rate,
        'resolution_methods': resolution_methods
    }


def generate_validation_sample(segments: List[Dict[str, Any]], sample_size: int = 25) -> List[Dict[str, Any]]:
    """
    Generate stratified sample for manual validation.
    
    Args:
        segments: All segments
        sample_size: Target sample size
        
    Returns:
        List of segments selected for validation
    """
    if not segments:
        return []
    
    import random
    
    # Separate announcements and speeches
    announcements = [s for s in segments if s.get('is_announcement', False)]
    speeches = [s for s in segments if not s.get('is_announcement', False)]
    
    # Calculate sample distribution
    total_segments = len(segments)
    announcement_ratio = len(announcements) / total_segments if total_segments > 0 else 0
    
    announcement_sample_size = int(sample_size * announcement_ratio)
    speech_sample_size = sample_size - announcement_sample_size
    
    sample_segments = []
    
    # Sample announcements
    if announcements and announcement_sample_size > 0:
        ann_sample = random.sample(announcements, min(announcement_sample_size, len(announcements)))
        sample_segments.extend(ann_sample)
    
    # Sample speeches
    if speeches and speech_sample_size > 0:
        speech_sample = random.sample(speeches, min(speech_sample_size, len(speeches)))
        sample_segments.extend(speech_sample)
    
    # Add validation metadata
    for i, segment in enumerate(sample_segments):
        segment['validation_id'] = f"SAMPLE_{i+1:03d}"
        segment['validation_type'] = 'announcement' if segment.get('is_announcement') else 'speech'
    
    logger.info("Generated validation sample", 
               total_size=len(sample_segments),
               announcements=announcement_sample_size,
               speeches=speech_sample_size)
    
    return sample_segments


def validate_pydantic_model(segment_dict: Dict[str, Any]) -> Tuple[bool, Optional[SpeechSegment], List[str]]:
    """
    Validate segment against Pydantic model.
    
    Args:
        segment_dict: Segment data dictionary
        
    Returns:
        Tuple of (is_valid, speech_segment_object, validation_errors)
    """
    try:
        speech_segment = SpeechSegment(**segment_dict)
        return True, speech_segment, []
    except Exception as e:
        error_msg = str(e)
        errors = [error_msg]
        return False, None, errors