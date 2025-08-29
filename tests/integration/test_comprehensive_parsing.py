"""
Comprehensive parsing tests with real EU Parliament data
Tests the complete parsing pipeline with realistic scenarios
"""

import pytest
from pathlib import Path
from typing import List, Dict, Any
import logging

from src.parsers.verbatim_parser import VerbatimParser
from src.parsers.speech_classifier import SpeechClassifier
from src.validators.parsing_validator import ParsingValidator
from src.models.speech import RawSpeechSegment
from src.core.logging import get_logger

logger = get_logger(__name__)

class TestComprehensiveParsing:
    """Test complete parsing pipeline with real EU Parliament data"""
    
    @pytest.fixture
    def parser(self):
        return VerbatimParser()
    
    @pytest.fixture
    def classifier(self):
        return SpeechClassifier()
    
    @pytest.fixture
    def validator(self):
        return ParsingValidator()
    
    @pytest.fixture
    def real_sample_html(self):
        """Load real EU Parliament HTML sample"""
        sample_path = Path("tests/data/real_eu_parliament_sample.html")
        return sample_path.read_text(encoding='utf-8')
    
    @pytest.fixture
    def complex_sample_html(self):
        """Load complex EU Parliament HTML sample"""
        sample_path = Path("tests/data/complex_eu_parliament_sample.html")
        return sample_path.read_text(encoding='utf-8')
    
    def test_real_parliament_data_parsing(self, parser, real_sample_html):
        """Test parsing with real EU Parliament HTML structure"""
        logger.info("Testing real EU Parliament data parsing")
        
        segments = parser.parse_html(real_sample_html, session_id="TEST-2025-01-15")
        
        # Should extract meaningful segments
        assert len(segments) >= 5, f"Expected at least 5 segments, got {len(segments)}"
        
        # Check for proper speaker identification
        speakers = [seg.speaker_raw for seg in segments]
        expected_speakers = ["President", "García Pérez", "Müller, Hans", "Rossi, Maria", "Johnson, David"]
        
        for expected_speaker in expected_speakers:
            found = any(expected_speaker in speaker for speaker in speakers)
            assert found, f"Expected speaker '{expected_speaker}' not found in {speakers}"
        
        # Verify speech content is meaningful
        speeches_with_content = [seg for seg in segments if len(seg.speech_text) > 50]
        assert len(speeches_with_content) >= 4, "Should have substantial speech content"
        
        logger.info(f"Successfully parsed {len(segments)} segments from real Parliament data")
    
    def test_complex_parliament_data_parsing(self, parser, complex_sample_html):
        """Test parsing with complex EU Parliament debate"""
        logger.info("Testing complex EU Parliament debate parsing")
        
        segments = parser.parse_html(complex_sample_html, session_id="TEST-2025-01-16")
        
        # Should extract many segments from complex debate
        assert len(segments) >= 10, f"Expected at least 10 segments, got {len(segments)}"
        
        # Check for diverse speakers with party affiliations
        speakers = [seg.speaker_raw for seg in segments]
        party_speakers = [speaker for speaker in speakers if any(party in speaker for party in ["PPE", "S&D", "Renew", "ECR", "Greens", "GUE", "ID"])]
        assert len(party_speakers) >= 6, f"Should identify multiple party-affiliated speakers, found {len(party_speakers)}"
        
        # Verify Commissioner speech is captured
        commissioner_segments = [seg for seg in segments if "Commissioner" in seg.speaker_raw]
        assert len(commissioner_segments) >= 2, "Should capture Commissioner speeches"
        
        # Check for substantial policy discussion content
        policy_segments = [seg for seg in segments if len(seg.speech_text) > 100]
        assert len(policy_segments) >= 8, "Should have substantial policy discussion content"
        
        logger.info(f"Successfully parsed {len(segments)} segments from complex debate")
    
    def test_classification_accuracy(self, parser, classifier, complex_sample_html):
        """Test speech classification accuracy on real data"""
        logger.info("Testing classification accuracy")
        
        segments = parser.parse_html(complex_sample_html, session_id="TEST-2025-01-16")
        
        speech_count = 0
        procedural_count = 0
        
        for segment in segments:
            result = classifier.classify_segment(segment)
            
            if result.category == "SPEECH":
                speech_count += 1
                # Speeches should have high confidence
                assert result.confidence >= 0.7, f"Speech confidence too low: {result.confidence}"
            elif result.category == "PROCEDURAL":
                procedural_count += 1
        
        # Should have more speeches than procedural in a debate
        assert speech_count > procedural_count, f"Expected more speeches ({speech_count}) than procedural ({procedural_count})"
        
        # Should classify substantial content correctly
        total_classified = speech_count + procedural_count
        assert total_classified >= len(segments) * 0.8, "Should classify at least 80% of segments"
        
        logger.info(f"Classification results: {speech_count} speeches, {procedural_count} procedural")
    
    def test_validation_quality_assessment(self, parser, validator, real_sample_html):
        """Test validation framework with real data"""
        logger.info("Testing validation quality assessment")
        
        segments = parser.parse_html(real_sample_html, session_id="TEST-2025-01-15")
        validation_result = validator.validate_batch(segments)
        
        # Should achieve good quality score on real data
        assert validation_result.quality_score >= 0.8, f"Quality score too low: {validation_result.quality_score}"
        
        # Should not have critical issues with real Parliament data
        critical_issues = [issue for issue in validation_result.issues if issue.severity == "CRITICAL"]
        assert len(critical_issues) == 0, f"Critical issues found: {critical_issues}"
        
        # May have minor issues but should be manageable
        total_issues = len(validation_result.issues)
        assert total_issues <= len(segments) * 0.3, f"Too many validation issues: {total_issues} for {len(segments)} segments"
        
        logger.info(f"Validation completed: quality={validation_result.quality_score:.3f}, issues={total_issues}")
    
    def test_end_to_end_pipeline(self, parser, classifier, validator, complex_sample_html):
        """Test complete parsing pipeline end-to-end"""
        logger.info("Testing end-to-end parsing pipeline")
        
        # Step 1: Parse HTML
        segments = parser.parse_html(complex_sample_html, session_id="TEST-PIPELINE")
        assert len(segments) > 0, "Parser should extract segments"
        
        # Step 2: Classify segments
        classified_segments = []
        for segment in segments:
            result = classifier.classify_segment(segment)
            segment.parsing_metadata['classification'] = {
                'category': result.category,
                'confidence': result.confidence,
                'subcategory': result.subcategory
            }
            classified_segments.append(segment)
        
        # Step 3: Validate results
        validation_result = validator.validate_batch(classified_segments)
        
        # Pipeline should produce quality results
        assert validation_result.quality_score >= 0.75, "End-to-end pipeline should achieve good quality"
        
        # Should have meaningful distribution of content types
        speech_segments = [seg for seg in classified_segments 
                          if seg.parsing_metadata.get('classification', {}).get('category') == 'SPEECH']
        assert len(speech_segments) >= len(classified_segments) * 0.6, "Should identify majority as speeches in debate"
        
        logger.info(f"End-to-end pipeline: {len(segments)} segments, "
                   f"{len(speech_segments)} speeches, quality={validation_result.quality_score:.3f}")
    
    def test_speaker_pattern_coverage(self, parser, complex_sample_html):
        """Test coverage of different speaker name patterns"""
        logger.info("Testing speaker pattern coverage")
        
        segments = parser.parse_html(complex_sample_html, session_id="TEST-PATTERNS")
        
        # Track different speaker pattern types found
        patterns_found = {
            'president': False,
            'commissioner': False,
            'party_affiliation': False,
            'full_name': False,
            'simple_name': False
        }
        
        for segment in segments:
            speaker = segment.speaker_raw.lower()
            
            if 'president' in speaker:
                patterns_found['president'] = True
            elif 'commissioner' in speaker:
                patterns_found['commissioner'] = True
            elif any(party in segment.speaker_raw for party in ['PPE', 'S&D', 'Renew', 'ECR', 'Greens', 'GUE', 'ID']):
                patterns_found['party_affiliation'] = True
            elif ',' in segment.speaker_raw:  # Format: "Surname, Name"
                patterns_found['full_name'] = True
            elif len(segment.speaker_raw.split()) >= 2:
                patterns_found['simple_name'] = True
        
        # Should recognize multiple speaker patterns
        patterns_recognized = sum(patterns_found.values())
        assert patterns_recognized >= 4, f"Should recognize at least 4 speaker patterns, found {patterns_recognized}"
        
        logger.info(f"Speaker patterns recognized: {patterns_found}")
    
    def test_multilingual_content_handling(self, parser):
        """Test handling of multilingual content with accented characters"""
        logger.info("Testing multilingual content handling")
        
        multilingual_html = """
        <div class="contents">
            <p class="contents">
                <span class="bold">García Pérez (PPE)</span>. – Señora Presidenta, quiero hablar sobre los derechos humanos.
            </p>
            <p class="contents">
                <span class="bold">Müller, Klaus (S&D)</span>. – Frau Präsidentin, ich möchte mich den Ausführungen meines Kollegen anschließen.
            </p>
            <p class="contents">
                <span class="bold">Dubois, François (Renew)</span>. – Madame la Présidente, je souhaiterais aborder la question européenne.
            </p>
        </div>
        """
        
        segments = parser.parse_html(multilingual_html, session_id="TEST-MULTILINGUAL")
        
        # Should handle accented characters correctly
        assert len(segments) == 3, f"Expected 3 multilingual segments, got {len(segments)}"
        
        # Verify accented characters are preserved
        speakers = [seg.speaker_raw for seg in segments]
        assert any('García Pérez' in speaker for speaker in speakers), "Should preserve Spanish accents"
        assert any('Müller' in speaker for speaker in speakers), "Should preserve German umlauts"
        assert any('François' in speaker for speaker in speakers), "Should preserve French accents"
        
        # Verify multilingual content is captured
        texts = [seg.speech_text for seg in segments]
        multilingual_indicators = ['Señora', 'Frau', 'Madame']
        for indicator in multilingual_indicators:
            found = any(indicator in text for text in texts)
            assert found, f"Multilingual indicator '{indicator}' not found"
        
        logger.info(f"Successfully handled multilingual content: {len(segments)} segments")
    
    def test_performance_benchmarks(self, parser, classifier, validator, complex_sample_html):
        """Test performance benchmarks for parsing pipeline"""
        import time
        
        logger.info("Testing performance benchmarks")
        
        # Benchmark parsing performance
        start_time = time.time()
        segments = parser.parse_html(complex_sample_html, session_id="PERF-TEST")
        parse_time = time.time() - start_time
        
        # Should parse reasonably quickly
        assert parse_time < 5.0, f"Parsing took too long: {parse_time:.2f}s"
        
        # Benchmark classification performance
        start_time = time.time()
        for segment in segments:
            classifier.classify_segment(segment)
        classify_time = time.time() - start_time
        
        # Should classify reasonably quickly
        assert classify_time < 3.0, f"Classification took too long: {classify_time:.2f}s"
        
        # Benchmark validation performance
        start_time = time.time()
        validator.validate_batch(segments)
        validate_time = time.time() - start_time
        
        # Should validate reasonably quickly
        assert validate_time < 2.0, f"Validation took too long: {validate_time:.2f}s"
        
        total_time = parse_time + classify_time + validate_time
        segments_per_second = len(segments) / total_time if total_time > 0 else 0
        
        logger.info(f"Performance: Parse={parse_time:.2f}s, Classify={classify_time:.2f}s, "
                   f"Validate={validate_time:.2f}s, Total={total_time:.2f}s, "
                   f"Rate={segments_per_second:.1f} segments/sec")
        
        # Should achieve reasonable throughput
        assert segments_per_second >= 5.0, f"Processing rate too slow: {segments_per_second:.1f} segments/sec"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])