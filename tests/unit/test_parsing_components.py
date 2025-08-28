"""Unit tests for parsing components with real-world data scenarios."""

import pytest
from datetime import datetime
from typing import List

from src.parsers.verbatim_parser import VerbatimParser, ParsedSegment
from src.parsers.speech_classifier import SpeechClassifier, ContentType
from src.validators.parsing_validator import ParsingValidator, ValidationSeverity
from src.models.speech import RawSpeechSegment
from src.core.logging import get_logger

logger = get_logger(__name__)


class TestVerbatimParser:
    """Test cases for the enhanced verbatim parser."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return VerbatimParser()
    
    @pytest.fixture
    def sample_html_complex(self):
        """Complex HTML sample with real EU Parliament patterns."""
        return """
        <html>
        <body>
            <div class="contents">
                <p>The sitting opened at 09:00</p>
                <p><strong>President.</strong> − Good morning, colleagues. We shall now proceed with the agenda.</p>
                <p>GARCÍA PÉREZ, María (PPE). − Mr President, I would like to address the critical issue of climate change policy. 
                The Commission's recent report clearly demonstrates that we need immediate and comprehensive action across all member states. 
                This is not just an environmental issue, but a fundamental question of economic sustainability and social justice.</p>
                <p>We must ensure that our transition to renewable energy sources does not leave any community behind.</p>
                <p><strong>MÜLLER, Hans (S&D).</strong> − Thank you, Mr President. I completely agree with my colleague from the PPE Group. 
                However, I would like to add that we also need to consider the social impact of these policies on workers in traditional energy sectors.</p>
                <p>(Applause from the S&D Group)</p>
                <p><strong>President.</strong> − Thank you. Next item on the agenda is the report by Ms Schmidt on digital transformation.</p>
                <p>SCHMIDT, Anna (Renew). − Madam President, digital transformation is reshaping our economies and societies at an unprecedented pace.</p>
                <p>(The sitting was suspended at 12:30 for lunch break)</p>
            </div>
        </body>
        </html>
        """
    
    @pytest.fixture
    def sample_html_procedural(self):
        """HTML with heavy procedural content."""
        return """
        <html>
        <body>
            <div class="doceo-content">
                <p><strong>President.</strong> − The sitting is opened.</p>
                <p>Written statements (Rule 171)</p>
                <p>Voting time</p>
                <p>The vote is taken on Amendment 1</p>
                <p>(Amendment 1 is adopted by 345 votes to 234, with 45 abstentions)</p>
                <p><strong>President.</strong> − That concludes the voting time.</p>
                <p>Explanations of vote</p>
                <p>Point of order by Mr Johnson</p>
                <p>JOHNSON, Robert (ECR). − On a point of order, Madam President.</p>
                <p><strong>President.</strong> − The sitting is suspended.</p>
            </div>
        </body>
        </html>
        """
    
    def test_parse_complex_session(self, parser, sample_html_complex):
        """Test parsing of complex session with mixed content."""
        session_id = "test_session_2024_01_15"
        session_date = datetime(2024, 1, 15, 9, 0)
        
        segments = parser.parse_verbatim_report(sample_html_complex, session_id, session_date)
        
        # Should extract multiple segments
        assert len(segments) >= 5
        
        # First segment should be President opening
        assert segments[0].speaker_raw == "President"
        assert "good morning" in segments[0].speech_text.lower()
        assert segments[0].is_procedural
        
        # Find García Pérez segment
        garcia_segments = [s for s in segments if "garcía" in s.speaker_raw.lower()]
        assert len(garcia_segments) >= 1
        
        garcia_seg = garcia_segments[0]
        assert "climate change" in garcia_seg.speech_text.lower()
        assert not garcia_seg.is_procedural
        assert garcia_seg.confidence_score > 0.7
        
        # Should detect timestamps
        timestamped = [s for s in segments if s.timestamp_hint]
        assert len(timestamped) >= 1
    
    def test_parse_procedural_heavy(self, parser, sample_html_procedural):
        """Test parsing of procedural-heavy content."""
        session_id = "test_procedural_2024_01_15"
        session_date = datetime(2024, 1, 15, 10, 0)
        
        segments = parser.parse_verbatim_report(sample_html_procedural, session_id, session_date)
        
        # Most segments should be procedural
        procedural_count = sum(1 for s in segments if s.is_procedural)
        assert procedural_count >= len(segments) * 0.7  # At least 70% procedural
        
        # Should identify voting procedures
        voting_segments = [s for s in segments if "voting" in s.speech_text.lower()]
        assert len(voting_segments) >= 1
        
        # Johnson point of order should not be procedural
        johnson_segments = [s for s in segments if "johnson" in s.speaker_raw.lower()]
        if johnson_segments:
            assert not johnson_segments[0].is_procedural
    
    def test_enhanced_speaker_patterns(self, parser):
        """Test enhanced speaker recognition patterns."""
        test_cases = [
            ('<strong>García Pérez, María (PPE)</strong>. − Thank you.', 'García Pérez, María'),
            ('MÜLLER, Hans (S&D). − I agree completely.', 'MÜLLER, Hans'),
            ('Vice-President Timmermans. − The Commission position is clear.', 'Vice-President Timmermans'),
            ('The President. − We shall now proceed.', 'The President'),
            ('President of the European Parliament. − Welcome.', 'President of the European Parliament'),
            ('LÓPEZ-ISTÚRIZ WHITE, Antonio (PPE): In Spain, we face challenges.', 'LÓPEZ-ISTÚRIZ WHITE, Antonio')
        ]
        
        for test_text, expected_speaker in test_cases:
            result = parser._extract_speaker_enhanced(test_text, type('MockElement', (), {'find': lambda self, tags: None})())
            assert result is not None, f"Failed to extract speaker from: {test_text}"
            assert expected_speaker.lower() in result[0].lower(), f"Expected {expected_speaker}, got {result[0]}"
    
    def test_enhanced_procedural_detection(self, parser):
        """Test enhanced procedural content detection."""
        procedural_cases = [
            "The sitting opened at 09:00",
            "Voting time",
            "(Applause from the PPE Group)",
            "Next item on the agenda",
            "Written statements (Rule 171)",
            "Point of order by Mr Smith",
            "The debate is closed",
            "Ladies and gentlemen, we shall proceed"
        ]
        
        speech_cases = [
            "Mr President, I believe we must address climate change urgently.",
            "The Commission's proposal fails to address the concerns of our constituents.",
            "In my country, we have implemented successful policies in this area.",
            "I would like to thank the rapporteur for this excellent work.",
            "This directive will have significant impact on small businesses."
        ]
        
        for text in procedural_cases:
            assert parser._is_procedural_content_enhanced(text), f"Should be procedural: {text}"
        
        for text in speech_cases:
            assert not parser._is_procedural_content_enhanced(text), f"Should not be procedural: {text}"
    
    def test_parsing_stats_collection(self, parser, sample_html_complex):
        """Test that parsing statistics are collected correctly."""
        session_id = "test_stats_2024_01_15"
        session_date = datetime(2024, 1, 15, 9, 0)
        
        segments = parser.parse_verbatim_report(sample_html_complex, session_id, session_date)
        stats = parser.get_enhanced_parsing_stats()
        
        assert stats['parsing_method'] == 'enhanced_patterns_v2'
        assert stats['total_segments'] == len(segments)
        assert stats['raw_stats']['total_elements'] > 0
        assert stats['raw_stats']['speaker_patterns_matched'] > 0
        assert 'quality_metrics' in stats


class TestSpeechClassifier:
    """Test cases for the speech classifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return SpeechClassifier()
    
    @pytest.fixture
    def sample_segments(self):
        """Create sample segments for testing."""
        return [
            RawSpeechSegment(
                session_id="test_session_2024",
                sequence_number=1,
                speaker_raw="President",
                speech_text="The sitting is opened. We shall proceed with the agenda.",
                is_procedural=True,
                confidence_score=0.9
            ),
            RawSpeechSegment(
                session_id="test_session_2024",
                sequence_number=2,
                speaker_raw="García Pérez, María (PPE)",
                speech_text="Mr President, I would like to address the critical issue of climate change policy. We need comprehensive action across all member states.",
                is_procedural=False,
                confidence_score=0.85
            ),
            RawSpeechSegment(
                session_id="test_session_2024",
                sequence_number=3,
                speaker_raw="President",
                speech_text="Voting time. The vote is taken on Amendment 1.",
                is_procedural=True,
                confidence_score=0.95
            ),
            RawSpeechSegment(
                session_id="test_session_2024",
                sequence_number=4,
                speaker_raw="MÜLLER, Hans (S&D)",
                speech_text="Thank you, Mr President. I completely agree with my colleague. However, we must also consider the social impact on workers.",
                is_procedural=False,
                confidence_score=0.8
            )
        ]
    
    def test_classify_single_segment_speech(self, classifier, sample_segments):
        """Test classification of speech segments."""
        speech_segment = sample_segments[1]  # García Pérez segment
        result = classifier.classify_segment(speech_segment)
        
        assert result.content_type == ContentType.SPEECH
        assert result.confidence > 0.6
        assert "policy" in result.reasoning or "speech" in result.reasoning
    
    def test_classify_single_segment_procedural(self, classifier, sample_segments):
        """Test classification of procedural segments."""
        procedural_segment = sample_segments[2]  # Voting time segment
        result = classifier.classify_segment(procedural_segment)
        
        assert result.content_type in [ContentType.PROCEDURAL, ContentType.ANNOUNCEMENT]
        assert result.confidence > 0.7
    
    def test_classify_batch_with_context(self, classifier, sample_segments):
        """Test batch classification with context awareness."""
        results = classifier.classify_batch(sample_segments)
        
        assert len(results) == len(sample_segments)
        
        # First segment (President opening) should be announcement
        assert results[0].content_type == ContentType.ANNOUNCEMENT
        
        # Speech segments should be classified as speech
        speech_results = [r for i, r in enumerate(results) if not sample_segments[i].is_procedural]
        for result in speech_results:
            assert result.content_type == ContentType.SPEECH
    
    def test_classification_stats(self, classifier, sample_segments):
        """Test classification statistics generation."""
        results = classifier.classify_batch(sample_segments)
        stats = classifier.get_classification_stats(results)
        
        assert stats['total_segments'] == len(sample_segments)
        assert 'content_type_distribution' in stats
        assert 'average_confidence' in stats
        assert 'high_confidence_count' in stats
    
    def test_speech_indicators_patterns(self, classifier):
        """Test speech indicator patterns."""
        speech_texts = [
            "Mr President, I believe we must act now",
            "In my opinion, this proposal is flawed",
            "We must ensure that our citizens are protected",
            "I would like to thank the rapporteur for this excellent work",
            "This directive will have significant impact on small businesses"
        ]
        
        for text in speech_texts:
            # Create test segment
            segment = RawSpeechSegment(
                session_id="test", sequence_number=1, speaker_raw="Test Speaker",
                speech_text=text, confidence_score=0.8
            )
            result = classifier.classify_segment(segment)
            
            # Should lean towards speech classification
            assert result.content_type == ContentType.SPEECH or result.confidence < 0.8


class TestParsingValidator:
    """Test cases for the parsing validator."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ParsingValidator()
    
    @pytest.fixture
    def valid_segment(self):
        """Create a valid segment for testing."""
        return RawSpeechSegment(
            session_id="test_session_2024",
            sequence_number=1,
            speaker_raw="García Pérez, María (PPE)",
            speech_text="Mr President, I would like to address the important issue of environmental policy. The European Union must take decisive action to meet our climate goals.",
            timestamp_hint="09:15",
            is_procedural=False,
            confidence_score=0.85,
            parsing_metadata={
                'parser_version': '2.0',
                'position': 0,
                'parsing_method': 'enhanced_patterns',
                'procedural_detected': False,
                'timestamp_extracted': True
            }
        )
    
    @pytest.fixture
    def invalid_segments(self):
        """Create segments with various validation issues."""
        return [
            # Empty speaker
            RawSpeechSegment(
                session_id="test_session_2024", sequence_number=1, speaker_raw="X",  # Minimal speaker
                speech_text="Some content", confidence_score=0.5
            ),
            # Too short content
            RawSpeechSegment(
                session_id="test_session_2024", sequence_number=2, speaker_raw="Speaker",
                speech_text="Short text", confidence_score=0.5  # Still short but meets min_length
            ),
            # Invalid timestamp
            RawSpeechSegment(
                session_id="test_session_2024", sequence_number=3, speaker_raw="Speaker",
                speech_text="Good content here with sufficient length",
                timestamp_hint="25:70", confidence_score=0.5
            ),
            # Low confidence
            RawSpeechSegment(
                session_id="test_session_2024", sequence_number=4, speaker_raw="Speaker",
                speech_text="Content with sufficient length for validation",
                confidence_score=0.1
            )
        ]
    
    def test_validate_valid_segment(self, validator, valid_segment):
        """Test validation of a valid segment."""
        result = validator.validate_segment(valid_segment)
        
        assert result.is_valid
        assert result.quality_score > 0.7
        assert len(result.issues) == 0
    
    def test_validate_invalid_segments(self, validator, invalid_segments):
        """Test validation of segments with issues."""
        for segment in invalid_segments:
            result = validator.validate_segment(segment)
            
            # Should have issues
            assert len(result.issues) > 0
            
            # Quality score should reflect issues
            if any(issue.severity == ValidationSeverity.CRITICAL for issue in result.issues):
                assert not result.is_valid
    
    def test_validate_batch_consistency(self, validator, valid_segment):
        """Test batch validation with consistency checks."""
        # Create batch with duplicate sequence numbers
        segments = [
            valid_segment,
            RawSpeechSegment(
                session_id="test_session_2024", sequence_number=1,  # Duplicate!
                speaker_raw="Another Speaker", 
                speech_text="Different content with sufficient length",
                confidence_score=0.8
            )
        ]
        
        result = validator.validate_batch(segments)
        
        # Should detect duplicate sequence numbers
        structural_issues = [i for i in result.issues 
                           if i.category.value == 'structural_integrity']
        assert len(structural_issues) > 0
    
    def test_cross_segment_validation(self, validator):
        """Test cross-segment validation features."""
        segments = [
            RawSpeechSegment(
                session_id="session1", sequence_number=1, speaker_raw="Speaker A",
                speech_text="First segment content", confidence_score=0.8
            ),
            RawSpeechSegment(
                session_id="session2", sequence_number=2, speaker_raw="Speaker B",  # Different session!
                speech_text="Second segment content", confidence_score=0.8
            )
        ]
        
        result = validator.validate_batch(segments)
        
        # Should detect mixed session IDs
        session_issues = [i for i in result.issues 
                         if 'session' in i.message.lower()]
        assert len(session_issues) > 0
    
    def test_timestamp_sequence_validation(self, validator):
        """Test timestamp sequence validation."""
        segments = [
            RawSpeechSegment(
                session_id="test", sequence_number=1, speaker_raw="Speaker A",
                speech_text="Content at 10:00", timestamp_hint="10:00", confidence_score=0.8
            ),
            RawSpeechSegment(
                session_id="test", sequence_number=2, speaker_raw="Speaker B",
                speech_text="Content at 09:30", timestamp_hint="09:30", confidence_score=0.8  # Goes backward!
            )
        ]
        
        result = validator.validate_batch(segments)
        
        # Should detect timestamp going backwards
        timestamp_issues = [i for i in result.issues 
                          if 'backward' in i.message.lower()]
        assert len(timestamp_issues) > 0
    
    def test_validation_report_generation(self, validator, valid_segment, invalid_segments):
        """Test validation report generation."""
        all_segments = [valid_segment] + invalid_segments
        result = validator.validate_batch(all_segments)
        
        report = validator.generate_validation_report(result, include_details=True)
        
        assert "EU PARLIAMENT PARSING VALIDATION REPORT" in report
        assert "SUMMARY:" in report
        assert "Quality Score:" in report
        assert "DETAILED ISSUES:" in report
        
        # Should include issue details
        for segment in invalid_segments:
            assert str(segment.sequence_number) in report
    
    def test_quality_score_calculation(self, validator):
        """Test quality score calculation logic."""
        # High quality segment
        high_quality = RawSpeechSegment(
            session_id="test", sequence_number=1,
            speaker_raw="García Pérez, María (PPE)",
            speech_text="Mr President, I would like to address the comprehensive environmental policy framework that we need to implement across all member states to achieve our climate neutrality goals by 2050.",
            timestamp_hint="09:15",
            confidence_score=0.9,
            parsing_metadata={'complete': True}
        )
        
        result = validator.validate_segment(high_quality)
        assert result.quality_score > 0.8
        
        # Low quality segment
        low_quality = RawSpeechSegment(
            session_id="test", sequence_number=1,
            speaker_raw="X",  # Very short
            speech_text="Short",  # Too short
            confidence_score=0.2  # Low confidence
        )
        
        result = validator.validate_segment(low_quality)
        assert result.quality_score < 0.4


@pytest.mark.integration
class TestParsingIntegration:
    """Integration tests for the complete parsing pipeline."""
    
    def test_complete_parsing_pipeline(self):
        """Test the complete parsing pipeline from HTML to validated segments."""
        parser = VerbatimParser()
        classifier = SpeechClassifier()
        validator = ParsingValidator()
        
        # Sample realistic EU Parliament HTML
        html_content = """
        <html>
        <body>
            <div class="contents">
                <p><strong>President.</strong> − Good morning. The sitting is opened.</p>
                <p>GARCÍA PÉREZ, María (PPE). − Mr President, climate change represents one of the greatest challenges of our time. 
                The European Green Deal provides a comprehensive framework, but we must ensure that the transition is just and inclusive. 
                Small businesses and rural communities must not be left behind in our pursuit of carbon neutrality.</p>
                <p><strong>MÜLLER, Hans (S&D).</strong> − Thank you, Mr President. I agree with my colleague, but we also need to address 
                the social dimension. The Just Transition Fund must be adequately financed to support workers in fossil fuel industries.</p>
                <p>(Applause)</p>
                <p><strong>President.</strong> − Thank you. Next item on the agenda.</p>
            </div>
        </body>
        </html>
        """
        
        session_id = "integration_test_2024"
        session_date = datetime(2024, 1, 15, 9, 0)
        
        # Step 1: Parse HTML
        raw_segments = parser.parse_verbatim_report(html_content, session_id, session_date)
        assert len(raw_segments) >= 3
        
        # Step 2: Classify segments
        classification_results = classifier.classify_batch(raw_segments)
        assert len(classification_results) == len(raw_segments)
        
        # Step 3: Validate results
        validation_result = validator.validate_batch(raw_segments)
        
        # Verify integration
        assert validation_result.segment_count == len(raw_segments)
        assert validation_result.quality_score > 0.5
        
        # Check that we have both speech and procedural content
        speech_segments = [s for s in raw_segments if not s.is_procedural]
        procedural_segments = [s for s in raw_segments if s.is_procedural]
        
        assert len(speech_segments) >= 2  # García Pérez and Müller
        assert len(procedural_segments) >= 1  # President statements
        
        # Verify parsing statistics
        stats = parser.get_enhanced_parsing_stats()
        assert stats['total_segments'] == len(raw_segments)
        assert stats['speech_segments'] >= 2
        assert stats['procedural_segments'] >= 1
        
        logger.info("Complete parsing pipeline test passed", 
                   segments_parsed=len(raw_segments),
                   validation_score=validation_result.quality_score,
                   stats=stats)
    
    def test_parsing_performance_with_large_document(self):
        """Test parsing performance with larger document."""
        parser = VerbatimParser()
        
        # Generate larger HTML document
        segments_html = []
        for i in range(50):
            segments_html.append(f'<p><strong>SPEAKER_{i:02d}, Name (PPE).</strong> − Mr President, this is speech number {i} with substantial content about European policy matters. We must address the challenges facing our union and work together for a better future for all European citizens.</p>')
        
        html_content = f"""
        <html>
        <body>
            <div class="contents">
                {''.join(segments_html)}
            </div>
        </body>
        </html>
        """
        
        session_id = "performance_test_2024"
        session_date = datetime(2024, 1, 15, 9, 0)
        
        import time
        start_time = time.time()
        
        raw_segments = parser.parse_verbatim_report(html_content, session_id, session_date)
        
        end_time = time.time()
        parsing_time = end_time - start_time
        
        # Should parse reasonable number of segments efficiently
        assert len(raw_segments) >= 40  # Most should be extracted
        assert parsing_time < 5.0  # Should complete in reasonable time
        
        # Quality should remain high even with large document
        stats = parser.get_enhanced_parsing_stats()
        assert stats['average_confidence'] > 0.7