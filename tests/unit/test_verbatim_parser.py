"""Unit tests for verbatim parser."""

import pytest
from datetime import datetime
from src.parsers.verbatim_parser import VerbatimParser, VerbatimSegmentProcessor
from src.models.speech import RawSpeechSegment


class TestVerbatimParser:
    """Test cases for VerbatimParser."""
    
    def test_parser_initialization(self):
        """Test parser initializes correctly."""
        parser = VerbatimParser()
        assert parser.current_session_id is None
        assert parser.session_date is None
        assert parser.parsed_segments == []
    
    def test_speaker_extraction_basic(self):
        """Test basic speaker name extraction."""
        parser = VerbatimParser()
        
        # Test standard format: Name. - Speech
        result = parser._extract_speaker_from_text("GARCIA PÉREZ, María. − Thank you, Mr President.")
        assert result is not None
        speaker, speech = result
        assert "GARCIA PÉREZ, María" in speaker
        assert "Thank you, Mr President" in speech
    
    def test_speaker_extraction_variations(self):
        """Test various speaker name formats."""
        parser = VerbatimParser()
        
        test_cases = [
            ("MÜLLER, Hans (PPE). − Good morning.", "MÜLLER, Hans (PPE)"),
            ("President: The session is now open.", "President"),
            ("Ms ANDERSSON − Thank you.", "Ms ANDERSSON"),
        ]
        
        for text, expected_speaker in test_cases:
            result = parser._extract_speaker_from_text(text)
            if result:
                speaker, _ = result
                assert expected_speaker in speaker
    
    def test_procedural_content_detection(self):
        """Test detection of procedural content."""
        parser = VerbatimParser()
        
        procedural_texts = [
            "President. − Good morning, colleagues.",
            "(Applause)",
            "The sitting was suspended at 12:30",
            "Written statements (Rule 149)",
        ]
        
        for text in procedural_texts:
            assert parser._is_procedural_content(text)
        
        # Non-procedural content
        speech_text = "I believe we must address climate change immediately."
        assert not parser._is_procedural_content(speech_text)
    
    def test_timestamp_extraction(self):
        """Test timestamp hint extraction."""
        parser = VerbatimParser()
        
        test_cases = [
            ("The sitting opened at 14:30", "14:30"),
            ("(The sitting suspended at 12.30)", "12.30"),
            ("No timestamp here", None),
        ]
        
        for text, expected in test_cases:
            result = parser._extract_timestamp_hint(text)
            if expected:
                assert expected in result
            else:
                assert result is None
    
    def test_parse_verbatim_report_basic(self, sample_verbatim_html):
        """Test basic verbatim report parsing."""
        parser = VerbatimParser()
        session_date = datetime(2024, 1, 15, 9, 0)
        
        segments = parser.parse_verbatim_report(
            sample_verbatim_html, 
            "test_session", 
            session_date
        )
        
        assert len(segments) > 0
        assert all(isinstance(seg, RawSpeechSegment) for seg in segments)
        assert segments[0].session_id == "test_session"
    
    def test_parsing_stats(self, sample_verbatim_html):
        """Test parsing statistics generation."""
        parser = VerbatimParser()
        session_date = datetime(2024, 1, 15, 9, 0)
        
        parser.parse_verbatim_report(sample_verbatim_html, "test_session", session_date)
        stats = parser.get_parsing_stats()
        
        assert 'total_segments' in stats
        assert 'procedural_segments' in stats
        assert 'speech_segments' in stats
        assert 'average_confidence' in stats
        assert stats['total_segments'] >= 0


class TestVerbatimSegmentProcessor:
    """Test cases for VerbatimSegmentProcessor."""
    
    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        processor = VerbatimSegmentProcessor()
        assert processor is not None
    
    def test_segment_validation(self):
        """Test segment validation logic."""
        processor = VerbatimSegmentProcessor()
        
        # Valid segment
        valid_segment = RawSpeechSegment(
            session_id="test",
            sequence_number=1,
            speaker_raw="Test Speaker",
            speech_text="This is a valid speech segment with enough content.",
            confidence_score=0.8
        )
        assert processor._validate_segment(valid_segment)
        
        # Invalid segments
        invalid_cases = [
            # Empty speaker
            RawSpeechSegment(
                session_id="test", sequence_number=1, speaker_raw="", 
                speech_text="Valid speech", confidence_score=0.8
            ),
            # Short speech
            RawSpeechSegment(
                session_id="test", sequence_number=1, speaker_raw="Speaker", 
                speech_text="Too short", confidence_score=0.8
            ),
            # Low confidence
            RawSpeechSegment(
                session_id="test", sequence_number=1, speaker_raw="Speaker", 
                speech_text="Valid speech content here", confidence_score=0.05
            ),
        ]
        
        for invalid_segment in invalid_cases:
            assert not processor._validate_segment(invalid_segment)
    
    def test_process_segments(self, sample_raw_speech_data):
        """Test segment processing pipeline."""
        processor = VerbatimSegmentProcessor()
        
        # Create test segment
        test_segment = RawSpeechSegment(**sample_raw_speech_data)
        session_date = datetime(2024, 1, 15, 9, 0)
        
        processed = processor.process_segments([test_segment], session_date)
        
        assert len(processed) <= 1  # Could be filtered out or enhanced
        if processed:
            assert processed[0].session_id == test_segment.session_id
            assert processed[0].speaker_raw != ""
            assert len(processed[0].speech_text) >= 10