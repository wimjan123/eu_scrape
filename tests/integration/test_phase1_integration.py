#!/usr/bin/env python3
"""
Phase 1 Integration Tests
Tests complete integration of all Phase 1 components working together
"""

import sys
import os
from pathlib import Path
import time
from typing import List, Dict, Any
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.parsers.verbatim_parser import VerbatimParser
from src.parsers.speech_classifier import SpeechClassifier
from src.validators.parsing_validator import ParsingValidator
from src.models.speech import RawSpeechSegment
from src.core.logging import get_logger

logger = get_logger(__name__)

class Phase1IntegrationTests:
    """Complete Phase 1 integration testing"""
    
    def __init__(self):
        self.parser = VerbatimParser()
        self.classifier = SpeechClassifier()
        self.validator = ParsingValidator()
        self.test_results = []
        
    def run_integration_test_suite(self):
        """Run complete Phase 1 integration test suite"""
        logger.info("Starting Phase 1 integration test suite")
        
        tests = [
            ("Multi-Component Pipeline Integration", self.test_multi_component_pipeline),
            ("Cross-Component Data Flow", self.test_cross_component_data_flow),
            ("Error Handling Integration", self.test_error_handling_integration),
            ("Performance Integration", self.test_performance_integration),
            ("Quality Assurance Integration", self.test_quality_integration),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Running integration test: {test_name}")
                start_time = time.time()
                
                test_func()
                
                duration = time.time() - start_time
                result = f"✅ PASS: {test_name} ({duration:.2f}s)"
                self.test_results.append(result)
                passed += 1
                logger.info(result)
                
            except Exception as e:
                result = f"❌ FAIL: {test_name} - {str(e)}"
                self.test_results.append(result)
                failed += 1
                logger.error(result)
        
        # Print integration test summary
        total = passed + failed
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print("\n" + "="*70)
        print("PHASE 1 INTEGRATION TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("="*70)
        
        for result in self.test_results:
            print(result)
        
        logger.info(f"Phase 1 integration tests completed: {passed}/{total} passed ({success_rate:.1f}%)")
        return failed == 0
    
    def test_multi_component_pipeline(self):
        """Test complete multi-component pipeline integration"""
        # Create test HTML content
        test_html = """
        <div class="contents">
            <h1>Tuesday, 15 January 2025 - Strasbourg</h1>
            <h2>1. Opening of the session</h2>
            <p class="contents">President. – Good morning. The sitting is open.</p>
            <p class="contents">García Pérez (PPE). – Madam President, I rise to speak about human rights.</p>
            <p class="contents">The situation in Myanmar remains concerning. We must take action.</p>
            <p class="contents">Müller, Hans (S&D). – Thank you, Madam President. I agree with the previous speaker.</p>
            <p class="contents">President. – The debate is closed.</p>
        </div>
        """
        
        # Step 1: Parse HTML content
        segments = self.parser.parse_verbatim_report(
            test_html, 
            session_id="INTEGRATION-TEST", 
            session_date=datetime(2025, 1, 15)
        )
        
        assert len(segments) >= 3, f"Expected at least 3 segments, got {len(segments)}"
        
        # Step 2: Classify all segments
        classified_segments = []
        for segment in segments:
            classification = self.classifier.classify_segment(segment)
            segment.parsing_metadata['classification'] = {
                'content_type': classification.content_type.value,
                'confidence': classification.confidence,
                'announcement_type': classification.announcement_type.value if classification.announcement_type else None
            }
            classified_segments.append(segment)
        
        # Step 3: Validate complete batch
        validation_result = self.validator.validate_batch(classified_segments)
        
        # Integration assertions
        assert validation_result.quality_score > 0.0, "Quality score should be positive"
        assert len(classified_segments) == len(segments), "All segments should be classified"
        
        # Check that metadata was properly added
        metadata_count = sum(1 for seg in classified_segments if 'classification' in seg.parsing_metadata)
        assert metadata_count == len(segments), "All segments should have classification metadata"
        
        logger.info(f"Multi-component pipeline: {len(segments)} segments processed, "
                   f"quality={validation_result.quality_score:.3f}")
    
    def test_cross_component_data_flow(self):
        """Test data flow between components maintains integrity"""
        test_html = """
        <div class="contents">
            <p class="contents">Commissioner Vestager. – Thank you for this opportunity to present the Commission's assessment.</p>
            <p class="contents">The European economy is showing signs of resilience despite global uncertainties.</p>
            <p class="contents">Schmidt, Andreas (PPE). – Commissioner, I would like to thank you for this overview.</p>
        </div>
        """
        
        # Parse with detailed tracking
        initial_segments = self.parser.parse_verbatim_report(
            test_html,
            session_id="DATA-FLOW-TEST",
            session_date=datetime(2025, 1, 16)
        )
        
        # Verify parsing preserved essential data
        for segment in initial_segments:
            assert segment.session_id == "DATA-FLOW-TEST", "Session ID should be preserved"
            assert isinstance(segment.sequence_number, int), "Sequence number should be integer"
            assert len(segment.speech_text.strip()) > 0, "Speech text should not be empty"
        
        # Classify and verify classification doesn't corrupt original data
        for segment in initial_segments:
            original_text = segment.speech_text
            original_speaker = segment.speaker_raw
            
            classification = self.classifier.classify_segment(segment)
            
            # Verify original data unchanged
            assert segment.speech_text == original_text, "Classification shouldn't modify speech text"
            assert segment.speaker_raw == original_speaker, "Classification shouldn't modify speaker"
            assert classification.confidence >= 0.0, "Confidence should be non-negative"
        
        # Validate and verify validation doesn't corrupt data
        pre_validation_count = len(initial_segments)
        validation_result = self.validator.validate_batch(initial_segments)
        
        assert len(initial_segments) == pre_validation_count, "Validation shouldn't modify segment count"
        assert validation_result.segment_count == pre_validation_count, "Should process all segments"
        
        logger.info(f"Cross-component data flow: {len(initial_segments)} segments maintained integrity")
    
    def test_error_handling_integration(self):
        """Test integrated error handling across components"""
        # Test with problematic but recoverable content
        problematic_html = """
        <div class="contents">
            <p class="contents">. – </p>
            <p class="contents">Some Speaker. – </p>
            <p class="contents">Valid Speaker. – This is a proper speech with sufficient content for processing.</p>
            <p class="contents"></p>
        </div>
        """
        
        # Should handle gracefully without crashing
        segments = self.parser.parse_verbatim_report(
            problematic_html,
            session_id="ERROR-TEST",
            session_date=datetime(2025, 1, 17)
        )
        
        # Should have at least one valid segment
        valid_segments = [seg for seg in segments if len(seg.speech_text.strip()) > 10]
        assert len(valid_segments) >= 1, "Should extract at least one valid segment"
        
        # Classification should handle edge cases
        for segment in segments:
            classification = self.classifier.classify_segment(segment)
            assert classification.confidence >= 0.0, "Confidence should be non-negative even for problematic content"
        
        # Validation should identify issues but not crash
        validation_result = self.validator.validate_batch(segments)
        assert len(validation_result.issues) >= 0, "Validation should complete successfully"
        assert validation_result.quality_score >= 0.0, "Quality score should be non-negative"
        
        logger.info(f"Error handling: processed {len(segments)} segments with {len(validation_result.issues)} issues")
    
    def test_performance_integration(self):
        """Test integrated performance across all components"""
        # Create larger test content
        large_html = """
        <div class="contents">
            <h1>Wednesday, 16 January 2025 - Strasbourg</h1>
        """ + "".join([
            f'<p class="contents">Speaker {i} (PPE). – This is speech number {i} with substantial content about European policy matters and important legislative issues.</p>'
            for i in range(1, 21)
        ]) + """
        </div>
        """
        
        # Measure end-to-end performance
        start_time = time.time()
        
        # Parse
        segments = self.parser.parse_verbatim_report(
            large_html,
            session_id="PERFORMANCE-TEST",
            session_date=datetime(2025, 1, 16)
        )
        parse_time = time.time() - start_time
        
        # Classify
        classify_start = time.time()
        for segment in segments:
            self.classifier.classify_segment(segment)
        classify_time = time.time() - classify_start
        
        # Validate
        validate_start = time.time()
        self.validator.validate_batch(segments)
        validate_time = time.time() - validate_start
        
        total_time = time.time() - start_time
        segments_per_second = len(segments) / total_time if total_time > 0 else 0
        
        # Performance assertions
        assert parse_time < 1.0, f"Parsing took too long: {parse_time:.2f}s"
        assert classify_time < 1.0, f"Classification took too long: {classify_time:.2f}s"  
        assert validate_time < 1.0, f"Validation took too long: {validate_time:.2f}s"
        assert segments_per_second >= 50.0, f"Processing rate too slow: {segments_per_second:.1f} segments/sec"
        
        logger.info(f"Performance integration: {len(segments)} segments in {total_time:.2f}s "
                   f"({segments_per_second:.1f} segments/sec)")
    
    def test_quality_integration(self):
        """Test integrated quality assurance across pipeline"""
        high_quality_html = """
        <div class="contents">
            <h1>Thursday, 17 January 2025 - Strasbourg</h1>
            <h2>1. Economic and Monetary Affairs</h2>
            <p class="contents">President. – The next item is the debate on economic policy.</p>
            <p class="contents">Commissioner Dombrovskis. – Madam President, honourable Members, thank you for this opportunity to discuss the current economic situation in the European Union.</p>
            <p class="contents">Our latest economic forecasts show continued growth across member states, with particular strength in domestic consumption and business investment.</p>
            <p class="contents">Schmidt, Andreas (PPE). – Commissioner, I would like to thank you for this comprehensive assessment of our economic situation.</p>
            <p class="contents">While the overall picture is indeed positive, we must not ignore the challenges facing small and medium enterprises across Europe.</p>
            <p class="contents">Dubois, Marie (S&D). – Thank you, Madam President. I appreciate the Commission's optimistic assessment, but we must also address the social dimension of economic recovery.</p>
        </div>
        """
        
        # Process with quality focus
        segments = self.parser.parse_verbatim_report(
            high_quality_html,
            session_id="QUALITY-TEST", 
            session_date=datetime(2025, 1, 17)
        )
        
        # Classify with quality tracking
        high_confidence_classifications = 0
        for segment in segments:
            classification = self.classifier.classify_segment(segment)
            if classification.confidence >= 0.4:
                high_confidence_classifications += 1
        
        # Validate with quality expectations
        validation_result = self.validator.validate_batch(segments)
        
        # Quality assertions
        assert len(segments) >= 4, "Should extract reasonable number of segments from quality content"
        assert validation_result.quality_score >= 0.4, f"Quality score too low: {validation_result.quality_score}"
        
        # At least some classifications should be high confidence
        confidence_rate = high_confidence_classifications / len(segments) if segments else 0
        assert confidence_rate >= 0.2, f"High confidence rate too low: {confidence_rate:.1%}"
        
        logger.info(f"Quality integration: {len(segments)} segments, "
                   f"quality={validation_result.quality_score:.3f}, "
                   f"high confidence={confidence_rate:.1%}")

if __name__ == "__main__":
    integration_tests = Phase1IntegrationTests()
    success = integration_tests.run_integration_test_suite()
    sys.exit(0 if success else 1)