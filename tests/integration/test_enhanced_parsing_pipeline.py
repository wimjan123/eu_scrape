"""Integration tests for enhanced parsing pipeline."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.parsers.verbatim_parser import VerbatimParser
from src.parsers.speech_classifier import SpeechClassifier
from src.validators.parsing_validator import ParsingValidator
from src.models.speech import RawSpeechSegment


@pytest.mark.integration
class TestEnhancedParsingPipeline:
    """Integration tests for the enhanced parsing system."""
    
    @pytest.fixture
    def parsing_components(self):
        """Set up parsing components."""
        return {
            'parser': VerbatimParser(),
            'classifier': SpeechClassifier(),
            'validator': ParsingValidator()
        }
    
    @pytest.fixture
    def real_world_html_sample(self):
        """Realistic HTML sample based on actual EU Parliament structure."""
        return """
        <html>
        <head>
            <title>Verbatim report - European Parliament</title>
        </head>
        <body>
            <div class="contents">
                <div class="document-content">
                    <p class="doc-title">Debates of the European Parliament</p>
                    <p class="sitting-date">Monday, 15 January 2024 - Strasbourg</p>
                    
                    <p class="sitting-info">The sitting opened at 09:00</p>
                    
                    <h3>1. Opening of the session</h3>
                    <p><strong>President.</strong> − Good morning, colleagues. I declare open the session of the European Parliament.</p>
                    <p>(The sitting opened at 09:00)</p>
                    
                    <h3>2. Climate change and environmental policy</h3>
                    <p><strong>President.</strong> − The next item is the debate on the report by Ms García Pérez on climate action.</p>
                    
                    <p><strong>GARCÍA PÉREZ, María (PPE), rapporteur.</strong> − Mr President, Commissioners, dear colleagues, 
                    climate change represents the defining challenge of our generation. The scientific evidence is overwhelming: 
                    we are facing unprecedented changes to our planet's climate system.</p>
                    
                    <p>The European Green Deal, adopted in 2019, set out our roadmap to become the first climate-neutral continent by 2050. 
                    However, the path ahead requires not just ambition, but concrete action at all levels of governance.</p>
                    
                    <p>In my home country of Spain, we have witnessed firsthand the devastating effects of extreme weather events. 
                    From prolonged droughts affecting our agricultural sector to unprecedented flooding in urban areas, 
                    the impacts are already being felt by our citizens.</p>
                    
                    <p><strong>MÜLLER, Hans (S&D).</strong> − Thank you, Mr President. I would like to thank the rapporteur for her excellent work on this vital issue.</p>
                    
                    <p>While I fully support the ambitious targets set out in the European Green Deal, we must ensure that the transition 
                    is socially just and leaves no one behind. The Just Transition Fund, worth €17.5 billion, is a crucial instrument, 
                    but it must be properly implemented to support workers and communities dependent on fossil fuel industries.</p>
                    
                    <p>In Germany, we have experience with coal phase-out, and we know that successful transition requires early dialogue 
                    with stakeholders, comprehensive retraining programs, and alternative economic opportunities.</p>
                    
                    <p>(Applause from the S&D Group)</p>
                    
                    <p><strong>ANDERSON, Sarah (Renew).</strong> − Mr President, I want to focus on the innovation aspect of our climate response.</p>
                    
                    <p>The European Union must lead in developing clean technologies. Our Horizon Europe program allocates €95 billion for research and innovation, 
                    with climate and environmental goals as cross-cutting priorities. But we need to accelerate the deployment of these technologies.</p>
                    
                    <p><strong>President.</strong> − Thank you. I now call on Commissioner Timmermans.</p>
                    
                    <p><strong>TIMMERMANS (Commission).</strong> − Mr President, honourable Members, I want to thank all speakers for their contributions.</p>
                    
                    <p>The Commission is fully committed to implementing the European Green Deal. We have already presented over 50 legislative proposals, 
                    including the Climate Law, which makes our 2050 climate neutrality target legally binding.</p>
                    
                    <p>(Interruption by Mr Johnson from the ECR Group)</p>
                    
                    <p><strong>JOHNSON, Robert (ECR).</strong> − On a point of order, Mr President. I believe the Commissioner is exceeding his allocated speaking time.</p>
                    
                    <p><strong>President.</strong> − Thank you, Mr Johnson. Commissioner, please continue.</p>
                    
                    <p><strong>TIMMERMANS (Commission).</strong> − As I was saying, the Fit for 55 package represents the most comprehensive climate legislation ever proposed. 
                    It will reshape our economy and put us on track to reduce emissions by at least 55% by 2030.</p>
                    
                    <p>(The sitting was suspended at 12:30 and resumed at 14:30)</p>
                    
                    <h3>3. Voting time</h3>
                    <p><strong>President.</strong> − We shall now proceed to the vote.</p>
                    <p>Voting time</p>
                    
                    <p>The vote is taken on the García Pérez report (A9-0001/2024)</p>
                    <p>(Parliament adopted the resolution by 356 votes to 275, with 63 abstentions)</p>
                    
                    <p><strong>President.</strong> − That concludes the voting. The sitting is closed.</p>
                    <p>(The sitting closed at 18:00)</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def test_enhanced_parser_real_world_content(self, parsing_components, real_world_html_sample):
        """Test enhanced parser with realistic EU Parliament content."""
        parser = parsing_components['parser']
        session_id = "EP_2024_01_15_AM"
        session_date = datetime(2024, 1, 15, 9, 0)
        
        # Parse the realistic HTML
        segments = parser.parse_verbatim_report(real_world_html_sample, session_id, session_date)
        
        # Should extract significant number of segments
        assert len(segments) >= 10
        
        # Check for key speakers
        speakers = [s.speaker_raw for s in segments]
        assert any("garcía" in speaker.lower() for speaker in speakers)
        assert any("müller" in speaker.lower() for speaker in speakers)
        assert any("timmermans" in speaker.lower() for speaker in speakers)
        assert any("president" in speaker.lower() for speaker in speakers)
        
        # Check content quality
        speech_segments = [s for s in segments if not s.is_procedural]
        assert len(speech_segments) >= 5
        
        # Verify long speeches are captured properly
        long_speeches = [s for s in speech_segments if len(s.speech_text.split()) > 50]
        assert len(long_speeches) >= 2
        
        # Check timestamp extraction
        timestamped = [s for s in segments if s.timestamp_hint]
        assert len(timestamped) >= 2  # Should capture opening and closing times
        
        # Verify parsing statistics
        stats = parser.get_enhanced_parsing_stats()
        assert stats['parsing_method'] == 'enhanced_patterns_v2'
        assert stats['average_confidence'] > 0.7
        assert stats['timestamp_coverage_pct'] > 10
    
    def test_classifier_accuracy_real_content(self, parsing_components, real_world_html_sample):
        """Test classifier accuracy with real-world content."""
        parser = parsing_components['parser']
        classifier = parsing_components['classifier']
        
        session_id = "EP_2024_01_15_AM"
        session_date = datetime(2024, 1, 15, 9, 0)
        
        # Parse and classify
        segments = parser.parse_verbatim_report(real_world_html_sample, session_id, session_date)
        classifications = classifier.classify_batch(segments)
        
        assert len(classifications) == len(segments)
        
        # Check classification distribution
        content_types = [c.content_type.value for c in classifications]
        
        # Should have mix of content types
        assert 'speech' in content_types
        assert 'announcement' in content_types or 'procedural' in content_types
        
        # High confidence classifications
        high_confidence = [c for c in classifications if c.confidence >= 0.8]
        assert len(high_confidence) >= len(classifications) * 0.5  # At least 50% high confidence
        
        # Verify specific classifications
        for i, segment in enumerate(segments):
            classification = classifications[i]
            
            # Long policy speeches should be classified as speech
            if (len(segment.speech_text.split()) > 100 and 
                not segment.is_procedural and 
                any(keyword in segment.speech_text.lower() for keyword in ['policy', 'climate', 'transition'])):
                assert classification.content_type.value == 'speech'
            
            # Voting procedures should be procedural/announcement
            if 'voting' in segment.speech_text.lower() or 'vote is taken' in segment.speech_text.lower():
                assert classification.content_type.value in ['procedural', 'announcement']
    
    def test_validator_comprehensive_analysis(self, parsing_components, real_world_html_sample):
        """Test comprehensive validation with real content."""
        parser = parsing_components['parser']
        validator = parsing_components['validator']
        
        session_id = "EP_2024_01_15_AM"
        session_date = datetime(2024, 1, 15, 9, 0)
        
        # Parse and validate
        segments = parser.parse_verbatim_report(real_world_html_sample, session_id, session_date)
        validation_result = validator.validate_batch(segments)
        
        # Overall validation should pass
        assert validation_result.is_valid or validation_result.quality_score > 0.7
        
        # Quality metrics
        assert validation_result.quality_score > 0.6
        assert validation_result.passed_count > 0
        
        # Check validation coverage
        summary = validation_result.get_summary()
        assert summary['total_segments'] == len(segments)
        
        # Generate validation report
        report = validator.generate_validation_report(validation_result, include_details=True)
        assert "EU PARLIAMENT PARSING VALIDATION REPORT" in report
        assert "Quality Score:" in report
        
        # Should have reasonable error rates
        if validation_result.errors_count > 0:
            error_rate = validation_result.errors_count / len(segments)
            assert error_rate < 0.3  # Less than 30% error rate
    
    def test_end_to_end_pipeline_performance(self, parsing_components, real_world_html_sample):
        """Test complete pipeline performance and accuracy."""
        parser = parsing_components['parser']
        classifier = parsing_components['classifier']
        validator = parsing_components['validator']
        
        session_id = "EP_2024_01_15_AM"
        session_date = datetime(2024, 1, 15, 9, 0)
        
        import time
        
        # Measure parsing performance
        start_time = time.time()
        segments = parser.parse_verbatim_report(real_world_html_sample, session_id, session_date)
        parsing_time = time.time() - start_time
        
        # Measure classification performance
        start_time = time.time()
        classifications = classifier.classify_batch(segments)
        classification_time = time.time() - start_time
        
        # Measure validation performance
        start_time = time.time()
        validation = validator.validate_batch(segments)
        validation_time = time.time() - start_time
        
        # Performance assertions
        assert parsing_time < 2.0  # Should parse quickly
        assert classification_time < 1.0  # Classification should be fast
        assert validation_time < 1.0  # Validation should be fast
        
        # Quality assertions
        assert len(segments) >= 10  # Should extract reasonable number of segments
        assert validation.quality_score > 0.6  # Should achieve good quality
        
        # Accuracy assertions
        speech_segments = [s for s in segments if not s.is_procedural]
        procedural_segments = [s for s in segments if s.is_procedural]
        
        assert len(speech_segments) >= 5  # Should identify substantial speeches
        assert len(procedural_segments) >= 3  # Should identify procedural content
        
        # Statistical validation
        stats = parser.get_enhanced_parsing_stats()
        class_stats = classifier.get_classification_stats(classifications)
        
        assert stats['average_confidence'] > 0.7
        assert class_stats['average_confidence'] > 0.6
        assert class_stats['high_confidence_count'] >= len(segments) * 0.4
        
        print(f"Pipeline Performance Summary:")
        print(f"  - Parsing time: {parsing_time:.3f}s")
        print(f"  - Classification time: {classification_time:.3f}s") 
        print(f"  - Validation time: {validation_time:.3f}s")
        print(f"  - Segments extracted: {len(segments)}")
        print(f"  - Speech segments: {len(speech_segments)}")
        print(f"  - Procedural segments: {len(procedural_segments)}")
        print(f"  - Overall quality: {validation.quality_score:.3f}")
    
    def test_edge_cases_and_robustness(self, parsing_components):
        """Test parser robustness with edge cases."""
        parser = parsing_components['parser']
        validator = parsing_components['validator']
        
        edge_cases = [
            # Minimal HTML
            "<html><body><p>President. − Short session.</p></body></html>",
            
            # HTML with special characters
            """<html><body><div class="contents">
            <p><strong>LÓPEZ-ISTÚRIZ WHITE, António (PPE).</strong> − Señor Presidente, la política europea debe considerar las especificidades nacionales.</p>
            <p><strong>MÜLLER, Hans-Georg (S&D).</strong> − Herr Präsident, ich möchte auf die Ausführungen meines Kollegen antworten.</p>
            </div></body></html>""",
            
            # HTML with nested content
            """<html><body><div class="contents">
            <p><strong>President.</strong> − <em>The sitting is opened</em>.</p>
            <p>SPEAKER, Name. − Content with <a href="#">links</a> and <span class="highlight">highlighted text</span>.</p>
            </div></body></html>""",
            
            # Malformed HTML
            "<html><body><p><strong>Speaker</strong> Content without proper closure<p>More content</body></html>"
        ]
        
        session_date = datetime(2024, 1, 15, 9, 0)
        
        for i, html_content in enumerate(edge_cases):
            session_id = f"edge_case_{i}"
            
            # Should not crash
            try:
                segments = parser.parse_verbatim_report(html_content, session_id, session_date)
                
                # Should extract at least something from non-empty content
                if "President" in html_content or "SPEAKER" in html_content:
                    assert len(segments) > 0
                
                # Validation should handle edge cases gracefully
                validation = validator.validate_batch(segments)
                assert validation.segment_count == len(segments)
                
            except Exception as e:
                pytest.fail(f"Parser failed on edge case {i}: {e}")
    
    def test_multilingual_content_handling(self, parsing_components):
        """Test handling of multilingual content (common in EU Parliament)."""
        parser = parsing_components['parser']
        
        multilingual_html = """
        <html>
        <body>
            <div class="contents">
                <p><strong>GARCÍA PÉREZ, María (PPE).</strong> − Señor Presidente, la transición ecológica es fundamental para Europa.</p>
                <p><strong>MÜLLER, Hans (S&D).</strong> − Herr Präsident, ich stimme meiner Kollegin vollständig zu.</p>
                <p><strong>DUBOIS, Pierre (Renew).</strong> − Monsieur le Président, la France soutient cette initiative européenne.</p>
                <p><strong>ROSSI, Marco (ID).</strong> − Signor Presidente, l'Italia deve proteggere i propri interessi nazionali.</p>
                <p><strong>KOWALSKI, Jan (ECR).</strong> − Panie Przewodniczący, Polska potrzebuje sprawiedliwej transformacji.</p>
            </div>
        </body>
        </html>
        """
        
        session_id = "multilingual_test"
        session_date = datetime(2024, 1, 15, 14, 0)
        
        segments = parser.parse_verbatim_report(multilingual_html, session_id, session_date)
        
        # Should extract all speakers regardless of language
        assert len(segments) >= 5
        
        # Check speaker extraction works for different name formats
        speakers = [s.speaker_raw for s in segments]
        expected_speakers = ["GARCÍA PÉREZ, María", "MÜLLER, Hans", "DUBOIS, Pierre", "ROSSI, Marco", "KOWALSKI, Jan"]
        
        for expected in expected_speakers:
            assert any(expected in speaker for speaker in speakers), f"Missing speaker: {expected}"
        
        # Content should be preserved properly
        for segment in segments:
            assert len(segment.speech_text) > 10  # Should have substantial content
            assert not segment.is_procedural  # These are all speeches
    
    @pytest.mark.slow
    def test_large_document_processing(self, parsing_components):
        """Test processing of large documents (marked as slow test)."""
        parser = parsing_components['parser']
        validator = parsing_components['validator']
        
        # Generate large document
        segments_html = []
        for i in range(200):  # Large document with 200 segments
            speaker_type = "procedural" if i % 10 == 0 else "speech"
            
            if speaker_type == "procedural":
                segments_html.append(f'<p><strong>President.</strong> − Procedural announcement number {i}.</p>')
            else:
                segments_html.append(f'''
                <p><strong>SPEAKER_{i:03d}, Name (Group).</strong> − Mr President, this is a comprehensive speech about European policy matters. 
                We must address the various challenges facing our union in the 21st century. This includes economic competitiveness, 
                social cohesion, environmental sustainability, and democratic governance. The European Union has shown remarkable 
                resilience throughout its history, adapting to changing circumstances while maintaining its core values of democracy, 
                human rights, and the rule of law. Today, we face new challenges that require innovative solutions and unprecedented cooperation.</p>
                ''')
        
        large_html = f"""
        <html>
        <body>
            <div class="contents">
                <p>The sitting opened at 09:00</p>
                {''.join(segments_html)}
                <p>The sitting closed at 18:00</p>
            </div>
        </body>
        </html>
        """
        
        session_id = "large_document_test"
        session_date = datetime(2024, 1, 15, 9, 0)
        
        import time
        start_time = time.time()
        
        segments = parser.parse_verbatim_report(large_html, session_id, session_date)
        
        parsing_time = time.time() - start_time
        
        # Performance requirements for large documents
        assert parsing_time < 10.0  # Should complete within 10 seconds
        assert len(segments) >= 180  # Should extract most segments
        
        # Quality should remain reasonable even with large documents
        validation = validator.validate_batch(segments[:50])  # Sample validation
        assert validation.quality_score > 0.6
        
        # Statistical checks
        stats = parser.get_enhanced_parsing_stats()
        assert stats['average_confidence'] > 0.6
        assert stats['validation_failure_rate'] < 20  # Less than 20% failure rate