#!/usr/bin/env python3
"""
API Connectivity Test Script for EU Parliament Scraper

This script tests connectivity to all required data sources as specified 
in the mission requirements:
- European Parliament Open Data Portal: https://data.europarl.europa.eu/
- EUR-Lex SPARQL Endpoint: http://publications.europa.eu/webapi/rdf/sparql
- Verbatim Reports: https://www.europarl.europa.eu/doceo/document/
- MEP Database: Accessible via Open Data Portal

CRITICAL: Test connectivity before proceeding with implementation.
"""

import sys
import requests
import time
from typing import Dict, List, Tuple
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.logging import setup_logging, get_logger
from core.rate_limiter import RateLimiter


def test_opendata_portal() -> Tuple[bool, str, Dict]:
    """Test European Parliament Open Data Portal connectivity."""
    logger = get_logger(__name__)
    
    base_url = "https://data.europarl.europa.eu"
    test_endpoints = [
        "/api",
        "/data", 
        "/en"  # Homepage endpoint
    ]
    
    results = {}
    overall_success = False
    
    rate_limiter = RateLimiter(0.5)  # Conservative 0.5 req/sec
    
    for endpoint in test_endpoints:
        rate_limiter.wait_if_needed()
        
        try:
            url = base_url + endpoint
            logger.info(f"Testing Open Data Portal: {url}")
            
            response = requests.get(
                url,
                headers={
                    'User-Agent': 'EU-Parliament-Research-Tool/1.0',
                    'Accept': 'application/json,text/html'
                },
                timeout=30
            )
            
            success = 200 <= response.status_code < 400
            results[endpoint] = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'success': success,
                'content_type': response.headers.get('content-type', '')
            }
            
            if success:
                overall_success = True
                logger.info(f"✅ Open Data Portal {endpoint}: {response.status_code}")
            else:
                logger.warning(f"⚠️ Open Data Portal {endpoint}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Open Data Portal {endpoint}: {str(e)}")
            results[endpoint] = {'error': str(e), 'success': False}
    
    summary = "✅ ACCESSIBLE" if overall_success else "❌ NOT ACCESSIBLE"
    return overall_success, summary, results


def test_eurlex_sparql() -> Tuple[bool, str, Dict]:
    """Test EUR-Lex SPARQL Endpoint connectivity."""
    logger = get_logger(__name__)
    
    sparql_endpoint = "http://publications.europa.eu/webapi/rdf/sparql"
    
    # Simple SPARQL query to test connectivity
    test_query = """
    SELECT ?s ?p ?o WHERE { 
        ?s ?p ?o 
    } LIMIT 1
    """
    
    rate_limiter = RateLimiter(0.5)  # Conservative 0.5 req/sec
    rate_limiter.wait_if_needed()
    
    try:
        logger.info(f"Testing EUR-Lex SPARQL: {sparql_endpoint}")
        
        response = requests.get(
            sparql_endpoint,
            params={
                'query': test_query,
                'format': 'json'
            },
            headers={
                'User-Agent': 'EU-Parliament-Research-Tool/1.0',
                'Accept': 'application/sparql-results+json,application/json'
            },
            timeout=60
        )
        
        success = 200 <= response.status_code < 400
        results = {
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'success': success,
            'content_type': response.headers.get('content-type', '')
        }
        
        if success:
            logger.info(f"✅ EUR-Lex SPARQL: {response.status_code}")
            summary = "✅ ACCESSIBLE"
        else:
            logger.warning(f"⚠️ EUR-Lex SPARQL: {response.status_code}")
            summary = "❌ NOT ACCESSIBLE"
            
    except Exception as e:
        logger.error(f"❌ EUR-Lex SPARQL: {str(e)}")
        results = {'error': str(e), 'success': False}
        success = False
        summary = "❌ NOT ACCESSIBLE"
    
    return success, summary, results


def test_verbatim_reports() -> Tuple[bool, str, Dict]:
    """Test Verbatim Reports accessibility."""
    logger = get_logger(__name__)
    
    base_url = "https://www.europarl.europa.eu/doceo/document"
    
    # Test general document access (we'll use a pattern that should exist)
    test_endpoints = [
        "/",  # Document base
        "/CRE-9-2024-01-15_EN.html",  # Example verbatim report format
        "/help"  # Help/info page
    ]
    
    results = {}
    overall_success = False
    
    rate_limiter = RateLimiter(0.33)  # Conservative 0.33 req/sec
    
    for endpoint in test_endpoints:
        rate_limiter.wait_if_needed()
        
        try:
            url = base_url + endpoint
            logger.info(f"Testing Verbatim Reports: {url}")
            
            response = requests.get(
                url,
                headers={
                    'User-Agent': 'EU-Parliament-Research-Tool/1.0',
                    'Accept': 'text/html,application/xhtml+xml'
                },
                timeout=45
            )
            
            # For verbatim reports, even 404s on specific documents are OK
            # as long as the server responds
            success = 200 <= response.status_code < 500
            results[endpoint] = {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'success': success,
                'content_type': response.headers.get('content-type', '')
            }
            
            if success:
                overall_success = True
                logger.info(f"✅ Verbatim Reports {endpoint}: {response.status_code}")
            else:
                logger.warning(f"⚠️ Verbatim Reports {endpoint}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Verbatim Reports {endpoint}: {str(e)}")
            results[endpoint] = {'error': str(e), 'success': False}
    
    summary = "✅ ACCESSIBLE" if overall_success else "❌ NOT ACCESSIBLE"
    return overall_success, summary, results


def main():
    """Run comprehensive API connectivity tests."""
    print("🔍 EU Parliament Scraper - API Connectivity Test")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("Starting API connectivity tests")
    
    # Test each data source
    test_results = {}
    
    print("\\n📡 Testing European Parliament Open Data Portal...")
    success, summary, details = test_opendata_portal()
    test_results['opendata_portal'] = {'success': success, 'summary': summary, 'details': details}
    print(f"   Result: {summary}")
    
    print("\\n🔍 Testing EUR-Lex SPARQL Endpoint...")
    success, summary, details = test_eurlex_sparql()
    test_results['eurlex_sparql'] = {'success': success, 'summary': summary, 'details': details}
    print(f"   Result: {summary}")
    
    print("\\n📄 Testing Verbatim Reports Access...")
    success, summary, details = test_verbatim_reports()
    test_results['verbatim_reports'] = {'success': success, 'summary': summary, 'details': details}
    print(f"   Result: {summary}")
    
    # Overall assessment
    print("\\n" + "=" * 60)
    print("📊 CONNECTIVITY TEST RESULTS")
    print("=" * 60)
    
    all_accessible = all(result['success'] for result in test_results.values())
    
    for source, result in test_results.items():
        status_icon = "✅" if result['success'] else "❌"
        print(f"{status_icon} {source.replace('_', ' ').title()}: {result['summary']}")
    
    print("\\n" + "=" * 60)
    
    if all_accessible:
        print("🎉 SUCCESS: All data sources are accessible!")
        print("✅ Ready to proceed with Phase 1 implementation.")
        logger.info("All API connectivity tests passed")
        return 0
    else:
        print("⚠️  WARNING: Some data sources are not accessible!")
        print("❌ Review connection issues before proceeding.")
        
        # Log detailed results for debugging
        for source, result in test_results.items():
            if not result['success']:
                logger.error(f"Failed connectivity test: {source}", details=result['details'])
        
        return 1


if __name__ == "__main__":
    sys.exit(main())