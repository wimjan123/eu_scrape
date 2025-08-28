#!/usr/bin/env python3
"""
Simple API Connectivity Test for EU Parliament Scraper

Tests basic connectivity to all required data sources.
"""

import requests
import time
from typing import Tuple


def test_connectivity() -> bool:
    """Test connectivity to all required APIs."""
    
    print("🔍 EU Parliament Scraper - Simple Connectivity Test")
    print("=" * 60)
    
    tests = [
        ("European Parliament Website", "https://www.europarl.europa.eu", {}),
        ("Open Data Portal", "https://data.europarl.europa.eu", {}),
        ("EUR-Lex Publications", "http://publications.europa.eu", {}),
        ("Verbatim Reports Base", "https://www.europarl.europa.eu/doceo", {}),
    ]
    
    all_passed = True
    
    for name, url, headers in tests:
        try:
            print(f"\\nTesting {name}...")
            print(f"URL: {url}")
            
            default_headers = {
                'User-Agent': 'EU-Parliament-Research-Tool/1.0',
                'Accept': 'text/html,application/json'
            }
            default_headers.update(headers)
            
            response = requests.get(
                url,
                headers=default_headers,
                timeout=30,
                allow_redirects=True
            )
            
            if 200 <= response.status_code < 400:
                print(f"✅ SUCCESS: {response.status_code} - {response.reason}")
                print(f"   Response time: {response.elapsed.total_seconds():.2f}s")
            else:
                print(f"⚠️  WARNING: {response.status_code} - {response.reason}")
                if response.status_code < 500:  # Client error but server responding
                    print("   Server responding (may need specific endpoints)")
                else:
                    all_passed = False
            
            # Be respectful with rate limiting
            time.sleep(1)
            
        except requests.exceptions.Timeout:
            print(f"❌ TIMEOUT: Server did not respond within 30 seconds")
            all_passed = False
            
        except requests.exceptions.ConnectionError as e:
            print(f"❌ CONNECTION ERROR: {e}")
            all_passed = False
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            all_passed = False
    
    print("\\n" + "=" * 60)
    
    if all_passed:
        print("🎉 OVERALL RESULT: All servers are reachable!")
        print("✅ Network connectivity confirmed for EU Parliament APIs.")
        return True
    else:
        print("⚠️  OVERALL RESULT: Some connectivity issues detected.")
        print("❌ Check network connection and try again.")
        return False


if __name__ == "__main__":
    success = test_connectivity()
    exit(0 if success else 1)