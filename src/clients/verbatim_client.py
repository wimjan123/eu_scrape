"""Verbatim Reports client for EU Parliament scraper."""

import requests
from typing import Dict, List, Optional, Any
import time
from urllib.parse import urljoin, urlparse
import re
from bs4 import BeautifulSoup

from ..core.config import APIConfig
from ..core.rate_limiter import RateLimiter, ExponentialBackoff
from ..core.exceptions import APIError, ParsingError
from ..core.logging import get_logger, log_api_request

logger = get_logger(__name__)


class VerbatimClient:
    """Client for downloading verbatim reports from EU Parliament."""
    
    def __init__(self, config: APIConfig):
        """
        Initialize Verbatim Reports client.
        
        Args:
            config: API configuration
        """
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.session = requests.Session()
        
        self.session.headers.update({
            'User-Agent': 'EU-Parliament-Research-Tool/1.0',
            'Accept': 'text/html,application/xhtml+xml,text/xml'
        })
        
        # Common verbatim report URL patterns
        self.verbatim_patterns = [
            r'CRE-\d+-\d{4}-\d{2}-\d{2}',  # CRE-9-2024-01-15
            r'CRE-\d+-\d{4}-\d{2}-\d{2}_[A-Z]{2}',  # CRE-9-2024-01-15_EN
            r'PV-\d+-\d{4}-\d{2}-\d{2}',  # PV format
        ]
        
        logger.info(
            "Verbatim client initialized",
            base_url=config.base_url,
            rate_limit=config.rate_limit
        )
    
    def _make_request(self, url: str) -> str:
        """
        Make rate-limited request for verbatim document.
        
        Args:
            url: Document URL
            
        Returns:
            Document HTML content
            
        Raises:
            APIError: On request errors
        """
        self.rate_limiter.wait_if_needed()
        
        backoff = ExponentialBackoff()
        while backoff.wait():
            start_time = time.time()
            
            try:
                logger.debug("Fetching verbatim document", url=url)
                
                response = self.session.get(url, timeout=self.config.timeout)
                
                response_time = time.time() - start_time
                
                logger.info(
                    "Verbatim document request completed",
                    **log_api_request(url, "GET", response_time, response.status_code)
                )
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning("Verbatim rate limit exceeded", retry_after=retry_after)
                    time.sleep(retry_after)
                    continue
                
                if response.status_code == 404:
                    raise APIError(f"Verbatim document not found: {url}", response.status_code)
                
                if response.status_code >= 400:
                    error_msg = f"Verbatim request failed: {response.status_code}"
                    logger.error(
                        error_msg,
                        url=url,
                        status_code=response.status_code
                    )
                    
                    if response.status_code >= 500:
                        continue  # Retry server errors
                    else:
                        raise APIError(error_msg, response.status_code)
                
                # Return HTML content
                return response.text
                
            except requests.exceptions.Timeout:
                logger.warning("Verbatim request timeout", url=url, attempt=backoff.attempt)
                continue
                
            except requests.exceptions.ConnectionError as e:
                logger.warning("Verbatim connection error", error=str(e), attempt=backoff.attempt)
                continue
                
            except requests.exceptions.RequestException as e:
                logger.error("Verbatim request exception", error=str(e), url=url)
                raise APIError(f"Verbatim request failed: {e}")
        
        raise APIError(f"Maximum retries exceeded for: {url}")
    
    def download_verbatim_report(self, session_id: str, language: str = "EN") -> str:
        """
        Download verbatim report by session ID.
        
        Args:
            session_id: Session identifier (e.g., "CRE-9-2024-01-15")
            language: Language code (EN, FR, DE, etc.)
            
        Returns:
            HTML content of verbatim report
            
        Raises:
            APIError: On download errors
        """
        logger.info("Downloading verbatim report", session_id=session_id, language=language)
        
        # Construct URL based on session ID format
        if language and not session_id.endswith(f"_{language}"):
            document_id = f"{session_id}_{language}"
        else:
            document_id = session_id
        
        # Try common URL formats
        url_patterns = [
            f"{self.config.base_url}/{document_id}.html",
            f"{self.config.base_url}/{document_id}.xml",
            f"{self.config.base_url}/CRE/{document_id}.html",
            f"{self.config.base_url}/document/{document_id}"
        ]
        
        last_error = None
        
        for url in url_patterns:
            try:
                content = self._make_request(url)
                logger.info("Successfully downloaded verbatim report", 
                          session_id=session_id, url=url)
                return content
                
            except APIError as e:
                last_error = e
                if e.status_code == 404:
                    logger.debug("Verbatim URL not found, trying next pattern", url=url)
                    continue
                else:
                    # Non-404 error, don't continue
                    raise
        
        # If all patterns failed
        error_msg = f"Verbatim report not found for session: {session_id}"
        logger.error(error_msg, session_id=session_id, tried_urls=url_patterns)
        raise APIError(error_msg, getattr(last_error, 'status_code', 404))
    
    def get_verbatim_url(self, session_date: str, language: str = "EN") -> Optional[str]:
        """
        Construct verbatim report URL from session date.
        
        Args:
            session_date: Session date (YYYY-MM-DD format)
            language: Language code
            
        Returns:
            Constructed URL or None if format unknown
        """
        try:
            # Parse date to construct session ID
            date_parts = session_date.split('-')
            if len(date_parts) != 3:
                return None
            
            year, month, day = date_parts
            
            # Common formats for verbatim report IDs
            session_formats = [
                f"CRE-9-{year}-{month}-{day}_{language}",
                f"PV-9-{year}-{month}-{day}_{language}",
                f"CRE-{year}-{month}-{day}_{language}"
            ]
            
            # Return first format as default
            session_id = session_formats[0]
            url = f"{self.config.base_url}/{session_id}.html"
            
            logger.debug("Constructed verbatim URL", 
                        session_date=session_date, url=url)
            
            return url
            
        except Exception as e:
            logger.error("Failed to construct verbatim URL", 
                        error=str(e), session_date=session_date)
            return None
    
    def validate_verbatim_content(self, html_content: str, session_id: str = None) -> bool:
        """
        Validate that content appears to be a valid verbatim report.
        
        Args:
            html_content: HTML content to validate
            session_id: Optional session ID for context
            
        Returns:
            True if content appears valid
        """
        try:
            if not html_content or len(html_content) < 100:
                return False
            
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Check for common verbatim report indicators
            indicators = [
                'plenary',
                'sitting',
                'president',
                'debate',
                'parliament',
                'agenda'
            ]
            
            text_content = soup.get_text().lower()
            
            # Must contain at least 2 indicators
            found_indicators = sum(1 for indicator in indicators if indicator in text_content)
            
            if found_indicators >= 2:
                logger.debug("Verbatim content validation passed", 
                           session_id=session_id, indicators_found=found_indicators)
                return True
            else:
                logger.warning("Verbatim content validation failed", 
                             session_id=session_id, indicators_found=found_indicators)
                return False
                
        except Exception as e:
            logger.error("Error validating verbatim content", 
                        error=str(e), session_id=session_id)
            return False
    
    def extract_basic_metadata(self, html_content: str) -> Dict[str, Any]:
        """
        Extract basic metadata from verbatim report.
        
        Args:
            html_content: HTML content
            
        Returns:
            Basic metadata dictionary
        """
        metadata = {
            'session_date': None,
            'session_type': 'plenary',
            'language': None,
            'title': None,
            'total_length': len(html_content)
        }
        
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract title
            title_elem = soup.find('title')
            if title_elem:
                metadata['title'] = title_elem.get_text().strip()
            
            # Look for date patterns in content
            text_content = soup.get_text()
            date_patterns = [
                r'(\d{1,2}\s+\w+\s+\d{4})',  # 15 January 2024
                r'(\d{4}-\d{2}-\d{2})',      # 2024-01-15
                r'(\d{2}/\d{2}/\d{4})'       # 15/01/2024
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, text_content)
                if match:
                    metadata['session_date'] = match.group(1)
                    break
            
            # Detect language
            if 'english' in text_content.lower() or 'en.html' in html_content:
                metadata['language'] = 'EN'
            elif 'français' in text_content.lower() or 'fr.html' in html_content:
                metadata['language'] = 'FR'
            elif 'deutsch' in text_content.lower() or 'de.html' in html_content:
                metadata['language'] = 'DE'
            
            logger.debug("Extracted verbatim metadata", metadata=metadata)
            
        except Exception as e:
            logger.error("Failed to extract verbatim metadata", error=str(e))
        
        return metadata
    
    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()
        logger.info("Verbatim client closed")