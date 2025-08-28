"""EUR-Lex SPARQL endpoint client for EU Parliament scraper."""

import requests
from typing import Dict, List, Optional, Any
import time
from urllib.parse import urlencode

from ..core.config import APIConfig
from ..core.rate_limiter import RateLimiter, ExponentialBackoff
from ..core.exceptions import APIError
from ..core.logging import get_logger, log_api_request

logger = get_logger(__name__)


class EURLexClient:
    """Client for EUR-Lex SPARQL endpoint."""
    
    def __init__(self, config: APIConfig):
        """
        Initialize EUR-Lex SPARQL client.
        
        Args:
            config: API configuration
        """
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.session = requests.Session()
        
        self.session.headers.update({
            'User-Agent': 'EU-Parliament-Research-Tool/1.0',
            'Accept': 'application/sparql-results+json,application/json'
        })
        
        logger.info(
            "EUR-Lex client initialized",
            sparql_endpoint=config.base_url,
            rate_limit=config.rate_limit
        )
    
    def _execute_sparql_query(self, query: str, format_type: str = "json") -> Dict[str, Any]:
        """
        Execute SPARQL query with error handling.
        
        Args:
            query: SPARQL query string
            format_type: Response format (json, xml, turtle)
            
        Returns:
            Query results
            
        Raises:
            APIError: On query execution errors
        """
        self.rate_limiter.wait_if_needed()
        
        params = {
            'query': query,
            'format': format_type
        }
        
        backoff = ExponentialBackoff()
        while backoff.wait():
            start_time = time.time()
            
            try:
                logger.debug("Executing SPARQL query", query_length=len(query))
                
                response = self.session.get(
                    self.config.base_url,
                    params=params,
                    timeout=self.config.timeout
                )
                
                response_time = time.time() - start_time
                
                logger.info(
                    "SPARQL query completed",
                    **log_api_request(self.config.base_url, "GET", response_time, response.status_code)
                )
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning("SPARQL rate limit exceeded", retry_after=retry_after)
                    time.sleep(retry_after)
                    continue
                
                if response.status_code >= 400:
                    error_msg = f"SPARQL query failed: {response.status_code}"
                    logger.error(
                        error_msg,
                        status_code=response.status_code,
                        response_text=response.text[:500]
                    )
                    
                    if response.status_code >= 500:
                        continue  # Retry server errors
                    else:
                        raise APIError(error_msg, response.status_code, response.text)
                
                # Parse JSON response
                try:
                    return response.json()
                except ValueError as e:
                    logger.error("Failed to parse SPARQL JSON response", error=str(e))
                    raise APIError(f"Invalid SPARQL response: {e}")
                
            except requests.exceptions.Timeout:
                logger.warning("SPARQL query timeout", attempt=backoff.attempt)
                continue
                
            except requests.exceptions.ConnectionError as e:
                logger.warning("SPARQL connection error", error=str(e), attempt=backoff.attempt)
                continue
                
            except requests.exceptions.RequestException as e:
                logger.error("SPARQL request exception", error=str(e))
                raise APIError(f"SPARQL request failed: {e}")
        
        raise APIError("SPARQL maximum retries exceeded")
    
    def get_plenary_documents(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Query plenary session documents using SPARQL.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of document metadata
        """
        logger.info("Querying plenary documents", start_date=start_date, end_date=end_date)
        
        # SPARQL query for plenary documents
        query = f"""
        PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT DISTINCT ?document ?title ?date ?language ?identifier WHERE {{
            ?document a cdm:expression ;
                      cdm:expression_title ?title ;
                      cdm:expression_date ?date ;
                      cdm:expression_language ?language ;
                      cdm:expression_identifier ?identifier ;
                      cdm:expression_belongs_to_work ?work .
            
            ?work cdm:work_part_of_collection ?collection .
            ?collection skos:prefLabel ?collectionLabel .
            
            FILTER(CONTAINS(LCASE(str(?collectionLabel)), "plenary"))
            FILTER(?date >= "{start_date}"^^xsd:date)
            FILTER(?date <= "{end_date}"^^xsd:date)
            FILTER(?language = "ENG")
        }}
        ORDER BY ?date
        LIMIT 1000
        """
        
        try:
            results = self._execute_sparql_query(query)
            
            # Extract bindings from SPARQL results
            documents = []
            if 'results' in results and 'bindings' in results['results']:
                for binding in results['results']['bindings']:
                    doc = {
                        'document_uri': binding.get('document', {}).get('value', ''),
                        'title': binding.get('title', {}).get('value', ''),
                        'date': binding.get('date', {}).get('value', ''),
                        'language': binding.get('language', {}).get('value', ''),
                        'identifier': binding.get('identifier', {}).get('value', '')
                    }
                    documents.append(doc)
            
            logger.info("Retrieved plenary documents", count=len(documents))
            return documents
            
        except APIError as e:
            logger.error("Failed to query plenary documents", error=str(e))
            raise
    
    def get_document_metadata(self, document_uri: str) -> Dict[str, Any]:
        """
        Get detailed metadata for a specific document.
        
        Args:
            document_uri: URI of the document
            
        Returns:
            Document metadata
        """
        logger.info("Fetching document metadata", document_uri=document_uri)
        
        query = f"""
        PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        
        SELECT ?property ?value WHERE {{
            <{document_uri}> ?property ?value .
        }}
        LIMIT 100
        """
        
        try:
            results = self._execute_sparql_query(query)
            
            metadata = {}
            if 'results' in results and 'bindings' in results['results']:
                for binding in results['results']['bindings']:
                    prop = binding.get('property', {}).get('value', '')
                    value = binding.get('value', {}).get('value', '')
                    
                    # Extract property name from URI
                    prop_name = prop.split('#')[-1] if '#' in prop else prop.split('/')[-1]
                    metadata[prop_name] = value
            
            logger.info("Retrieved document metadata", document_uri=document_uri)
            return metadata
            
        except APIError as e:
            logger.error(
                "Failed to fetch document metadata", 
                error=str(e),
                document_uri=document_uri
            )
            raise
    
    def search_mep_documents(self, mep_name: str, start_date: str = None, 
                           end_date: str = None) -> List[Dict[str, Any]]:
        """
        Search for documents by MEP name.
        
        Args:
            mep_name: Name of MEP to search for
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of documents by the MEP
        """
        logger.info("Searching MEP documents", mep_name=mep_name)
        
        # Build date filters
        date_filter = ""
        if start_date and end_date:
            date_filter = f"""
            FILTER(?date >= "{start_date}"^^xsd:date)
            FILTER(?date <= "{end_date}"^^xsd:date)
            """
        
        query = f"""
        PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT DISTINCT ?document ?title ?date ?author WHERE {{
            ?document a cdm:expression ;
                      cdm:expression_title ?title ;
                      cdm:expression_date ?date ;
                      cdm:expression_author ?authorEntity .
            
            ?authorEntity foaf:name ?author .
            
            FILTER(CONTAINS(LCASE(?author), "{mep_name.lower()}"))
            {date_filter}
        }}
        ORDER BY ?date
        LIMIT 500
        """
        
        try:
            results = self._execute_sparql_query(query)
            
            documents = []
            if 'results' in results and 'bindings' in results['results']:
                for binding in results['results']['bindings']:
                    doc = {
                        'document_uri': binding.get('document', {}).get('value', ''),
                        'title': binding.get('title', {}).get('value', ''),
                        'date': binding.get('date', {}).get('value', ''),
                        'author': binding.get('author', {}).get('value', '')
                    }
                    documents.append(doc)
            
            logger.info("Retrieved MEP documents", mep_name=mep_name, count=len(documents))
            return documents
            
        except APIError as e:
            logger.error("Failed to search MEP documents", error=str(e), mep_name=mep_name)
            raise
    
    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()
        logger.info("EUR-Lex client closed")