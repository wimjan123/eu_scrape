"""MEP (Member of European Parliament) database client."""

from typing import Dict, List, Optional, Any
from ..clients.opendata_client import OpenDataClient
from ..core.exceptions import APIError
from ..core.logging import get_logger
from ..models.speaker import MEPData, SpeakerDatabase

logger = get_logger(__name__)


class MEPClient:
    """Client for MEP database operations."""
    
    def __init__(self, opendata_client: OpenDataClient):
        """
        Initialize MEP client using OpenData client.
        
        Args:
            opendata_client: Configured OpenData client
        """
        self.opendata_client = opendata_client
        self.mep_cache: Optional[List[MEPData]] = None
        
        logger.info("MEP client initialized")
    
    def get_current_meps(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get current MEPs from Open Data Portal.
        
        Args:
            force_refresh: Force refresh of cached data
            
        Returns:
            List of MEP data dictionaries
        """
        if self.mep_cache is None or force_refresh:
            logger.info("Fetching current MEPs", force_refresh=force_refresh)
            
            try:
                mep_data = self.opendata_client.get_mep_data(active_only=True)
                logger.info("Retrieved MEP data", count=len(mep_data))
                return mep_data
                
            except APIError as e:
                logger.error("Failed to fetch MEP data", error=str(e))
                raise
        
        return self.mep_cache
    
    def get_historical_meps(self, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """
        Get historical MEP data for date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of historical MEP data
        """
        logger.info("Fetching historical MEPs", start_date=start_date, end_date=end_date)
        
        try:
            # Get all MEPs (not just active)
            mep_data = self.opendata_client.get_mep_data(active_only=False)
            
            # TODO: Implement date filtering when API supports it
            # For now, return all historical data
            
            logger.info("Retrieved historical MEP data", count=len(mep_data))
            return mep_data
            
        except APIError as e:
            logger.error("Failed to fetch historical MEP data", error=str(e))
            raise
    
    def build_mep_database(self) -> SpeakerDatabase:
        """
        Build comprehensive MEP database.
        
        Returns:
            SpeakerDatabase with MEP information
        """
        logger.info("Building MEP database")
        
        try:
            # Get current MEPs
            current_meps_raw = self.get_current_meps()
            
            # Convert to MEPData objects
            meps = []
            for mep_raw in current_meps_raw:
                try:
                    mep = self._convert_raw_mep_data(mep_raw)
                    if mep:
                        meps.append(mep)
                except Exception as e:
                    logger.warning("Failed to convert MEP data", error=str(e), mep_id=mep_raw.get('id'))
                    continue
            
            # Create speaker database
            database = SpeakerDatabase(meps=meps)
            
            logger.info("Built MEP database", mep_count=len(meps))
            return database
            
        except Exception as e:
            logger.error("Failed to build MEP database", error=str(e))
            raise
    
    def _convert_raw_mep_data(self, raw_data: Dict[str, Any]) -> Optional[MEPData]:
        """
        Convert raw API data to MEPData object.
        
        Args:
            raw_data: Raw MEP data from API
            
        Returns:
            MEPData object or None if conversion fails
        """
        try:
            # Extract fields based on typical Open Data Portal structure
            # Note: Actual field names may differ - this is based on common patterns
            
            mep_id = raw_data.get('identifier') or raw_data.get('id', '')
            
            # Handle different name field structures
            if 'label' in raw_data:
                if isinstance(raw_data['label'], dict):
                    full_name = raw_data['label'].get('en', '') or str(raw_data['label'])
                else:
                    full_name = str(raw_data['label'])
            else:
                full_name = raw_data.get('name', '')
            
            # Extract first and family names
            first_name = raw_data.get('hasFirstName', '') or raw_data.get('firstName', '')
            family_name = raw_data.get('hasFamilyName', '') or raw_data.get('familyName', '')
            
            # Extract country
            country = ''
            if 'represents' in raw_data:
                if isinstance(raw_data['represents'], dict):
                    country = raw_data['represents'].get('label', {}).get('en', '') or str(raw_data['represents'])
                else:
                    country = str(raw_data['represents'])
            else:
                country = raw_data.get('country', '')
            
            # Extract political group
            political_group = ''
            if 'memberOf' in raw_data:
                if isinstance(raw_data['memberOf'], dict):
                    political_group = raw_data['memberOf'].get('label', {}).get('en', '') or str(raw_data['memberOf'])
                else:
                    political_group = str(raw_data['memberOf'])
            else:
                political_group = raw_data.get('politicalGroup', '')
            
            # Extract dates
            start_date = raw_data.get('membershipStartDate') or raw_data.get('startDate')
            end_date = raw_data.get('membershipEndDate') or raw_data.get('endDate')
            
            # Validate required fields
            if not mep_id or not full_name:
                logger.warning("MEP missing required fields", mep_id=mep_id, full_name=full_name)
                return None
            
            return MEPData(
                mep_id=mep_id,
                full_name=full_name,
                first_name=first_name,
                family_name=family_name,
                country=country,
                political_group=political_group,
                start_date=start_date,
                end_date=end_date
            )
            
        except Exception as e:
            logger.error("Failed to convert MEP data", error=str(e), raw_data_keys=list(raw_data.keys()))
            return None
    
    def search_mep_by_name(self, name: str) -> List[MEPData]:
        """
        Search for MEPs by name.
        
        Args:
            name: Name to search for
            
        Returns:
            List of matching MEPs
        """
        logger.info("Searching MEPs by name", name=name)
        
        try:
            database = self.build_mep_database()
            
            # Simple name matching - can be enhanced with fuzzy matching later
            matches = []
            name_lower = name.lower()
            
            for mep in database.meps:
                if (name_lower in mep.full_name.lower() or 
                    name_lower in mep.first_name.lower() or 
                    name_lower in mep.family_name.lower()):
                    matches.append(mep)
            
            logger.info("Found MEP matches", name=name, count=len(matches))
            return matches
            
        except Exception as e:
            logger.error("Failed to search MEPs by name", error=str(e), name=name)
            return []
    
    def get_mep_by_id(self, mep_id: str) -> Optional[MEPData]:
        """
        Get specific MEP by ID.
        
        Args:
            mep_id: MEP identifier
            
        Returns:
            MEPData or None if not found
        """
        logger.info("Fetching MEP by ID", mep_id=mep_id)
        
        try:
            database = self.build_mep_database()
            
            for mep in database.meps:
                if mep.mep_id == mep_id:
                    logger.info("Found MEP by ID", mep_id=mep_id, name=mep.full_name)
                    return mep
            
            logger.warning("MEP not found", mep_id=mep_id)
            return None
            
        except Exception as e:
            logger.error("Failed to get MEP by ID", error=str(e), mep_id=mep_id)
            return None