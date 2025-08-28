"""Custom exceptions for EU Parliament scraper."""


class EUParliamentScraperError(Exception):
    """Base exception for EU Parliament scraper."""
    pass


class ConfigurationError(EUParliamentScraperError):
    """Configuration-related errors."""
    pass


class APIError(EUParliamentScraperError):
    """API-related errors."""
    
    def __init__(self, message: str, status_code: int = None, response_data: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class RateLimitExceededError(APIError):
    """Rate limit exceeded error."""
    pass


class ParsingError(EUParliamentScraperError):
    """Document parsing errors."""
    
    def __init__(self, message: str, document_url: str = None, document_type: str = None):
        super().__init__(message)
        self.document_url = document_url
        self.document_type = document_type


class SpeakerResolutionError(EUParliamentScraperError):
    """Speaker resolution errors."""
    
    def __init__(self, message: str, speaker_name: str = None, confidence: float = None):
        super().__init__(message)
        self.speaker_name = speaker_name
        self.confidence = confidence


class DataValidationError(EUParliamentScraperError):
    """Data validation errors."""
    
    def __init__(self, message: str, field_name: str = None, field_value = None):
        super().__init__(message)
        self.field_name = field_name
        self.field_value = field_value


class CheckpointError(EUParliamentScraperError):
    """Checkpoint save/load errors."""
    pass


class ProcessingError(EUParliamentScraperError):
    """General processing errors."""
    
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.context = context or {}