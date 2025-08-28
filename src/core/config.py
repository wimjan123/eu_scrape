"""Configuration management for EU Parliament scraper."""

import yaml
from typing import Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator


class APIConfig(BaseModel):
    """API configuration settings."""
    base_url: str
    timeout: int = 30
    rate_limit: float = 0.5  # requests per second
    
    @validator('rate_limit')
    def validate_rate_limit(cls, v):
        if v <= 0:
            raise ValueError('rate_limit must be positive')
        return v


class ProcessingConfig(BaseModel):
    """Processing configuration settings."""
    date_range: Dict[str, str]
    languages: List[str] = ["en"]
    max_workers: int = 4
    chunk_size: int = 100
    
    @validator('max_workers')
    def validate_max_workers(cls, v):
        if v < 1 or v > 16:
            raise ValueError('max_workers must be between 1 and 16')
        return v


class QualityConfig(BaseModel):
    """Quality assurance configuration."""
    min_speech_length: int = 10
    max_speech_length: int = 10000
    speaker_confidence_threshold: float = 0.8
    validation_sample_size: int = 25
    
    @validator('speaker_confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('speaker_confidence_threshold must be between 0.0 and 1.0')
        return v


class OutputConfig(BaseModel):
    """Output configuration settings."""
    formats: List[str] = ["csv", "jsonl"]
    include_metadata: bool = True
    timestamp_format: str = "iso8601_utc"


class Settings(BaseModel):
    """Main application settings."""
    api: Dict[str, APIConfig]
    processing: ProcessingConfig
    quality: QualityConfig
    output: OutputConfig
    
    @classmethod
    def load_from_file(cls, config_path: str = "config/settings.yaml") -> "Settings":
        """Load settings from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Convert nested API configs
        api_configs = {}
        for api_name, api_data in config_data.get('api', {}).items():
            api_configs[api_name] = APIConfig(**api_data)
        
        return cls(
            api=api_configs,
            processing=ProcessingConfig(**config_data['processing']),
            quality=QualityConfig(**config_data['quality']),
            output=OutputConfig(**config_data['output'])
        )
    
    def get_api_config(self, api_name: str) -> APIConfig:
        """Get API configuration by name."""
        if api_name not in self.api:
            raise KeyError(f"API configuration '{api_name}' not found")
        return self.api[api_name]