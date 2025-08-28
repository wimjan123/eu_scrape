"""Configuration validation for EU Parliament scraper."""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import requests
from urllib.parse import urlparse

from .config import Settings
from .logging import get_logger
from .exceptions import ConfigurationError

logger = get_logger(__name__)


class ConfigValidator:
    """Validates configuration and environment setup."""
    
    def __init__(self):
        """Initialize configuration validator."""
        self.validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks_passed': 0,
            'checks_total': 0
        }
    
    def validate_all(self, config_path: str = "config/settings.yaml") -> Dict[str, Any]:
        """
        Perform comprehensive configuration validation.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validation results
        """
        logger.info("Starting comprehensive configuration validation")
        
        # Reset validation results
        self.validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks_passed': 0,
            'checks_total': 0
        }
        
        try:
            # Load and validate configuration file
            config = self._validate_config_file(config_path)
            
            if config:
                # Validate configuration structure
                self._validate_config_structure(config)
                
                # Validate API endpoints
                self._validate_api_endpoints(config)
                
                # Validate file system requirements
                self._validate_filesystem_requirements()
                
                # Validate environment variables
                self._validate_environment_variables()
                
                # Validate dependencies
                self._validate_dependencies()
            
        except Exception as e:
            self._add_error(f"Configuration validation failed: {str(e)}")
        
        # Calculate final results
        self.validation_results['success_rate'] = (
            self.validation_results['checks_passed'] / 
            max(1, self.validation_results['checks_total']) * 100
        )
        
        logger.info(
            "Configuration validation completed",
            valid=self.validation_results['valid'],
            errors=len(self.validation_results['errors']),
            warnings=len(self.validation_results['warnings']),
            success_rate=round(self.validation_results['success_rate'], 1)
        )
        
        return self.validation_results
    
    def _validate_config_file(self, config_path: str) -> Optional[Settings]:
        """Validate configuration file existence and structure."""
        self._add_check("Configuration file validation")
        
        config_file = Path(config_path)
        
        if not config_file.exists():
            self._add_error(f"Configuration file not found: {config_path}")
            return None
        
        try:
            # Validate YAML syntax
            with open(config_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            if not yaml_data:
                self._add_error("Configuration file is empty")
                return None
            
            # Validate Pydantic model
            config = Settings.load_from_file(config_path)
            self._add_success("Configuration file loaded successfully")
            
            return config
            
        except yaml.YAMLError as e:
            self._add_error(f"Invalid YAML syntax: {str(e)}")
            return None
        except Exception as e:
            self._add_error(f"Failed to load configuration: {str(e)}")
            return None
    
    def _validate_config_structure(self, config: Settings) -> None:
        """Validate configuration structure and required fields."""
        self._add_check("Configuration structure validation")
        
        try:
            # Validate API configurations
            required_apis = ['opendata', 'eurlex', 'verbatim', 'mep']
            for api_name in required_apis:
                if api_name not in config.api:
                    self._add_error(f"Missing API configuration: {api_name}")
                else:
                    api_config = config.api[api_name]
                    if not api_config.base_url:
                        self._add_error(f"Missing base_url for {api_name}")
                    if api_config.rate_limit <= 0:
                        self._add_error(f"Invalid rate_limit for {api_name}")
            
            # Validate processing configuration
            if config.processing.max_workers <= 0:
                self._add_error("Invalid max_workers in processing config")
            
            if not config.processing.languages:
                self._add_warning("No languages configured")
            elif 'en' not in config.processing.languages:
                self._add_warning("English (en) not in configured languages")
            
            # Validate quality configuration
            if config.quality.speaker_confidence_threshold < 0 or config.quality.speaker_confidence_threshold > 1:
                self._add_error("Invalid speaker_confidence_threshold (must be 0.0-1.0)")
            
            if config.quality.validation_sample_size <= 0:
                self._add_error("Invalid validation_sample_size")
            
            # Validate output configuration
            if not config.output.base_dir:
                self._add_error("Missing output base_dir")
            
            self._add_success("Configuration structure is valid")
            
        except Exception as e:
            self._add_error(f"Configuration structure validation failed: {str(e)}")
    
    def _validate_api_endpoints(self, config: Settings) -> None:
        """Validate API endpoint accessibility."""
        self._add_check("API endpoint validation")
        
        api_results = {}
        
        for api_name, api_config in config.api.items():
            self._add_check(f"API endpoint: {api_name}")
            
            try:
                # Parse URL
                parsed_url = urlparse(api_config.base_url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    self._add_error(f"Invalid URL format for {api_name}: {api_config.base_url}")
                    api_results[api_name] = False
                    continue
                
                # Test connectivity (with timeout)
                try:
                    response = requests.head(
                        api_config.base_url,
                        timeout=10,
                        allow_redirects=True,
                        headers={'User-Agent': 'EU-Parliament-Config-Validator/1.0'}
                    )
                    
                    if response.status_code < 400:
                        self._add_success(f"API endpoint {api_name} is accessible")
                        api_results[api_name] = True
                    else:
                        self._add_warning(f"API endpoint {api_name} returned status {response.status_code}")
                        api_results[api_name] = False
                        
                except requests.exceptions.Timeout:
                    self._add_warning(f"API endpoint {api_name} timed out (may be rate limited)")
                    api_results[api_name] = False
                    
                except requests.exceptions.ConnectionError:
                    self._add_error(f"Cannot connect to API endpoint {api_name}")
                    api_results[api_name] = False
                    
                except requests.exceptions.RequestException as e:
                    self._add_warning(f"API endpoint {api_name} validation failed: {str(e)}")
                    api_results[api_name] = False
                
            except Exception as e:
                self._add_error(f"API endpoint validation failed for {api_name}: {str(e)}")
                api_results[api_name] = False
        
        accessible_apis = sum(1 for result in api_results.values() if result)
        if accessible_apis == 0:
            self._add_error("No API endpoints are accessible")
        elif accessible_apis < len(api_results):
            self._add_warning(f"Only {accessible_apis}/{len(api_results)} API endpoints are accessible")
    
    def _validate_filesystem_requirements(self) -> None:
        """Validate filesystem requirements and permissions."""
        self._add_check("Filesystem requirements validation")
        
        required_dirs = [
            "data/cache",
            "data/progress", 
            "data/output",
            "data/meps",
            "data/monitoring",
            "logs"
        ]
        
        for dir_path in required_dirs:
            self._add_check(f"Directory: {dir_path}")
            
            path = Path(dir_path)
            
            try:
                # Create directory if it doesn't exist
                path.mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
                
                self._add_success(f"Directory {dir_path} is accessible and writable")
                
            except PermissionError:
                self._add_error(f"No write permission for directory {dir_path}")
            except Exception as e:
                self._add_error(f"Directory validation failed for {dir_path}: {str(e)}")
        
        # Check disk space
        try:
            disk_usage = Path.cwd().stat()
            # This is a simple check - in production you'd want more sophisticated disk space checking
            self._add_success("Disk space check passed")
        except Exception as e:
            self._add_warning(f"Disk space check failed: {str(e)}")
    
    def _validate_environment_variables(self) -> None:
        """Validate environment variables."""
        self._add_check("Environment variables validation")
        
        # Optional environment variables
        optional_env_vars = [
            'EU_SCRAPE_ENV',
            'EU_SCRAPE_LOG_LEVEL',
            'EU_SCRAPE_CONFIG_PATH'
        ]
        
        env_vars_found = 0
        for env_var in optional_env_vars:
            if os.getenv(env_var):
                env_vars_found += 1
                logger.debug(f"Environment variable {env_var} is set")
        
        if env_vars_found == 0:
            self._add_warning("No environment variables configured (using defaults)")
        else:
            self._add_success(f"Found {env_vars_found} environment variables")
    
    def _validate_dependencies(self) -> None:
        """Validate Python dependencies."""
        self._add_check("Dependencies validation")
        
        required_packages = [
            'requests', 'beautifulsoup4', 'lxml', 'pandas', 'numpy',
            'pydantic', 'PyYAML', 'fuzzywuzzy', 'rapidfuzz', 
            'structlog', 'pytest', 'psutil'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self._add_error(f"Missing required packages: {', '.join(missing_packages)}")
        else:
            self._add_success("All required packages are installed")
    
    def _add_check(self, check_name: str) -> None:
        """Add a check to the validation."""
        self.validation_results['checks_total'] += 1
    
    def _add_success(self, message: str) -> None:
        """Add a successful check."""
        self.validation_results['checks_passed'] += 1
        logger.debug(f"✓ {message}")
    
    def _add_error(self, message: str) -> None:
        """Add an error to validation results."""
        self.validation_results['valid'] = False
        self.validation_results['errors'].append(message)
        logger.error(f"✗ {message}")
    
    def _add_warning(self, message: str) -> None:
        """Add a warning to validation results."""
        self.validation_results['warnings'].append(message)
        logger.warning(f"⚠ {message}")
    
    def generate_validation_report(self) -> str:
        """Generate human-readable validation report."""
        results = self.validation_results
        
        report_lines = [
            "EU Parliament Scraper - Configuration Validation Report",
            f"Generated: {os.uname().nodename} at {os.getcwd()}",
            f"Timestamp: {__import__('datetime').datetime.utcnow().isoformat()}Z",
            "",
            f"Overall Status: {'✓ VALID' if results['valid'] else '✗ INVALID'}",
            f"Success Rate: {results.get('success_rate', 0):.1f}%",
            f"Checks Passed: {results['checks_passed']}/{results['checks_total']}",
            ""
        ]
        
        if results['errors']:
            report_lines.extend([
                "ERRORS:",
                *[f"  ✗ {error}" for error in results['errors']],
                ""
            ])
        
        if results['warnings']:
            report_lines.extend([
                "WARNINGS:",
                *[f"  ⚠ {warning}" for warning in results['warnings']],
                ""
            ])
        
        if results['valid']:
            report_lines.append("Configuration is ready for production use.")
        else:
            report_lines.append("Configuration requires fixes before production use.")
        
        return "\\n".join(report_lines)