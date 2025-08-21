import os
import yaml
from typing import Any, Dict
from configs.logger import setup_logger

logger = setup_logger("Config")

class Config:
    """General configuration loader that reads from YAML files."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

        self.config_path = config_path
        self._config = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        logger.info(f"Loading configuration from {self.config_path}")
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        logger.debug(f"Configuration loaded: {cfg}")
        return cfg

    def get(self, key: str, default: Any = None) -> Any:
        """Access config using dot notation, e.g., config.get('dataset.name')"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                logger.warning(f"Config key '{key}' not found. Returning default: {default}")
                return default
        return value

    def as_dict(self) -> Dict[str, Any]:
        """Return the entire configuration as a dictionary."""
        return self._config
