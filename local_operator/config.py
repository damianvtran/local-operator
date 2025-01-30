"""Configuration management for Local Operator.

This module handles reading and writing configuration settings from a YAML file.
It provides default configurations and methods to update them.
"""

import argparse
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration settings for Local Operator.

    Attributes:
        version (str): Configuration schema version for compatibility
        metadata (Dict): Metadata about the configuration
        values (Dict): Configuration settings
            conversation_length (int): Number of conversation messages to retain
            detail_length (int): Maximum length of detailed conversation history
            hosting (str): AI model hosting provider
            model_name (str): Name of the AI model to use
    """

    version: str
    metadata: Dict[str, Any]
    values: Dict[str, Any]

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize the config with default or existing settings.

        Creates a new Config instance that manages configuration settings.
        If a config file exists at the specified path, loads settings from it.
        """
        # Set version and metadata first
        self.version = config_dict.get("version", version("local-operator"))
        self.metadata = config_dict.get(
            "metadata",
            {
                "created_at": "",
                "last_modified": "",
                "description": "Local Operator configuration file",
            },
        )

        # Set metadata values with defaults if not provided
        if not self.metadata["created_at"]:
            self.metadata["created_at"] = datetime.now().isoformat()
        if not self.metadata["last_modified"]:
            self.metadata["last_modified"] = datetime.now().isoformat()

        # Set config values
        self.values = {}
        for key, value in config_dict.get("values", {}).items():
            self.values[key] = value

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration value.

        Args:
            key (str): The configuration key to retrieve

        Returns:
            Any: The configuration value for the key, or default if not found
        """
        return self.values.get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        """Set a specific configuration value.

        Args:
            key (str): The configuration key to set
            value (Any): The value to set for the key
        """
        self.values[key] = value


# Default configuration settings for Local Operator
DEFAULT_CONFIG = Config(
    {
        "version": version("local-operator"),
        "metadata": {
            "created_at": "",
            "last_modified": "",
            "description": "Local Operator configuration file",
        },
        "values": {
            "conversation_length": 100,
            "detail_length": 10,
            "hosting": "",
            "model_name": "",
        },
    }
)

# Name of the YAML configuration file
CONFIG_FILE_NAME = "config.yml"


class ConfigManager:
    """Manages configuration settings for Local Operator.

    Handles reading and writing configuration settings to a YAML file,
    with fallback to default values if no config exists.

    Attributes:
        config_dir (Path): Directory where config file is stored
        config_file (Path): Path to the config.yml file
        config (dict): Current configuration settings
    """

    config_dir: Path
    config_file: Path
    config: Config

    def __init__(self, config_dir: Path):
        """Initialize the config manager with default or existing settings.

        Creates a new ConfigManager instance that manages configuration settings.
        If a config file exists at the specified path, loads settings from it.
        Otherwise creates a new config file with default settings.

        Args:
            config_dir (Path): Directory path where the config file should be stored

        The config file will be named according to CONFIG_FILE_NAME and stored
        in the specified directory. Configuration is loaded immediately upon
        initialization.
        """
        self.config_dir = config_dir
        self.config_file = self.config_dir / CONFIG_FILE_NAME
        self.config = self._load_config()

    def _load_config(self) -> Config:
        """Load configuration from file or create with defaults if none exists.

        Returns:
            Config: The configuration object
        """
        if not self.config_file.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            return DEFAULT_CONFIG

        with open(self.config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or vars(DEFAULT_CONFIG)

            # Check if config version is older than current version
            config_version = config_dict.get("version", "0.0.0")
            current_version = version("local-operator")
            if config_version < current_version:
                print(
                    f"\n\033[1;33mWarning: Your config file version ({config_version}) "
                    f"is older than the current version ({current_version}). "
                    "Please upgrade to ensure compatibility.\033[0m"
                )

            # Fill in any missing values with defaults
            if "values" not in config_dict:
                config_dict["values"] = vars(DEFAULT_CONFIG)["values"]
            else:
                default_values = vars(DEFAULT_CONFIG)["values"]
                for key, value in default_values.items():
                    if key not in config_dict["values"]:
                        config_dict["values"][key] = value

            return Config(config_dict)

    def _write_config(self, config: Dict[str, Any]) -> None:
        """Write configuration to YAML file.

        Creates the config file first if it doesn't exist.

        Args:
            config (Dict[str, Any]): Configuration dictionary to write
        """
        if not self.config_file.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.config_file.touch()

        # Ensure version and metadata are included
        if "version" not in config:
            config["version"] = DEFAULT_CONFIG.version
        if "metadata" not in config:
            config["metadata"] = DEFAULT_CONFIG.metadata

        # Ensure created_at and last_modified are included
        if "created_at" not in config["metadata"]:
            config["metadata"]["created_at"] = datetime.now().isoformat()

        config["metadata"]["last_modified"] = datetime.now().isoformat()

        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

    def get_config(self) -> Config:
        """Get the current configuration settings.

        Returns:
            Config: Current configuration settings
        """
        return self.config

    def update_config(self, updates: Dict[str, Any], write: bool = True) -> None:
        """Update configuration with new values.

        Args:
            updates (Dict[str, Any]): Dictionary of configuration updates
        """
        # Update each field individually to work with Config class
        for key, value in updates.items():
            self.config.set_value(key, value)

        if write:
            self._write_config(vars(self.config))

    def update_config_from_args(self, args: argparse.Namespace) -> None:
        """Update configuration with values from command line arguments.

        Only updates values that were explicitly provided via CLI args.

        Args:
            args (argparse.Namespace): Parsed command line arguments
        """
        updates = {}
        if args.hosting:
            updates["hosting"] = args.hosting
        if args.model:
            updates["model_name"] = args.model

        self.update_config(updates, write=False)

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = DEFAULT_CONFIG
        self._write_config(vars(self.config))

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration variable.

        Args:
            key (str): The configuration key to retrieve
            default (Any, optional): Default value if key doesn't exist. Defaults to None.

        Returns:
            Any: The configuration value for the key, or default if not found
        """
        return self.config.get_value(key, default)

    def set_config_value(self, key: str, value: Any) -> None:
        """Set a specific configuration variable.

        Args:
            key (str): The configuration key to set
            value (Any): The value to set for the key
        """
        self.config.set_value(key, value)
        self._write_config(vars(self.config))
