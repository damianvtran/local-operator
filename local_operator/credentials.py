"""Credentials management for Local Operator.

This module handles API key storage and retrieval for various AI services.
It securely stores credentials in a local config file and provides methods
for accessing them when needed.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


class CredentialManager:
    """Manages API credentials storage and retrieval."""

    def __init__(self):
        self.config_dir = Path.home() / ".local-operator"
        self.config_file = self.config_dir / "config.env"
        self._ensure_config_exists()
        # Load environment variables from config file
        load_dotenv(self.config_file)

    def _ensure_config_exists(self):
        """Ensure config directory and file exist, prompt for API key if needed."""
        self.config_dir.mkdir(exist_ok=True, mode=0o700)
        if not self.config_file.exists():
            print("\033[1;36m╭──────────────────────────────────────────────────╮\033[0m")
            print("\033[1;36m│ DeepSeek API Key Setup                           │\033[0m")
            print("\033[1;36m│──────────────────────────────────────────────────│\033[0m")
            print("\033[1;36m│ API key not found. Let's set it up.              │\033[0m")
            print("\033[1;36m╰──────────────────────────────────────────────────╯\033[0m")

            api_key = input("\033[1m\033[94mPlease enter your DeepSeek API key: \033[0m").strip()
            if not api_key:
                raise ValueError("\033[1;31mAPI key is required to use this application\033[0m")

            with open(self.config_file, "w") as f:
                f.write(f"DEEPSEEK_API_KEY={api_key}\n")
            self.config_file.chmod(0o600)

            print("\n\033[1;32m✓ API key successfully saved!\033[0m")

    def get_api_key(self, key: str):
        """Retrieve the API key from config file.

        Args:
            key (str): The environment variable key to retrieve

        Returns:
            str: The API key value
        """
        return os.getenv(key)

    def prompt_for_api_key(self, key: str) -> str:
        """Prompt the user to enter an API key if not present in environment.

        Args:
            key (str): The environment variable key to check

        Returns:
            str: The API key value
        """
        api_key = self.get_api_key(key)
        if not api_key:
            print(f"{key} not found in configuration.")
            api_key = input(f"Please enter your {key}: ").strip()
            if not api_key:
                raise ValueError(f"{key} is required to use this application")

            # Save the new API key to config file
            with open(self.config_file, "a") as f:
                f.write(f"\n{key}={api_key}\n")
            self.config_file.chmod(0o600)

            # Reload environment variables
            load_dotenv(self.config_file, override=True)

        return api_key
