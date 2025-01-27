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

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_file = self.config_dir / "config.env"
        self._ensure_config_exists()
        # Load environment variables from config file
        load_dotenv(self.config_file)

    def _ensure_config_exists(self):
        """Ensure config directory and file exist, prompt for API key if needed."""
        if not self.config_file.exists():
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.touch()
            self.config_file.chmod(0o600)

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
            # Calculate border length based on key length
            line_length = max(50, len(key) + 12)
            border = "─" * line_length

            # Create box components with colors
            cyan = "\033[1;36m"
            blue = "\033[1;94m"
            reset = "\033[0m"

            # Print the setup box
            print(f"{cyan}╭{border}╮{reset}")
            setup_padding = " " * (line_length - len(key) - 9)
            print(f"{cyan}│ {key} Setup{setup_padding}│{reset}")
            print(f"{cyan}├{border}┤{reset}")
            not_found_padding = " " * (line_length - len(key) - 28)
            print(f"{cyan}│ {key} not found in configuration.{not_found_padding}│{reset}")
            print(f"{cyan}╰{border}╯{reset}")

            # Prompt for API key
            api_key = input(f"{blue}Please enter your {key}: {reset}").strip()
            if not api_key:
                raise ValueError(f"\033[1;31m{key} is required to use this application\033[0m")

            # Save the new API key to config file
            with open(self.config_file, "a") as f:
                f.write(f"\n{key}={api_key}\n")
            self.config_file.chmod(0o600)

            print("\n\033[1;32m✓ API key successfully saved!\033[0m")

            # Reload environment variables
            load_dotenv(self.config_file, override=True)

        return api_key
