"""Credentials management for Local Operator.

This module handles API key storage and retrieval for various AI services.
It securely stores credentials in a local config file and provides methods
for accessing them when needed.
"""

import getpass
import os
from pathlib import Path

from dotenv import load_dotenv

# Name of the file used to store credentials in .env format
CREDENTIALS_FILE_NAME: str = "credentials.env"


class CredentialManager:
    """Manages secure storage and retrieval of API credentials.

    This class handles storing API keys and other sensitive credentials in a local
    encrypted configuration file. It provides methods for safely reading and writing
    credentials while maintaining proper file permissions.

    Attributes:
        config_dir (Path): Directory where credential files are stored
        config_file (Path): Path to the credentials file
    """

    config_dir: Path
    config_file: Path

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_file = self.config_dir / CREDENTIALS_FILE_NAME
        self._ensure_config_exists()
        # Load environment variables from config file
        load_dotenv(self.config_file)

    def _ensure_config_exists(self):
        """Ensure the credentials configuration file exists and has proper permissions.

        Creates the config directory and credentials file if they don't exist.
        Sets restrictive file permissions (600) to protect sensitive credential data.
        The file permissions ensure only the owner can read/write the credentials.

        The config file is created as an empty file that will be populated later
        when credentials are added via set_credential().
        """
        if not self.config_file.exists():
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.touch()
            self.config_file.chmod(0o600)

    def get_credential(self, key: str) -> str:
        """Retrieve the credential from config file.

        Args:
            key (str): The environment variable key to retrieve

        Returns:
            str: The credential value
        """
        return os.getenv(key, "")

    def set_credential(self, key: str, value: str):
        """Set the credential in the config file.
        If the key already exists, it will be updated.
        If the key does not exist, it will be added.

        Args:
            key (str): The environment variable key to set
            value (str): The credential value to set
        """
        with open(self.config_file, "r") as f:
            lines = f.readlines()

        with open(self.config_file, "w") as f:
            line_updated = False
            for line in lines:
                if line.startswith(f"{key}="):
                    f.write(f"{key}={value}\n")
                    line_updated = True
                else:
                    f.write(line)
            if not line_updated:
                f.write(f"\n{key}={value}\n")

        self.config_file.chmod(0o600)

        # Reload environment variables
        load_dotenv(self.config_file, override=True)

    def prompt_for_credential(self, key: str, reason: str = "not found in configuration") -> str:
        """Prompt the user to enter a credential if not present in environment.

        Args:
            key (str): The environment variable key to check
            reason (str): The reason for prompting the user

        Returns:
            str: The credential value
        """
        # Calculate border length based on key length
        line_length = max(50, len(key) + 12)
        border = "─" * line_length

        # Create box components with colors
        cyan = "\033[1;36m"
        blue = "\033[1;94m"
        reset = "\033[0m"

        # Print the setup box
        print(f"{cyan}╭{border}╮{reset}")
        setup_padding = " " * (line_length - len(key) - 7)
        print(f"{cyan}│ {key} Setup{setup_padding}│{reset}")
        print(f"{cyan}├{border}┤{reset}")
        reason_padding = " " * (line_length - len(key) - len(reason) - 3)
        print(f"{cyan}│ {key} {reason}.{reason_padding}│{reset}")
        print(f"{cyan}╰{border}╯{reset}")

        # Prompt for API key using getpass to hide input
        credential = getpass.getpass(f"{blue}Please enter your {key}: {reset}").strip()
        if not credential:
            raise ValueError(f"\033[1;31m{key} is required for this step.\033[0m")

        # Save the new API key to config file
        with open(self.config_file, "a") as f:
            f.write(f"\n{key}={credential}\n")
        self.config_file.chmod(0o600)

        print("\n\033[1;32m✓ Credential successfully saved!\033[0m")

        # Reload environment variables
        load_dotenv(self.config_file, override=True)

        return credential
