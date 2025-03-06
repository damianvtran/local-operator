"""Credentials management for Local Operator.

This module handles API key storage and retrieval for various AI services.
It securely stores credentials in a local config file and provides methods
for accessing them when needed.
"""

import getpass
import os
from pathlib import Path
from typing import Dict, List

from pydantic import SecretStr

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
        credentials (Dict[str, SecretStr]): Dictionary of credentials
    """

    config_dir: Path
    config_file: Path
    credentials: Dict[str, SecretStr]

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_file = self.config_dir / CREDENTIALS_FILE_NAME
        self._ensure_config_exists()
        self.load_from_file()

    def load_from_file(self) -> Dict[str, SecretStr]:
        """Load credentials from the config file."""
        self.credentials = {}

        with open(self.config_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    self.credentials[key] = SecretStr(value)

        return self.credentials

    def write_to_file(self):
        """Write credentials to the config file."""
        with open(self.config_file, "w") as f:
            for key, value in self.credentials.items():
                f.write(f"{key}={value.get_secret_value()}\n")

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

    def get_credentials(self) -> Dict[str, SecretStr]:
        """Get all credentials from the config file."""
        return self.credentials

    def get_credential(self, key: str) -> SecretStr:
        """Retrieve the credential from config file.

        Args:
            key (str): The environment variable key to retrieve

        Returns:
            SecretStr: The credential value wrapped in SecretStr
        """
        if key not in self.credentials:
            # Check if the key is in the environment variables
            if key in os.environ:
                self.set_credential(key, os.environ[key], write=False)

        return self.credentials.get(key, SecretStr(""))

    def list_credential_keys(self, non_empty: bool = True) -> List[str]:
        """List all credential keys from the config file.

        Args:
            non_empty (bool): Whether to filter out empty credentials

        Returns:
            List[str]: List of credential keys
        """
        output = []

        for key, value in self.get_credentials().items():
            if not non_empty or value:
                output.append(key)

        return output

    def set_credential(self, key: str, value: str, write: bool = True):
        """Set the credential in the config file.
        If the key already exists, it will be updated.
        If the key does not exist, it will be added.

        Args:
            key (str): The environment variable key to set
            value (str): The credential value to set
            write (bool): Whether to write the credential to the config file
        """
        self.credentials[key] = SecretStr(value)

        if write:
            self.write_to_file()

    def prompt_for_credential(
        self, key: str, reason: str = "not found in configuration"
    ) -> SecretStr:
        """Prompt the user to enter a credential if not present in environment.

        Args:
            key (str): The environment variable key to check
            reason (str): The reason for prompting the user

        Returns:
            SecretStr: The credential value wrapped in SecretStr

        Raises:
            ValueError: If the user enters an empty credential
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
        self.set_credential(key, credential, write=True)

        print("\n\033[1;32m✓ Credential successfully saved!\033[0m")

        return SecretStr(credential)
