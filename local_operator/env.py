"""
Environment configuration module for local_operator.

Loads environment variables from a .env file using python-dotenv,
and provides a typed EnvConfig for dependency injection.

EnvConfig currently supports:
- RADIENT_API_BASE_URL: Optional[str]
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


@dataclass(frozen=True)
class EnvConfig:
    """
    Typed environment configuration for the application.

    Attributes:
        radient_api_base_url: Optional base URL for the Radient API.
    """

    radient_api_base_url: Optional[str] = None


def get_env_config() -> EnvConfig:
    """
    Loads environment variables and returns an EnvConfig instance.

    Returns:
        EnvConfig: The loaded environment configuration.
    """
    return EnvConfig(radient_api_base_url=os.getenv("RADIENT_API_BASE_URL"))
