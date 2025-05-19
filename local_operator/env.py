"""
Environment configuration module for local_operator.

Loads environment variables from a .env file using python-dotenv,
and provides a typed EnvConfig for dependency injection.

EnvConfig currently supports:
- RADIENT_API_BASE_URL: Optional[str]
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field

# Always load .env from the project root, regardless of working directory
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path, override=True)


@dataclass(frozen=True)
class EnvConfig:
    """
    Typed environment configuration for the application.

    Attributes:
        radient_api_base_url: Base URL for the Radient API.
    """

    radient_api_base_url: str = Field(
        default="https://api.radienthq.com/v1",
        description="Base URL for the Radient API.",
    )

    radient_client_id: str = Field(
        default="",
        description="Client ID for Radient API OAuth flows.",
    )


def get_env_config() -> EnvConfig:
    """
    Loads environment variables and returns an EnvConfig instance.

    Returns:
        EnvConfig: The loaded environment configuration.
    """
    return EnvConfig(
        radient_api_base_url=os.getenv("RADIENT_API_BASE_URL", "https://api.radienthq.com/v1"),
        radient_client_id=os.getenv("RADIENT_CLIENT_ID", "b0fd1aa8-05a2-4ca2-bac2-82db293e7584"),
    )
