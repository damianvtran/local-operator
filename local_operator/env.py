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

# Always load .env from the project root, regardless of working directory
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path, override=True)

os.environ["ANONYMIZED_TELEMETRY"] = "false"


@dataclass(frozen=True)
class EnvConfig:
    """
    Typed environment configuration for the application.

    Attributes:
        radient_api_base_url: Base URL for the Radient API.
        radient_client_id: Client ID for Radient API OAuth flows.
    """

    # Plain dataclass defaults: this is a stdlib dataclass, not a pydantic
    # model, so a ``pydantic.Field(...)`` here would be stored verbatim as the
    # default and hand callers a FieldInfo where the annotation promises a str.
    radient_api_base_url: str = "https://api.radienthq.com/v1"
    radient_client_id: str = ""


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
