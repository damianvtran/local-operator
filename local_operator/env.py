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
from typing import Optional
from urllib.parse import urlsplit

# Always load .env from the project root, regardless of working directory.
dotenv_path = Path(__file__).parent.parent / ".env"

# STILL AN IMPORT-TIME LOAD, AND STILL `override=True`; only the no-file case
# became free.
#
# The precedence this encodes is a guarantee about WHEN the values land, not
# which of two sources wins: `override=True` writes every key in `.env` into
# ``os.environ`` at the moment this module is first imported, so a `.env` value
# beats a variable that was already exported, and a write made AFTER the import
# beats `.env`. Deferring the load to the first :func:`get_env_config` call
# would move that window, and the window is observable: `providers/registry.py`
# answers ``env_keys`` (``OPENROUTER_API_KEY`` and friends) straight out of
# ``os.environ``, so a caller that reads a provider key without ever calling
# ``get_env_config`` would stop seeing `.env` entirely, and a value set between
# import and that first call would silently invert precedence. A correctness
# regression there costs far more than the ~12 ms this saves, so the window does
# NOT move.
#
# What the guard removes is the cost of asking about a file that is not there.
# ``load_dotenv`` on a missing path is a no-op — measured: it returns ``False``
# and touches no environment variable — but reaching it means importing the
# ``dotenv`` package first, which is ~125 ms standalone and ~12.7 ms of marginal
# CPU on this path (most of dotenv's own dependencies are already loaded by the
# package ``__init__``). Importing it inside the branch keeps that import off
# every run with no ``.env`` — every CI run, every isolated test run, and this
# worktree — while a checkout that HAS a `.env` behaves exactly as before,
# because then the import and the call both still happen, here, at import time.
if dotenv_path.is_file():
    from dotenv import load_dotenv

    load_dotenv(dotenv_path, override=True)

os.environ["ANONYMIZED_TELEMETRY"] = "false"


#: The canonical Radient Agent Hub API root — the ONE place this host is written.
#:
#: The version segment belongs HERE rather than at each join site. Every hub path
#: is built by joining onto whatever base a caller hands
#: :class:`~local_operator.clients.radient.RadientClient` (``{base}/agents/{id}``,
#: ``{base}/agents/publish``), so a base missing ``/v1`` addresses an unversioned
#: route that answers a bare ``{"error":"Not Found"}`` 404, and a caller that
#: appends ``/v1`` itself doubly prefixes it. The CLI shipped one literal per call
#: site and two of the three were wrong — ``https://api.radienthq.com`` (no
#: version segment) as the paste-time default in ``agents delete`` and
#: ``agents push``, and the non-existent ``https://api.radientlabs.ai`` in
#: ``agents pull`` — so a delete, a push and a pull could each resolve a
#: different host from the same configuration. :func:`resolve_radient_api_base_url`
#: is the single rule that replaced them.
DEFAULT_RADIENT_API_BASE_URL = "https://api.radienthq.com/v1"


def normalize_radient_api_base_url(value: str) -> str:
    """Return ``value`` as a hub API root that carries its version segment.

    A value naming no path at all (``https://api.radienthq.com``) is completed
    with ``/v1``; a value that already names a path is the operator's own gateway
    and is returned otherwise untouched, so the documented ``…/v1`` shape — the
    one the credential resolver matches a destination on — survives
    byte-identical rather than being rewritten under a caller that compared it.

    Args:
        value: A hub API root, with or without its version segment.

    Returns:
        str: The same root, versioned and free of a trailing slash.
    """
    base = value.strip().rstrip("/")
    if urlsplit(base).path in {"", "/"}:
        return base + "/v1"
    return base


def resolve_radient_api_base_url(configured: Optional[str] = None) -> str:
    """Resolve the Radient Agent Hub API root a caller should address.

    Precedence: ``configured`` (the ``radient_base_url`` a caller read from
    ``config.yml``), then ``RADIENT_API_BASE_URL``, then
    :data:`DEFAULT_RADIENT_API_BASE_URL`. That is the same rule
    :func:`get_env_config` already applies, so a caller cannot answer this
    question a fourth way.

    The version segment is completed at each of the two places a value is
    PRODUCED — here for the configured one, and in :func:`get_env_config` for the
    environment one — rather than at each consumer, which is what makes this the
    whole answer for every surface. Normalizing only on this path left the CLI
    on ``/v1`` and the server and desktop transport, which read
    ``EnvConfig.radient_api_base_url`` directly, on the bare host: one
    configuration, two destinations.

    Args:
        configured: An explicit base, or None to fall through to the environment.

    Returns:
        str: A versioned hub API root.
    """
    if configured:
        return normalize_radient_api_base_url(configured)
    return get_env_config().radient_api_base_url


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
    radient_api_base_url: str = DEFAULT_RADIENT_API_BASE_URL
    radient_client_id: str = ""


def get_env_config() -> EnvConfig:
    """
    Loads environment variables and returns an EnvConfig instance.

    ``RADIENT_API_BASE_URL`` is normalized HERE, where the value is produced,
    not at each consumer: the server routes, the desktop transport and the
    speech/models clients all address ``EnvConfig.radient_api_base_url``
    directly, so a hand-set bare host has to arrive at them already carrying the
    ``/v1`` segment the hub's routes live under — the same completion
    :func:`resolve_radient_api_base_url` applies to the configured value. Without
    it one configuration resolved two destinations, working in the CLI and
    404ing in the desktop transport.

    Returns:
        EnvConfig: The loaded environment configuration.
    """
    return EnvConfig(
        radient_api_base_url=normalize_radient_api_base_url(
            os.getenv("RADIENT_API_BASE_URL", DEFAULT_RADIENT_API_BASE_URL)
        ),
        radient_client_id=os.getenv("RADIENT_CLIENT_ID", "b0fd1aa8-05a2-4ca2-bac2-82db293e7584"),
    )
