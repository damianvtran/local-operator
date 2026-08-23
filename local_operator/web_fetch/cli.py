"""CLI configuration and smoke-test experience for built-in web fetch.

Mirrors ``web_search/cli.py``: a ``fetch`` subcommand with ``status`` (is the
render extra active, what is the SSRF/TTL posture), ``test`` (one live fetch),
and focused ``set`` mutators that each persist a single field. The status view
matters because the good renderer is behind an extra — a user needs to see
whether markdownify or the stdlib fallback is actually in effect.
"""

from __future__ import annotations

import argparse
import asyncio

from local_operator.config import ConfigManager
from local_operator.paths import config_dir


def add_fetch_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    parent_parser: argparse.ArgumentParser,
) -> None:
    """Install ``fetch`` and its focused configuration subcommands."""
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Configure and test headless web fetching",
        parents=[parent_parser],
    )
    commands = fetch_parser.add_subparsers(dest="fetch_command")
    commands.add_parser("status", help="Show fetch config, render backend, and cache stats")

    test = commands.add_parser("test", help="Run one real fetch through the configured engine")
    test.add_argument("url")
    test.add_argument("--raw", action="store_true", help="Return the source verbatim")
    test.add_argument("--refresh", action="store_true", help="Bypass the cache")
    test.add_argument("--max-bytes", type=int, default=None, dest="max_bytes")

    setter = commands.add_parser("set", help="Set a single fetch config field")
    setter.add_argument(
        "key",
        choices=("enabled", "ttl", "allow-private", "backend"),
    )
    setter.add_argument(
        "value",
        help="on/off for enabled & allow-private; seconds for ttl; auto/stdlib for backend",
    )


def _manager() -> ConfigManager:
    return ConfigManager(config_dir())


def format_fetch_status(manager: ConfigManager) -> str:
    """Plain, width-tolerant status table shared by CLI and TUI.

    The render backend line is computed by ATTEMPTING the import, not by reading
    the config's ``render_backend`` alone: what a user needs to know is which
    renderer will actually run, which depends on whether the ``[fetch]`` extra is
    importable in this environment.
    """
    from local_operator.web_fetch.render import html_backend_available
    from local_operator.web_fetch.service import cache_dir, load_fetch_settings

    settings = load_fetch_settings(manager)
    if settings.render_backend == "stdlib":
        backend = "stdlib (forced)"
    elif html_backend_available():
        backend = "markdownify"
    else:
        backend = "stdlib (extra absent — install local-operator[fetch] for better rendering)"

    cache_count = 0
    try:
        cache_count = sum(1 for p in cache_dir().iterdir() if p.suffix == ".json")
    except OSError:
        cache_count = 0

    rows = [
        f"Web fetch: {'on' if settings.enabled else 'off'}",
        f"Render backend: {backend}",
        f"Timeout: {settings.timeout_seconds:g}s | max bytes: {settings.max_bytes} | "
        f"max redirects: {settings.max_redirects}",
        f"Cache TTL: {settings.cache_ttl_seconds}s | cached URLs: {cache_count}",
        f"Allow private/loopback targets: {'yes' if settings.allow_private else 'no'}",
        f"Enrichment (.md / llms.txt): {'on' if settings.enrich else 'off'}",
    ]
    return "\n".join(rows)


async def _test_fetch(args: argparse.Namespace) -> int:
    from local_operator.web_fetch.tool import run_fetch

    preview, details, is_error = await run_fetch(
        args.url,
        tool_name="web_fetch",
        raw=bool(getattr(args, "raw", False)),
        max_bytes=getattr(args, "max_bytes", None),
        refresh=bool(getattr(args, "refresh", False)),
    )
    if is_error:
        print(f"error: {preview}")
        return 1
    spill = details.get("spill") if isinstance(details, dict) else None
    handle = spill.get("handle") if isinstance(spill, dict) else None
    print(preview)
    print(
        "\n--- "
        f"status={details.get('status')} method={details.get('render_method')} "
        f"bytes={details.get('bytes')} lines={details.get('lines')} "
        f"cache={details.get('cache')}" + (f" spill={handle}" if handle else " spill=(inline)")
    )
    return 0


def _set_field(args: argparse.Namespace) -> int:
    from local_operator.web_fetch.service import (
        set_allow_private,
        set_cache_ttl,
        set_fetch_enabled,
        set_render_backend,
    )

    manager = _manager()
    key = args.key
    value = str(args.value).strip().lower()
    try:
        if key == "enabled":
            set_fetch_enabled(manager, value in ("on", "true", "1", "yes"))
        elif key == "allow-private":
            set_allow_private(manager, value in ("on", "true", "1", "yes"))
        elif key == "ttl":
            set_cache_ttl(manager, int(value))
        elif key == "backend":
            if value not in ("auto", "stdlib"):
                raise ValueError("backend must be 'auto' or 'stdlib'")
            set_render_backend(manager, value)
        else:  # pragma: no cover - argparse choices guard this
            print(f"error: unknown fetch field {key}")
            return 1
    except ValueError as error:
        print(f"error: {error}")
        return 1
    print(f"web_fetch {key} set to {value}. Reload a running session to apply the master switch.")
    return 0


def fetch_command(args: argparse.Namespace) -> int:
    """Dispatch ``fetch`` configuration and smoke-test commands."""
    command = args.fetch_command or "status"
    if command == "status":
        print(format_fetch_status(_manager()))
        return 0
    if command == "test":
        return asyncio.run(_test_fetch(args))
    if command == "set":
        return _set_field(args)
    print(f"error: unknown fetch command {command}")
    return 1
