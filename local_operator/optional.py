"""Helpers for features that live behind an install extra.

The default ``pip install local-operator`` deliberately carries only what the
core agent needs: the provider clients, the tool loop, and the terminal UI.
Everything with a heavy or platform-fragile install footprint — the HTTP API
server, the MCP client SDK, image decoding, the BPE tokenizer — sits behind a
named extra so that a plain install stays small and fast, which matters most
on Windows where several of those chains ship compiled wheels.

The cost of that split is that a user can reach a code path whose dependency
is not installed. When that happens the failure must name the missing extra
and the exact command that fixes it; a bare ``ModuleNotFoundError: No module
named 'uvicorn'`` tells the user nothing about how this package is packaged.

Every optional-dependency guard in the codebase routes its message through
this module so the wording, quoting, and extra names stay consistent.
"""

from __future__ import annotations

__all__ = ["EXTRAS", "missing_extra_error", "MissingExtraError", "require_extra"]

#: The install extras this package declares, mapped to a short human
#: description of what each unlocks. Keep in sync with the
#: ``[project.optional-dependencies]`` table in ``pyproject.toml`` — the
#: assertion in :func:`missing_extra_error` turns a drift into a loud failure
#: during development rather than a misleading hint shown to a user.
EXTRAS: dict[str, str] = {
    "server": "the HTTP API server and background scheduler",
    "mcp": "Model Context Protocol client support",
    "images": "HEIC/HEIF image attachment decoding",
    "tokenizer": "exact BPE token counting for context management",
    "all": "every optional feature",
}


class MissingExtraError(ImportError):
    """Raised when a feature is used without its extra installed.

    Subclasses :class:`ImportError` on purpose: the call sites that guard
    optional features already catch ``ImportError``, so raising this from a
    deeper helper stays compatible with the degradation paths that wrap them.
    """


def missing_extra_error(extra: str, feature: str) -> str:
    """Build the user-facing message for a missing install extra.

    Args:
        extra: The extra name, e.g. ``"server"``. Must be a declared extra.
        feature: What the user was trying to do, phrased as a sentence
            subject — ``"The HTTP API server"``, ``"MCP support"``. It is
            used verbatim at the start of the message.

    Returns:
        A single-line message naming the feature, the extra, and the exact
        install command. The command quotes the requirement because the
        bracket syntax is glob-expanded by zsh and most shells otherwise.
    """
    assert extra in EXTRAS, f"Unknown extra {extra!r}; declared extras are {sorted(EXTRAS)}"
    return (
        f'{feature} requires the "{extra}" extra. '
        f'Install it with: pip install "local-operator[{extra}]"'
    )


def require_extra(module: str, extra: str, feature: str) -> object:
    """Import ``module``, converting a missing install into a clear error.

    This is for call sites that cannot usefully degrade and would otherwise
    propagate a raw ``ModuleNotFoundError``. Sites that *can* degrade should
    keep their own ``try``/``except ImportError`` and use
    :func:`missing_extra_error` for the warning text instead.

    Args:
        module: Dotted module path to import, e.g. ``"uvicorn"``.
        extra: The extra that provides it.
        feature: Human description of the feature, as for
            :func:`missing_extra_error`.

    Returns:
        The imported module.

    Raises:
        MissingExtraError: If the module is not installed.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(missing_extra_error(extra, feature)) from exc
