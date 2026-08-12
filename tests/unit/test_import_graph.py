"""Import-graph guards for the startup path.

Startup cost is invisible in review: adding one module-level ``import`` to a
module the CLI already touches costs every invocation — ``--version``, shell
completion, every scheduler tick — and nothing fails. These tests pin the
modules that were deliberately moved OFF the startup path so a revert shows up
as a red test rather than as a slow CLI nobody profiles.

Every module listed below has a measured cost recorded next to it. The numbers
come from ``scripts/bench_base_overhead.py`` on this repo (Python 3.13, Apple
M3 Max); re-run it before changing any of them.

Each check runs in a FRESH SUBPROCESS. An in-process ``sys.modules`` assertion
is worthless here: pytest has already imported half the tree by the time a test
body runs, so a module the CLI wrongly pulls would look "already imported" and
the assertion would pass on a real regression.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent

# Dump the exact module set a virgin interpreter ends up with after importing
# one target. Printed as JSON on stdout so a stray warning on stderr cannot
# corrupt the result.
_PROBE = """
import json, importlib, sys
importlib.import_module(sys.argv[1])
print(json.dumps(sorted(sys.modules)))
"""


def _imported_modules(target: str) -> set[str]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, target],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, f"importing {target} failed:\n{proc.stderr[-3000:]}"
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


@pytest.fixture(scope="module")
def cli_modules() -> set[str]:
    """Modules loaded by importing the console-script entry point."""
    return _imported_modules("local_operator.cli")


@pytest.fixture(scope="module")
def session_factory_modules() -> set[str]:
    """Modules loaded by importing the composition root every host funnels
    through (server, scheduler, exec worker — not just the CLI)."""
    return _imported_modules("local_operator.session_factory")


@pytest.fixture(scope="module")
def configure_modules() -> set[str]:
    """Modules loaded by importing ``local_operator.model.configure``, which
    ``session_factory._prepare`` imports on every session build."""
    return _imported_modules("local_operator.model.configure")


def _assert_absent(modules: set[str], top_level: str, why: str) -> None:
    """Fail when ``top_level`` (or any submodule of it) got imported."""
    offenders = sorted(m for m in modules if m == top_level or m.startswith(top_level + "."))
    assert not offenders, f"{top_level} is back on the startup path ({why}); saw {offenders[:5]}"


# --- CLI entry point ---------------------------------------------------------


def test_cli_import_does_not_load_asyncio(cli_modules: set[str]) -> None:
    # The single heaviest item on the CLI's import graph: 34.4 ms, +6.5 MB RSS
    # and +77 modules. Only the interactive TUI/REPL tail in main() needs an
    # event loop, so `import asyncio` lives inside that branch — `--version`,
    # `--help`, shell completion and the config/credential/agents/login
    # subcommands all return before it, and `exec`/`serve` bring their own loop
    # from exec_mode/the server module. A module-scope `import asyncio` in
    # cli.py (or in anything cli.py imports) silently reverts the batch's
    # largest saving, which is exactly what this pin exists to make loud.
    _assert_absent(cli_modules, "asyncio", "34.4 ms / 6.5 MB; only the interactive tail needs it")


def test_cli_import_does_not_load_pillow(cli_modules: set[str]) -> None:
    # Pillow + pillow-heif cost 23.4 ms and +7.6 MB RSS (+75 modules) and exist
    # solely to decode image inputs — HEIC/HEIF attachments, and files the
    # `read` tool returns as image blocks. cli.py imports helpers.py for
    # setup_cross_platform_environment; helpers.py used to probe these two at
    # module scope, so every run paid for an input almost no run has. They now
    # load inside helpers.pillow_image_module() / helpers.heif_image_module(),
    # and read's magic-byte sniff decides whether to call them at all so a text
    # read never pays either.
    _assert_absent(cli_modules, "PIL", "23.4 ms / 7.6 MB; only image decoding needs it")
    _assert_absent(cli_modules, "pillow_heif", "part of the same 23.4 ms / 7.6 MB probe")


def test_cli_import_does_not_load_local_operator_types(cli_modules: set[str]) -> None:
    # local_operator.types builds its pydantic model classes at import time:
    # 51.7 ms total, ~35 ms of which is model construction on top of pydantic
    # itself. helpers.py needed exactly one name from it (ResponseJsonSchema, in
    # process_json_response), so it moved to a function-local import plus a
    # TYPE_CHECKING binding for the annotation.
    _assert_absent(
        cli_modules,
        "local_operator.types",
        "~35 ms of pydantic model construction; keep it lazy in helpers.py",
    )


def test_cli_import_does_not_load_tokenizer(cli_modules: set[str]) -> None:
    # tiktoken itself is only 7.8 ms, but the cl100k_base encoding it exists to
    # provide costs 84 ms and +43.6 MB RSS. compaction/tokens.py loads it on
    # first estimate, never at import.
    _assert_absent(cli_modules, "tiktoken", "84 ms / 43.6 MB once the encoding loads")


def test_cli_import_does_not_load_tui_or_browser_stacks(cli_modules: set[str]) -> None:
    # The TUI is one of several front ends and the headless paths (exec, serve,
    # config, credential) must never pay for it; likewise no browser automation
    # SDK belongs on an import that `--version` performs.
    _assert_absent(cli_modules, "textual", "TUI front end; headless paths never render")
    _assert_absent(cli_modules, "playwright", "browser automation; not a startup concern")


def test_cli_import_does_not_load_http_stacks(cli_modules: set[str]) -> None:
    # requests costs 53.7 ms and +12.6 MB RSS (+228 modules) from cold; httpx
    # costs 88.2 ms / +16.5 MB. Neither belongs on an import that may only be
    # about to print a version string — both arrive with the provider layer,
    # which is lazy-imported at the point of use.
    _assert_absent(cli_modules, "requests", "53.7 ms / 12.6 MB; provider layer loads it lazily")
    _assert_absent(cli_modules, "httpx", "88.2 ms / 16.5 MB; provider layer loads it lazily")


def test_cli_import_does_not_load_scientific_stack(cli_modules: set[str]) -> None:
    # Neither is a declared runtime dependency of the harness. They are pinned
    # anyway because tool code is the natural place for someone to reach for a
    # dataframe, and a module-level import there would land on every startup.
    _assert_absent(cli_modules, "numpy", "not a startup dependency at all")
    _assert_absent(cli_modules, "pandas", "not a startup dependency at all")


# --- Composition root --------------------------------------------------------


def test_session_factory_import_stays_off_the_heavy_stacks(
    session_factory_modules: set[str],
) -> None:
    # session_factory is the composition root for the server, the scheduler and
    # the exec worker as well as the CLI, and its whole design is that the
    # engine, registry and provider modules load inside the functions that need
    # them. These four are the ones that would silently undo that.
    _assert_absent(session_factory_modules, "PIL", "23.4 ms / 7.6 MB")
    _assert_absent(session_factory_modules, "pillow_heif", "23.4 ms / 7.6 MB")
    _assert_absent(session_factory_modules, "tiktoken", "84 ms / 43.6 MB with the encoding")
    _assert_absent(session_factory_modules, "textual", "TUI front end; the server has no terminal")


# --- Per-session model configuration -----------------------------------------


def test_configure_import_does_not_load_requests(configure_modules: set[str]) -> None:
    # model/configure.py is imported by session_factory._prepare on EVERY
    # session build, and its only requests caller is validate_model, an
    # interactive credential check. Cold, requests costs 53.7 ms / +12.6 MB /
    # +228 modules; alongside the httpx stack a real session already loads it
    # still costs +5.8 ms / +2.9 MB / +127 modules. It now imports inside
    # validate_model.
    _assert_absent(
        configure_modules,
        "requests",
        "+2.9 MB / +127 modules per session; only validate_model needs it",
    )
