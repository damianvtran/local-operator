"""The headless CLI's boot activation: ONE locale move, before any command renders.

WHAT THIS FILE PROVES, and why each case exists rather than only the happy one:

* a headless command renders the language the shared chain resolves — the
  defect this change closes was that ``cli.main`` never called
  ``set_active_locale`` at all, so every keyed line under ``lop <command>``
  rendered English whatever ``language:`` said, while ``--help`` in the same
  shell was translated;
* the ENGLISH bytes do not change. The control is this same code with the
  activation switched off (``boot.set_active_locale`` made a no-op), which is
  exactly the pre-change path — so a same-locale move that quietly
  re-materialised a constant in another language fails here;
* the move happens ONCE per process, and only where it is correct: a ``--json``
  invocation is left alone, and the ``--remote`` dispatch (which activates
  before ``cli.main`` reaches its own dispatch step) still activates once;
* a locale with no catalogue for the namespace falls back to ENGLISH — the
  renderer's additive contract — never to raw message codes.

Every case drives the real ``cli.main`` entry point with an isolated config
root, so what is asserted is the surface an operator runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator import cli as cli_module
from local_operator.i18n import boot as i18n_boot
from local_operator.i18n import catalogues, runtime

NAMESPACE = "cli.browser"
HEALTHY = "cli.browser.daemon_healthy"
STATUS_NO = "cli.browser.status_no"


@pytest.fixture()
def headless_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An isolated config root, with no login-shell probe on the startup path.

    ``LOP_LANG`` is REMOVED rather than set: the chain is config -> env -> OS,
    and a developer's shell must not decide what these assertions mean. Every
    case that cares writes ``language:``, which outranks the env and the OS.
    """
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("LOP_LANG", raising=False)
    # Neutralize environment repair (a login-shell PATH round-trip): the seam
    # every other `main()` test patches, and unrelated to copy.
    monkeypatch.setattr(cli_module, "setup_cross_platform_environment", lambda: None)
    yield root


@pytest.fixture()
def fixed_browser_status(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """``browser status`` with a deterministic answer and NO daemon probe.

    The real probe reaches a fixed port, so on a developer's machine it reports
    the live bridge — which would make an English byte comparison depend on
    whose laptop ran it.
    """
    from local_operator.browser_bridge import install as browser_install

    result: dict[str, Any] = {
        "installed": False,
        "healthy": False,
        "health": {},
        "port": 4099,
        "state": None,
        "paired": False,
        "supervisor": browser_install.LABEL,
        "config_root": None,
    }
    monkeypatch.setattr(browser_install, "status", lambda port=None: dict(result))
    monkeypatch.setattr(browser_install, "stale_heartbeat_age", lambda: None)
    return result


@pytest.fixture()
def activation_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every locale move the boot path attempts, without performing it."""
    calls: list[str] = []
    monkeypatch.setattr(i18n_boot, "set_active_locale", calls.append)
    return calls


def _write_language(root: Path, locale: str) -> None:
    """Write the ``language:`` row the resolution chain reads, as the app persists it."""
    (root / "config.yml").write_text(f"values:\n  language: {locale}\n", encoding="utf-8")


def _render(locale: str, code: str, *, namespace: str = NAMESPACE, **params: Any) -> str:
    """One line rendered from the catalogue FILE, never through the active locale.

    Reading the file directly is what keeps the expectation independent of the
    mechanism under test: if this went through ``tr``, a broken activation would
    make the "expected" string English too and the assertion would pass on the
    very defect it exists to catch.
    """
    source = catalogues.load_catalogue(locale, namespace)[code]
    return runtime.render_message(source, dict(params), locale)


def _run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> tuple[int, str]:
    """Drive the real entry point and return ``(exit code, stdout)``."""
    monkeypatch.setattr(sys, "argv", ["lop", *argv])
    code = cli_module.main()
    return code, capsys.readouterr().out


def _browser_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str]:
    return _run(monkeypatch, capsys, ["browser", "status"])


# --- the fix ---------------------------------------------------------------


def test_a_headless_command_renders_the_configured_language(
    headless_config: Path,
    fixed_browser_status: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``language: es`` reaches ``lop browser status`` with no TUI in the process.

    Spanish is the surface the QA round measured — the same command printed
    ``daemon healthy:`` under an ``es`` config while ``lop browser --help`` was
    Spanish. The fallback test below covers a locale whose namespace has no
    catalogue.
    """
    _write_language(headless_config, "es")
    code, out = _browser_status(monkeypatch, capsys)
    assert code == 1  # the fake status reports an unhealthy daemon
    assert _render("es", HEALTHY, value=_render("es", STATUS_NO)) in out
    assert _render("en", HEALTHY, value=_render("en", STATUS_NO)) not in out


def test_english_output_is_identical_to_the_hook_removed_control(
    headless_config: Path,
    fixed_browser_status: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The regression half, with a control that can actually differ.

    Two things are asserted, because ``set_active_locale("en")`` is a documented
    no-op (``i18n.text`` returns before any refresher): the bytes must equal the
    run where the boot hook is REMOVED — the true pre-change path — and the
    measured run must have made the move at all. The second half is what makes
    this fail when the call disappears; the first is what makes it fail if an
    English dispatch ever re-materialises copy in another language.
    """
    _write_language(headless_config, "en")
    with monkeypatch.context() as off:
        off.setattr(cli_module, "_install_headless_locale", lambda args: None)
        _, control = _browser_status(monkeypatch, capsys)
    calls: list[str] = []
    with monkeypatch.context() as spy:
        spy.setattr(i18n_boot, "set_active_locale", calls.append)
        _, measured = _browser_status(monkeypatch, capsys)
    assert measured == control
    assert calls == ["en"]
    assert _render("en", HEALTHY, value=_render("en", STATUS_NO)) in measured


def test_the_boot_path_activates_once_with_the_resolved_locale(
    headless_config: Path,
    fixed_browser_status: dict[str, Any],
    activation_calls: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ONE move, carrying the chain's answer — this fails if the call is removed.

    The count is the contract (``set_active_locale`` re-materialises module
    constants on every real move) and the value is the other half: activating
    with anything but the resolved locale would be a language nobody asked for.
    """
    _write_language(headless_config, "es")
    _browser_status(monkeypatch, capsys)
    assert activation_calls == ["es"]


def test_without_the_activation_the_same_command_stays_english(
    headless_config: Path,
    fixed_browser_status: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The DISCRIMINATOR: the Spanish above is produced by the activation itself.

    Same command, same config, same catalogue — with the move suppressed the
    copy is English, so the first test cannot pass for want of some other cause.
    """
    _write_language(headless_config, "es")
    with monkeypatch.context() as off:
        off.setattr(i18n_boot, "set_active_locale", lambda locale: None)
        _, out = _browser_status(monkeypatch, capsys)
    assert _render("en", HEALTHY, value=_render("en", STATUS_NO)) in out
    assert _render("es", HEALTHY, value=_render("es", STATUS_NO)) not in out


# --- the additive fallback -------------------------------------------------


def test_a_locale_without_a_catalogue_renders_english_not_codes(
    headless_config: Path,
    fixed_browser_status: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No catalogue for the namespace -> English, never a message CODE.

    ``ru`` is a shipped locale with no ``cli.browser`` wave on this head, and
    the catalogue root is ALSO re-pointed at a tree holding English only — so
    the contract holds whether or not tomorrow's ledger adds the namespace:
    ``i18n.text`` resolves the root at every call (and keys its cache by it).
    """
    _write_language(headless_config, "ru")
    expected = _render("en", HEALTHY, value=_render("en", STATUS_NO))
    partial = tmp_path / "partial"
    (partial / "en").mkdir(parents=True)
    shutil.copy(catalogues.catalogue_root() / "en" / f"{NAMESPACE}.json", partial / "en")
    monkeypatch.setattr(catalogues, "catalogue_root", lambda: partial)

    _, out = _browser_status(monkeypatch, capsys)
    assert expected in out
    assert f"{NAMESPACE}." not in out


# --- where the move must NOT happen ----------------------------------------


def test_a_json_invocation_is_not_activated(
    headless_config: Path,
    activation_calls: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--json`` is a machine contract: the document must not follow a language.

    ``browser tabs --json`` is the cheapest real one — the inventory comes from
    the isolated config root, so the document is deterministic.
    """
    from local_operator.browser_bridge import install as browser_install
    from local_operator.browser_bridge import resources as browser_resources
    from local_operator.browser_bridge import state as browser_state

    _write_language(headless_config, "es")
    monkeypatch.setattr(browser_resources, "read_inventory", lambda root: [])
    monkeypatch.setattr(browser_state, "read", lambda: None)
    monkeypatch.setattr(browser_install, "health", lambda port: None)

    code, out = _run(monkeypatch, capsys, ["browser", "tabs", "--json"])
    assert code == 0
    assert json.loads(out)["resources"] == []
    assert activation_calls == []


def test_the_machine_readable_predicate_covers_exec_json() -> None:
    """``exec --json`` is the SECOND spelling of the exclusion, not a lookalike.

    It parses into ``json_mode`` (``--json`` with ``dest="json_mode"``), so a
    predicate that only read ``args.json`` would activate a stream that is one
    ``printable_event`` document per line. Exercised here rather than end to end
    because a real ``exec`` run reaches a provider.
    """
    assert cli_module._machine_readable(argparse.Namespace(json_mode=True)) is True
    assert cli_module._machine_readable(argparse.Namespace(json=True)) is True
    assert cli_module._machine_readable(argparse.Namespace()) is False
    assert cli_module._machine_readable(argparse.Namespace(json=False, json_mode=False)) is False


def test_a_remote_invocation_activates_exactly_once(
    headless_config: Path,
    activation_calls: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ``--remote`` dispatch keeps its own activation, and gets only one.

    ``cli.main`` returns into ``remote_command`` before its own dispatch step,
    so this path must not be activated twice — ``--detach`` without ``--remote``
    is a pure in-process refusal that reaches the copy, so the count is all this
    needs to prove.
    """
    _write_language(headless_config, "es")
    code, out = _run(monkeypatch, capsys, ["exec", "--detach"])
    assert code == 1
    assert out == ""
    assert activation_calls == ["es"]


def test_the_activation_runs_after_the_startup_migrations(
    headless_config,
    fixed_browser_status,
    monkeypatch,
    capsys,
) -> None:
    """Order pinned: migrations first, then the move.

    The move must see the config the startup repairs left behind, so moving the
    call above ``base_dir``/``run_startup_migrations`` is a regression this test
    fails on — the comment at the call site argues the position, and this is
    what enforces it.
    """
    order: list[str] = []
    monkeypatch.setattr(
        "local_operator.config_migrations.run_startup_migrations",
        lambda *args, **kwargs: order.append("migrations"),
    )
    # ``cli`` binds the helper at import, so the seam is ITS name, not boot's.
    monkeypatch.setattr(cli_module, "activate_resolved_locale", lambda: order.append("activate"))
    _browser_status(monkeypatch, capsys)
    assert order == ["migrations", "activate"]


def test_a_tui_launch_is_never_activated(headless_config, activation_calls) -> None:
    """The TUI exclusion branch, on a PARSED namespace rather than a hand-built one.

    ``OperatorApp.on_mount`` owns the move for the full-screen app, so the CLI
    must not make one on that launch; the bare launch beside it (piped, no tty)
    is a headless surface and must still get its one move.
    """
    _write_language(headless_config, "es")
    parser = cli_module.build_cli_parser()
    cli_module._install_headless_locale(parser.parse_args(["--tui"]))
    assert activation_calls == []
    cli_module._install_headless_locale(parser.parse_args([]))
    assert activation_calls == ["es"]


def test_the_boot_creates_nothing_on_disk(tmp_path, monkeypatch) -> None:
    """Resolving the language must not CREATE the store it is only reading.

    The agent-shell guard asserts a refused ``lop`` leaves ``~/.local-operator``
    absent, and resolving through a bare ``resolve_language()`` violates it:
    without a watcher it builds a ``ConfigManager``, whose construction makes the
    directory (measured). ``boot.configured_language`` reads through
    ``read_config_values``, which never writes, moves or reports anything.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)
    monkeypatch.delenv("LOP_LANG", raising=False)
    args = cli_module.build_cli_parser().parse_args([])
    cli_module._install_headless_locale(args)
    assert not (home / ".local-operator").exists()


def test_the_spawned_exec_worker_activates_before_it_runs(headless_config, monkeypatch) -> None:
    """``lop exec --background`` is its own process and had no activation at all.

    The launcher's boot activation belongs to the launcher, so a detached run
    rendered ``cli.exec_runtime`` chrome (``headless_print``'s import-time
    constants) and ``cli.exec_worker_help`` in English under an ``es`` config.
    ``run`` is stubbed: this asserts the move happens before the work starts and
    that the chrome constant re-materialised through the refresh it registered.
    """
    from local_operator import exec_worker, headless_print
    from local_operator.i18n.text import active_locale

    _write_language(headless_config, "es")
    seen: list[str] = []

    def _run(parsed, *, stop_facts=None):
        seen.append(active_locale())
        return 0

    monkeypatch.setattr(exec_worker, "run", _run)
    monkeypatch.setattr(sys, "argv", ["exec_worker", "--prompt", "x"])
    assert exec_worker.main() == 0
    assert seen == ["es"]
    assert headless_print._THINKING == _render(
        "es", "cli.exec_runtime.thinking", namespace="cli.exec_runtime"
    )


def test_the_exec_worker_leaves_a_json_stream_untranslated(headless_config, monkeypatch) -> None:
    """``--json`` workers keep the renderer on ``en``, like every other JSON path."""
    from local_operator import exec_worker
    from local_operator.i18n.text import active_locale

    _write_language(headless_config, "es")
    seen: list[str] = []

    def _run(parsed, *, stop_facts=None):
        seen.append(active_locale())
        return 0

    monkeypatch.setattr(exec_worker, "run", _run)
    monkeypatch.setattr(sys, "argv", ["exec_worker", "--prompt", "x", "--json"])
    assert exec_worker.main() == 0
    assert seen == ["en"]


@pytest.mark.parametrize(
    ("argv", "machine_readable"),
    [
        (["memory", "list", "--json"], True),
        (["search", "test", "q", "--json"], True),
        (["browser", "tabs", "--json"], True),
        (["exec", "x", "--json"], True),
        (["fetch", "test", "https://example.invalid", "--raw"], False),
        (["browser", "status"], False),
    ],
)
def test_every_declared_json_flag_reaches_the_predicate(
    argv: list[str], machine_readable: bool
) -> None:
    """Parsed through the REAL parser, so a renamed dest cannot pass silently.

    ``memory``'s ``--json`` used to land on ``raw`` and ``search test``'s on
    ``as_json``, which the predicate did not read; ``fetch test --raw`` shares the
    ``raw`` dest but prints TEXT, which is why those two flags were normalised to
    argparse's default instead of widening the predicate.
    """
    args = cli_module.build_cli_parser().parse_args(argv)
    assert cli_module._machine_readable(args) is machine_readable
