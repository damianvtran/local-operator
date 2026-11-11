"""``/move --to`` refusals render through ``tui.session_move`` in the active locale.

The grammar and the subprocess wrapper used to hand the composer English
sentences (and ``app.py`` forwarded ``request.error`` straight to the notice
line). The French rows below are a synthetic fixture (see ``synthetic_locale``):
the audited translations land in a later wave. Flag names and commands inside
the sentences are typed tokens and must survive in every language.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from local_operator.i18n import catalogues, text
from local_operator.tui import session_move
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_move import MoveTo, ambiguous_move, parse_move_to
from tests.unit.i18n import synthetic_locale

NAMESPACE = "tui.session_move"

_FR = {
    "tui.session_move.ambiguous": (
        "Utilisez /move <chemin> pour un dossier, ou /move <session> --to <pair> "
        "pour déplacer une session."
    ),
    "tui.session_move.to_needs_device": "--to attend un appareil : /move --to <pair|local>",
    "tui.session_move.cannot_wait": (
        "cette vue ne peut pas attendre — depuis un shell : "
        "`lop sessions move <id> --to <to> --wait <seconds>`"
    ),
    "tui.session_move.unsupported_flag": (
        "/move --to n'accepte que --keep, --queue et --drop-session-roles, pas {flag}"
    ),
    "tui.session_move.cli_unavailable": "impossible de lancer la CLI ({reason})",
    "tui.session_move.deadline_exceeded": (
        "le déplacement n'a pas fini à temps ; `lop sessions move {session} --to {to}` "
        "indique où il en est"
    ),
    "tui.session_move.cli_no_answer": "la CLI s'est arrêtée ({code}) sans réponse",
}

_ENGLISH_FRAGMENTS = (
    "Use /move",
    "needs a device",
    "cannot wait",
    "takes only",
    "could not start",
    "did not finish in time",
    "with no answer",
)


@pytest.fixture
def french(monkeypatch: pytest.MonkeyPatch) -> None:
    synthetic_locale.install(monkeypatch, "fr", NAMESPACE, _FR)
    # The autouse ``restore_active_locale`` fixture puts English back afterwards.
    text.set_active_locale("fr")
    assert text.active_locale() == "fr"


def _assert_no_english(value: str) -> None:
    for fragment in _ENGLISH_FRAGMENTS:
        assert fragment not in value, fragment


def test_english_source_keeps_the_pre_extraction_bytes() -> None:
    """Control: English is byte-identical to the literals these keys replaced."""
    assert ambiguous_move() == (
        "Use /move <path> for a working directory, "
        "or /move <session> --to <peer> to move a session."
    )
    assert parse_move_to("--to").error == "--to needs a device: /move --to <peer|local>"
    assert parse_move_to("--to x --wait 30").error == (
        "this view cannot wait — from a shell: "
        "`lop sessions move <id> --to <to> --wait <seconds>`"
    )
    assert parse_move_to("--to x --force").error == (
        "/move --to takes only --keep, --queue and --drop-session-roles, not '--force'"
    )


@pytest.mark.parametrize(
    "arg, key, params",
    [
        ("--to", "to_needs_device", {}),
        ("--to=", "to_needs_device", {}),
        ("--to --keep", "to_needs_device", {}),
        ("a b --to local", "ambiguous", {}),
        ("~/src --to pixel-8", "ambiguous", {}),
        ("./x --to local", "ambiguous", {}),
        ("--to x --wait 30", "cannot_wait", {}),
        ("--wait=30 --to x", "cannot_wait", {}),
        ("--to x --force", "unsupported_flag", {"flag": "'--force'"}),
    ],
)
def test_grammar_refusals_render_the_catalogue_in_the_active_locale(
    french: None, arg: str, key: str, params: dict[str, str]
) -> None:
    refused = parse_move_to(arg)
    assert refused is not None
    expected = _FR[f"tui.session_move.{key}"].format(**params)
    assert refused.error == expected
    _assert_no_english(refused.error)
    # Typed tokens stay English inside the translated sentence.
    assert "--to" in refused.error or "--wait" in refused.error
    # An accepted request carries no error and no copy at all.
    assert parse_move_to("--to pixel-8") == MoveTo(to="pixel-8")


class _FakePopen:
    """A child that never starts, times out, or exits silently (per ``mode``)."""

    mode = "oserror"

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        if self.mode == "oserror":
            raise OSError("Exec format error")
        self.returncode = 3
        self._calls = 0

    def communicate(self, timeout=None):  # noqa: ANN001
        self._calls += 1
        if self.mode == "timeout" and self._calls == 1:
            raise subprocess.TimeoutExpired("lop", timeout or 0)
        return "", ""


@pytest.mark.parametrize(
    "mode, code, key, params, changed",
    [
        ("oserror", "cli_unavailable", "cli_unavailable", {"reason": "Exec format error"}, False),
        (
            "timeout",
            "deadline_exceeded",
            "deadline_exceeded",
            {"session": "9f3ac1e0b7d2", "to": "pixel-8"},
            True,
        ),
        ("silent", "cli_failed", "cli_no_answer", {"code": "3"}, False),
    ],
)
def test_run_session_move_refusals_render_in_the_active_locale(
    french: None,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    code: str,
    key: str,
    params: dict[str, str],
    changed: bool,
) -> None:
    _FakePopen.mode = mode
    monkeypatch.setattr(session_move.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(session_move, "_reap_group", lambda child: None)
    result = session_move.run_session_move("9f3ac1e0b7d2", "pixel-8")
    # The machine category is the stable ``code`` field: only ``message`` moves.
    assert result["ok"] is False
    assert result["code"] == code
    assert result["changed"] is changed
    assert result["message"] == _FR[f"tui.session_move.{key}"].format(**params)
    _assert_no_english(result["message"])


def test_the_app_forwards_the_localized_request_error_to_the_notice_line(french: None) -> None:
    """``_cmd_move_session`` paints ``request.error`` verbatim: it is already keyed."""
    shown: list[tuple[str, str]] = []
    fake = SimpleNamespace(
        _session=SimpleNamespace(session_id="abc"),
        _system_notice=lambda body, kind: shown.append((body, kind)),
    )
    request = parse_move_to("--to x --wait 30")
    assert request is not None
    OperatorApp._cmd_move_session(fake, request, lambda *a, **k: None)  # type: ignore[arg-type]
    assert shown == [(_FR["tui.session_move.cannot_wait"], "error")]
    _assert_no_english(shown[0][0])


def test_every_message_is_catalogued_in_english() -> None:
    source = catalogues.load_catalogue("en", NAMESPACE)
    assert set(source) == set(_FR), "fixture rows must mirror the English keys exactly"
