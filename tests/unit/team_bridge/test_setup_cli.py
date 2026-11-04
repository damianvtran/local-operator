"""`lop team-bridge setup` is a readiness check — it can NEVER grant consent.

Operator decision, #3116 round 4: consent is granted in the Local Operator app
only (#3022's bearer boundary). An earlier revision accepted it at a terminal
behind a spoofable "human gate"; that path is removed rather than hardened, so
these cells pin three things:

* a machine with no enrolment is REFUSED, with the exact guidance copy a user
  reads — consent is in the app, not a terminal;
* nothing on any path writes consent, a designation or the enabled setting, and
  the service is only asked to place a receiver once the app's own decision is
  already committed;
* the command has no option that could carry a consent, a session or a tenant
  (the "can't happen" is enforced at the argument parser, not just in code).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.team_bridge import setup_cli
from local_operator.team_bridge.connector import install
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store

OWNER = OwnerSnapshot(
    credential_id=1,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url="https://hub.example/v1",
    enrollment_generation=1,
)


def _enrol(root: Path, *, designated: str | None = "session-42") -> None:
    store = Store(root)
    store.commit_enrollment(OWNER, OwnerReceipt("bridge_consent_v2", "bep_1"))
    if designated is not None:
        store.update_designated_session(OWNER, designated)


def _enable(root: Path, value: bool = True) -> None:
    settings_io.write_setting(ConfigManager(root), settings_io.BY_KEY["team_bridge.enabled"], value)


@pytest.fixture
def placed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every service placement request instead of touching a supervisor."""

    calls: list[str] = []

    class Outcome:
        code = "installed"
        detail = ""

    def _sync(root):  # noqa: ANN001
        calls.append(str(root))
        return Outcome()

    monkeypatch.setattr(install, "sync_connector_service", _sync)
    return calls


def test_a_machine_with_no_enrolment_is_refused_with_the_app_guidance(
    tmp_path: Path, placed: list[str]
) -> None:
    """THE cell this round exists for: no consent, no CLI substitute, no write."""

    result = setup_cli.setup(tmp_path)

    assert result == {"ok": False, "code": setup_cli.CODE_CONSENT_UI_REQUIRED}
    assert placed == []
    # Nothing was enrolled and nothing was enabled by running the check.
    assert Store(tmp_path).active_owner_readonly() is None
    assert Store(tmp_path).committed_owner_count() == 0


def test_the_refusal_copy_says_consent_is_in_the_app_and_names_the_action() -> None:
    """The user-facing sentence is part of the contract, so it is pinned verbatim."""

    from local_operator.i18n.keys import cli_team_bridge as _k

    text = str(_k.setup_consent_ui_required().envelope(locale="en")["text"])
    assert text == (
        "Team Bridge consent is granted in the Local Operator app, not from a terminal. "
        "Open the app once and enable Agent messages."
    )


def test_the_human_gate_is_gone_not_merely_disabled() -> None:
    """No literal, no flag and no function may remain that could accept consent."""

    assert not hasattr(setup_cli, "confirm_consent")
    source = Path(setup_cli.__file__).read_text(encoding="utf-8")
    for forbidden in ("accept_consent_version", "in_agent_shell", "isatty", "input("):
        assert forbidden not in source, forbidden


def test_the_command_has_no_option_that_could_carry_a_consent(tmp_path: Path) -> None:
    """Enforced at the parser: there is nothing to pass a consent, session or tenant in."""

    from local_operator.cli import build_cli_parser

    parser = build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["team-bridge", "setup", "--accept-consent-version", "v2"])
    with pytest.raises(SystemExit):
        parser.parse_args(["team-bridge", "setup", "--session", "session-42"])
    args = parser.parse_args(["team-bridge", "setup"])
    assert not hasattr(args, "accept_consent_version")
    assert not hasattr(args, "session")


def test_enrolled_but_not_enabled_asks_for_the_app_and_places_nothing(
    tmp_path: Path, placed: list[str]
) -> None:
    """An owner without the user's ON flag is "not asked yet", not "install it"."""

    _enrol(tmp_path)
    _enable(tmp_path, False)

    result = setup_cli.setup(tmp_path)

    assert result["ok"] is False
    assert result["code"] == setup_cli.CODE_ENABLE_IN_APP_REQUIRED
    assert result["account_id"] == "acc_1"
    assert placed == []


def test_a_params_carrying_outcome_renders_without_raising(
    tmp_path: Path, placed: list[str]
) -> None:
    """The render path must survive an outcome whose copy takes a parameter.

    Review MINOR-1: thirteen passing cells all rendered outcomes whose copy was
    parameterless, so nothing caught the case that actually crashed in the field.
    ``enable_in_app_required`` interpolates ``account_id``; when the parameters
    were forwarded through ``text(**kwargs)`` into ``Msg.envelope`` - which takes
    only ``code`` and ``locale`` - this raised ``TypeError`` and, because
    ``render_lines`` sits outside ``main()``'s guard, dumped a traceback AFTER
    the receiver had already been placed. The assertion is on the RENDERED TEXT,
    so a future drift back to forwarding fails here rather than in a terminal.
    """

    _enrol(tmp_path)
    _enable(tmp_path, False)

    result = setup_cli.setup(tmp_path)
    lines = setup_cli.render_lines(result)

    assert result["code"] == setup_cli.CODE_ENABLE_IN_APP_REQUIRED
    rendered = "\n".join(lines)
    assert "acc_1" in rendered, f"the parameter never reached the copy: {rendered}"
    assert "{" not in rendered, f"an unformatted placeholder survived: {rendered}"


def test_an_absent_enabled_key_is_treated_as_not_enabled(tmp_path: Path, placed: list[str]) -> None:
    _enrol(tmp_path)

    result = setup_cli.setup(tmp_path)

    assert result["code"] == setup_cli.CODE_ENABLE_IN_APP_REQUIRED
    assert placed == []


def test_enrolled_and_enabled_without_a_designation_asks_for_the_app(
    tmp_path: Path, placed: list[str]
) -> None:
    _enrol(tmp_path, designated=None)
    _enable(tmp_path)

    result = setup_cli.setup(tmp_path)

    assert result["code"] == setup_cli.CODE_DESIGNATION_MISSING
    assert placed == []


def test_a_ready_machine_places_the_receiver_through_the_one_shared_sync(
    tmp_path: Path, placed: list[str]
) -> None:
    """Placement goes through sync_connector_service — the same function, same guards."""

    _enrol(tmp_path)
    _enable(tmp_path)

    result = setup_cli.setup(tmp_path)

    assert result["ok"] is True
    assert result["code"] == setup_cli.CODE_READY
    assert result["service"] == "installed"
    assert result["session"] == "session-42"
    assert placed == [str(tmp_path)]


def test_a_refused_placement_is_reported_rather_than_called_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enrol(tmp_path)
    _enable(tmp_path)

    class Outcome:
        code = "source_checkout"
        detail = "runs from a checkout"

    monkeypatch.setattr(install, "sync_connector_service", lambda root: Outcome())

    result = setup_cli.setup(tmp_path)

    assert result["ok"] is False
    assert result["code"] == setup_cli.CODE_NOT_READY
    assert result["service"] == "source_checkout"


def test_a_written_but_unaddressed_unit_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``not_addressable`` means the service manager was never reached (round 5)."""

    _enrol(tmp_path)
    _enable(tmp_path)

    class Outcome:
        code = "not_addressable"
        detail = "unit written; systemd not addressed"

    monkeypatch.setattr(install, "sync_connector_service", lambda root: Outcome())

    result = setup_cli.setup(tmp_path)

    assert result["ok"] is False
    assert result["code"] == setup_cli.CODE_NOT_READY
    assert result["service"] == "not_addressable"


def test_a_corrupt_store_is_a_refusal_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable state database must report, not explode (round 5, MAJOR-2)."""

    from local_operator.team_bridge.store import state_db_path

    path = state_db_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a sqlite database")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)

    assert setup_cli.main(_args(json=True)) == 1
    assert '"setup_failed"' in capsys.readouterr().out


def test_plain_english_is_the_default_output_and_json_is_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal a person sees is the guidance copy, not a JSON blob."""

    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)

    assert setup_cli.main(_args(json=False)) == 1
    out = capsys.readouterr().out
    assert "granted in the Local Operator app, not from a terminal" in out
    assert "{" not in out

    assert setup_cli.main(_args(json=True)) == 1
    out = capsys.readouterr().out
    assert '"consent_ui_required"' in out


def _args(*, json: bool):
    import argparse

    return argparse.Namespace(json=json)


def test_the_guide_is_the_only_place_that_could_promise_a_terminal_consent() -> None:
    """A stale sentence in the guide would undo this round in the user's hands."""

    from pathlib import Path as _Path

    guide = _Path(__file__).resolve().parents[3] / "local_operator" / "guides" / "team-bridge"
    text = (guide / "GUIDE.md").read_text(encoding="utf-8")
    lowered = text.lower()
    for forbidden in ("speed bump", "human gate", "--accept-consent-version"):
        assert forbidden not in lowered, forbidden
    assert "not from a terminal" in lowered
