"""The ``/rules`` slash command: listing, mutations, refusals, and the flag promise.

The command is the in-app surface for ``session/rules.py`` (POC behind
``LOP_SESSION_RULES=1``): a bare ``/rules`` lists the session's standing rules,
and ``add``/``edit``/``rm`` mutate the record the system-blocks provider renders
into the prompt's volatile tail on the next call.

Every mutation here is driven through the REAL command path — the editor, Enter,
``OperatorApp._run_slash_command`` and the handler — or through the routed
``run_slash_authoritative`` seam a follower uses, never by calling
``session.rules.add_rule`` from the test. The one exception is staged evidence:
the over-cap case seeds 20 rules with the same runner the command uses, because
the interesting assertion is the 21st.

``wire.rules`` is English-only today (the POC's coverage debt is recorded in
``i18n/coverage-allowance.json``), so these assertions pin the English copy
under the ``en`` locale — the RFC §7 shape the ``wire.slash`` tests also use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.i18n import catalogues
from local_operator.paths import config_dir
from local_operator.session import rules as rules_module
from local_operator.slash_commands import (
    SLASH_COMMANDS,
    describe,
    run_rules_slash_op,
    slash_command_for,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.command_picker import command_suggestions
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _renderable_plain
from tests.unit.tui.test_slash_goal_loop_flags import _boot, _notice_texts, _submit

#: ``FakeSession.session_id`` — the directory the command writes under, through
#: the same ``config_dir()/sessions/<id>`` derivation every other reader uses.
SESSION_ID = "sess"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect the config root and pin English for every case in this module.

    A slash command derived from ``config_dir()`` must never write into the
    operator's live ``~/.local-operator`` (AGENTS.md "Isolating a run"), and the
    copy assertions below are English-pinned on purpose — the same reason the
    ``wire.slash`` tests pin ``en``.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOP_LANG", "en")
    monkeypatch.delenv("LOP_SESSION_RULES", raising=False)


def _session_dir() -> Path:
    return config_dir() / "sessions" / SESSION_ID


def _record() -> dict[str, Any]:
    return json.loads((_session_dir() / "rules.json").read_text(encoding="utf-8"))


def _last_notice(app: OperatorApp) -> str:
    return _notice_texts(app)[-1]


def test_the_registry_offers_rules_with_a_catalogued_description() -> None:
    """The palette and ``/help`` both read the registry; the copy comes from the catalogue.

    ``command_suggestions("", …)`` is the picker's own derivation (bare ``/``
    offers the whole registry in registration order), and ``_help_block`` is the
    table ``/help`` paints — so both surfaces are asserted where a user meets
    them, not on the registry object alone.
    """
    spec = slash_command_for("/rules")
    assert spec is not None, "/rules must resolve through the shared registry"
    assert spec.name == "rules"
    assert [command.name for command in SLASH_COMMANDS].count("rules") == 1

    catalogue = catalogues.load_catalogue("en", "wire.rules")
    assert spec.description == catalogue["wire.rules.command"]
    assert describe(spec, "en") == catalogue["wire.rules.command"]
    # The on-demand render (desktop catalogue, phone sheet) under another
    # locale falls back to the English source while the POC carries no
    # translation — an English sentence, not an empty string or a key.
    assert describe(spec, "fr") == catalogue["wire.rules.command"]

    assert ("rules", spec) in command_suggestions("", SLASH_COMMANDS)

    app = OperatorApp(lambda: _factory(FakeSession()))
    rows = _renderable_plain(app._help_block().renderable).splitlines()
    assert any(row.startswith("/rules") for row in rows), "no /rules row in /help"


@pytest.mark.asyncio
async def test_bare_rules_on_a_fresh_session_says_how_to_add_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOP_SESSION_RULES", "1")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/rules")
        assert _last_notice(app) == "No rules set. /rules add <text> adds one."

    assert not (_session_dir() / "rules.json").exists(), "a listing must not create the record"
    assert session.prompts == [], "a bare listing starts no turn"


@pytest.mark.asyncio
async def test_add_list_edit_and_remove_round_trip_through_the_editor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One rule, then many: every mutation through the typed command, and the record on disk.

    The disk assertion is the other half of "through the real path": a receipt
    could be composed without the write ever landing, and the prompt tail reads
    the FILE, not the receipt.
    """
    monkeypatch.setenv("LOP_SESSION_RULES", "1")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)

        await _submit(pilot, app, "/rules add never push to main")
        assert _last_notice(app) == "Rule r1 added: never push to main"

        await _submit(pilot, app, "/rules add answer in British English")
        assert _last_notice(app) == "Rule r2 added: answer in British English"

        # Many: one line each, in record order, ids and text verbatim.
        await _submit(pilot, app, "/rules")
        assert _last_notice(app) == ("r1: never push to main\nr2: answer in British English")

        await _submit(pilot, app, "/rules edit r1 never force-push main")
        assert _last_notice(app) == "Rule r1 updated: never force-push main"

        await _submit(pilot, app, "/rules rm r2")
        assert _last_notice(app) == "Rule r2 removed: answer in British English"

        # Back to one: the edit stuck and the removal took only its own rule.
        await _submit(pilot, app, "/rules")
        assert _last_notice(app) == "r1: never force-push main"

    record = _record()
    assert [row["id"] for row in record["rules"]] == ["r1"]
    assert record["rules"][0]["text"] == "never force-push main"
    # The counter keeps climbing so a removed id is never reissued.
    assert record["next"] == 3
    assert session.prompts == [], "no form of /rules starts a turn"


@pytest.mark.asyncio
async def test_a_rule_added_through_the_command_reaches_the_prompt_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command's write is the FILE the provider renders on the next call.

    The other half of this path — the sidecar reaching the outbound ``<rules>``
    section — is ``test_session_rules_live_tail``'s subject. This closes the loop
    at the command: type ``/rules add …`` and the next system blocks the model is
    given carry that rule, because the provider re-reads the file it wrote.
    """
    monkeypatch.setenv("LOP_SESSION_RULES", "1")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/rules add End every reply with ALPHA.")

    from local_operator.session.transcript import Transcript
    from tests.unit.test_session_rules_acceptance import _make_provider

    blocks = await _make_provider(Transcript(_session_dir()), _session_dir())()
    rendered = "\n".join(blocks)
    assert "<rules>" in rendered, "the flag-on tail must carry the section"
    assert "1. [r1] End every reply with ALPHA." in rendered


@pytest.mark.asyncio
async def test_the_routed_host_prints_the_same_sentences_as_the_local_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A follower's routed ``/rules`` is the owner's own sentence (the `/goal` rule).

    Both hosts run ``run_rules_slash_op``; this asserts the seam that carries it
    still answers with the exact text the local terminal prints.
    """
    monkeypatch.setenv("LOP_SESSION_RULES", "1")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/rules add never push to main")
        local_add = _last_notice(app)

        routed = await app.run_slash_authoritative("rules", "add answer in British English")
        await _submit(pilot, app, "/rules")
        local_list = _last_notice(app)

    assert routed["kind"] == "notice"
    assert routed["text"] == "Rule r2 added: answer in British English"
    assert local_add == "Rule r1 added: never push to main"
    assert local_list == "r1: never push to main\nr2: answer in British English"


@pytest.mark.asyncio
async def test_the_detached_runtime_answers_from_its_own_transcript_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A runtime-owned session answers ``/rules`` from ITS directory, not ``config_dir()``.

    This is the host a follower's attach actually lands on. The runner is
    shared, but the directory is the runtime's own (``_session_directory`` off
    the transcript), which is why ``_rules_slash`` takes no directory argument —
    and why this asserts the record is written under the transcript's directory
    and nowhere near this process's config root.
    """
    import asyncio
    from types import SimpleNamespace

    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.unit.session.runtime.test_serving import FakeSession as RuntimeSession

    monkeypatch.setenv("LOP_SESSION_RULES", "1")
    runtime_session = RuntimeSession()
    runtime_session.transcript = SimpleNamespace(directory=tmp_path / "session")
    handle = ServingSessionHandle(runtime_session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        added = await handle.run_slash_authoritative("rules", "add never push to main", None)
        listed = await handle.run_slash_authoritative("rules", "", None)
    finally:
        await handle.dispose()

    assert added["text"] == "Rule r1 added: never push to main"
    assert listed["text"] == "r1: never push to main"
    assert (tmp_path / "session" / "rules.json").is_file()
    assert not _session_dir().exists(), "the runtime wrote into this process's config root"


@pytest.mark.asyncio
async def test_every_refusal_is_a_notice_carrying_the_records_own_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad id, over-long text, a section tag, an unknown word and the cap.

    The record's ``RuleError`` message is what the user reads — quoted after the
    localized refusal sentence, never a traceback — and a refused mutation
    leaves the record exactly as it was.
    """
    monkeypatch.setenv("LOP_SESSION_RULES", "1")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/rules add keep this")
        before = (_session_dir() / "rules.json").read_bytes()

        await _submit(pilot, app, "/rules rm r9")
        assert _last_notice(app) == "Rule refused: no rule with id 'r9'"

        await _submit(pilot, app, "/rules add " + "x" * 301)
        assert _last_notice(app) == "Rule refused: rule text is 301 characters; the limit is 300"

        # A tail-section marker would let one rule pose as host context outside
        # the binding block, so the record refuses it and the user is told.
        await _submit(pilot, app, "/rules add do not mention <goal> here")
        assert _last_notice(app) == "Rule refused: rule text must not contain '<goal>'"

        await _submit(pilot, app, "/rules frobnicate")
        assert _last_notice(app) == (
            "Use /rules, /rules add <text>, /rules edit <id> <text> or /rules rm <id>."
        )

        # A newline cannot be typed through the composer (Enter submits), so the
        # multi-line refusal is driven through the routed seam, which carries
        # the raw argument — the same runner, the same sentence.
        routed = await app.run_slash_authoritative("rules", "add two\nlines")
        assert routed["text"] == "Rule refused: rule text must be a single line"

        assert (_session_dir() / "rules.json").read_bytes() == before, "a refusal wrote"

        # The cap: 20 is the record's own limit; the 21st is refused, not
        # silently dropped. Seeded with the command's own runner.
        for number in range(2, 21):
            text, _style = run_rules_slash_op(f"add rule {number}", session_dir=_session_dir())
            assert text.startswith(f"Rule r{number} added"), text
        await _submit(pilot, app, "/rules add one too many")
        assert _last_notice(app) == (
            "Rule refused: this session already has 20 rules; remove one first"
        )
        await _submit(pilot, app, "/rules")
        assert len(_last_notice(app).splitlines()) == 20

    assert session.prompts == []


@pytest.mark.asyncio
async def test_flag_off_refuses_names_the_flag_and_touches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``LOP_SESSION_RULES`` unset, ``/rules`` is a sentence and nothing else.

    The record is seeded on disk and the module's read/write entry points are
    replaced with a hard failure: the flag-off path must not read the file, not
    write it, and not start a turn. The routed seam is asserted too, because a
    follower reaches the same promise through a different host.
    """
    monkeypatch.delenv("LOP_SESSION_RULES", raising=False)
    _session_dir().mkdir(parents=True)
    record_path = _session_dir() / "rules.json"
    record_path.write_text(
        json.dumps(
            {
                "v": 1,
                "next": 2,
                "rules": [
                    {
                        "id": "r1",
                        "text": "keep this",
                        "created": "2026-11-03T00:00:00+00:00",
                        "updated": "2026-11-03T00:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    before = record_path.read_bytes()

    def _never(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the flag-off path read or wrote the rules record")

    monkeypatch.setattr(rules_module, "load_rules_with_reason", _never)
    monkeypatch.setattr(rules_module, "add_rule", _never)
    monkeypatch.setattr(rules_module, "remove_rule", _never)

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/rules")
        assert _last_notice(app).startswith("Per-session rules are off")
        assert "LOP_SESSION_RULES=1" in _last_notice(app)

        await _submit(pilot, app, "/rules add never push to main")
        assert _last_notice(app).startswith("Per-session rules are off")

        routed = await app.run_slash_authoritative("rules", "rm r1")
        assert routed["text"].startswith("Per-session rules are off")

    assert record_path.read_bytes() == before, "the flag-off path rewrote the record"
    assert sorted(path.name for path in _session_dir().iterdir()) == ["rules.json"]
    assert session.prompts == []
