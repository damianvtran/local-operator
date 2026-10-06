"""The pilot verbs: ``lop network sessions --send/--steer/--slash``.

WHAT THESE PIN, and what they deliberately leave to other evidence. The MESH half
— a viewer that reaches a conversation on another device, and the acts landing on
THAT device's runtime — is proven over real relays in
``tests/unit/network/test_remote_viewer.py``, and for this verb against the real
paired peer in the PR's own evidence. These tests are the verb's half: which act
each flag is, the text's two routes, every refusal and the code it carries, that
the OWNER's words are the receipt rather than a sentence invented on this side,
and that a viewer this path opened is always given back.

The viewer is faked rather than dialled because every one of these is a decision
made BEFORE the frame is written, and a fake is what lets a refusal be asserted
without a peer to refuse. The end-to-end claim is not made here and is not made
by any test in this file.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network.types import MeshRefusal
from local_operator.resume import SessionRow
from local_operator.session.errors import RuntimeRetiring

ok = json.loads


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers)
    return parser


PEER = "cloud-node-1"
SESSION = "9e7d35e4e41f"


class _FakeViewer:
    """The slice of ``AttachedSession`` the pilot verbs touch, and nothing else.

    Deliberately narrow: a fake with more surface than the code under test would
    keep passing after the code stopped using it, which is how a green suite
    stops being evidence (the file's own warning, one level up).
    """

    def __init__(
        self,
        *,
        streaming: bool = False,
        steer_receipt: str = "steering queued",
        send_error: Exception | None = None,
        send_sleep_s: float = 0.0,
        slash_receipt: Any = None,
        rows: list[Any] | None = None,
    ) -> None:
        self.is_streaming = streaming
        self.steer_receipt = steer_receipt
        self.send_error = send_error
        self.send_sleep_s = send_sleep_s
        self.slash_receipt = (
            slash_receipt
            if slash_receipt is not None
            else {
                "kind": "notice",
                "text": "renamed to a better name",
                "style": "info",
                "data": {},
            }
        )
        self._rows = rows or []
        self.bound = False
        self.disposed = False
        self.prompts: list[str] = []
        self.steers: list[str] = []
        self.slashes: list[tuple[str, str]] = []
        self.waited: list[str] = []

    async def bind_runtime(self) -> None:
        self.bound = True

    async def dispose(self) -> None:
        self.disposed = True

    async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> str:
        # THE ROUTE THE CODE MUST TAKE FOR A STEER: `prompt` is what routes to the
        # owner's steer op when the session is streaming (``send_command(...,
        # streaming=self._streaming)``), and it returns the owner's receipt.
        self.prompts.append(text)
        return self.steer_receipt

    async def prompt_and_wait(self, text: str, images: Any = None, **kwargs: Any) -> None:
        self.waited.append(text)
        if self.send_sleep_s:
            await asyncio.sleep(self.send_sleep_s)
        if self.send_error is not None:
            raise self.send_error

    async def route_shared_slash(self, command: str, args: str, images: Any = None) -> Any:
        self.slashes.append((command, args))
        return self.slash_receipt

    def display_history_window(self) -> list[Any]:
        return self._rows


class _Message:
    """The two fields ``_pilot_last_reply`` reads off a display row."""

    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text


class _Opened:
    def __init__(self) -> None:
        self.calls = 0
        #: What the verb told the SEAM, per call. The viewer's declaration of the
        #: action receipts it consumes is made here and nowhere else, so asserting
        #: it needs the kwargs rather than the viewer's behaviour.
        self.viewer_kwargs: list[dict[str, Any]] = []

    def __call__(self, viewer: _FakeViewer | None, row: Any = None):  # type: ignore[no-untyped-def]
        async def _open(session_id: str, **kwargs: Any) -> Any:
            self.calls += 1
            self.viewer_kwargs.append(dict(kwargs))
            return viewer

        return _open


def _remote_row(
    *,
    session_id: str = SESSION,
    name: str = "pilot",
    live_state: str = "busy",
    reachable: bool = True,
    reason: str = "",
    device_id: str = "d_428cb39f92ebbd094acf5d30a2db7bfb",
    device_name: str = PEER,
) -> SessionRow:
    return SessionRow(
        id=session_id,
        mtime=1.0,
        name=name,
        live_state=live_state,
        locality="remote",
        owner_device=device_id,
        owner_device_name=device_name,
        reachable=reachable,
        unreachable_reason=reason,
    )


def _patch_peer_rows(monkeypatch: pytest.MonkeyPatch, rows: list[SessionRow]) -> None:
    """Stand in for the name tier's catalogue read, as ``_patch_unresolved`` does
    for the failure path's: the rows the named peer reported, no relay dialled."""

    import local_operator.session.peer_rows as peer_rows_mod

    monkeypatch.setattr(peer_rows_mod, "peer_session_rows", lambda _root=None: tuple(rows))


def _patch(monkeypatch: pytest.MonkeyPatch, viewer: _FakeViewer | None, row: Any = None) -> _Opened:
    """Point the seam the verb uses at ``viewer``/``row``.

    Patched on ``session.remote_open`` itself because that is where the verb
    imports them from at call time — the same seam ``test_remote_open.py`` drives
    the app through (`remote_open.open_remote_viewer`), so a change of entry point
    breaks this test rather than silently moving under it.

    The name tier's read is stubbed to NOTHING here as well: every cell that
    wants name resolution overrides it via :func:`_patch_peer_rows`, and a cell
    that does not must never reach a live relay through this seam.
    """
    import local_operator.session.remote_open as remote_open

    opened = _Opened()
    monkeypatch.setattr(remote_open, "open_remote_viewer", opened(viewer, row))
    monkeypatch.setattr(remote_open, "remote_row_for", lambda _session_id, _root: row)
    _patch_peer_rows(monkeypatch, [])
    return opened


def _run(argv: list[str]) -> int:
    return int(net_cli.main(_parser().parse_args(["network", *argv])))


def _real_parse(argv: list[str]) -> argparse.Namespace:
    """Parse through the REAL top-level parser (``cli.build_cli_parser``).

    The payload cells need the entry point a person runs: ``--force`` and ``--json``
    are options the PARENT parser owns alongside this subcommand's own, and the
    local ``_parser`` above declares neither — a payload question asked through it
    would be easier than the one round 1 reproduced.
    """
    from local_operator.cli import build_cli_parser

    return build_cli_parser().parse_args(["network", *argv])


def _run_real(argv: list[str]) -> int:
    """``_run`` through that same parser, so a cell exercises one path end to end."""
    return int(net_cli.main(_real_parse(argv)))


# ---------------------------------------------------------------------------
# the surface
# ---------------------------------------------------------------------------


def test_the_pilot_flags_take_a_session_and_the_text_as_a_positional() -> None:
    """The parser half, asserted so the flags cannot be dropped while the help text
    keeps promising them (the same reason `--stop --force` is pinned in test_cli).

    THE TEXT IS A POSITIONAL AND NOT THE FLAG'S VALUE, deliberately: ``--prompt``
    already means "--create's first turn", and a second meaning for one flag is
    what makes `--prompt X` do two things depending on a token three words left.
    """
    parsed = _parser().parse_args(
        ["network", "sessions", "--peer", PEER, "--send", SESSION, "hello", "there"]
    )
    assert parsed.send == SESSION
    assert parsed.text == ["hello", "there"]
    for name in ("steer", "slash"):
        assert getattr(parsed, name) == ""


def test_two_pilot_acts_in_one_command_are_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["sessions", "--peer", PEER, "--send", SESSION, "--steer", SESSION, "hi"]) == 2
    assert "name one of --send/--steer/--slash" in capsys.readouterr().err


def test_a_pilot_act_mixed_with_create_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    """A create would mint a NEW session and drop the turn the caller meant to send."""
    code = _run(["sessions", "--peer", PEER, "--create", "--name", "x", "--send", SESSION, "hi"])
    assert code == 2
    assert "--send acts on the session you name; --create" in capsys.readouterr().err


def test_without_a_peer_the_verb_says_which_flag_it_needs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Refused by NAME and with the machine code, as a real refusal should be.

    Asserted through the surface a caller actually reads (``main`` renders a
    ``MeshRefusal`` as rc 1 plus a ``{"code", "message"}`` body under ``--json``)
    rather than by raising it out of the handler: the rendering IS the contract —
    a code a script branches on and a sentence a person can act on.
    """
    assert _run(["sessions", "--json", "--send", SESSION, "hi"]) == 1
    body = capsys.readouterr()
    assert '{"ok": false, "code": "peer_required"' in body.out
    assert "--send needs --peer" in body.out
    assert "--send needs --peer" in body.err


def test_without_text_the_verb_refuses_before_it_dials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened = _patch(monkeypatch, _FakeViewer(), _remote_row())
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    assert _run(["sessions", "--peer", PEER, "--send", SESSION]) == 2
    assert "needs some text" in capsys.readouterr().err
    assert opened.calls == 0, "a verb refused for its own arguments must not open a viewer"


def test_a_pipe_carries_the_text_when_the_positional_is_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`lop send`'s own rule: a body may arrive on stdin, which is how a caller puts
    a newline in a prompt without inventing a quoting scheme."""
    viewer = _FakeViewer(rows=[_Message("assistant", "done")])
    _patch(monkeypatch, viewer, _remote_row())
    piped = io.StringIO("line one\nline two\n")
    piped.isatty = lambda: False  # type: ignore[method-assign]
    monkeypatch.setattr("sys.stdin", piped)
    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION]) == 0
    capsys.readouterr()
    assert viewer.waited == ["line one\nline two"]


# ---------------------------------------------------------------------------
# the refusals that happen before any frame
# ---------------------------------------------------------------------------


def _patch_unresolved(
    monkeypatch: pytest.MonkeyPatch,
    *,
    answer: dict[str, Any] | None = None,
    refusal: Exception | None = None,
) -> None:
    """Stand in for the catalogue read the unresolved path makes."""

    def _answer(op: str, **kwargs: Any) -> dict[str, Any]:
        assert op == "peer_session_rows", op
        if refusal is not None:
            raise refusal
        return answer or {}

    monkeypatch.setattr(net_cli, "_relay_answer", _answer)
    # The NAME tier reads the catalogue itself (``_resolve_act_target``), so a
    # cell that wants an unresolved target must stub that read too — otherwise
    # this helper would stand in for one read while a second one dialled a real
    # relay. Empty rows are the unresolved state; name cells override this.
    _patch_peer_rows(monkeypatch, [])


#: The three states one ``None`` from the resolver can mean, and the sentence each
#: one owes. A single "not a session on another device" for all three told a user
#: with a stopped peer that their conversation did not exist.
_UNRESOLVED_CASES: tuple[tuple[str, dict[str, Any] | None, Exception | None, str, str], ...] = (
    (
        "no relay here",
        None,
        MeshRefusal(
            "relay_unavailable", "the relay is not running; start it with `lop network start`"
        ),
        "relay_unavailable",
        "lop network start",
    ),
    (
        "the peer did not answer",
        {"peers": {PEER: {"name": PEER, "reachable": False, "reason": "no-peer-link"}}},
        None,
        "peer_unreachable",
        "--peer",
    ),
    (
        "the peer answered and holds no such id",
        {"peers": {PEER: {"name": PEER, "reachable": True, "reason": ""}}},
        None,
        "session_unknown",
        f"--peer {PEER}",
    ),
)


@pytest.mark.parametrize(
    ("what", "answer", "refusal", "code", "named"),
    _UNRESOLVED_CASES,
    ids=[case[0] for case in _UNRESOLVED_CASES],
)
def test_an_id_that_resolves_to_nothing_says_which_of_the_three_it_is(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    what: str,
    answer: dict[str, Any] | None,
    refusal: Exception | None,
    code: str,
    named: str,
) -> None:
    _patch(monkeypatch, _FakeViewer(), None)
    _patch_unresolved(monkeypatch, answer=answer, refusal=refusal)

    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hi"]) == 1
    printed = capsys.readouterr().out
    assert f'"code": "{code}"' in printed, printed
    assert named in printed, printed
    if code == "relay_unavailable":
        # The relay's own sentence is about the RELAY and is reused verbatim
        # across this family; naming a session would be this verb's invention.
        return
    assert SESSION in printed, printed
    assert "doctor" in printed or "lists" in printed, printed


class _BlindViewer(_FakeViewer):
    """A viewer whose DIAL fails — at the rung the caller names.

    ``open`` is the cold-to-working seam (the viewer is cold until
    ``bind_runtime``), so a dial that never produced a session can fail at either
    rung depending on whether a row was cached; both are the same situation to the
    person reading it, and both must answer the same way.
    """

    def __init__(self, rung: str) -> None:
        super().__init__()
        self.rung = rung

    async def bind_runtime(self) -> None:
        if self.rung == "bind":
            raise ConnectionError("the remote owner did not send its state")
        self.bound = True


async def _blind_open(_session_id: str, **_kwargs: Any) -> Any:
    raise ConnectionError("the remote owner did not send its state")


@pytest.mark.parametrize("rung", ["open", "bind"])
@pytest.mark.parametrize("relay_up", [True, False], ids=["relay-up", "no-relay-here"])
def test_a_dial_that_produced_no_session_names_the_far_end(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rung: str,
    relay_up: bool,
) -> None:
    """The state CI caught answering two ways, pinned at BOTH rungs.

    One stopped peer reported ``session_unreachable`` for a cached row (the dial
    failed inside the viewer) and ``peer_unreachable`` for an uncached one (it was
    refused before the dial), because the first version classified by asking the
    peer's catalogue a second time — a read that is not usable at that moment: a
    link this device's relay still believes in reports the peer as reachable for
    seconds after it died. Both rungs answer ONE thing now: a dial that produced no
    stream is the DEVICE, whether it died at the open or at the bind, which is what
    the reachability read already says.

    Round 1's MINOR-4 offered the other settlement — ``session_unreachable`` at the
    bind rung — and CI refused it: with the read still answering
    ``peer_unreachable``, one stopped peer answered two codes again depending on
    which rung was unlucky. So the GUIDE's row moved instead, to what that code
    really covers: a bind that ran out of its BUDGET (the timeout rung below), and
    the act as a whole running out of ``PILOT_ACT_TIMEOUT_S``.
    """
    import local_operator.network.store as store_mod
    import local_operator.session.remote_open as remote_open

    _patch(monkeypatch, _BlindViewer(rung), _remote_row())
    if rung == "open":
        monkeypatch.setattr(remote_open, "open_remote_viewer", _blind_open)
    monkeypatch.setattr(
        store_mod, "find_own_relay", lambda root=None: object() if relay_up else None
    )

    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hi"]) == 1
    printed = capsys.readouterr().out
    if relay_up:
        assert '"code": "peer_unreachable"' in printed, printed
        assert "doctor" in printed, printed
        assert PEER in printed, printed
    else:
        # The local relay is the component, and the family's own sentence names
        # the remedy rather than this verb inventing one.
        assert '"code": "relay_unavailable"' in printed, printed
        assert "lop network start" in printed, printed


def test_an_unreachable_peer_gets_the_one_sentence_and_no_dial(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened = _patch(monkeypatch, _FakeViewer(), _remote_row(reachable=False, reason="timeout"))
    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hi"]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["code"] == "peer_unreachable"
    assert payload["ok"] is False
    # The device's name, the reason in words and the diagnosing command — composed
    # by `remote_open.unreachable_peer_sentence`, so both surfaces say one thing.
    assert PEER in payload["message"]
    assert "network doctor" in payload["message"]
    assert opened.calls == 0


# ---------------------------------------------------------------------------
# --send
# ---------------------------------------------------------------------------


def test_send_waits_for_the_owners_outcome_and_prints_its_reply(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    viewer = _FakeViewer(rows=[_Message("user", "hello"), _Message("assistant", "the reply")])
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--peer", PEER, "--send", SESSION, "hello"]) == 0
    out = capsys.readouterr().out
    assert "the turn finished" in out
    assert "the reply" in out
    # THE WAIT IS THE WHOLE POINT: `prompt` admits, `prompt_and_wait` completes.
    assert viewer.waited == ["hello"]
    assert viewer.disposed


def test_send_reports_a_failed_turn_as_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The owner's own error is the receipt, and the exit code does not launder it."""
    viewer = _FakeViewer(send_error=RuntimeError("No API key configured for that provider"))
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hi"]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["code"] == "turn_failed"
    assert payload["outcome"] == "failed"
    assert "No API key configured" in payload["error"]


def test_send_reports_a_retiring_owner_as_queued_and_not_as_ran(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A draining runtime spools the message for its successor: the message is safe,
    the turn did not run here, and the exit code says which of those happened."""
    viewer = _FakeViewer(send_error=RuntimeRetiring(leaving="a build handover", queued=True))
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hi"]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["outcome"] == "queued"
    assert payload["ok"] is False


def test_send_reports_a_turn_that_outlived_the_bound_as_still_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADMISSION IS NOT COMPLETION: at the bound the verb says the turn is still
    running THERE and names the session, rather than claiming an outcome nobody saw."""
    viewer = _FakeViewer(send_sleep_s=30.0)
    _patch(monkeypatch, viewer, _remote_row())
    monkeypatch.setattr(net_cli, "PILOT_TURN_TIMEOUT_S", 0.05)
    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hi"]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["outcome"] == "running"
    assert payload["code"] == "turn_running"
    assert viewer.disposed, "the boundary is a report, not a leak"


# ---------------------------------------------------------------------------
# --steer
# ---------------------------------------------------------------------------


def test_steer_refuses_a_session_with_no_turn_to_correct(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE DIFFERENCE FROM --send, and the reason this flag reads the streaming edge
    first: `AttachedSession.prompt` routes an idle session to a PROMPT, so a steer
    that skipped this check would quietly start a turn the caller did not ask for."""
    viewer = _FakeViewer(streaming=False)
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--json", "--peer", PEER, "--steer", SESSION, "wait, not that"]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["code"] == "turn_not_running"
    assert payload["ok"] is False
    assert viewer.prompts == [], "nothing may be written when there is no turn to steer"
    assert viewer.disposed


def test_steer_delivers_the_owners_queue_receipt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    viewer = _FakeViewer(streaming=True, steer_receipt="steering queued")
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--peer", PEER, "--steer", SESSION, "focus on the tests"]) == 0
    out = capsys.readouterr().out
    assert "steering queued" in out
    assert viewer.prompts == ["focus on the tests"]


# ---------------------------------------------------------------------------
# --slash
# ---------------------------------------------------------------------------


def test_slash_splits_the_command_from_its_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    viewer = _FakeViewer()
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--peer", PEER, "--slash", SESSION, "/rename a better name"]) == 0
    assert viewer.slashes == [("rename", "a better name")]
    assert "renamed to a better name" in capsys.readouterr().out


def test_slash_works_without_the_leading_slash_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    viewer = _FakeViewer()
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--peer", PEER, "--slash", SESSION, "model"]) == 0
    assert viewer.slashes == [("model", "")]
    capsys.readouterr()


def test_slash_exit_code_comes_from_the_owners_own_receipt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The OWNER ran the command and it failed: this side must not guess, and the
    receipt's own ``style`` is what decides."""
    viewer = _FakeViewer(
        slash_receipt={"kind": "notice", "text": "no such command", "style": "error"}
    )
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--json", "--peer", PEER, "--slash", SESSION, "/nope"]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["outcome"] == "refused"
    assert payload["text"] == "no such command"


def test_slash_refuses_an_empty_command_before_dialling(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened = _patch(monkeypatch, _FakeViewer(), _remote_row())
    assert _run(["sessions", "--peer", PEER, "--slash", SESSION, "/"]) == 2
    assert "needs a command" in capsys.readouterr().err
    assert opened.calls == 0


# ---------------------------------------------------------------------------
# the shape a script reads
# ---------------------------------------------------------------------------


def test_json_prints_the_payload_and_the_human_run_prints_sentences(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One or the other, never both — the family's contract, because a payload in a
    pipe that asked for lines and prose in a pipe that parses JSON are both wrong."""
    viewer = _FakeViewer(streaming=True, steer_receipt="steering queued")
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--json", "--peer", PEER, "--steer", SESSION, "go"]) == 0
    printed = capsys.readouterr().out
    payload = ok(printed)
    assert payload["ok"] is True
    assert payload["verb"] == "steer"
    assert payload["session_id"] == SESSION
    assert payload["peer"] == PEER
    assert payload["receipt"] == "steering queued"


# ---------------------------------------------------------------------------
# the payload is the user's, verbatim (round 1, MAJOR-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "words",
    [
        ["check", "the", "--name", "field"],
        ["use", "--force"],
        ["pass", "--yes", "to", "it"],
        ["explain", "the", "--peer", "flag"],
        ["a", "--", "b"],
    ],
    ids=["name", "force", "yes", "peer", "dash-in-prose"],
)
def test_a_flag_shaped_prompt_is_delivered_whole(
    monkeypatch: pytest.MonkeyPatch, words: list[str]
) -> None:
    """The defect that made a prompt undeliverable: with ``nargs="*"`` every word
    matching a declared option was consumed as that option, wherever it appeared,
    so this verb ran a turn on ``check the`` and exited 0 — and a prompt containing
    ``--force`` (a flag of ``--stop``) never became a turn at all, just a usage
    error about the wrong flag. The payload is a REMAINDER now, so the owner is
    asked the user's own words and the command still says what it did."""
    viewer = _FakeViewer()
    _patch(monkeypatch, viewer, _remote_row())
    assert _run_real(["sessions", "--peer", PEER, "--send", SESSION, *words]) == 0
    assert viewer.waited == [" ".join(words)], viewer.waited


def test_a_leading_separator_is_the_separator_and_not_a_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--`` is documented as the explicit separator, and argparse does NOT
    consume it under a REMAINDER — measured through this parser: the token stays
    in the list. So the spelling every other CLI taught the user is taken back off
    rather than delivered as the prompt's first word."""
    viewer = _FakeViewer()
    _patch(monkeypatch, viewer, _remote_row())
    assert (
        _run_real(
            [
                "sessions",
                "--peer",
                PEER,
                "--send",
                SESSION,
                "--",
                "explain",
                "the",
                "--peer",
                "flag",
            ]
        )
        == 0
    )
    assert viewer.waited == ["explain the --peer flag"], viewer.waited


def test_a_second_separator_keeps_a_dash_leading_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One token, so a payload that genuinely begins with a dash keeps an escape
    — and a ``--`` in the MIDDLE of the text is a word the user typed, because
    nothing between the session id and the end of the line is reinterpreted."""
    viewer = _FakeViewer()
    _patch(monkeypatch, viewer, _remote_row())
    assert (
        _run_real(["sessions", "--peer", PEER, "--send", SESSION, "--", "--", "dash", "prose"]) == 0
    )
    assert viewer.waited == ["-- dash prose"], viewer.waited


def test_a_separator_with_nothing_after_it_is_the_empty_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half of taking the text as-is: the separator alone is NOT a
    prompt whose text is ``--``. An empty payload is refused before dialling,
    which is the rule the sibling cell below pins for the plain empty case too."""
    opened = _patch(monkeypatch, _FakeViewer(), _remote_row())
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert _run_real(["sessions", "--peer", PEER, "--send", SESSION, "--"]) == 2
    assert "needs some text" in capsys.readouterr().err
    assert opened.calls == 0


def test_this_commands_own_flags_come_before_the_session_id() -> None:
    """The COST of "the payload is what you typed", pinned so it is a documented
    trade rather than a surprise: a flag written after the session id is payload,
    and one written before it is a flag. Both halves are asserted because only the
    pair makes the rule checkable."""
    first = _real_parse(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hello", "world"])
    assert first.json is True
    assert first.text == ["hello", "world"]

    last = _real_parse(["sessions", "--peer", PEER, "--send", SESSION, "hello", "--json"])
    assert last.json is False
    assert last.text == ["hello", "--json"]


# ---------------------------------------------------------------------------
# the receipt names the DEVICE, not the string typed (QA round 1, Q1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "act",
    [
        ["--send", SESSION, "hello"],
        ["--steer", SESSION, "hello"],
        ["--slash", SESSION, "/rename x"],
    ],
    ids=["send", "steer", "slash"],
)
def test_a_success_receipt_names_the_device_the_work_happened_on(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], act: list[str]
) -> None:
    """QA round 1, Q1: `--peer no-such-device --send <id> hello` answered rc 0 with
    `peer: no-such-device` while the turn ran on the real peer, because the receipt
    echoed the string the user typed and only a FAILURE path ever resolved a name.

    `--peer` is not what routes the act — the session id is — so the receipt names
    the device the session's own row says holds it, which is a fact this act already
    read and which therefore costs no second fan-out. The string the user typed is
    kept beside it when the two disagree, so a wrong `--peer` is visible instead of
    silently believed.
    """
    viewer = _FakeViewer(
        streaming=True, slash_receipt={"kind": "notice", "text": "renamed", "style": "info"}
    )
    _patch(monkeypatch, viewer, _remote_row())

    assert _run(["sessions", "--json", "--peer", "no-such-device", *act]) == 0
    payload = ok(capsys.readouterr().out)
    assert payload["peer"] == PEER, payload
    assert payload["peer_named"] == "no-such-device", payload


def test_a_matching_peer_name_carries_no_mismatch_note(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ordinary case stays ordinary: the name the user typed IS the device, so
    nothing is repeated back at them."""
    viewer = _FakeViewer(rows=[_Message("assistant", "the reply")])
    _patch(monkeypatch, viewer, _remote_row())

    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hi"]) == 0
    payload = ok(capsys.readouterr().out)
    assert payload["peer"] == PEER
    assert "peer_named" not in payload


def test_the_device_id_is_an_accepted_way_to_name_the_peer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same exact-match rule ``_pilot_peer_block`` already uses (name OR device
    id), so naming the peer by its id does not read as a mismatch — and the receipt
    still answers with the canonical NAME, which is what a person reads."""
    row = _remote_row()
    viewer = _FakeViewer(rows=[_Message("assistant", "the reply")])
    _patch(monkeypatch, viewer, row)

    assert _run(["sessions", "--json", "--peer", row.owner_device, "--send", SESSION, "hi"]) == 0
    payload = ok(capsys.readouterr().out)
    assert payload["peer"] == PEER
    assert "peer_named" not in payload


def test_the_human_run_names_the_device_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The lines are a receipt as well, so the same fact is in them — and the wrong
    name is called out rather than dropped."""
    viewer = _FakeViewer(rows=[_Message("assistant", "the reply")])
    _patch(monkeypatch, viewer, _remote_row())

    assert _run(["sessions", "--peer", "no-such-device", "--send", SESSION, "hi"]) == 0
    printed = capsys.readouterr().out
    assert PEER in printed, printed
    assert "no-such-device" in printed, printed


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--name", "a-title"),
        ("--cwd", "/tmp"),
        ("--prompt", "first turn"),
        ("--team", "a-team"),
        ("--profile", "a-role"),
        ("--effort", "high"),
    ],
    ids=["name", "cwd", "prompt", "team", "profile", "effort"],
)
def test_a_create_only_flag_with_an_act_is_refused_not_dropped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
) -> None:
    """Round 2, MINOR-1, the half a refusal can fix: the flag was accepted, then
    DROPPED, and the words it ate were the user's own — `--send <s> --model X is the
    field` came back as "is the field" with rc 0.

    A payload cannot be told from a flag (that is the parser's boundary), so the
    flags that can never mean anything with an act are refused, and the sentence
    names `--` as the way to send those words as text. Same class as ``--force`` and
    ``--create`` beside an act: accepted-and-dropped is an untruth.

    These are the ones ``network sessions`` declares ITSELF; the parent's launch
    flags are the sibling cell below, because they parse in a different place.
    """
    opened = _patch(monkeypatch, _FakeViewer(), _remote_row())

    assert _run(["sessions", "--peer", PEER, "--send", SESSION, flag, value, "is", "the"]) == 2
    err = capsys.readouterr().err
    assert flag in err, err
    assert "`--`" in err, err
    assert opened.calls == 0, "nothing may be dialled for a refused command line"


@pytest.mark.parametrize("flag", ["--model", "--hosting", "--run-in"])
def test_a_parent_launch_flag_with_an_act_is_refused_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    """The reviewer's own shape: `--model` is a PARENT flag (it parses after the
    subcommand as well as before it), and with an act it is just as dropped as the
    ones above — `--send <s> --model X is the field` delivered "is the field".

    Parsed through the REAL entry point because that is where the parent's flags
    exist at all; the local stand-in in this file has no parent.
    """
    opened = _patch(monkeypatch, _FakeViewer(), _remote_row())

    assert _run_real(["sessions", "--peer", PEER, "--send", SESSION, flag, "X", "is", "the"]) == 2
    err = capsys.readouterr().err
    assert flag in err, err
    assert opened.calls == 0


def test_a_leading_flag_token_is_read_as_that_flag_and_the_rest_is_the_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The BOUNDARY the narrowed help text and guide now describe, pinned so the
    prose cannot drift from the parser again (that gap is what round 2 found).

    A token that IS this verb's flag is still that flag: `--send <s> --json is the
    field I mean` turns JSON output ON and delivers `is the field I mean`. The
    honest shape is the documented one — flags before the session id, `--` to send a
    dash-shaped first word deliberately.
    """
    viewer = _FakeViewer(rows=[_Message("assistant", "the reply")])
    _patch(monkeypatch, viewer, _remote_row())

    argv = ["sessions", "--json", "--peer", PEER, "--send", SESSION, "--json", "is", "the", "field"]
    assert _run(argv) == 0
    payload = ok(capsys.readouterr().out)
    assert payload["ok"] is True, payload
    assert viewer.waited == ["is the field"], viewer.waited


def test_a_dash_shaped_first_word_can_be_sent_with_the_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The escape the refusal sentences name, asserted on the exact string the
    reviewer used: with `--`, the flag-shaped first word IS the payload."""
    viewer = _FakeViewer()
    _patch(monkeypatch, viewer, _remote_row())

    assert (
        _run_real(
            [
                "sessions",
                "--peer",
                PEER,
                "--send",
                SESSION,
                "--",
                "--json",
                "is",
                "the",
                "field",
                "I",
                "mean",
            ]
        )
        == 0
    )
    assert viewer.waited == ["--json is the field I mean"], viewer.waited


# ---------------------------------------------------------------------------
# what the viewer declares (round 1, MAJOR-2)
# ---------------------------------------------------------------------------


def test_the_one_shot_viewer_declares_no_action_receipt_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owner completes an action receipt only when its client did NOT declare
    it (``runtime_must_complete``), so a viewer that declared the attached
    vocabulary and rendered nothing left ``/goal``, ``/agent`` and ``/team``
    setting state on the peer and starting no turn — at exit 0. A process that
    prints the owner's receipt and exits declares ``()``, and this is the only
    place that decision is visible before a frame is written."""
    viewer = _FakeViewer()
    opened = _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--peer", PEER, "--send", SESSION, "hi"]) == 0
    assert [call["slash_consumers"] for call in opened.viewer_kwargs] == [()], opened.viewer_kwargs


def test_the_tui_child_budget_is_derived_from_this_verbs_own_worst_case() -> None:
    """Round-1 MAJOR-3: the composer's child budget claimed to sit above what this
    verb waits and did not — 480 s against an act whose parts add up to 540 s — so
    a `/network sessions --send …` typed in a session reaped its child part-way
    through a wait the verb itself promises, and the composer showed a timeout
    about a command that was still working.

    The number is DERIVED now (``tui/network_cli.PILOT_CALL_TIMEOUT_S`` off
    ``PILOT_ACT_TIMEOUT_S``, which is off the three budgets), and this cell is what
    keeps a future edit from hardcoding it again: it fails the moment either term
    stops following the others.
    """
    from local_operator.tui.network_cli import PILOT_CALL_TIMEOUT_S

    assert net_cli.PILOT_ACT_TIMEOUT_S == 2 * net_cli.PILOT_BIND_TIMEOUT_S + max(
        net_cli.PILOT_TURN_TIMEOUT_S, net_cli.PILOT_REPLY_TIMEOUT_S
    )
    assert PILOT_CALL_TIMEOUT_S > net_cli.PILOT_ACT_TIMEOUT_S


def test_the_whole_act_is_bounded_by_that_worst_case(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half of MAJOR-3: the number is ENFORCED, not merely stated.

    One bound wraps the whole act at ``PILOT_ACT_TIMEOUT_S``, so the reads with no
    rung of their own — the row lookup, the teardown — are inside it too. The
    constant is turned down here rather than waiting 540 s for the real one: what
    the cell proves is that a hung act is reaped and REPORTED, which is the
    property a caller holds.
    """
    viewer = _FakeViewer(send_sleep_s=5.0)
    _patch(monkeypatch, viewer, _remote_row())
    monkeypatch.setattr(net_cli, "PILOT_ACT_TIMEOUT_S", 0.5)

    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hi"]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["code"] == "session_unreachable"
    assert "worst case this verb budgets for" in payload["message"], payload
    # The viewer is torn down on the way out, cancelled turn or not.
    assert viewer.disposed


# ---------------------------------------------------------------------------
# a receipt this build cannot read (round 1, NIT-6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "receipt",
    [
        {"kind": "notice", "text": "goal set", "data": {}},
        {"kind": "notice", "text": "goal set", "style": "celebratory", "data": {}},
        "goal set",
    ],
    ids=["no-style", "unknown-style", "prose"],
)
def test_a_receipt_this_build_cannot_read_is_unreported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], receipt: Any
) -> None:
    """``ok`` used to be ``style != "error"``, which is TRUE for a receipt carrying
    no style at all — a success claim about a field the owner never sent — and a
    non-dict answer was reported ``ok=True, outcome="answered"`` even when it was
    a refusal. Both are ``slash_unreported`` now, the family's own word for a
    receipt that does not carry the fact it is a receipt for."""
    viewer = _FakeViewer(slash_receipt=receipt)
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--json", "--peer", PEER, "--slash", SESSION, "/goal ship it"]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["code"] == "slash_unreported"
    assert payload["ok"] is False


# ---------------------------------------------------------------------------
# id-or-name targets (sessions-remote-tools.md §3B)
#
# A pilot target may be an id OR a conversation name: the exact id resolves
# anywhere on the mesh (unchanged), and a name resolves through the pure
# selector over the rows THIS --peer reported. What these cells pin is the
# wiring: which read runs, what the receipt names, and that an ambiguous word
# refuses BEFORE a viewer exists or a frame is written.


def test_a_send_target_resolves_a_name_through_the_peers_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--send <name>`: the selector resolves, and the act lands on the RESOLVED id.

    The typed word is a name no wire op in this family has ever taken, so the
    receipt must read back the session's own id — two callers who addressed one
    conversation two ways must get one answer.
    """
    viewer = _FakeViewer()
    opened = _patch(monkeypatch, viewer, None)
    _patch_peer_rows(monkeypatch, [_remote_row(name="pilot-run")])
    assert _run(["sessions", "--json", "--peer", PEER, "--send", "pilot-run", "hello"]) == 0
    payload = ok(capsys.readouterr().out)
    assert payload["ok"] is True, payload
    assert payload["session_id"] == SESSION, "the receipt names the resolved id"
    assert viewer.waited == ["hello"]
    assert opened.calls == 1


def test_a_slash_target_resolves_a_name_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same resolution for `--slash`: one helper, so every pilot verb agrees."""
    viewer = _FakeViewer()
    _patch(monkeypatch, viewer, None)
    _patch_peer_rows(monkeypatch, [_remote_row(name="slash-by-name")])
    assert (
        _run(["sessions", "--json", "--peer", PEER, "--slash", "slash-by-name", "/rename better"])
        == 0
    )
    payload = ok(capsys.readouterr().out)
    assert payload["session_id"] == SESSION, payload
    assert viewer.slashes and viewer.slashes[0][0] == "rename", viewer.slashes


def test_an_ambiguous_name_lists_the_candidates_and_opens_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two whole conversations behind one word: refuse WITH the rows, guess at neither.

    This is the wrong-recipient case the whole resolution exists for, so the
    cell asserts all three halves: the code, BOTH retypeable ids in the
    sentence, and that no viewer was opened for either of them.
    """
    viewer = _FakeViewer()
    opened = _patch(monkeypatch, viewer, None)
    _patch_peer_rows(
        monkeypatch,
        [
            _remote_row(session_id="aaaa11112222", name="checklist one"),
            _remote_row(session_id="bbbb33334444", name="checklist two"),
        ],
    )
    assert _run(["sessions", "--json", "--peer", PEER, "--send", "checklist", "hello"]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["code"] == "session_ambiguous", payload
    assert "aaaa11112222" in payload["message"], payload
    assert "bbbb33334444" in payload["message"], payload
    assert "checklist one" in payload["message"], payload
    assert opened.calls == 0, "an ambiguous target must not open a viewer"
    assert viewer.waited == []


def test_a_name_on_another_device_is_not_this_peers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The name tier is scoped to the device the caller NAMED (`--peer`).

    A conversation is only addressed on the device that holds it, and reaching
    for it elsewhere is how an act lands on a namesake: the rows the OTHER
    device reported must not satisfy a name resolved against this one.
    """
    _patch(monkeypatch, _FakeViewer(), None)
    _patch_unresolved(
        monkeypatch,
        answer={"peers": {PEER: {"name": PEER, "reachable": True, "reason": ""}}},
    )
    _patch_peer_rows(
        monkeypatch,
        [_remote_row(name="on-elsewhere", device_id="d_other", device_name="other-node")],
    )
    assert _run(["sessions", "--json", "--peer", PEER, "--send", "on-elsewhere", "hi"]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["code"] == "session_unknown", payload
    assert "does not hold" in payload["message"], payload


def test_a_stop_target_may_be_the_conversations_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--stop`` gets the same id-or-name translation: the relay sees the id."""
    frames: list[dict[str, Any]] = []

    def _capture(op: str, **fields: Any) -> dict[str, Any]:
        frames.append({"op": op, **fields})
        return {"rung": "socket", "outcome": "stopped", "pid": 1, "detail": "stopped"}

    monkeypatch.setattr(net_cli, "_relay_answer", _capture)
    _patch(monkeypatch, None, None)
    _patch_peer_rows(monkeypatch, [_remote_row(name="the-session")])
    assert _run(["sessions", "--json", "--peer", PEER, "--stop", "the-session"]) == 0
    capsys.readouterr()
    assert len(frames) == 1, frames
    assert frames[0]["op"] == "peer_session_stop"
    assert frames[0]["session_id"] == SESSION, "the frame carries the RESOLVED id"
    assert frames[0]["peer"] == PEER
    assert frames[0]["mode"] == "graceful"


def test_an_ambiguous_stop_is_refused_before_any_frame(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ambiguous stop refuses locally; nothing was sent to the peer."""
    frames: list[dict[str, Any]] = []
    monkeypatch.setattr(
        net_cli, "_relay_answer", lambda op, **fields: frames.append(op) or {}  # noqa: ARG005
    )
    _patch(monkeypatch, None, None)
    _patch_peer_rows(
        monkeypatch,
        [
            _remote_row(session_id="aaaa11112222", name="checklist one"),
            _remote_row(session_id="bbbb33334444", name="checklist two"),
        ],
    )
    assert _run(["sessions", "--json", "--peer", PEER, "--stop", "checklist"]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["code"] == "session_ambiguous", payload
    assert "aaaa11112222" in payload["message"] and "bbbb33334444" in payload["message"]
    assert frames == [], "an ambiguous stop must not reach the relay"


def test_an_unresolved_stop_target_passes_through_unchanged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A miss is NOT this side's verdict: the peer answers for what it holds (§9.2).

    The cache that resolved nothing may be seconds old, and for these two verbs
    the owner runs the ladder — so the typed value travels unchanged and the
    owner's own answer comes back (the pinned contract these verbs already had).
    """
    frames: list[dict[str, Any]] = []

    def _capture(op: str, **fields: Any) -> dict[str, Any]:
        frames.append({"op": op, **fields})
        return {"rung": "socket", "outcome": "stopped", "pid": 1, "detail": "stopped"}

    monkeypatch.setattr(net_cli, "_relay_answer", _capture)
    _patch(monkeypatch, None, None)
    _patch_peer_rows(monkeypatch, [])
    assert _run(["sessions", "--json", "--peer", PEER, "--stop", "deadbeefcafe"]) == 0
    capsys.readouterr()
    assert frames[0]["session_id"] == "deadbeefcafe", frames


def test_an_engage_target_may_be_the_conversations_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--engage`` too: the same translation, one spelling."""
    frames: list[dict[str, Any]] = []

    def _capture(op: str, **fields: Any) -> dict[str, Any]:
        frames.append({"op": op, **fields})
        return {"engaged": True, "session_id": SESSION, "detail": "warm"}

    monkeypatch.setattr(net_cli, "_relay_answer", _capture)
    _patch(monkeypatch, None, None)
    _patch_peer_rows(monkeypatch, [_remote_row(name="warm-me", live_state="")])
    assert _run(["sessions", "--json", "--peer", PEER, "--engage", "warm-me"]) == 0
    capsys.readouterr()
    assert frames[0]["op"] == "peer_session_engage"
    assert frames[0]["session_id"] == SESSION, frames


# ---------------------------------------------------------------------------
# --peek: the tail window, and the live-only gate (§3C/§8.3)
# ---------------------------------------------------------------------------


def test_a_peek_reads_the_requested_tail_window(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The newest `--steps` rows, in order, with has_older telling the truth."""
    viewer = _FakeViewer(
        rows=[
            _Message("user", "one"),
            _Message("assistant", "two"),
            _Message("user", "three"),
            _Message("assistant", "four"),
        ]
    )
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--json", "--peer", PEER, "--peek", SESSION, "--steps", "2"]) == 0
    payload = ok(capsys.readouterr().out)
    assert payload["ok"] is True and payload["verb"] == "peek", payload
    assert payload["session_id"] == SESSION
    assert payload["steps"] == 2
    assert [row["text"] for row in payload["rows"]] == ["three", "four"]
    assert payload["has_older"] is True
    assert viewer.bound is True and viewer.disposed is True


def test_a_peek_defaults_to_twelve_steps(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No `--steps`: the sessions tool's own default, so one verb, one window size."""
    viewer = _FakeViewer(rows=[_Message("user", f"line {i}") for i in range(20)])
    _patch(monkeypatch, viewer, _remote_row())
    assert _run(["sessions", "--json", "--peer", PEER, "--peek", SESSION]) == 0
    payload = ok(capsys.readouterr().out)
    assert payload["steps"] == 12
    assert [row["text"] for row in payload["rows"]] == [f"line {i}" for i in range(8, 20)]


def test_a_stored_session_refuses_peek_and_is_never_bound(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """§8.3, LOCKED: a read never starts a runtime on a device nobody is watching.

    The refusal must land BEFORE the viewer exists — whether a bind ever ran is
    what this cell proves, because a gate that merely raised after opening
    would still have warmed the runtime a peek must not wake. The sentence
    names the route out (``--engage`` / the agent's resume) so the next action
    is a warm, not a retried read.
    """
    viewer = _FakeViewer(rows=[_Message("assistant", "should not be seen")])
    opened = _patch(monkeypatch, viewer, _remote_row(live_state=""))
    assert _run(["sessions", "--json", "--peer", PEER, "--peek", SESSION]) == 1
    payload = ok(capsys.readouterr().out)
    assert payload["code"] == "session_stored", payload
    assert SESSION in payload["message"] and PEER in payload["message"], payload
    assert "--engage" in payload["message"], payload
    assert "resume" in payload["message"], payload
    assert opened.calls == 0, "the gate must fire before a viewer exists"
    assert viewer.bound is False, "a stored session must never be bound or warmed"
    assert viewer.waited == []


def test_peek_takes_no_text_and_never_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A read has no payload — and skipping the text path is load-bearing.

    ``_pilot_text`` reads STDIN when the positional is empty and stdin is not a
    TTY, so a peek that went through it would hang a script on a pipe it never
    asked for; this cell's stdin raises the moment anything reads it.
    """
    viewer = _FakeViewer(rows=[_Message("assistant", "x")])
    _patch(monkeypatch, viewer, _remote_row())

    class _NoRead(io.StringIO):
        def read(self, *args: Any, **kwargs: Any) -> str:  # noqa: ARG002
            raise AssertionError("--peek read stdin")

    piped = _NoRead("a body that must not be consumed")
    monkeypatch.setattr("sys.stdin", piped)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    assert _run(["sessions", "--json", "--peer", PEER, "--peek", SESSION]) == 0
    capsys.readouterr()
    assert _run(["sessions", "--peer", PEER, "--peek", SESSION, "loud", "words"]) == 2
    assert "takes no text" in capsys.readouterr().err


def test_a_peek_beside_another_act_is_two_acts(capsys: pytest.CaptureFixture[str]) -> None:
    """The one-act rule covers the read too."""
    assert _run(["sessions", "--peer", PEER, "--peek", SESSION, "--send", SESSION, "hi"]) == 2
    assert "not a pipeline" in capsys.readouterr().err


def test_steps_without_peek_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Same class as ``--force`` without ``--stop``: meaning one thing, dropped otherwise."""
    assert _run(["sessions", "--peer", PEER, "--steps", "5"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--steps applies to --peek only" in captured.err


@pytest.mark.parametrize("value", ["0", "-1", "51", "1000"])
def test_steps_out_of_range_reads_nothing(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    """The window's bounds are the family's (1..``comms.PEEK_MAX_STEPS``), enforced."""
    assert _run(["sessions", "--peer", PEER, "--peek", SESSION, "--steps", value]) == 2
    assert "1..50" in capsys.readouterr().err


def test_the_peek_bound_carries_no_spawn_and_no_turn() -> None:
    """§3A's budget table, one row over: the READ rung is what bounds a peek.

    The driving act's bound pays for a spawn and an unbounded turn; a peek can
    reach neither (the gate above refuses stored sessions before any bind), so
    its bound is the same derivation with the REPLY rung in place of the turn,
    and it must actually be SMALLER — a bound that kept the 540 s spawn budget
    would be a budget for work this verb never does.
    """
    from local_operator.tui import network_cli as tui_mod

    assert (
        net_cli.PILOT_PEEK_TIMEOUT_S
        == 2 * net_cli.PILOT_BIND_TIMEOUT_S + net_cli.PILOT_REPLY_TIMEOUT_S
    )
    assert net_cli.PILOT_PEEK_TIMEOUT_S < net_cli.PILOT_ACT_TIMEOUT_S
    # And the TUI's child budget still outlasts it, so a composer shows THIS
    # verb's own answer rather than a timeout about a call that was working.
    assert net_cli.PILOT_PEEK_TIMEOUT_S < tui_mod.PILOT_CALL_TIMEOUT_S
