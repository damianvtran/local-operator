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

    def __call__(self, viewer: _FakeViewer | None, row: Any = None):  # type: ignore[no-untyped-def]
        async def _open(session_id: str, **kwargs: Any) -> Any:
            self.calls += 1
            return viewer

        return _open


def _remote_row(*, reachable: bool = True, reason: str = "") -> SessionRow:
    return SessionRow(
        id=SESSION,
        mtime=1.0,
        name="pilot",
        locality="remote",
        owner_device="d_428cb39f92ebbd094acf5d30a2db7bfb",
        owner_device_name=PEER,
        reachable=reachable,
        unreachable_reason=reason,
    )


def _patch(monkeypatch: pytest.MonkeyPatch, viewer: _FakeViewer | None, row: Any = None) -> _Opened:
    """Point the seam the verb uses at ``viewer``/``row``.

    Patched on ``session.remote_open`` itself because that is where the verb
    imports them from at call time — the same seam ``test_remote_open.py`` drives
    the app through (`remote_open.open_remote_viewer`), so a change of entry point
    breaks this test rather than silently moving under it.
    """
    import local_operator.session.remote_open as remote_open

    opened = _Opened()
    monkeypatch.setattr(remote_open, "open_remote_viewer", opened(viewer, row))
    monkeypatch.setattr(remote_open, "remote_row_for", lambda _session_id, _root: row)
    return opened


def _run(argv: list[str]) -> int:
    return int(net_cli.main(_parser().parse_args(["network", *argv])))


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
    assert _run(["sessions", "--send", SESSION, "hi", "--json"]) == 1
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
    assert _run(["sessions", "--peer", PEER, "--send", SESSION, "--json"]) == 0
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

    assert _run(["sessions", "--peer", PEER, "--send", SESSION, "hi", "--json"]) == 1
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
    seconds after it died. The classification is now by construction, and only this
    device's own relay record — a local file read — can change it.
    """
    import local_operator.network.store as store_mod

    import local_operator.session.remote_open as remote_open

    _patch(monkeypatch, _BlindViewer(rung), _remote_row())
    if rung == "open":
        monkeypatch.setattr(remote_open, "open_remote_viewer", _blind_open)
    monkeypatch.setattr(
        store_mod, "find_own_relay", lambda root=None: object() if relay_up else None
    )

    assert _run(["sessions", "--peer", PEER, "--send", SESSION, "hi", "--json"]) == 1
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
    assert _run(["sessions", "--peer", PEER, "--send", SESSION, "hi", "--json"]) == 1
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
    assert _run(["sessions", "--peer", PEER, "--send", SESSION, "hi", "--json"]) == 1
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
    assert _run(["sessions", "--peer", PEER, "--send", SESSION, "hi", "--json"]) == 1
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
    assert _run(["sessions", "--peer", PEER, "--send", SESSION, "hi", "--json"]) == 1
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
    assert _run(["sessions", "--peer", PEER, "--steer", SESSION, "wait, not that", "--json"]) == 1
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
    assert _run(["sessions", "--peer", PEER, "--slash", SESSION, "/nope", "--json"]) == 1
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
    assert _run(["sessions", "--peer", PEER, "--steer", SESSION, "go", "--json"]) == 0
    printed = capsys.readouterr().out
    payload = ok(printed)
    assert payload["ok"] is True
    assert payload["verb"] == "steer"
    assert payload["session_id"] == SESSION
    assert payload["peer"] == PEER
    assert payload["receipt"] == "steering queued"
