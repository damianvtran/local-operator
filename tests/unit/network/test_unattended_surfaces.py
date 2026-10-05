"""The defect-2 SURFACE wires on the CREATE side (remote-onboarding §6).

WHAT THESE PIN, AND WHY HERE. The mechanism shipped before these wires did: the
receiver-side ``unattended`` gate on ``yolo`` (``relay._op_session_create``, pinned
in ``test_unattended_create.py``) and the carry itself were live while NOTHING on
the requesting device ever consulted its saved ``tool_approval_mode`` — so an
operator with full-auto locally still got an attended (prompting) session on a
peer: defect 2's exact shape, "full-auto does not follow a placement". These cells
drive ``cli._cmd_sessions`` — through the REAL parser the user runs, with the
relay answer stubbed at its one seam (``_relay_answer``) — and pin the four
surface behaviours:

* config ``auto`` => the create REQUESTS ``yolo`` (the implied request);
* a peer's refusal of that IMPLIED request falls back to an ATTENDED create, rc 0,
  with the notice on the receipt and in the ``--json`` payload;
* an EXPLICIT ``--yolo`` keeps the refusal verbatim (the design's acceptance): the
  user asked for unattended itself, so a fallback would hide the refusal;
* config ``ask`` (or no config at all) sends today's exact body — ``yolo=false``
  present, no notice — so the change is invisible unless the device says auto.

The move half of the same defect is pinned in ``test_mobility.py``; the desktop
builder's half in ``tests/unit/server/test_desktop_mesh.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.cli import build_cli_parser
from local_operator.config import ConfigManager
from local_operator.network import cli as net_cli
from local_operator.network.types import MeshRefusal

PEER = "cloud-node-1"
SESSION = "9f3ac1e0b7d2"
#: The peer's own refusal sentence, verbatim (``relay._op_session_create``'s copy
#: for a named device). The tests treat it as an opaque fixture: what this side
#: must not do is EDIT it — the fallback drops it in favour of the notice, and the
#: explicit arm raises it untouched.
REFUSAL_SENTENCE = (
    "a session created on another device can start unattended (yolo) only when "
    "cloud-node-1 grants the requesting member 'unattended'. Its operator makes that "
    "grant there — approve setup for cloud-node-1 in the Mesh tab. Until then, create "
    "it here or start it on your own device with yolo."
)


def _parse(extra: list[str]) -> Any:
    return build_cli_parser().parse_args(
        ["network", "sessions", "--peer", PEER, "--create", *extra]
    )


def _save_mode(root: Path, mode: str) -> None:
    """The operator's saved mode, written the product's own way."""
    ConfigManager(root).set_config_value("tool_approval_mode", mode)


class _Recorder:
    """``_relay_answer``, recording every call; ``answers`` is consumed per call.

    An entry that is an Exception INSTANCE is raised; anything else is returned.
    The last entry repeats, so the two-call fallback shape is three lines of setup.
    """

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, op: str, *, timeout: float = 5.0, **fields: Any) -> dict[str, Any]:
        assert op == "peer_session_create", op
        self.calls.append(dict(fields))
        index = min(len(self.calls) - 1, len(self.answers) - 1)
        answer = self.answers[index]
        if isinstance(answer, Exception):
            raise answer
        return dict(answer)


def test_a_config_auto_create_requests_unattended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """(a) The implied request: ``auto`` on THIS device sends ``yolo=true``.

    This is the wire the defect was about: before it, the create body said
    ``yolo=false`` whatever the operator's saved mode, so a full-auto send parked
    on an approval card on the node. The read goes through
    ``session_factory.saved_tool_approval_is_auto`` — the one derivation — so this
    cell also fails if the request stops following the saved mode.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _save_mode(tmp_path, "auto")
    recorder = _Recorder({"session_id": SESSION})
    monkeypatch.setattr(net_cli, "_relay_answer", recorder)

    rc = net_cli.main(_parse([]))
    assert rc == 0, capsys.readouterr().out
    assert recorder.calls and recorder.calls[0]["yolo"] is True, recorder.calls
    # And nothing extra: the request succeeded, so there is no fallback to explain.
    out = capsys.readouterr().out
    assert "created on cloud-node-1" in out, out
    assert "created attended" not in out, out


def test_the_implied_request_falls_back_attended_with_the_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """(b) Refusal of the IMPLIED request => attended create, rc 0, notice carried.

    A full-auto send must not dead-end because a grant is missing: the peer
    refused ``yolo`` with ``not_permitted`` (raised above its mint — nothing
    durable behind), the retry drops ``yolo`` only, and the notice says what the
    session will do instead, how its cards are answered, and what the grant does
    and does not change — future sends from this device, never a live switch
    (design round 1, D1). Both the human receipt and the ``--json`` payload carry
    it; the payload field is additive.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _save_mode(tmp_path, "auto")
    recorder = _Recorder(MeshRefusal("not_permitted", REFUSAL_SENTENCE), {"session_id": SESSION})
    monkeypatch.setattr(net_cli, "_relay_answer", recorder)

    rc = net_cli.main(_parse(["--json"]))
    assert rc == 0, "the fallback's create succeeded; rc says so"
    assert [call["yolo"] for call in recorder.calls] == [True, False], recorder.calls
    payload = capsys.readouterr().out
    assert '"unattended_notice"' in payload, payload
    assert f"approve setup for {PEER} in the Mesh tab on {PEER}" in payload, payload
    assert "ask for approvals" in payload, payload
    # The grant's TRUE scope (design round 1, D1): it governs future sends and
    # cannot quiet this conversation — the sentence must not read as if it could.
    assert "covers future sends from this device, not this conversation" in payload, payload
    # The peer's refusal sentence is NOT echoed as the notice: it advises "create
    # it here", which is the advice for a FAILED create, and this one succeeded.
    assert "Until then, create it here" not in payload, payload


def test_the_fallback_receipt_says_it_on_the_human_lines_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """(b, receipt half) ``--json`` is not the only surface: the lines say it too."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _save_mode(tmp_path, "auto")
    recorder = _Recorder(MeshRefusal("not_permitted", REFUSAL_SENTENCE), {"session_id": SESSION})
    monkeypatch.setattr(net_cli, "_relay_answer", recorder)

    rc = net_cli.main(_parse([]))
    assert rc == 0
    out = capsys.readouterr().out
    assert "created attended" in out, out
    assert f"approve setup for {PEER} in the Mesh tab on {PEER}" in out, out


def test_an_explicit_yolo_keeps_the_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """(c) The explicit ask keeps the structural refusal, byte for byte.

    ``--yolo`` typed by a user is a request for unattended ITSELF; the fallback
    exists for the IMPLIED form only. One call, ``yolo=true``, the peer's own
    sentence on stderr, rc 1 — exactly the pre-change behaviour.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _save_mode(tmp_path, "ask")
    recorder = _Recorder(MeshRefusal("not_permitted", REFUSAL_SENTENCE))
    monkeypatch.setattr(net_cli, "_relay_answer", recorder)

    rc = net_cli.main(_parse(["--yolo"]))
    assert rc == 1
    assert len(recorder.calls) == 1, "an explicit refusal must not be retried"
    assert recorder.calls[0]["yolo"] is True, recorder.calls
    captured = capsys.readouterr()
    assert REFUSAL_SENTENCE in captured.err, captured


@pytest.mark.parametrize("mode", [None, "ask"])
def test_a_non_auto_device_sends_todays_exact_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str | None,
) -> None:
    """(d) With config ``ask`` (or no config file), the body is byte-identical.

    ``yolo=false`` is present exactly as it always was, there is no notice, and
    no second call happens: the whole change is invisible unless the device says
    ``auto``. The no-file arm also pins the fresh-root guard — the read must not
    materialise a config directory to answer ``ask``.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    if mode is not None:
        _save_mode(tmp_path, mode)
    recorder = _Recorder({"session_id": SESSION})
    monkeypatch.setattr(net_cli, "_relay_answer", recorder)

    rc = net_cli.main(_parse(["--json"]))
    assert rc == 0, capsys.readouterr().out
    if mode is None:
        materialised = (tmp_path / "config.yml").exists()
        assert not materialised, "the fresh-root read must not materialise a config file"
    assert len(recorder.calls) == 1, recorder.calls
    assert recorder.calls[0]["yolo"] is False, recorder.calls
    payload = capsys.readouterr().out
    assert "unattended_notice" not in payload, payload
