"""The model-choice front door: a create names both halves of a model, or neither.

WHAT THIS FILE EXISTS TO PIN, AND HOW THE DEFECT WAS REPRODUCED. A create's model
choice is a PAIR — ``provider`` and ``model_id`` — and nothing downstream can
derive either half from the other. The natural spelling at the shell is the pair
itself, and it used to fall on the floor:

    lop network sessions --peer cloud-node-1 --create --model anthropic/claude-sonnet-5-5

The CLI read ``--model`` as a bare model id, so the frame carried
``{"provider": "", "model_id": "anthropic/claude-sonnet-5-5"}``; the relay's
``_model_choice`` drops a half-filled choice, and the reply answered
``model: {"applied": false, "detail": ""}`` — a fallback to the node's default
reported as if nothing had been asked for.

THE TWO HALVES:

* THE RELAY SAYS WHY. A model that was NAMED but could not be taken answers a
  ``model.detail`` sentence naming the shape it saw — the half-filled pair, or a
  value that is not an object at all — while the empty detail for "nothing was
  asked" and for a full choice the runtime accepted stays exactly as it was.
* THE CLI READS THE PAIR. ``--model provider/model-id`` (no ``--hosting``)
  splits on the FIRST ``/``; ``--hosting provider --model id`` keeps today's
  exact reading; a bare model id with no hosting is REFUSED at the parse
  (rc 2) before a frame is built, because the relay's sentence would otherwise
  arrive only after the session had already been minted on the peer.

THE RECEIPT HALF rides along: a non-empty ``model.detail`` now prints on the
human create receipt for a pure ``--model`` create too (it was gated on an
identity being named — the only producer that sentence had when the gate was
written).

The relay cells follow ``test_create_team_owns_agent_slot.py``'s idiom (a
``_warmed_server`` whose spawn/engage/prompt are stubbed — the create REPLY is
the subject); the CLI cells follow ``test_unattended_surfaces.py``'s (the REAL
parser the user runs, with the relay answer stubbed at its one seam,
``_relay_answer``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.cli import build_cli_parser
from local_operator.network import cli as net_cli
from local_operator.network import identity, relay

PEER = "cloud-node-1"
SESSION = "9f3ac1e0b7d2"

#: The relay's sentence for the half-filled frame the repro produced — the exact
#: value the caller's ``model.detail`` carries.
HALF_FILLED_SENTENCE = (
    "the model choice needs both a provider and a model id "
    "(got provider='', model_id='anthropic/claude-sonnet-5-5')"
)
NOT_AN_OBJECT_SENTENCE = (
    "the model choice must be an object with a provider and a model id "
    "(got 'anthropic/claude-sonnet-5-5')"
)
#: The CLI's refusal for a value that names no provider at all.
BARE_MODEL_SENTENCE = (
    "--model must name both halves on a peer create: write --model <provider>/<model-id> "
    "(for example --model anthropic/claude-sonnet-5-5), or pair a bare model id with "
    "--hosting <provider>"
)


# ---------------------------------------------------------------------------
# The relay: the reply names the reason a named choice could not be taken
# ---------------------------------------------------------------------------


def _server(root: Path) -> relay.RelayServer:
    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )


def _warmed_server(root: Path) -> relay.RelayServer:
    """A server whose spawn/engage/prompt are stubbed: the create REPLY is the subject."""
    server = _server(root)
    server._warm_after_create = lambda *args, **kwargs: None
    server._engage_locally = lambda *args, **kwargs: ""
    server._prompt_on = lambda *args, **kwargs: (True, "")
    return server


def _link() -> Any:
    """A stub link carrying exactly what ``_op_session_create`` reads."""
    return SimpleNamespace(
        device_id="d_" + "a" * 32,
        network_id="n_0123456789abcdef0123456789abcdef",
        epoch=1,
        context=SimpleNamespace(
            capabilities=frozenset({"prompt"}),
            device_id="d_" + "a" * 32,
            network_id="n_0123456789abcdef0123456789abcdef",
            epoch=1,
        ),
    )


def _frame(**fields: Any) -> dict[str, Any]:
    return {"op": "net_session_create", "req": 41, "cwd": "", "prompt": "hi", **fields}


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        pytest.param(
            {"provider": "", "model_id": "anthropic/claude-sonnet-5-5"},
            HALF_FILLED_SENTENCE,
            id="model-id-only",
        ),
        pytest.param(
            {"provider": "anthropic", "model_id": ""},
            "the model choice needs both a provider and a model id "
            "(got provider='anthropic', model_id='')",
            id="provider-only",
        ),
    ],
)
def test_a_named_but_half_filled_choice_answers_the_reason_it_was_dropped(
    root: Path, choice: dict[str, str], expected: str
) -> None:
    """The defect: this reply used to be ``{"applied": false, "detail": ""}``.

    Empty detail is the vocabulary's "nothing was asked", so the caller read a
    fallback to the node's default as the absence of a request. The sentence
    names the SHAPE and the values it saw, because both halves are what the wire
    needs and this is the one reply that can say which half is missing.
    """
    identity.mint(root, name="cloud-node-1")
    asked: list[Any] = []
    server = _warmed_server(root)
    server._set_model_on = lambda *args, **kwargs: asked.append((args, kwargs)) or {
        "applied": True,
        "detail": "",
    }

    detail = server._op_session_create(_link(), _frame(model=choice))  # noqa: SLF001

    assert detail["model"] == {"applied": False, "detail": expected}
    assert asked == [], "a half-filled choice must not reach the runtime's set_model"


def test_a_choice_that_is_not_an_object_answers_too(root: Path) -> None:
    """A truthy non-dict is as unusable as a half-filled pair — and was as silent.

    ``_model_choice``'s predicate accepts only an object with both halves, so a
    client that sent the pair as one string was dropped exactly like the
    half-filled frame; the sentence names the shape the wire takes.
    """
    identity.mint(root, name="cloud-node-1")
    server = _warmed_server(root)

    detail = server._op_session_create(  # noqa: SLF001
        _link(), _frame(model="anthropic/claude-sonnet-5-5")
    )

    assert detail["model"] == {"applied": False, "detail": NOT_AN_OBJECT_SENTENCE}


def test_a_full_choice_still_reaches_the_runtime_and_keeps_the_silent_success(root: Path) -> None:
    """The control: the applied path is untouched — empty detail on success.

    The refusal must be exactly the half that was MISSING from the old reply: a
    full pair goes to ``_set_model_on`` as it always did, and its outcome (here
    the stub's success) is reported verbatim.
    """
    identity.mint(root, name="cloud-node-1")
    asked: list[dict[str, Any]] = []
    server = _warmed_server(root)

    def _set_model(session_id: str, model: dict[str, Any]) -> dict[str, Any]:
        asked.append(dict(model))
        return {"applied": True, "detail": ""}

    server._set_model_on = _set_model  # noqa: SLF001
    choice = {"provider": "anthropic", "model_id": "claude-sonnet-5-5"}

    detail = server._op_session_create(_link(), _frame(model=choice))  # noqa: SLF001

    assert asked == [choice]
    assert detail["model"] == {"applied": True, "detail": ""}


def test_a_promptless_create_names_the_reason_on_its_cold_reply(root: Path) -> None:
    """The warm branch is a second reply shape, and the sentence must cross it too.

    A promptless create answers before the runtime joins (``warming: true``), so
    its ``model`` block is composed in a different expression; a half-filled
    choice used to come back empty there as well. A full choice keeps the join
    sentence — the model is applied when the runtime arrives.
    """
    identity.mint(root, name="cloud-node-1")
    server = _warmed_server(root)

    cold = server._op_session_create(  # noqa: SLF001
        _link(),
        _frame(prompt="", model={"provider": "", "model_id": "anthropic/claude-sonnet-5-5"}),
    )
    assert cold["warming"] is True
    assert cold["model"] == {"applied": False, "detail": HALF_FILLED_SENTENCE}

    joining = server._op_session_create(  # noqa: SLF001
        _link(),
        _frame(prompt="", model={"provider": "anthropic", "model_id": "claude-sonnet-5-5"}),
    )
    assert joining["model"] == {
        "applied": False,
        "detail": "the runtime is joining; the model is applied when it arrives",
    }


_NOT_ASKED = object()


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(_NOT_ASKED, id="absent"),
        pytest.param(None, id="null"),
        pytest.param({}, id="empty-object"),
        pytest.param({"provider": "", "model_id": ""}, id="both-halves-empty"),
    ],
)
def test_nothing_asked_stays_silent(root: Path, model: object) -> None:
    """The empty detail on ABSENCE is the direction this change must keep.

    "Nothing was asked" and "a request that cannot be taken" are different
    facts; only the second earns a sentence. The empty spellings (absent, null,
    an empty object, both halves empty) must keep answering ``""``.
    """
    identity.mint(root, name="cloud-node-1")
    server = _warmed_server(root)
    fields = {} if model is _NOT_ASKED else {"model": model}

    detail = server._op_session_create(_link(), _frame(**fields))  # noqa: SLF001

    assert detail["model"] == {"applied": False, "detail": ""}


def test_the_predicate_and_its_reason_agree_about_which_shapes_are_choices() -> None:
    """``_model_choice`` and ``_model_choice_refusal`` read one boundary, both ways.

    The first is the guard the runtime's ``set_model`` is gated on (``None`` for
    every unusable shape); the second is the reply's sentence, and it must be
    non-empty exactly where the first is ``None`` for a NAMED model — never for
    an applied choice and never for an absent one.
    """
    from local_operator.network.relay import _model_choice, _model_choice_refusal

    full = {"provider": "anthropic", "model_id": "claude-sonnet-5-5"}
    assert _model_choice(full) is full
    assert _model_choice_refusal(full) == ""
    for unusable in (
        {"provider": "anthropic", "model_id": ""},
        {"provider": "", "model_id": "claude-sonnet-5-5"},
        "anthropic/claude-sonnet-5-5",
    ):
        assert _model_choice(unusable) is None
        assert _model_choice_refusal(unusable) != ""
    for silent in (None, {}, {"provider": "", "model_id": ""}, ""):
        assert _model_choice(silent) is None
        assert _model_choice_refusal(silent) == ""


# ---------------------------------------------------------------------------
# The CLI: the pair's own spelling, read at the parse
# ---------------------------------------------------------------------------


def _parse(extra: list[str]) -> Any:
    return build_cli_parser().parse_args(
        ["network", "sessions", "--peer", PEER, "--create", *extra]
    )


class _Recorder:
    """``_relay_answer``, recording the create frame it was asked to send.

    An entry that is an Exception INSTANCE is raised; anything else is returned;
    the last entry repeats, so a cell that expects one call is one line of
    setup. ``peer_session_create`` is the only op any cell here sends.
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


def test_the_slash_form_travels_as_the_pair_the_wire_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--model provider/model-id`` is ONE argument naming both halves.

    The split is on the FIRST ``/``: an aggregator's vendor namespace
    (``ollama/hf.co/...``) must keep its second slash inside the model id, and
    only the leading segment is taken as the provider.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    recorder = _Recorder({"session_id": SESSION})
    monkeypatch.setattr(net_cli, "_relay_answer", recorder)

    rc = net_cli.main(_parse(["--model", "anthropic/claude-sonnet-5-5"]))
    assert rc == 0, capsys.readouterr()
    assert recorder.calls[0]["model"] == {
        "provider": "anthropic",
        "model_id": "claude-sonnet-5-5",
    }, recorder.calls

    rc = net_cli.main(_parse(["--model", "ollama/hf.co/some-model"]))
    assert rc == 0, capsys.readouterr()
    assert recorder.calls[1]["model"] == {
        "provider": "ollama",
        "model_id": "hf.co/some-model",
    }, recorder.calls


def test_hosting_keeps_days_exact_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With ``--hosting`` the model is an ID, verbatim — slash or not.

    Two facts are pinned: the value reaches the frame UNSPLIT (a hosting is
    already the provider half), and ``--hosting`` alone keeps today's exact
    reading — a half-filled dict the relay now explains, not a usage error
    invented here (the read this change owns is ``--model``'s).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    recorder = _Recorder({"session_id": SESSION})
    monkeypatch.setattr(net_cli, "_relay_answer", recorder)

    rc = net_cli.main(_parse(["--hosting", "anthropic", "--model", "claude-x"]))
    assert rc == 0, capsys.readouterr()
    assert recorder.calls[0]["model"] == {"provider": "anthropic", "model_id": "claude-x"}

    rc = net_cli.main(_parse(["--hosting", "anthropic", "--model", "a/b"]))
    assert rc == 0, capsys.readouterr()
    assert recorder.calls[1]["model"] == {
        "provider": "anthropic",
        "model_id": "a/b",
    }, "a slash in the id is not a provider once --hosting named one"

    rc = net_cli.main(_parse(["--hosting", "anthropic"]))
    assert rc == 0, capsys.readouterr()
    assert recorder.calls[2]["model"] == {"provider": "anthropic", "model_id": ""}


@pytest.mark.parametrize("value", ["claude-sonnet-5-5", "/claude-sonnet-5-5", "anthropic/"])
def test_a_model_id_that_cannot_name_a_pair_is_refused_at_the_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    """No usable pair and no ``--hosting``: refuse in a sentence, before anything moves.

    The wire needs both halves and nothing on this side can derive a provider
    from a bare id; letting the create through would mint a session on the
    peer's default and explain afterwards — after the session already exists.
    The refusal lands before ``_relay_answer``, so no frame is built and no
    definition is pushed.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    recorder = _Recorder({"session_id": SESSION})
    monkeypatch.setattr(net_cli, "_relay_answer", recorder)

    rc = net_cli.main(_parse(["--model", value]))

    assert rc == 2
    assert BARE_MODEL_SENTENCE in capsys.readouterr().err
    assert recorder.calls == [], "a refused parse must not reach the relay"


def test_no_model_and_no_hosting_sends_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The byte-identical floor: a create that names no model sends no model key."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    recorder = _Recorder({"session_id": SESSION})
    monkeypatch.setattr(net_cli, "_relay_answer", recorder)

    assert net_cli.main(_parse([])) == 0, capsys.readouterr()
    assert recorder.calls[0]["model"] is None


def test_the_receipt_prints_the_peers_model_sentence_for_a_pure_model_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reason reaches the person, not only ``--json``.

    The receipt's model line was gated on an identity being named — when the
    override sentence was the only thing it could carry. A pure ``--model``
    create is the caller who TYPED the model; a non-empty detail must print,
    and an empty one must print nothing (byte-identical to the old receipt).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    reason = "no such model on that device"
    recorder = _Recorder(
        {"session_id": SESSION, "model": {"applied": False, "detail": reason}},
        {"session_id": SESSION, "model": {"applied": True, "detail": ""}},
        {"session_id": SESSION},
    )
    monkeypatch.setattr(net_cli, "_relay_answer", recorder)

    assert net_cli.main(_parse(["--model", "anthropic/claude-sonnet-5-5"])) == 0
    with_reason = capsys.readouterr().out
    assert net_cli.main(_parse(["--model", "anthropic/claude-sonnet-5-5"])) == 0
    empty_detail = capsys.readouterr().out
    assert net_cli.main(_parse(["--model", "anthropic/claude-sonnet-5-5"])) == 0
    no_block = capsys.readouterr().out

    assert "created on cloud-node-1" in with_reason
    assert reason in with_reason
    assert reason not in empty_detail
    assert empty_detail == no_block, "an empty detail prints no line, exactly as before"
