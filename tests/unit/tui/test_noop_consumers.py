"""A routed slash outcome that nothing renders is a command that vanishes.

**This file exists because `/team lopdev <request>` did nothing on every fresh
`lop` for ten releases and no test failed.**

The mechanism is worth stating precisely, because it is a defect SHAPE rather
than a typo. Since 0.46.0 a fresh `lop` boots as a viewer over a
``RemoteSession``, so a command scoped ``authoritative_session`` runs on the
owner and comes back as a typed ``SlashResult``. ``kind="noop"`` is a legal and
useful answer: it means *the invoking terminal hosts this interaction itself*,
the way bare ``/model`` opens its own picker up front. The contract is that
every ``noop`` has a counterpart in ``_render_authoritative_slash`` that opens
that surface.

``/team``'s mutating form and every ``/agent`` form returned
``noop {"type": "team_mutate"}`` / ``{"type": "agent_mutate"}`` — and nothing
consumed either. ``_render_authoritative_slash`` returns immediately on a
``noop``, so the command sent no prompt, wrote no transcript row, and printed
no notice. Total silence, which is exactly how the operator reported it
("it just fails silently").

Why the existing audits could not catch it:

* ``tests/unit/session/runtime/test_capability_surface.py`` audits the runtime
  HANDLE surface — whether the owner can answer what the server asks. The owner
  answered fine. The answer had no reader.
* ``tests/unit/tui/test_slash_echo.py`` actually MEASURED this silence and
  described it accurately, then asserted only that the refusal copy promised no
  false retry. The behaviour was known and pinned as a wording property.

So this audits the SEAM instead of either side: every ``data["type"]`` a
producer can emit must be named by the renderer that receives it. It is a
static audit over the real source on both sides, so a new producer added in a
year fails here without anyone remembering this bug.
"""

from __future__ import annotations

import ast
from pathlib import Path

_APP = Path(__file__).resolve().parents[3] / "local_operator" / "tui" / "app.py"
_OWNED = Path(__file__).resolve().parents[3] / "local_operator" / "session" / "runtime" / "owned.py"


def _slash_result_payload_types(source: str) -> set[str]:
    """Every ``type`` string a ``SlashResult(...)`` call can carry.

    Matches the call by NAME rather than by enclosing function, because the
    producers are spread across ``_team_slash``, ``_agent_slash``,
    ``_team_slash_result`` and friends in two modules, and a list of function
    names is the kind of thing that goes stale silently — which is the failure
    mode this whole file exists to prevent.
    """
    tree = ast.parse(source)
    types: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "SlashResult":
            continue
        for keyword in node.keywords:
            if keyword.arg != "data" or not isinstance(keyword.value, ast.Dict):
                continue
            for key, value in zip(keyword.value.keys, keyword.value.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "type"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    types.add(value.value)
                # ``{"type": "agent_mutate" if arg else "agent_list"}`` — the
                # conditional spelling, which is how one of the two silent
                # types was actually written. Both branches count.
                if isinstance(key, ast.Constant) and key.value == "type":
                    if isinstance(value, ast.IfExp):
                        for branch in (value.body, value.orelse):
                            if isinstance(branch, ast.Constant) and isinstance(branch.value, str):
                                types.add(branch.value)
    return types


def _types_named_by_the_renderer(source: str) -> set[str]:
    """Every payload type string ``_render_authoritative_slash`` mentions.

    Deliberately the whole function body rather than only its ``==``
    comparisons: the renderer dispatches on ``block_type ==`` in one place and
    on ``data.get("type") in (...)`` in another, and an audit keyed to one
    spelling would be blind to the other — the same mistake that let the
    original bug through a capability audit.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_render_authoritative_slash":
            return {
                sub.value
                for sub in ast.walk(node)
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            }
    raise AssertionError("_render_authoritative_slash not found in app.py")


def test_every_routed_slash_payload_type_has_a_renderer() -> None:
    """The defect class, as an enforced property.

    A producer emitting a ``data["type"]`` the renderer never names is a
    command that runs on the owner and then evaporates: no output, no error,
    nothing for the user to act on. If this fails, either give the type a
    branch in ``_render_authoritative_slash`` or stop producing it.
    """
    produced = _slash_result_payload_types(_APP.read_text()) | _slash_result_payload_types(
        _OWNED.read_text()
    )
    rendered = _types_named_by_the_renderer(_APP.read_text())
    orphaned = sorted(produced - rendered)
    assert not orphaned, (
        "these routed slash payload types are produced but never rendered, so the "
        f"command silently does nothing on a viewer: {orphaned}. Add a branch in "
        "_render_authoritative_slash or stop emitting the type."
    )


def test_the_two_types_that_shipped_broken_are_covered() -> None:
    """The regression itself, named.

    A property test can be satisfied by deleting the producer, so this pins the
    specific outcome the operator needs: the attach receipts exist and the
    renderer knows them.
    """
    rendered = _types_named_by_the_renderer(_APP.read_text())
    assert "team_attached" in rendered, "/team <name> <request> must render its attach receipt"
    assert "agent_attached" in rendered, "/agent <name> must render its attach receipt"
    produced = _slash_result_payload_types(_OWNED.read_text())
    assert (
        "team_mutate" not in produced
    ), "the owner must not reintroduce the unconsumed team_mutate noop"
    assert (
        "agent_mutate" not in produced
    ), "the owner must not reintroduce the unconsumed agent_mutate noop"


def test_the_extractor_sees_a_type_however_it_is_spelled() -> None:
    """The audit must survive the spellings the real code actually uses.

    Both forms below occur in production today; an extractor that saw only the
    plain constant would have reported ``agent_mutate`` as covered.
    """
    source = (
        "def f():\n"
        "    a = SlashResult(kind='noop', data={'type': 'plain'})\n"
        "    b = SlashResult(kind='noop', data={'type': 'yes' if x else 'no'})\n"
    )
    assert _slash_result_payload_types(source) == {"plain", "yes", "no"}


def test_the_credential_guard_does_not_promise_a_wait_that_never_ends() -> None:
    """The guard-copy half of the regression.

    ``/credential`` answered "session is still starting…" on a session that was
    fully attached and never going to grow a local store. A refusal that names
    a TRANSIENT state for a PERMANENT absence is worse than a plain failure: it
    tells the user to wait for something that is not coming, and the workaround
    they reach for is pasting the secret into the chat — the exact leak the
    command exists to prevent.

    Asserted on the source of ``_cmd_credential`` because the string is the
    defect; the behavioural proof that the routed path works lives in
    ``tests/e2e/test_viewer_attach_e2e.py``.
    """
    tree = ast.parse(_APP.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_cmd_credential":
            strings = {
                sub.value
                for sub in ast.walk(node)
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            }
            assert not any("still starting" in text for text in strings), (
                "/credential must not report a permanent capability absence as a "
                "transient boot state; route the work or say it cannot run"
            )
            return
    raise AssertionError("_cmd_credential not found in app.py")


def test_no_local_config_command_asks_is_remote_directly() -> None:
    """The third shape of this bug, as an enforced property.

    A local-config write must not key on ``is_remote``. That flag answered
    "is there a socket between me and the session", which USED to imply "the
    runtime is someone else's machine" and stopped implying it in 0.46.0, when
    `lop` began building a ``RemoteSession`` for every local user. Every guard
    still asking it refuses the exact person it was written to serve.

    This has now been fixed three times independently, each site-locally:

    * #576 moved ``/model default`` onto ``_session_runs_elsewhere()``;
    * #609 moved the ``/mcp`` grant verbs onto the runtime ("run grant verbs on
      the runtime instead of refusing the local user");
    * this PR moves the model picker's ``d`` key, which #576 missed even though
      it is the OTHER HALF of the same feature, and fixes the cold-viewer case
      ``_session_runs_elsewhere()`` itself got wrong.

    Nothing was left behind by the first two, which is why there was a third.
    This is that artifact: the handlers that persist to this machine's
    ``config.yml`` must all ask the shared predicate, so a fourth instance
    fails here instead of reaching a user.
    """
    tree = ast.parse(_APP.read_text())
    #: Handlers that write this machine's config.yml and must therefore ask
    #: "would this write govern the runtime?" rather than "is there a socket?".
    persisting_handlers = {
        "_persist_default_from_picker",
        "_cmd_model",
    }
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name in persisting_handlers):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "getattr"
                and len(sub.args) >= 2
                and isinstance(sub.args[1], ast.Constant)
                and sub.args[1].value == "is_remote"
            ):
                offenders.append(node.name)
    assert not offenders, (
        f"{sorted(set(offenders))} gate a LOCAL CONFIG WRITE on is_remote, which is "
        "true for every local user since 0.46.0. Ask _session_runs_elsewhere() "
        "instead — see #576 and #609, the two previous fixes of this same bug."
    )


def test_a_cold_viewer_may_still_set_its_default_model() -> None:
    """The cold case, which the predicate itself got wrong.

    A viewer whose runtime has not spawned publishes no discovery record, and
    ``_session_runs_elsewhere`` treated "no proof of local" as "elsewhere". But
    a session with no id has no runtime AT ALL, so there is nothing anywhere for
    the wrong config to govern — and this is the single most common moment a
    user sets a default: open `lop`, open the picker, press ``d``, before
    typing anything.

    Asserted through the real method on a stub shaped like the session `lop`
    actually builds, rather than on source, because the bug is in the branch
    the predicate takes.
    """
    from local_operator.tui.app import OperatorApp

    class _ColdViewer:
        is_remote = True
        session_id = ""

    app = OperatorApp.__new__(OperatorApp)
    app._session = _ColdViewer()  # type: ignore[attr-defined]
    assert app._session_runs_elsewhere() is False, (
        "a cold viewer has no runtime at all, so a config write cannot govern "
        "the wrong machine — refusing here blocks the default-model flow at "
        "the exact moment users reach for it"
    )
