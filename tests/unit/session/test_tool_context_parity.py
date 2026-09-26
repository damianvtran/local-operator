"""Nothing a host hands the Session may vanish before the executor sees it.

Two production bugs had the identical shape. ``session_factory._prepare``
builds a ToolContext with everything the host configured and passes it to
``create_tools``, but that context only ever reaches the createIf checks that
decide which tools EXIST. The context a tool actually runs against is
``Session._build_tool_context()``, rebuilt from a fixed kwarg list at the top
of every turn. A field set on the first and forgotten on the second is
therefore invisible in exactly the way that is hardest to notice: the tool is
advertised, it runs, and it silently answers from a fallback.

- ``variables``: a configured VariableStore reached the createIf check and
  never the executor, so ``list_variables``/``read_variable`` read a bare
  process-environment store in EVERY session.
- ``job_id``: added for subagent approval provenance and dropped the same way
  on its first attempt.

Two occurrences is a pattern, so the guard below is written to fail when a
THIRD field is added the same way rather than to enumerate today's fields.
"""

from __future__ import annotations

import inspect
from typing import Any

from local_operator.harness.types import ModelSpec, ToolContext
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.variables import VariableStore

MODEL = ModelSpec(provider="test", model_id="m", context_window=1000)


def never_streams(request: Any, signal: Any) -> Any:
    """These tests build a Session to inspect its wiring; none runs a turn."""

    async def gen():
        raise AssertionError("no turn is run here")
        yield

    return gen()


#: A distinguishable value per ToolContext field that ``Session.__init__``
#: also accepts BY THE SAME NAME. The pairing is derived, not listed: a new
#: same-named field with no entry here fails
#: :func:`test_every_paired_field_has_a_sentinel` immediately, which is the
#: prompt to add one and thereby get the drop-detection below for free.
SENTINELS: dict[str, Any] = {
    "cwd": "/tmp/sentinel-cwd",
    "session_id": "sentinel-session",
    "agent_id": "sentinel-agent",
    "has_ui": True,
    "job_id": "sentinel-job",
    # The short name a subagent was delegated under. Dropped on the way to the
    # executor, a child's browser tab group falls back to its PARENT's cwd —
    # which every sibling shares — and a fleet of children renders as one
    # repeated pill distinguished only by an ordinal.
    "job_label": "sentinel-job-label",
    "variables": VariableStore(cwd="/tmp", config_values={"SENTINEL_VAR": "1"}),
    "request_approval": lambda tool_name, description: None,
    "resolve_internal_url": lambda url: None,
    # A child is handed its PARENT's comms instance; if that were dropped on
    # the way to the executor the child's hub tool would answer into a private
    # object nobody is waiting on.
    "subagent_comms": object(),
    # The agent registry backs the ``agent`` tool and role resolution for
    # ``task(agent=...)``. Dropped on the way to the executor, the tool would
    # silently see no registry: role lookups would fall back to the packaged
    # starters and every profile the operator authored would be invisible.
    "agent_registry": object(),
    # The team registry backs the ``team`` / ``team_delete`` tools and the
    # ``/team`` slash command. Dropped on the way to the executor, teams would
    # silently vanish from a session that was built with them.
    "team_registry": object(),
    # The projects registry backs the ``project`` / ``project_delete`` tools and
    # the ``/project`` slash command. Dropped on the way to the executor, a
    # session built with a store would advertise neither tool and every
    # ``/project`` surface would answer "unavailable" while the store sat right
    # there on disk.
    "project_registry": object(),
}

#: ToolContext fields the Session takes under a DIFFERENT name. Kept tiny and
#: asserted to be accurate, so a rename cannot leave a dead entry behind. An
#: entry here maps a field into the guard; it never excuses one from it —
#: subtracting ``ALIASES`` from the tripwire would make registering a field
#: the way to opt it OUT, which is how the third occurrence of this bug would
#: have walked straight through.
ALIASES: dict[str, str] = {"resolve_internal_url": "skill_resolver"}


def paired_fields() -> dict[str, str]:
    """ToolContext field -> the ``Session.__init__`` parameter that feeds it."""
    session_params = set(inspect.signature(Session.__init__).parameters)
    pairs: dict[str, str] = {}
    for field in ToolContext.model_fields:
        parameter = ALIASES.get(field, field)
        if parameter in session_params:
            pairs[field] = parameter
    return pairs


def test_the_alias_table_still_describes_real_names() -> None:
    """A dead alias entry would silently drop its field out of the guard."""
    session_params = set(inspect.signature(Session.__init__).parameters)
    for field, parameter in ALIASES.items():
        assert field in ToolContext.model_fields, f"{field} is no longer a ToolContext field"
        assert parameter in session_params, f"Session no longer takes {parameter}"


def test_every_paired_field_has_a_sentinel() -> None:
    """The tripwire. A field added to BOTH ToolContext and ``Session.__init__``
    under one name lands here with no sentinel and fails, which is the moment
    to check it is also plumbed through ``_build_tool_context``."""
    missing = sorted(set(paired_fields()) - set(SENTINELS))
    assert not missing, (
        f"ToolContext fields with no sentinel in this test: {missing}. Add one, "
        "and confirm Session._build_tool_context passes the field through."
    )


def test_no_host_supplied_field_is_dropped_before_the_executor(tmp_path) -> None:
    """Give the Session a distinguishable value for every field it accepts and
    assert the per-turn context still carries it. This is the assertion the
    ``variables`` and ``job_id`` bugs would both have failed."""
    pairs = paired_fields()
    kwargs: dict[str, Any] = {
        pairs[field]: SENTINELS[field] for field in pairs if field in SENTINELS
    }
    # ``yolo`` would legitimately blank the approval gate; the drop under test
    # is the accidental kind, so the session is built with the gate live.
    session = Session(
        model=MODEL,
        stream_fn=never_streams,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["sys"],
        yolo=False,
        **kwargs,
    )
    context = session._build_tool_context()
    dropped = [
        field
        for field in pairs
        if field in SENTINELS and getattr(context, field) is not SENTINELS[field]
    ]
    assert not dropped, (
        f"Session._build_tool_context() does not pass through: {dropped}. The host "
        "configured these and the running tool cannot see them."
    )


def test_the_guard_would_have_caught_the_variables_bug(tmp_path) -> None:
    """Proof the guard is load-bearing rather than tautological: reproduce the
    original defect by blanking the session's held store, and watch the same
    comparison fail."""
    store = SENTINELS["variables"]
    session = Session(
        model=MODEL,
        stream_fn=never_streams,
        tools=[],
        transcript=Transcript(tmp_path / "sess2"),
        system_blocks_provider=lambda: ["sys"],
        variables=store,
    )
    assert session._build_tool_context().variables is store
    session._variables = None  # exactly what the pre-fix code amounted to
    assert session._build_tool_context().variables is not store


# ---------------------------------------------------------------------------
# scratchpad_dir: DERIVED from the transcript, which is why it is not in the table
# ---------------------------------------------------------------------------


def _session_for(tmp_path, transcript_dir) -> Session:
    return Session(
        model=MODEL,
        stream_fn=never_streams,
        tools=[],
        transcript=Transcript(transcript_dir),
        system_blocks_provider=lambda: ["sys"],
    )


def test_scratchpad_dir_follows_the_transcript_directory(tmp_path) -> None:
    """``scratchpad://`` resolves under the session's OWN directory, so the value
    is derived per turn rather than accepted as a constructor argument — a
    derived value cannot be configured by a host and then dropped, which is the
    failure this whole module exists to catch."""
    session = _session_for(tmp_path, tmp_path / "sessions" / "abc123")

    assert session._build_tool_context().scratchpad_dir == str(
        tmp_path / "sessions" / "abc123" / "scratchpad"
    )


def test_scratchpad_dir_is_none_for_an_agent_directory(tmp_path) -> None:
    """``--train`` (and a named agent) keeps its transcript in ``agents/<id>/``,
    which ``AgentRegistry.export_agent`` zips whole and publishes — a scratch
    folder there would ship to strangers. No scratchpad root, and none
    created."""
    transcript_dir = tmp_path / "agents" / "abc123"
    session = _session_for(tmp_path, transcript_dir)

    assert session._build_tool_context().scratchpad_dir is None
    assert not (transcript_dir / "scratchpad").exists()


# ---------------------------------------------------------------------------
# attached_probe: DERIVED from the session's own goal state, which is why it is
# not in the table either — there is no host-supplied value to hand and compare,
# so the sentinel guard cannot see it. It is pinned directly instead, because
# dropping it is not a hypothetical: the field is what the browser flow reads to
# say whether a question can be presented, and a context that lost it would
# silently return every access prompt to the pre-fix text.
# ---------------------------------------------------------------------------


def test_the_attached_probe_reads_the_sessions_live_goal_state(tmp_path) -> None:
    """The field is populated, and it is a LIVE read rather than a snapshot."""
    from local_operator.session.goal import GoalState

    session = _session_for(tmp_path, tmp_path / "sessions" / "attached")
    session._goal_state = GoalState()
    state = {"attached": False}
    session._goal_state.interactive_probe = lambda: state["attached"]

    context = session._build_tool_context()
    assert context.attached_probe is not None
    assert context.attached_probe() is False
    # A context is built once per TURN, so a value captured at build time would
    # freeze the answer for the rest of the turn — the browser flow asks after
    # it has waited.
    state["attached"] = True
    assert context.attached_probe() is True


def test_an_unprobed_session_reads_as_attached(tmp_path) -> None:
    """``None`` probe means attached: a host with a person in front of it.

    This is the direction every uncertain path in this design falls. A wrong
    "attached" costs a parked gate and a late answer; a wrong "unattached" costs
    a turn that gives up on a question the operator was ready to answer.
    """
    from local_operator.session.goal import GoalState

    session = _session_for(tmp_path, tmp_path / "sessions" / "bare")
    session._goal_state = GoalState()

    context = session._build_tool_context()
    assert context.attached_probe is not None
    assert context.attached_probe() is True


# ---------------------------------------------------------------------------
# may_delegate: DERIVED from the live tool inventory (``self._tools``), which is
# why it is not in the table either. It is the field this module's shape of
# failure would bite hardest — a delegating shell that reads False is REFUSED
# with a message telling it to delegate with a tool it holds — so its derivation
# is pinned in the guard's own suite (``tests/unit/test_agent_shell_guard.py``),
# beside the predicate that consumes it, rather than a second time here. Recorded
# in this file only so a reader auditing "what a host hands the Session" does not
# have to work out whether ``may_delegate`` was missed: it is not handed at all.
# ---------------------------------------------------------------------------
