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
