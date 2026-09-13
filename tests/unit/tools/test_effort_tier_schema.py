"""The ``effort`` field of ``task``/``agent`` advertises only CONFIGURED tiers.

The incident: both tools hard-coded ``effort: Literal["lo", "med", "hi"]``
regardless of ``values.subagents.models``. On a machine with no tier
configured, the delegating model read the enum, picked ``effort="hi"``, and
the strict launch path (``SubagentModelUnavailable``) refused it — every value
the schema offered was a guaranteed failure, and nothing told the model that
omitting the field was the one working choice.

These tests pin the class of fix: the schema is built from the live config
(0, 1, 3 tiers; malformed selectors excluded), a zero-tier schema never emits
``enum: []`` (Gemini rejects it), validation refuses an unconfigured tier with
a message naming the working alternative, and a mid-session config edit
re-renders the two tools that carry the field.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.config_watch import ConfigWatcher
from local_operator.harness.subagent import (
    MODEL_CHOICE_MODEL,
    MODEL_CHOICE_OPERATOR,
    configured_effort_tiers,
    describe_effort_tiers,
    effort_tier_rejection,
    read_effort_tier_selectors,
    read_model_choice,
)
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    ToolContext,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.registry import create_tools

THREE = {
    "lo": "openai/gpt-5-mini",
    "med": "openrouter/moonshotai/kimi-k3",
    "hi": "anthropic/claude-opus-5",
}


def write_tiers(
    config_dir: Path,
    models: dict[str, Any] | None,
    *,
    model_choice: Any = MODEL_CHOICE_MODEL,
) -> None:
    """The on-disk shape ``/settings`` writes; ``None`` leaves ``subagents`` unset.

    ``model_choice`` defaults to ``"model"``, and that default is the whole
    reason this parameter exists: every one of these tests describes the arm in
    which tiers ARE advertised, which since the ``subagents.model_choice`` key
    shipped is no longer the default arm. Spelling it here rather than at each
    of the ~20 call sites keeps one fact in one place, and a test that means the
    operator arm passes ``MODEL_CHOICE_OPERATOR`` and says so.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    subagents: dict[str, Any] = {} if models is None else {"models": models}
    if model_choice is not None:
        subagents["model_choice"] = model_choice
    values: dict[str, Any] = {} if not subagents else {"subagents": subagents}
    (config_dir / "config.yml").write_text(yaml.safe_dump({"values": values}))


@pytest.fixture()
def config_dir(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(path))
    return path


def _launcher(label, prompt, *, agent="task", effort=None):
    return f"job:{label}:{effort}"


def _context(tmp_path) -> ToolContext:
    return ToolContext(
        cwd=str(tmp_path),
        session_id="s",
        subagent_launcher=_launcher,
        agent_registry=AgentRegistry(tmp_path / "agents"),
    )


def _schemas(tmp_path) -> dict[str, dict[str, Any]]:
    return {
        t.name: t.parameters for t in create_tools(_context(tmp_path), enabled=["task", "agent"])
    }


def _effort(schema: dict[str, Any], *path: str) -> dict[str, Any] | None:
    node: Any = schema
    for part in path:
        node = node[part]
    return node.get("effort")


def _enum(prop: dict[str, Any]) -> list[str]:
    return prop["anyOf"][0]["enum"]


def _enum_nodes(node: Any) -> list[list[Any]]:
    found: list[list[Any]] = []
    if isinstance(node, dict):
        if "enum" in node:
            found.append(node["enum"])
        for child in node.values():
            found.extend(_enum_nodes(child))
    elif isinstance(node, list):
        for child in node:
            found.extend(_enum_nodes(child))
    return found


# ---------------------------------------------------------------------------
# the shared reader
# ---------------------------------------------------------------------------


def test_no_config_file_means_no_tiers(config_dir) -> None:
    assert configured_effort_tiers() == {}
    # ...and no config file means the restrictive default: the choice belongs
    # to the operator, so nothing may be advertised to the model at all.
    assert read_model_choice() == MODEL_CHOICE_OPERATOR


# ---------------------------------------------------------------------------
# the model-choice reader: fail closed on everything but one exact spelling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "written,expected",
    [
        (None, MODEL_CHOICE_OPERATOR),  # key absent — the shipped default
        ("model", MODEL_CHOICE_MODEL),
        (" model ", MODEL_CHOICE_MODEL),  # lop config edit preserves whitespace
        ("MODEL", MODEL_CHOICE_MODEL),
        ("operator", MODEL_CHOICE_OPERATOR),
        ("modle", MODEL_CHOICE_OPERATOR),  # a typo must gate, not authorise
        ("", MODEL_CHOICE_OPERATOR),
        ("   ", MODEL_CHOICE_OPERATOR),
        (1, MODEL_CHOICE_OPERATOR),
        (True, MODEL_CHOICE_OPERATOR),
        ([], MODEL_CHOICE_OPERATOR),
    ],
)
def test_read_model_choice_fails_closed_on_everything_but_one_spelling(
    config_dir, written, expected
) -> None:
    """Only an exact ``str`` spelling enables model choice.

    The direction is the point. A reader that failed OPEN on an unrecognised
    value would restore exactly the spend this key was added to stop, on
    nothing worse than a typo — silently, and looking like the key working.
    ``1``/``True``/``[]`` are here because YAML produces them for free
    (``model_choice: 1``, ``yes``, an empty list), and they are the shapes a
    truthiness check would have accepted.
    """
    write_tiers(config_dir, None, model_choice=written)
    assert read_model_choice() == expected


def test_an_unrecognised_model_choice_warns_once(config_dir, caplog) -> None:
    """An operator who typed ``modle`` must not be gated in silence.

    Warned once per distinct value rather than per read: this is read on every
    tool build and every spawn, so a per-read warning would bury the TUI in
    repeats of one typo.
    """
    import logging

    from local_operator.harness.subagent import _WARNED_MODEL_CHOICES

    _WARNED_MODEL_CHOICES.discard("modle")
    write_tiers(config_dir, None, model_choice="modle")
    with caplog.at_level(logging.WARNING, logger="local_operator.harness.subagent"):
        for _ in range(3):
            assert read_model_choice() == MODEL_CHOICE_OPERATOR
    assert sum("model_choice" in record.message for record in caplog.records) == 1


def test_a_recognised_model_choice_never_warns(config_dir, caplog) -> None:
    import logging

    write_tiers(config_dir, None, model_choice="operator")
    with caplog.at_level(logging.WARNING, logger="local_operator.harness.subagent"):
        read_model_choice()
    assert not [r for r in caplog.records if "model_choice" in r.message]


def test_three_tiers_come_back_in_canonical_order(config_dir) -> None:
    # YAML order is hi/lo/med here on purpose: the schema must not follow it,
    # or a reordering edit would move the enum and bust the prompt cache.
    write_tiers(config_dir, {"hi": THREE["hi"], "lo": THREE["lo"], "med": THREE["med"]})
    assert list(configured_effort_tiers()) == ["lo", "med", "hi"]
    assert configured_effort_tiers() == THREE


def test_malformed_selectors_are_excluded(config_dir) -> None:
    write_tiers(
        config_dir,
        {
            "lo": "just-a-model",  # no provider
            "med": "   ",  # blank
            "hi": THREE["hi"],
        },
    )
    assert configured_effort_tiers() == {"hi": THREE["hi"]}


def test_a_tier_outside_the_registry_set_is_inert_in_every_consumer(config_dir) -> None:
    """``subagents.models`` is a free mapping in YAML, but the watcher's
    per-registry-key diff can only report lo/med/hi — a hand-added ``xl``
    would be read and advertised yet never trigger the live schema re-render,
    so the schema would promise a tier whose edits the rebuild cannot reach.

    The narrowing therefore lives in the shared reader, and this asserts all
    three consumers agree: dropping it in the schema alone would leave ``xl``
    unadvertised but launchable through a role pin, and would refuse it as
    'lacks provider/model' — false, since the selector is well-formed."""
    write_tiers(config_dir, {"hi": THREE["hi"], "xl": "openai/gpt-5"})

    assert read_effort_tier_selectors() == {"hi": THREE["hi"]}
    assert configured_effort_tiers() == {"hi": THREE["hi"]}
    # Refused as absent, not as malformed — and it names the tier that works.
    rejection = effort_tier_rejection("xl")
    assert rejection is not None
    assert "not configured at subagents.models.xl" in rejection
    assert "lacks provider/model" not in rejection


def test_whitespace_padded_selectors_are_stripped_once(config_dir) -> None:
    """``lop config edit`` preserves surrounding whitespace verbatim. The
    strip lives in the shared reader so the schema and the strict launch
    path see the SAME value: before this, the schema stripped and launch
    did not, so a padded selector was advertised, passed the strict check,
    and failed on the first provider call with ``provider='  openai'``."""
    write_tiers(config_dir, {"lo": "  openai/gpt-5-mini  "})
    assert configured_effort_tiers() == {"lo": "openai/gpt-5-mini"}


def test_yaml_coerced_non_string_tier_keys_cannot_break_the_schema(config_dir, tmp_path) -> None:
    """The exact F1 repro: YAML parses ``1:``/``on:``/``yes:`` as int/bool
    keys, and a mixed-type ``sorted()`` on them raised ``TypeError`` OUTSIDE
    the reader's ``try`` — through ``create_tools`` (a session could not
    boot) and through ``TaskParams(effort=...)`` as a non-ValidationError.
    Non-string keys are dropped rather than stringified: ``1: openai/…`` is
    a YAML accident, not a tier anyone can ask for by name."""
    config_dir.mkdir(parents=True, exist_ok=True)
    # Written as YAML text rather than safe_dump: the point is YAML's silent
    # key coercion (``1:`` → int, ``on:`` → bool True), and a dict literal
    # with both would trip F601 (``1 == True`` hashes equal).
    (config_dir / "config.yml").write_text(
        "values:\n  subagents:\n    model_choice: model\n    models:\n"
        f"      1: openai/gpt-5-mini\n      on: x/y\n      hi: {THREE['hi']}\n"
    )
    assert configured_effort_tiers() == {"hi": THREE["hi"]}

    # Schema construction — the session-boot path — must survive it, and the
    # surviving tier must still be accepted as a plain ValidationError-checked
    # argument rather than escaping as a TypeError.
    from local_operator.tools.builtin import TaskParams

    TaskParams(label="r", prompt="p", effort="hi")
    task = next(t for t in create_tools(_context(tmp_path), enabled=["task"]))
    assert _enum(task.parameters["properties"]["effort"]) == ["hi"]


def test_an_unreadable_config_reports_no_tiers_rather_than_raising(
    config_dir, monkeypatch, tmp_path
) -> None:
    """Schema construction runs while the tool inventory is being built; a
    corrupt config.yml must cost the operator a tier picker, not a session.
    The read failure is SYNTHESIZED here: ``ConfigManager`` on a corrupt
    file moves it aside and returns defaults rather than raising, so only a
    monkeypatched reader actually exercises the except branch the docstring
    promises."""
    import local_operator.harness.subagent as subagent_mod

    def boom() -> dict[str, Any]:
        raise OSError("config.yml is on fire")

    monkeypatch.setattr(subagent_mod, "read_effort_tier_selectors", boom)
    assert configured_effort_tiers() == {}
    # And the failure must not reach the tool inventory either.
    task = next(t for t in create_tools(_context(tmp_path), enabled=["task"]))
    assert "effort" not in task.parameters["properties"]


def test_describe_names_the_model_behind_each_tier() -> None:
    assert describe_effort_tiers({"lo": "a/b", "hi": "c/d"}) == "lo → a/b, hi → c/d"


def test_rejection_names_the_working_alternative(config_dir) -> None:
    assert effort_tier_rejection("hi") is not None
    assert "no tiers are configured" in str(effort_tier_rejection("hi"))
    assert "omit 'effort'" in str(effort_tier_rejection("hi"))
    write_tiers(config_dir, {"lo": THREE["lo"]})
    assert effort_tier_rejection("lo") is None
    message = effort_tier_rejection("hi")
    assert message is not None
    assert "not configured at subagents.models.hi" in message
    assert "lo → openai/gpt-5-mini" in message
    # Present but unusable is named as such, in the launch path's own words.
    write_tiers(config_dir, {"lo": THREE["lo"], "hi": "gpt-5"})
    assert "subagents.models.hi='gpt-5' lacks provider/model" in str(effort_tier_rejection("hi"))


# ---------------------------------------------------------------------------
# the task tool schema
# ---------------------------------------------------------------------------


def test_zero_tiers_drops_the_field_from_both_task_forms(config_dir, tmp_path) -> None:
    """No tier means no enum to pick from: the property is REMOVED, from the
    single form and the batch item alike, and no ``enum: []`` exists anywhere
    in the schema (Gemini function declarations reject an empty enum)."""
    write_tiers(config_dir, None, model_choice=MODEL_CHOICE_MODEL)
    task = _schemas(tmp_path)["task"]

    assert _effort(task, "properties") is None
    assert _effort(task, "$defs", "TaskItem", "properties") is None
    assert all(members for members in _enum_nodes(task)), "an empty enum leaked"


def test_zero_tiers_tells_the_model_not_to_pass_effort(config_dir, tmp_path) -> None:
    write_tiers(config_dir, None, model_choice=MODEL_CHOICE_MODEL)
    (task,) = create_tools(_context(tmp_path), enabled=["task"])
    assert "No effort tiers are configured" in task.description
    assert "inherits this session's model and reasoning effort" in task.description
    assert "do not pass 'effort'" in task.description


def test_one_tier_is_the_whole_enum_and_names_its_model(config_dir, tmp_path) -> None:
    write_tiers(config_dir, {"hi": THREE["hi"]})
    task = _schemas(tmp_path)["task"]

    single = _effort(task, "properties")
    item = _effort(task, "$defs", "TaskItem", "properties")
    assert single is not None and item is not None
    assert _enum(single) == ["hi"]
    assert _enum(item) == ["hi"]
    assert "hi → anthropic/claude-opus-5" in single["description"]
    assert "Omit to inherit" in single["description"]
    # The nullable shape pydantic renders for ``Literal | None`` is kept, so
    # providers see the field they always saw with different members.
    assert single["anyOf"][1] == {"type": "null"}


def test_three_tiers_list_all_three_with_their_models(config_dir, tmp_path) -> None:
    write_tiers(config_dir, THREE)
    (task,) = create_tools(_context(tmp_path), enabled=["task"])
    single = _effort(task.parameters, "properties")
    assert single is not None
    assert _enum(single) == ["lo", "med", "hi"]
    for tier, selector in THREE.items():
        assert f"{tier} → {selector}" in single["description"]
    assert "set it to one of the configured tiers" in task.description


def test_a_malformed_tier_is_not_advertised(config_dir, tmp_path) -> None:
    write_tiers(config_dir, {"lo": "no-provider", "hi": THREE["hi"]})
    single = _effort(_schemas(tmp_path)["task"], "properties")
    assert single is not None
    assert _enum(single) == ["hi"]


# ---------------------------------------------------------------------------
# validation: what the schema does not offer, the tool refuses — helpfully
# ---------------------------------------------------------------------------


async def _call(tmp_path, name: str, args: dict[str, Any]):
    context = _context(tmp_path)
    (tool,) = create_tools(context, enabled=[name])
    return await tool.execute("call-1", args, None, None, context)


@pytest.mark.asyncio
async def test_task_refuses_an_unconfigured_tier_with_zero_tiers(config_dir, tmp_path) -> None:
    # ``model_choice=model`` because this is the arm that advertises the field:
    # under the shipped default there is no enum for this refusal to be about.
    write_tiers(config_dir, None, model_choice=MODEL_CHOICE_MODEL)
    result = await _call(tmp_path, "task", {"label": "r", "prompt": "p", "effort": "hi"})
    assert result.is_error
    assert "invalid arguments" in result.text
    assert "effort tier 'hi' is unavailable" in result.text
    assert "no tiers are configured under values.subagents.models" in result.text
    assert "omit 'effort' to inherit this session's model and reasoning effort" in result.text
    # Refused at validation: the launcher was never reached.
    assert "job:" not in result.text


@pytest.mark.asyncio
async def test_task_batch_item_is_validated_the_same_way(config_dir, tmp_path) -> None:
    write_tiers(config_dir, {"lo": THREE["lo"]})
    result = await _call(
        tmp_path,
        "task",
        {"context": "c", "tasks": [{"label": "r", "prompt": "p", "effort": "hi"}]},
    )
    assert result.is_error
    assert "tasks.0.effort" in result.text
    assert "not configured at subagents.models.hi" in result.text
    assert "lo → openai/gpt-5-mini" in result.text


@pytest.mark.asyncio
async def test_task_accepts_a_configured_tier_and_omission(config_dir, tmp_path) -> None:
    write_tiers(config_dir, {"hi": THREE["hi"]})
    ok = await _call(tmp_path, "task", {"label": "r", "prompt": "p", "effort": "hi"})
    assert not ok.is_error and "job:r:hi" in ok.text
    inherit = await _call(tmp_path, "task", {"label": "r", "prompt": "p"})
    assert not inherit.is_error and "job:r:None" in inherit.text


# ---------------------------------------------------------------------------
# the agent tool mirrors it, keeping ``inherit`` and the legacy "" spelling
# ---------------------------------------------------------------------------


def test_agent_schema_lists_configured_tiers_then_inherit(config_dir, tmp_path) -> None:
    write_tiers(config_dir, THREE)
    prop = _effort(_schemas(tmp_path)["agent"], "properties")
    assert prop is not None
    assert _enum(prop) == ["lo", "med", "hi", "inherit"]
    assert "hi → anthropic/claude-opus-5" in prop["description"]


def test_agent_schema_with_zero_tiers_keeps_only_inherit(config_dir, tmp_path) -> None:
    write_tiers(config_dir, None, model_choice=MODEL_CHOICE_MODEL)
    prop = _effort(_schemas(tmp_path)["agent"], "properties")
    assert prop is not None
    assert _enum(prop) == ["inherit"]
    assert "no model tiers are configured" in prop["description"]


@pytest.mark.asyncio
async def test_agent_refuses_an_unconfigured_pin_and_accepts_inherit(config_dir, tmp_path) -> None:
    write_tiers(config_dir, {"lo": THREE["lo"]})
    refused = await _call(
        tmp_path,
        "agent",
        {"op": "create", "name": "r1", "description": "d", "instructions": "i", "effort": "hi"},
    )
    assert refused.is_error
    assert "effort tier 'hi' is unavailable" in refused.text
    for value in ("inherit", ""):
        ok = await _call(
            tmp_path,
            "agent",
            {
                "op": "create",
                "name": f"r-{value or 'legacy'}",
                "description": "d",
                "instructions": "i",
                "effort": value,
            },
        )
        assert not ok.is_error, ok.text


# ---------------------------------------------------------------------------
# a mid-session edit re-renders the schema the model reads
# ---------------------------------------------------------------------------


class _Stream:
    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        async def gen():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def _session_effort_enum(session: Session) -> list[str] | None:
    task = next(tool for tool in session._tools if tool.name == "task")
    prop = task.parameters["properties"].get("effort")
    return None if prop is None else _enum(prop)


@pytest.mark.asyncio
async def test_configuring_a_tier_mid_session_reaches_the_task_schema(config_dir, tmp_path) -> None:
    """Tools are built once at session construction, so a tier configured
    afterwards would stay invisible for the life of the session without the
    live re-render — and a tier removed would stay advertised, which is the
    original incident with extra steps."""
    ConfigManager(config_dir).set_config_value("hosting", "")  # a real file to diff against
    session = Session(
        model=ModelSpec(provider="test", model_id="m", context_window=1000, max_output_tokens=100),
        stream_fn=_Stream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )
    watcher = ConfigWatcher(config_dir)
    session.add_dispose_hook(watcher.subscribe(session._apply_config_change))
    try:
        order_before = [tool.name for tool in session._tools]
        assert _session_effort_enum(session) is None

        # BOTH keys, in one write: with the shipped `model_choice=operator`
        # default a tier alone advertises nothing, so a test that wrote only
        # `models` here would assert the watcher works against a schema that
        # has no field to move.
        ConfigManager(config_dir).set_config_value(
            "subagents", {"model_choice": MODEL_CHOICE_MODEL, "models": {"hi": THREE["hi"]}}
        )
        change = watcher.poll_now()
        assert change is not None and "subagents.models.hi" in change.changed_keys
        assert _session_effort_enum(session) == ["hi"]
        # Replacement in place: the provider-visible order is part of the
        # prompt-cache prefix and must not move.
        assert [tool.name for tool in session._tools] == order_before

        ConfigManager(config_dir).set_config_value(
            "subagents", {"model_choice": MODEL_CHOICE_MODEL}
        )
        watcher.poll_now()
        assert _session_effort_enum(session) is None
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_handing_the_choice_back_mid_session_removes_the_field(config_dir, tmp_path) -> None:
    """``subagents.model_choice`` must fire the rebuild ON ITS OWN.

    This is the failure the registry makes possible and the watcher diff cannot
    see: ``changed`` holds REGISTRY keys only, so a key that existed in the
    reader but not in ``settings_io.SETTINGS`` would never appear there and an
    operator flipping it mid-session would keep a tool that still offered — and
    still accepted — a tier they had just taken back. Asserted against the
    TOOL-BUILD product (the rendered schema), not against the reader: the
    reader is live by construction and would pass even with no rebuild at all.
    """
    ConfigManager(config_dir).set_config_value(
        "subagents", {"model_choice": MODEL_CHOICE_MODEL, "models": {"hi": THREE["hi"]}}
    )
    session = Session(
        model=ModelSpec(provider="test", model_id="m", context_window=1000, max_output_tokens=100),
        stream_fn=_Stream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )
    watcher = ConfigWatcher(config_dir)
    session.add_dispose_hook(watcher.subscribe(session._apply_config_change))
    try:
        assert _session_effort_enum(session) == ["hi"]

        ConfigManager(config_dir).set_config_value(
            "subagents", {"model_choice": MODEL_CHOICE_OPERATOR, "models": {"hi": THREE["hi"]}}
        )
        change = watcher.poll_now()
        assert change is not None and "subagents.model_choice" in change.changed_keys
        assert _session_effort_enum(session) is None, "the tier is still offered"
        # And the tier is still CONFIGURED — the field went because the operator
        # took the choice back, not because the tier vanished.
        assert configured_effort_tiers() == {"hi": THREE["hi"]}

        ConfigManager(config_dir).set_config_value(
            "subagents", {"model_choice": MODEL_CHOICE_MODEL, "models": {"hi": THREE["hi"]}}
        )
        watcher.poll_now()
        assert _session_effort_enum(session) == ["hi"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_shipped_default_advertises_no_tier_mid_session_change(
    config_dir, tmp_path
) -> None:
    """A tier configured while ``model_choice`` is ``operator`` stays unadvertised.

    The half-fix this guards: firing the rebuild on ``models.*`` and rendering
    the members without consulting the policy would re-advertise every tier the
    operator had just taken back, every time they touched a selector.
    """
    ConfigManager(config_dir).set_config_value("subagents", {"model_choice": MODEL_CHOICE_OPERATOR})
    session = Session(
        model=ModelSpec(provider="test", model_id="m", context_window=1000, max_output_tokens=100),
        stream_fn=_Stream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )
    watcher = ConfigWatcher(config_dir)
    session.add_dispose_hook(watcher.subscribe(session._apply_config_change))
    try:
        assert _session_effort_enum(session) is None
        ConfigManager(config_dir).set_config_value(
            "subagents", {"model_choice": MODEL_CHOICE_OPERATOR, "models": {"hi": THREE["hi"]}}
        )
        watcher.poll_now()
        assert configured_effort_tiers() == {"hi": THREE["hi"]}
        assert _session_effort_enum(session) is None
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_child_whose_task_tool_was_pruned_does_not_get_it_back(
    config_dir, tmp_path
) -> None:
    """The rebuild replaces tools already present and never appends: a
    non-delegating child had ``task`` pruned on purpose."""
    ConfigManager(config_dir).set_config_value("hosting", "")
    session = Session(
        model=ModelSpec(provider="test", model_id="m", context_window=1000, max_output_tokens=100),
        stream_fn=_Stream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )
    watcher = ConfigWatcher(config_dir)
    session.add_dispose_hook(watcher.subscribe(session._apply_config_change))
    try:
        session.refresh_tools([tool for tool in session._tools if tool.name != "task"])
        ConfigManager(config_dir).set_config_value(
            "subagents", {"model_choice": MODEL_CHOICE_MODEL, "models": {"hi": THREE["hi"]}}
        )
        watcher.poll_now()
        assert "task" not in {tool.name for tool in session._tools}
    finally:
        await session.dispose()
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# ``subagents.model_choice=operator``: the choice is not the model's to make
# ---------------------------------------------------------------------------


def test_operator_mode_removes_the_field_from_both_task_forms(config_dir, tmp_path) -> None:
    """The shipped default advertises no tier on ``task`` — with tiers CONFIGURED.

    That is the whole difference from the zero-tier arm: the tiers are there and
    launchable (the operator launches them, and a role may pin one), they are
    simply not the delegating model's to pick. The property is removed rather
    than advertised empty for the same reason as before — Gemini function
    declarations reject ``enum: []``.
    """
    write_tiers(config_dir, THREE, model_choice=MODEL_CHOICE_OPERATOR)
    task = _schemas(tmp_path)["task"]

    assert _effort(task, "properties") is None
    assert _effort(task, "$defs", "TaskItem", "properties") is None
    assert all(members for members in _enum_nodes(task)), "an empty enum leaked"
    # ...and the tiers really are configured, so this is the policy at work and
    # not an empty selector map.
    assert configured_effort_tiers() == THREE


def test_operator_mode_tells_the_model_not_to_pass_effort(config_dir, tmp_path) -> None:
    write_tiers(config_dir, THREE, model_choice=MODEL_CHOICE_OPERATOR)
    (task,) = create_tools(_context(tmp_path), enabled=["task"])
    assert "do not pass 'effort'" in task.description
    assert "values.subagents.model_choice=operator" in task.description
    assert "A role may pin its own model." in task.description


def test_operator_mode_leaves_the_agent_tool_only_the_sentinel(config_dir, tmp_path) -> None:
    """``agent`` keeps ``effort`` alive for one reason: to clear a pin.

    A model with no way to spell "no pin" cannot undo one, so the enumerations
    differ on purpose between the two tools — while neither offers a tier.
    """
    write_tiers(config_dir, THREE, model_choice=MODEL_CHOICE_OPERATOR)
    prop = _effort(_schemas(tmp_path)["agent"], "properties")
    assert prop is not None
    assert _enum(prop) == ["inherit"]
    assert "no effort tiers are yours to choose" in prop["description"]
    assert "operator sets tier pins" in prop["description"]


def test_operator_mode_with_zero_tiers_says_the_same_thing(config_dir, tmp_path) -> None:
    """Same copy in both zero-tier states: the sentence a model reads must not
    depend on a tier count it cannot see."""
    write_tiers(config_dir, None, model_choice=MODEL_CHOICE_MODEL)
    (with_tiers,) = create_tools(_context(tmp_path), enabled=["task"])
    write_tiers(config_dir, None, model_choice=MODEL_CHOICE_OPERATOR)
    (operator_mode,) = create_tools(_context(tmp_path), enabled=["task"])

    assert "No effort tiers are yours to choose" in operator_mode.description
    assert "do not pass 'effort'" in operator_mode.description
    assert operator_mode.description != with_tiers.description


def test_operator_mode_drops_the_model_choice_description_entirely(config_dir, tmp_path) -> None:
    """The field's own description cannot leak into the schema it was removed from."""
    write_tiers(config_dir, THREE, model_choice=MODEL_CHOICE_OPERATOR)
    (task,) = create_tools(_context(tmp_path), enabled=["task"])
    assert "Swaps this child's MODEL" not in json.dumps(task.parameters)


def test_model_mode_names_the_swap_in_its_description(config_dir, tmp_path) -> None:
    write_tiers(config_dir, {"hi": THREE["hi"]}, model_choice=MODEL_CHOICE_MODEL)
    prop = _effort(_schemas(tmp_path)["task"], "properties")
    assert prop is not None
    assert prop["description"] == (
        "Swaps this child's MODEL (not its reasoning level): hi → anthropic/claude-opus-5. "
        "Omit to inherit this session's model and reasoning effort."
    )


@pytest.mark.asyncio
async def test_operator_mode_refuses_a_tier_before_any_launch(config_dir, tmp_path) -> None:
    """Refused at validation, so no job row exists and no launcher ran."""
    write_tiers(config_dir, THREE, model_choice=MODEL_CHOICE_OPERATOR)
    result = await _call(tmp_path, "task", {"label": "r", "prompt": "p", "effort": "hi"})

    assert result.is_error
    assert "invalid arguments" in result.text
    assert "effort is the operator's to choose" in result.text
    assert "'hi' would run it on anthropic/claude-opus-5 instead of this session's model" in (
        result.text
    )
    assert "Relaunch without 'effort'" in result.text
    assert "job:" not in result.text


@pytest.mark.asyncio
async def test_operator_mode_omits_the_model_name_for_an_unconfigured_tier(
    config_dir, tmp_path
) -> None:
    write_tiers(config_dir, None, model_choice=MODEL_CHOICE_OPERATOR)
    result = await _call(tmp_path, "task", {"label": "r", "prompt": "p", "effort": "hi"})
    assert result.is_error
    assert "effort is the operator's to choose" in result.text
    assert "would run it on" not in result.text


@pytest.mark.asyncio
async def test_operator_mode_batch_item_is_refused_with_its_own_locator(
    config_dir, tmp_path
) -> None:
    write_tiers(config_dir, THREE, model_choice=MODEL_CHOICE_OPERATOR)
    result = await _call(
        tmp_path,
        "task",
        {"context": "c", "tasks": [{"label": "r", "prompt": "p", "effort": "hi"}]},
    )
    assert result.is_error
    assert "tasks.0.effort" in result.text
    assert "effort is the operator's to choose" in result.text
    assert "job:" not in result.text


@pytest.mark.asyncio
async def test_operator_mode_refuses_a_role_pin_and_accepts_the_sentinel(
    config_dir, tmp_path
) -> None:
    write_tiers(config_dir, THREE, model_choice=MODEL_CHOICE_OPERATOR)
    refused = await _call(
        tmp_path,
        "agent",
        {"op": "create", "name": "r1", "description": "d", "instructions": "i", "effort": "hi"},
    )
    assert refused.is_error
    assert "effort is the operator's to choose" in refused.text
    assert "a pin runs that role on a different MODEL ('hi' → anthropic/claude-opus-5)" in (
        refused.text
    )
    assert "pass 'inherit' to clear an existing pin" in refused.text

    for value in ("inherit", ""):
        ok = await _call(
            tmp_path,
            "agent",
            {
                "op": "create",
                "name": f"r-{value or 'legacy'}",
                "description": "d",
                "instructions": "i",
                "effort": value,
            },
        )
        assert not ok.is_error, ok.text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [MODEL_CHOICE_OPERATOR, MODEL_CHOICE_MODEL])
async def test_inherit_is_a_synonym_for_omission_in_both_modes(config_dir, tmp_path, mode) -> None:
    """The ``agent`` token a model sees in its own inventory must not be a trap.

    ``task(effort="inherit")`` means what omission means — the child inherits —
    in BOTH modes, because a model copying the sentinel out of the ``agent``
    schema sits one field away from the ``task`` schema and reads one tool's
    vocabulary as the harness's.
    """
    tiers = {"hi": THREE["hi"]} if mode == MODEL_CHOICE_MODEL else {}
    write_tiers(config_dir, tiers, model_choice=mode)
    named = await _call(tmp_path, "task", {"label": "r", "prompt": "p", "effort": "inherit"})
    omitted = await _call(tmp_path, "task", {"label": "r", "prompt": "p"})

    assert not named.is_error, named.text
    assert named.text == omitted.text
    assert "job:r:None" in named.text, "the sentinel must reach the launcher as no tier at all"
