"""Stable agent facts must not infer semantics from opaque display text."""

import json
from dataclasses import FrozenInstanceError

import pytest
from rich.cells import cell_len
from rich.style import Style

from local_operator.agent_profiles import (
    AgentFact,
    agent_fact_message,
    agent_fact_record,
    agent_listing_facts,
    agent_listing_rows,
)
from local_operator.i18n import text as i18n_text
from local_operator.i18n.messages import Msg
from local_operator.tui.widgets.settings_view import (
    _paint_facts,
    _paint_facts_lines,
    _paint_team_facts_lines,
)

MUTED = Style(color="white")
FAINT = Style(color="grey50")


def raw_display(text):
    """Measure source payload cells, not render-only bidi control codepoints."""
    return text.translate(dict.fromkeys(map(ord, "\u2066\u2067\u2068\u2069")))


@pytest.mark.parametrize("locale", ["en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"])
@pytest.mark.parametrize("kind", ["role", "specialist"])
def test_actual_audited_locales_keep_whole_markers_in_shared_budget(locale, kind):
    from local_operator.i18n import catalogues

    i18n_text.set_active_locale(locale)
    assert i18n_text.active_locale() == locale
    facts = (
        AgentFact("kind", kind),
        AgentFact("source", "packaged"),
        AgentFact("action_class", "proactive"),
        AgentFact("scope", "session"),
        AgentFact("model", "proactive · opaque-model"),
        AgentFact("effort", "custom-tier"),
    )
    records = [agent_fact_record(fact) for fact in facts]
    assert (
        records[1].text
        == catalogues.load_catalogue(locale, "wire.agent_badges")["wire.agent_badges.packaged"]
    )
    assert records[1].source_locale == locale
    lines = _paint_facts_lines(facts, 27, muted=MUTED, faint=FAINT)
    bright = [raw_display(text).strip() for line in lines for text, style in line if style == MUTED]
    assert records[2].text in bright, "the entire actual proactive marker must survive"
    assert records[3].text in bright, "the entire actual scope label must survive"
    assert all(cell_len(raw_display(text)) <= 31 for text in plain(lines))
    assert all(raw_display(text).startswith("    ") for text in plain(lines))
    for text in plain(lines):
        assert text.count("\u2066") + text.count("\u2067") == text.count("\u2069")
    # Fitting never mutates technical data or introduces terminal controls into wire strings.
    assert agent_fact_record(facts[-2]).text == "proactive · opaque-model"
    assert agent_fact_record(facts[-1]).text.endswith("custom-tier")


def test_legacy_agent_text_does_not_infer_proactive_from_model_id() -> None:
    line = _paint_facts("role · proactive", 27, muted=MUTED, faint=FAINT)
    assert all(style == FAINT for _, style in line), line


def test_legacy_agent_text_does_not_extract_model_named_like_scope() -> None:
    lines = _paint_facts_lines(
        "role · this conversation · model-too-long", 27, muted=MUTED, faint=FAINT
    )
    assert len(lines) == 1, lines


def plain(lines):
    return ["".join(text for text, _ in line) for line in lines]


@pytest.fixture(autouse=True)
def english():
    before = i18n_text.active_locale()
    i18n_text.set_active_locale("en")
    yield
    i18n_text.set_active_locale(before)


@pytest.mark.parametrize("model", ["proactive", "this conversation", "role · proactive"])
def test_opaque_model_never_acquires_style_or_scope(model):
    facts = (AgentFact("kind", "role"), AgentFact("model", model))
    lines = _paint_facts_lines(facts, 80, muted=MUTED, faint=FAINT)
    assert plain(lines) == ["    role · " + model]
    assert all(style == FAINT for line in lines for _, style in line)
    # A narrow model may truncate, but cannot be lifted into a semantic line.
    assert len(_paint_facts_lines(facts, 10, muted=MUTED, faint=FAINT)) == 1


def test_fact_is_immutable_and_unknown_codes_are_raw():
    fact = AgentFact("kind", "a-future-kind")
    with pytest.raises(FrozenInstanceError):
        setattr(fact, "value", "role")
    for fact in (fact, AgentFact("unknown", "proactive"), AgentFact("model", "role")):
        assert agent_fact_message(fact) == fact.value
        assert agent_fact_record(fact).source_locale is None
        line = _paint_facts((fact,), 80, muted=MUTED, faint=FAINT)
        assert all(style == FAINT for _, style in line)
    effort_message = agent_fact_message(AgentFact("effort", "custom-tier"))
    assert isinstance(effort_message, Msg)
    assert effort_message.params == {"value": "custom-tier"}


def test_single_enumeration_seed_suppression_and_string_wire_shape(tmp_path, monkeypatch):
    from local_operator.agents import AgentRegistry
    from tests.unit.tui.test_agent_class_slash import _fields

    registry = AgentRegistry(tmp_path)
    registry.create_agent(
        _fields(name="reviewer", model="proactive", tags=["role", "class:reactive"])
    )
    calls = []
    original = registry.list_agents

    def listed():
        calls.append(True)
        return original()

    monkeypatch.setattr(registry, "list_agents", listed)
    rows = agent_listing_facts(registry)
    assert len(calls) == 1
    installed = next(row for row in rows if row[0] == "reviewer")
    assert installed[2] == (AgentFact("kind", "role"), AgentFact("model", "proactive"))
    assert any(AgentFact("source", "packaged") in row[2] for row in rows)
    assert all(AgentFact("source", "packaged") in row[2] for row in agent_listing_facts(None))
    calls.clear()
    strings = agent_listing_rows(registry)
    assert len(calls) == 1
    assert all(len(row) == 4 and all(isinstance(slot, str) for slot in row) for row in strings)
    triples = [(name, facts, summary) for name, _, facts, summary in strings]
    payload = json.loads(json.dumps({"type": "agent_list", "items": triples}))
    assert payload["type"] == "agent_list"
    assert all(
        len(row) == 3 and all(isinstance(slot, str) for slot in row) for row in payload["items"]
    )
    serialized = json.dumps(payload, ensure_ascii=False)
    assert not any(char in serialized for char in "\u2066\u2067\u2068\u2069")


def test_team_legacy_scope_geometry_and_ink_are_retained():
    facts = "team · proactive · this conversation · alias: roster"
    team = _paint_team_facts_lines(facts, 27, muted=MUTED, faint=FAINT)
    assert len(team) == 2
    assert team[1] == [("    this conversation", MUTED)]
    assert ("proactive", MUTED) in team[0]
    # The same legacy string in the agent domain is opaque, never parsed.
    agent = _paint_facts_lines(facts, 27, muted=MUTED, faint=FAINT)
    assert len(agent) == 1
    assert all(style == FAINT for _, style in agent[0])


def test_whole_safety_and_scope_markers_share_one_27_cell_budget(monkeypatch):
    facts = (
        AgentFact("kind", "role"),
        AgentFact("source", "packaged"),
        AgentFact("action_class", "proactive"),
        AgentFact("scope", "session"),
        AgentFact("model", "very-long-model-id"),
        AgentFact("effort", "high"),
    )
    lines = _paint_facts_lines(facts, 27, muted=MUTED, faint=FAINT)
    assert ("proactive", MUTED) in lines[0]
    assert lines[1] == [("    this conversation", MUTED)]
    assert all(cell_len(line) <= 31 for line in plain(lines))
    assert all(line.startswith("    ") and not line.startswith("     ") for line in plain(lines))
    # Synthetic expansion of a reused label: safety spills instead of clipping.
    import local_operator.tui.widgets.settings_view as view

    original = view.agent_fact_record
    from local_operator.i18n.messages import RenderedMessage

    def expanded_record(fact):
        return RenderedMessage("K" * 24, "en") if fact.code == "kind" else original(fact)

    monkeypatch.setattr(view, "agent_fact_record", expanded_record)
    expanded = _paint_facts_lines(facts, 27, muted=MUTED, faint=FAINT)
    assert expanded[1] == [("    proactive", MUTED)]
    assert expanded[2] == [("    this conversation", MUTED)]
    assert all(cell_len(line) <= 31 for line in plain(expanded))


def test_spilling_duplicate_facts_projects_only_first_occurrences(monkeypatch):
    import local_operator.tui.widgets.settings_view as view
    from local_operator.i18n.messages import RenderedMessage

    original = view.agent_fact_record

    def expanded_record(fact):
        return RenderedMessage("K" * 24, "en") if fact.code == "kind" else original(fact)

    monkeypatch.setattr(view, "agent_fact_record", expanded_record)
    kind = AgentFact("kind", "role")
    scope = AgentFact("scope", "session")
    marker = AgentFact("action_class", "proactive")
    model = AgentFact("model", "opaque · proactive")
    facts = (kind, scope, model, marker, scope, marker)
    before = tuple(facts)
    painted_facts = []
    original_paint = view._paint_facts

    def record_paint(items, room, *, muted, faint):
        painted_facts.append(items)
        return original_paint(items, room, muted=muted, faint=faint)

    monkeypatch.setattr(view, "_paint_facts", record_paint)
    # Both first occurrences spill, but equal later facts must remain in order.
    # This pins list.remove semantics, not a remove-all filtering shortcut.
    lines = _paint_facts_lines(facts, 27, muted=MUTED, faint=FAINT)
    assert lines == [
        _paint_facts((kind, model, scope, marker), 27, muted=MUTED, faint=FAINT),
        _paint_facts((marker,), 27, muted=MUTED, faint=FAINT),
        _paint_facts((scope,), 27, muted=MUTED, faint=FAINT),
    ]
    # Inspect the projection too: truncation can hide later duplicate tokens,
    # so pixel equality alone would not catch accidentally removing them all.
    assert painted_facts == [(kind, model, scope, marker), (marker,), (scope,)]
    assert facts == before


def test_urdu_fallback_provenance_and_raw_fit_before_controls(monkeypatch):
    monkeypatch.setattr(i18n_text._locale, "eligible_locale", lambda locale: locale)
    original = i18n_text._sources

    def missing_packaged(root, namespace, locale):
        if locale == "ur" and namespace == "wire.agent_badges":
            return {}
        return original(root, namespace, locale)

    monkeypatch.setattr(i18n_text, "_sources", missing_packaged)
    i18n_text.set_active_locale("ur")
    packaged = AgentFact("source", "packaged")
    assert agent_fact_record(packaged).source_locale == "en"
    assert plain([_paint_facts((packaged,), 5, muted=MUTED, faint=FAINT)]) == ["    pack…"]
    model = AgentFact("model", "proactive · model")
    fitted = plain([_paint_facts((model,), 9, muted=MUTED, faint=FAINT)])[0]
    assert fitted == "    \u2066proactiv…\u2069"


def test_french_reuse_is_live_and_proactive_ink_is_semantic(monkeypatch):
    # Not translations: deterministic fixture tokens prove the paint seam, while
    # independent LLM authors/auditors own the seven real target translations.
    original = i18n_text._sources

    def sources(root, namespace, locale):
        values = original(root, namespace, locale)
        if locale == "fr" and namespace == "tui.authoring":
            return {
                **values,
                "tui.authoring.role": "KIND_FR",
                "tui.authoring.proactive": "CLASS_FR",
            }
        return values

    monkeypatch.setattr(i18n_text, "_sources", sources)
    monkeypatch.setattr(i18n_text._locale, "eligible_locale", lambda locale: locale)
    facts = (
        AgentFact("kind", "role"),
        AgentFact("action_class", "proactive"),
        AgentFact("model", "proactive"),
    )
    for locale, expected in (("en", "proactive"), ("fr", "CLASS_FR"), ("en", "proactive")):
        i18n_text.set_active_locale(locale)
        line = _paint_facts(facts, 80, muted=MUTED, faint=FAINT)
        assert (expected, MUTED) in line
        assert ("proactive", FAINT) in line
        if locale == "fr":
            assert plain([line])[0] == "    KIND_FR · CLASS_FR · proactive"


@pytest.mark.asyncio
async def test_real_app_language_switch_keeps_raw_snapshot_without_registry_io(
    tmp_path,
    monkeypatch,
):
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    original = i18n_text._sources

    def sources(root, namespace, locale):
        values = original(root, namespace, locale)
        if locale == "fr" and namespace == "tui.authoring":
            return {**values, "tui.authoring.role": "KIND_FR"}
        return values

    monkeypatch.setattr(i18n_text, "_sources", sources)
    monkeypatch.setattr(i18n_text._locale, "eligible_locale", lambda locale: locale)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app._open_settings_view()
        await pilot.pause()
        view = app._settings_view
        assert view is not None
        view._pane = "agents"
        snapshot = view._agents
        assert snapshot and all(isinstance(row[1], tuple) for row in snapshot)

        def forbidden(*args, **kwargs):
            raise AssertionError("registry enumeration during repaint")

        import local_operator.agent_profiles as profiles

        monkeypatch.setattr(profiles, "agent_listing_facts", forbidden)
        monkeypatch.setattr(profiles, "agent_listing_rows", forbidden)
        monkeypatch.setattr(view, "load", forbidden)
        for locale in ("en", "fr", "en"):
            app._apply_language(locale)
            await pilot.pause()
            assert view._agents is snapshot
            assert ("KIND_FR" in view._pane_text.plain) == (locale == "fr")
        # Crowded structured rows still reserve the read-only caption; fitting
        # counts each actual spill row rather than admitting a half-entry.
        crowded = (
            AgentFact("kind", "role"),
            AgentFact("action_class", "proactive"),
            AgentFact("scope", "session"),
        )
        view._agents = [(f"raw-{n}", crowded, "summary") for n in range(40)]
        view._repaint()
        text = view._pane_text.plain
        assert "read-only" in text
        assert "more" in text
        assert len(text.splitlines()) <= view._pane_height()


@pytest.mark.parametrize("kind", ["role", "specialist"])
@pytest.mark.parametrize("action", [None, "reactive", "proactive", "future"])
def test_semantic_construction_keeps_raw_fields_and_filters_ordinary(tmp_path, kind, action):
    from local_operator.agents import AgentRegistry
    from tests.unit.tui.test_agent_class_slash import _fields

    registry = AgentRegistry(tmp_path)
    # Delegation effort is a role tag, not AgentData's reasoning-effort field.
    tags = ["role", "effort:custom-tier"] if kind == "role" else []
    if action is not None:
        tags.append(f"class:{action}")
    registry.create_agent(
        _fields(
            name="rawname",
            label="role proactive packaged",
            description="role proactive packaged",
            categories=["specialist"] if kind == "specialist" else [],
            tags=tags,
            model="proactive",
            effort="custom-tier",
        )
    )
    registry.create_agent(_fields(name="ordinary", tags=[], categories=[]))
    rows = agent_listing_facts(registry)
    assert not any(row[0] == "ordinary" for row in rows)
    name, label, facts, summary = next(row for row in rows if row[0] == "rawname")
    assert name == "rawname" and label == summary == "role proactive packaged"
    assert facts[0] == AgentFact("kind", kind)
    assert (AgentFact("action_class", "proactive") in facts) == (action == "proactive")
    assert all(fact.value != "reactive" for fact in facts)
    if kind == "role":
        assert facts[-2:] == (AgentFact("model", "proactive"), AgentFact("effort", "custom-tier"))


def test_model_equal_to_french_scope_label_is_still_opaque():
    i18n_text.set_active_locale("fr")
    scope_label = agent_fact_record(AgentFact("scope", "session")).text
    facts = (AgentFact("kind", "role"), AgentFact("model", scope_label))
    lines = _paint_facts_lines(facts, 80, muted=MUTED, faint=FAINT)
    assert len(lines) == 1
    assert (scope_label, FAINT) in lines[0]


@pytest.mark.parametrize("tier", ["custom-tier", "tier/(]", "高级"])
def test_opaque_effort_tiers_do_not_enter_restricted_command_span_grammar(tier, monkeypatch):
    original = i18n_text._sources

    def sources(root, namespace, locale):
        values = original(root, namespace, locale)
        if namespace == "tui.small" and locale == "ur":
            return {**values, "tui.small.model_picker.effort": "UR_EFFORT {value}"}
        return values

    monkeypatch.setattr(i18n_text, "_sources", sources)
    monkeypatch.setattr(i18n_text._locale, "eligible_locale", lambda locale: locale)
    i18n_text.set_active_locale("ur")
    facts = (AgentFact("effort", tier),)
    text = plain([_paint_facts(facts, 80, muted=MUTED, faint=FAINT)])[0]
    assert "\u2066" + tier + "\u2069" in text
    assert text.count("\u2066") + text.count("\u2067") == text.count("\u2069")
