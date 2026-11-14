"""S1 (`wire.settings`): the registry's en copy is byte-identical to extraction time.

The fixture next to this file is a SNAPSHOT of every user-facing string in the
settings registry, taken from `origin/main` BEFORE the extraction
(`22498f1a59`; the branch is rebased onto `b92f4e6148`, byte-identical for
`settings_io.py`). This test re-derives the same
enumeration from the live registry and requires every rendered string to
match the golden EXACTLY — the committed form of the slice's byte-identical
proof, re-runnable by review and QA, and the trip-wire for any later edit that
changes a registry string without regenerating the catalogue.

The second test pins the resolution PATH: the registry's strings must come
through the `wire.settings` catalogue (not from leftovers), and an unknown
message degrades to its own code the way `messages.render` documents.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from local_operator import settings_io
from local_operator.i18n import catalogues, messages

FIXTURE = Path(__file__).parent / "fixtures" / "wire_settings_en_golden.json"


def _snake(text: object) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return value or "unset"


def _live_slots() -> dict[str, str]:
    """Every user-facing registry string, keyed the slice's way."""
    slots: dict[str, str] = {}

    def record(key: str, value: object) -> None:
        if isinstance(value, str) and value != "":
            slots[key] = value

    for setting in settings_io.SETTINGS:
        record(f"wire.settings.{_snake(setting.key)}.label", setting.label)
        record(f"wire.settings.{_snake(setting.key)}.help", setting.help)
        if setting.warning:
            record(f"wire.settings.{_snake(setting.key)}.warning", setting.warning)
        if setting.placeholder:
            record(f"wire.settings.{_snake(setting.key)}.placeholder", setting.placeholder)
        for choice in setting.choices:
            record(
                f"wire.settings.{_snake(setting.key)}.choice.{_snake(choice.value)}.label",
                choice.label,
            )
            if choice.description:
                record(
                    f"wire.settings.{_snake(setting.key)}.choice."
                    f"{_snake(choice.value)}.description",
                    choice.description,
                )
    for section in settings_io.SECTIONS:
        record(f"wire.settings.{_snake(section.name)}.title", section.title)
        if section.description:
            record(f"wire.settings.{_snake(section.name)}.description", section.description)
    return slots


def test_registry_copy_is_byte_identical_to_the_extraction_time_golden() -> None:
    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    current = _live_slots()
    missing = sorted(set(golden) - set(current))
    added = sorted(set(current) - set(golden))
    changed = sorted(k for k in golden if k in current and current[k] != golden[k])
    assert not missing, f"slots lost since extraction: {missing[:10]}"
    assert not added, f"new slots need the catalogue + golden update: {added[:10]}"
    assert not changed, "strings changed since extraction: " + repr(
        {k: (golden[k], current[k]) for k in changed[:5]}
    )


def test_registry_copy_resolves_through_the_catalogue() -> None:
    # Each SHAPE of the slice must be addressable in the catalogue, and its
    # server-side rendering must equal the live registry string (the provider
    # rows are the message-with-parameter case).
    samples = [
        ("wire.settings.model.title", {}),
        ("wire.settings.model.description", {}),
        ("wire.settings.model_effort.choice.high.description", {}),
        ("wire.settings.bool.on", {}),
        ("wire.settings.bash_shell.help", {}),
        ("wire.settings.shell_environment_inherit.placeholder", {}),
        ("wire.settings.local_providers.base_url.label", {"name": "Ollama"}),
    ]
    for key, params in samples:
        rendered = messages.render(key, params, locale="en")
        assert rendered != key, f"{key} is missing from the catalogue"
        assert "\n" not in rendered


def test_unknown_message_degrades_to_its_own_code() -> None:
    text = settings_io._t(messages.msg("wire.settings.does_not_exist"))
    assert text == "wire.settings.does_not_exist"


def test_provider_rows_render_the_product_name_from_the_preset() -> None:
    # The LOCAL_PRESETS labels interpolate the preset's display name (a
    # product name, untranslated per the style guide); the message carries it
    # as a parameter, so every provider resolves the same template.
    labels = {
        setting.label
        for setting in settings_io.SETTINGS
        if setting.key.startswith("providers.") and setting.key.endswith(".base_url")
    }
    assert labels, "expected provider rows"
    assert all(label.endswith(" endpoint") for label in labels)
    assert (
        catalogues.load_catalogue("en", "wire.settings")[
            "wire.settings.local_providers.base_url.label"
        ]
        == "{name} endpoint"
    )


def test_guardrails_extraction_slots_reuse_audited_bytes_in_every_locale() -> None:
    """The registry snapshot uses historical wire.settings names; guardrails
    renders its child namespace. Compatibility slots must copy the audited
    source values exactly, not introduce independently drifting translations.
    See docs/guardrails-settings-extraction.md for the two audit lineages.
    """
    aliases = {
        "wire.settings.guardrails.title": "wire.guardrails.section.title",
        "wire.settings.guardrails.description": "wire.guardrails.section.description",
    }
    for row in ("enabled", "critical_override", "judge_timeout_s", "rules_dir"):
        for field in ("label", "help"):
            aliases[f"wire.settings.guardrails_{row}.{field}"] = f"wire.guardrails.{row}.{field}"
    for locale in ("en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"):
        parent = catalogues.load_catalogue(locale, "wire.settings")
        child = catalogues.load_catalogue(locale, "wire.guardrails")
        for dest, src in aliases.items():
            assert parent[dest].encode("utf-8") == child[src].encode("utf-8"), (locale, dest)
            assert messages.render(dest, {}, locale=locale).encode("utf-8") == child[src].encode(
                "utf-8"
            ), (locale, dest)


def test_no_registry_string_is_its_own_code() -> None:
    """A label, help, title or description that renders as its message CODE is a miss.

    `_t` degrades to the code when the message's namespace is not among the ones
    it reads (`wire.settings` + `_CHILD_SETTINGS_NAMESPACES`). The golden above
    cannot catch that on its own — it was regenerated once with the raw codes in
    it (#3945 round 1, D1/Q1: the Phone access section printed
    `wire.phone_access.auto_setup.label`) — so this asserts the property itself
    over every registry string that carries its message.
    """
    leaked: list[tuple[str, str]] = []

    def check(where: str, value: object) -> None:
        message = getattr(value, "message", None)
        if message is not None and str(value) == message.code:
            leaked.append((where, message.code))

    for setting in settings_io.SETTINGS:
        for field in ("label", "help", "warning", "placeholder"):
            check(f"{setting.key}.{field}", getattr(setting, field, None))
        for choice in setting.choices:
            check(f"{setting.key}.choice.{choice.value}.label", choice.label)
            check(f"{setting.key}.choice.{choice.value}.description", choice.description)
    for section in settings_io.SECTIONS:
        check(f"section {section.name}.title", section.title)
        check(f"section {section.name}.description", section.description)
    assert not leaked, f"registry strings rendering as their own code: {leaked[:10]}"
