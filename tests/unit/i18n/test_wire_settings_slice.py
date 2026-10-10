"""S1 (`wire.settings`): the registry's en copy is byte-identical to extraction time.

The fixture next to this file is a SNAPSHOT of every user-facing string in the
settings registry, taken from `origin/main` at the merge base of the slice
(`22498f1a59`) BEFORE the extraction. This test re-derives the same
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
                    f"wire.settings.{_snake(setting.key)}.choice.{_snake(choice.value)}.description",
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
    assert catalogues.load_catalogue("en", "wire.settings")[
        "wire.settings.local_providers.base_url.label"
    ] == "{name} endpoint"
