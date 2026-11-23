"""Names and description fingerprints, never skill bodies, for mesh session transfer."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_operator.skills.discovery import Skill

SkillManifest = dict[str, str]


def description_fingerprint(skill: Skill) -> str:
    return hashlib.sha256(skill.description.encode("utf-8")).hexdigest()


def selected_skill_manifest(skills: Iterable[Skill]) -> SkillManifest:
    # Packaged guides are receiver-local harness procedures, not synced skills.
    return {
        skill.name: description_fingerprint(skill)
        for skill in skills
        if skill.resource_type == "skill"
    }


def parse_skill_manifest(value: object) -> SkillManifest | None:
    """Validate all entries or none; an empty selection is a real pin.

    The enclosing create record and copied sidecar already have byte bounds.
    Discovery also admits legacy names outside the managed-skill slug grammar.
    Keep exact spelling, but never admit multiline or control text; callers
    escape names when displaying an omission.
    """
    if not isinstance(value, dict):
        return None
    result: SkillManifest = {}
    for name, fingerprint in value.items():
        if not isinstance(name, str) or name != name.strip() or not name.isprintable():
            return None
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            return None
        result[name] = fingerprint
    return result
