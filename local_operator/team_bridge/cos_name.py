"""The chief of staff's published identity: her bridge name and her picture.

WHY THIS EXISTS. Team Bridge lists a person's agent under the endpoint
``label`` — a name chosen once at enrollment (default: the account's display
name) and unique per organization. The operator's chief of staff, though, has
her own name — "Aida" out of the box, and whatever the operator renames her to
afterwards — and teammates should be able to message "Aida (Ben)", not "Ben's
agent". The agent-server therefore takes an additive, NON-unique
``agent_name`` on ``PUT /v1/messaging/settings`` (same sanitizer rules as
``label``) and echoes it in the directory's per-member ``agent`` block. This
module answers the one question the runtime owns: what name to send.

It answers the picture question too (:func:`published_agent`): the LOCAL agent
definition holds the image the owner chose, and that definition is the source
of truth ``team_bridge.avatar_publish`` publishes from. Reading it here, in one
place, keeps the publish path's notion of "her agent" identical to the name
path's.

WHERE THE NAME COMES FROM, and why only there.

* The DESIGNATED CoS is ``team_bridge.session`` (settings.py): the one session
  the connector may wake, chosen by the user. Empty means "no chief of staff on
  the bridge" — badge-only arrivals — so there is no agent name to publish and
  the resolver answers ``None`` (the client then omits the field entirely).
* When that session IS Aida's (``aida/state.json``'s ``session_id``), the name
  is ``aida.name`` — the config-canonical key every other surface reads, which
  follows every rename gesture because ``Session.set_conversation_name`` syncs
  her title into it — with the SAME rules as
  :func:`local_operator.aida.naming.display_name` (absent or unusable →
  :data:`~local_operator.aida.naming.DEFAULT_NAME`). It is read from the
  config WATCHER's last-good snapshot, not through ``display_name``: that
  reader constructs a ``ConfigManager``, which moves a malformed
  ``config.yml`` aside, and this resolver runs on read paths (every active
  desktop settings write, a fresh bridge status read) that must never do that
  to a half-written hand edit (review R1-4).
* When the designated session is ANY OTHER session, the answer is ``None``.
  Deliberately: an ordinary session's only "name" is its conversation title,
  which is usually generated from the conversation's content ("Q3 layoffs
  plan"). Publishing that to every member of the organization would be a
  disclosure the user never chose; the server then falls back to the endpoint
  label, which IS a name the user chose for this purpose.

READS NEVER FAIL. A display name is decoration on an enrollment write: any
read problem (an unreadable config, a missing or corrupt Aida state file)
answers ``None`` rather than raising, so the consent write it rides on can
never be broken by it.

WIRE SHAPING. :func:`bridge_agent_name` repairs a name into the server's
label grammar (the same repair the server's ``DefaultLabel`` performs for a
name the user did not type into the bridge surface): unsafe characters are
dropped, spaces tidied, and the result cut to :data:`AGENT_NAME_MAX_CHARS`.
``aida.name`` allows 80 characters; the bridge allows 64, so a long name is
cut rather than refused — refusing would turn a cosmetic field into a failed
opt-in.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: The bridge's label cap, in code points (agent-server ``bridge.LabelMaxChars``).
#: ``agent_name`` follows the label rules, so the same cap applies.
AGENT_NAME_MAX_CHARS = 64

#: The hub's avatar ceiling (its ``avatar_too_large`` ``details.limit_bytes``).
#: A local file above it could never publish, so the resolver refuses it here
#: rather than uploading a body the hub is guaranteed to reject. Defined here
#: because this module is the one place that decides WHAT the local picture is;
#: ``avatar_publish`` and the client both read it from here.
AVATAR_MAX_BYTES = 1024 * 1024

#: An avatar file name, by the mime the block records. A block written before
#: ``filename`` existed carries only ``{sha256, mime, bytes, updated_at}``, and
#: the stored name is canonical per type, so the mime is a total fallback.
_AVATAR_BY_MIME = {
    "image/png": "avatar.png",
    "image/jpeg": "avatar.jpg",
    "image/webp": "avatar.webp",
}

#: The hub's source-digest grammar, in ONE place: what ``X-Source-SHA256``
#: carries, what the status answer echoes back, and what a stored block records
#: (review R1-F10b: it was declared three times).
SHA256_HEX = re.compile(r"[0-9a-f]{64}")

#: Characters the server treats as invisible beyond the Cc/Cf/Co categories
#: (its ``isLabelInvisible``: Other_Default_Ignorable_Code_Point, variation
#: selectors, U+2800 BRAILLE BLANK). ``unicodedata`` exposes no ODI property,
#: so the assigned, non-Cf members are listed; anything this misses is still
#: covered by the client's one retry without the field on a server refusal.
_INVISIBLE = frozenset(
    {
        0x034F,  # COMBINING GRAPHEME JOINER
        0x115F,  # HANGUL CHOSEONG FILLER
        0x1160,  # HANGUL JUNGSEONG FILLER
        0x17B4,  # KHMER VOWEL INHERENT AQ
        0x17B5,  # KHMER VOWEL INHERENT AA
        0x2800,  # BRAILLE PATTERN BLANK
        0x3164,  # HANGUL FILLER
        0xFFA0,  # HALFWIDTH HANGUL FILLER
    }
)


def _invisible(char: str) -> bool:
    point = ord(char)
    return (
        point in _INVISIBLE
        or 0x180B <= point <= 0x180F  # Mongolian free variation selectors
        or 0xFE00 <= point <= 0xFE0F  # variation selectors
        or 0xE0100 <= point <= 0xE01EF  # variation selectors supplement
    )


def bridge_agent_name(raw: object) -> str | None:
    """``raw`` repaired into the bridge's label grammar, or ``None`` if nothing is left.

    Mirrors agent-server ``bridge.DefaultLabel`` (a REPAIR, not a refusal) so a
    name the server would refuse is never sent: control, format, private-use,
    unassigned and invisible characters are dropped; every other space becomes
    U+0020; runs collapse; the result is cut to :data:`AGENT_NAME_MAX_CHARS`
    code points. A remainder of only combining marks is "blank-looking" to the
    server and reads ``None`` here.
    """

    if not isinstance(raw, str):
        return None
    kept: list[str] = []
    for char in raw:
        category = unicodedata.category(char)
        # Cs: a lone surrogate is invalid UTF-8 (the server's DefaultLabel
        # drops it via ToValidUTF8) and would make httpx's encoder raise.
        if category in {"Cc", "Cf", "Co", "Cn", "Cs", "Zl", "Zp"} or _invisible(char):
            continue
        kept.append(" " if char.isspace() or category == "Zs" else char)
    tidy = " ".join("".join(kept).split())[:AGENT_NAME_MAX_CHARS].strip()
    if not tidy or all(unicodedata.category(char) == "Mn" for char in tidy):
        return None
    return tidy


def _config_values(config_dir: Path) -> Mapping[str, Any]:
    """The directory-scoped watcher's last-good config values.

    The watcher, not a ``ConfigManager``: constructing one on a malformed hand
    edit would move the user's file aside from a read path (the reason
    ``client._local_settings_snapshot`` reads it the same way).
    """

    from local_operator.config_watch import process_watcher

    watcher = process_watcher(config_dir)
    watcher.poll_now()
    return watcher.values


def _aida_name(values: Mapping[str, Any]) -> str:
    """``aida.name`` from a config snapshot, defaulted exactly like ``display_name``."""

    from local_operator.aida import naming as aida_naming

    section = values.get("aida")
    raw = section.get("name", aida_naming.DEFAULT_NAME) if isinstance(section, Mapping) else None
    if not isinstance(raw, str):
        return aida_naming.DEFAULT_NAME
    try:
        return aida_naming.validate_name(raw)
    except (ValueError, TypeError):
        return aida_naming.DEFAULT_NAME


def cos_display_name(config_dir: Path | str | None) -> str | None:
    """The designated chief of staff's current display name, wire-ready; else ``None``.

    ``None`` means "send no ``agent_name``": no config root, no designated CoS,
    a designated session that is not Aida's (see the module docstring for why
    its title is never published), or a name that repairs to nothing. Never
    raises.

    A THIN WRAPPER over the same gate :func:`published_agent` uses, on purpose:
    the callers of this one (the enrollment write and the rename re-sync) must
    keep their exact pre-avatar behaviour, so the agent-definition lookup the
    picture needs can never turn a name into ``None``.
    """

    resolved = _designated_aida(config_dir)
    return None if resolved is None else resolved[1]


def degenerate_agent_name(agent_name: object, label: object) -> bool:
    """Whether ``agent_name`` is only the owner's OWN name rather than a chosen one.

    The status payload carries no account display name, so the endpoint LABEL
    stands in for it: the hub's ``DefaultLabel`` derives the label from the
    owner's display name, and the rows this rule exists for are exactly the
    default-on enrollments nobody typed a name into. An approximation, and a
    deliberately one-sided one — the hub's own write-time refusal
    (``details.rule = owner_name``) stays the authority for the cases the label
    has since been renamed past (a profile change, a hand-edited label).

    False for an empty name: "no name of its own" is already the state this
    rule drives towards, so a cleared row must not be re-PUT on every boundary.
    """

    if not isinstance(agent_name, str) or not agent_name.strip():
        return False
    if not isinstance(label, str) or not label.strip():
        return False
    return _name_key(agent_name) == _name_key(label)


def _name_key(value: str) -> str:
    """``value`` folded the way the hub's ``NormalizeNameKey`` folds it.

    Case and whitespace runs only: the point is to catch "Sherman  Tsui" and
    "sherman tsui" as the SAME name, not to re-implement the server's full
    identifier normalisation for a cosmetic comparison.
    """

    return " ".join(value.split()).casefold()


def published_agent(config_dir: Path | str | None) -> PublishedAgent | None:
    """The local agent definition behind the designated chief of staff, or ``None``.

    ``None`` is the answer for every case where this machine has nothing to say
    about the hub's copy — no config root, no designated CoS, a designated
    session that is not Aida's, or no agent definition named after her role. It
    means PUBLISH NOTHING and, just as importantly, DELETE NOTHING: a machine
    that cannot see the local picture must never be the reason the hub's copy
    disappears.

    A resolved agent may still carry no picture. ``avatar_path is None`` with
    ``avatar_broken`` False is the user's own choice (the local writer stores
    the ``avatar`` block together with the file, so the block's absence is a
    cleared picture); ``avatar_broken`` True is a DOUBT — a block that names a
    file this side will not resolve — and a doubt publishes nothing.
    """

    resolved = _designated_aida(config_dir)
    if resolved is None:
        return None
    root, name = resolved
    if root is None:
        return None
    found = _aida_definition(root)
    if found is None:
        return None
    agent_id, agent_dir, definition = found
    fields = _avatar_of(agent_dir, definition.get("avatar"))
    return PublishedAgent(local_agent_id=agent_id, name=name, **fields)


def _designated_aida(config_dir: Path | str | None) -> tuple[Path | None, str | None] | None:
    """``(config root, wire-ready name)`` when the designated CoS is Aida's; else ``None``.

    The gate both public readers share. The name may be ``None`` — an unusable
    ``aida.name`` is still a designated CoS with a picture to publish.
    """

    if config_dir is None:
        return None
    root = Path(config_dir)
    try:
        from local_operator.team_bridge.settings import TeamBridgeSettings

        values = _config_values(root)
        designated = TeamBridgeSettings.from_values(values).session
        if not designated:
            return None
        from local_operator.aida import state as aida_state

        if not aida_state.is_aida_session(root, designated):
            return None
        return root, bridge_agent_name(_aida_name(values))
    except Exception as exc:  # noqa: BLE001 — a name must never break the write it rides on
        # The TYPE only. A traceback's last line is the exception's message, and
        # a FileNotFoundError here names a config path (review R1-S8).
        logger.debug("team_bridge: could not resolve the CoS identity (%s)", type(exc).__name__)
        return None


def _aida_definition(root: Path) -> tuple[str, Path, Mapping[str, Any]] | None:
    """``(agent id, directory, agent.yml)`` for Aida's role definition, or ``None``.

    Addresses the definition the way the rest of the runtime does — by ``name``,
    the key ``/agent aida`` and the session attachment's ``agent`` field resolve
    — but reads the YAML directly instead of through ``AgentRegistry``: building
    a registry creates and migrates directories, and this runs on read paths
    (every fresh status read) that must write nothing. Reading the mapping also
    keeps this working before ``AgentData`` grows a first-class ``avatar`` field:
    pydantic drops an unknown key on the next registry save.
    """

    from local_operator.aida.bootstrap import ROLE_NAME

    try:
        import yaml
    except Exception:  # noqa: BLE001 — no YAML reader means no definition, not an error
        return None
    agents_dir = root / "agents"
    try:
        candidates = sorted(agents_dir.iterdir())
    except OSError:
        return None
    for agent_dir in candidates:
        try:
            if not agent_dir.is_dir():
                continue
            definition = yaml.safe_load((agent_dir / "agent.yml").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        if not isinstance(definition, Mapping):
            continue
        raw_name = definition.get("name")
        if isinstance(raw_name, str) and raw_name.strip().casefold() == ROLE_NAME.casefold():
            return agent_dir.name, agent_dir, definition
    return None


def _avatar_of(agent_dir: Path, block: object) -> dict[str, Any]:
    """The :class:`PublishedAgent` avatar fields for one stored ``avatar`` block.

    A block that is PRESENT but names a file this side will not resolve answers
    ``avatar_broken`` — publishing nothing, and never a delete — while a block
    that is ABSENT is a DELIBERATE clear... but only when the agent store can
    preserve the block in the first place. ``AgentData`` is a pydantic model with
    the default ``extra="ignore"``, so until ``avatar`` is one of its fields both
    writers drop the key on the next ordinary save (a rename, a settings edit).
    Absence would then mean "the registry erased it", not "the user cleared it",
    and treating it as a clear would silently delete the teammate-visible copy
    for good (review R1-F4). When the field is missing from the model, absence is
    a DOUBT and :attr:`PublishedAgent.avatar_broken` says so.
    """

    if not isinstance(block, Mapping):
        return {} if _avatar_survives_registry() else {"avatar_broken": True}
    filename = _avatar_filename(block)
    if filename is None:
        return {"avatar_broken": True}
    candidate = _resolve_avatar(agent_dir, filename)
    if candidate is None:
        return {"avatar_broken": True}
    stored_sha = block.get("sha256")
    stored_bytes = block.get("bytes")
    return {
        "avatar_path": candidate,
        "avatar_sha256": (
            stored_sha if isinstance(stored_sha, str) and SHA256_HEX.fullmatch(stored_sha) else None
        ),
        "avatar_bytes": (
            stored_bytes
            if isinstance(stored_bytes, int) and not isinstance(stored_bytes, bool)
            else None
        ),
    }


#: Cached answer for :func:`_avatar_survives_registry`; ``None`` = not asked yet.
_AVATAR_FIELD: bool | None = None


def _avatar_survives_registry() -> bool:
    """Whether an ``avatar`` block survives the agent store's own round trip.

    The design's MEASURED CONSTRAINT is that ``avatar`` becomes a first-class
    ``AgentData`` field; both writers serialise ``model_dump()`` and the read path
    builds rows through the model, so a key the model does not declare is dropped
    on the next save. Until that field lands, an absent block proves nothing.

    An import failure answers ``False`` — "cannot prove it survives" is exactly
    the doubt this exists to express, and a picture must not be deleted on a
    guess.
    """

    global _AVATAR_FIELD
    if _AVATAR_FIELD is None:
        try:
            from local_operator.agents import AgentData

            _AVATAR_FIELD = "avatar" in AgentData.model_fields
        except Exception:  # noqa: BLE001 - an unreadable model is a doubt, not a crash
            return False
    return _AVATAR_FIELD


def _avatar_filename(block: Mapping[str, Any]) -> str | None:
    """The stored file name: ``filename`` first, then the legacy mime fallback."""

    raw = block.get("filename")
    if isinstance(raw, str) and raw.strip():
        # ``./avatar.png`` and ``avatar.png`` name the same file in the same
        # directory, so the prefix is dropped here rather than read as a doubt by
        # the separator rule below (review R1-F10).
        name = raw.strip()
        return name[2:] if name.startswith("./") else name
    mime = block.get("mime")
    if isinstance(mime, str):
        return _AVATAR_BY_MIME.get(mime.strip().lower())
    return None


def _resolve_avatar(agent_dir: Path, filename: str) -> Path | None:
    """The avatar file ``filename`` names INSIDE ``agent_dir``, or ``None``.

    The block is a hand-editable YAML file, so its ``filename`` is untrusted
    input: an absolute path, a separator, ``..``, a NUL or a symlink pointing out
    of the agent's own directory all read as "no picture" rather than as a file
    to publish. The containment check runs AFTER ``resolve`` so a symlink swapped
    in between counts as the escape it is.
    """

    if (
        not filename
        or filename in {".", ".."}
        or any(sep and sep in filename for sep in (os.sep, os.altsep, "/", "\\"))
        or os.path.isabs(filename)
        or "\x00" in filename
    ):
        return None
    try:
        root = agent_dir.resolve(strict=True)
        candidate = (root / filename).resolve(strict=False)
    except OSError:
        return None
    if candidate.parent != root or not candidate.name:
        return None
    try:
        if not candidate.is_file():
            return None
        size = candidate.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > AVATAR_MAX_BYTES:
        return None
    return candidate


@dataclass(frozen=True)
class PublishedAgent:
    """One local agent definition, as the hub's copy of it is derived from.

    ``local_agent_id`` is the definition's directory name (the id every surface
    addresses it by), never a path: the publish path needs an identity, and a
    path in a dataclass that gets logged is how a home directory reaches a log.
    """

    local_agent_id: str
    name: str | None = None
    #: The RESOLVED file, read at publish time. ``None`` means "no picture".
    #: ``repr=False`` on purpose: this class says a path in a dataclass "is how
    #: a home directory reaches a log", so it must not put one in its own repr
    #: (review R1-S5).
    avatar_path: Path | None = field(default=None, repr=False)
    #: What ``agent.yml`` recorded. Trusted for the upload only while the file's
    #: size still matches ``avatar_bytes`` (see ``avatar_publish._avatar_bytes``).
    avatar_sha256: str | None = None
    avatar_bytes: int | None = None
    #: An ``avatar`` block is present but yields no usable file — a DOUBT, which
    #: publishes nothing and never deletes.
    avatar_broken: bool = False


__all__ = [
    "AGENT_NAME_MAX_CHARS",
    "AVATAR_MAX_BYTES",
    "PublishedAgent",
    "SHA256_HEX",
    "bridge_agent_name",
    "cos_display_name",
    "degenerate_agent_name",
    "published_agent",
]
