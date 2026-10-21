"""The chief of staff's CURRENT display name, as teammates see it on the bridge.

WHY THIS EXISTS. Team Bridge lists a person's agent under the endpoint
``label`` — a name chosen once at enrollment (default: the account's display
name) and unique per organization. The operator's chief of staff, though, has
her own name — "Aida" out of the box, and whatever the operator renames her to
afterwards — and teammates should be able to message "Aida (Ben)", not "Ben's
agent". The agent-server therefore takes an additive, NON-unique
``agent_name`` on ``PUT /v1/messaging/settings`` (same sanitizer rules as
``label``) and echoes it in the directory's per-member ``agent`` block. This
module answers the one question the runtime owns: what name to send.

WHERE THE NAME COMES FROM, and why only there.

* The DESIGNATED CoS is ``team_bridge.session`` (settings.py): the one session
  the connector may wake, chosen by the user. Empty means "no chief of staff on
  the bridge" — badge-only arrivals — so there is no agent name to publish and
  the resolver answers ``None`` (the client then omits the field entirely).
* When that session IS Aida's (``aida/state.json``'s ``session_id``), the name
  is ``aida.name`` via :func:`local_operator.aida.naming.display_name` — the
  config-canonical reader every other surface (TUI receipts, the desktop
  payload, the greeting) already uses, which follows every rename gesture
  because ``Session.set_conversation_name`` syncs her title into that key.
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
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

#: The bridge's label cap, in code points (agent-server ``bridge.LabelMaxChars``).
#: ``agent_name`` follows the label rules, so the same cap applies.
AGENT_NAME_MAX_CHARS = 64

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
        if category in {"Cc", "Cf", "Co", "Cn", "Zl", "Zp"} or _invisible(char):
            continue
        kept.append(" " if char.isspace() or category == "Zs" else char)
    tidy = " ".join("".join(kept).split())[:AGENT_NAME_MAX_CHARS].strip()
    if not tidy or all(unicodedata.category(char) == "Mn" for char in tidy):
        return None
    return tidy


def _designated_session(config_dir: Path) -> str:
    """``team_bridge.session`` from the watcher's last-good snapshot ("" when unset).

    The directory-scoped watcher, not a ``ConfigManager``: constructing one on
    a malformed hand edit would move the user's file aside from a read path
    (the reason ``client._local_settings_snapshot`` reads it the same way).
    """

    from local_operator.config_watch import process_watcher
    from local_operator.team_bridge.settings import TeamBridgeSettings

    watcher = process_watcher(config_dir)
    watcher.poll_now()
    return TeamBridgeSettings.from_values(watcher.values).session


def cos_display_name(config_dir: Path | str | None) -> str | None:
    """The designated chief of staff's current display name, wire-ready; else ``None``.

    ``None`` means "send no ``agent_name``": no config root, no designated CoS,
    a designated session that is not Aida's (see the module docstring for why
    its title is never published), or a name that repairs to nothing. Never
    raises.
    """

    if config_dir is None:
        return None
    root = Path(config_dir)
    try:
        designated = _designated_session(root)
        if not designated:
            return None
        from local_operator.aida import naming as aida_naming
        from local_operator.aida import state as aida_state

        if not aida_state.is_aida_session(root, designated):
            return None
        return bridge_agent_name(aida_naming.display_name(root))
    except Exception:  # noqa: BLE001 — a display name must never break the write it rides on
        logger.debug("team_bridge: could not resolve the CoS display name", exc_info=True)
        return None


__all__ = ["AGENT_NAME_MAX_CHARS", "bridge_agent_name", "cos_display_name"]
