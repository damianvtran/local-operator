"""The sanctioned settings-write seam for NON-interactive (agent) callers.

WHY THIS MODULE EXISTS. ``source="local"`` means "a write the operator made
through the settings facade in the process holding the gate"
(``harness/approval.py::loosening_is_authorised``), and the codebase treats that
equivalence as a PROPERTY, not a per-call-site claim:
``tests/unit/test_approval_source_boundary.py`` pins the static half for the
agent-facing trees, and this module is the architectural half. The agent's
``setting`` tool calls HERE, never the facade, and every write below goes out
with ``notify=False`` — an UNATTRIBUTED write.

WHY UNATTRIBUTED IS THE HONEST SHAPE, not merely an out-of-tree relocation.
Relocating a notified call would leave the runtime hole intact: the tool
executes in processes that can hold a live approval gate (the embedded TUI,
``lop exec``), and a notified write there produces ``source="local"`` in exactly
that process — the gated party minting the operator's attribution and loosening
the gate that constrains it (#1282's attack, one caller removed). With the
fan-out suppressed, the write is what it factually is — a file edit made on the
agent's behalf — and every process, including the writer's own watcher (whose
next stat-poll reads the file), classifies it the way the predicate's docstring
classifies "another process's edit": it may TIGHTEN a live gate and may never
loosen one. New sessions read the file as any edit; no value semantics change.

WHAT STAYS SHARED. Everything that is not attribution: ``write_setting`` is
still the one funnel (validation, the sub-mapping merge, refusals, the trigger
snapshot, the aida hold marker), so the /settings page, ``PATCH /v1/settings``,
``lop config edit`` and this seam cannot disagree about what a legal write is.
The cascade refusal and the route's clear-to-unset arm live here because they
are properties of the agent-facing CONTRACT, not of any single caller.

WHAT THIS MODULE IS NOT. It is not the agent's policy layer: the tool keeps its
own params validation, its read/JSON shapes and its result/refusal formatting.
It is also NOT for interactive callers — the TUI page, the HTTP route and the
CLI call ``settings_io`` directly, with the notification they are entitled to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from local_operator import settings_io
from local_operator.settings_io import ConfigUnreadableError

if TYPE_CHECKING:
    from local_operator.config import ConfigManager
    from local_operator.settings_io import Setting

__all__ = [
    "ConfigUnreadableError",
    "apply",
    "is_default",
    "read",
    "resolve",
]


def resolve(key: str) -> "Setting | None":
    """The registry row for ``key`` — ``None`` for the unknown-key answer."""
    return settings_io.resolve_key(key)


def read(manager: "ConfigManager", setting: "Setting") -> Any:
    """The row's current value, as the registry reads it."""
    return settings_io.read_setting(manager, setting)


def is_default(manager: "ConfigManager", setting: "Setting") -> bool:
    """Whether the row currently reads as its default."""
    return settings_io.is_default(manager, setting)


def apply(manager: "ConfigManager", setting: "Setting", value: Any) -> None:
    """Perform one agent-origin settings write, UNATTRIBUTED (see module doc).

    The route's semantics, mirrored in one place: cascade keys are refused (the
    base-snapshot contract has no field here), an empty string on an
    ``empty_unsets`` row is the clear-to-unset gesture, everything else is
    ``write_setting``. Raises ``ConfigUnreadableError`` (file unreadable) and
    ``ValueError`` (the registry's own refusal), so callers catch one surface.
    """
    if setting.kind is settings_io.Kind.CASCADE:
        # The route edits chains under an original-snapshot contract
        # (`SettingEdit.base`): an unchanged chain from a stale screen must
        # never overwrite a newer chain. This seam has no snapshot field, and
        # writing raw would accept shapes `_normalize_chains` drops on read —
        # a silent divergence is the one outcome the shared funnel exists to
        # prevent — so the key is REFUSED, explicitly, with the alternative
        # named.
        raise ValueError(
            f"Setting '{setting.key}' is a cascade edited with an original-snapshot "
            "contract. Use the /settings page, `lop config edit`, or PATCH "
            "/v1/settings with `base` — the agent write path does not carry the snapshot."
        )
    settings_io._require_readable_config(manager)
    if setting.empty_unsets and value == "":
        # The clear-to-unset gesture, mirrored from the route and the CLI
        # (their arm is exactly this one: ``empty_unsets`` + the empty string):
        # deleting the key is what makes it read as its default again — storing
        # a literal "" would report a deliberate choice the user never made.
        settings_io.reset_setting(manager, setting, notify=False)
        return
    settings_io.write_setting(manager, setting, value, notify=False)
