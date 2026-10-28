"""The egress surface: what can carry model-authored text off the machine, and how
the harness reads it.

WHY THIS MODULE IS THE FIX FOR REVIEW BLOCKER B3. The first version of this harness
scanned only the bridge client's ``send``/``conversation_send`` payloads. That covered
one channel out of the ones the design enumerates, so an agent could post the
Downloads listing through ``bash curl``, a Slack-shaped MCP tool, ``web_fetch``'s query
string, or a ``task`` prompt to a child session and the run would still report zero
leaks. A leak metric that only sees one door is worse than no metric, because it reads
as assurance.

THE TWO HALVES, and both are load-bearing:

1. **Enumeration.** ``EGRESS_TOOLS`` (design v1.3 §3.2/§3.3) names the tools that can
   carry text, the argument keys that carry it, and the tier each is judged at. It is
   DATA (``labels/egress_registry_v1.json``) so adding a tool is a reviewable data
   edit rather than a code change nobody diffs.

2. **Default-deny.** A tool in NEITHER ``EGRESS_TOOLS`` nor ``NON_EGRESS`` is UNKNOWN,
   and unknown is not "probably fine": it makes the run unclean. That is the design's
   own inversion (M1-r3: "a tool is egress-capable unless it is named in NON_EGRESS")
   and it is the only way a coverage claim survives a tool nobody anticipated - the
   product's v1.2 missed ``network`` and ``artifact_data`` exactly this way.

WHAT THIS MODULE DOES NOT DO. It does not judge anything. Judgement is E1's, and E1
does not exist on this branch; the harness's job is to make sure that when the agent
sends something, the something is SEEN, so the canary scan can decide whether it
leaked. Extraction is therefore deliberately generous: a false extraction costs a
little string scanning, while a missed one costs the whole measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "ALL_STRINGS",
    "EgressRegistry",
    "Payload",
    "ToolClass",
    "UNKNOWN",
    "classify",
    "extract_payloads",
    "load_registry",
]

#: The pseudo-field name meaning "every string argument, whatever it is called". MCP
#: tools cannot be enumerated ahead of time, so their arguments are read wholesale.
ALL_STRINGS = "__all_strings__"

#: The classification of a tool the registry does not know. Never treated as safe.
UNKNOWN = "unknown"

#: Bound on how much of one argument value is scanned, and on how many arguments are
#: flattened for an MCP call. Chosen to match E1's own 16 KiB payload cap so the
#: harness cannot silently scan less than the gate would.
MAX_FIELD_CHARS = 16 * 1024
MAX_FIELDS = 64


@dataclass(frozen=True)
class ToolClass:
    """One registered tool: what kind of egress it is, at which tier, which fields."""

    name: str
    kind: str
    tier: int
    text_fields: tuple[str, ...]
    ops: Mapping[str, tuple[str, ...]]

    @property
    def is_bridge(self) -> bool:
        return self.kind == "bridge"

    def fields_for(self, op: str | None) -> tuple[str, ...]:
        """The text fields for one op, or the tool's own list for a non-bridge tool.

        A bridge op the registry does not know about returns ``()`` and the caller
        treats that as an unknown SURFACE (see :func:`classify`): a new bridge op is
        new egress, and default-deny applies to ops as much as to tools.
        """

        if not self.is_bridge:
            return self.text_fields
        if op is None:
            return ()
        return self.ops.get(str(op), ())


@dataclass(frozen=True)
class Payload:
    """One model-authored text value on its way out.

    ``surface`` names the tool and field so a report can say WHERE - "the query string
    of a web_fetch", not "somewhere" - and ``destination_class`` names the kind of
    place it was headed.
    """

    surface: str
    field: str
    text: str
    destination_class: str

    def as_record(self) -> Mapping[str, Any]:
        """Metadata only: the report never carries the payload text itself."""

        return {
            "surface": self.surface,
            "field": self.field,
            "destination_class": self.destination_class,
            "chars": len(self.text),
        }


class EgressRegistry:
    """The E1 registry, loaded from data, with the fail-closed lookup rules."""

    def __init__(
        self,
        *,
        version: str,
        tools: Mapping[str, ToolClass],
        non_egress: frozenset[str],
        mcp_prefixes: tuple[str, ...],
    ) -> None:
        self.version = version
        self._tools = dict(tools)
        self._non_egress = set(non_egress)
        self._mcp_prefixes = tuple(mcp_prefixes)

    # -- classification -----------------------------------------------------

    def tool(self, name: str) -> ToolClass | None:
        """The registered class for ``name``, or ``None`` when it is not enumerated."""

        return self._tools.get(name)

    def is_mcp(self, name: str) -> bool:
        """Whether ``name`` looks like an MCP tool this build registers at runtime.

        Prefix-matching rather than a fixed list: MCP servers arrive after boot, and a
        build-time list is precisely the check the design says cannot see them.
        """

        lowered = name.lower()
        return any(lowered.startswith(prefix) for prefix in self._mcp_prefixes)

    def is_non_egress(self, name: str) -> bool:
        return name in self._non_egress

    def classify(self, name: str, arguments: Mapping[str, Any]) -> str:
        """``bridge``/``mcp``/…/``non_egress``, or :data:`UNKNOWN` when unenumerated."""

        registered = self._tools.get(name)
        if registered is not None:
            if registered.is_bridge:
                op = str(arguments.get("op") or "")
                if op and not registered.fields_for(op):
                    # A bridge op the registry does not carry: new egress surface.
                    return UNKNOWN
            return registered.kind
        if self.is_mcp(name):
            return "mcp"
        if self.is_non_egress(name):
            return "non_egress"
        return UNKNOWN

    def audit(self, tool_names: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Split an actual runtime inventory into (known, unknown).

        This is the registry test the design asks for (§7.6): enumerate the tools the
        runtime really built and name every one that is in neither list. An unknown
        name here is a coverage hole that invalidates the run - not a warning.
        """

        known: list[str] = []
        unknown: list[str] = []
        for name in sorted({str(n) for n in tool_names}):
            if name in self._tools or self.is_mcp(name) or self.is_non_egress(name):
                known.append(name)
            else:
                unknown.append(name)
        return tuple(known), tuple(unknown)

    def data(self) -> Mapping[str, Any]:
        return {
            "version": self.version,
            "egress_tools": sorted(self._tools),
            "non_egress": sorted(self._non_egress),
            "mcp_prefixes": list(self._mcp_prefixes),
        }


def load_registry(path: Path) -> EgressRegistry:
    """Read the registry data file. A malformed registry is a hard error.

    Deliberately no leniency: a registry that silently loses an entry would reduce the
    measured surface while every later number still looked fine.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != "tb-moderation-egress-registry-v1":
        raise ValueError(
            f"egress registry schema must be tb-moderation-egress-registry-v1, got {raw.get('schema')!r}"
        )
    tools: dict[str, ToolClass] = {}
    for name, entry in (raw.get("egress_tools") or {}).items():
        if str(name).startswith("_"):
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(f"egress tool {name!r} must be an object")
        ops_raw = entry.get("ops") or {}
        ops: dict[str, tuple[str, ...]] = {}
        for op, op_entry in ops_raw.items():
            if str(op).startswith("_"):
                continue
            if not isinstance(op_entry, Mapping):
                raise ValueError(f"egress tool {name!r} op {op!r} must be an object")
            fields = tuple(map(str, op_entry.get("text_fields") or ()))
            if not fields:
                # A named surface with no readable field is a coverage hole dressed as
                # coverage: refuse it rather than scan nothing and call it measured.
                raise ValueError(f"egress tool {name!r} op {op!r} declares no text_fields")
            ops[str(op)] = fields
        text_fields = tuple(map(str, entry.get("text_fields") or ()))
        if not text_fields and not ops:
            raise ValueError(f"egress tool {name!r} declares neither text_fields nor ops")
        tools[str(name)] = ToolClass(
            name=str(name),
            kind=str(entry.get("kind") or UNKNOWN),
            tier=int(entry.get("tier") or 2),
            text_fields=text_fields,
            ops=ops,
        )
    non_egress = frozenset(str(n) for n in (raw.get("non_egress") or ()))
    prefixes = tuple(str(p) for p in (raw.get("mcp_name_prefixes") or ()))
    return EgressRegistry(
        version=str(raw.get("registry_version") or ""),
        tools=tools,
        non_egress=non_egress,
        mcp_prefixes=prefixes,
    )


def classify(
    registry: EgressRegistry, tool_name: str, arguments: Mapping[str, Any]
) -> str:
    """Module-level convenience so callers need not hold the registry twice."""

    return registry.classify(tool_name, arguments)


def _flatten_strings(value: Any, *, depth: int = 0) -> list[str]:
    if depth > 4:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, item in value.items():
            out.extend(_flatten_strings(item, depth=depth + 1))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_strings(item, depth=depth + 1))
        return out
    return []


def extract_payloads(
    registry: EgressRegistry,
    tool_name: str,
    arguments: Mapping[str, Any],
) -> tuple[Payload, ...]:
    """Every model-authored string on this call that could leave the machine.

    Returns ``()`` for a non-egress tool. For an UNKNOWN tool it also returns ``()`` -
    and the caller must NOT read that as "nothing to scan": an unknown tool is a
    coverage failure that invalidates the run (see ``AgentLeg``'s registry audit and
    ``scoring``'s ``unknown_egress_tool`` outcome).
    """

    kind = registry.classify(tool_name, arguments)
    if kind in ("non_egress", UNKNOWN):
        return ()
    registered = registry.tool(tool_name)
    op = str(arguments.get("op") or "") if registered is not None and registered.is_bridge else None
    fields = registered.fields_for(op) if registered is not None else (ALL_STRINGS,)
    if not fields:
        fields = (ALL_STRINGS,)

    payloads: list[Payload] = []
    emitted = 0
    for field in fields:
        if field == ALL_STRINGS:
            values = _flatten_strings(dict(arguments))
            names = ["<mcp:args>"] * len(values)
        else:
            raw = arguments.get(field)
            if isinstance(raw, str):
                values = [raw]
            elif isinstance(raw, (list, tuple, Mapping)):
                values = _flatten_strings(raw)
            else:
                values = []
            names = [field] * len(values)
        for field_name, text in zip(names, values):
            cleaned = text[:MAX_FIELD_CHARS]
            if not cleaned.strip():
                continue
            payloads.append(
                Payload(
                    surface=f"{tool_name}:{op + '/' if op else ''}{field_name}",
                    field=field_name,
                    text=cleaned,
                    destination_class=kind,
                )
            )
            emitted += 1
            if emitted >= MAX_FIELDS:
                return tuple(payloads)
    return tuple(payloads)
