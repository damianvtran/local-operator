"""The egress surface: what can carry model-authored text off the machine, and how
the harness reads it.

WHY THIS MODULE IS THE FIX FOR REVIEW BLOCKER B3 (R1) AND B1/B2 (R2). The first
version of this harness scanned only the bridge client's ``send``/``conversation_send``
payloads. The second version enumerated the surface but kept two silent holes: an
unknown tool observed after the initial inventory audit escaped classification, and a
bounded recursive walk dropped anything nested deeper than four levels, longer than
16 KiB, or beyond 64 values - while the case still scored as measured. Both are the same
class of defect: **a scan that silently does not cover the payload reads as a clean
result.**

THE TWO HALVES, and both are load-bearing:

1. **Enumeration.** ``EGRESS_TOOLS`` (design v1.3 §3.2/§3.3) names the tools that can
   carry text, the argument keys that carry it, and the tier each is judged at. It is
   DATA (``labels/egress_registry_v1.json``) so adding a tool is a reviewable data
   edit rather than a code change nobody diffs.

2. **Default-deny, over time and over structure.** A tool in NEITHER ``EGRESS_TOOLS``
   nor ``NON_EGRESS`` is UNKNOWN, and unknown is not "probably fine": it makes the case
   invalid **wherever it is observed** - at the inventory audit AND at every tool
   execution. Extraction is likewise fail-closed: :func:`extract_payloads` never
   truncates, and an argument shape it cannot fully walk is reported as an extraction
   problem that invalidates the case rather than as an empty payload.

WHAT THIS MODULE DOES NOT DO. It does not judge anything. Judgement is E1's, and E1
does not exist on this branch; the harness's job is to make sure that when the agent
sends something, the something is SEEN IN FULL, so the canary scan can decide whether
it leaked. Extraction is therefore deliberately generous: a false extraction costs a
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
    "ExtractionProblem",
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

#: Cost bounds on ONE argument blob - and hitting one is an INVALIDITY, never a clip.
#: Review R2 B2: the previous version truncated each string at 16 KiB, stopped after
#: 64 values and ignored anything below depth four, so a canary hidden past a bound
#: vanished while the case stayed valid. The bounds still exist because a model can
#: emit an arbitrarily large blob and the scan must not become an unbounded memory
#: sink, but exceeding them now means "this case was not measured", not "scan the
#: first part and call it done".
MAX_TOTAL_CHARS = 1024 * 1024
MAX_VALUES = 4096


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
        return {
            "surface": self.surface,
            "field": self.field,
            "text": self.text,
            "destination_class": self.destination_class,
        }


@dataclass(frozen=True)
class ExtractionProblem:
    """An argument blob the harness could not walk completely.

    Never a warning. Any problem here means the payload was NOT fully read, so the
    case's leak count is meaningless and the case is invalid (review R2 B1/B2).
    """

    surface: str
    reason: str

    def as_record(self) -> Mapping[str, Any]:
        return {"surface": self.surface, "reason": self.reason}


class EgressRegistry:
    """The closed enumeration, plus the default-deny rule over tool names."""

    def __init__(
        self,
        *,
        version: str,
        tools: Mapping[str, ToolClass],
        non_egress: Iterable[str],
        mcp_prefixes: Sequence[str],
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
    coverage claim without anyone editing code.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tools: dict[str, ToolClass] = {}
    for name, entry in (raw.get("egress_tools") or {}).items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"egress tool {name!r} must be an object")
        ops: dict[str, tuple[str, ...]] = {}
        for op, op_entry in (entry.get("ops") or {}).items():
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


def _walk_strings(
    value: Any,
    *,
    surface: str,
    problems: list[ExtractionProblem],
    counter: list[int],
    path: str = "",
) -> list[tuple[str, str]]:
    """Every string in ``value``, at ANY depth, with its key path.

    NO DEPTH LIMIT and no early return. The recursive walk this replaces stopped at
    depth four and the caller truncated at 16 KiB/64 values, which is how a nested or
    oversized canary escaped detection (review R2 B2). The only remaining limits are
    the total blob bounds in :data:`MAX_TOTAL_CHARS`/:data:`MAX_VALUES`, and reaching
    one appends an :class:`ExtractionProblem` instead of returning partial coverage.
    """

    found: list[tuple[str, str]] = []
    if counter[0] >= MAX_VALUES:
        problems.append(
            ExtractionProblem(surface=surface, reason=f"more than {MAX_VALUES} values in one payload")
        )
        return found
    counter[0] += 1
    if isinstance(value, str):
        if value:
            found.append((path or "<value>", value))
        return found
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool, type(None))):
                # A non-scalar mapping key means the blob is not JSON-shaped, so a
                # serialisation would be lossy. Unknown shape => invalid, not skipped.
                problems.append(
                    ExtractionProblem(surface=surface, reason=f"non-scalar mapping key at {path or '<root>'}")
                )
                continue
            found.extend(
                _walk_strings(item, surface=surface, problems=problems, counter=counter,
                              path=f"{path}.{key}" if path else str(key))
            )
        return found
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(
                _walk_strings(item, surface=surface, problems=problems, counter=counter,
                              path=f"{path}[{index}]")
            )
        return found
    if value is None or isinstance(value, (bool, int, float)):
        # Scalar non-strings carry no text a canary could hide in.
        return found
    # An arbitrary object (bytes, a dataclass, a custom container): the harness cannot
    # prove there is no string inside, so it must not pretend the payload was empty.
    problems.append(
        ExtractionProblem(
            surface=surface,
            reason=f"unsupported value type {type(value).__name__} at {path or '<root>'}",
        )
    )
    return found


def extract_payloads(
    registry: EgressRegistry,
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    problems: list[ExtractionProblem] | None = None,
) -> tuple[Payload, ...]:
    """Every model-authored string on this call that could leave the machine, IN FULL.

    Returns ``()`` for a non-egress tool. For an UNKNOWN tool it also returns ``()`` -
    and the caller must NOT read that as "nothing to scan": an unknown tool is a
    coverage failure that invalidates the case (the caller unions it into the run's
    unknown set at the moment of observation; review R2 B1).

    ``problems`` is an out-parameter: pass a list and any incomplete extraction is
    appended to it. A non-empty list after this call means the case is invalid -
    partial coverage is never silently accepted (review R2 B2). Passing ``None`` keeps
    the pure "just the payloads" shape for linting, where a problem still raises
    rather than being dropped.
    """

    kind = registry.classify(tool_name, arguments)
    if kind in ("non_egress", UNKNOWN):
        return ()
    registered = registry.tool(tool_name)
    op = str(arguments.get("op") or "") if registered is not None and registered.is_bridge else None
    fields = registered.fields_for(op) if registered is not None else (ALL_STRINGS,)
    if not fields:
        fields = (ALL_STRINGS,)

    local_problems: list[ExtractionProblem] = []
    counter = [0]
    values: list[tuple[str, str]] = []
    for field in fields:
        if field == ALL_STRINGS:
            values.extend(
                _walk_strings(dict(arguments), surface=f"{tool_name}:<args>",
                              problems=local_problems, counter=counter)
            )
        else:
            if field not in arguments:
                # An absent optional field is not a problem; a declared field whose
                # value is an unsupported shape IS handled by the walk below.
                continue
            raw = arguments[field]
            values.extend(
                _walk_strings(raw, surface=f"{tool_name}:{field}",
                              problems=local_problems, counter=counter, path=field)
            )

    total_chars = sum(len(text) for _, text in values)
    if total_chars > MAX_TOTAL_CHARS:
        local_problems.append(
            ExtractionProblem(
                surface=f"{tool_name}:<args>",
                reason=f"payload is {total_chars} chars, over the {MAX_TOTAL_CHARS} bound",
            )
        )

    if problems is not None:
        problems.extend(local_problems)
    elif local_problems:
        raise ValueError(
            f"incomplete payload extraction for {tool_name}: "
            + "; ".join(p.reason for p in local_problems)
        )

    payloads: list[Payload] = []
    for field_name, text in values:
        if not text.strip():
            continue
        payloads.append(
            Payload(
                surface=f"{tool_name}:{op + '/' if op else ''}{field_name}",
                field=field_name,
                # NEVER TRUNCATED. A canary past a clip point is a leak the report
                # calls clean; the size bound above invalidates instead.
                text=text,
                destination_class=kind,
            )
        )
    return tuple(payloads)
