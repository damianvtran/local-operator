"""Prompt rendering for the new harness.

Why this exists
---------------
The legacy ``prompts.py`` kept every prompt as Python string constants (176 KB
of source, 19 vertical instruction blobs), which made prompts undiffable,
unhot-reloadable, and the main reason a classifier LLM call existed at all.
The rewrite externalizes prompt text to markdown templates in
``local_operator/prompts_md/`` and renders them with a deliberately tiny
handlebars-ish engine — no dependency, no partials, no helpers: just
``{{var}}``, ``{{#if var}}...{{/if}}``, and ``{{#each items}}...{{/each}}``.

Block layout and caching
------------------------
:func:`build_system_blocks` returns the system prompt as blocks ordered from
most-stable to most-volatile:

- ``[0]`` instruction block — the persona and standing rules from
  ``system.md`` rendered WITHOUT skills, date, or environment. Byte-stable
  across every turn of a session (and across sessions for the same build).
- ``[1]`` tool inventory block — one compact line per visible tool. Stable
  across a turn; changes only when the tool set changes.
- ``[2]`` (present only when skills matched this turn) skills block — the
  per-turn selected skills listing, verbatim. Its absence never shifts the
  stable prefix: the volatile env block is always LAST.
- last block — environment block: date (a calendar date, never a timestamp,
  so the prefix is byte-stable for a whole local day) and volatile
  environment details (cwd, platform, session facts).

Providers put prompt-cache breakpoints BETWEEN these blocks (e.g. Anthropic
``cache_control`` after block 0/1) so a change in the volatile tail never
invalidates the cached stable prefix. Consumers must pass the blocks to
``ChatRequest.system_blocks`` as a list and never join them into one string.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib.resources import files
from typing import Any, Literal, TypeAlias

from local_operator.harness.types import AgentTool

# ---------------------------------------------------------------------------
# Template engine
# ---------------------------------------------------------------------------

#: One template tag: ``{{name}}``, ``{{#if name}}``, ``{{#each name}}``,
#: ``{{/if}}``, ``{{/each}}``. Names may be dotted paths.
_TAG_RE = re.compile(r"\{\{\s*(#if\s+[\w.]+|#each\s+[\w.]+|/if|/each|[\w.]+)\s*\}\}")

#: A lexed template piece: literal text, or the body of one ``{{...}}`` tag.
Token: TypeAlias = "tuple[Literal['text', 'tag'], str]"

#: A parsed template node. ``text``/``var`` carry a payload string (the
#: literal, or the dotted data path); ``if``/``each`` carry the path plus the
#: body they guard or repeat. Kept as tuples rather than classes because the
#: renderer walks them in the hot path of every prompt build.
TextNode: TypeAlias = "tuple[Literal['text', 'var'], str]"
BlockNode: TypeAlias = "tuple[Literal['if', 'each'], str, list[Node]]"
Node: TypeAlias = "TextNode | BlockNode"

#: Compiled template cache keyed by template file name. Templates ship in the
#: package and never change at runtime, so one parse per name is enough.
_TEMPLATE_CACHE: dict[str, list[Node]] = {}


def _lookup(data: dict[str, Any], name: str) -> Any:
    """Resolve a dotted path against the data dict; missing -> ``None``."""
    current: Any = data
    for part in name.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _tokenize(text: str) -> list[Token]:
    """Split template text into ('text', str) / ('tag', str) tokens."""
    tokens: list[Token] = []
    pos = 0
    for match in _TAG_RE.finditer(text):
        if match.start() > pos:
            tokens.append(("text", text[pos : match.start()]))
        tokens.append(("tag", match.group(1)))
        pos = match.end()
    if pos < len(text):
        tokens.append(("text", text[pos:]))
    return tokens


def _parse(tokens: list[Token], index: int, terminators: tuple[str, ...]) -> tuple[list[Node], int]:
    """Recursive-descent parse of tokens into a node list.

    Nodes: ``("text", s)``, ``("var", name)``, ``("if", name, children)``,
    ``("each", name, children)``. Unbalanced blocks raise ``ValueError`` —
    templates are authored by us, so a malformed template is a build bug that
    should fail loudly, not render half-way: a stray closing tag is as much a
    bug as a missing one.
    """
    nodes: list[Node] = []
    while index < len(tokens):
        kind, value = tokens[index]
        if kind == "text":
            nodes.append(("text", value))
            index += 1
            continue
        tag = value
        if tag in terminators:
            return nodes, index
        index += 1
        if tag in ("/if", "/each"):
            raise ValueError(f"stray {{{{{tag}}}}} without a matching opener")
        if tag.startswith("#if "):
            name = tag[4:].strip()
            children, index = _parse(tokens, index, ("/if",))
            index += 1  # consume /if
            nodes.append(("if", name, children))
        elif tag.startswith("#each "):
            name = tag[6:].strip()
            children, index = _parse(tokens, index, ("/each",))
            index += 1  # consume /each
            nodes.append(("each", name, children))
        else:
            nodes.append(("var", tag))
    if terminators:
        raise ValueError(f"missing closing tag (expected {terminators[0]})")
    return nodes, index


def _render_nodes(nodes: list[Node], data: dict[str, Any], out: list[str]) -> None:
    for node in nodes:
        # Narrowing reads node[0] directly: assigning it to a local first
        # would not discriminate the tuple union for a type checker.
        if node[0] == "text":
            out.append(node[1])
        elif node[0] == "var":
            value = _lookup(data, node[1])
            if value is not None:
                out.append(str(value))
        elif node[0] == "if":
            if _lookup(data, node[1]):
                _render_nodes(node[2], data, out)
        elif node[0] == "each":
            items = _lookup(data, node[1])
            if not isinstance(items, (list, tuple)):
                continue
            for item in items:
                child = dict(data)
                if isinstance(item, dict):
                    child.update(item)
                child["this"] = item
                _render_nodes(node[2], child, out)


def render_string(template: str, data: dict[str, Any]) -> str:
    """Render template text against ``data`` (no file loading).

    Missing variables render as empty strings; ``{{#if}}`` on a missing or
    falsy value drops its body. Inside ``{{#each}}``, ``{{this}}`` is the
    current item and dict items also expose their keys as variables.
    """
    nodes, _ = _parse(_tokenize(template), 0, ())
    out: list[str] = []
    _render_nodes(nodes, data, out)
    return "".join(out)


def _read_template_text(name: str) -> str:
    """Read a template by name; resources first, filesystem fallback.

    The fallback matters for editable/dev installs where package-data wiring
    may not ship ``*.md`` through ``importlib.resources``.
    """
    try:
        return files("local_operator.prompts_md").joinpath(name).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        from pathlib import Path

        return (Path(__file__).parent / "prompts_md" / name).read_text(encoding="utf-8")


def _load_template(name: str) -> list[Node]:
    nodes = _TEMPLATE_CACHE.get(name)
    if nodes is None:
        nodes, _ = _parse(_tokenize(_read_template_text(name)), 0, ())
        _TEMPLATE_CACHE[name] = nodes
    return nodes


def render_template(name: str, data: dict[str, Any]) -> str:
    """Render the named template file from ``local_operator/prompts_md/``.

    Loading goes through ``importlib.resources`` so the templates work from
    installed wheels too, not only source checkouts.
    """
    out: list[str] = []
    _render_nodes(_load_template(name), data, out)
    return "".join(out)


# ---------------------------------------------------------------------------
# System prompt blocks
# ---------------------------------------------------------------------------


def _render_tool_inventory(tools: Sequence[AgentTool]) -> str:
    """One compact line per visible tool; schemas ride in the provider tools
    array, so the prompt only needs name + one-line description."""
    lines = [
        f"- {tool.name}: {tool.description}"
        for tool in tools
        if not tool.hidden and tool.description
    ]
    return "\n".join(lines)


def build_system_blocks(
    tools: Sequence[AgentTool],
    skills_block: str,
    env_details: str,
    date_str: str,
    goal: str = "",
) -> list[str]:
    """Build the system prompt blocks; see the module docstring.

    Block order is a cache-layout decision: the stable head (instructions,
    tool inventory, env) is byte-stable for the session, and the per-turn
    volatile content rides LAST. A volatile block mid-prefix would
    invalidate every message after it on each selection change — the bench
    measured 40% stability with skills at index 2; at the tail the
    conversation prefix stays warm. The list is fixed-arity (placeholder when
    nothing matched) so the wire clients' breakpoint derivation never shifts.
    The env block carries the calendar date — never a timestamp — plus
    volatile environment facts.

    ``goal`` (the session objective set by ``/goal``) shares that volatile
    tail block rather than adding a fifth one: that keeps the arity fixed,
    and an edited goal then invalidates only the tail instead of the whole
    conversation prefix.
    """
    instructions = render_template("system.md", {})
    inventory = f"## Available tools\n\n{_render_tool_inventory(tools)}"
    env_block = f"Today is {date_str}."
    if env_details:
        env_block = f"{env_block}\n\n{env_details}"

    tail = skills_block or "<skills/>"
    if goal:
        # Phrased as a standing objective so the model carries it as context
        # for every turn instead of re-acknowledging a fresh instruction.
        tail = (
            f"{tail}\n\n<goal>\nThe user's standing objective for this "
            f"session:\n{goal}\n</goal>"
        )

    return [instructions, inventory, env_block, tail]
