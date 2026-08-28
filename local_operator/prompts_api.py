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


#: Spellings of the closing tag that a language model reads as a close, since
#: the consumer is a model rather than a strict parser: mixed case, whitespace
#: either side of the slash, a hyphen or space or repeat for the underscore, and
#: a trailing self-closing slash. Neutralized before interpolation because an
#: AGENT PROFILE prompt reaches this string and ``import_agent`` copies that
#: verbatim out of a downloaded marketplace archive, so a third-party agent
#: could otherwise close the tag early and have its remainder render as though
#: it were packaged prompt.
#:
#: NOT exhaustive, and deliberately not claimed to be: a blocklist of spellings
#: is a losing game against homoglyphs. Zero-width separators inside the name
#: are covered below; a fullwidth or Cyrillic lookalike letter is not, and
#: normalizing the operator's own prose to catch it costs more than it buys.
#: The escape is defence-in-depth on prompt text, not an authorization
#: boundary; nothing downstream trusts the delimiter for a security decision.
_ZERO_WIDTH = r"\u200b-\u200f\u2060\ufeff"
_CLOSING_TAG_RE = re.compile(
    rf"<[\s{_ZERO_WIDTH}]*/[\s{_ZERO_WIDTH}]*user[\s{_ZERO_WIDTH}_-]*instructions"
    rf"[\s{_ZERO_WIDTH}]*/?[\s{_ZERO_WIDTH}]*>",
    re.IGNORECASE,
)


def _render_tool_inventory(tools: Sequence[AgentTool]) -> str:
    """One compact line per visible tool; schemas ride in the provider tools
    array, so the prompt only needs name + one-line description."""
    lines = [
        f"- {tool.name}: {tool.description}"
        for tool in tools
        if not tool.hidden and tool.description
    ]
    return "\n".join(lines)


#: Appended to the tool inventory when the session has no browser tool. The
#: builder is createIf-gated (a browser needs cmux, and this package ships no
#: browser engine), so on a host without cmux the model can only observe an
#: ABSENCE — and an absence reads as "arrange your own". Measured: asked for
#: before/after screenshots of a local dev server, a session wrote a playwright
#: script and spent 23 s on ``playwright install chromium``. A downloaded
#: browser cannot carry the user's logins and the user cannot reach into it, so
#: it is not a smaller version of the real thing; it is a dead end that looks
#: like progress. Naming the absence and the reason costs three lines and only
#: ships when the tool is genuinely missing — the inventory is never told about
#: a tool that cannot work.
_NO_BROWSER_NOTE = (
    "\n\nThis session has NO browser tool: browser automation here runs through "
    "the cmux terminal, and no cmux CLI is reachable on this host. Do not "
    "substitute one — never install or script a browser engine (playwright, "
    "puppeteer, a downloaded Chromium) to load a page or capture a screenshot. "
    "For page text use `bash` with curl; when a task genuinely needs a rendered "
    "screenshot, say it is unavailable and why."
)


def build_system_blocks(
    tools: Sequence[AgentTool],
    skills_block: str,
    env_details: str,
    date_str: str,
    goal: str = "",
    user_instructions: str = "",
    repo_guidance: str = "",
    credentials: Sequence[str] | None = None,
    team_brief: str = "",
    agent_brief: str = "",
    model_label: str = "",
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

    ``goal`` (the session objective set by ``/goal``) and ``credentials``
    (the names of session secrets set by ``/credential`` or ``ask``) share
    that volatile tail block rather than adding a fifth one: that keeps the
    arity fixed, and an edited goal or a newly stored key then invalidates
    only the tail instead of the whole conversation prefix. Sessions that
    never use either feature pay nothing for them.

    ``user_instructions`` (the operator's standing customization, read once at
    session start from ``system_prompt.md``) rides the HEAD block instead,
    appended to the packaged persona. It belongs there because it is exactly
    as stable as the persona — a file the operator edits between sessions,
    never within one — so it costs nothing in cache churn, and because it must
    outrank nothing: standing user preference is part of who the assistant is,
    not a per-turn instruction competing with the live conversation. Keeping
    it out of the tail also stops a long instructions file from being re-sent
    ahead of every volatile change.
    """
    instructions = render_template("system.md", {})
    if repo_guidance.strip():
        # Same head-block, read-once discipline as user_instructions: the
        # files are part of the project's standing state, edited between
        # sessions, never within one.
        instructions = f"{instructions}\n\n{repo_guidance.strip()}"
    if user_instructions.strip():
        # Tagged, not merged: the model must be able to tell the operator's
        # standing customization apart from the packaged rules above it, and
        # a delimiter is what stops a long instructions file from reading as
        # a continuation of the persona's final bullet.
        #
        # The closing tag is neutralized first. The global file is
        # self-authored, so escaping it there is only tidiness; the same
        # string also carries an imported agent profile's prompt, which is
        # untrusted text.
        safe = _CLOSING_TAG_RE.sub("<\\/user_instructions>", user_instructions.strip())
        instructions = (
            f"{instructions}\n\n## User's custom instructions\n\n"
            "The operator set these standing preferences for every session on "
            "this machine. Follow them as their default expectations; a "
            "direct instruction in the conversation still wins.\n\n"
            f"<user_instructions>\n{safe}\n</user_instructions>"
        )
    inventory = f"## Available tools\n\n{_render_tool_inventory(tools)}"
    # Membership, not visibility: a hidden tool is still callable, and telling
    # the model a browser does not exist while one answers would be worse than
    # saying nothing.
    if not any(tool.name == "browser" for tool in tools):
        inventory = f"{inventory}{_NO_BROWSER_NOTE}"
    env_block = f"Today is {date_str}."
    if env_details:
        env_block = f"{env_block}\n\n{env_details}"
    if model_label.strip():
        # The running model, so the assistant knows which model it currently is
        # rather than guessing (a subagent naming itself in a review byline, a
        # model reasoning about its own context window or capabilities). This
        # rides the byte-stable env HEAD block, not the volatile tail, because
        # within one turn-loop the model does not change: a deliberate
        # ``set_model`` or a failover fallback takes effect at the NEXT turn
        # boundary, which re-renders this block from the session's live model,
        # and the switch itself is separately announced as a
        # ``session_model_switch`` message so the model notices the change
        # rather than only seeing a different static line.
        env_block = f"{env_block}\n\nModel: {model_label.strip()}"

    tail = skills_block or "<skills/>"
    if goal:
        # Phrased as a standing objective so the model carries it as context
        # for every turn instead of re-acknowledging a fresh instruction.
        tail = (
            f"{tail}\n\n<goal>\nThe user's standing objective for this "
            f"session:\n{goal}\n</goal>"
        )
    if team_brief.strip():
        # A /team launch stamps the group's collaboration and project briefs
        # here rather than in the cached head: attaching a team mid-session
        # must not invalidate the persona prefix, and a team is a grouping
        # for THIS conversation, not a machine-wide preference.
        tail = f"{tail}\n\n<team>\n{team_brief.strip()}\n</team>"
    if agent_brief.strip():
        # `/agent <name>` rides the tail for the same cache reason as the team
        # brief. AFTER `<team>` deliberately: an agent attached mid-session is
        # the more recent, more specific instruction, and later placement is
        # how the model reads precedence when the two briefs disagree.
        tail = f"{tail}\n\n<agent>\n{agent_brief.strip()}\n</agent>"
    names = [name for name in (credentials or ()) if name]
    if names:
        # Names only. The values live in process memory and are injected into
        # bash; putting a value (or even a reversible placeholder) here would
        # ship the secret to the provider on every later turn.
        listed = "\n".join(f"- `{name}`" for name in names)
        tail = (
            f"{tail}\n\n<session-credentials>\n"
            "The operator has handed this session credentials you can USE but "
            "never READ. Each name below is an environment variable on every "
            "`bash` command; the real value is never visible to you. When the "
            "user says they added a key or credential, these names are what to "
            "use — `list_variables` also lists them.\n\n"
            "- NEVER print, echo, log, commit, or write one of these values. "
            "If a command would display it, do not run that command.\n"
            "- Prefer letting the child process inherit the variable over "
            "inlining it in a command string, so the value never reaches a "
            "shell history or a rendered command line.\n"
            "- These live in memory for this session only. Asked to persist "
            "one, put it in a real secrets manager or vault — never a "
            "dotfile in the repo.\n\n"
            f"{listed}\n"
            "</session-credentials>"
        )

    return [instructions, inventory, env_block, tail]
