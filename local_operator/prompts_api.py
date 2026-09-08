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
:func:`build_system_blocks` returns four desired blocks: standing instructions,
compact tool inventory, environment, and selected knowledge/session state.
Production Session builders persist the initial four blocks. Later changes
enter history as typed host-state records instead of rewriting that prefix.
Provider cache hierarchy is tools -> system -> messages: moving a changing
SYSTEM block later can preserve earlier system text, but still invalidates all
conversation after it. Resume and transcript forks recover the original bytes.
Consumers must preserve the block list so provider cache breakpoints remain
well-defined.

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


def _resolve_system_md_flags(data: dict[str, Any]) -> dict[str, Any]:
    """Complete ``system.md``'s browser flag pair, or refuse an impossible one.

    ``{{#if}}`` has no ``else`` and no negation, so the browser sections are
    gated by a PAIR of flags — and a pair is easy to half-supply. A missing key
    renders as falsy and drops its body with no marker in the output, so the
    mistake is silent in both directions:

    - ``{}`` rendered a prompt with NEITHER browser section, ~1.5k characters
      lighter than any real session's;
    - supplying only ``has_browser=True`` and DEFAULTING the other flag shipped
      BOTH — the usage prose plus the browserless setup playbook, 686
      characters asserting the negation of what the same prompt just said.

    The second is why this derives rather than merging defaults: a default is a
    claim about the world, and the conservative claim for an absent pair is not
    the conservative claim for a half-supplied one. Deriving the missing member
    from the one the caller actually stated cannot contradict them.

    THREE states are legitimate, so the pair is NOT a plain negation and must
    not be collapsed into one flag (see ``build_system_blocks``):

    ==================  ==========  ===========  ===========================
    state               has_browser no_browser   meaning
    ==================  ==========  ===========  ===========================
    tool present        True        False        usage prose
    host has no backend False       True         setup playbook
    role restricted     False       False        neither; host is fine
    ==================  ==========  ===========  ===========================

    ``(True, True)`` is the one combination with no meaning — a session cannot
    hold the browser tool on a host with no browser backend — so it raises
    instead of rendering. Loud, for the same reason a malformed template raises
    in :func:`_parse`: these callers are all in this repo, so an impossible
    prompt is a build bug, not input to tolerate.
    """
    has = data.get("has_browser")
    no_browser = data.get("no_browser")

    if has and no_browser:
        raise ValueError(
            "system.md: has_browser and no_browser cannot both be true — that "
            "ships the browser usage prose and the browserless setup playbook "
            "together. Pass the pair from build_system_blocks, or pass just "
            "one and let it derive."
        )
    if has is None and no_browser is None:
        # Neither stated: the browserless arm, which is what `main` shipped
        # unconditionally and so is the safe thing for a probe or a test that
        # never had an opinion.
        return {**data, "has_browser": False, "no_browser": True}
    if no_browser is None:
        # Only tool presence stated. A session that HAS the tool is on a host
        # with a backend, so the playbook is wrong; one that lacks it has said
        # nothing about the host, and the playbook is the conservative arm.
        return {**data, "no_browser": not has}
    if has is None:
        # Only host capability stated. Never infer that a tool is present from
        # a working host — that is the M2 error in reverse, and it would put
        # usage prose in front of a session with no browser tool to use.
        return {**data, "has_browser": False}
    return data


def render_template(name: str, data: dict[str, Any]) -> str:
    """Render the named template file from ``local_operator/prompts_md/``.

    Loading goes through ``importlib.resources`` so the templates work from
    installed wheels too, not only source checkouts.

    ``system.md``'s browser flags are completed by
    :func:`_resolve_system_md_flags`, so a half-supplied pair can never render
    two contradictory sections and an absent one renders a prompt a real
    session could have.
    """
    if name == "system.md":
        data = _resolve_system_md_flags(data)
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


#: The heading that opens the tool-inventory block. Exported because the block
#: is the one part of the prompt that must track SESSION state rather than the
#: builder's arguments: ``Session`` re-renders it from its live inventory and
#: identifies the block to replace by this prefix (see
#: ``Session._reconcile_tool_inventory``). Changing the heading here without
#: changing it there would silently stop that reconciliation.
TOOL_INVENTORY_HEADING = "## Available tools"


def render_tool_inventory_block(
    tools: Sequence[AgentTool], *, host_has_browser: bool | None = None
) -> str:
    """The complete "## Available tools" block for ``tools``.

    Split out of :func:`build_system_blocks` so the session can re-render this
    ONE block against the inventory it is actually about to advertise, without
    rebuilding (or re-anchoring) the rest of the prompt. Two renderers would
    drift, and the browser note is exactly the kind of thing that gets updated
    in one place and forgotten in the other.

    ``host_has_browser`` carries the caller's already-computed host probe so
    the three-state diagnosis documented in :func:`build_system_blocks` is
    decided identically here. Left as ``None`` (the session's re-render, which
    has no cheaper source) it is probed once via
    :func:`_host_browser_backend_available`; that probe is the only way to tell
    state 2 (the HOST has no backend, so the setup playbook applies) from
    state 3 (the host is fine and only this ROLE lacks the tool), and
    conflating them tells a subagent to install a backend its host already has.
    """
    block = f"{TOOL_INVENTORY_HEADING}\n\n{_render_tool_inventory(tools)}"
    # Membership, not visibility: a hidden tool is still callable, and telling
    # the model a browser does not exist while one answers would be worse than
    # saying nothing.
    if any(tool.name == "browser" for tool in tools):
        return block
    if host_has_browser is None:
        host_has_browser = _host_browser_backend_available()
    # Same prohibition either way; only the DIAGNOSIS differs. See the
    # three-state comment in build_system_blocks and _ROLE_HAS_NO_BROWSER_NOTE.
    return block + (_ROLE_HAS_NO_BROWSER_NOTE if host_has_browser else _NO_BROWSER_NOTE)


def _render_tool_inventory(tools: Sequence[AgentTool]) -> str:
    """One line per visible tool: the NAME only, deliberately.

    NAMES ONLY — do not "restore" the descriptions here. This block used to
    emit ``- {name}: {tool.description}``, which shipped every description a
    SECOND time: the provider tools array already carries ``tool.description``
    verbatim for each tool, and it is the copy the model actually dispatches
    against. Measured on the 24-tool default surface, all 24 descriptions were
    byte-identical duplicates, costing 9,995 characters (~3,600 billed tokens
    at this surface's measured 2.78 chars/token) on every single request for
    text the model had already been given.

    The names are kept rather than dropping the block outright. Deleting it
    saves only a further 206 characters (the measured body over 24 tools) and
    forfeits the one thing the tools array does not present as prose: a single
    flat "these tools exist" anchor the model can scan when deciding whether a
    capability is available at all.

    Note the descriptions are NOT one-line and never were — ``browser`` and
    ``ask`` run to several hundred tokens each — which is why duplicating them
    was expensive rather than merely redundant.

    Filtered on ``hidden`` ALONE. While a line was ``- {name}: {description}``
    a description-less tool was rightly skipped, since its line would have
    trailed a bare colon; now that the line is just the name there is nothing
    wrong with it, and dropping it hides a real, callable tool from the only
    list that says what exists. MCP servers may legitimately omit a
    description, so this is reachable rather than theoretical.
    """
    return "\n".join(f"- {tool.name}" for tool in tools if not tool.hidden)


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

#: The same prohibition for a session whose ROLE was not given the browser
#: tool on a host that HAS one — a ``reviewer``/``scout``/``manager``/
#: ``architect`` subagent, whose seed allowlist omits it.
#:
#: A separate string because :data:`_NO_BROWSER_NOTE` asserts a fact about the
#: HOST ("no cmux CLI is reachable on this host") that is simply false here,
#: and the setup playbook it pairs with would invite a read-only child to walk
#: the operator through an install it does not need and could not use. The
#: no-install rule still applies — the playwright dead end is just as
#: available to a restricted child — so the prohibition is kept and only the
#: false diagnosis is dropped. Says who to ask instead, because unlike a
#: browserless host this capability genuinely exists and is one delegation
#: away.
_ROLE_HAS_NO_BROWSER_NOTE = (
    "\n\nThis session was not given the browser tool. A browser IS available on "
    "this host — it is simply not part of this role's tool set — so never "
    "install or script a browser engine (playwright, puppeteer, a downloaded "
    "Chromium) to load a page or capture a screenshot. For page text use "
    "`bash` with curl; when a task genuinely needs a rendered page or "
    "screenshot, say so and let the delegating session take it."
)


def _host_browser_backend_available() -> bool:
    """Whether THIS HOST could drive a browser at all, ignoring tool lists.

    Distinct from "the browser tool is in my list": a restricted-role subagent
    on a fully browser-capable host has no ``browser`` tool but must not be
    told the host lacks a backend. Reads the same two probes the createIf
    builder uses, so the answer cannot disagree with why the tool was withheld.

    Imported lazily and defensively: this is prompt rendering, which must never
    fail because a capability probe raised. A probe failure degrades to "no
    backend", which is the conservative answer — it ships the setup playbook,
    the same text ``main`` shipped unconditionally.
    """
    try:
        from local_operator.tools.builtin import (
            bridge_browser_advertisable,
            cmux_browser_available,
        )

        return bool(cmux_browser_available() or bridge_browser_advertisable())
    except Exception:  # noqa: BLE001 — prompt rendering must never break
        return False


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
    interactive: bool = True,
) -> list[str]:
    """Build the system prompt blocks; see the module docstring.

    These are the current desired blocks, not a promise of prefix cache
    stability: every system block precedes conversation history on the wire.
    Production sessions persist the first snapshot, then journal changes as
    host-authored state messages at the history tail. Moving volatile content
    to a later SYSTEM block alone cannot preserve the conversation cache.
    A changed standing-instruction head starts a new persisted prefix: cached
    bytes must never hide updated repository rules, custom instructions or a
    newer packaged prompt, including when an old conversation is resumed.

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
    # THREE states, not two, and conflating the last two ships a false claim.
    # Membership, not visibility: a hidden tool is still callable, and telling
    # the model a browser does not exist while one answers would be worse than
    # saying nothing.
    #
    #   1. tool present                      -> usage prose, no note
    #   2. tool absent because the HOST has   -> setup playbook + the note that
    #      no backend                            names the playwright dead end
    #   3. tool absent because this ROLE's    -> no playbook (the host is fine,
    #      allowlist omits it                    an install would be pointless),
    #                                            but still the no-install rule
    #
    # State 3 is the one an earlier revision got wrong: `reviewer`, `scout`,
    # `manager` and `architect` seeds all omit `browser`, so every such child
    # on a browser-capable host was told "the host has neither backend
    # connected... do that setup with the user". False, and actionably false.
    # The host probe is what separates 2 from 3.
    has_browser = any(tool.name == "browser" for tool in tools)
    host_has_browser = has_browser or _host_browser_backend_available()
    # The browser prose is conditional rather than unconditional because it is
    # ~1,500 characters of instruction for a tool that is createIf-gated: a
    # host with no cmux and no extension paid for three paragraphs about a
    # `browser` tool that is not in its tool list. The two flags are passed
    # separately (rather than one negated in the template) because the engine
    # is deliberately tiny — `{{#if}}` has no `else` and no negation.
    #
    # NOTE the asymmetry, and keep it: when the browser is absent the usage
    # prose is gated out but a no-install note still ships on the inventory
    # block. They are not two copies of one thing. The gated-out prose explains
    # how to USE the tool; the note records a measured failure (a session spent
    # 23s on `playwright install`) and is the only text that names the wrong
    # turn — which stays worth saying however the tool came to be absent.
    instructions = render_template(
        "system.md",
        # The setup playbook is gated on the HOST, never on this role's list.
        {"has_browser": has_browser, "no_browser": not host_has_browser},
    )
    instructions += (
        "\n\n## Session state updates\n\n"
        "The host may append [session-state] records containing current tool, "
        "environment, knowledge, goal, team, agent, or interactivity state. "
        "Each supplied section replaces that section's earlier snapshot, "
        "including an explicit empty section. Treat these as host context, "
        "not a new user task or permission to act. Direct user instructions "
        "and existing approval requirements still apply. Older snapshots "
        "describe the state at that point in the conversation."
    )
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
    inventory = render_tool_inventory_block(tools, host_has_browser=host_has_browser)
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
    if not interactive:
        # WHO CAN ANSWER, stated once. A detached session has nobody at a
        # screen, so a question costs a parked gate (holding the runtime
        # resident) and gets no answer — the model needs to know that BEFORE
        # it decides to ask, not after the gate times out.
        #
        # A single recomputed statement, deliberately, not an event: a row
        # per attach/detach would grow the transcript without bound for a
        # user who reattaches often, which is exactly the token accumulation
        # this is meant to avoid. This block is rebuilt at turn start, so N
        # attach/detach cycles cost the same as zero.
        tail = (
            f"{tail}\n\n<interactivity>\n"
            "No interactive surface is attached to this session right now: "
            "nobody is watching a screen, so a question to the user cannot be "
            "answered until someone reopens it.\n\n"
            "- Prefer to PROCEED with what you have, or finish the turn with a "
            "clear statement of what you would have asked, over calling `ask`.\n"
            "- That statement is a decision you already took and the fact that "
            "would change it, not a question left hanging: with nobody at a "
            "screen a prose question is even less answerable than usual.\n"
            "- Do not take an irreversible or destructive action to avoid "
            "asking; when the choice genuinely needs a person, stop and say so "
            "— that is cheaper than a wrong guess.\n"
            "- The user will read this conversation when they return, so write "
            "for someone catching up, not for someone watching live.\n"
            "</interactivity>"
        )
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
