"""Deferred tool schemas: which built-in tools ship their schema on demand.

WHY THIS EXISTS. Every tool in a session's inventory used to put its full
schema in the provider ``tools`` array on every request. Seventeen tools that
under 2% of sessions ever call carried ~35k characters (~12k billed tokens) on
every call of every session (measured 2026-10-08 against ``analytics.db``; see
the PR that introduced this module). A deferred tool stays in the session's
INVENTORY — it resolves, validates, passes the approval gate and answers
``tool://`` exactly as before — and only its schema is left out of the array
the provider receives, until the session activates it.

WHAT THIS IS NOT. It is not a second gating convention beside ``createIf``
(``AGENTS.md``, "The tool-surface footprint ladder"): ``createIf`` decides
whether a tool EXISTS in a session; this decides only whether its schema rides
the request prefix. Enforcement (role allowlists, declared inventories, the
approval gate) is untouched because it reads the inventory, never the array.

Why the provider accepts a call to a tool absent from ``tools``: measured, not
assumed. One request per wire in active use (Anthropic, DeepSeek, OpenRouter,
Radient), each carrying a history with a call + result for a tool NOT in that
request's array, and each asked to call a tool it saw only in prompt text —
all accepted the history and all emitted the call. The table is on the PR.

A leaf module (it imports nothing from ``tools.builtin``) so the prompt
renderer can read the purpose phrases without pulling the tool builders in.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

#: Tools whose schema is withheld from the request's tools array.
#:
#: ONE SET FOR EVERY SESSION, and that is a measurement rather than a
#: simplification. The design memo proposed a LARGER set for subagents — the
#: session-management tools a top-level session lives in and a child almost
#: never calls (``send`` 2.1% of children, ``secret`` 0.9%, ``agent`` 0.23%,
#: ``project`` 0.17%, ``sessions`` 0.02%). Every one of those five was measured
#: on a live model and EVERY one showed an adoption regression where its schema
#: was absent: the child reached for an equivalent CLI plus a guide instead of
#: the tool, at 2-4x the provider calls. A child-only set therefore bought
#: schema tokens by making the tools the child was given harder to use, which is
#: not a saving. The per-tool counts and the route each run took are on the PR.
#:
#: Chosen from usage counted PER KIND rather than blended: 93% of the 30-day
#: ledger's sessions are subagents, so a blended "share of sessions that called
#: it" would hide that ``send`` is a tool ~50% of top-level sessions use and a
#: child almost never does. Every name here is called by at most ~2% of
#: sessions of EITHER kind; ``lsp`` and ``patience`` by roughly none. ``ask`` is
#: deliberately NOT here: the ``<interactivity>`` bodies name it as the channel
#: to the operator.
# notify deliberately stays published: the completed live deferred pilot
# repeatedly called todo and falsely claimed delivery with no durable row.
# Correct reachability is worth its measured 573-billed-token publication cost.
DEFERRED_TOOLS: frozenset[str] = frozenset(
    {
        "console",
        # ``network`` was deferred and was RE-ADMITTED after measurement: QA
        # round 1 (Q1) ran the same prompt 3x on each arm and the deferred arm
        # called it 0/2 whenever ``bash`` was available, reaching instead for
        # ``read guide://network`` + ``lop network status --json`` (12 and 9
        # provider calls against base's 3 and 7). The answers were right either
        # way, so this was an ADOPTION regression rather than a capability
        # loss — and a tool whose use collapses is not a saving worth having.
        # Its schema is one of the largest, and external-surface correctness
        # outweighs the tokens. Every other deferred tool with a CLI equivalent
        # was measured the same way and kept: see the PR's adoption table.
        "team",
        "lsp",
        "patience",
        # ``ask_withdraw`` was deferred and was RE-ADMITTED after measurement
        # (CI round 4). It exists for ONE moment — the user answered a QUEUED
        # ask in chat, or the question went moot — and the only text that tells
        # a model to use it then is its OWN description: neither the queued
        # receipt nor ``ask``'s description names it. Withheld, that sentence
        # never reaches the model. Measured live (claude-sonnet-5-5, a real
        # queued ask answered in chat, one variable): deferred, the model
        # settled the ask 0/3 and it stayed ``open``; published, 3/3 and it
        # settled ``answered``. A deferred tool the model needs exactly when it
        # does not know to look is not a saving.
        #
        # Un-deferring it is the same as "publish it whenever ``ask`` is", and
        # needs no pairing rule: both tools gate on the same host ask hook
        # (``Session.set_ask_handler`` merges and removes them together), so
        # ``ask_withdraw`` is only ever held where ``ask`` is.
        "web_read",
        "project_delete",
        "team_delete",
        "read_variable",
        "list_variables",
        # ``code_requests`` joins the deferred set on the design's own
        # measurement (§E): most sessions never touch a PR, and 93% of sessions
        # are subagents. The known failure mode is adoption (``network`` was
        # re-admitted for exactly that), so the deterministic recommendation
        # and the tracked note exist to keep the tool findable, and the
        # deferral PR's A/B measurement ships as the evidence.
        "code_requests",
        "profile_proposal",
        # ``skill`` became a default tool with the profile-proposals change
        # (owner-requested skill authoring outside the Agents config run). Its
        # schema is large and most turns never author a skill, so it is held
        # off the wire like ``profile_proposal``: still resolvable by name, so
        # an owner's "create a skill" request reaches it on demand.
        "skill",
        # ``library`` is deferred on the start-context budget's arithmetic: the
        # tool is new and its published schema + description is 1,379 chars
        # (514 billed tokens) on a clean arm whose base already sits 115 tokens
        # under the 31,000 ceiling. The tool stays in the inventory, resolves,
        # validates and answers ``tool://library``; the first read publishes
        # its schema. This is the only way to keep every field while passing
        # the unraised ceiling.
        "library",
        # ``no_reply`` joins on the design's own reasoning (docs/design/
        # quiet-turns.md §4): the call has NO arguments, so no schema is needed
        # to form it, and calls to tools absent from the array were measured
        # accepted on four wires (see the module docstring). The system-prompt
        # sentence naming the rule (slice S0b, c938c60759b7, 2026-10-10) renders
        # with the tool (``has_no_reply`` in ``prompts_api``). Cost avoided:
        # roughly 60 schema tokens per request on every top-level session that
        # holds it. Its purpose phrase says "no arguments, call it directly"
        # because the 7-day ledger to 2026-11-19 showed models still reading
        # ``tool://no_reply`` ~2.2x per session that used it: a round trip for
        # a schema that has nothing in it. The named probe guarding this
        # choice: 20 scripted peer-ping turns per arm on one live model; under
        # 80% adoption removes it from this set (one line — ``network`` and
        # ``ask_withdraw`` were re-admitted on exactly this signal).
        "no_reply",
        # ``artifact_data`` is deferred on the start-context budget's arithmetic,
        # the ``library`` precedent: unpublished it measured +785 billed tokens
        # over the ceiling on both arms, and it is a niche tool (only sessions
        # that hold a claude.ai artifact link ever call it). Skills that use it
        # name it, and the purpose phrase below keeps it findable; the first
        # call or ``tool://`` read publishes its schema.
        "artifact_data",
        # ZERO-USE SCHEMAS (wave 2, 2026-11-19). A read-only 7-day audit of
        # ``analytics.db`` (380,917 turn calls; tool use counted from 470,471
        # model tool_calls) found these ten carried ~1.19B schema tokens a week
        # between them — ~1.15B net of the extra round trip a miss costs —
        # while being called 0 times each in 7 days, except ``generate_image``
        # (19) and ``tags`` (2); 30 days: ``generate_image`` 53, every other
        # one 0-6. Media tools are createIf-gated on a configured provider, so
        # the sessions paying for them are exactly the configured installs.
        # A VALID direct call needs no ``read tool://`` (``Session`` dispatches
        # it from the inventory; an invalid one publishes the schema and points
        # at the doc), so each purpose phrase below names what the tool does and
        # its required argument, making the direct call the natural move.
        # Adoption was measured live per tool family before landing (base vs
        # head, 5 prompts each; table on the PR); a family under 80%, or more
        # than one run below base, comes back out of this set — the
        # ``network``/``ask_withdraw`` precedent above.
        "generate_image",
        "generate_video",
        "video_status",
        "generate_voiceover",
        "generate_sound_effect",
        "generate_music",
        "modify_music",
        "music_status",
        "tags",
        "harness_import",
        # ``integrations`` (RFC "Radient Team Integrations" §8.1 R4): org admin
        # provisioning is rare and its op schema is wide; it is already
        # createIf-gated to interactive, signed-in sessions, and deferral keeps
        # even those at the purpose phrase until ``read tool://integrations``.
        "integrations",
    }
)

#: One short purpose phrase per deferrable tool, rendered in the inventory's
#: "schema on demand" line. The phrase is what lets a model decide to reach for
#: a tool whose description it has not been sent; keep each to a few words —
#: the full purpose is one ``read tool://<name>`` away. A deferrable tool with
#: no entry still renders (by name alone), so a missing phrase degrades rather
#: than hiding a tool; ``test_every_deferred_tool_has_a_purpose`` pins the map.
DEFERRED_TOOL_PURPOSES: Mapping[str, str] = {
    "console": "drive an interactive terminal",
    "team": "author or list teams",
    "lsp": "code intelligence",
    "patience": "proactive reply-wait timers",
    "web_read": "answer from already-searched pages",
    "project_delete": "delete a project row",
    "team_delete": "delete a team",
    "read_variable": "read one variable",
    "list_variables": "list variable names",
    "code_requests": "PR/MR status, review rounds, CI",
    "profile_proposal": "Agents profile proposals",
    "skill": "author skills",
    "library": "long-term asset catalog",
    "no_reply": "end a turn silently; takes no arguments so call it directly",
    "artifact_data": "claude.ai artifact shared data store",
    # No commas inside a phrase: the line joins entries with ", ".
    # Each media generator names its guide HERE as well as in its description:
    # a valid direct call never publishes the description, so without this the
    # "read the guide first" pointer was lost on exactly the calls that skip
    # ``tool://`` (PR #4326 review R4; image "any read first" fell 5/5 -> 4/5).
    "generate_image": "generate or edit an image from a prompt; guide://image-generation first",
    "generate_video": "generate a video from a prompt; guide://video-generation first",
    "video_status": "check a generate_video job",
    "generate_voiceover": "speak text aloud as a voiceover; guide://sound-generation first",
    "generate_sound_effect": (
        "make a sound effect from prompt + duration_s; guide://sound-generation first"
    ),
    "generate_music": "compose music from prompt + duration_ms; guide://music-generation first",
    "modify_music": "rework sections of a generated track; guide://music-generation first",
    "music_status": "check or cancel a music job",
    "tags": "list/create/assign session tag categories",
    "harness_import": "import Claude Code/Codex/omp sessions and memory",
    "integrations": "Radient org integration apps and tokens",
}

#: ``tools.defer`` — the kill switch. On by default; off publishes every
#: schema again (the pre-deferral behaviour) from the next turn.
TOOL_DEFERRAL_PATH: tuple[str, ...] = ("tools", "defer")
DEFAULT_TOOL_DEFERRAL = True


def tool_deferral_enabled(values: Mapping[str, object] | None = None) -> bool:
    """Read ``tools.defer``; anything but an explicit ``false`` means on.

    ``values`` is a config mapping already in hand (the config watcher's
    snapshot); ``None`` reads a fresh ``ConfigManager``. Never raises: a
    corrupt ``config.yml`` must not cost a session its tools array, and the
    default is the behaviour this ships as.
    """
    if values is None:
        try:
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir

            section = ConfigManager(config_dir()).get_config_value(TOOL_DEFERRAL_PATH[0], None)
        except Exception:  # noqa: BLE001 — a config read never breaks a session
            return DEFAULT_TOOL_DEFERRAL
    else:
        section = values.get(TOOL_DEFERRAL_PATH[0]) if isinstance(values, Mapping) else None
    stored = section.get(TOOL_DEFERRAL_PATH[1]) if isinstance(section, Mapping) else None
    if isinstance(stored, bool):
        return stored
    return DEFAULT_TOOL_DEFERRAL


def deferred_tool_names(pinned: Collection[str] = ()) -> frozenset[str]:
    """The deferred names, minus any the session PINS.

    A caller that NAMES a tool — a role's ``tools:`` list, a team manager's, or
    a host declaration (``lop exec --tools console,bash``) — asked for it as
    part of what the run IS, so withholding its schema would make a tool the
    caller already chose the one the model has to discover. The lopdev
    ``manager`` seed names ``project`` and ``read_variable``/``list_variables``.
    """
    return DEFERRED_TOOLS - frozenset(pinned)


def render_deferred_tools_line(names: Collection[str]) -> str:
    """The inventory's single line naming the tools whose schema is on demand.

    Fixed for a session's life — it lists the DEFERRABLE tools the session
    holds, not the ones still unpublished — so activating a tool does not move
    the inventory block and therefore costs no ``[session-state]`` delta.
    Ordered by name so the bytes do not depend on inventory order.
    """
    if not names:
        return ""
    entries = []
    for name in sorted(names):
        purpose = DEFERRED_TOOL_PURPOSES.get(name)
        entries.append(f"{name} ({purpose})" if purpose else name)
    return (
        "Schema on demand — call these directly by name, or read tool://<name> "
        "for the parameters: " + ", ".join(entries) + "."
    )


__all__ = [
    "DEFAULT_TOOL_DEFERRAL",
    "DEFERRED_TOOLS",
    "DEFERRED_TOOL_PURPOSES",
    "TOOL_DEFERRAL_PATH",
    "deferred_tool_names",
    "render_deferred_tools_line",
    "tool_deferral_enabled",
]
