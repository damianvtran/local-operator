"""Read-only enforcement — the safety core (contract §6).

One module, one consumer-facing entry point: :func:`monitor_call_verdict` —
:func:`readonly_verdict` (the safety verdict) followed by an additive shape
check, so a call that arms is a call that can run. A
monitor may wrap a call **iff the harness's effective approval tier for that
call is ``read``** — the same computation the loop gates on
(``tool.call_approval_tier(args) if tool.call_approval_tier else
tool.approval_tier``). What makes a monitor different from an interactive read
is that it repeats *unattended*, so the verdict is enforced at arm time (fail
loudly) and re-checked at run time.

Three classes need this module because the existing tier fields alone cannot
answer for them:

- **bash.** Its tier is static ``exec``, so §6.4 defines what "provably
  read-only" means: exactly one logical line, split at unquoted ``|`` into
  stages, every stage a bare-named command from a deliberately short
  allow-list with default-deny flags. Two layers — command words, then flags —
  both deny by default; what is not enumerated is refused.
- **MCP.** ``annotations.readOnlyHint is True`` on the server's ``tools/list``
  row is the only key (§6.5); absent or false is refused.
- **Per-op read-tier tools** (hub, jobs, todo, agent, team, project, lsp,
  network, console). A read-tier tool whose op surface is not wholly observing
  is monitorable only for the verbs in :data:`OBSERVING_VERBS` — and an
  op-bearing read-tier tool absent from the map is refused **fail-closed**: a
  new such tool must add itself and its test, it cannot be admitted by
  accident (round-1 review F6, the class not one tool).

Two deliberate divergences from the bare text of §6.2, both fail-closed, both
recorded here because a reviewer will look:

1. ``network`` and ``console`` carry dynamic tiers whose read answers ARE
   listed in §6.2's "any other dynamic-tier tool … honored as-is" row. The
   F6 rule and that row pull apart for them (they are op-bearing AND
   dynamic); this module resolves the tension toward the F6 rule by
   enumerating their observing verbs explicitly — nothing is admitted without
   a line and a test.
2. ``console {method:"screenshot"}`` is refused even though the console tool's
   own tier calls it read: it writes a PNG to disk, and the monitor definition
   (contract Appendix B) is stricter than the tier vocabulary — "no filesystem
   writes". The other three console read methods are observing.

Refusals are full sentences naming the tool, the reason, and (where one
exists) a compliant alternative; the §6.9 matrix asserts each one's
discriminating phrase so a reordered message fails a test rather than
drifting.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from typing import Any

from local_operator.harness.types import AgentTool

# ---------------------------------------------------------------------------
# Tool classes with a fixed verdict
# ---------------------------------------------------------------------------

_EVAL_REFUSAL = (
    "monitor can't watch eval: arbitrary Python cannot be proven read-only, and "
    "monitors run unattended. Watch a read-only command (bash), a URL (web_fetch), "
    "or a tool that declares readOnlyHint instead."
)

#: Read-tier tools that are nevertheless never observable by a monitor,
#: each with its reason. ``task`` is static write; ``ask``/``wait`` are static
#: read but send a message / park the turn respectively, which a check that
#: runs unattended must never do (§6.2).
_HARD_REJECT: dict[str, str] = {
    "task": (
        'monitor can\'t watch task: its approval tier is "write" and it delegates '
        "autonomous work — monitors re-run unattended, so every check must be a "
        "read-tier call."
    ),
    "ask": (
        "monitor can't watch ask: asking a question sends a message to the parent, "
        "and monitors run unattended."
    ),
    "wait": (
        "monitor can't watch wait: waiting parks the turn, which a scheduled check " "cannot do."
    ),
}

# ---------------------------------------------------------------------------
# The per-op observing map (F6)
# ---------------------------------------------------------------------------
#
#: tool name -> (verb argument name, observing verbs). ONLY these verbs may be
#: monitored; every other verb the tool offers is a refusal. An op-bearing
#: read-tier tool NOT in this map is refused fail-closed (see
#: :func:`_unmapped_verb_reason`).
OBSERVING_VERBS: dict[str, tuple[str, frozenset[str]]] = {
    "hub": ("op", frozenset({"list", "peek"})),
    "jobs": ("op", frozenset({"list", "peek"})),
    "todo": ("op", frozenset({"view"})),
    "agent": ("op", frozenset({"list", "show", "search"})),
    "team": ("op", frozenset({"list", "show"})),
    "project": ("op", frozenset({"list", "show"})),
    # lsp's verbs are all observing; keyed on its own verb parameter (the §6.2
    # "action for lsp" note).
    "lsp": ("action", frozenset({"definitions", "references", "symbols", "rename_preview"})),
    # network's READ_ACTIONS, plus `sessions` whose bare form is a listing (its
    # tier upgrades per call on the arguments, so a mutating `sessions` call is
    # refused by the tier before this map is consulted).
    "network": (
        "action",
        frozenset(
            {
                "status",
                "ls",
                "show",
                "peers",
                "log",
                "doctor",
                "credentials",
                "definitions_state",
                "sessions",
            }
        ),
    ),
    # console's four read methods minus `screenshot` (writes a PNG — see the
    # module docstring).
    "console": ("method", frozenset({"list", "status", "read"})),
    # code_requests' two ops: `list` reads this session's derived index and the
    # local fetch cache, `show` reads the cache and (at most) one conditional
    # fetch for a ref the cache does not know — a network READ with no side
    # effect, which is what a scheduled "tell me when round 2 lands" check
    # needs to re-run unattended.
    "code_requests": ("op", frozenset({"list", "show"})),
}

#: Refusals for specific (tool, verb) pairs whose reason is worth naming
#: beyond "not in the observing set" — §6.9's console-screenshot row. These
#: are deliberately the ONLY per-pair carve-outs: everything else either is
#: observing or gets the generic sentence.
_VERB_REFUSALS: dict[tuple[str, str], str] = {
    ("console", "screenshot"): (
        'monitor can\'t watch "console" method="screenshot": it writes a file to '
        "disk, and monitors admit no filesystem writes."
    ),
}

#: Property names by which a schema can declare "this tool takes a verb".
_VERB_PROPERTIES = ("op", "action", "method")


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


def readonly_verdict(
    tool: AgentTool,
    args: Mapping[str, Any],
    *,
    mcp_annotations: Mapping[str, Any] | None = None,
) -> str | None:
    """``None`` = read-only (accept). A sentence = the refusal reason (reject).

    Reads the same fields the loop reads; bash and MCP are the two classes
    where the existing fields do not answer, and this module is where their
    answer lives.
    """
    name = tool.name

    if name == "eval":
        return _EVAL_REFUSAL

    if name.startswith("mcp__"):
        annotations = mcp_annotations
        if annotations is None:
            annotations = getattr(tool, "mcp_annotations", None)
        if not _read_only_hint(annotations):
            return (
                f'monitor can\'t watch "{name}": it does not declare readOnlyHint, '
                "so the harness cannot rule out side effects."
            )
        return None

    hard = _HARD_REJECT.get(name)
    if hard is not None:
        return hard

    # bash BEFORE the tier computation: its tier is static `exec` at this head
    # (§6.2's appendix correction — PR #1696's "bash scope" is the evaluation
    # confinement, not a per-command tier), so the §6.4 evaluator is what
    # answers for it. The tier check below would refuse every bash call.
    if name == "bash":
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return 'monitor can\'t watch "bash": a bash monitor needs a "command".'
        return _bash_verdict(command)

    tier = (
        tool.call_approval_tier(dict(args))
        if tool.call_approval_tier is not None
        else tool.approval_tier
    )
    if tier != "read":
        return (
            f'monitor can\'t watch "{name}": its approval tier is "{tier}" — '
            "monitors re-run unattended, so every check must be a read-tier call."
        )

    entry = OBSERVING_VERBS.get(name)
    if entry is not None:
        key, allowed = entry
        verb = str(args.get(key) or "").strip().lower()
        if verb not in allowed:
            special = _VERB_REFUSALS.get((name, verb))
            if special is not None:
                return special
            if not verb:
                return (
                    f'monitor can\'t watch "{name}": its "{key}" argument is missing, '
                    "so the call cannot be proven observing."
                )
            allowed_terms = " or ".join(f'{key}="{item}"' for item in sorted(allowed))
            return (
                f'monitor can\'t watch "{name}" {key}="{verb}": "{name}" is '
                f"read-only only for {allowed_terms}."
            )
        return None

    unmapped = _unmapped_verb_reason(tool, args)
    if unmapped is not None:
        return unmapped
    return None


def monitor_call_verdict(
    tool: AgentTool,
    args: Mapping[str, Any],
    *,
    mcp_annotations: Mapping[str, Any] | None = None,
) -> str | None:
    """The ONE arm-and-tick validator: the read-only verdict, then the call's shape.

    ``readonly_verdict`` answers only "is this class of call observing?"; it
    never looks at the argument schema. That gap let ``glob({path: ...})`` arm
    cleanly (``glob`` is read-tier) and then fail on every tick, because the
    tool's own params model forbids the extra key — five strikes later the
    monitor was disabled having never run. The shape check closes it so the
    arm refuses exactly what a tick would.

    ORDER IS THE SAFETY CONTRACT: ``readonly_verdict`` runs first and is not
    touched, so every posture sentence stays byte-identical and the shape check
    can only ADD refusals — it never admits a call the verdict refused.
    """
    reason = readonly_verdict(tool, args, mcp_annotations=mcp_annotations)
    if reason is not None:
        return reason
    return _shape_reason(tool, args)


def monitor_call_arguments(tool: AgentTool, args: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments a monitor tick should RUN: the harness intent lifted off.

    LOOP PARITY, and the reason this exists. Every tool schema advertises the
    injected ``i`` property (``registry.apply_intent_schema``), so a model
    arming a monitor naturally includes one — and every builtin params model is
    ``extra="forbid"``, so leaving it in makes the tick fail deterministically
    with ``invalid arguments:\n- i: Extra inputs are not permitted``. Measured
    live: two monitors (`4eabc50d61bd` m1/m2) failing every tick on exactly
    that. Refusing ``i`` at arm instead would leave them dead; lifting it heals
    them with no re-arm, and it is what the loop already does
    (``harness/loop.py``, "Lift the intent off BEFORE validation").

    The lift is conditional exactly as the loop's is: only when the schema
    carries OUR intent property. A tool that declares its own ``i`` never had
    ours injected, so its value is a real argument and is kept.
    """
    from local_operator.harness.intent import INTENT_FIELD, intent_is_injected

    view = dict(args)
    if INTENT_FIELD in view and intent_is_injected(tool.parameters):
        view.pop(INTENT_FIELD)
    return view


def _shape_reason(tool: AgentTool, args: Mapping[str, Any]) -> str | None:
    """Refuse a call the tool itself would reject on every execution.

    Mirrors what the tick path rejects and nothing more:

    - required keys and scalar types through the loop's own
      ``validate_tool_arguments`` (one definition of "valid", imported lazily
      because this module is import-light and the loop is not);
    - for a builtin whose schema is closed (``additionalProperties: false`` —
      every builtin params model is ``extra="forbid"``), any key the schema
      does not declare. MCP tools are deliberately NOT held to this: the
      manager's ``prepare_outbound_args`` drops extras before the call, so a
      tick tolerates them and refusing at arm would be stricter than the run.

    Validation runs on the STRIPPED view (:func:`monitor_call_arguments`), so
    an injected ``i`` — which the tick lifts before ``execute`` — is judged the
    way the run will judge it: arm refuses exactly what a tick refuses, and the
    intent the harness injected into every real session's schemas is never a
    refusal. (A COLD arm resolves a tool through its raw builder, which has not
    been through ``apply_intent_schema``; there an ``i`` is an undeclared key
    like any other, and the tick still lifts it.)
    """
    from local_operator.harness.intent import INTENT_FIELD, intent_is_injected
    from local_operator.harness.loop import validate_tool_arguments

    name = tool.name
    view = monitor_call_arguments(tool, args)
    errors = validate_tool_arguments(tool, view)
    if errors:
        return f'monitor can\'t watch "{name}": ' + "; ".join(errors) + "."

    schema = tool.parameters or {}
    if name.startswith("mcp__") or schema.get("additionalProperties") is not False:
        return None
    # The ACCEPTS list omits our injected intent property: it is not an
    # argument the caller may pass (the tick lifts it either way), and naming it
    # would tell the model to keep sending the key this fix exists to absorb.
    injected_intent = INTENT_FIELD if intent_is_injected(schema) else None
    declared = {str(key) for key in (schema.get("properties") or {}) if str(key) != injected_intent}
    unknown = sorted(key for key in view if key not in declared)
    if not unknown:
        return None
    unknown_terms = ", ".join(f'"{key}"' for key in unknown)
    accepts = ", ".join(sorted(declared)) or "no arguments"
    return (
        f'monitor can\'t watch "{name}": unknown argument(s) {unknown_terms} — '
        f"{name} accepts: {accepts}."
    )


def _read_only_hint(annotations: Mapping[str, Any] | None) -> bool:
    """``annotations.readOnlyHint is True`` — the only accepted spelling."""
    if not isinstance(annotations, Mapping):
        return False
    return annotations.get("readOnlyHint") is True


def _unmapped_verb_reason(tool: AgentTool, args: Mapping[str, Any]) -> str | None:
    """Fail-closed: an op-bearing read-tier tool absent from the map (§6.2)."""
    verb_key = _declared_verb_property(tool)
    if verb_key is None:
        return None
    verb = str(args.get(verb_key) or "")
    return (
        f'monitor can\'t watch "{tool.name}" {verb_key}="{verb}": it takes a verb '
        "argument and is not in the monitor observing map (monitors/readonly.py), "
        "so it cannot be admitted by accident — add it and its test to allow it."
    )


def _declared_verb_property(tool: AgentTool) -> str | None:
    """The tool's verb parameter name, from its JSON schema, or ``None``."""
    parameters = tool.parameters if isinstance(tool.parameters, Mapping) else {}
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    if not isinstance(properties, Mapping):
        return None
    for candidate in _VERB_PROPERTIES:
        if candidate in properties:
            return candidate
    return None


# ---------------------------------------------------------------------------
# The external resolver — the arm-time gate for writers OUTSIDE a session
# ---------------------------------------------------------------------------
#
# A session's own validator (``Session._validate_monitor_call``) resolves the
# tool from its LIVE inventory, which a writer standing outside the session
# does not have: the cold paths (``monitors/arm.py``, reached by the desktop
# routes) have no runtime to ask. The resolver below is the honest
# approximation available without booting one:
#
# - the tool is built by the SAME registry the session's own tool list comes
#   from (``tools.registry.TOOL_BUILDERS`` — the one place a builtin tool's
#   tier and schema are defined), so a verdict here is the verdict the session
#   would reach for the same call;
# - what a cold resolver cannot prove stays REFUSED, never admitted: ``mcp__``
#   tools (their ``readOnlyHint`` arrives from a live connection) and builders
#   that only exist inside a running session (``wake``, ``monitor``, ``task``,
#   ``console`` …) answer with a sentence saying so;
# - the residue is session inventory GATING (a host config that disables a
#   builtin). That is not a safety question — the run-time re-check (§6.8)
#   refuses such a call at the tick and counts the failure — so it is
#   deliberately not guessed at here.
#
# The import is inside the function because ``tools.registry`` reaches the
# whole builtin tool tree, while this module is imported by ``session.py`` for
# one predicate: the resolver pays for that tree only when an external arm
# actually runs.


def external_monitor_verdict(tool_name: str, arguments: Mapping[str, Any]) -> str | None:
    """The read-only gate resolved WITHOUT a live session (§6, cold paths).

    ``None`` = read-only (accept). A sentence = the refusal reason, and where
    the call is one the evaluator itself can judge (``bash``, a read-tier
    builtin, a hard-rejected tool) the sentence is ``readonly_verdict``'s own,
    so the agent's tool, the CLI and the desktop route cannot describe one
    call two ways.
    """
    from local_operator.harness.types import ToolContext
    from local_operator.tools.registry import TOOL_BUILDERS

    # THE HARD REJECTS COME FIRST, with ``readonly_verdict``'s own sentences:
    # all three entries (``task``/``ask``/``wait``) are session-only builders
    # that resolve to ``None`` here, so without this line the builder-missing branch
    # below would answer a call the evaluator CAN judge with "not available
    # without a running session" — true, but it buries the reason the call is
    # refused ("delegates autonomous work" / "sends a message" / "parks the
    # turn") and sends the caller to a session that still cannot watch it. The
    # map holds only plain static tool names, so this pre-check answers the same
    # calls ``readonly_verdict`` refuses, with the same sentences.
    hard = _HARD_REJECT.get(tool_name)
    if hard is not None:
        return hard

    builder = TOOL_BUILDERS.get(tool_name)
    if builder is None:
        if tool_name.startswith("mcp__"):
            return (
                f'monitor can\'t watch "{tool_name}" from outside its conversation: '
                "an MCP tool's read-only hint can only be checked inside a running "
                "session — ask that conversation's agent to arm the monitor."
            )
        return (
            f'monitor can\'t watch "{tool_name}" from outside its conversation: '
            "it is not a tool every session builds — ask that conversation's agent "
            "to arm the monitor."
        )
    try:
        tool = builder(ToolContext())
    except Exception:  # noqa: BLE001 — fail closed: an unbuildable tool is unwatchable
        return (
            f'monitor can\'t watch "{tool_name}" from outside its conversation: '
            "the tool could not be resolved here, so the call cannot be proven "
            "read-only — ask that conversation's agent to arm the monitor."
        )
    if tool is None:
        return (
            f'monitor can\'t watch "{tool_name}" from outside its conversation: '
            "it is not available without a running session — ask that "
            "conversation's agent to arm the monitor."
        )
    # THE SAME SCHEMA TRANSFORM A SESSION APPLIES, and it is load-bearing for
    # the parity this function exists to keep: ``registry.create_tools`` runs
    # ``apply_intent_schema`` over every tool it builds, so a real session's
    # schema advertises the injected ``i`` and the tick lifts it. A raw builder
    # has not been through that transform, so without this line the external
    # arm sees ``intent_is_injected(...) is False``, keeps the ``i`` and refuses
    # a call the session arm accepts and the tick runs (review round 1, MAJOR
    # R1 — reproduced with ``glob``).
    from local_operator.harness.intent import apply_intent_schema

    tool.parameters = apply_intent_schema(tool.parameters)
    # The shape check is shared with the in-session path so an arm from the CLI or
    # the desktop route refuses exactly what the agent's own arm would.
    return monitor_call_verdict(tool, arguments)


# ---------------------------------------------------------------------------
# bash (§6.4)
# ---------------------------------------------------------------------------
#
# Stage lexing comes first (round-4 review R4-F1), then tokenisation
# (round-2 review R2-F1). The raw line is scanned once, character-wise — like
# the shell, not from shlex tokens — tracking quotes and backslash escapes,
# and split into pipeline stages at every UNQUOTED `|`, spacing-independent.
# Each stage's resolved words are then judged by the rules below.

#: The v1 allow-list of command words, matched by basename with no path
#: prefix. Deliberately short and grown on demand: each addition is a
#: reviewable line plus tests.
_ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        "cat",
        "head",
        "tail",
        "wc",
        "ls",
        "stat",
        "file",
        "date",
        "uname",
        "whoami",
        "id",
        "nproc",
        "df",
        "du",
        "ps",
        "tree",
        "grep",
        "rg",
        "jq",
        "git",
        "gh",
        "glab",
        "find",
        # kubectl is allow-listed by SUBCOMMAND (``_kubectl_reason``): the
        # read verbs are a strict set and every flag that retargets the
        # cluster, the identity or an arbitrary API path is denied by name.
        "kubectl",
    }
)

#: Flags with a special refusal because a general "not allowed" sentence
#: under-describes why the flag is a trust boundary (the §6.4 examples).
_SPECIAL_FLAG_REFUSALS: dict[str, dict[str, str]] = {
    "rg": {
        "--pre": "runs a program for every file — a monitor must be provably read-only",
        "--pre-glob": "runs a program's per-file filter — a monitor must be provably read-only",
    },
    "gh": {
        "--web": "launches a browser",
        "-w": "launches a browser",
        "--watch": "waits for events instead of returning — a monitor check must return",
    },
    "glab": {
        "--web": "launches a browser",
        "-w": "launches a browser",
        "--watch": "waits for events instead of returning — a monitor check must return",
    },
}


def _flags(
    short: str,
    *,
    valued_short: str = "",
    opt_short: str = "",
    long: str = "",
    valued_long: str = "",
    opt_long: str = "",
    glued_long: str = "",
    consume_long: str = "",
) -> dict[str, Any]:
    """One command's complete flag set; default-deny beyond it.

    The value-taking kinds mirror each program's REAL argument semantics
    (round-1 review F1: git's ``--pretty`` is optional-argument — the separate
    spelling ``git log --pretty --output=/tmp/x`` does NOT feed ``--output`` to
    ``--pretty``, so a model that consumed the next token let a refused flag
    ride through to live git, which wrote the file):

    - ``valued_*``: the value is REQUIRED; glued and separate spellings both
      carry it, the separate form consumes the next token as data, and a
      missing next token is refused (fail-closed, matching the tools, which
      all error on a dangling value).
    - ``opt_*``: the value is OPTIONAL and GLUED-only where a value exists
      (git OPTARG: ``--pretty``, ``--color``, ``-U``, ``--short``, ``-u``);
      the bare form is the flag, and the next token is ALWAYS judged as its
      own word — never consumed.
    - ``glued_long``: the value is required but only git's ``=`` form is a
      flag at all (``--format=<fmt>``; a bare ``--format`` reaches git as an
      unrecognized word), so bare and separate are refused.
    - ``consume_long``: the value is optional but the program DOES consume the
      next token when one is present (``git branch --merged [<commit>]``
      takes ``--merged main`` AND ``--merged --format=…`` — probed against
      git 2.55.0); bare at end of line is the flag.
    """
    return {
        "short": set(short),
        "short_valued": set(valued_short),
        "short_opt": set(opt_short),
        "long": set(long.split()),
        "long_valued": set(valued_long.split()),
        "long_opt": set(opt_long.split()),
        "long_glued": set(glued_long.split()),
        "long_consume": set(consume_long.split()),
    }


#: The v1 allow-list flags per command — the COMPLETE set for that command,
#: everything else about the command denied (operands excepted where a command
#: takes data). Transcribed from contract §6.4.
_FLAG_TABLE: dict[str, dict[str, Any]] = {
    "cat": _flags("nbsAvetTu"),
    "head": _flags("qv", valued_short="nc"),
    "tail": _flags("qv", valued_short="nc"),
    "wc": _flags("lwcmL"),
    "ls": _flags("lahtrSd1n"),
    "stat": _flags("fLt", valued_short="c"),
    "file": _flags("bi", long="mime-type mime-encoding"),
    "date": _flags("uR"),
    "uname": _flags("asrmnvo"),
    "whoami": _flags(""),
    "id": _flags(""),
    "nproc": _flags(""),
    "df": _flags("hkmPiT"),
    "du": _flags("hkmPiTsac", valued_short="d"),
    "ps": _flags("efAaxuowp", valued_short="p"),
    "tree": _flags("adfin", valued_short="L"),
    "grep": _flags(
        "irRnlLcvwxEFGqsho",
        valued_short="efABCm",
        valued_long="include exclude exclude-dir",
    ),
    "rg": _flags(
        "inlcvwxFsqo",
        valued_short="efgtTABCm",
        long="hidden no-ignore",
        valued_long="iglob",
    ),
    "jq": _flags("rcnesRjaS"),
    "find": _flags(""),  # expression-allow-listed instead of flag-allow-listed
}

# git is the complete per-subcommand flag sets (round-2 review R2-F2).
_GIT_SUBCOMMAND_FLAGS: dict[str, dict[str, Any]] = {
    "status": _flags(
        "sb",
        opt_short="u",
        long="short porcelain ignored no-renames",
        opt_long="untracked-files",
    ),
    "log": _flags(
        "ps",
        valued_short="n",
        long=(
            "oneline graph stat name-only name-status decorate no-decorate "
            "abbrev-commit no-merges merges follow all no-patch"
        ),
        valued_long="date since until",
        opt_long="pretty",
        glued_long="format",
    ),
    "diff": _flags(
        "wb",
        opt_short="U",
        long="stat name-only name-status numstat shortstat cached staged no-index",
        opt_long="color",
    ),
    "show": _flags(
        "s",
        long="stat name-only name-status abbrev-commit no-patch",
        opt_long="pretty",
        glued_long="format",
    ),
    "blame": _flags("wp", valued_short="L", long="porcelain line-porcelain", valued_long="date"),
    "rev-parse": _flags(
        "",
        long="verify abbrev-ref show-toplevel is-inside-work-tree is-bare-repository",
        opt_long="short",
    ),
    "ls-files": _flags("scomdc", long="stage others modified deleted cached exclude-standard"),
    "grep": _flags("niIlLcwEF", valued_short="eABC", long="cached untracked"),
    # list forms ONLY; no operands (a ref argument CREATES or edits).
    "branch": _flags("var", long="list all remotes", consume_long="format merged no-merged"),
}

#: gh/glab: subcommand families and the complete flag set.
_GH_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "pr": frozenset({"view", "list", "diff", "checks", "status"}),
    "issue": frozenset({"view", "list"}),
    "run": frozenset({"view", "list"}),
}
_GH_FLAGS = _flags(
    "qt",
    valued_short="qt",
    long="comments log",
    valued_long="json jq template repo",
)

#: ``find`` is expression-allow-listed (no flag/operand grammar a deny-list can
#: bound): read tests and the boolean operators in, everything else out.
_FIND_VALUE_PRIMARIES = frozenset(
    {
        "-name",
        "-iname",
        "-type",
        "-maxdepth",
        "-mindepth",
        "-path",
        "-ipath",
        "-newer",
        "-mtime",
        "-mmin",
        "-size",
        "-perm",
        "-user",
        "-group",
    }
)
_FIND_OPERATORS = frozenset({"-a", "-o", "-and", "-or", "-not", "!", "(", ")", "-print", "-printf"})
_FIND_WRITES = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fls"})

#: The refusal clause for each unquoted operator character. Position is
#: appended by the scanner, so each clause stays position-free.
_OPERATOR_REASONS: dict[str, str] = {
    ";": '";" chains a second command',
    "&": '"&" chains a second command',
    "<": '"<" redirects input from a file',
    ">": "output redirection writes to disk",
}


def _bash_verdict(command: str) -> str | None:
    """The §6.4 verdict for one bash ``command`` string."""
    reason = _bash_readonly_reason(command)
    if reason is None:
        return None
    return f'monitor can\'t watch "{command}": {reason}'


def _bash_readonly_reason(command: str) -> str | None:
    """``None`` when the command is provably read-only, else the reason."""
    # -- layer 1: character-wise scan of the raw line --------------------------
    stages: list[str] = []
    current: list[str] = []
    quote: str | None = None
    quote_pos = 0
    escaped = False
    i = 0
    length = len(command)
    while i < length:
        ch = command[i]
        if ch in "\n\r":
            what = "newline" if ch == "\n" else "carriage return"
            return (
                f"a raw {what} at position {i + 1} starts a second command — "
                "monitors admit exactly one line."
            )
        if ch in "{}":
            return (
                f'"{ch}" at position {i + 1} — brace expansion invents words after '
                "this check; a monitor must be provably read-only."
            )
        if escaped:
            current.append(ch)
            escaped = False
            i += 1
            continue
        if quote == "'":
            if ch == "'":
                quote = None
            current.append(ch)
            i += 1
            continue
        if quote == '"':
            if ch == '"':
                quote = None
            elif ch == "\\":
                escaped = True
            current.append(ch)
            i += 1
            continue
        if ch == "\\":
            escaped = True
            current.append(ch)
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            quote_pos = i + 1
            current.append(ch)
            i += 1
            continue
        if ch == "|":
            nxt = command[i + 1] if i + 1 < length else ""
            if nxt in "|&":
                return (
                    f'"{"||" if nxt == "|" else "|&"}" at position {i + 1} is not a '
                    'pipeline — monitors admit exactly one operator, the plain "|".'
                )
            stages.append("".join(current))
            current = []
            i += 1
            continue
        if ch in _OPERATOR_REASONS:
            return f"{_OPERATOR_REASONS[ch]} at position {i + 1}."
        current.append(ch)
        i += 1
    if escaped:
        return (
            f"a backslash at position {length} is the line's final character — a "
            "continuation is not one line; monitors admit exactly one line."
        )
    if quote is not None:
        return (
            "the command line does not parse — unterminated quote starting at "
            f"position {quote_pos}. Fix the quoting."
        )
    stages.append("".join(current))
    for stage_text in stages:
        if not stage_text.strip():
            return 'an empty pipeline stage — every "|" must separate two commands.'
        reason = _stage_reason(stage_text)
        if reason is not None:
            return reason
    return None


def _stage_reason(stage_text: str) -> str | None:
    """One pipeline stage, judged on its RESOLVED words."""
    try:
        words = shlex.split(stage_text, posix=True)
    except ValueError as exc:
        return f"the command line does not parse — {exc}. Fix the quoting."
    if not words:
        return "an empty pipeline stage."
    for word in words:
        if "$" in word or "`" in word:
            return (
                f'"{word}" contains a shell substitution — it expands after this '
                "check, so the command cannot be proven read-only."
            )
    command = words[0]
    rest = words[1:]
    if "/" in command:
        return (
            f'"{command}" has a path prefix — monitors match the bare command name '
            "so the allow-list is the whole verdict."
        )
    if command not in _ALLOWED_COMMANDS:
        return (
            "monitors re-run unattended, so a bash command must be provably "
            f'read-only — "{command}" is not on the read-only allow-list. Try a '
            "read-only command, web_fetch, or an MCP tool that declares "
            "readOnlyHint."
        )
    if command == "find":
        return _find_reason(rest)
    if command == "git":
        return _git_reason(rest)
    if command in ("gh", "glab"):
        return _gh_reason(command, rest)
    if command == "kubectl":
        return _kubectl_reason(rest)
    if command == "date":
        return _date_reason(rest)
    return _flags_reason(command, rest, _FLAG_TABLE[command])


def _date_reason(tokens: list[str]) -> str | None:
    """``date`` accepts ``+<format>`` operands only; a bare operand SETS."""
    table = _FLAG_TABLE["date"]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return (
                'the "--" separator is not accepted here: a v1 operand that begins '
                'with "-" has no form, and each program\'s own "--" handling would '
                "be a new per-command trust surface."
            )
        if token.startswith("-") and token != "-":
            reason, consumed = _check_flag("date", token, tokens, index, table)
            if reason is not None:
                return reason
            index += consumed
            continue
        if not token.startswith("+"):
            return (
                f'"{token}" is the SET form of date (a "+<format>" operand prints; '
                "anything else sets the clock) — monitors must not set the clock."
            )
        index += 1
    return None


def _flags_reason(cmd: str, tokens: list[str], table: dict[str, Any]) -> str | None:
    """Default-deny flag scan over one stage's tokens."""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return (
                'the "--" separator is not accepted here: a v1 operand that begins '
                'with "-" has no form, and each program\'s own "--" handling would '
                "be a new per-command trust surface."
            )
        if token.startswith("-") and token != "-":
            reason, consumed = _check_flag(cmd, token, tokens, index, table)
            if reason is not None:
                return reason
            index += consumed
            continue
        index += 1
    return None


def _check_flag(
    cmd: str, token: str, tokens: list[str], index: int, table: dict[str, Any]
) -> tuple[str | None, int]:
    """One flag token. Returns ``(refusal | None, tokens consumed)``.

    Matching is by base flag name and covers the glued and separate
    spellings. Whether the SEPARATE form exists at all, and whether it
    consumes the next token, is per-flag-kind (``_flags``): a required value
    consumes, an optional value never does, a glued-only value has no
    separate form. Tokens consumed as a value are data — judged by no rule.
    """
    # Special refusals are keyed by the command FAMILY ("gh", not "gh pr
    # view"): the hazard belongs to the tool, not to one subcommand.
    special = _SPECIAL_FLAG_REFUSALS.get(cmd.split()[0] if cmd.split() else "", {})
    if token.startswith("--"):
        base, sep, _value = token.partition("=")
        if base in special:
            return f'"{base}" {special[base]}.', 1
        # The special-refusal keys keep their dashes; the allow-list tables are
        # keyed by the bare flag name ("oneline", not "--oneline").
        name = base[2:]
        if name in table["long_glued"]:
            if sep:
                return None, 1
            return f'"{base}" only accepts a value glued to it ("{base}=…").', 1
        if name in table["long_valued"]:
            if sep:
                return None, 1
            if index + 1 >= len(tokens):
                # Fail-closed (round-1 review F5): every allow-listed tool
                # errors on a dangling value; refusing at arm time names the
                # flag instead of counting a mystery failure at check time.
                return f'"{base}" expects a value.', 1
            return None, 2
        if name in table["long_opt"]:
            return None, 1
        if name in table["long_consume"]:
            if sep:
                return None, 1
            return None, 2 if index + 1 < len(tokens) else 1
        if name in table["long"]:
            if sep:
                return f'"{base}" does not take a value.', 1
            return None, 1
        return f'"{base}" is not an allowed flag of {cmd}.', 1
    # short cluster: every letter individually allowed; a required-value letter
    # consumes the rest of the token (glued) or the next token (separate); an
    # optional-value letter never consumes the next token.
    if token in special:
        return f'"{token}" {special[token]}.', 1
    body = token[1:]
    position = 0
    while position < len(body):
        ch = body[position]
        if ch in table["short_valued"]:
            if position + 1 < len(body):
                return None, 1
            if index + 1 >= len(tokens):
                return f'"-{ch}" expects a value.', 1
            return None, 2
        if ch in table["short_opt"]:
            return None, 1
        if ch in table["short"]:
            position += 1
            continue
        return f'"-{ch}" is not an allowed flag of {cmd}.', 1
    return None, 1


def _git_reason(tokens: list[str]) -> str | None:
    if not tokens:
        return "git needs a read-only subcommand."
    sub = tokens[0]
    table = _GIT_SUBCOMMAND_FLAGS.get(sub)
    if table is None:
        allowed = " ".join(sorted(_GIT_SUBCOMMAND_FLAGS))
        return (
            f'"{sub}" is not on the read-only git allow-list ({allowed}) — every '
            "other subcommand can write."
        )
    rest = tokens[1:]
    if sub == "branch":
        reason = _flags_reason("git branch", rest, table)
        if reason is not None:
            return reason
        # The operand scan must SKIP the tokens a value-flag consumed: the
        # flags' own values are data the flag owns (`--merged main`,
        # `--format '%(refname)'` — both are the §6.4 branch row's list
        # forms), not ref operands that create or edit anything. Only tokens
        # no flag claimed can be operands. (Round-1 review F3: the scan
        # walked every token, so it rejected the flags' own values.)
        index = 0
        while index < len(rest):
            token = rest[index]
            if token.startswith("-") and token != "-":
                _, consumed = _check_flag("git branch", token, rest, index, table)
                index += consumed
                continue
            return (
                f'"{token}" is an operand — "git branch" is list-only here: '
                "creating, moving or deleting a ref is not read-only."
            )
        return None
    return _flags_reason(f"git {sub}", rest, table)


def _gh_reason(cmd: str, tokens: list[str]) -> str | None:
    if len(tokens) < 2:
        return f'{cmd} needs a subcommand like "pr view".'
    family = tokens[0]
    verbs = _GH_SUBCOMMANDS.get(family)
    if verbs is None:
        return (
            f'"{family}" is not a read-only {cmd} subcommand (pr view/list/diff/'
            "checks/status, issue view/list, run view/list) — everything else can "
            "mutate, including the method-flippable api."
        )
    verb = tokens[1]
    if verb not in verbs:
        return (
            f'"{family} {verb}" is not read-only for {cmd} (that family allows: '
            f"{' '.join(sorted(verbs))})."
        )
    return _flags_reason(f"{cmd} {family} {verb}", tokens[2:], _GH_FLAGS)


#: kubectl's allow-list, by subcommand. Deliberately three verbs: every other
#: verb is either a write (``apply``/``delete``/``cordon``/``scale``/…) or an
#: escape hatch that can execute code or reach an arbitrary API path
#: (``exec``, ``cp``, ``port-forward``, ``proxy``, ``auth``, ``config``, …).
#: No plugin dispatch is possible either, because the first token must be
#: exactly one of the three.
_KUBECTL_SUBCOMMAND_FLAGS: dict[str, dict[str, Any]] = {
    "get": _flags(
        "A",
        valued_short="nlo",
        long="all-namespaces no-headers show-labels",
        valued_long="namespace context selector field-selector output sort-by",
    ),
    "describe": _flags(
        "A",
        valued_short="nl",
        long="all-namespaces show-events",
        valued_long="namespace context selector",
    ),
    "logs": _flags(
        "p",
        valued_short="nlc",
        long="previous timestamps",
        valued_long=(
            "namespace context container tail since since-time selector "
            "max-log-requests limit-bytes"
        ),
    ),
}

#: Flags refused with a NAMED reason rather than the generic "not an allowed
#: flag", because each one is a trust boundary a reader would not guess: the
#: first four never return, the rest retarget what the call reads.
_KUBECTL_DENIED_FLAGS: dict[str, str] = {
    "--watch": "waits for events instead of returning — a monitor check must return",
    "-w": "waits for events instead of returning — a monitor check must return",
    "--follow": "follows the stream instead of returning — a monitor check must return",
    "-f": "follows the stream instead of returning — a monitor check must return",
    "--raw": "requests an arbitrary API path, which this allow-list cannot bound",
    "--kubeconfig": "retargets which cluster and identity every later call uses",
    "--server": "retargets which cluster the call reads",
    "-s": "retargets which cluster the call reads",
    "--token": "supplies a bearer token, so the call runs as another identity",
    "--user": "selects another identity from the kubeconfig",
    "--as": "impersonates another identity",
    "--as-group": "impersonates another group",
    "--as-uid": "impersonates another uid",
    "--cluster": "selects another cluster from the kubeconfig",
    "--certificate-authority": "retargets which server certificate is trusted",
    "--client-certificate": "supplies another client identity",
    "--client-key": "supplies another client identity",
    "--insecure-skip-tls-verify": "accepts an unverified server certificate",
}

#: The ``-o/--output`` values that are RENDERING only. ``go-template*`` is
#: deliberately absent: template functions are a small evaluator, and a monitor
#: must be provably read-only rather than probably.
_KUBECTL_OUTPUT_OK: frozenset[str] = frozenset({"name", "wide", "json", "yaml"})

#: Operands naming a secret are refused because monitor output is copied into
#: the transcript and from there into the provider request: a delta carrying a
#: decoded credential is the one datum no later redaction pass can recall.
#: Matched as a CASE-INSENSITIVE SUBSTRING (``secret``, ``secrets``,
#: ``secrets/x``, ``all,secrets``), and without a regex import: this module is
#: deliberately import-light.
_KUBECTL_SECRET_WORD = "secret"


def _kubectl_reason(tokens: list[str]) -> str | None:
    """``kubectl get|describe|logs`` with a default-deny flag set (§6.4).

    Three rules beyond the flag table:

    - the SUBCOMMAND must be the first token. A global flag before it (or a
      second subcommand) is refused rather than skipped, because kubectl's own
      global flags are exactly the ones that retarget the cluster;
    - ``-o/--output`` must be a rendering format. ``go-template*`` is an
      evaluator, so it is refused by name while ``jsonpath=``/``custom-columns=``
      (pure selectors) are allowed;
    - no operand may name a secret, and none may begin with ``-`` (there is no
      ``--`` form: an operand that needs to look like a flag is not one of the
      shapes this allow-list covers).
    """
    if not tokens:
        return 'kubectl needs a read-only subcommand ("get", "describe" or "logs").'
    sub = tokens[0]
    table = _KUBECTL_SUBCOMMAND_FLAGS.get(sub)
    if table is None:
        if sub.startswith("-"):
            return (
                f'"{sub}" comes before the subcommand — a global flag is one of the '
                "shapes that retargets the cluster, so the subcommand must come first."
            )
        allowed = " ".join(sorted(_KUBECTL_SUBCOMMAND_FLAGS))
        return (
            f'"{sub}" is not on the read-only kubectl allow-list ({allowed}) — every '
            "other verb either writes or can execute code."
        )
    cmd = f"kubectl {sub}"
    index = 0
    rest = tokens[1:]
    while index < len(rest):
        token = rest[index]
        if token == "--":
            return (
                'the "--" separator is not accepted here: an operand that needs to look '
                "like a flag is not one of the shapes this allow-list covers."
            )
        if token.startswith("-") and token != "-":
            denied = _kubectl_denied_reason(token, table)
            if denied is not None:
                return denied
            output = _kubectl_output_value(token, rest, index)
            if output is not None and not _kubectl_output_ok(output):
                return (
                    f'"-o {output}" is not a rendering format this allow-list accepts '
                    "(name, wide, json, yaml, jsonpath=…, custom-columns=…); "
                    "go-template runs template functions."
                )
            reason, consumed = _check_flag(cmd, token, rest, index, table)
            if reason is not None:
                return reason
            index += consumed
            continue
        if _KUBECTL_SECRET_WORD in token.lower():
            return (
                f'"{token}" reads secret data — monitor output is copied into the '
                "transcript and the provider request, so secret reads are refused."
            )
        index += 1
    return None


def _kubectl_output_ok(value: str) -> bool:
    """Whether one ``-o/--output`` value renders output without evaluating it.

    ``name``/``wide``/``json``/``yaml`` are fixed renderings; the two selector
    forms are accepted by PREFIX (their syntax owns everything after ``=``).
    """
    if value in _KUBECTL_OUTPUT_OK:
        return True
    return value.startswith("jsonpath=") or value.startswith("custom-columns=")


def _kubectl_denied_reason(token: str, table: dict[str, Any]) -> str | None:
    """A named refusal for one flag token, or ``None`` when it is not denied.

    Only the head of a short cluster is judged, and only up to the first
    value-taking letter: ``-nw`` is ``-n`` with the value ``w``, not ``-w``.
    """
    if token.startswith("--"):
        base = token.partition("=")[0]
        reason = _KUBECTL_DENIED_FLAGS.get(base)
        if reason is not None:
            return f'"{base}" {reason}.'
        return None
    head: list[str] = []
    for ch in token[1:]:
        if ch in table["short_valued"]:
            break
        head.append(ch)
    for ch in head:
        reason = _KUBECTL_DENIED_FLAGS.get(f"-{ch}")
        if reason is not None:
            return f'"-{ch}" {reason}.'
    return None


def _kubectl_output_value(token: str, tokens: list[str], index: int) -> str | None:
    """The value of ``-o/--output`` in this token, or ``None`` if it is not one.

    Handles all four spellings (``-o json``, ``-ojson``, ``--output json``,
    ``--output=json``) because the check must see the value the flag will
    actually receive, whatever shape the caller used.
    """
    if token.startswith("--output"):
        _base, sep, value = token.partition("=")
        if sep:
            return value
        return tokens[index + 1] if index + 1 < len(tokens) else None
    if not token.startswith("-") or token.startswith("--"):
        return None
    body = token[1:]
    for position, ch in enumerate(body):
        if ch == "o":
            glued = body[position + 1 :]
            if glued:
                # ``-o=json`` is a spelling kubectl accepts, so the glued value
                # may carry the separator: judge the VALUE, not the remainder
                # (review round 1, R5 — the raw remainder was refused as
                # "-o =json").
                return glued[1:] if glued.startswith("=") else glued
            return tokens[index + 1] if index + 1 < len(tokens) else None
        if ch in _KUBECTL_SUBCOMMAND_FLAGS["get"]["short_valued"]:
            # A value-taking letter before ``o`` means the rest of the token
            # is THAT flag's value, not an output format.
            return None
    return None


def _find_reason(tokens: list[str]) -> str | None:
    """``find`` is expression-allow-listed: read tests and boolean operators."""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _FIND_OPERATORS or not token.startswith("-"):
            index += 1
            continue
        if token in _FIND_WRITES or token.startswith("-fprint"):
            return f'"{token}" makes find a write.'
        if token in _FIND_VALUE_PRIMARIES:
            index += 2
            continue
        return (
            f'"{token}" is not a permitted find primary — the permitted set is the '
            "read tests (-name, -type, -size, …) and the boolean operators."
        )
    return None
