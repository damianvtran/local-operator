"""The one code-memory (execution variables) verb table, shared by every session shape.

WHY this module exists
----------------------
The desktop canvas's "Code memory" panel renders a session's LIVE eval-kernel
namespace (``local_operator.tools.eval._KERNELS``). That namespace only exists
in the process running the session's turn loop, and it dies with the kernel
(idle reap, LRU eviction, timeout, crash) — nothing persists it, so there is no
"stored variables" to read; there is only what is in the interpreter right now.

The panel used to read it through ``GET /v1/agents/{id}/execution-variables``
with a canonical session id. That route resolves via
``AgentRegistry.load_agent_context``, which is keyed by legacy agent-directory
UUIDs, so it answered ``404 Agent with ID <12-hex> not found`` for every session
and the panel could never load at all. This module is the session-addressed
replacement.

One table, two shapes — the same doctrine as ``credential_ops.py``. A session
that runs its tools in this process executes the verbs against its own kernel
registry (``Session.variables_op``); a session that is a window onto a runtime
routes them there (``AttachedSession.variables_op`` → the runtime's
``ServingSessionHandle.variables_op`` → this same table). Keeping ONE copy is
what prevents the two shape's verbs from drifting apart; a second table beside
this one would be the defect, not a convenience.

The worker is the authority
---------------------------
The namespace lives in the eval worker subprocess, so existence, coercion and
enumeration all have to be decided THERE or not at all: a parent-side
"does this key exist" probe is a second round trip and loses the race against
the cell that is still running. This module therefore owns only what both
sides must agree on — the type vocabulary, the key denylist and its sentence,
and the response envelope — while the enumeration and the assignment happen in
``tools/eval_worker.py`` and the ownership/lease rules in ``tools/eval.py``.
``run_variable_verb`` is the seam: it validates, forwards, and redacts.

Secret policy (stated honestly, not implied)
--------------------------------------------
1. The worker renders each value through ``_safe_repr`` → ``_scrub_secrets``,
   the same sink a cell's trailing expression uses, so a value retrieved through
   ``secrets["NAME"]`` is scrubbed at the source.
2. The assembled list then passes the session's ``VariableStore.redact``
   (``redaction_values()`` = session credentials + registered redactions) before
   it leaves the process — that is the ``redact`` argument here.
3. ``secrets`` is never listed at all, and the harness's own ``display``/``tool``
   bindings are excluded with it (they are rebound every request, not memory).
4. Residual, and it is real: a plain string a cell read from a non-secret path
   (``open(".env").read()``) is indistinguishable from any other string and WILL
   be shown — the same exposure that value already has in the transcript.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Literal, get_args

logger = logging.getLogger(__name__)

#: The ONE type vocabulary. ``Literal`` is the source and the tuple is derived
#: from it, so the pydantic body validation on the route, the worker's coercion
#: table and the UI's type list cannot drift apart. Adding a member here is the
#: whole change; a second list anywhere else is the defect this prevents.
VariableType = Literal["str", "int", "float", "bool", "list", "dict"]
VARIABLE_TYPES: tuple[str, ...] = get_args(VariableType)

#: Names the kernel itself owns. ``secrets`` is the lazy secret alias, and
#: ``display``/``tool`` are rebound on every request — none of them is memory the
#: user made, and listing them would offer an "edit" that the next request
#: silently overwrites. Dunders (``__builtins__`` especially, which ``exec``
#: inserts) are excluded by predicate rather than by name.
RESERVED_NAMES = frozenset({"secrets", "display", "tool"})

#: Bounds mirrored from the frozen design: a key is a path segment that stays
#: addressable as a dict key, and a value is capped where the worker's render cap
#: already sits, so an accepted write cannot come back truncated.
MAX_KEY_CHARS = 128
MAX_VALUE_CHARS = 4096

#: Total response budget for one enumeration, charged in the ENCODED bytes the
#: control socket will carry. NOT cosmetic: that socket reads at a 1 MiB line
#: limit, so an unbounded namespace would sever the connection instead of
#: answering — and a panel that cannot READ a namespace cannot DELETE out of it
#: either, so the failure is not merely cosmetic on the client side. Charging
#: characters of the rendered value (the first accounting) under-counted the
#: frame by ~1.7x, because JSON escaping, the key, the type name and the
#: envelope all ride the same line; see the worker's ``_variables_list``.
TOTAL_BUDGET_CHARS = 256 * 1024

#: Refusal codes the desktop routes lift into ``detail.code`` (see §1 of the
#: design). Declared here as well as on the route so both sides of the wire name
#: the same strings; the UI switches on the code and shows ``message``.
RefusalCode = Literal[
    "no_kernel",
    "kernel_busy",
    "runtime_cold",
    "reserved_name",
    "already_exists",
    "not_found",
    "invalid_value",
    "too_large",
]

#: ``(session_id, action, key=, value=, value_type=) -> answer``. The default is
#: resolved lazily so importing this module never drags the eval tool in — it is
#: imported by the eval WORKER too, which must stay cheap to start.
VariableCompleter = Callable[..., Awaitable[dict[str, Any]]]


def is_reserved_name(name: str) -> bool:
    """Whether ``name`` belongs to the kernel rather than to the user.

    Dunders are matched by predicate, not by a list: ``exec`` inserts
    ``__builtins__`` into exactly this dict, and every future dunder the
    interpreter adds must stay hidden without a code change here.
    """
    return name in RESERVED_NAMES or (name.startswith("__") and name.endswith("__"))


def name_is_addressable(name: str) -> bool:
    """Whether a URL path segment can address ``name`` at all.

    The routes address a variable as ``.../variables/{key}``, and a ``/``
    decodes into extra path segments BEFORE the router matches — percent-encoding
    it does not help (``%2F`` is decoded by the server, measured), so no PATCH or
    DELETE can ever reach such a key. A cell can still create one, and hiding a
    binding the user's own code made would be the "Nothing stored yet" lie the
    ``state`` discriminator exists to prevent, so the row stays LISTED — but it
    must not advertise as editable, and the write paths refuse to create one.
    """
    return "/" not in name


#: The sentence for each way a NAME can be unusable. One copy, used by the verb
#: table AND by the worker: the worker's own wording was previously unreachable
#: (both session shapes pre-validate through the table), which is how two
#: sentences for one rule came to exist.
_NAME_REFUSAL_MESSAGES: dict[str, str] = {
    "unnamed": "Enter a variable name.",
    "too_long": f"A variable name is limited to {MAX_KEY_CHARS} characters.",
    "control": "A variable name cannot contain control characters.",
    "unaddressable": (
        "A variable name cannot contain '/': a variable is addressed as a URL "
        "path segment, so such a name cannot be edited or deleted."
    ),
    "reserved": "That name belongs to the interpreter and cannot be used.",
}


def name_refusal(key: str) -> dict[str, Any] | None:
    """The complete refusal for an unusable ``key``, or ``None`` when it is fine.

    Returns the envelope rather than a code so the caller cannot pair the right
    code with the wrong sentence (the worker used to answer the "1-128
    characters" bound for every ``invalid_value`` name, including a reserved
    one).
    """
    if not key:
        return refusal("invalid_value", _NAME_REFUSAL_MESSAGES["unnamed"])
    if len(key) > MAX_KEY_CHARS:
        return refusal("invalid_value", _NAME_REFUSAL_MESSAGES["too_long"])
    if any(character < " " or character == "\x7f" for character in key):
        return refusal("invalid_value", _NAME_REFUSAL_MESSAGES["control"])
    if not name_is_addressable(key):
        return refusal("invalid_value", _NAME_REFUSAL_MESSAGES["unaddressable"])
    if is_reserved_name(key):
        return refusal("reserved_name", _NAME_REFUSAL_MESSAGES["reserved"])
    return None


def key_refusal(key: str) -> str | None:
    """The refusal CODE for ``key``, or ``None`` when it is usable.

    The code half of :func:`name_refusal`, kept for callers that only need to
    ask "is this name usable?" (the worker's existence check runs after it).

    Deliberately a DENYLIST, not an identifier check: the write path is
    ``namespace[key] = value``, so ``globals()["a b"]`` is addressable and must
    keep working. What is refused is what the protocol cannot carry (an empty
    name, a control character, a ``/``), what the runtime cannot address (the
    same ``/``), or what the kernel owns (a reserved name).
    """
    rejected = name_refusal(key)
    return None if rejected is None else str(rejected["code"])


def coerce_variable_value(value: str, value_type: str) -> Any:
    """Build the object a write stores from ``(value, type)``.

    A TABLE, never interpolation and never ``exec``: the string arrives from an
    HTTP body, so building it by executing anything would be an injection point
    on the one surface whose whole job is to put objects into a live namespace.

    Raises :class:`ValueError` when the value cannot be read as the named type.
    The message names the TARGET TYPE ONLY — never the submitted value, which
    may be a credential the user is storing for a cell to use.
    """
    if value_type == "str":
        return value
    if value_type == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ValueError("Enter a whole number") from None
    if value_type == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError("Enter a number") from None
    if value_type == "bool":
        # Legacy parity (`routes/agents.py`): the truthy spellings are the set,
        # and anything else is False. Kept rather than tightened because the old
        # surface is still live for UUID agents and the two must agree on what a
        # given body means.
        return value.strip().lower() in ("true", "1", "yes", "on")
    if value_type in ("list", "dict"):
        import json

        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            raise ValueError(f"Enter JSON that reads as a {value_type}") from None
        if not isinstance(parsed, list if value_type == "list" else dict):
            raise ValueError(f"Enter JSON that reads as a {value_type}")
        return parsed
    raise ValueError(f"Unsupported variable type: {value_type}")


def refusal(code: str, message: str) -> dict[str, Any]:
    """One refusal in the wire shape every layer above passes through."""
    return {"ok": False, "code": code, "message": message}


#: Sentences live beside the codes so a caller cannot invent a new wording that
#: quotes the submitted value. ``{limit}`` is the only interpolation, and it is a
#: bound, never user input.
_REFUSAL_MESSAGES: dict[str, str] = {
    "no_kernel": (
        "This chat's Python interpreter is not running, so it has no code memory "
        "to read or change. Run a cell in this chat first."
    ),
    "kernel_busy": "A cell is running in this chat. Try again when it finishes.",
    "runtime_cold": (
        "This chat has not started yet. Send a message first, then its code memory "
        "becomes readable."
    ),
    "reserved_name": "That name belongs to the interpreter and cannot be used.",
    "already_exists": "A variable with that name already exists.",
    "not_found": "There is no variable with that name.",
    "invalid_value": "That value cannot be used.",
    "too_large": f"Values are limited to {MAX_VALUE_CHARS} characters.",
}


def refusal_for(code: str) -> dict[str, Any]:
    """The canned refusal for ``code`` (``invalid_value`` when unknown)."""
    return refusal(code, _REFUSAL_MESSAGES.get(code, _REFUSAL_MESSAGES["invalid_value"]))


async def run_variable_verb(
    session_id: str,
    action: str,
    key: str = "",
    value: str = "",
    value_type: str = "",
    *,
    redact: Callable[[str], str] | None = None,
    complete: VariableCompleter | None = None,
) -> dict[str, Any]:
    """Run one code-memory verb for ``session_id``; return plain data.

    ``redact`` is the session's ``VariableStore.redact`` (or ``None`` on a
    session built without a store — embedded callers and test doubles): the
    assembled values pass through it here, at the single point where the answer
    leaves the session, rather than at each producer. ``complete`` defaults to
    the eval tool's :func:`~local_operator.tools.eval.complete_session_variables`
    and is injectable so both shapes' behaviour is testable without a kernel.

    Validation happens HERE as well as in the worker. Not redundancy for its own
    sake: the route can refuse a reserved name or an oversized value without a
    round trip to a kernel that is idle by definition, and the worker still
    enforces the same rules because it is the authority on what is in the
    namespace (a caller reaching it by another road must not bypass them).
    """
    if action not in ("list", "set", "update", "delete"):
        # Unreachable from both callers (the route's body model and the control
        # frame validator refuse an unknown verb first) — a programming error
        # rather than a refusal, so it is not dressed up as one.
        raise ValueError(f"unknown code-memory action: {action!r}")

    if action != "list":
        rejected = name_refusal(key)
        if rejected is not None:
            return rejected
        if action in ("set", "update") and len(value) > MAX_VALUE_CHARS:
            return refusal_for("too_large")

    if complete is None:
        from local_operator.tools.eval import complete_session_variables

        complete = complete_session_variables

    answer = await complete(session_id, action, key=key, value=value, value_type=value_type)
    if not isinstance(answer, dict):
        logger.warning("code-memory verb returned %s", type(answer).__name__)
        return refusal_for("invalid_value")
    if redact is not None and answer.get("ok"):
        answer = _redacted_answer(answer, redact)
    return answer


def _redacted_answer(answer: dict[str, Any], redact: Callable[[str], str]) -> dict[str, Any]:
    """Apply the session's redactor to every value in an answer.

    Only VALUES are redacted. A key is a variable name the user typed and the
    panel must print it to be useful at all; ``redaction_values()`` contains the
    credential store's values, whose presence in a *name* would already be the
    leak. ``variable`` is covered alongside ``variables`` because a write echoes
    what it stored, and that echo leaves the process by the same road.
    """
    variables = answer.get("variables")
    if isinstance(variables, list):
        answer["variables"] = [
            (
                {**entry, "value": redact(str(entry.get("value", "")))}
                if isinstance(entry, dict)
                else entry
            )
            for entry in variables
        ]
    variable = answer.get("variable")
    if isinstance(variable, dict):
        answer["variable"] = {**variable, "value": redact(str(variable.get("value", "")))}
    return answer
