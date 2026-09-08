"""The ``secret`` agent tool — one createIf-gated verb tool (design §5.2).

**Why one tool and not six.** ``AGENTS.md``'s tool-surface footprint ladder is
explicit: every core tool's schema rides in the same prompt-cache prefix as the
system prompt, on every request, in every session and every subagent, whether
or not it is ever called. Six tools (``secret_store``, ``secret_retrieve``, …)
would be six schemas of permanent tax. One tool with an ``op`` parameter is a
fraction of that, and this is rung 3 rather than rung 5 because the builder
returns ``None`` when no store is reachable — a session that cannot use it pays
nothing at all. The CLI (``lop secret``, rung 2) remains the PRIMARY surface
and carries the full verb set including rotation and hardening; this tool
carries only what an agent decides in the middle of a task.

**``retrieve`` does not return the value.** It returns a receipt naming the
secret and stating how to USE it — ``$(lop secret get NAME)`` in bash,
``secrets["NAME"]`` in eval. That inversion is the same one the session
credential path already implements (``variables.py``): the agent uses a secret
it cannot read. Returning the bytes here would write them straight into the
transcript, which is persisted and replayed to the provider on every
subsequent turn — the single worst outcome this whole design exists to prevent.

**``store`` is the verb that makes persistence a decision the agent makes**,
which is what the operator asked for. Guidance lives in ``guide://credentials``
and the tool description points at it.

**Residual risk is not overclaimed here or in the description** (design §9):
this store raises the cost of credential theft a great deal against the
scan-the-disk malware that a bad link actually drops, and it is not a vault.
Anything running as the operator that is willing to run ``lop`` can read these
values.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import AbortSignal, AgentTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

#: What the model is told after a successful ``retrieve``. Written as an
#: instruction rather than a description because it is the only place an agent
#: learns HOW to use a secret it just proved exists, and a receipt that only
#: says "ok" invites a retry with a different verb looking for the value.
_USE_HINT = (
    "The value is NOT shown here and never will be. Use it without reading it:\n"
    "  bash:  interpolate it directly into the command that needs it, e.g.\n"
    '         curl -H "Authorization: Bearer $(lop secret get {name})" ...\n'
    '  eval:  secrets["{name}"] returns the value inside the worker process.\n'
    "Never echo it, assign-and-print it, write it to a file, or paste it into "
    "a commit, a PR or a message."
)


class SecretParams(BaseModel):
    """Arguments for the ``secret`` tool. One model, ``op`` selects the verb."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["store", "retrieve", "list", "describe", "update", "delete"] = Field(
        description="retrieve makes a secret usable without returning its value."
    )
    name: str | None = Field(default=None, description="Secret name; required except for list.")
    value: str | None = Field(
        default=None, description="The secret (store/update). Only a value you already hold."
    )
    description: str | None = Field(
        default=None, description="What it is for; shown in list. Not secret."
    )


#: The tool description IS the interface — the only guidance a model reliably
#: reads — so it states WHEN to reach for this, not merely what it does. It is
#: also schema that ships on EVERY request in every session and subagent (the
#: footprint ladder), so each clause has to earn its tokens: "store it as soon
#: as you get one" and "retrieve NEVER returns the value" change behaviour and
#: stay; the rationale behind them lives in `guide://credentials`, which costs
#: nothing until it is read.
_DESCRIPTION = (
    "Store and use long-term secrets in the operator's encrypted store (same store as "
    "`lop secret`). Store a credential as soon as you get one that outlives this session "
    "— a token you minted, a key the user pasted — rather than a plaintext .env or only "
    "this conversation; that is your decision to make. retrieve NEVER returns the value: "
    "use secrets without reading them, via $(lop secret get NAME) in bash or "
    'secrets["NAME"] in eval. list before asking the user for something they already '
    "gave you. Do not store a one-off value or provider keys the harness manages. Never "
    "echo a secret, write it to a file, or put it in a commit or PR. "
    "See guide://credentials."
)


def build_secret_tool(context: ToolContext) -> AgentTool | None:
    """CreateIf builder: exists only where a secret store is actually reachable.

    Gated on REACHABILITY, which is what ``AGENTS.md`` says a ``createIf``
    factory may ask. Two conditions, and both are about the dependency rather
    than about which host spawned the session:

    * the storage stack imports (a source checkout without the optional crypto
      dependency installed must not advertise a tool whose every call errors),
      and
    * a store already exists on disk, OR one could be created.

    The second is deliberately permissive: ``store`` is the verb that CREATES
    the store, so gating on "a store already exists" would make the first
    secret unstorable through the tool and leave the agent no way in.
    """
    del context  # gating is on the machine's store, not on session state
    try:
        from local_operator.secrets import keys  # noqa: F401
        from local_operator.secrets.access import open_store  # noqa: F401
    except Exception:
        # The crypto stack is an optional-in-practice dependency; a tree where
        # it will not import must not carry a tool that can only fail.
        logger.debug("secret tool unavailable: store modules will not import", exc_info=True)
        return None
    return AgentTool(
        name="secret",
        label="Secret",
        description=_DESCRIPTION,
        parameters=SecretParams.model_json_schema(),
        # write tier: store/update/delete mutate a durable, encrypted store the
        # operator owns, and `delete` is irreversible — the value is kept
        # nowhere else. `retrieve` is a read, but the approval tier is per tool
        # rather than per verb and the safe direction is to prompt.
        approval_tier="write",
        # Exclusive: the verbs open the same SQLite store and `rotate` (through
        # the CLI) re-seals every record. WAL handles concurrent readers, but
        # two writes racing from one session is a self-inflicted conflict with
        # no benefit.
        concurrency="exclusive",
        interruptible=False,
        execute=execute_secret,
    )


def _receipt(name: str) -> str:
    return _USE_HINT.format(name=name)


async def execute_secret(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Any = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Dispatch one ``secret`` verb against the encrypted store."""
    from local_operator.tools.builtin import _error, _text

    del signal, on_update
    try:
        params = SecretParams(**args)
    except ValidationError as exc:
        from local_operator.tools.builtin import _validation_error

        return _validation_error(tool_call_id, "secret", exc)

    from local_operator.secrets.access import open_store, session_id
    from local_operator.secrets.errors import SecretStoreError

    if params.op != "list" and not (params.name or "").strip():
        return _error(tool_call_id, "secret", f"'{params.op}' requires a secret name.")
    name = (params.name or "").strip()

    from local_operator.secrets.keys import store_path

    if params.op == "list" and not store_path().exists():
        # "Nothing stored yet" rather than an error. `list` is the verb an
        # agent uses to ORIENT — the guidance tells it to check what exists
        # before asking the user for a credential they already gave — and an
        # error here reads as "the store is broken" and invites a retry, when
        # the true and actionable answer is that this machine has no secrets
        # yet. Every other verb still reports the absent store, because for
        # them it is genuinely the reason they could not do the job.
        return _text(
            tool_call_id,
            "secret",
            "No secrets are stored on this machine yet. Use op='store' to add the first.",
            details={"op": "list", "count": 0},
        )

    try:
        # `create=True` only for the verb that legitimately creates a store.
        # A `retrieve` against a store that does not exist must say so rather
        # than initialise an empty one and then report the secret missing —
        # two different problems with two different fixes (the same reasoning
        # `access.open_store` documents for the CLI).
        store = open_store(create=params.op == "store")
        if params.op == "store":
            if not (params.value or "").strip():
                return _error(
                    tool_call_id,
                    "secret",
                    "'store' needs the value. If you do not have it, ask the user for it "
                    "with the ask tool (secret=true) instead of guessing.",
                )
            # `set` initializes the store itself, so the first `store` verb on
            # a machine with no store creates one.
            store.set(
                name,
                (params.value or "").encode("utf-8"),
                description=(params.description or "").strip(),
                session_id=session_id(),
            )
            return _text(
                tool_call_id,
                "secret",
                f"Stored {name} in the encrypted long-term store.\n{_receipt(name)}",
                details={"op": "store", "name": name},
            )
        if params.op == "update":
            if not (params.value or "").strip():
                return _error(tool_call_id, "secret", "'update' needs the replacement value.")
            store.update(
                name,
                (params.value or "").encode("utf-8"),
                description=params.description,
                session_id=session_id(),
            )
            return _text(
                tool_call_id,
                "secret",
                f"Updated {name}.\n{_receipt(name)}",
                details={"op": "update", "name": name},
            )
        if params.op == "retrieve":
            # `describe`, NOT `get`. The value is not returned, so fetching it
            # would move plaintext into this process and write a `get` audit
            # row for a retrieval that never handed out a byte — misreporting
            # the one trail the operator relies on. Existence is the whole
            # question this verb answers.
            record = store.describe(name)
            summary = f"{record.name} is stored"
            if record.description:
                summary += f" — {record.description}"
            return _text(
                tool_call_id,
                "secret",
                f"{summary}.\n{_receipt(record.name)}",
                details={"op": "retrieve", "name": record.name},
            )
        if params.op == "describe":
            record = store.describe(name)
            lines = [
                f"name: {record.name}",
                f"kind: {record.kind}",
                f"description: {record.description or '(none)'}",
            ]
            return _text(
                tool_call_id,
                "secret",
                "\n".join(lines),
                details={"op": "describe", "name": record.name},
            )
        if params.op == "delete":
            store.delete(name, session_id=session_id())
            return _text(
                tool_call_id,
                "secret",
                f"Deleted {name}. The value is not recoverable.",
                details={"op": "delete", "name": name},
            )
        records = store.list()
        if not records:
            return _text(
                tool_call_id,
                "secret",
                "No secrets are stored yet.",
                details={"op": "list", "count": 0},
            )
        width = max(len(record.name) for record in records)
        rows = [
            f"  {record.name.ljust(width)}  {record.description or ''}".rstrip()
            for record in records
        ]
        return _text(
            tool_call_id,
            "secret",
            "\n".join([f"{len(records)} stored secret(s) — names only, never values:", *rows]),
            details={"op": "list", "count": len(records)},
        )
    except SecretStoreError as exc:
        # Every store failure carries a message written for a human at a
        # terminal; passing it through verbatim is what lets the model act on
        # it (missing secret, damaged record, store written by a newer runtime,
        # broker down) instead of retrying blindly.
        return _error(tool_call_id, "secret", str(exc))
