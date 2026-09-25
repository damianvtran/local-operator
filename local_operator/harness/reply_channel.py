"""Offer a structured reply as the model's own tool channel, not only as prose.

Some requests do not want the model to ACT — they want one structured answer
back: the benchmark runner's action envelope, a compaction advisor's hint, any
"reply with this exact JSON object" errand. The established way to ask is prose
("reply with a single JSON object and nothing else") plus ``tool_choice="none"``.

That works until it meets a model whose tool channel is the strongest thing in
its post-training. Under pressure such a model answers on the channel it was
trained to answer on, emitting its own native call syntax as TEXT, and a strict
prose decoder throws the whole reply away — a full paid round trip returning
``not valid JSON: Expecting value: line 1 column 1``, with the model's INTENT
correct and only its CHANNEL wrong.

Measured over three cohorts on the same frozen benchmark tasks and the same
harness family, counting accepted batches against rejected replies:

    cohort   accepted   rejected   rejection rate
    luna          125         18            12.6%
    opus          149         16             9.7%
    kimi           46         36            43.9%

33 of kimi's 36 rejections were exactly this: native tool syntax where the
envelope belonged.

The fix is to stop making the two channels rivals. When the model spec says
tools are supported, the SAME envelope is also offered as one function whose
parameters ARE the envelope schema, and a call to it is read as the reply.
Nothing executes it — it is a reply channel, not a capability — so the caller
keeps driving whatever it drove before and both channels converge on one
validated structure.

Three properties this module exists to hold, each of which is load-bearing:

* **One validated structure.** :func:`envelope_from_tool_call` returns the
  argument JSON as TEXT for the caller's existing decoder. There is no second
  parser and no salvage path: a malformed reply on the tool channel fails in
  the same decoder, with the same message, as a malformed reply in prose. The
  channel changes where the bytes come from, never what counts as valid.
* **Cache stability.** The offered tool rides in the request prefix (the tools
  array sits at the FRONT of it — tools -> system -> messages on Anthropic, and
  inside the cached body on the OpenAI-compatible and Gemini wires), so it is
  built from static inputs and is byte-identical on every turn of a request
  family. A schema that varied per turn would re-write the prefix on each call
  and cost more than the rejections it saves.
* **No model, provider, or task knowledge.** The only question asked is the
  capability the spec already reports (``supports_tools``). Sniffing for a
  vendor's marker syntax and salvaging it would be a per-model rule that ages
  badly and silently widens what the harness accepts.

Deliberately NOT ``tool_choice="required"``. Forcing the call would remove the
prose path the other cohorts use successfully at a ~10% rejection rate, and a
forced call is a worse failure when a model has genuinely nothing to say. The
channel is OFFERED (``auto``); the prompt still describes the prose envelope,
and both are accepted.

**Which callers should adopt this, and which must not.** The test is whether
the request already sends a tool array, because this one is APPENDED to it:

* **Adopt it** when the request currently sends ``tools=[]`` and asks for a
  strict structured reply in prose. The benchmark runner's decision envelope
  (``evaluation/runner/provider_client.py``) is the first such caller: with an
  empty array the wire carries no ``tools`` and no ``tool_choice`` key at all,
  so the request neither offered the channel nor forbade it, and the tool this
  module adds is the entire prefix delta.
* **Do NOT adopt it** in ``Session.complete_aside`` or
  ``Session.advise_compaction``. Those deliberately send the WORKING TURN's
  live tool schema so their request rides the turn's already-cached prefix;
  the tools block is the front of that prefix, so appending one more function
  would diverge it at position 0 and force a full re-process at cache-WRITE
  price. That was measured (``scripts/measure_advisor_cache.py``: 0% hit,
  ``cache_write=14590`` against ``cache_write=568``), and it is the economics
  the whole advisor feature rests on. Their rejection path is also cheap —
  ``parse_hint`` returning ``None`` costs a fallback, not a re-prompt loop —
  so they have neither the problem nor the headroom this trades against.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from local_operator.harness.types import AgentTool, ToolCall

#: The reply channel is a CHANNEL, never a capability: nothing dispatches it,
#: so a name collision with a real tool would be a silent misroute rather than
#: a loud one. It is namespaced and stable — the name is part of the request
#: prefix, so renaming it per turn would break the prompt cache.
REPLY_CHANNEL_TOOL_NAME = "lop_structured_reply"


async def _never_executes(
    _name: str,
    _arguments: dict[str, Any],
    _signal: Any,
    _update: Any,
    _context: Any,
) -> Any:
    """Fail loudly if this tool is ever dispatched.

    The reply channel is consumed by the caller reading the stream, never by
    the tool loop. Reaching this function means some caller put the channel in
    a request whose tool calls are EXECUTED, which would run the model's reply
    as if it were an action — so it raises instead of returning a benign
    result that would hide the wiring mistake.
    """
    raise RuntimeError(
        f"{REPLY_CHANNEL_TOOL_NAME} is a reply channel and must never be executed; "
        "the caller is expected to read it from the stream"
    )


def build_reply_channel_tool(
    schema: dict[str, Any],
    *,
    description: str,
    name: str = REPLY_CHANNEL_TOOL_NAME,
) -> "AgentTool":
    """One function whose parameters ARE the caller's reply envelope.

    ``schema`` is passed through unchanged rather than rebuilt here: the
    envelope's shape stays owned by whoever defines the envelope, so this
    module cannot drift from the decoder that validates it. Callers that
    already publish a JSON Schema for their prose contract should hand that
    exact object over, which is what makes the two channels the same contract.
    """
    from local_operator.harness.types import AgentTool

    return AgentTool(
        name=name,
        label="Structured reply",
        description=description,
        parameters=schema,
        # Read tier and non-interruptible are the honest values for something
        # that performs no side effect; they matter only if a host renders the
        # tool, since nothing ever runs it.
        approval_tier="read",
        concurrency="shared",
        interruptible=False,
        # Kept out of any surface that lists callable tools to a user: this is
        # transport, not a capability someone can invoke.
        hidden=True,
        execute=_never_executes,
    )


def reply_channel_tools(
    model_spec: Any,
    schema: dict[str, Any],
    *,
    description: str,
    name: str = REPLY_CHANNEL_TOOL_NAME,
) -> list["AgentTool"]:
    """The channel when the spec supports tools, and an empty list when not.

    Gating on the spec's own capability flag is what keeps this general. A
    model with no tool support must still get the request it always got — an
    empty tools array, no ``tool_choice`` key on the wire at all — because
    offering a function to a model that cannot take one is at best ignored and
    at worst a provider-side rejection of the whole request.
    """
    if not getattr(model_spec, "supports_tools", False):
        return []
    return [build_reply_channel_tool(schema, description=description, name=name)]


def envelope_from_tool_call(
    calls: Iterable["ToolCall"],
    *,
    name: str,
    offered_names: Iterable[str] | None = None,
) -> str | None:
    """The raw argument JSON of the reply-channel call, or ``None``.

    Returns TEXT, not a parsed object, so the caller feeds it to the very
    decoder that validates a prose reply. That is the whole point: one
    validated structure, one set of rejection messages, and no second notion
    of what a well-formed envelope is.

    ``raw_arguments`` is preferred over the parsed ``arguments`` because it is
    what the provider actually sent. Round-tripping through ``json.dumps``
    would quietly REPAIR a duplicate key or a trailing-comma artefact that the
    strict decoder is supposed to reject, turning a channel meant to be
    transparent into a lenient one. It falls back to re-serializing only when
    the provider gave no raw string.

    A call that is present but carries nothing usable returns the empty string
    rather than ``None`` — a distinction the caller needs, because "the model
    used the channel and sent garbage" is a rejection to report, while "the
    model did not use the channel" means read the prose instead.

    ``offered_names`` is the tool names the REQUEST put on the wire, and
    passing it is what makes the channel survive the model naming the call
    something else. When the reply channel is the SOLE name offered, every call
    the stream carried is the channel's, whatever it was named: the request
    advertised no capability, so there is nothing else a call could be, and the
    name is a label the model invented for the one function it was given.

    That distinction is measured, not theoretical, and it is why the default is
    the strict name test rather than the tolerant one. The measurement is the
    arm's OSWorld bundles (``~/worktrees/osworld/runs``, all ``openrouter`` /
    ``deepseek-v4.1-flash``), counted 2026-09-25 when the corpus stood at 56
    episodes in 29 runs: of its 204 ``leading-delimiter`` refusal artifacts, 178
    published NO reply text at all — ``content_deltas=0`` with a streamed call
    (``tool_call_deltas`` 3-1124, ``stop=toolUse``), i.e. the decision arrived as
    a tool call and the harness read only the empty prose, then refused the turn
    for not beginning with ``{``. Both figures move with the corpus, so they
    carry the moment they were taken; the ``tool_call_deltas`` RANGE did not move
    across three recounts, and the mechanism is what the class is.

    The corroboration half of that argument is not something the bundles can
    produce: a reply that was READ leaves no stream record, so "other replies at
    the same shape were read and published" is not a figure the artifacts can
    support, and an earlier revision of this docstring quoted one anyway. The
    journal carries it instead — across the same corpus, 2085 ``model_response``
    events (counted the same day) stopped ``toolUse`` with a call and a non-empty
    published reply, i.e. the correctly-named channel being read at scale.

    Omitted (or naming any other set): only a call named ``name`` is the
    channel, which is the correct reading for a request that ALSO offered real
    tools — there a foreign name is a real capability being invoked, not a
    reply, and reading its arguments as one would execute a label the harness
    never advertised. This coercion is bounded by the PROSE as well as by
    everything downstream of this function, and the bound is applied by the
    CALLER: it passes ``offered_names`` only when the prose stated no decision of
    its own, because widening is a RECOVERY of a decision that would otherwise
    be lost and must not supersede one the model wrote
    (``provider_client._stream``, gated on ``public_reply._states_a_decision``).

    The widening is bounded by everything downstream of this function, which is
    unchanged: the bytes go to the same decoder, so a foreign-named call
    carrying two batches, a duplicate key, a non-object, or an object that
    fails validation is refused by exactly the rules a prose reply is refused
    by. No shape that was refused on the channel before is accepted now.
    """
    calls = list(calls)
    if offered_names is not None and list(offered_names) == [name]:
        matched = calls
    else:
        matched = [call for call in calls if call.name == name]
    if not matched:
        return None

    # Two calls naming the channel is the SAME ambiguity the prose decoder
    # refuses -- "which one did the model mean?" -- and taking the first would
    # execute a decision the model may have superseded. That holds for two
    # FOREIGN-named calls too, which is the case the sole-offer rule above now
    # admits: two calls carrying two batches are refused here rather than
    # resolved by order, so widening WHICH calls are read cannot widen how many
    # of them may be executed. MEASURED, and narrower than the sentence above
    # reads: the refusal fires on two batches bound to the SAME observation,
    # while two batches naming DIFFERENT observations are accepted on both
    # channels by the decoder's older-observation rule (see below). What the
    # channel guarantees is that at most one batch EXECUTES, which is the
    # property this comment needs.
    #
    # Rather than inventing a second rejection, hand the decoder every envelope
    # the model sent, concatenated, and let its own rules judge them. The
    # property being preserved is PARITY, not any particular message: the same
    # concatenated text fails identically whether it arrives as prose or as
    # calls, so the channel cannot accept a shape prose refuses.
    #
    # Which rule fires is MEASURED rather than assumed (probe over head and
    # base, 2026-09-25). The framing account this comment used to give was wrong
    # in three ways: trailing text after a complete JSON value is TOLERATED on
    # both bindings — the decoder logs it and accepts the batch, and
    # ``MAX_TOLERATED_TRAILING_CHARS`` bounds only the quoting in that warning —
    # so nothing here is refused "on framing"; two batches for DIFFERENT
    # observations are ACCEPTED on both bindings, under the older-observation
    # quotation rule; and the competing-batch refusal is keyed on
    # ``observation_id`` and fires only on the LEGACY binding — under
    # ``compact`` even two distinct envelopes both bound to the current
    # observation are accepted. The arm's own path is legacy
    # (``scripts/run_episode.py --action-binding`` defaults to ``legacy``), so
    # the same-observation bound this comment needs does hold where the widening
    # is used. Either way the channel inherits the decoder's verdict: this
    # function adds no rule of its own beyond which calls carry text.
    #
    # Only calls that actually CARRY an envelope are joined. Joining empty ones
    # would produce a string of separators -- truthy, so the caller would
    # prefer it over a perfectly good prose reply and destroy it, which is
    # exactly the regression the empty-single-call path already guards against.
    if len(matched) > 1:
        carried = [text for text in map(_call_text, matched) if text.strip()]
        if not carried:
            return ""
        if len(carried) == 1:
            return carried[0]
        return "\n".join(carried)

    return _call_text(matched[0])


def _call_text(call: "ToolCall") -> str:
    """The envelope text one channel call carries, empty when it carries none.

    ``raw_arguments`` is preferred over the parsed ``arguments`` because it is
    what the provider actually sent; re-serializing would quietly REPAIR a
    duplicate key or trailing-comma artefact the strict decoder must reject.
    """

    raw = call.raw_arguments
    if raw is not None:
        return raw
    return json.dumps(call.arguments) if call.arguments else ""
