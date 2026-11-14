"""Whether a provider turn carried NOTHING -- the discriminator for the empty-response retry.

Lives beside ``harness.loop`` (its one caller) rather than in ``harness.types``:
that module is imported by most of the tree, and a predicate this narrow should
not make every importer's tests "affected".
"""

from __future__ import annotations

from local_operator.harness.types import Message


def is_provider_empty(message: Message) -> bool:
    """Whether the provider sent NOTHING for this turn -- not "the model said nothing".

    A clean ``stop`` with no visible text and no tool calls has two very
    different causes the loop's wordless-stop arm cannot tell apart on its own:
    (a) the provider's response carried no content at all (a 200 stream with
    zero ``choices``/content blocks, a dropped completion), where re-issuing
    the SAME request is the remedy and a "you said nothing" prompt asks the
    model to explain something it never produced; or (b) the model thought and
    produced no final text, where the existing re-ask is the remedy. Only (a)
    returns ``True``.

    The test is the ABSENCE of every trace of generation, so a wire this
    function does not know how to read errs towards (b), the established
    behaviour. Each wire leaves a different trace, and each is checked:

    * ``usage.reasoning_tokens`` -- the ONLY trace on the OpenAI Responses wire
      (its reasoning item carries no ``encrypted_content`` unless asked, so no
      ``native_replay`` payload is kept), and on Gemini's ``thoughtsTokenCount``.
    * ``usage.output_tokens`` -- generation the client could not show: hidden
      reasoning on a route that does not break it out, or output the client's
      markup filter removed (a bare ``<think>`` wrapper). A provider that
      reports output tokens did not send nothing.
    * ``provider_payload["native_replay"]["items"]`` -- Anthropic ``thinking`` /
      ``redacted_thinking`` blocks, Chat-Completions ``reasoning_content`` /
      ``reasoning`` fragments, Gemini ``thought`` parts. Every wire builds this
      payload only when the response carried such content, so ANY item counts.
    """
    if message.role != "assistant" or message.stop_reason != "stop":
        return False
    # RAW text, not stripped: a bare newline is something the provider sent (the
    # measured OSWorld wordless shape), so it stays on the re-ask path.
    if message.tool_calls or message.text:
        return False
    usage = message.usage
    if usage is not None and (usage.reasoning_tokens > 0 or usage.output_tokens > 0):
        return False
    replay = (message.provider_payload or {}).get("native_replay")
    if isinstance(replay, dict) and replay.get("items"):
        return False
    return True
