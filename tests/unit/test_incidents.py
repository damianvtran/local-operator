"""Session incident classification and formatting."""

from __future__ import annotations

import pytest

from local_operator.harness.message_types import (
    SESSION_INCIDENT_MESSAGE_TYPE,
    SESSION_MCP_RECOVERY_MESSAGE_TYPE,
    SESSION_MODEL_SWITCH_MESSAGE_TYPE,
)
from local_operator.incidents import (
    classify_incident,
    format_incident_message,
    format_mcp_recovery_message,
    format_model_switch_message,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("429 Too Many Requests", "rate-limit"),
        ("RATE LIMIT exceeded for claude-opus-5", "rate-limit"),
        ("usage limit reached on this account", "rate-limit"),
        ("401 Unauthorized: invalid api key", "auth"),
        ("403 permission denied for this model", "auth"),
        ("refresh token expired", "auth"),
        ("402 payment required", "billing"),
        ("503 service unavailable", "provider"),
        ("bad gateway from upstream", "provider"),
        ("read timeout while streaming", "network"),
        ("connection reset by peer", "network"),
        ("SSL certificate verify failed", "network"),
        # The 2026-09-15 incident's exact text: anyio's aggregate carried up
        # through the harness's own transport wrapper.
        ("transient provider error: ConnectError: All connection attempts failed", "network"),
        ("maximum context length is 200000 tokens", "context-length"),
        ("the request was too large", "context-length"),
        ("MCP server 'linear' unavailable", "mcp"),
        # The DeepSeek thinking-mode validator's refusal, in the rendered form
        # the operator's own incidents carry (and as an aggregator relays it,
        # which is why it is named before the generic ``provider`` rule).
        (
            "invalid request (HTTP 400): The `reasoning_content` in the thinking "
            "mode must be passed back to the API.",
            "reasoning-echo",
        ),
        (
            "invalid request (HTTP 400): upstream error: The `reasoning_content` in "
            "the thinking mode must be passed back to the API.",
            "reasoning-echo",
        ),
        # The three texts the ORed rule misclassified as this category (review
        # round 1, MAJOR 1). Each names the field and none is the refusal, so
        # each must keep the category its OWN fault earns: a throttle is a
        # throttle, a relayed 502 is the provider, and the legacy rows' reject of
        # an input ``reasoning_content`` is the error the capability exists to
        # avoid -- reporting it as our recovery having failed inverted it.
        (
            '429 Too Many Requests: {"error":{"message":"rate limited",'
            ' "metadata":{"requested":{"reasoning_content":null}}}}',
            "rate-limit",
        ),
        (
            "502 Bad Gateway from upstream: the relay could not resolve "
            "reasoning_content for this turn",
            "provider",
        ),
        (
            "invalid request (HTTP 400): unsupported field 'reasoning_content' for "
            "model deepseek-reasoner",
            "unknown",
        ),
        ("something completely novel happened", "unknown"),
    ],
)
def test_classification(raw: str, expected: str) -> None:
    assert classify_incident(raw).category == expected


def test_render_carries_category_source_hint_and_raw():
    text = format_incident_message("429 quota exceeded", "anthropic", "claude-opus-5")
    assert text.startswith("[session incident (anthropic/claude-opus-5)] rate-limit:")
    assert "429 quota exceeded" in text
    assert "suggested action:" in text
    assert "previous turn ended" in text


def test_the_reasoning_echo_refusal_names_a_next_step():
    """A refusal the harness could not clear must not read as a generic 400.

    It is the one provider refusal whose recovery the harness attempted itself,
    so a user seeing it needs to know the attempt happened and that re-sending
    the same request unchanged is not the move.
    """
    incident = classify_incident(
        "invalid request (HTTP 400): The `reasoning_content` in the thinking "
        "mode must be passed back to the API."
    )
    assert incident.category == "reasoning-echo"
    assert "switching model" in incident.hint


def test_the_reasoning_echo_hint_does_not_claim_a_retry_that_may_not_have_run():
    """The hint is one static string; whether a retry happened is per-model.

    On a model with no thinking-off rung -- which includes the live aggregator
    routes to these weights -- the loop never re-asks, so a hint saying a retry
    "did not clear" this describes a call that was never made (QA round 1, Q2).
    The wording has to be true of both cases, since the category is the same in
    both and nothing on the incident says which one it was.
    """
    hint = classify_incident(
        "invalid request (HTTP 400): The `reasoning_content` in the thinking "
        "mode must be passed back to the API."
    ).hint
    assert "did not clear" not in hint
    assert "no such rung" in hint


def test_unknown_has_no_invented_hint():
    incident = classify_incident("a novel failure mode")
    assert incident.category == "unknown"
    assert incident.hint == ""


@pytest.mark.parametrize(
    "raw",
    [
        # The incident, verbatim. This classified ``provider`` because the
        # harness's own transport wrapper prefixes every wrapped failure with
        # the kind label "transient provider error", and "provider error" was
        # a token in the ``provider`` rule — which is ordered AHEAD of
        # ``network``. A pre-connect failure therefore never reached the rule
        # written for it, and the card told the operator the provider was
        # failing server-side while the machine could not open a socket at all.
        "transient provider error: ConnectError: All connection attempts failed",
        "transient provider error: ConnectError: [Errno 49] can't assign requested address",
        "transient provider error: ConnectError: [Errno 61] Connection refused",
    ],
)
def test_a_pre_connect_connect_failure_is_a_network_incident(raw: str) -> None:
    incident = classify_incident(raw, "deepseek", "deepseek-flash")
    assert incident.category == "network"
    assert incident.render().startswith("[session incident (deepseek/deepseek-flash)] network:")
    # The advice has to name the MACHINE: switching provider cannot help a box
    # that cannot reach the network, and telling the operator to do so is what
    # made them read a working cascade as a broken one.
    assert "machine" in incident.hint
    assert "switching provider" in incident.hint


@pytest.mark.parametrize(
    "raw",
    [
        "transient provider error (HTTP 503): upstream reset",
        "transient provider error (HTTP 502): upstream unavailable",
        # A provider narrating its OWN upstream trouble keeps ``provider``: it
        # is the provider's words, not ours, and it is the provider that is
        # unwell — see the KNOWN IMPRECISION note on the failover classifier,
        # which leans the same way for the backoff it chooses.
        "transient provider error: upstream: network is unreachable at edge",
    ],
)
def test_a_genuine_provider_side_failure_stays_a_provider_incident(raw: str) -> None:
    """The other direction. Removing the harness-authored token must not cost
    the 5xx/upstream cases their category — their evidence is the provider's
    own status and wording, which is what the rule now matches on alone."""
    incident = classify_incident(raw, "deepseek", "deepseek-flash")
    assert incident.category == "provider"
    assert "provider" in incident.hint


@pytest.mark.parametrize("status", [501, 505, 506, 507, 508, 510, 520, 529, 597])
def test_a_5xx_outside_the_enumerated_four_keeps_its_category_and_its_hint(status: int) -> None:
    """Agent review R1-1: the deleted label was these strings' ONLY matcher.

    Enumerating four statuses (500/502/503/504) left 501, 505-510, Cloudflare's
    52x and nginx's 597 falling through to ``unknown``, whose
    ``Incident.render`` omits the ``suggested action:`` line entirely — the
    model was told strictly less than before the change. The statused token
    ``http 5`` is what covers them now, and the harness's own rendering makes it
    unambiguous (``ProviderError.__str__`` writes ``(HTTP <status>)``).
    """
    incident = classify_incident(f"transient provider error (HTTP {status}): something went wrong")
    assert incident.category == "provider"
    assert incident.hint, "the suggested action must not disappear"
    assert "suggested action:" in incident.render()


@pytest.mark.parametrize(
    "name",
    ["ReadError", "WriteError", "CloseError", "ProtocolError", "RemoteProtocolError"],
)
def test_a_statusless_transport_class_keeps_its_category_and_its_hint(name: str) -> None:
    """The same hole, for the class names ``wrap_transport_error`` writes.

    These are routinely raised with an EMPTY detail — ``httpx.ReadError('')`` is
    what a TCP RST mid-body surfaces as — so with no status and no message there
    is no other token to match, and the message is exactly
    ``"transient provider error: <ClassName>"``.
    """
    incident = classify_incident(f"transient provider error: {name}")
    assert incident.category == "network"
    assert incident.hint, "the suggested action must not disappear"


def test_the_network_hint_speaks_for_both_halves_of_its_category() -> None:
    """Agent review R1-2: the category holds two opposite situations.

    A refused or reset connection means the far end DID answer, and the failover
    layer deliberately keeps rotating on it; a pre-connect failure means this
    machine could not reach anyone. Both land on ``network``, so the hint has to
    carry both branches — the previous wording told the refusal reader to
    distrust their own machine instead of switching target.
    """
    refused = classify_incident(
        "transient provider error: ConnectError: [Errno 61] Connection refused"
    )
    offline = classify_incident(
        "transient provider error: ConnectError: All connection attempts failed"
    )
    assert refused.category == offline.category == "network"
    assert "refused or reset" in refused.hint  # the far end answered
    assert "this machine" in offline.hint  # the far end was never reached
    assert "switching provider" in offline.hint


def test_message_type_constant_is_stable():
    # Persisted into transcripts; renaming it would orphan old sessions' replay.
    assert SESSION_INCIDENT_MESSAGE_TYPE == "session_incident"


def test_model_switch_deliberate_names_old_and_new_and_persists():
    text = format_model_switch_message(
        "anthropic/claude-opus-4-8",
        "zai/glm-5.3",
        reason="model switched",
    )
    assert text.startswith("[model switch] You are now running as anthropic/claude-opus-4-8")
    assert "was zai/glm-5.3" in text
    assert "applies from now on" in text
    # A deliberate switch is not transient, so it must not carry the fallback caveat.
    assert "temporary fallback" not in text


def test_model_switch_transient_fallback_is_marked_temporary():
    text = format_model_switch_message(
        "kimi/k3",
        "anthropic/claude-opus-4-8",
        reason="anthropic 429 — falling back",
        transient=True,
    )
    assert "You are now running as kimi/k3" in text
    assert "temporary fallback" in text
    assert "Reason: anthropic 429 — falling back" in text
    assert "applies from now on" not in text


def test_model_switch_without_previous_label_reads_cleanly():
    # The return-to-primary edge passes no previous label.
    text = format_model_switch_message("anthropic/claude-opus-4-8")
    assert text.startswith("[model switch] You are now running as anthropic/claude-opus-4-8.")
    assert "(was" not in text


def test_model_switch_message_type_constant_is_stable():
    # Persisted into transcripts; renaming it would orphan replay of old sessions.
    assert SESSION_MODEL_SWITCH_MESSAGE_TYPE == "session_model_switch"


@pytest.mark.parametrize(
    "raw",
    [
        "This model's maximum context length is 16385 tokens",
        "prompt is too long: 250000 tokens > 200000 maximum",
        "The input token count (1200000) exceeds the maximum number of tokens allowed",
        "The request exceeds the model's maximum context window",
        "Too many tokens in prompt",
        "too many tokens: input is larger than the model's context length",
        "request too large for model",
        "Input is too long for requested model",
    ],
)
def test_every_vendor_overflow_wording_is_recognised(raw: str) -> None:
    """The overflow rule has to cover how vendors ACTUALLY phrase it.

    An audit against real vendor output found the list recognising 6 of 10
    wordings: google/vertex counts input tokens, mistral and bedrock phrase it
    differently again. The gap mattered beyond this classifier, because
    `providers.clients` shares the list to decide that a relayed overflow is
    deterministic and must not be retried as upstream weather.
    """
    assert classify_incident(raw).category == "context-length"


@pytest.mark.parametrize(
    "raw",
    [
        # Verbatim AWS Bedrock ThrottlingException wording.
        "ThrottlingException: Too many tokens, please wait before trying again.",
        # A TPM limit that quotes a token count in passing.
        "Rate limit reached for gpt-4: Limit 90000 token count per min",
        "Number of request tokens has exceeded your per-minute rate limit",
    ],
)
def test_a_throttle_that_mentions_tokens_is_not_an_overflow(raw: str) -> None:
    """Rate limits must not be read as context overflows.

    context-length is the FIRST rule, so it outranks rate-limit: a bare "too
    many tokens" or "token count" marker captures these and tells the user to
    /compact a request whose only problem is that it arrived too soon. The
    markers are therefore qualified to name the INPUT, which a throttle never
    does.
    """
    assert classify_incident(raw).category != "context-length"


def test_recovery_text_names_server_count_and_supersedes() -> None:
    """The recovery must name the server, the count, and CANCEL the incident.

    All three are load-bearing. The server name is what ties it to the specific
    incident it supersedes. The count is concrete evidence the connection is
    real (and is the REGISTERED count, so it agrees with what the model can
    actually call). The supersede clause is the reason the message exists: the
    model is simultaneously holding an ``mcp`` incident whose hint says "its
    tools are gone ... Do not call its tools", and a bare "reconnected" leaves
    both claims live for it to choose between.
    """
    text = format_mcp_recovery_message("minerva-qa", 41)
    assert "minerva-qa" in text
    assert "41 tools are available again" in text
    assert "supersedes the earlier session incident" in text
    assert text.startswith("[mcp recovery]")
    # Tag is server-scoped, not session-scoped: [session incident] already owns
    # the session register, and the subject here is one server.
    assert "[session recovery]" not in text


def test_recovery_text_agrees_in_number() -> None:
    """Singular and plural, because the model reads this text.

    The zero case is a different sentence entirely and is pinned separately
    below; all this asserts here is that it never claims a count.
    """
    assert "1 tool is available again" in format_mcp_recovery_message("files", 1)
    assert "2 tools are available again" in format_mcp_recovery_message("files", 2)
    assert "0 tool" not in format_mcp_recovery_message("files", 0)


def test_recovery_text_promises_no_tools_when_none_are_enabled() -> None:
    """Zero registered tools must not be told to "call them normally".

    A server can be connected with every tool filtered out by
    ``disabledTools``, so this is a reachable state and not a degenerate one.
    The earlier count-free phrasing avoided "0 tools" but still said the tools
    "are available again" and "are usable now" — false against an empty
    inventory, and false in the expensive direction: the model spends a turn
    looking for tools that do not exist (review round 1, R2).

    What must survive is the SUPERSEDE clause. The model is still holding an
    incident saying this server is unreachable, and that part is genuinely no
    longer true, so the notice must still cancel it.
    """
    zero = format_mcp_recovery_message("files", 0)
    assert "is connected again" in zero
    assert "no enabled tools" in zero
    assert "supersedes the earlier session incident" in zero
    # The three over-claims the old zero branch made, none of which are true
    # with an empty inventory.
    assert "available again" not in zero
    assert "usable now" not in zero
    assert "call them normally" not in zero


def test_recovery_is_not_an_incident_type() -> None:
    """The distinct type is what keeps the classifier off this message.

    ``journal_incident`` classifies its input, and ``classify_incident``
    matches the substring "mcp" — so routing a recovery through the incident
    type would append "its tools are gone until it reconnects" and "This is why
    the previous turn ended" to a message announcing the opposite. This asserts
    both halves: the type is separate, and the classifier really would do that.
    """
    assert SESSION_MCP_RECOVERY_MESSAGE_TYPE != SESSION_INCIDENT_MESSAGE_TYPE
    assert SESSION_MCP_RECOVERY_MESSAGE_TYPE != SESSION_MODEL_SWITCH_MESSAGE_TYPE
    contradiction = classify_incident(format_mcp_recovery_message("files", 3))
    assert contradiction.category == "mcp"
    assert "tools are gone" in contradiction.render()


# ---------------------------------------------------------------------------
# The stop attribution: one act, one vocabulary, read back intact
# ---------------------------------------------------------------------------


def test_a_deliberate_stops_attribution_speaks_the_receipts_language() -> None:
    """Design round 1, D2/D3: the durable sentence must not spell a signal.

    The operator reads the stop RECEIPT at the keypress — ``killed "X"`` for the
    terminal rung, ``stopped "X" (sigterm)`` for the one before it — and the
    durable row lands in the list beside that receipt, so ``SIGKILL`` there split
    one act into two vocabularies on two surfaces a reader compares directly.
    The rung is named in the receipt's nouns, and an unrecognised rung renders
    as the neutral word rather than echoing the writer's token.
    """
    from local_operator.incidents import STOP_RUNG_LABELS, render_stop_attribution

    assert STOP_RUNG_LABELS == {
        "socket": "stopped on request",
        "sigterm": "stopped with a signal",
        "sigkill": "killed",
    }
    assert render_stop_attribution(rung="sigkill", command="lop stop") == (" (killed by lop stop)")
    assert render_stop_attribution(rung="socket", command="/stop") == (
        " (stopped on request by /stop)"
    )
    # A rung this build does not know: the operator's word, never the token.
    assert render_stop_attribution(rung="future-rung", command="lop stop") == (
        " (stopped by lop stop)"
    )
    for token in ("SIGKILL", "SIGTERM"):
        assert token not in render_stop_attribution(rung="sigkill", command="lop stop")


def test_the_killers_pid_is_labelled_because_the_neighbouring_one_is_not() -> None:
    """D2: one token, two opposite referents, one list.

    An error row's parenthetical names the runtime that DIED
    (``attention._record_detail``: build, pid, started-at); a deliberate row's
    names the process that killed it. Unlabelled, a reader comparing the two
    concludes the same relationship about both.
    """
    from local_operator.incidents import render_stop_attribution

    assert (
        render_stop_attribution(rung="sigkill", killer_pid=40609) == " (killed by killer pid 40609)"
    )
    assert render_stop_attribution(rung="sigkill", command="/stop", killer_pid=40609) == (
        " (killed by /stop, killer pid 40609)"
    )


def test_the_row_phrase_reads_back_out_of_the_durable_sentence() -> None:
    """The INVERSE, pinned as a round trip: three surfaces have only the sentence.

    ``rows.completion_notice`` and the sidebar entry are handed the stored reason
    and nothing else, so ``stop_rung_phrase`` recovers the attribution from the
    sentence and must spend it exactly when the ladder ESCALATED — a rung-1
    request is what the word ``Interrupted`` already means, and appending its own
    sentence would read ``Interrupted — the session was stopped by the user``.
    """
    from local_operator.incidents import (
        DELIBERATE_CUT_OFF_CAUSE,
        render_cut_off_reason,
        render_stop_attribution,
        stop_rung_phrase,
    )

    def reason(rung: str, *, command: str = "lop stop", pid: object = 4242) -> str:
        return render_cut_off_reason(
            DELIBERATE_CUT_OFF_CAUSE,
            detail=render_stop_attribution(rung=rung, command=command, killer_pid=pid),
        )

    assert stop_rung_phrase(reason("sigkill", command="/stop --all")) == "killed by /stop --all"
    assert stop_rung_phrase(reason("sigterm")) == "stopped with a signal by lop stop"
    # Rung 1 is the plain request, and an unrecognised rung has nothing the row
    # does not already say.
    assert stop_rung_phrase(reason("socket")) == ""
    assert stop_rung_phrase(reason("future-rung")) == ""
    # Not our sentence at all: a provider's prose, a pre-attribution reason, or
    # an involuntary cause must all leave the surface printing what it did.
    assert stop_rung_phrase("") == ""
    assert stop_rung_phrase("the session was stopped by the user") == ""
    assert stop_rung_phrase(render_cut_off_reason("runtime-killed")) == ""
    assert stop_rung_phrase("429 Too Many Requests") == ""


def test_the_list_column_explains_a_plain_request_the_row_word_covers() -> None:
    """``outcome_summary`` for a surface with no deliberate word of its own.

    ``lop sessions`` prints state, not ``Interrupted``, so its WHY column has to
    say something for a stop on the plain request rung — and for a death nobody
    asked for it is the reason's own first clause, with the build/pid
    parenthetical left to the ``--json`` row.
    """
    from local_operator.incidents import (
        DELIBERATE_CUT_OFF_CAUSE,
        outcome_summary,
        render_cut_off_reason,
        render_stop_attribution,
    )

    socket_stop = render_cut_off_reason(
        DELIBERATE_CUT_OFF_CAUSE,
        detail=render_stop_attribution(rung="socket", command="/stop", killer_pid=7),
    )
    assert outcome_summary(socket_stop) == "the session was stopped by the user"
    escalated = render_cut_off_reason(
        DELIBERATE_CUT_OFF_CAUSE,
        detail=render_stop_attribution(rung="sigkill", command="/stop --all", killer_pid=7),
    )
    assert outcome_summary(escalated) == "killed by /stop --all"
    killed = render_cut_off_reason("runtime-killed", detail=" (0.54.39@dec7933, pid 1)")
    assert outcome_summary(killed) == (
        "the runtime disappeared without exiting cleanly while this turn was running, "
        "and nothing recorded a stop"
    )


def test_runtime_killed_says_nothing_recorded_a_stop() -> None:
    """D4: the post-marker meaning of the token, in the sentence itself.

    The row above a deliberate stop reads ``Interrupted``, so "asked for" versus
    "never asked" is decided by this copy — and after this change that is
    decidable from the artifacts rather than guessed at.
    """
    from local_operator.incidents import cause_from_reason, render_cut_off_reason

    sentence = render_cut_off_reason("runtime-killed", detail=" (build, pid 1)")
    assert "nothing recorded a stop" in sentence
    # The inverse still recovers the token from the longer sentence.
    assert cause_from_reason(sentence) == "runtime-killed"


def test_a_detail_is_separated_from_the_sentence_it_rides_with() -> None:
    """The separator is the RENDERER's, because the callers disagree about it.

    ``render_cut_off_reason``'s output is what every surface prints (the live
    notice, the sidebar tooltip, the phone frame, the next turn's incident
    card) and what the durable outcome stores, so a detail glued to the
    sentence is a defect in all of them at once. Two callers hand over a string
    that already opens with a space and a bracket; ``process._drain_detail``
    hands over ``declined 3x (0.56.2 → 0.56.6)`` with neither, and rendering
    that by concatenation produced ``…would run a newer builddeclined 3x
    (0.56.2 → 0.56.6)`` on every retirement of a stale runtime (measured on the
    reporting host, 2026-09-17). Pinned as the three spellings that must all
    land: bare, already-parenthesised, and empty.
    """
    from local_operator.incidents import render_cut_off_reason

    sentence = "the runtime retired so the next engage would run a newer build"
    bare = render_cut_off_reason("runtime-retired", detail="declined 3x (0.56.2 → 0.56.6)")
    wrapped = render_cut_off_reason("runtime-retired", detail=" (0.56.2 → 0.56.6)")
    assert bare == f"{sentence} (declined 3x (0.56.2 → 0.56.6))"
    # ONE pair of brackets, not two: the detail was already a parenthetical.
    assert wrapped == f"{sentence} (0.56.2 → 0.56.6)"
    assert "builddeclined" not in bare
    assert render_cut_off_reason("runtime-retired") == sentence
    assert render_cut_off_reason("runtime-retired", detail="   ") == sentence
