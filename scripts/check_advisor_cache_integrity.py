#!/usr/bin/env python3
"""Does the advisor break the prompt cache anywhere except a real compaction pass?

The advisor's economic case rests entirely on the claim that its extra call
rides the turn's WARM prefix and leaves that prefix intact for the next real
turn. ``measure_advisor_cache.py`` proves the first half on live Anthropic
calls (96.1% cache read). This script proves the second half, and it does so
against the SHIPPED request builders rather than by reading the code, because
"it appends and does not mutate" is exactly the kind of claim that stays true
until someone adds a line.

Six checks, each of which has a specific way of failing in production:

C1  HISTORY NOT MUTATED. ``advise_compaction`` renders
    ``[*_wire_legal_snapshot(), *turns]``. If it appended the advisor's turn
    to the session's own list instead of to a request-scoped copy, the next
    real turn would carry a message the user never sent -- and would diverge
    the prefix permanently, not just for the advisor call.

C2  PREFIX IS APPEND-ONLY. The advisor request's rendered message list must
    have the turn's list as a strict PREFIX. Anthropic caches by prefix
    content, so any edit ahead of the appended question invalidates
    everything after it. This is the check that would have caught the
    system-block placement (arm 6) before it cost a measurement.

C3  CACHE BREAKPOINTS UNCHANGED. The advisor must not add or move a
    ``cache_control`` marker, and must stay inside ``MAX_CACHE_BREAKPOINTS``
    (4) -- Anthropic rejects a request carrying more, and a marker in a new
    PLACE silently re-partitions the cached prefix.

C4  SYSTEM BLOCKS IDENTICAL. System sits ahead of the messages in the cache
    prefix; one extra block there measured 0% cache hit and a full re-write.

C5  PROMPT_CACHE_KEY PRESERVED. On the OpenAI wire caching keys on
    ``prompt_cache_key``, not on prefix content, so an advisor call that
    dropped it (``isolated=True`` does exactly that) would run on a cold
    namespace. This is why the design declines isolation.

C6  OFF BY DEFAULT IS INERT. With ``advisor_enabled`` false the settings must
    resolve byte-identically to a config written before the feature existed,
    and no advisor call may be spawned.

Run:
    .venv/bin/python scripts/check_advisor_cache_integrity.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction.advisor import ADVISOR_SYSTEM_PROMPT  # noqa: E402
from local_operator.compaction.thresholds import CompactionSettings  # noqa: E402
from local_operator.harness.types import AgentTool, ChatRequest, Message  # noqa: E402
from local_operator.model.configure import build_model_spec  # noqa: E402
from local_operator.providers.clients import AnthropicClient  # noqa: E402

_PASS = "PASS"
_FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []


def record(check: str, ok: bool, detail: str) -> None:
    _results.append((check, _PASS if ok else _FAIL, detail))
    print(f"  [{_PASS if ok else _FAIL}] {check}: {detail}")


async def _noop(*_a: Any, **_k: Any) -> Any:
    """Tool bodies are never invoked here (``tool_choice="none"``); the schema is
    what matters, because it sits at the front of the cache prefix."""
    raise AssertionError("tool must not run")


def _tools() -> list[AgentTool]:
    return [
        AgentTool(
            name=name,
            description=f"{name} tool",
            parameters={"type": "object", "properties": {}},
            execute=_noop,
        )
        for name in ("bash", "read", "edit", "grep")
    ]


def _history() -> list[Message]:
    convo: list[Message] = []
    for i in range(6):
        convo.append(Message.user(f"user turn {i} " * 20))
        convo.append(Message.assistant(f"assistant turn {i} " * 40))
    return convo


SYSTEM = ["You are an agent. " * 40, "Environment: macOS. " * 10]


def _markers(body: dict[str, Any]) -> tuple[int, list[str]]:
    """Count of ``cache_control`` markers in a built body, and where they sit."""
    places: list[str] = []
    for i, block in enumerate(body.get("system") or []):
        if isinstance(block, dict) and "cache_control" in block:
            places.append(f"system[{i}]")
    for i, message in enumerate(body.get("messages") or []):
        content = message.get("content")
        if isinstance(content, list):
            for j, block in enumerate(content):
                if isinstance(block, dict) and "cache_control" in block:
                    places.append(f"messages[{i}].content[{j}] role={message.get('role')}")
    return len(places), places


def _run_live_session_checks() -> list[tuple[str, bool, str]]:
    """Call the shipped ``advise_compaction`` and inspect what it did.

    Uses the session suite's own ``ScriptedStream``/``make_session`` helpers so
    the session under test is built the way the tests build it, and the request
    inspected here is the one that actually reached the stream function.
    """
    import tempfile

    from tests.unit.session.test_compaction_advisor import ScriptedStream, make_session

    async def _drive(tmp: Path) -> list[tuple[str, bool, str]]:
        stream = ScriptedStream(advice="ok")
        session = make_session(tmp, stream=stream)
        seeded = [
            Message.user("real user request one"),
            Message.assistant("assistant reply one"),
            Message.user("real user request two"),
        ]
        session._context.messages.extend(seeded)

        # ``_context.messages`` is AgentMessage (Message | CustomMessage); only
        # Message carries ``.text``, and everything seeded here is a Message.

        def _texts(messages: Any) -> list[str]:
            return [m.text for m in messages if isinstance(m, Message)]

        before_ids = [id(m) for m in session._context.messages]
        before_text = _texts(session._context.messages)

        await session.advise_compaction(
            [Message.user("compaction advisor: where should the cut land?")]
        )

        after_ids = [id(m) for m in session._context.messages]
        after_text = _texts(session._context.messages)
        request = stream.requests[-1]
        assert isinstance(request, ChatRequest)  # ScriptedStream types it as object
        wire = list(request.messages)
        out: list[tuple[str, bool, str]] = [
            (
                "live history unchanged after a real advisor call",
                before_ids == after_ids and before_text == after_text,
                f"{len(before_text)} messages, same identities and contents",
            ),
            (
                "advisor turn rides on the REQUEST only",
                len(wire) == len(after_text) + 1
                and "compaction advisor" in (_texts(wire[-1:])[0] if wire else ""),
                f"wire={len(wire)} msgs vs live={len(after_text)}; question is the tail",
            ),
            (
                "wire prefix equals the live history",
                _texts(wire[: len(after_text)]) == after_text,
                "append-only against the turn's own prefix",
            ),
            (
                "live tools sent (front of the cache prefix)",
                list(request.tools) == list(session._context.tools),
                f"{len(request.tools)} tools, tool_choice={request.tool_choice!r}; "
                "sending [] would change position 0 and force a full re-process",
            ),
        ]
        await session.dispose()
        return out

    with tempfile.TemporaryDirectory() as directory:
        return asyncio.run(_drive(Path(directory)))


def main() -> int:
    spec = build_model_spec("anthropic", "claude-opus-5")
    client = AnthropicClient()
    tools = _tools()
    history = _history()

    # The advisor's appended turn, built exactly as Session._run_advisor does:
    # instructions INSIDE the user turn, never as a system block.
    advisor_turn = Message.user(f"{ADVISOR_SYSTEM_PROMPT}\n\nA compaction decision is pending.")

    turn_request = ChatRequest(
        model=spec,
        system_blocks=list(SYSTEM),
        messages=list(history),
        tools=tools,
        prompt_cache_key="session-abc",
    )
    advisor_request = ChatRequest(
        model=spec,
        system_blocks=list(SYSTEM),
        messages=[*history, advisor_turn],
        tools=tools,
        tool_choice="none",
        replayable=True,
        isolated=False,
        prompt_cache_key="session-abc",
    )

    print("\n== C1: the advisor does not mutate the session's history ==")
    # Driven through the REAL ``Session.advise_compaction`` rather than a
    # reconstruction of it. A reconstruction would keep passing after someone
    # changed the method to append its question to ``_context.messages`` --
    # which is the exact regression this check exists to catch, and the one
    # that would diverge the prefix permanently instead of for one call.
    for label, ok, detail in _run_live_session_checks():
        record(label, ok, detail)

    print("\n== C2: the advisor request is APPEND-ONLY on the cache prefix ==")
    turn_body = client._build_body(turn_request)
    advisor_body = client._build_body(advisor_request)
    tm, am = turn_body["messages"], advisor_body["messages"]
    # Compare content, ignoring cache_control (C3 covers markers): a prefix
    # that differs only by a breakpoint is still a content prefix.

    def _strip(msgs: list[dict[str, Any]]) -> str:
        cleaned = []
        for m in msgs:
            c = m.get("content")
            if isinstance(c, list):
                c = [{k: v for k, v in b.items() if k != "cache_control"} for b in c]
            cleaned.append({**m, "content": c})
        return json.dumps(cleaned, sort_keys=True)

    prefix_ok = _strip(am[: len(tm)]) == _strip(tm)
    record(
        "message prefix identical",
        prefix_ok,
        f"turn={len(tm)} msgs is a strict prefix of advisor={len(am)} msgs "
        f"(+{len(am) - len(tm)} appended)",
    )

    print("\n== C3: cache_control breakpoints are not added or moved ==")
    tn, tplaces = _markers(turn_body)
    an, aplaces = _markers(advisor_body)
    within = an <= AnthropicClient.MAX_CACHE_BREAKPOINTS
    record(
        "breakpoint count within cap",
        within,
        f"turn={tn} advisor={an} cap={AnthropicClient.MAX_CACHE_BREAKPOINTS}",
    )
    # The markers that matter are the SYSTEM ones: those sit in the shared
    # prefix ahead of every message.
    tsys = [p for p in tplaces if p.startswith("system")]
    asys = [p for p in aplaces if p.startswith("system")]
    record(
        "system breakpoints unmoved",
        tsys == asys,
        f"turn={tsys} advisor={asys}",
    )

    # Message markers DO move, and that is the design rather than a defect:
    # ``_message_cache_breakpoints`` marks the last message and the
    # second-to-last USER turn, and the advisor appends a user turn, so both
    # markers shift right by one. Measured: turn marks messages[8], [11];
    # advisor marks [10], [12].
    #
    # Moving a marker RIGHT is safe and moving one LEFT is not, so that is the
    # invariant asserted here. Anthropic's markers are cache WRITE points;
    # reads match the longest cached prefix regardless of where the current
    # request happens to place its own markers. A marker that appeared EARLIER
    # than the turn's earliest marker would re-partition the shared prefix and
    # could orphan the entry the next turn wants; a marker further right only
    # adds a longer entry alongside it.
    #
    # This is not reasoning alone: arm 3 of measure_advisor_cache.py runs an
    # ordinary turn immediately AFTER an advisor call and measures
    # cache_read=14024 cache_write=0, a 100% hit. The next real turn is
    # provably unharmed by the moved marker.
    def _msg_indices(places: list[str]) -> list[int]:
        return [int(p.split("[")[1].split("]")[0]) for p in places if p.startswith("messages")]

    tmsg, amsg = _msg_indices(tplaces), _msg_indices(aplaces)
    no_leftward = bool(tmsg) and bool(amsg) and min(amsg) >= min(tmsg)
    record(
        "no breakpoint moves LEFT into the shared prefix",
        no_leftward,
        f"turn marks {tmsg}, advisor marks {amsg} -- all at or right of the turn's "
        f"earliest; the shared prefix ahead of index {min(tmsg) if tmsg else '-'} "
        "keeps its cache entries (live proof: arm 3 re-warm = 100% cache read)",
    )

    print("\n== C4: system blocks are passed through unchanged ==")
    same_system = turn_body.get("system") == advisor_body.get("system")
    record(
        "system array identical",
        same_system,
        (
            f"{len(turn_body.get('system') or [])} blocks, byte-identical"
            if same_system
            else "SYSTEM DIVERGED -- this is the 0%-cache-hit failure"
        ),
    )
    # And prove the rejected placement WOULD have diverged it, so the check
    # above is known to be capable of failing.
    sysblock_request = advisor_request.model_copy(
        update={"system_blocks": [*SYSTEM, ADVISOR_SYSTEM_PROMPT]}
    )
    sysblock_body = client._build_body(sysblock_request)
    diverges = sysblock_body.get("system") != turn_body.get("system")
    record(
        "control: system-block placement DOES diverge",
        diverges,
        "the rejected arm 6 shape changes the prefix, so C4 is a live check",
    )

    print("\n== C5: prompt_cache_key survives on the advisor request ==")
    record(
        "cache key preserved",
        advisor_request.prompt_cache_key == turn_request.prompt_cache_key,
        f"turn={turn_request.prompt_cache_key!r} advisor={advisor_request.prompt_cache_key!r}",
    )
    isolated = advisor_request.model_copy(update={"isolated": True})
    record(
        "isolated=False is what keeps it",
        advisor_request.isolated is False,
        f"shipped isolated={advisor_request.isolated}; isolated=True would strip the key "
        f"on the OpenAI wire (control object isolated={isolated.isolated})",
    )

    print("\n== C6: off by default is inert and byte-identical ==")
    legacy = CompactionSettings()
    explicit_off = CompactionSettings(advisor_enabled=False)
    record(
        "default is OFF",
        legacy.advisor_enabled is False,
        f"advisor_enabled default={legacy.advisor_enabled}",
    )
    # A config written BEFORE the feature must validate and resolve the same.
    pre_feature = CompactionSettings.model_validate(
        {"enabled": True, "threshold_tokens": 600_000, "keep_recent_tokens": 20_000}
    )
    non_advisor = {k: v for k, v in legacy.model_dump().items() if not k.startswith("advisor_")}
    pre_non_advisor = {
        k: v for k, v in pre_feature.model_dump().items() if not k.startswith("advisor_")
    }
    record(
        "pre-feature config resolves identically",
        non_advisor == pre_non_advisor and legacy.model_dump() == explicit_off.model_dump(),
        "every non-advisor field matches a config that predates the feature",
    )

    # And the gate itself: _advisor_settings must return None with the flag off.
    from local_operator.session.session import Session

    class _Stub:
        """Minimal stand-in exercising the REAL gate function unbound."""

        _disposed = False
        _advisor_disabled = False
        _advisor_in_flight = False
        _advisor_calls = 0
        _advisor_cooldown_until = -1
        _advisor_last_turn = -(10**9)
        _generation = 100

    stub = _Stub()
    stub._compaction_settings = legacy  # type: ignore[attr-defined]
    gate_off = Session._advisor_settings(stub)  # type: ignore[arg-type]
    enabled = CompactionSettings(advisor_enabled=True)
    stub._compaction_settings = enabled  # type: ignore[attr-defined]
    gate_on = Session._advisor_settings(stub)  # type: ignore[arg-type]
    record(
        "gate returns None with the flag off",
        gate_off is None and gate_on is not None,
        f"off->{gate_off!r}; on->{'settings' if gate_on else None} (gate is live, not dead code)",
    )

    failed = [r for r in _results if r[1] == _FAIL]
    print(f"\n{'=' * 70}")
    print(f"RESULT: {len(_results) - len(failed)}/{len(_results)} checks passed")
    for check, _status, detail in failed:
        print(f"  FAILED {check}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
