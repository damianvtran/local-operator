#!/usr/bin/env python3
"""Live proof that a FORKED session keeps OpenAI prompt-cache warmth.

Why this script exists
----------------------
``/fork`` (PR #437) makes a fork inherit its PARENT's ``prompt_cache_key`` so
the branch's first request is routed to the prefix the parent already warmed.
The plumbing shipped, but the acceptance evidence
(``docs/evidence/fork/cache-warmth.txt``) could only measure the ANTHROPIC path
and said so honestly: Anthropic keys its cache on prefix CONTENT, so a fork hits
there on byte-identity alone and the inherited key is NEUTRAL. That evidence
proved nothing about the one path where the key is actually written to the wire
-- ``OpenAICompatClient._build_responses_body`` -- and the shipped file called
the inheritance "unverified-but-plausible on the one path where it actually
matters". This script removes that asterisk.

The measurement vehicle, and why it is OpenRouter
-------------------------------------------------
The stored OpenAI credential is a ChatGPT/Codex OAuth account, which rejects
direct ``gpt-5`` calls ("The 'gpt-5' model is not supported when using Codex
with a ChatGPT account"), and the Codex-backed model that IS reachable reports
``cached_tokens`` of 0 even for a trivially-warm same-key control -- so that
endpoint cannot demonstrate a difference either way. There is no direct OpenAI
API key in the store.

OpenRouter serves real ``openai/gpt-5.x`` over a real ``/responses`` endpoint,
honours ``prompt_cache_key``, and RETURNS ``input_tokens_details.cached_tokens``.
Each leg of that is verified by arm 0 below rather than assumed.

The harness pins every non-``openai`` provider to ``chat_completions``
(``client_for_spec``), and a chat-completions body never carries
``prompt_cache_key`` -- only ``_build_responses_body`` sets it. Driving
OpenRouter through the ordinary provider path would therefore measure a request
WITHOUT the key and prove nothing. This script constructs
``OpenAICompatClient`` with ``openai_api="responses"`` explicitly, which is the
same code path an OpenAI API key takes, so the body on the wire is the one under
test.

The design, and why each arm is PAIRED on its own fresh prefix
--------------------------------------------------------------
A first attempt ran all five arms against ONE prefix in sequence, and arms 3-5
all read warm -- an ordering artifact, not a result: by the time they ran, the
prefix had already been sent twice and OpenAI's content-addressed cache had it
regardless of key. The corrected design gives EVERY arm its own nonce'd prefix
and its own cold write, so an arm can only read what its own pair warmed:

    trial = [ call 1: fresh prefix, PARENT key   -> must be cold (write) ]
            [ call 2: same prefix, the arm's key -> does it read?        ]

Arms, each repeated N times with a fresh prefix per trial:

  inherited -- call 2 carries the key a real ``fork_session()`` inherited.
               THIS IS THE NUMBER THAT MATTERS.
  own       -- call 2 carries its OWN distinct key. Separates "the inherited
               key did it" from "the provider keys on content anyway". On
               Anthropic this arm READ, which is what proved the key neutral
               there.
  none      -- call 2 carries no key at all.
  wrong     -- call 2 carries a deliberately WRONG key (negative control):
               shows the key is load-bearing rather than incidental.

Reading the result: the non-matching arms are NOT expected to be uniformly
cold. OpenAI's cache is content-addressed underneath, so a keyless or
wrong-keyed repeat sometimes still lands on the warm prefix; the key is a
routing/stickiness hint on top of that, not a lock. The signal is the RATE
difference between ``inherited`` and the others, which is why every arm is
repeated and the trials are interleaved rather than run in blocks.

Usage
-----
    OPENROUTER_API_KEY=... .venv/bin/python \
        docs/evidence/fork-cache/measure_fork_cache_openai.py [trials]

The key is read from the environment and is NEVER printed. Exit code is always
0 -- this is a measurement tool, not a gate.
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
import tempfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from local_operator.fork import fork_parent, fork_session  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    ChatRequest,
    Message,
    StreamUsageEvent,
)
from local_operator.model.configure import (  # noqa: E402
    build_model_spec,
    create_stream_fn,
)
from local_operator.providers.clients import OpenAICompatClient  # noqa: E402
from local_operator.resume import TRANSCRIPT_NAME  # noqa: E402

MODEL = "openai/gpt-5.4"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

#: Large enough to clear OpenAI's 1024-token cache floor by a wide margin. A
#: toy prefix would never cache and the measurement would be meaningless.
PREFIX_LINES = 1400

#: Courtesy spacing between live calls; also keeps a trial's two calls far
#: enough apart that the second is a genuine cache lookup rather than a
#: coalesced duplicate.
PAUSE_S = 1.5


def build_prefix(nonce: str) -> str:
    """A big prefix, unique per trial so nothing is warm before its first call."""
    body = "".join(
        f"Fact {i}: widget {i} tolerance {i * 7 % 13} microns fallback path.\n"
        for i in range(PREFIX_LINES)
    )
    return f"Session nonce {nonce}.\n{body}"


def make_request(prefix: str, cache_key: str | None) -> ChatRequest:
    spec = build_model_spec("openrouter", MODEL)
    # The registry marks this model cache-capable from its quoted cache-read
    # price; assert rather than assume, because `_build_responses_body` gates
    # the key on exactly this flag.
    assert spec.supports_prompt_cache, f"{MODEL} is not cache-capable in the registry"
    return ChatRequest(
        model=spec,
        system_blocks=[prefix],
        messages=[Message.user("Reply with one word: ok")],
        max_tokens=16,
        prompt_cache_key=cache_key,
    )


def client() -> OpenAICompatClient:
    """The harness' own wire client, forced onto the Responses path.

    See the module docstring: ``client_for_spec`` would pin OpenRouter to
    ``chat_completions``, whose body never carries the key under test.
    """
    return OpenAICompatClient(base_url=OPENROUTER_BASE, openai_api="responses")


async def call(api_key: str, prefix: str, cache_key: str | None) -> tuple[int, int]:
    """One live call. Returns ``(cache_read_tokens, input_tokens)``."""
    usage = None
    async for event in client().stream(make_request(prefix, cache_key), api_key):
        if isinstance(event, StreamUsageEvent):
            usage = event.usage
    return (
        (usage.cache_read_tokens or 0) if usage else 0,
        (usage.input_tokens or 0) if usage else 0,
    )


async def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("OPENROUTER_API_KEY not set -- skipping live measurement")
        return
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    run_nonce = uuid.uuid4().hex

    # --- A REAL fork, through the REAL fork_session() ------------------------
    cfg = Path(tempfile.mkdtemp(prefix="fork-cache-"))
    parent_id = f"parent-{run_nonce[:12]}"
    parent_dir = cfg / "sessions" / parent_id
    parent_dir.mkdir(parents=True)
    (parent_dir / TRANSCRIPT_NAME).write_text('{"kind":"user","text":"hello"}\n')

    fork_id = fork_session(cfg, parent_id)
    fork_dir = cfg / "sessions" / fork_id

    def stream_fn_for(session_dir: Path):
        """Wired EXACTLY as ``session_factory`` wires one at boot."""

        class _Auth:
            pass

        return create_stream_fn(
            _Auth(),  # type: ignore[arg-type]
            settings={},
            session_id=session_dir.name,
            cache_lineage_id=fork_parent(session_dir) or None,
        )

    parent_fn = stream_fn_for(parent_dir)
    fork_fn = stream_fn_for(fork_dir)

    print("=== Fork lineage, resolved through the real session_factory wiring ===\n")
    print(f"parent session id         : {parent_fn._session_id}")
    print(f"fork session id           : {fork_fn._session_id}")
    print(f"fork origin.json parent   : {fork_parent(fork_dir)}")
    print(f"parent cache key          : {parent_fn._cache_lineage_id}")
    print(f"fork cache key (inherited): {fork_fn._cache_lineage_id}")
    print(f"fork sticky-cred scope    : {fork_fn._session_id}")
    separate = fork_fn._session_id != fork_fn._cache_lineage_id
    print(f"cache key vs cred scope   : {'SEPARATE (correct)' if separate else 'SHARED (BUG)'}")

    parent_key = parent_fn._cache_lineage_id
    fork_key = fork_fn._cache_lineage_id
    assert fork_key == parent_id, "the fork did not inherit the parent's cache key"

    # --- Arm 0: the wire check ----------------------------------------------
    print("\n=== 0 WIRE CHECK: is the key actually on the wire? ===\n")
    body = client()._build_responses_body(make_request("probe", parent_key))
    print(f"body carries prompt_cache_key      : {'prompt_cache_key' in body}")
    print(f"prompt_cache_key == parent id      : {body.get('prompt_cache_key') == parent_id}")
    print(f"body carries prompt_cache_retention: {body.get('prompt_cache_retention')}")

    # --- The measurement -----------------------------------------------------
    # ``pk`` is THIS TRIAL's inherited key, derived from that trial's own real
    # fork -- see the loop below for why it cannot be a single fixed value.
    arms = {
        "inherited": lambda n, pk: pk,
        "own": lambda n, pk: f"own-{n}",
        "none": lambda n, pk: None,
        "wrong": lambda n, pk: f"wrong-{uuid.uuid4().hex}",
    }
    print("\n=== Cache warmth on the OpenAI Responses path ===\n")
    print(f"model: openrouter/{MODEL} (real openai/gpt-5.x over /responses)")
    print(f"design: paired, fresh nonce'd prefix per trial, {trials} trials per arm")
    print(f"run nonce: {run_nonce}\n")

    results: dict[str, list[tuple[int, int]]] = {name: [] for name in arms}
    # Interleaved, not blocked: a block per arm would confound the arm with
    # whatever the endpoint was doing during that window.
    schedule = [name for name in arms for _ in range(trials)]
    random.shuffle(schedule)

    for name in schedule:
        nonce = uuid.uuid4().hex
        prefix = build_prefix(nonce)
        # EVERY TRIAL GETS ITS OWN REAL PARENT AND ITS OWN REAL FORK, because a
        # real parent session id is unique per session and that uniqueness is
        # load-bearing here. An earlier revision reused ONE parent id across
        # every trial and the arm separation vanished entirely -- all four arms
        # read 100%, including the wrong-key negative control. Reusing one key
        # across ~32 calls evidently pins the whole run onto warm state that a
        # non-matching key can still reach; whatever the mechanism, it is an
        # artifact of the harness rather than a property of forking, and it
        # models nothing that happens in production. Keep the per-trial fork.
        trial_parent_id = f"parent-{nonce[:12]}"
        trial_parent_dir = cfg / "sessions" / trial_parent_id
        trial_parent_dir.mkdir(parents=True)
        (trial_parent_dir / TRANSCRIPT_NAME).write_text('{"kind":"user","text":"hello"}\n')
        trial_fork_dir = cfg / "sessions" / fork_session(cfg, trial_parent_id)
        # Derived through the real wiring, not assumed equal to the parent id.
        trial_key = stream_fn_for(trial_fork_dir)._cache_lineage_id

        # Call 1 is the PARENT warming its own prefix -- the state a fork
        # branches from.
        first_read, _ = await call(api_key, prefix, trial_parent_id)
        await asyncio.sleep(PAUSE_S)
        second_read, _ = await call(api_key, prefix, arms[name](nonce, trial_key))
        await asyncio.sleep(PAUSE_S)
        results[name].append((first_read, second_read))
        if first_read:
            print(f"  !! {name}: first call was not cold ({first_read}) -- trial discarded")

    print(f"{'arm':12s} {'second-call READ':>18s}  {'rate':>6s}   cached tokens per trial")
    for name in arms:
        pairs = [p for p in results[name] if p[0] == 0]  # only trials that started cold
        hits = sum(1 for _, second in pairs if second > 0)
        rate = (hits / len(pairs) * 100) if pairs else 0.0
        cached = [second for _, second in pairs]
        print(f"{name:12s} {f'{hits}/{len(pairs)}':>18s}  {rate:5.0f}%   {cached}")

    inherited = [s for f, s in results["inherited"] if f == 0]
    others = [s for name in ("own", "none", "wrong") for f, s in results[name] if f == 0]
    ih = sum(1 for c in inherited if c) / len(inherited) * 100 if inherited else 0
    oh = sum(1 for c in others if c) / len(others) * 100 if others else 0
    print(f"\nRESULT: inherited key reads {ih:.0f}% of the time; " f"non-matching keys {oh:.0f}%.")


if __name__ == "__main__":
    asyncio.run(main())
