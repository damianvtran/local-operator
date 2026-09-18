"""Measure what the classification layer adds to a user message, on the real path.

    ISO=$(mktemp -d)
    env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" \
      PATH="$PATH" TERM=xterm-256color OPENROUTER_API_KEY_DEV=<key> \
      .venv/bin/python scripts/classification_latency_probe.py [--prewarm]

The turn path exactly: ``session_factory._select_knowledge_block`` over a real
skill roster (27 skills — the size the QA measured on the operator's own library)
with the REAL ``ClassificationService``, against the real OpenRouter decision route
when a key is in the environment. It prints, in order:

* the roster build, once per roster;
* the per-message added wall-clock with the layer OFF and ON, the ON arm split into
  a session's first message (cold), warm uncached messages, and cache hits, all
  reported as MEDIANS because this host runs many concurrent sessions and a single
  run is worth little;
* how many answers arrive late (delivered by the following message) rather than
  inside the 50 ms wait.

``--prewarm`` calls ``ClassificationService.warm_up`` before timing anything, which
is what the composition root does at session build. The OFF arm is the control:
subtracting it is the only honest way to report OUR added cost, since the first
message of a session pays the embedder's own cold build either way.

``--clients-only`` answers the other half and must be run as its OWN process: it
builds five ``httpx.AsyncClient`` objects and prints each one, so the first figure is
what a session's first call pays if nobody prewarms. Running it inside the timing run
would warm the process for the arms that follow, which is exactly the mistake that
made an earlier version of these numbers look like the prewarm saved nothing.

WHY A SCRIPT AND NOT A TEST: the vendor leg is a real network call and the numbers
move with machine load, so this is evidence a human reads, not an assertion CI can
carry. The hermetic guarantees (byte-identical block when off, the bounded wait, the
single notice) live in ``tests/unit/test_session_factory_classification.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def plant_skills(root: Path, count: int = 27) -> None:
    """A roster of a realistic SIZE, with descriptions of realistic length."""
    directory_root = root / ".local-operator" / "skills"
    for index in range(count):
        directory = directory_root / f"skill-{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(
            "---\n"
            f"name: skill-{index:03d}\n"
            f"description: Operational playbook {index} for the platform's {index}th procedure, "
            "with its constraints, the vendored tools it needs and the evidence it keeps.\n"
            "---\n\nBody.\n",
            encoding="utf-8",
        )


QUERIES = [
    "which iam role does the operator use in prod",
    "add criminal and legal to the tenant's allowed screening feeds",
    "why can't this tenant run Legal searches",
    "restrict them to sanctions-only monitoring",
    "deploy this MR to qa",
    "roll back prod-2 after a bad release",
    "who owns the equifax billing member number",
    "check the datadog error rate for core",
]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prewarm", action="store_true", help="call warm_up before timing")
    parser.add_argument("--skills", type=int, default=27, help="roster size (default 27)")
    parser.add_argument(
        "--clients-only",
        action="store_true",
        help="print the first five http client constructions and exit (run alone)",
    )
    options = parser.parse_args()

    # Off the repo's own path: the role of this script is to be run from a checkout,
    # and the roster it plants must not land in the operator's library.
    sys.path.insert(0, str(REPO))
    scratch = Path(tempfile.mkdtemp(prefix="classify-probe-"))
    plant_skills(scratch, options.skills)
    os.environ.setdefault("LOCAL_OPERATOR_CONFIG_DIR", str(scratch / ".local-operator"))

    from local_operator import session_factory
    from local_operator.classification.service import ClassificationService
    from local_operator.credentials import CredentialManager
    from local_operator.paths import config_dir
    from local_operator.skills.api import default_skill_roots, discover_skills
    from local_operator.skills.embeddings import LocalEmbedder
    from local_operator.skills.index import SkillIndex

    skills, _warnings = discover_skills(default_skill_roots(scratch))

    def hooks(*, classifier: object) -> session_factory._KnowledgeHooks:
        return session_factory._KnowledgeHooks(
            index=SkillIndex(skills, LocalEmbedder()), classifier=classifier
        )

    async def timed(hooks_obj: session_factory._KnowledgeHooks, query: str, task_id: str) -> float:
        started = time.monotonic()
        await session_factory._select_knowledge_block(hooks_obj, query, task_id=task_id)
        return (time.monotonic() - started) * 1000

    if options.clients_only:
        builds: list[float] = []
        for _ in range(5):
            # A fresh service each time: the client is memoized per service, and the
            # question this answers is what the FIRST construction costs a session.
            fresh = ClassificationService(
                manager=CredentialManager(config_dir()),
                settings={"classification": {"auto": True}},
            )
            started = time.perf_counter()
            fresh.warm_up()
            builds.append((time.perf_counter() - started) * 1000)
            await fresh.aclose()
        print(
            "http client construction (warm_up): "
            + " ".join(f"{value:.1f}" for value in builds)
            + " ms (first, then four more in this process)"
        )
        return 0

    off_hooks = hooks(classifier=None)
    off = [await timed(off_hooks, query, f"off-{i}") for i, query in enumerate(QUERIES)]

    service = ClassificationService(
        manager=CredentialManager(config_dir()),
        settings={"classification": {"auto": True}},
    )
    on_hooks = hooks(classifier=service)
    if options.prewarm:
        started = time.perf_counter()
        service.warm_up()
        print(f"warm_up at session build: {(time.perf_counter() - started) * 1000:.1f} ms")
    print(f"vendor: {service.vendor_name} | waitMs: {on_hooks.classification_wait_s * 1000:.0f}")
    started = time.perf_counter()
    roster = session_factory._classification_roster(on_hooks)
    build_ms = (time.perf_counter() - started) * 1000
    print(f"roster build (once per roster): {build_ms:.2f} ms, {len(roster)} rows")

    miss: list[float] = []
    deferred = 0
    for index_, query in enumerate(QUERIES):
        miss.append(await timed(on_hooks, query, f"miss-{index_}"))
        # The wait is 50 ms and the vendor takes ~250 ms, so the answer for THIS
        # message should be waiting on the next one.
        block = await session_factory._select_knowledge_block(
            on_hooks, f"follow-up {index_}", task_id=f"follow-{index_}"
        )
        deferred += 1 if "- skill://" in block or "- guide://" in block else 0
        await asyncio.sleep(0.35)

    hit = []
    for index_, query in enumerate(QUERIES):
        hit.append(await timed(on_hooks, query, f"hit-{index_}"))
        await asyncio.sleep(0.05)

    def report(label: str, values: list[float]) -> None:
        if not values:
            return
        print(
            f"{label:30} median {statistics.median(values):7.2f} ms   "
            f"worst {max(values):7.2f} ms   n={len(values)}"
        )

    print()
    report("off (layer off)", off)
    report("off, first message (cold)", off[:1])
    report("on, first message (cold)", miss[:1])
    report("on, uncached (rest)", miss[1:])
    report("on, cache hit", hit)
    print()
    print(
        "added, warm uncached: "
        f"median {statistics.median(miss[1:]):.2f} ms worst {max(miss[1:]):.2f} ms"
    )
    print(f"added, session's first message: {miss[0] - off[0]:.2f} ms (on minus off)")
    print(f"late answers delivered by the next message: {deferred}/{len(QUERIES)}")
    await service.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
