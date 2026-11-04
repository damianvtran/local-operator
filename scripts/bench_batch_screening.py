"""Screening spend for a busy Team Bridge group, through the REAL connector (PR #3400).

WHY THIS EXISTS. Agent replies are no longer capped by the hub (agent-server
#294), so the recipient's screening budget is what bounds its spend. This
script measures what one recipient's connector spends on one busy group, and
how long the owner's own DM to their agent waits for its screen when it lands
mid-flood.

WHAT IS REAL. Round 1 of #3400 (QA Q-7) found that the first version of this
bench modelled the scheduler and charged every message its true tier itself,
so it reported a reserve the shipped connector did not have. This version
drives the shipped path end to end: ``connector.loop.run_iteration`` (feed,
the batch pre-screen pass, the per-message handler, the held-retry pass, the
coalesced wake), down into ``screening.screen_message_async`` /
``screen_batch_async``, the real ``Store`` budget ledger, holds and receipts
on a real SQLite file. Stand-ins: the hub (the in-memory ``Hub`` of
``tests/unit/team_bridge/test_wake_every_message.py``), the policy publisher
(the test ``Policy``), the vendor (always ``deliver``, in whichever grammar it
was asked), the session wake, and the clock (``time.time``).

BEFORE vs AFTER. Run the SAME script from a checkout of the base commit for
the "before" arm; it adapts to the tree it lives in (a base tree has no batch
seam, so every message is screened alone, one unit each):

    git archive 7f93aad | tar -x -C <dir> && cp scripts/bench_batch_screening.py <dir>/scripts/
    .venv/bin/python <dir>/scripts/bench_batch_screening.py before
    .venv/bin/python scripts/bench_batch_screening.py after

THE SCENARIO (the brief). One human and six agents in one group, posting at
agent speed; the recipient is a seventh agent in the group. Each agent posts
every ``AGENT_GAP_S`` seconds (jittered), the human every ``HUMAN_GAP_S``; at
minute 30 of each hour the human sends ONE DM to the recipient's agent. One
connector iteration per coalescing window (3 s). Deterministic for a seed.
"""

from __future__ import annotations

import asyncio
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from local_operator.team_bridge import screening  # noqa: E402
from local_operator.team_bridge.connector import conversation as conversation_mod  # noqa: E402
from local_operator.team_bridge.connector import loop as loop_mod  # noqa: E402
from local_operator.team_bridge.connector import wake as wake_mod  # noqa: E402
from local_operator.team_bridge.store import Store  # noqa: E402
from tests.unit.team_bridge import test_wake_every_message as rig  # noqa: E402
from tests.unit.team_bridge.test_screening import Policy  # noqa: E402

HOUR_S = 3600.0
WINDOW_S = 3.0  # coalesce.DEFAULT_COALESCE_WINDOW_S
AGENTS = "bcdefg"  # six agent endpoints; the recipient is bep_a
AGENT_GAP_S = 20.0  # each agent posts ~3 times a minute ("agent speed")
HUMAN_GAP_S = 120.0
DM_AT_S = 1800.0
SEED = 3400


@dataclass
class Planned:
    at: float
    sender: str | None  # None = the human
    dm: bool = False


def traffic(seed: int, hours: int) -> list[Planned]:
    rng = random.Random(seed)
    out: list[Planned] = []
    end = hours * HOUR_S
    for agent in AGENTS:
        at = rng.uniform(0, AGENT_GAP_S)
        while at < end:
            out.append(Planned(at, f"bep_{agent}"))
            at += rng.uniform(0.5, 1.5) * AGENT_GAP_S
    at = rng.uniform(0, HUMAN_GAP_S)
    while at < end:
        out.append(Planned(at, None))
        at += rng.uniform(0.5, 1.5) * HUMAN_GAP_S
    for hour in range(hours):
        out.append(Planned(hour * HOUR_S + DM_AT_S, None, dm=True))
    return sorted(out, key=lambda m: m.at)


def _vendor(calls: list[str]) -> Any:
    def runner(prompt: str, *, default_model: str) -> tuple[str, str]:
        calls.append(prompt)
        ids = re.findall(r"^Message (m[0-9]+)$", prompt, re.M)
        if ids:
            lines = [f"ID: {i}\nVERDICT: deliver\nREASON: ordinary." for i in ids]
            return "\n".join(lines), "qa/default-model"
        return "VERDICT: deliver\nREASON: ordinary.", "qa/default-model"

    return runner


def run(hours: int, workdir: Path) -> list[dict[str, Any]]:
    monkeypatch = pytest.MonkeyPatch()
    try:
        clock = rig.Clock()
        start = clock.now
        monkeypatch.setattr(time, "time", clock)
        owner = rig._owner("a", 1)
        config_dir = workdir / "a"
        config_dir.mkdir(parents=True)
        rig._enroll(config_dir, owner, "sess_a")
        hub = rig.Hub(members=("bep_a",), caps=False)
        rig._rig(monkeypatch, hub, {"a": config_dir})
        calls: list[str] = []
        runner = _vendor(calls)
        refused_at: list[float] = []

        def _note(results: Any) -> Any:
            for result in results if isinstance(results, list) else [results]:
                if getattr(result, "hold_code", None) == "budget_exhausted":
                    refused_at.append(clock.now - start)
            return results

        async def one(owner_: Any, item: Any, *, config_dir: Path, **_kw: Any) -> Any:
            return _note(
                await screening.screen_message_async(
                    owner_, item, config_dir=config_dir, policy=Policy(), runner=runner
                )
            )

        monkeypatch.setattr(conversation_mod, "screen_item", one)
        batched = hasattr(screening, "screen_batch_async") and hasattr(wake_mod, "screen_items")
        if batched:

            async def many(owner_: Any, items: Any, *, config_dir: Path, **_kw: Any) -> Any:
                return _note(
                    await screening.screen_batch_async(
                        owner_, items, config_dir=config_dir, policy=Policy(), runner=runner
                    )
                )

            monkeypatch.setattr(wake_mod, "screen_items", many)

        store = Store(config_dir)
        plan = traffic(SEED, hours)
        posted: list[tuple[float, str, str]] = []  # (at, kind, message id)
        dm_wait: dict[str, float] = {}  # dm id -> seconds until its receipt existed
        dm_open: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
        index = 0
        tick = 0.0
        hour_calls_from = 0
        while tick < hours * HOUR_S:
            clock.now = start + tick
            while index < len(plan) and plan[index].at <= tick:
                p = plan[index]
                kind = "dm" if p.dm else ("human" if p.sender is None else "agent")
                body = f"{kind} line {index}: status update"  # unique bodies
                msg = hub.post(p.sender, body, kind="dm" if p.dm else "channel")
                posted.append((tick, kind, msg.id))
                if p.dm:
                    dm_open[msg.id] = tick
                index += 1
            asyncio.run(
                loop_mod.run_iteration(
                    owner,
                    config_dir=config_dir,
                    access=SimpleNamespace(credential_id=owner.credential_id),
                    auth_store=None,
                )
            )
            for dm_id, at in list(dm_open.items()):
                if store.receipt_for_message(owner, dm_id) is not None:
                    dm_wait[dm_id] = tick - at
                    del dm_open[dm_id]
            tick += WINDOW_S
            if tick % HOUR_S < WINDOW_S:
                hour = int(tick // HOUR_S)
                lo, hi = (hour - 1) * HOUR_S, hour * HOUR_S
                hour_calls = calls[hour_calls_from:]
                hour_calls_from = len(calls)
                screened = {
                    m: store.receipt_for_message(owner, m) is not None for _, _, m in posted
                }
                arriving = [p for p in posted if lo <= p[0] < hi]
                dm = next(p for p in arriving if p[1] == "dm")
                humans = [p for p in arriving if p[1] == "human"]
                rows.append(
                    {
                        "hour": hour,
                        "arriving": len(arriving),
                        "batch_calls": sum(1 for c in hour_calls if "Message ids are" in c),
                        "single_calls": sum(1 for c in hour_calls if "Message ids are" not in c),
                        "screened": sum(1 for p in arriving if screened[p[2]]),
                        "refused_min": round(
                            len({int(t // WINDOW_S) for t in refused_at if lo <= t < hi})
                            * WINDOW_S
                            / 60.0,
                            1,
                        ),
                        "dm": (f"{dm_wait[dm[2]]:.0f} s" if dm[2] in dm_wait else "not screened"),
                        "humans": f"{sum(1 for p in humans if screened[p[2]])}/{len(humans)}",
                        "unscreened": sum(1 for v in screened.values() if not v),
                        "batched": batched,
                    }
                )
        return rows
    finally:
        monkeypatch.undo()


def main() -> None:
    import tempfile

    label = sys.argv[1] if len(sys.argv) > 1 else "after"
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    with tempfile.TemporaryDirectory() as tmp:
        rows = run(hours, Path(tmp))
    print(f"<!-- screening module: {screening.__file__} -->")
    for row in rows:
        print(
            f"| {label} | {row['hour']} | {row['arriving']} | "
            f"{row['batch_calls'] + row['single_calls']} "
            f"({row['batch_calls']} batch + {row['single_calls']} single) | "
            f"{row['screened']} | {row['refused_min']} | {row['dm']} | {row['humans']} | "
            f"{row['unscreened']} |"
        )


if __name__ == "__main__":
    main()
