"""Before/after screening spend for a busy Team Bridge group (PR #3400).

WHY THIS EXISTS. Agent replies are no longer capped by the hub (agent-server
#294), so the recipient's screening budget is what bounds its spend. This
script measures what one recipient's connector spends for one busy group over
an hour, under the OLD unit (one screening call and one budget unit per
message) and the NEW one (one call per coalesced wake batch, one unit per
call, with the ambient reserve), and how long a human's DM to the owner's
agent waits for its screen when it lands mid-flood.

WHAT IS REAL AND WHAT IS SIMULATED. The budget decisions are the real
``Store`` code in a throwaway config dir: every screening call is a real
``reserve_screening_attempt`` (before) or ``reserve_screening_batch`` (after)
against a real SQLite ledger, so the 40/h account ceiling, the 30/h ambient
ceiling and the per-sender 20/h ceiling refuse exactly as they would in the
connector. Time is a simulated clock (``time.time`` patched), the group's
traffic is a seeded generator, and the connector loop is reduced to its
scheduling: one iteration per coalescing window (3 s), the wake batches of
that window, highest tier first (DM, then a person's line, then agent
chatter), oldest first within a tier. No model
is called; the verdict is irrelevant to spend.

THE SCENARIO (the brief). One human and six agents in one group, posting at
agent speed for one hour; the recipient is a seventh agent in the group. Each
agent replies to the room every ``AGENT_GAP_S`` seconds (jittered); the human
posts every ``HUMAN_GAP_S`` seconds; at minute 30 the human sends ONE DM to
the recipient's own agent.

    .venv/bin/python scripts/bench_batch_screening.py

Prints a markdown table. Deterministic for a given seed.
"""

from __future__ import annotations

import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.team_bridge.store import (  # noqa: E402
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    StoreRefused,
)

HOUR_S = 3600.0
WINDOW_S = 3.0  # coalesce.DEFAULT_COALESCE_WINDOW_S
AGENTS = 6
AGENT_GAP_S = 20.0  # each agent posts ~3 times a minute ("agent speed")
HUMAN_GAP_S = 120.0
DM_AT_S = 1800.0
BATCH_MAX = 20  # screening.BATCH_MAX_MESSAGES
RETRY_BATCH_LIMIT = 8  # connector.loop.RETRY_BATCH_LIMIT (due holds per pass)
RETRY_BASE_S = 60.0  # connector.loop.RETRY_BASE_S
RETRY_MAX_S = 3600.0  # connector.loop.RETRY_MAX_S


@dataclass
class Message:
    id: str
    at: float
    sender: str
    human: bool
    dm: bool = False

    @property
    def tier(self) -> int:
        # Store.SCREENING_TIER_*: addressed (a DM), human (a room line), ambient.
        return 2 if self.dm else 1 if self.human else 0


def traffic(seed: int, hours: int) -> list[Message]:
    rng = random.Random(seed)
    out: list[Message] = []
    end = hours * HOUR_S
    for agent in range(AGENTS):
        at = rng.uniform(0, AGENT_GAP_S)
        while at < end:
            out.append(Message(f"a{agent}-{len(out)}", at, f"acc_agent{agent}", False))
            at += rng.uniform(0.5, 1.5) * AGENT_GAP_S
    at = rng.uniform(0, HUMAN_GAP_S)
    while at < end:
        out.append(Message(f"h-{len(out)}", at, "acc_human", True))
        at += rng.uniform(0.5, 1.5) * HUMAN_GAP_S
    for hour in range(hours):
        # ONE DM from the human to the recipient's own agent, mid-flood.
        out.append(Message(f"dm-{hour}", hour * HOUR_S + DM_AT_S, "acc_human", True, dm=True))
    return sorted(out, key=lambda m: m.at)


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_recipient",
        tenant_id="tnt_1",
        endpoint_id="bep_recipient",
        canonical_base_url="https://bench.invalid/v1",
        enrollment_generation=1,
    ).validated()


def run(mode: str, seed: int, hours: int = 2) -> list[dict[str, float | int | str]]:
    messages = traffic(seed, hours)
    clock = {"now": 1_800_000_000.0}
    start = clock["now"]
    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch("local_operator.team_bridge.store.time.time", lambda: clock["now"]),
    ):
        store = Store(Path(tmp))
        owner = _owner()
        store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"))
        held: dict[str, tuple[Message, float, int]] = {}  # id -> (msg, retry_at, attempts)
        screened_at: dict[str, float] = {}
        calls_at: list[float] = []
        exhausted_at: list[float] = []
        backlog_at: list[int] = []  # messages still held at each hour's end
        index = 0
        tick = 0.0

        def charge(chunk: list[Message], now: float) -> bool:
            try:
                if mode == "before":
                    store.reserve_screening_attempt(owner, message_id=chunk[0].id)
                else:
                    store.reserve_screening_batch(
                        owner,
                        [{"message_id": m.id, "sender_account_id": m.sender} for m in chunk],
                        tier=max(m.tier for m in chunk),
                    )
            except StoreRefused:
                return False
            calls_at.append(now)
            for m in chunk:
                screened_at[m.id] = now
                held.pop(m.id, None)
            return True

        def hold(chunk: list[Message], now: float) -> None:
            for m in chunk:
                _, _, attempts = held.get(m.id, (m, 0.0, 0))
                if mode == "before":
                    # The old connector: the loop's doubling backoff (60 s..1 h).
                    due = now + min(RETRY_BASE_S * (2**attempts), RETRY_MAX_S)
                else:
                    # After: the store's refill instant (connector ``_hold_due``).
                    refill = store.screening_budget_refill_at(owner, tier=m.tier)
                    due = min(refill, now + RETRY_MAX_S) if refill else now + RETRY_BASE_S
                held[m.id] = (m, due, attempts + 1)

        def screen(batch: list[Message], now: float) -> bool:
            """Screen ``batch`` (one wake or one retry pass); True if refused."""

            refused = False
            if mode == "before":
                for m in sorted(batch, key=lambda m: m.at):
                    if refused or not charge([m], now):
                        refused = True
                        hold([m], now)
                return refused
            for tier in (2, 1, 0):
                group = sorted((m for m in batch if m.tier == tier), key=lambda m: m.at)
                for offset in range(0, len(group), BATCH_MAX):
                    chunk = group[offset : offset + BATCH_MAX]
                    if not charge(chunk, now):
                        refused = True
                        hold(chunk, now)
            return refused

        while tick < hours * HOUR_S:
            now = start + tick
            clock["now"] = now
            # 1. Due held rows first, oldest first, bounded like the real pass.
            due = sorted(
                (entry for entry in held.values() if entry[1] <= now), key=lambda e: e[0].at
            )[:RETRY_BATCH_LIMIT]
            refused = screen([entry[0] for entry in due], now) if due else False
            # 2. This window's new arrivals: one wake batch.
            arrivals: list[Message] = []
            while index < len(messages) and messages[index].at <= tick:
                arrivals.append(messages[index])
                index += 1
            if arrivals:
                refused = screen(arrivals, now) or refused
            if refused:
                exhausted_at.append(tick)
            tick += WINDOW_S
            if tick % HOUR_S < WINDOW_S:
                backlog_at.append(len(held))
        rows = []
        for hour in range(hours):
            lo, hi = hour * HOUR_S, (hour + 1) * HOUR_S
            in_hour = [m for m in messages if lo <= m.at < hi]
            dms = [m for m in in_hour if m.dm]
            humans = [m for m in in_hour if m.human and not m.dm]
            first = next((t for t in exhausted_at if lo <= t < hi), None)
            rows.append(
                {
                    "mode": mode,
                    "hour": hour + 1,
                    "messages": len(in_hour),
                    "calls": sum(1 for t in calls_at if start + lo <= t < start + hi),
                    "screened": sum(
                        1 for t in screened_at.values() if start + lo <= t < start + hi
                    ),
                    "first_exhausted": (
                        f"{(first - lo) / 60.0:.1f} min" if first is not None else "never"
                    ),
                    "exhausted_min": round(
                        sum(1 for t in exhausted_at if lo <= t < hi) * WINDOW_S / 60.0, 1
                    ),
                    "dm_wait": (
                        f"{screened_at[dms[0].id] - (start + dms[0].at):.0f} s"
                        if dms and dms[0].id in screened_at
                        else ("not screened" if dms else "-")
                    ),
                    "human_screened": (
                        f"{sum(1 for m in humans if m.id in screened_at)}/{len(humans)}"
                    ),
                    "backlog": backlog_at[hour] if hour < len(backlog_at) else len(held),
                }
            )
        return rows


def main() -> None:
    seed = 3386
    print(
        "| unit | hour | msgs arriving | screening calls | msgs screened | budget first "
        "exhausted | minutes exhausted | human DM wait (lands :30) | human room lines "
        "screened | held backlog at end |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for mode, label in (("before", "before: per message"), ("after", "after: per wake batch")):
        for row in run(mode, seed):
            print(
                f"| {label} | {row['hour']} | {row['messages']} | {row['calls']} "
                f"| {row['screened']} | {row['first_exhausted']} | {row['exhausted_min']} "
                f"| {row['dm_wait']} | {row['human_screened']} | {row['backlog']} |"
            )


if __name__ == "__main__":
    main()
