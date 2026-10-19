"""Synthetic, network-free proof of per-dispatch payment provenance.

Run with an isolated HOME/config and ``--root`` pointing to scratch. Uses the
real failover and analytics-recording wrappers, then reads their SQLite rows;
it never consumes an operator credential or changes an existing ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.analytics.recorder import reset_recorder_for_test  # noqa: E402
from local_operator.analytics.store import AnalyticsStore  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamUsageEvent,
    Usage,
)
from local_operator.model.configure import SessionStreamFn  # noqa: E402
from local_operator.model.costs import cost_summary_by_basis  # noqa: E402
from local_operator.providers.auth_store import OAuthAccess  # noqa: E402
from local_operator.providers.failover import stream_with_failover  # noqa: E402
from local_operator.session.channel_spend import (  # noqa: E402
    ChildrenSnapshot,
    InferenceSnapshot,
    combine,
)
from local_operator.session.frontend_state import _aggregate_usage  # noqa: E402
from local_operator.session.spend import SessionSpend  # noqa: E402


class SyntheticAuth:
    kind = "oauth"

    async def get_oauth_access(self, *args, **kwargs):
        return OAuthAccess("synthetic-not-a-credential", 1, kind=self.kind)

    async def get_api_key(self, *args, **kwargs):
        return "synthetic-not-a-credential"


class SyntheticClient:
    async def stream(self, request, key, oauth_access=None):
        # The dollars are synthetic receipts so this probe stays independent
        # of catalogue availability and proves classification, not token rates.
        usage = Usage(input_tokens=100, output_tokens=20, usd_cost=0.79)
        yield StreamUsageEvent(usage=usage)
        yield StreamEndEvent(stop_reason="stop", usage=usage)


async def probe(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "basis-probe.db"
    if path.exists():
        raise RuntimeError("Refusing to overwrite an existing probe ledger")
    store = AnalyticsStore(path)
    recorder = reset_recorder_for_test(store)
    # This fixture wants the actual recorder wrapper, not auth/network setup.
    fn = object.__new__(SessionStreamFn)
    fn._session_id = "synthetic-basis-parent"
    fn._counts_as_child_request = False
    auth = SyntheticAuth()
    request = ChatRequest(model=ModelSpec(provider="anthropic", model_id="claude-test"))

    async def client_for(spec):
        return SyntheticClient()

    usages = []
    for kind in ("oauth", "api_key"):
        auth.kind = kind
        events = stream_with_failover(request, auth, {"retry": {"enabled": False}}, client_for)
        async for event in fn._record_stream(request, events):
            if isinstance(event, StreamUsageEvent):
                usages.append(event.usage)
    recorder.flush_for_test()
    aggregate = _aggregate_usage(usages)
    spend = SessionSpend()
    for usage in aggregate.cost_components:
        spend.accrue(
            790000,
            {"provider": usage.provider, "model_id": usage.model_id},
            basis=usage.billing_basis,
        )
    own = InferenceSnapshot(
        micro=spend.micro,
        calls=spend.calls,
        priced_calls=spend.priced_calls,
        knowledge="exact",
        by_identity=spend.by_identity,
    )
    _, _, child_bases = cost_summary_by_basis(
        [
            Usage(
                provider="anthropic",
                model_id="claude-test",
                usd_cost=11.62,
                billing_basis="subscription_api_equivalent",
            )
        ]
    )
    published = combine(
        [],
        inference=own,
        children=ChildrenSnapshot(total_micro=11620000, by_basis=child_bases),
        tracked=True,
    )
    with sqlite3.connect(path) as conn:
        calls = [
            dict(zip(("provider", "model_id", "billing_basis", "cost_micro"), row))
            for row in conn.execute(
                "SELECT provider, model_id, billing_basis, cost_micro FROM calls ORDER BY id"
            )
        ]
    # Exercise the export path on actual synthetic journal bytes as well as
    # SQLite: a cumulative record and a durable child accounting checkpoint.
    from local_operator.session.spend import SESSION_SPEND_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript
    from scripts.spend_ledger_probe import census

    session_root = root / "sessions" / "synthetic-basis-parent"
    transcript = Transcript(session_root)
    await transcript.append_message(Message(role="assistant", content=[], usage=aggregate))
    await transcript.append_custom(SESSION_SPEND_CUSTOM_TYPE, spend.to_details())
    await transcript.append_custom(
        "subagent_roster",
        {
            "accounting": [
                Usage(
                    provider="anthropic",
                    model_id="claude-test",
                    usd_cost=11.62,
                    billing_basis="subscription_api_equivalent",
                ).model_dump(mode="json")
            ]
        },
    )
    export = census(root / "sessions", path, None)
    store.close()
    return {
        "census_basis": {
            key: export[key] for key in ("inference_by_basis_micro", "children_by_basis_micro")
        },
        "turn_basis": aggregate.billing_basis,
        "component_bases": [u.billing_basis for u in aggregate.cost_components],
        "own_by_basis": combine([], inference=own, tracked=True)["by_basis"],
        "by_basis": published["by_basis"],
        "rows": published["rows"],
        "children": published["children"],
        "total_micro": published["total_micro"],
        "calls": calls,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(probe(args.root)), indent=2, sort_keys=True))
