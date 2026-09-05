"""Native Anthropic cache counters across a goal change; synthetic input only.

Three bounded requests per arm: seed, unchanged goal, changed goal. The
production prompt provider opts into append-only state in the candidate. This
measures provider-reported cache reuse, not task quality or general latency.
Native login is resolved from AuthStore without printing/copying credentials.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--arm", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
auth_db = Path.home() / ".local-operator" / "auth.db"
import local_operator  # noqa: E402

assert Path(local_operator.__file__).resolve().is_relative_to(args.repo.resolve())
from local_operator.model.configure import (  # noqa: E402
    build_model_spec,
    create_stream_fn,
)
from local_operator.providers.auth_store import AuthStore  # noqa: E402
from local_operator.session.goal import GoalState  # noqa: E402
from local_operator.session.session import Session  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from local_operator.session_factory import (  # noqa: E402
    _KnowledgeHooks,
    _make_system_blocks_provider,
)

logging.disable(logging.CRITICAL)


async def main():
    with tempfile.TemporaryDirectory(prefix="lo-live-cache-") as tmp:
        os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(Path(tmp) / "config")
        auth = AuthStore(auth_db)
        identity = f"cache-probe-{uuid.uuid4().hex}"
        settings = {
            "retry": {
                "enabled": False,
                "usageAwareFallback": False,
                "usageAwareAccountPick": False,
            },
            "effort": {"auto": False},
        }
        stream = create_stream_fn(auth, settings, session_id=identity)
        spec = build_model_spec("anthropic", "claude-sonnet-4-6").model_copy(
            update={"reasoning_effort": "low", "max_output_tokens": 1024}
        )
        transcript = Transcript(Path(tmp) / "transcript")
        goal = GoalState()
        goal.set("Cache experiment phase A. Respond READY only.")
        provider = _make_system_blocks_provider(
            [], transcript, _KnowledgeHooks(), cwd=tmp, goal_state=goal
        )
        records = []
        real = stream._client_for

        class Capture:
            def __init__(self, client):
                self.client = client

            async def stream(self, request, api_key, oauth_access=None):
                row: dict[str, Any] = {
                    "index": len(records),
                    "model": request.model.model_id,
                    "phase": ("seed", "unchanged_goal", "changed_goal")[len(records)],
                    "credential_scope": hashlib.sha256(
                        str(getattr(oauth_access, "credential_id", api_key)).encode()
                    ).hexdigest()[:16],
                    "append_only_state": bool(getattr(provider, "append_only_state", False)),
                }
                records.append(row)
                start = time.monotonic()
                async for event in self.client.stream(request, api_key, oauth_access):
                    if getattr(event, "usage", None) is not None:
                        u = event.usage
                        row["usage"] = {
                            k: getattr(u, k, 0)
                            for k in (
                                "input_tokens",
                                "cache_read_tokens",
                                "cache_write_tokens",
                                "output_tokens",
                                "context_tokens",
                            )
                        }
                    if getattr(event, "stop_reason", None):
                        row["stop_reason"] = event.stop_reason
                    yield event
                row["wall_s"] = round(time.monotonic() - start, 3)

        stream._client_for = lambda spec: Capture(real(spec))
        session = Session(
            model=spec,
            stream_fn=stream,
            tools=[],
            transcript=transcript,
            session_id=identity,
            cwd=tmp,
            system_blocks_provider=provider,
            goal_state=goal,
        )
        try:
            payload = "\n".join(
                f"record {n}: synthetic value {n * 13} group {n % 17}" for n in range(2400)
            )
            await asyncio.wait_for(
                session.prompt(
                    "Cache fixture "
                    + identity
                    + "\nReply READY only. No tools are needed.\n"
                    + payload
                ),
                120,
            )
            await asyncio.wait_for(session.prompt("Reply READY only."), 120)
            goal.set("Cache experiment phase B. Respond READY only.")
            await asyncio.wait_for(session.prompt("Reply READY only."), 120)
            valid = (
                len(records) == 3
                and len({r["credential_scope"] for r in records}) == 1
                and all(
                    r.get("stop_reason") == "stop" and r["model"] == spec.model_id and "usage" in r
                    for r in records
                )
            )
            report = {
                "arm": args.arm,
                "source": str(args.repo.resolve()),
                "accepted": valid,
                "source_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
                ).strip(),
                "credential_stable": len({r["credential_scope"] for r in records}) == 1,
                "purpose": "Actual Anthropic cache reuse across a goal change; synthetic fixture",
                "catalogue_cache": "Shared native home; not a cold catalogue benchmark",
                "calls": records,
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            if not valid:
                raise SystemExit("Provider run incomplete; no cache claim permitted")
        finally:
            await session.dispose()
            await stream.close()
            auth.close()


asyncio.run(main())
