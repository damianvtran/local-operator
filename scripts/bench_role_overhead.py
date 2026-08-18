#!/usr/bin/env python3
"""Always-on prompt overhead of the role/delegation surface.

Every token in the system prompt and in an advertised tool's schema is paid on
EVERY request of every session, cached-prefix or not. A feature that adds a tool
therefore has a standing cost that is invisible in review unless someone
measures it, and easy to misreport: two successive attempts to state this PR's
cost were wrong because rows were measured on different bases (schema only vs
the serialized entry a provider actually receives).

So the basis is fixed here, in code, and it is the same for every row:

- a tool costs the tokens of ``{"name", "description", "parameters"}`` as JSON,
  which is what rides in the provider's ``tools`` array;
- the system prompt costs the tokens of its rendered markdown.

Run it in a worktree of the branch and again in one of the base commit, then
diff the two JSON blobs. Reporting a delta measured any other way is how the
number drifts from what the user is billed.

    python scripts/bench_role_overhead.py

``cl100k_base`` is a stand-in for "a typical BPE tokenizer": absolute counts
differ per provider, but the deltas this is used for do not move enough to
change a decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken  # noqa: E402

from local_operator.harness.types import ToolContext  # noqa: E402
from local_operator.tools.registry import create_tools  # noqa: E402

#: Tools whose cost this benchmark tracks. Not every tool: these are the ones
#: the delegation surface owns, so a regression here is attributable.
TRACKED = ("task", "wait", "jobs", "hub", "agent")


class _StubComms:
    """Enough of ``SubagentComms`` for ``hub``'s createIf gate.

    ``is_child`` decides which SHAPE of the tool is built (address peers vs
    message your parent), and the two differ in schema size — so this reports
    the parent shape, which is what a top-level session pays.
    """

    def is_child(self, job_id: str | None) -> bool:
        return False


class _StubJobs:
    """Satisfies ``JobManagerProtocol`` so the job-gated tools are built."""

    def get(self, job_id: str, *, owner_id: str | None = None) -> None:
        return None

    def list(self, *, owner_id: str | None = None) -> list[object]:
        return []

    async def cancel(self, job_id: str, *, owner_id: str | None = None) -> bool:
        return True


def main() -> int:
    encoding = tiktoken.get_encoding("cl100k_base")

    def tokens(text: str) -> int:
        return len(encoding.encode(text))

    # A FULLY capable context: every createIf gate satisfied, so a tool missing
    # from the output is a tool that does not exist on this commit rather than
    # one this fixture failed to enable.
    context = ToolContext(
        cwd=".",
        subagent_launcher=lambda *args, **kwargs: "job",
        jobs=_StubJobs(),
        agent_registry=object(),
        subagent_comms=_StubComms(),
        has_ui=True,
    )
    built = {tool.name: tool for tool in create_tools(context)}

    report: dict[str, int] = {}
    missing: list[str] = []
    prompt_path = Path(__file__).resolve().parent.parent / "local_operator" / "prompts_md"
    report["system_prompt"] = tokens((prompt_path / "system.md").read_text(encoding="utf-8"))
    for name in TRACKED:
        tool = built.get(name)
        if tool is None:
            # RECORDED, not silently zeroed. A tool absent because this commit
            # does not have it and a tool absent because the fixture failed to
            # satisfy its createIf gate produce the same 0, and diffing two
            # commits is this script's whole purpose: a degraded fixture would
            # quietly report a SAVING where there is a cost. The caller decides
            # which case it is; the script refuses to guess.
            missing.append(name)
            report[name] = 0
            continue
        report[name] = tokens(
            json.dumps(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            )
        )
    report["total"] = sum(report.values())
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing:
        print(
            "WARNING: not built on this commit (a 0 above may mean a broken fixture "
            f"rather than an absent tool): {', '.join(missing)}",
            file=sys.stderr,
        )
        # Non-zero exit so a scripted before/after diff cannot silently compare
        # a full inventory against a degraded one.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
