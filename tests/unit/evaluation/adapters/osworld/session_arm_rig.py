"""Offline whole-episode driver for the session arm (the committed rig).

Runs ``scripts/run_episode.py --engagement session`` in THIS process against a
real spawned adapter -- the shipped wheel installed into a real copied
interpreter, a digest-pinned workspace selecting the FakeProvider -- with the
model replaced by the deterministic wire client below. Everything else is the
real machinery: the arm, the action server over MCP, the bridge, the adapter
lifecycle, scoring and cleanup. This is the session-arm posture of
``--model-client scripted-finish`` on the reply arm, and it is what
``test_session_arm_script.py`` spawns for each of its cases.

The scripted episode turn drives, in order: [``task`` delegation probe ->
``wait``,] an action ``wait`` batch (a step whose rendered frame must come back
through the MCP tool result), then a ``finish`` batch. With ``--child-acts``
the delegated SCOUT CHILD calls the action tool itself -- a finish or a wait
batch -- before answering PONG (the Q-1 probe): a delegated child must not be
able to drive or end the episode, so its call is refused and the run's steps
and terminal stay the parent's.

Runnable by hand for an evidence pass (see the module's own ``--help``); it
writes only under ``--run-root``, ``--log`` and the scratch roots handed in the
environment, and never edits the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

#: Fragment of the arm's prompt header; the episode prompt stays in history, so
#: every call of the episode's own turn matches it (the child's launch prompt
#: does not carry it).
SENTINEL = "Act on this computer by calling"

ACTION_TOOL = "mcp__episode_actions_apply_actions"

_JOB_RE = re.compile(r"job ([0-9a-f]{6,})")


def _tool_names(request: Any) -> list[str]:
    tools = getattr(request, "tools", None) or []
    names: list[str] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if name is None and isinstance(tool, dict):
            name = tool.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


class ScriptedClient:
    """Deterministic wire client for the session arm (see module docstring)."""

    def __init__(
        self, *, log_path: Path, child_acts: str | None, challenge_reply: str = "refinish"
    ) -> None:
        self.log_path = Path(log_path)
        self.calls = 0
        self.child_acts = child_acts
        #: How the script answers the completion challenge that the first
        #: ``done`` claim earns (the session-path gate): ``refinish`` re-sends
        #: the SAME finish declaration (one of the two replies the challenge
        #: names), ``act`` models the task_013 rescue -- one corrective action
        #: first, then the re-declaration.
        self.challenge_reply = challenge_reply

    # -- logging -----------------------------------------------------------

    def _log(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record["ts"] = time.time()
        record["call"] = self.calls
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # -- wire protocol -----------------------------------------------------

    async def stream(
        self, request: Any, api_key: str | None = None, oauth_access: str | None = None
    ):
        self.calls += 1
        messages = list(getattr(request, "messages", None) or [])
        tool_rows = [m for m in messages if getattr(m, "role", None) == "tool"]
        tool_names = _tool_names(request)
        episode = any(
            SENTINEL in (getattr(m, "text", "") or "")
            for m in messages
            if getattr(m, "role", None) == "user"
        )
        self._log(
            {
                "episode": episode,
                "tools": tool_names,
                "action_tool_advertised": ACTION_TOOL in tool_names,
                "last_tool_text": (
                    (getattr(tool_rows[-1], "text", "") or "")[:300] if tool_rows else None
                ),
            }
        )
        events = self._episode_call(tool_rows) if episode else self._child_call(tool_rows)
        for event in events:
            yield event

    def _tool_call(self, index: int, name: str, arguments: dict[str, Any]) -> list[Any]:
        from local_operator.harness.types import (
            StreamEndEvent,
            StreamStartEvent,
            StreamToolCallDelta,
            StreamUsageEvent,
            Usage,
        )

        self._log({"emit": "tool-call", "name": name, "arguments": arguments})
        return [
            StreamStartEvent(response_id=f"session-arm-rig-{index}"),
            StreamToolCallDelta(
                index=0,
                id=f"session-arm-rig-call-{index}",
                name=name,
                argument_delta=json.dumps(arguments),
            ),
            StreamUsageEvent(usage=Usage(input_tokens=10, output_tokens=5)),
            StreamEndEvent(stop_reason="toolUse", usage=Usage(input_tokens=10, output_tokens=5)),
        ]

    def _text_stop(self, text: str) -> list[Any]:
        from local_operator.harness.types import (
            StreamEndEvent,
            StreamStartEvent,
            StreamTextDelta,
            StreamUsageEvent,
            Usage,
        )

        return [
            StreamStartEvent(response_id="session-arm-rig-text"),
            StreamTextDelta(delta=text),
            StreamUsageEvent(usage=Usage(input_tokens=5, output_tokens=2)),
            StreamEndEvent(stop_reason="stop", usage=Usage(input_tokens=5, output_tokens=2)),
        ]

    # -- the two scripts ---------------------------------------------------

    def _episode_call(self, tool_rows: list[Any]) -> list[Any]:
        stage = len(tool_rows)
        if stage == 0:
            if self.child_acts:
                return self._tool_call(
                    0,
                    "task",
                    {
                        "label": "delegation-probe",
                        "prompt": "Reply with exactly: PONG",
                        "agent": "scout",
                    },
                )
            return self._tool_call(
                0, ACTION_TOOL, {"actions": [{"kind": "wait", "duration_ms": 50}]}
            )
        if self.child_acts and stage == 1:
            job_id = None
            for row in reversed(tool_rows):
                match = _JOB_RE.search(getattr(row, "text", "") or "")
                if match:
                    job_id = match.group(1)
                    break
            if job_id is None:
                return self._tool_call(stage, "jobs", {"op": "list"})
            return self._tool_call(stage, "wait", {"job_id": job_id, "wait_ms": 120000})
        action_stage = stage - (2 if self.child_acts else 0)
        if action_stage == 0:
            return self._tool_call(
                stage, ACTION_TOOL, {"actions": [{"kind": "wait", "duration_ms": 50}]}
            )
        if action_stage == 1:
            return self._tool_call(
                stage,
                ACTION_TOOL,
                {
                    "actions": [
                        {
                            "kind": "finish",
                            "status": "done",
                            "reason": "session-arm rig: episode complete",
                        }
                    ]
                },
            )
        # The first `done` claim above earns the completion challenge; these
        # stages answer it. Both are legitimate replies the challenge names.
        if action_stage == 2 and self.challenge_reply == "act":
            # The rescue shape (task_013's): after the challenge, one more
            # action -- the corrective move -- before the re-declaration.
            return self._tool_call(
                stage, ACTION_TOOL, {"actions": [{"kind": "wait", "duration_ms": 50}]}
            )
        if action_stage in (2, 3):
            return self._tool_call(
                stage,
                ACTION_TOOL,
                {
                    "actions": [
                        {
                            "kind": "finish",
                            "status": "done",
                            "reason": "session-arm rig: episode complete, re-checked",
                        }
                    ]
                },
            )
        return self._text_stop("Episode complete.")

    def _child_call(self, tool_rows: list[Any]) -> list[Any]:
        # Stateless on purpose: a fresh client instance can serve each child
        # request, so the script reads only what the tool rows show -- the
        # refusal the child's own call earned (``Tool not found``), or (on an
        # unfixed tree) a result row proving the call reached the bridge.
        acted = any(ACTION_TOOL in (getattr(row, "text", "") or "") for row in tool_rows)
        refused = any(
            f"Tool not found: {ACTION_TOOL}" in (getattr(row, "text", "") or "")
            for row in tool_rows
        )
        if refused:
            self._log({"emit": "child-refusal-seen"})
            return self._text_stop("PONG")
        if self.child_acts and not acted:
            if self.child_acts == "wait":
                actions: list[dict[str, Any]] = [{"kind": "wait", "duration_ms": 50}]
            else:
                actions = [
                    {
                        "kind": "finish",
                        "status": "done",
                        "reason": "child-act probe: a delegated child must not end the episode",
                    }
                ]
            return self._tool_call(90, ACTION_TOOL, {"actions": actions})
        return self._text_stop("PONG")


def _bootstrap(worktree: Path) -> None:
    sys.path.insert(0, str(worktree))
    import local_operator

    resolved = Path(local_operator.__file__).resolve()
    if not resolved.is_relative_to(worktree):
        raise SystemExit(f"FATAL: local_operator resolves outside the worktree: {resolved}")
    print(f"[rig] local_operator -> {resolved}", file=sys.stderr, flush=True)


def patch_clients(
    log_path: Path, child_acts: str | None, challenge_reply: str = "refinish"
) -> None:
    import local_operator.providers.clients as clients_mod

    original = clients_mod.client_for_spec

    def patched(spec: Any, **kwargs: Any) -> Any:
        client = original(spec, **kwargs)
        if getattr(spec, "provider", None) == "test":
            return ScriptedClient(
                log_path=log_path, child_acts=child_acts, challenge_reply=challenge_reply
            )
        return client

    clients_mod.client_for_spec = patched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--child-acts", choices=("finish", "wait"), default=None)
    parser.add_argument("--challenge-reply", choices=("refinish", "act"), default="refinish")
    args = parser.parse_args(argv)

    worktree = args.worktree.resolve()
    _bootstrap(worktree)
    patch_clients(args.log, args.child_acts, args.challenge_reply)

    import scripts.run_episode as run_episode

    cli = [
        "--selector",
        str(args.selector),
        "--task-id",
        "task_plain",
        "--route",
        "test/fake/model:free",
        "--run-root",
        str(args.run_root),
        "--secret-env",
        "AWS_ACCESS_KEY_ID",
        "--secret-env",
        "AWS_SECRET_ACCESS_KEY",
        "--max-steps",
        "6",
        "--max-usd",
        "0.01",
        "--infra",
        "AWS_REGION=us-east-1",
        "--infra",
        "AWS_SUBNET_ID=subnet-rig1687",
        "--infra",
        "AWS_SECURITY_GROUP_ID=sg-rig1687",
        "--infra",
        "AWS_SCHEDULER_ROLE_ARN=arn:aws:iam::0:role/rig1687",
        "--infra",
        "OSWORLD_CLIENT_PASSWORD=rigpw",
        "--infra",
        "OSWORLD_FILE_BASE_URL=http://assets.test",
        "--config-dir",
        os.environ["LOCAL_OPERATOR_CONFIG_DIR"],
        "--episode-id",
        args.episode_id,
        "--engagement",
        "session",
        "--session-route",
        "test/mock",
        "--no-store",
    ]
    rc = run_episode.main(cli)
    print(f"[rig] exit code: {rc}", file=sys.stderr, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
