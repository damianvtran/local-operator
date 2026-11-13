#!/usr/bin/env python3
"""Send a prompt to the Local Operator AgentCore runtime and stream the reply.

    python3 scripts/invoke.py --profile <aws-profile> [--region <region>] \\
        [--stack-name lop-agentcore | --runtime-arn <arn>] \\
        [--session-id <id>] [--json] "your prompt"

The runtime ARN is read from the ``<stack-name>-runtime`` CloudFormation
outputs unless ``--runtime-arn`` is given. ``--session-id`` selects the
AgentCore runtime session (one microVM, one lop session): reuse it for a
multi-turn conversation; omit it for a fresh session (the id is printed so you
can continue it). AgentCore requires session ids of at least 33 characters.

Requires only boto3. ``--profile`` is required so a run never silently uses
ambient credentials from another account.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any, Iterable, Iterator

# AgentCore rejects runtimeSessionId values shorter than this.
MIN_SESSION_ID_LENGTH = 33


def new_session_id() -> str:
    """A fresh runtime session id (a uuid4 hex is 32 chars; prefix to clear the minimum)."""
    return f"lop-{uuid.uuid4().hex}"


def parse_sse(lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Decode the runtime's ``text/event-stream`` body into event dicts.

    BedrockAgentCoreApp frames each yielded value as one ``data: <json>`` line.
    A non-JSON data line is passed through as text rather than dropped.
    """
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        try:
            event = json.loads(data)
        except ValueError:
            yield {"type": "text", "text": data}
            continue
        # A generator that yields strings is framed as a JSON string.
        yield event if isinstance(event, dict) else {"type": "text", "text": str(event)}


def render(event: dict[str, Any], out: Any = sys.stdout) -> None:
    """Human-readable rendering: text streams inline, tools on their own lines."""
    kind = event.get("type")
    if kind == "text":
        out.write(event.get("text", ""))
    elif kind == "tool_start":
        out.write(f"\n[tool] {event.get('tool')} {json.dumps(event.get('args'))}\n")
    elif kind == "tool_end":
        status = "error" if event.get("is_error") else "ok"
        result = str(event.get("result", "")).strip()
        if len(result) > 600:
            result = result[:600] + " …"
        out.write(f"[tool {status}] {result}\n")
    elif kind == "session":
        out.write(f"[session {event.get('session_id')} model={event.get('model')}]\n")
    elif kind == "end":
        if event.get("error"):
            out.write(f"\n[turn error] {event['error']}\n")
        else:
            out.write("\n")
    elif kind == "error":
        out.write(f"\n[error] {event.get('error')}\n")
    out.flush()


def resolve_runtime_arn(session: Any, stack_name: str) -> str:
    """The AgentRuntimeArn output of ``<stack_name>-runtime``, with a clear error."""
    from botocore.exceptions import ClientError

    cfn = session.client("cloudformation")
    stack = f"{stack_name}-runtime"
    try:
        outputs = cfn.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])
    except ClientError as exc:
        raise SystemExit(
            f"error: cannot read stack {stack!r} in {session.region_name}: "
            f"{exc.response['Error']['Message']} (deployed with scripts/deploy.sh?)"
        ) from None
    for output in outputs:
        if output["OutputKey"] == "AgentRuntimeArn":
            return output["OutputValue"]
    raise SystemExit(f"error: stack {stack!r} has no AgentRuntimeArn output (still deploying?)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("prompt")
    parser.add_argument("--profile", required=True, help="AWS CLI profile (required)")
    parser.add_argument("--region", help="AWS region (default: the profile's region)")
    parser.add_argument("--stack-name", default="lop-agentcore")
    parser.add_argument("--runtime-arn", help="skip the stack lookup")
    parser.add_argument("--session-id", help="continue this runtime session")
    parser.add_argument("--model", help="Bedrock model for a NEW session (role must allow it)")
    parser.add_argument("--json", action="store_true", help="print raw events as JSON lines")
    args = parser.parse_args(argv)
    # Validate before touching AWS so a typo never costs a round trip.
    if args.session_id is not None and len(args.session_id) < MIN_SESSION_ID_LENGTH:
        raise SystemExit(f"error: --session-id must be at least {MIN_SESSION_ID_LENGTH} characters")

    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    if not session.region_name:
        raise SystemExit("error: --region is required (the profile has no default region)")
    runtime_arn = args.runtime_arn or resolve_runtime_arn(session, args.stack_name)
    session_id = args.session_id or new_session_id()

    payload: dict[str, Any] = {"prompt": args.prompt}
    if args.model:
        payload["model"] = args.model
    # Agent turns can run for minutes between streamed events (a long tool call).
    client = session.client("bedrock-agentcore", config=Config(read_timeout=900))
    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            runtimeSessionId=session_id,
            payload=json.dumps(payload).encode(),
            contentType="application/json",
            accept="text/event-stream",
        )
    except ClientError as exc:
        error = exc.response["Error"]
        print(f"error: {error.get('Code')}: {error.get('Message')}", file=sys.stderr)
        return 1

    print(f"# session-id: {session_id}", file=sys.stderr)
    body = response["response"]
    content_type = response.get("contentType", "")
    failed = False
    if "text/event-stream" in content_type:
        # chunk_size=1: botocore buffers iter_lines() in 1 KiB reads by default,
        # which would hold small SSE events back until the turn ends.
        events = parse_sse(body.iter_lines(chunk_size=1))
    else:
        events = iter([json.loads(body.read() or b"{}")])
    for event in events:
        if event.get("type") == "error" or (event.get("type") == "end" and event.get("error")):
            failed = True
        if args.json:
            print(json.dumps(event), flush=True)
        else:
            render(event)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
