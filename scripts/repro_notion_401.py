#!/usr/bin/env python3
"""Two-process repro for the residual Notion MCP OAuth logout bug.

Proves the 401-adoption fix end to end across PROCESS boundaries, which is
where the bug lives: several long-lived local-operator sessions share ONE
stored grant row, Notion rotates the refresh token on every use, and rotating
REVOKES every previously issued access token. A session holding a locally-
valid but server-side-revoked token must recover by adopting a sibling's
fresh token on the 401 — NOT by running the full browser authorization grant
(the "Notion logged out again" the user saw several times a day).

Scenario (two OS processes, one shared auth.db, one fake Notion-like AS):

- A fake authorization server issues 8h access tokens, rotates the refresh
  token on every refresh, and revokes all prior access tokens when it rotates
  (exactly Notion's behaviour). Revocation state is kept in a small registry
  guarded by an flock'd file, mirroring how production already coordinates
  cross-process with ``fcntl.flock`` in ``_oauth_refresh_lock``.
- Process A boots with the original token, REFRESHES (rotates), and persists
  the fresh token to the shared store.
- Process B boots (in a SEPARATE process) with the token it loaded BEFORE the
  rotation — locally valid (unexpired), but revoked server-side by A's
  rotation. It makes a request, gets a 401, and must succeed by adopting A's
  token from the shared store WITHOUT any browser grant / discovery request.

Usage:
    .venv/bin/python scripts/repro_notion_401.py <scratch_dir>

Prints PASS/FAIL lines and exits non-zero on failure. A pre-fix tree FAILs:
process B runs the full authorization grant instead of adopting the token.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Ensure the repo is importable regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SERVER_URL = "https://mcp.notion.test/mcp"

# The access-token "registry" the fake AS consults. Kept as a JSON file under
# an flock'd lock so two processes agree on which tokens are revoked without
# any extra dependency (aiosqlite is not installed).
REGISTRY_NAME = "as_registry.json"
REGISTRY_LOCK_NAME = "as_registry.lock"


def _registry_paths(scratch: str):
    return os.path.join(scratch, REGISTRY_NAME), os.path.join(scratch, REGISTRY_LOCK_NAME)


def _with_registry(scratch: str, mutate) -> Any:
    """Read-modify-write the fake AS's token registry under an flock'd file.

    Cross-process safe the same way production's ``_oauth_refresh_lock`` is:
    a byte-range flock on a sidecar file, never on the data file itself.
    """
    import fcntl

    reg_path, lock_path = _registry_paths(scratch)
    os.makedirs(scratch, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            with open(reg_path) as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"live": [], "counter": 0}
        result = mutate(data)
        with open(reg_path, "w") as fh:
            json.dump(data, fh)
        return result
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _as_issue_initial(scratch: str) -> dict[str, Any]:
    """The authorization server issues the ORIGINAL grant (access + refresh)."""

    def mutate(data):
        data["counter"] += 1
        token = {
            "access_token": f"access-{data['counter']}",
            "refresh_token": f"refresh-{data['counter']}",
        }
        data["live"] = [token["access_token"]]
        return token

    return _with_registry(scratch, mutate)


def _as_refresh(scratch: str, presented_refresh: str) -> dict[str, Any]:
    """The AS rotates: a refresh REVOKES every prior access token (Notion).

    The presented refresh token must match the CURRENT one — a stale refresh
    (a second process spending the token it loaded at boot) is rejected, which
    is exactly the double-spend Notion punishes."""

    def mutate(data):
        current_refresh = f"refresh-{data['counter']}"
        if presented_refresh != current_refresh:
            raise RuntimeError(
                f"stale refresh token presented: {presented_refresh} != {current_refresh}"
            )
        data["counter"] += 1
        token = {
            "access_token": f"access-{data['counter']}",
            "refresh_token": f"refresh-{data['counter']}",
        }
        # Rotation revokes ALL previously issued access tokens.
        data["live"] = [token["access_token"]]
        return token

    return _with_registry(scratch, mutate)


def _as_token_live(scratch: str, access_token: str) -> bool:
    reg_path, _ = _registry_paths(scratch)
    try:
        with open(reg_path) as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return access_token in data.get("live", [])


def _endpoints():
    from mcp.shared.auth import OAuthMetadata

    from local_operator.mcp.auth import DiscoveredOAuthEndpoints

    return DiscoveredOAuthEndpoints(
        oauth_metadata=OAuthMetadata.model_validate(
            {
                "issuer": SERVER_URL,
                "authorization_endpoint": "https://a/authorize",
                "token_endpoint": "https://a/token",
            }
        )
    )


def _build_provider(scratch: str, role: str, browser_events: list[str]):
    """A refresh-coordinating provider over the SHARED auth.db, plus a fake
    httpx send function standing in for the caller's client.

    The provider is an httpx Auth: it only ever YIELDS requests; the CALLER's
    httpx client sends them. Nothing in the SDK or local-operator sets a
    transport on the provider, so this repro drives the flow exactly the way
    httpx does — sending each yielded request through ``fake_send`` and feeding
    the response back in. The provider is built NON-INTERACTIVE, so if the 401
    ever reaches the SDK's full authorization branch it raises
    McpAuthRequiredError instead of opening a browser — which is precisely the
    bug being proven absent for B."""
    import httpx

    from local_operator.mcp.auth import build_oauth_provider
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(db_path=os.path.join(scratch, "auth.db"))
    cfg = MCPHttpServerConfig(url=SERVER_URL, auth=MCPAuthConfig(type="oauth"))
    provider = build_oauth_provider(
        SERVER_URL, cfg, store=store, endpoints=_endpoints(), interactive=False
    )

    async def fake_send(request):
        url = str(request.url)
        # The fake AS's token endpoint: only process A refreshes here.
        if url.rstrip("/").endswith("/token"):
            from mcp.shared.auth import OAuthToken

            # Pull the presented refresh token out of the form body.
            body = request.content.decode()
            presented = ""
            for part in body.split("&"):
                if part.startswith("refresh_token="):
                    presented = part.split("=", 1)[1]
            new = _as_refresh(scratch, presented)
            token = OAuthToken(
                access_token=new["access_token"],
                refresh_token=new["refresh_token"],
                expires_in=3600,
            )
            return httpx.Response(200, json=token.model_dump(mode="json"), request=request)

        # The authorization endpoint: reaching it means the FULL browser grant
        # ran — the bug. Record it so the assertion can name the culprit.
        if "authorize" in url:
            browser_events.append(f"{role}: AUTHORIZATION-GRANT request to {url}")
            return httpx.Response(200, json={}, request=request)

        # The resource server: 401 any REVOKED token, 200 a live one.
        auth = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()
        if _as_token_live(scratch, token):
            return httpx.Response(200, json={"ok": True}, request=request)
        return httpx.Response(401, request=request)

    return provider, fake_send


async def _process_a(scratch: str) -> None:
    """Process A: load the original token, then REFRESH (rotating the grant).

    A's token was loaded at boot before the refresh; the refresh exchange is
    emulated by calling the fake AS's rotation directly and persisting the
    result through the SAME storage the provider uses, which is what the real
    in-flow refresh does. The point is the SIDE EFFECT on the shared store and
    on the AS's revocation registry, not which coroutine spent the token."""
    from mcp.shared.auth import OAuthToken

    from local_operator.mcp.auth import McpTokenStorage
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(db_path=os.path.join(scratch, "auth.db"))
    storage = McpTokenStorage(SERVER_URL, store)
    current = await storage.get_tokens()
    assert current is not None and current.refresh_token is not None
    # Rotate at the AS (revokes the original access token) and persist the
    # fresh pair, exactly as a successful refresh would.
    new = _as_refresh(scratch, current.refresh_token)
    await storage.set_tokens(
        OAuthToken(
            access_token=new["access_token"], refresh_token=new["refresh_token"], expires_in=3600
        )
    )
    print(f"A: rotated grant; live access token is now {new['access_token']}", flush=True)


async def _process_b(scratch: str) -> int:
    """Process B: hold the PRE-ROTATION token, request, get 401, must adopt.

    Returns 0 on success (adopted, no browser grant), 1 on the bug (a full
    authorization grant ran or the request failed)."""
    import httpx

    from local_operator.mcp.auth import McpAuthRequiredError, McpTokenStorage
    from local_operator.providers.auth_store import AuthStore

    browser_events: list[str] = []

    provider, send = _build_provider(scratch, "B", browser_events)

    # Reconstruct B's exact multi-process state: B BOOTED before A's rotation,
    # so its in-memory token is the ORIGINAL one — locally valid (unexpired),
    # but revoked server-side by A's rotation. The provider's ``_initialize``
    # would load the CURRENT (rotated) store token, so we pin the in-memory
    # state to the pre-rotation view directly: the original token, an unexpired
    # deadline, and the initialized flag. This is precisely the session that,
    # pre-fix, sent the revoked token, got a 401, and ran a full browser grant.
    import time as _time

    from mcp.shared.auth import OAuthToken

    original_access = "access-1"
    provider.context.current_tokens = OAuthToken(
        access_token=original_access, refresh_token="refresh-1", expires_in=3600
    )
    provider.context.token_expiry_time = _time.time() + 3600  # locally valid
    provider._initialized = True

    # The shared store already holds A's rotated token; note it for the log.
    store = AuthStore(db_path=os.path.join(scratch, "auth.db"))
    storage = McpTokenStorage(SERVER_URL, store)
    stored_now = await storage.get_tokens()
    print(
        f"B: booted holding '{original_access}' (locally valid, revoked by A); "
        f"store holds '{stored_now.access_token if stored_now else None}'",
        flush=True,
    )

    # Drive the auth flow the way httpx does: send each yielded request through
    # the fake client and feed the response back in. With the fix, the adoption
    # retry appears as a SECOND yield of the same request object bearing the
    # adopted token; without it, the 401 drives the SDK's full authorization
    # grant, which (non-interactive) raises McpAuthRequiredError — the bug.
    gen = provider.async_auth_flow(httpx.Request("POST", SERVER_URL, content=b'{"q":1}'))
    try:
        request = await gen.__anext__()
        final_status = None
        while True:
            response = await send(request)
            try:
                request = await gen.asend(response)
            except StopAsyncIteration:
                final_status = response.status_code
                break
    except McpAuthRequiredError as exc:
        print(f"B: FAIL — full authorization grant ran (McpAuthRequiredError): {exc}", flush=True)
        return 1
    finally:
        await gen.aclose()

    if browser_events:
        print(f"B: FAIL — browser/authorization grant ran: {browser_events}", flush=True)
        return 1
    if final_status != 200:
        print(f"B: FAIL — final status {final_status}, expected 200 via adoption", flush=True)
        return 1

    current = provider.context.current_tokens
    adopted = current.access_token if current else None
    if adopted == original_access:
        print(f"B: FAIL — still on the revoked token '{adopted}'", flush=True)
        return 1
    print(
        f"B: PASS — 401 recovered by ADOPTING sibling token '{adopted}', "
        f"no browser grant, final status 200",
        flush=True,
    )
    return 0


def _seed(scratch: str) -> None:
    """Seed the fake AS and the shared store with the ORIGINAL grant."""
    import asyncio

    from mcp.shared.auth import OAuthToken

    from local_operator.mcp.auth import McpTokenStorage
    from local_operator.providers.auth_store import AuthStore

    token = _as_issue_initial(scratch)
    store = AuthStore(db_path=os.path.join(scratch, "auth.db"))
    storage = McpTokenStorage(SERVER_URL, store)
    asyncio.run(
        storage.set_tokens(
            OAuthToken(
                access_token=token["access_token"],
                refresh_token=token["refresh_token"],
                expires_in=3600,
            )
        )
    )
    print(f"seed: issued original grant {token}", flush=True)


def _run_role(scratch: str, role: str) -> int:
    import asyncio

    if role == "A":
        asyncio.run(_process_a(scratch))
        return 0
    return asyncio.run(_process_b(scratch))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    scratch = sys.argv[1]

    # Subprocess entry point: `repro_notion_401.py <scratch> child <A|B>`.
    if len(sys.argv) >= 4 and sys.argv[2] == "child":
        # Isolate the child's config dir so _oauth_refresh_lock and AuthStore
        # resolve under the scratch dir, never the developer's real ~/.local-operator.
        os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = scratch
        return _run_role(scratch, sys.argv[3])

    # Orchestrator: seed, then run A and B as SEPARATE OS processes sharing the
    # one scratch dir (one auth.db, one AS registry) — the multi-process state
    # the bug lives in.
    import subprocess

    os.makedirs(scratch, exist_ok=True)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = scratch
    _seed(scratch)

    env = dict(os.environ, LOCAL_OPERATOR_CONFIG_DIR=scratch)
    a = subprocess.run(
        [sys.executable, os.path.abspath(__file__), scratch, "child", "A"],
        env=env,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(a.stdout)
    sys.stderr.write(a.stderr)
    if a.returncode != 0:
        print("ORCH: FAIL — process A errored", flush=True)
        return 1

    b = subprocess.run(
        [sys.executable, os.path.abspath(__file__), scratch, "child", "B"],
        env=env,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(b.stdout)
    sys.stderr.write(b.stderr)
    if b.returncode != 0:
        print("ORCH: FAIL — process B did not recover via adoption", flush=True)
        return 1

    print("ORCH: PASS — two-process 401 adoption recovered without a browser grant", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
