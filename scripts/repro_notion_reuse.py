#!/usr/bin/env python3
"""Two-process repro for the RESIDUAL Notion MCP OAuth reuse-detection window.

Sibling of ``repro_notion_401.py``. That script proves the 401-ADOPTION path
(PR #340): a session holding a locally-valid but server-side-revoked ACCESS
token recovers by adopting a peer's token on the 401. This script proves the
remaining hole that adoption does not cover: the SDK's own in-flow REFRESH,
which spends the refresh token the provider loaded ONCE at ``_initialize`` and
never re-reads, under no cross-process lock.

Why that hole is fatal for Notion specifically: Notion rotates its refresh
token on every use AND runs REFRESH-TOKEN REUSE DETECTION. Presenting an
already-rotated refresh token returns HTTP 400 ``{"error":"invalid_grant"}``
and revokes the ENTIRE token family — every one of the ~15 concurrent
local-operator sessions sharing the one grant is logged out at once. So a
sibling that boots with a now-stale in-memory refresh token must NEVER let the
SDK spend it: it has to re-read storage under the lock and spend the freshest
persisted refresh token instead.

The invariant under test: the SDK's ``_refresh_token`` must never run with a
``context.current_tokens.refresh_token`` older than what is in storage.

Scenario (two OS processes, one shared auth.db, one fake reuse-detecting AS):

- A fake authorization server rotates the refresh token on every refresh and
  RETIRES the spent one. Presenting a retired refresh token revokes the whole
  family (sets ``family_revoked`` and returns 400 invalid_grant), exactly like
  Notion. State lives in an flock'd JSON file so two processes agree.
- Process A boots with the original grant, REFRESHES once (rotating
  ``r-old`` -> ``r-new`` and retiring ``r-old``), and persists ``r-new`` to the
  shared store. Its access token is then aged to EXPIRED — modelling the real
  steady state where the whole fleet booted together and has crossed the 8h
  access-token boundary, so even the freshest STORED access token needs a
  refresh (this is what makes coordination proceed to refresh rather than
  simply adopt a still-valid access token).
- Process B boots (SEPARATE process) holding the PRE-rotation refresh token
  ``r-old`` in memory with an EXPIRED access token, and with metadata discovery
  FAILED (``endpoints=None``) — the exact fall-through the fix closes. It makes
  a request, which triggers the in-flow refresh. With the bug, the SDK spends
  the stale in-memory ``r-old`` unlocked -> the AS detects reuse -> the family
  is revoked (every session logged out). With the fix, B refreshes UNDER THE
  LOCK against the SDK's own synthesized fallback endpoint, re-reads storage,
  and spends the fresh ``r-new`` -> the family survives.

Usage:
    .venv/bin/python scripts/repro_notion_reuse.py <scratch_dir>

Prints PASS/FAIL lines and exits non-zero on failure. A pre-fix tree FAILs:
process B spends the retired token and the AS revokes the family.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Ensure the repo is importable regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SERVER_URL = "https://mcp.notion-reuse.test/mcp"

REGISTRY_NAME = "as_reuse_registry.json"
REGISTRY_LOCK_NAME = "as_reuse_registry.lock"


def _registry_paths(scratch: str):
    return os.path.join(scratch, REGISTRY_NAME), os.path.join(scratch, REGISTRY_LOCK_NAME)


def _with_registry(scratch: str, mutate) -> Any:
    """Read-modify-write the fake AS registry under an flock'd sidecar file.

    Cross-process safe the same way production's ``_oauth_refresh_lock`` is: a
    byte-range flock on a sidecar file, never on the data file itself.
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
            # current_refresh: the ONLY refresh token the AS will honour.
            # retired: refresh tokens already spent (presenting one = reuse).
            # family_revoked: set once reuse is detected — the whole grant dies.
            data = {
                "counter": 0,
                "current_refresh": None,
                "retired": [],
                "live_access": [],
                "family_revoked": False,
            }
        result = mutate(data)
        with open(reg_path, "w") as fh:
            json.dump(data, fh)
        return result
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _as_issue_initial(scratch: str) -> dict[str, Any]:
    """The AS issues the ORIGINAL grant (access + refresh)."""

    def mutate(data):
        data["counter"] += 1
        access = f"access-{data['counter']}"
        refresh = f"refresh-{data['counter']}"
        data["current_refresh"] = refresh
        data["live_access"] = [access]
        return {"access_token": access, "refresh_token": refresh}

    return _with_registry(scratch, mutate)


def _as_refresh(scratch: str, presented_refresh: str) -> dict[str, Any]:
    """Rotate the grant, with REUSE DETECTION.

    - Presenting the CURRENT refresh token rotates: the spent token is retired,
      a new access+refresh pair is issued, and all prior access tokens are
      revoked (Notion's behaviour).
    - Presenting a RETIRED refresh token is reuse: the entire family is revoked
      and an ``invalid_grant`` error is signalled. This is the event that logs
      every session out at once, and the thing the fix must prevent B from
      triggering.
    """

    def mutate(data):
        if presented_refresh in data["retired"] or data["family_revoked"]:
            # Reuse detected (or the family is already dead from an earlier
            # reuse): revoke everything and report invalid_grant.
            data["family_revoked"] = True
            data["current_refresh"] = None
            data["live_access"] = []
            return {"error": "invalid_grant", "error_description": "OAuth grant revoked"}
        if presented_refresh != data["current_refresh"]:
            # An unknown token: treat as a generic invalid_grant, not a rotation.
            return {"error": "invalid_grant", "error_description": "unknown refresh token"}
        data["retired"].append(data["current_refresh"])
        data["counter"] += 1
        access = f"access-{data['counter']}"
        refresh = f"refresh-{data['counter']}"
        data["current_refresh"] = refresh
        data["live_access"] = [access]  # rotation revokes prior access tokens
        return {"access_token": access, "refresh_token": refresh}

    return _with_registry(scratch, mutate)


def _as_family_revoked(scratch: str) -> bool:
    reg_path, _ = _registry_paths(scratch)
    try:
        with open(reg_path) as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return bool(data.get("family_revoked"))


def _as_token_live(scratch: str, access_token: str) -> bool:
    reg_path, _ = _registry_paths(scratch)
    try:
        with open(reg_path) as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return access_token in data.get("live_access", [])


def _refresh_lock_is_held(scratch: str) -> bool:
    """True when another holder owns the REAL ``_oauth_refresh_lock`` right now.

    Probes the exact lock file production uses (``config_dir()/
    mcp_oauth_refresh.lock``) with a NON-BLOCKING exclusive flock: if the
    acquire fails, some code path is inside the lock; if it succeeds, we release
    immediately and report "unheld". This is how the repro tells the fix's
    coordinated (locked) refresh apart from the SDK's unlocked one without
    instrumenting the code under test.
    """
    import fcntl

    from local_operator.paths import config_dir

    lock_path = os.path.join(str(config_dir()), "mcp_oauth_refresh.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True  # someone else holds it -> the refresh IS locked
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _inject_concurrent_sibling_rotation(scratch: str, about_to_present: str) -> None:
    """Simulate a peer rotating the grant in the unlocked seam.

    Called only when a refresh POST arrives with the refresh lock UNHELD — i.e.
    the SDK's unlocked in-flow refresh, the residual window. If the token this
    POST is about to present is still the current one, a sibling process rotates
    it away first (retiring it), so this POST then presents a RETIRED token and
    trips reuse detection. That is the real multi-process race: 15 sessions, one
    grant, and any unlocked spend can be overtaken.
    """

    def mutate(data):
        if data["family_revoked"]:
            return None
        if about_to_present == data.get("current_refresh"):
            data["retired"].append(data["current_refresh"])
            data["counter"] += 1
            data["current_refresh"] = f"refresh-{data['counter']}"
            data["live_access"] = [f"access-{data['counter']}"]
        return None

    _with_registry(scratch, mutate)


def _make_fake_as_handler(scratch: str, browser_events: list[str], role: str):
    """Build the single request handler standing in for the reuse-detecting AS.

    It answers every URL the flow can reach: any ``/token`` endpoint (the SDK's
    fallback OR a discovered one), the authorization endpoint (a full browser
    grant, which must never be reached), metadata discovery (forced 404 so the
    provider runs with ``endpoints=None`` — the fall-through the fix closes),
    and the resource server (200 a live token, 401 a revoked one).
    """
    import httpx
    from mcp.shared.auth import OAuthToken

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.rstrip("/").endswith("/token"):
            body = request.content.decode()
            presented = ""
            for part in body.split("&"):
                if part.startswith("refresh_token="):
                    presented = part.split("=", 1)[1]
            # MODEL THE RESIDUAL WINDOW. The bug is not "spend a stale token
            # loaded at boot" — coordination's re-read already adopts the
            # freshest STORED token before falling through. The bug is the SEAM:
            # coordination releases ``_oauth_refresh_lock``, and only THEN does
            # the SDK's unlocked ``_refresh_token`` POST — so one of the other
            # ~15 sessions can rotate the grant in that gap, retiring the token
            # this POST is about to present. We detect which path issued this
            # POST by probing the real refresh lock: if it is UNHELD, the POST is
            # the SDK's unlocked path (the bug), and we inject exactly that
            # concurrent sibling rotation before processing — retiring the
            # presented token so reuse detection fires. If the lock is HELD, the
            # POST is the fix's coordinated path and no sibling can slip in.
            if not _refresh_lock_is_held(scratch):
                _inject_concurrent_sibling_rotation(scratch, presented)
            result = _as_refresh(scratch, presented)
            if "error" in result:
                # Reuse / revoked grant: HTTP 400 invalid_grant, exactly Notion.
                return httpx.Response(400, json=result, request=request)
            token = OAuthToken(
                access_token=result["access_token"],
                refresh_token=result["refresh_token"],
                expires_in=3600,
            )
            return httpx.Response(200, json=token.model_dump(mode="json"), request=request)
        if "authorize" in url:
            browser_events.append(f"{role}: AUTHORIZATION-GRANT request to {url}")
            return httpx.Response(200, json={}, request=request)
        if "oauth-protected-resource" in url or "oauth-authorization-server" in url:
            return httpx.Response(404, request=request)
        auth = request.headers.get("Authorization", "")
        token_val = auth.removeprefix("Bearer ").strip()
        if _as_token_live(scratch, token_val):
            return httpx.Response(200, json={"ok": True}, request=request)
        return httpx.Response(401, request=request)

    return handler


def _install_fake_as_transport(handler) -> None:
    """Route this process's ``httpx.AsyncClient`` through ``handler``.

    The fix's ``_refresh_oauth_token_locked`` opens its OWN ``httpx.AsyncClient``
    to POST the refresh, so that internal client must also hit the fake AS.
    Patching ``AsyncClient`` process-wide with a MockTransport covers it. The
    yielded-request path (the SDK's in-flow refresh and the resource request) is
    driven by hand against the SAME handler in ``_process_b`` — never through a
    real client, which would nest a second auth flow.
    """
    import httpx

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def patched_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault("transport", transport)
        return real_client(*args, **kwargs)

    httpx.AsyncClient = patched_client  # type: ignore[misc, assignment]


def _build_provider(scratch: str):
    """A refresh-coordinating provider over the SHARED auth.db, built with
    ``endpoints=None`` (discovery failed) and NON-INTERACTIVE so a stray full
    grant raises instead of opening a browser."""
    from local_operator.mcp.auth import build_oauth_provider
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(db_path=os.path.join(scratch, "auth.db"))
    cfg = MCPHttpServerConfig(url=SERVER_URL, auth=MCPAuthConfig(type="oauth"))
    return build_oauth_provider(SERVER_URL, cfg, store=store, endpoints=None, interactive=False)


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


async def _process_a(scratch: str) -> None:
    """Process A: REFRESH once (rotating the grant), persist, then age it.

    A rotates ``r-old`` -> ``r-new`` at the AS (retiring ``r-old``) and persists
    ``r-new`` to the shared store, then ages the stored access token to EXPIRED.
    The ageing models the real steady state: the whole fleet booted together, so
    by the time B refreshes even the freshest STORED access token has crossed
    the 8h boundary and needs a refresh — which is precisely what makes B's
    coordination proceed to a refresh rather than adopt a still-valid token."""
    import time

    from mcp.shared.auth import OAuthToken

    from local_operator.mcp.auth import TOKENS_OBTAINED_AT_KEY, McpTokenStorage
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(db_path=os.path.join(scratch, "auth.db"))
    storage = McpTokenStorage(SERVER_URL, store)
    current = await storage.get_tokens()
    assert current is not None and current.refresh_token is not None
    rotated = _as_refresh(scratch, current.refresh_token)
    assert "error" not in rotated, f"A's own refresh should succeed: {rotated}"
    await storage.set_tokens(
        OAuthToken(
            access_token=rotated["access_token"],
            refresh_token=rotated["refresh_token"],
            expires_in=3600,
        )
    )
    # Age the stored access token so B's coordination proceeds to a refresh.
    creds = storage._read() or {}
    creds[TOKENS_OBTAINED_AT_KEY] = time.time() - 100000
    storage._write(creds)
    print(
        f"A: rotated r-old -> {rotated['refresh_token']}, persisted (access aged to expired)",
        flush=True,
    )


async def _process_b(scratch: str) -> int:
    """Process B: hold the PRE-rotation refresh token, refresh, must not reuse.

    Returns 0 on success (family survives, B spent the fresh token), 1 on the
    bug (B spent the retired token and the AS revoked the whole family)."""
    import time

    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    from local_operator.mcp.auth import McpAuthRequiredError, McpTokenStorage
    from local_operator.providers.auth_store import AuthStore

    browser_events: list[str] = []
    handler = _make_fake_as_handler(scratch, browser_events, "B")
    _install_fake_as_transport(handler)

    provider = _build_provider(scratch)

    # Client info is needed for can_refresh_token() to be True.
    store = AuthStore(db_path=os.path.join(scratch, "auth.db"))
    storage = McpTokenStorage(SERVER_URL, store)
    await storage.set_client_info(OAuthClientInformationFull(client_id="cid"))

    # Reconstruct B's multi-process state: B BOOTED before A's rotation, so its
    # in-memory refresh token is the ORIGINAL ``refresh-1`` (now RETIRED at the
    # AS), and its access token is EXPIRED (the fleet crossed the 8h boundary).
    # This is the session that, pre-fix, spends the retired refresh token via
    # the SDK's unlocked in-flow refresh and kills the whole family.
    original_refresh = "refresh-1"
    provider.context.current_tokens = OAuthToken(
        access_token="access-1", refresh_token=original_refresh, expires_in=3600
    )
    provider.context.token_expiry_time = time.time() - 10  # EXPIRED -> refresh fires
    provider.context.client_info = OAuthClientInformationFull(client_id="cid")
    provider._initialized = True

    stored_now = await storage.get_tokens()
    print(
        f"B: booted with in-memory refresh '{original_refresh}' (retired), "
        f"store holds refresh '{stored_now.refresh_token if stored_now else None}'",
        flush=True,
    )

    # Drive the auth flow the way httpx does — but answer each yielded request
    # by calling the fake-AS handler DIRECTLY, never through a real client
    # (which would nest a second auth flow). This is exactly how the SDK's own
    # tests pump an httpx Auth: the provider only YIELDS requests, and the
    # caller supplies the responses.
    import httpx

    gen = provider.async_auth_flow(httpx.Request("POST", SERVER_URL, content=b'{"q":1}'))
    final_status = None
    try:
        request = await gen.__anext__()
        while True:
            response = handler(request)
            try:
                request = await gen.asend(response)
            except StopAsyncIteration:
                final_status = response.status_code
                break
    except McpAuthRequiredError as exc:
        print(f"B: full authorization grant ran (McpAuthRequiredError): {exc}", flush=True)
    finally:
        await gen.aclose()

    if _as_family_revoked(scratch):
        print(
            "B: FAIL — reuse DETECTED: B spent the retired refresh token and the "
            "AS revoked the ENTIRE token family (every session logged out)",
            flush=True,
        )
        return 1
    if browser_events:
        print(f"B: FAIL — browser/authorization grant ran: {browser_events}", flush=True)
        return 1
    if final_status != 200:
        print(f"B: FAIL — final status {final_status}, expected 200", flush=True)
        return 1
    current = provider.context.current_tokens
    spent = current.refresh_token if current else None
    if spent == original_refresh:
        print(f"B: FAIL — still holding the retired token '{spent}'", flush=True)
        return 1
    print(
        f"B: PASS — refreshed under the lock without reusing the retired token; "
        f"now holds '{spent}', family intact, final status 200",
        flush=True,
    )
    return 0


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

    if len(sys.argv) >= 4 and sys.argv[2] == "child":
        # Isolate config so _oauth_refresh_lock and AuthStore resolve under the
        # scratch dir, never the developer's real ~/.local-operator.
        os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = scratch
        return _run_role(scratch, sys.argv[3])

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
        print("ORCH: FAIL — process B triggered reuse detection / did not recover", flush=True)
        return 1
    print("ORCH: PASS — the whole token family survived a stale sibling refresh", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
