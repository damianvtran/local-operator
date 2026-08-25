"""Loopback-only browser bridge daemon.

The extension is the WebSocket client because an MV3 worker cannot listen.
Local Operator sessions remain stateless HTTP callers; the daemon owns the
single extension connection and bounds every command so a dead worker can
never hang a tool call.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request as HttpRequest
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.protocol import (
    PROTO_VERSION,
    ErrorCode,
    ErrorDetail,
    Hello,
    HelloAck,
    PairRequest,
    PairResult,
    Request,
    Response,
)
from local_operator.paths import config_dir

logger = logging.getLogger(__name__)
DEFAULT_PORT = 4099
PING_INTERVAL_S = 20.0
PAIR_TTL_S = 120.0
PAIR_MAX_ATTEMPTS = 5
PAIRING_FILENAME = "browser/pairing.json"
PENDING_FILENAME = "run/browser/pairing-pending.json"
COMMAND_TIMEOUTS = {
    "open": 30.0,
    "goto": 30.0,
    "click": 25.0,
    "type": 25.0,
    "read": 20.0,
    "snapshot": 20.0,
    "screenshot": 20.0,
    "close": 20.0,
    "status": 20.0,
}


def _private_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _pairing_path(root: Path | None = None) -> Path:
    return (root or config_dir()) / PAIRING_FILENAME


def _pending_path(root: Path | None = None) -> Path:
    return (root or config_dir()) / PENDING_FILENAME


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def pairing_status(root: Path | None = None) -> dict[str, Any]:
    """Return only display-safe pairing metadata; token hashes stay private."""
    saved = _read_json(_pairing_path(root))
    pending = _read_json(_pending_path(root))
    now = time.time()
    return {
        "paired": saved is not None,
        "extension_id": str(saved.get("extension_id", "")) if saved else "",
        "pending_code": (
            str(pending.get("code", ""))
            if pending and float(pending.get("expires_at", 0)) > now
            else ""
        ),
        "pending_expires_at": float(pending.get("expires_at", 0)) if pending else 0.0,
    }


def reset_pairing(root: Path | None = None) -> None:
    for path in (_pairing_path(root), _pending_path(root)):
        with suppress(OSError):
            path.unlink()


class ExtensionLink:
    """The one connected extension plus in-flight request correlation."""

    def __init__(self) -> None:
        self.websocket: WebSocket | None = None
        self.extension_id = ""
        self.browser = ""
        self.paired = False
        self.pending: dict[str, asyncio.Future[Response]] = {}
        self.send_lock = asyncio.Lock()

    async def send(self, payload: dict[str, Any]) -> None:
        websocket = self.websocket
        if websocket is None:
            raise RuntimeError("extension disconnected")
        async with self.send_lock:
            await websocket.send_json(payload)

    def disconnect(self) -> None:
        self.websocket = None
        self.paired = False
        for future in self.pending.values():
            if not future.done():
                future.set_exception(RuntimeError("extension disconnected"))
        self.pending.clear()


class BridgeService:
    def __init__(self, port: int = DEFAULT_PORT, root: Path | None = None) -> None:
        self.port = port
        self.root = root
        self.link = ExtensionLink()
        self.started_at = time.time()
        self.state = state_store.BridgeState(
            pid=os.getpid(),
            port=port,
            session_key=secrets.token_urlsafe(32),
            proto=PROTO_VERSION,
            started_at=self.started_at,
        )
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._ping_task: asyncio.Task[None] | None = None

    def publish(self) -> None:
        self.state.extension_connected = self.link.websocket is not None
        self.state.paired = self.link.paired
        self.state.extension_id = self.link.extension_id
        self.state.browser_name = self.link.browser
        state_store.publish(self.state, self.root)

    async def _heartbeat(self) -> None:
        while True:
            self.publish()
            await asyncio.sleep(state_store.HEARTBEAT_INTERVAL_S)

    async def _ping(self) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL_S)
            if self.link.websocket is not None:
                try:
                    await self.link.send({"event": "ping"})
                except Exception:  # noqa: BLE001 - receive loop owns teardown
                    logger.debug("browser extension ping failed", exc_info=True)

    async def startup(self) -> None:
        self.publish()
        self._heartbeat_task = asyncio.create_task(self._heartbeat())
        self._ping_task = asyncio.create_task(self._ping())

    async def shutdown(self) -> None:
        for task in (self._heartbeat_task, self._ping_task):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        self.link.disconnect()
        state_store.remove(self.root)

    def _origin_extension_id(self, websocket: WebSocket) -> str:
        origin = websocket.headers.get("origin", "")
        prefix = "chrome-extension://"
        if not origin.startswith(prefix):
            return ""
        extension_id = origin[len(prefix) :]
        # Chromium IDs are 32 lowercase a-p characters. A strict check stops a
        # web origin from smuggling slashes or suffixes into the pinned value.
        if len(extension_id) != 32 or any(char not in "abcdefghijklmnop" for char in extension_id):
            return ""
        return extension_id

    def _ensure_pending(self, extension_id: str) -> None:
        pending = _read_json(_pending_path(self.root))
        if (
            pending
            and pending.get("extension_id") == extension_id
            and float(pending.get("expires_at", 0)) > time.time()
            and int(pending.get("attempts", 0)) < PAIR_MAX_ATTEMPTS
        ):
            return
        _private_write(
            _pending_path(self.root),
            {
                "extension_id": extension_id,
                "code": f"{secrets.randbelow(1_000_000):06d}",
                "expires_at": time.time() + PAIR_TTL_S,
                "attempts": 0,
            },
        )

    def _valid_saved_token(self, extension_id: str, token: str) -> bool:
        saved = _read_json(_pairing_path(self.root))
        if not saved or saved.get("extension_id") != extension_id or not token:
            return False
        digest = hashlib.sha256(token.encode()).hexdigest()
        return secrets.compare_digest(str(saved.get("token_sha256", "")), digest)

    async def _try_pair(self, request: PairRequest) -> PairResult:
        pending = _read_json(_pending_path(self.root))
        if not pending or pending.get("extension_id") != self.link.extension_id:
            self._ensure_pending(self.link.extension_id)
            return PairResult(ok=False, message="No live pairing code. Run lop browser pair again.")
        attempts = int(pending.get("attempts", 0)) + 1
        expired = float(pending.get("expires_at", 0)) <= time.time()
        matches = secrets.compare_digest(str(pending.get("code", "")), request.code)
        if expired or attempts > PAIR_MAX_ATTEMPTS or not matches:
            if attempts >= PAIR_MAX_ATTEMPTS or expired:
                self._ensure_pending(self.link.extension_id)
            else:
                pending["attempts"] = attempts
                _private_write(_pending_path(self.root), pending)
            return PairResult(
                ok=False,
                message=(
                    "That code did not match. Codes expire after two minutes; "
                    "check the app for a fresh one."
                ),
            )
        token = secrets.token_urlsafe(32)
        _private_write(
            _pairing_path(self.root),
            {
                "extension_id": self.link.extension_id,
                "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "paired_at": time.time(),
            },
        )
        with suppress(OSError):
            _pending_path(self.root).unlink()
        self.link.paired = True
        self.publish()
        return PairResult(ok=True, token=token)

    async def extension(self, websocket: WebSocket) -> None:
        extension_id = self._origin_extension_id(websocket)
        if not extension_id:
            await websocket.close(code=4004)
            return
        await websocket.accept()
        try:
            raw = await asyncio.wait_for(websocket.receive_json(), timeout=5)
            hello = Hello.model_validate(raw)
        except (asyncio.TimeoutError, ValidationError, ValueError):
            await websocket.close(code=4001)
            return
        if hello.proto != PROTO_VERSION:
            await websocket.close(code=4001)
            return
        saved = _read_json(_pairing_path(self.root))
        if saved and saved.get("extension_id") != extension_id:
            await websocket.close(code=4004)
            return

        # A later extension wins. This prevents two browser profiles from both
        # receiving commands while preserving reconnect after worker death.
        if self.link.websocket is not None:
            with suppress(Exception):
                await self.link.websocket.close(code=4000)
            self.link.disconnect()
        self.link.websocket = websocket
        self.link.extension_id = extension_id
        self.link.browser = hello.browser
        self.link.paired = self._valid_saved_token(extension_id, hello.token)
        if not self.link.paired:
            self._ensure_pending(extension_id)
        self.publish()
        await self.link.send(HelloAck(paired=self.link.paired).model_dump(mode="json"))
        try:
            while True:
                frame = await websocket.receive_json()
                if frame.get("event") == "pair":
                    try:
                        pair = PairRequest.model_validate(frame)
                    except ValidationError:
                        continue
                    result = await self._try_pair(pair)
                    await self.link.send(result.model_dump(mode="json"))
                    continue
                if frame.get("event") in ("pong", "tab_closed", "origin_decision"):
                    continue
                try:
                    response = Response.model_validate(frame)
                except ValidationError:
                    continue
                future = self.link.pending.pop(response.id, None)
                if future is not None and not future.done():
                    future.set_result(response)
        except WebSocketDisconnect:
            pass
        finally:
            if self.link.websocket is websocket:
                self.link.disconnect()
                self.publish()

    async def rpc(self, http_request: HttpRequest) -> JSONResponse:
        supplied = http_request.headers.get("x-bridge-key", "")
        if not secrets.compare_digest(supplied, self.state.session_key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            request = Request.model_validate(await http_request.json())
        except (ValidationError, ValueError) as exc:
            return JSONResponse({"error": "invalid_request", "detail": str(exc)}, status_code=422)
        if request.method == "ping":
            return JSONResponse({"id": request.id, "ok": True, "result": {"pong": True}})
        if self.link.websocket is None:
            return self._error_response(
                request.id, ErrorCode.EXTENSION_DISCONNECTED, "extension not connected"
            )
        if not self.link.paired:
            return self._error_response(request.id, ErrorCode.NOT_PAIRED, "extension is not paired")
        if request.method not in COMMAND_TIMEOUTS:
            return self._error_response(
                request.id, ErrorCode.INTERNAL, f"unknown method: {request.method}"
            )
        if request.id in self.link.pending:
            return self._error_response(request.id, ErrorCode.BUSY, "request id already in flight")
        future: asyncio.Future[Response] = asyncio.get_running_loop().create_future()
        self.link.pending[request.id] = future
        try:
            await self.link.send(request.model_dump(mode="json"))
            response = await asyncio.wait_for(future, timeout=COMMAND_TIMEOUTS[request.method])
            return JSONResponse(response.model_dump(mode="json", exclude_none=True))
        except asyncio.TimeoutError:
            code = (
                ErrorCode.NAV_TIMEOUT if request.method in ("open", "goto") else ErrorCode.INTERNAL
            )
            return self._error_response(
                request.id,
                code,
                f"{request.method} timed out",
                {"timeout_s": COMMAND_TIMEOUTS[request.method]},
            )
        except Exception as exc:  # noqa: BLE001 - transport failure becomes typed wire error
            return self._error_response(request.id, ErrorCode.EXTENSION_DISCONNECTED, str(exc))
        finally:
            self.link.pending.pop(request.id, None)

    @staticmethod
    def _error_response(
        request_id: str,
        code: ErrorCode,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> JSONResponse:
        response = Response(
            id=request_id,
            ok=False,
            error=ErrorDetail(code=code, message=message, data=data or {}),
        )
        return JSONResponse(response.model_dump(mode="json", exclude_none=True))

    async def health(self, _request: HttpRequest) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "proto": PROTO_VERSION,
                "extension_connected": self.link.websocket is not None,
                "paired": self.link.paired,
                "browser": self.link.browser,
            }
        )


def create_app(port: int = DEFAULT_PORT, root: Path | None = None) -> Starlette:
    service = BridgeService(port, root)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        await service.startup()
        try:
            yield
        finally:
            await service.shutdown()

    app = Starlette(
        routes=[
            Route("/health", service.health, methods=["GET"]),
            Route("/rpc", service.rpc, methods=["POST"]),
            WebSocketRoute("/extension", service.extension),
        ],
        lifespan=lifespan,
    )
    app.state.bridge = service
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Operator browser bridge daemon")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    uvicorn.run(create_app(args.port), host="127.0.0.1", port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
