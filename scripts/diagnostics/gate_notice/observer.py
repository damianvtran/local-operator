"""Passive, bounded gate observers for a NEVER-MERGE diagnostic carrier.

No coroutine wrapper, injected await, scheduling call, or product-state write is
allowed here. Instrumentation/policy failure is separate from product failure.
Only structural identities and whitelisted predicate inputs enter the trace;
message bodies, arbitrary locals, credentials and exception text never do.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import inspect
import ipaddress
import json
import os
import socket
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Any

APP = (
    "_note_gate_reply_undelivered_on_app_loop",
    "_retire_gate_reply_notice",
    "_is_current",
    "_gate_card_is_answerable",
    "_sidebar_gate_card_ready",
    "_build_approval_prompt",
    "_prearm_known_gate",
    "_system_notice_block",
    "_reconcile_gate_surface",
    "call_later",
)
ATTACHED = (
    "_maybe_start_gate",
    "_apply_pending_gate",
    "_on_disconnected",
    "_note_gate_reply_undelivered",
    "_go_cold",
)
FIELDS = (
    "_gate_task",
    "_recovery_task",
    "_gate_answered_key",
    "_gates_detached",
    "_background_approval",
    "_keep_gate_reply",
    "_ready_for_events",
    "_disposed",
    "_recovering",
    "_can_go_cold",
    "_deliberate_stop",
    "_runtime_ready",
    "_client",
    "_approval",
    "_ask_screen",
    "_gate_reply_notices",
    "_last_gate_receipt",
    "_interaction",
    "_session_id",
    "_pending_gate",
)
SOURCE_FIELDS = (
    "token",
    "retired",
    "display_only",
    "gate_view_generation",
    "aside_generation",
    "connection_error",
    "connect_attempts",
    "connection_task",
    "session",
    "kind",
    "request_id",
    "epoch",
    "generation",
    "version",
    "_connected",
    "_value",
    "source_binding",
    "_settled",
    "_is_mounted",
    "_reactive_disabled",
    "presentation_revision",
    "draft",
    "approve_all",
)
PROPERTIES = ("is_cold", "recovering", "can_ever_bind", "pending_gate")
OBSERVER = None
POLICY = {"dns_blocked": 0, "connect_blocked": 0, "signal_blocked": 0}
POLICY_ROOT = None


class InstrumentationError(RuntimeError):
    """The observer cannot justify an instrumented result."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data(obj: Any) -> dict[str, Any]:
    try:
        values = vars(obj)
    except TypeError:
        values = {}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Declared data fields include slots without evaluating arbitrary
        # properties merely to take an observer snapshot.
        values = {
            field.name: object.__getattribute__(obj, field.name)
            for field in dataclasses.fields(obj)
            if field.name in FIELDS + SOURCE_FIELDS
        }
    return values


def loopback(host: Any) -> bool:
    if host in ("localhost", b"localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


def signal_guard(original):
    @functools.wraps(original)
    def guarded(target, sig):
        if sig != 0:
            POLICY["signal_blocked"] += 1
            raise InstrumentationError("nonzero process signal prohibited")
        return original(target, sig)

    return guarded


def install_policy(root: Path) -> None:
    """Install before product imports; never probe the guard using a real signal."""
    global POLICY_ROOT
    if POLICY_ROOT is not None:
        if POLICY_ROOT != root.resolve():
            raise InstrumentationError("policy root changed")
        return
    POLICY_ROOT = root.resolve()
    for name in ("kill", "killpg"):
        setattr(os, name, signal_guard(getattr(os, name)))
    dns, connect = socket.getaddrinfo, socket.socket.connect

    def guarded_dns(host, *args, **kwargs):
        if not loopback(host):
            POLICY["dns_blocked"] += 1
            raise InstrumentationError("nonloopback DNS prohibited")
        return dns(host, *args, **kwargs)

    def guarded_connect(sock, address):
        allowed = (sock.family in (socket.AF_INET, socket.AF_INET6) and loopback(address[0])) or (
            sock.family == socket.AF_UNIX
            and isinstance(address, str)
            and Path(address).resolve().is_relative_to(root.resolve())
        )
        if not allowed:
            POLICY["connect_blocked"] += 1
            raise InstrumentationError("nonfixture socket connection prohibited")
        return connect(sock, address)

    socket.getaddrinfo = guarded_dns
    socket.socket.connect = guarded_connect
    connect_ex = socket.socket.connect_ex

    def guarded_connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and loopback(address[0]):
            return connect_ex(sock, address)
        if (
            sock.family == socket.AF_UNIX
            and isinstance(address, str)
            and Path(address).resolve().is_relative_to(root.resolve())
        ):
            return connect_ex(sock, address)
        POLICY["connect_blocked"] += 1
        raise InstrumentationError("nonfixture connect_ex prohibited")

    socket.socket.connect_ex = guarded_connect_ex
    for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        original = getattr(socket, name)

        def guarded_lookup(host, *, _original=original):
            if not loopback(host):
                POLICY["dns_blocked"] += 1
                raise InstrumentationError("nonloopback lookup prohibited")
            return _original(host)

        setattr(socket, name, guarded_lookup)


class Observer:
    def __init__(self, directory: Path, source: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory, self.source = directory, source
        self.pid = os.getpid()
        self.worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
        self.path = directory / f"trace-{self.pid}.ndjson"
        self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        self.count = self.bytes = 0
        self.node = None
        self.failed = []
        self.hooks = {}
        self.local = threading.local()
        self.tasks = weakref.WeakKeyDictionary()
        self.serial = 0
        self.predicates = {}
        self.watched = set()

    def value(self, obj, depth=0):
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            return {"hash": hashlib.sha256(obj.encode()).hexdigest()[:20]}
        if isinstance(obj, asyncio.Task):
            if obj not in self.tasks:
                self.serial += 1
                self.tasks[obj] = self.serial
            return {
                "task": self.tasks[obj],
                "id": id(obj),
                "done": obj.done(),
                "cancelled": obj.cancelled(),
                "coroutine": getattr(obj.get_coro(), "__qualname__", "completed"),
            }
        if depth < 3 and isinstance(obj, (list, tuple)):
            return [self.value(v, depth + 1) for v in obj[:12]]
        out = {"type": type(obj).__name__, "id": id(obj)}
        if depth < 2:
            raw = data(obj)
            for key in SOURCE_FIELDS:
                if key in raw:
                    out[key] = self.value(raw[key], depth + 1)
        return out

    def state(self, actor):
        raw = data(actor)
        return {
            "actor": self.value(actor),
            **{key: self.value(raw[key]) for key in FIELDS if key in raw},
        }

    def stack(self):
        # No inspect.getargvalues/repr/locals: retain provenance, not payloads.
        frames = []
        frame = sys._getframe(2)
        while frame is not None and len(frames) < 8:
            filename = Path(frame.f_code.co_filename)
            try:
                name = str(filename.relative_to(self.source))
            except ValueError:
                name = "external/" + filename.name
            if filename.name != "observer.py":
                frames.append(
                    {"file": name, "line": frame.f_lineno, "function": frame.f_code.co_qualname}
                )
            frame = frame.f_back
        return frames

    def emit(self, event, **values):
        # Exhaustion invalidates the diagnostic, never the product operation.
        if self.failed:
            return
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        record = {
            "event": event,
            "seq": self.count,
            "monotonic_ns": time.monotonic_ns(),
            "pid": self.pid,
            "worker": self.worker,
            "test": self.node,
            "task": self.value(task),
            **values,
        }
        line = (json.dumps(record, sort_keys=True) + "\n").encode()
        if len(line) > 8192 or self.count >= 12000 or self.bytes + len(line) > 16 * 1024**2:
            self.failed.append("trace payload/event/byte limit exceeded")
            return
        try:
            if os.write(self.fd, line) != len(line):
                self.failed.append("partial trace write")
                return
        except OSError:
            self.failed.append("trace write failed")
            return
        self.count += 1
        self.bytes += len(line)

    def wrap_property(self, owner, name, expected):
        descriptor = inspect.getattr_static(owner, name)
        if not isinstance(descriptor, property) or descriptor.fget is None:
            raise InstrumentationError(f"missing property: {name}")
        original = descriptor.fget
        actual = [(p.name, p.kind.name) for p in inspect.signature(original).parameters.values()]
        if actual != [tuple(p) for p in expected]:
            raise InstrumentationError(f"property signature changed: {name}")

        @functools.wraps(original)
        def observed(actor):
            result = original(actor)
            value = self.value(result)
            key = (id(actor), name)
            if (
                not getattr(self.local, "depth", 0)
                and key in self.predicates
                and self.predicates[key] == value
            ):
                return result
            self.predicates[key] = value
            self.emit(
                "predicate",
                hook=name,
                actor=self.value(actor),
                result=self.value(result),
                stack=self.stack(),
            )
            return result

        setattr(
            owner, name, property(observed, descriptor.fset, descriptor.fdel, descriptor.__doc__)
        )
        self.hooks[f"{owner.__name__}.{name}"] = {
            "property": True,
            "signature": str(inspect.signature(original)),
        }

    def wrap(self, owner, name, expected):
        original = getattr(owner, name, None)
        if not callable(original) or inspect.iscoroutinefunction(original):
            raise InstrumentationError(f"missing or asynchronous instrument: {name}")
        signature = inspect.signature(original)
        actual = [(p.name, p.kind.name) for p in signature.parameters.values()]
        if actual != [tuple(p) for p in expected]:
            raise InstrumentationError(f"signature mismatch: {name}")

        @functools.wraps(original)
        def observed(actor, *args, **kwargs):
            caller = sys._getframe(1).f_code.co_qualname
            if name == "_note_gate_reply_undelivered_on_app_loop":
                self.watched.add(id(actor))
            if name in (
                "_gate_card_is_answerable",
                "_sidebar_gate_card_ready",
                "_reconcile_gate_surface",
            ) and not (getattr(self.local, "depth", 0) or id(actor) in self.watched):
                return original(actor, *args, **kwargs)
            depth = getattr(self.local, "depth", 0)
            publication = "_note_gate_reply_undelivered_on_app_loop.<locals>.show" in caller
            if name in ("_is_current", "_system_notice_block") and not (depth or publication):
                return original(actor, *args, **kwargs)
            callback = args[0] if name == "call_later" and args else None
            callback_name = getattr(callback, "__qualname__", "")
            if (
                name == "call_later"
                and "_note_gate_reply_undelivered_on_app_loop" not in callback_name
            ):
                return original(actor, *args, **kwargs)
            bound = signature.bind(actor, *args, **kwargs)
            facts: dict[str, Any] = {
                key: self.value(value)
                for key, value in bound.arguments.items()
                if key
                in ("source", "pending", "identity", "kind", "require_mount", "require_paint")
            }
            if callback_name and inspect.isfunction(callback):
                facts["callback"] = callback_name
                closure = dict(zip(callback.__code__.co_freevars, callback.__closure__ or ()))
                if "source" in closure:
                    facts["callback_source"] = self.value(closure["source"].cell_contents)
            self.emit("enter", hook=name, state=self.state(actor), inputs=facts, stack=self.stack())
            self.local.depth = depth + 1
            try:
                result = original(actor, *args, **kwargs)
            except BaseException as error:
                self.emit(
                    "raise", hook=name, error_type=type(error).__name__, state=self.state(actor)
                )
                raise
            else:
                self.emit("leave", hook=name, result=self.value(result), state=self.state(actor))
                return result
            finally:
                self.local.depth = depth

        setattr(owner, name, observed)
        self.hooks[f"{owner.__name__}.{name}"] = {
            "signature": str(signature),
            "file": inspect.getsourcefile(original),
        }
        return observed

    def finish(self, status):
        os.fsync(self.fd)
        os.close(self.fd)
        summary = {
            "pid": self.pid,
            "worker": self.worker,
            "status": int(status),
            "events": self.count,
            "bytes": self.bytes,
            "hooks": self.hooks,
            "policy": POLICY,
            "instrumentation_errors": self.failed,
            "trace_sha256": digest(self.path),
        }
        (self.directory / f"summary-{self.pid}.json").write_text(json.dumps(summary, indent=2))


def source_probe(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    import importlib

    from local_operator.session.attached import AttachedSession
    from local_operator.session.runtime import launch
    from local_operator.tui.app import OperatorApp

    modules = (
        "local_operator",
        "local_operator.session.attached",
        "local_operator.tui.app",
        "local_operator.session.runtime.server",
        "local_operator.session.runtime.serving",
        "local_operator.tui.session_interaction",
    )
    winners = {}
    for name in modules:
        filename = importlib.import_module(name).__file__
        if filename is None:
            raise InstrumentationError(f"source module has no file: {name}")
        path = Path(filename).resolve()
        if not path.is_relative_to(root.resolve()):
            raise InstrumentationError(f"installed module winner escaped selected checkout: {name}")
        winners[name] = {"file": str(path), "sha256": digest(path)}
    for owner, names in ((OperatorApp, APP), (AttachedSession, ATTACHED + PROPERTIES)):
        for name in names:
            method = getattr(owner, name, None)
            if name in PROPERTIES and isinstance(method, property):
                method = method.fget
            if not callable(method) or inspect.iscoroutinefunction(method):
                raise InstrumentationError(f"invalid instrument: {name}")
            actual = [(p.name, p.kind.name) for p in inspect.signature(method).parameters.values()]
            if actual != [tuple(p) for p in contract[f"{owner.__name__}.{name}"]]:
                raise InstrumentationError(f"instrument signature changed: {name}")

    def forbid_spawn(*args, **kwargs):
        raise InstrumentationError("real runtime spawn prohibited")

    launch._spawn_runtime = forbid_spawn
    return winners


# Pytest imports explicitly selected plugins before conftest/product imports.
# Policy cannot wait for sessionstart: even import-time traffic must be denied.
if os.environ.get("GATE_DIAG_ISO"):
    install_policy(Path(os.environ["GATE_DIAG_ISO"]))


def pytest_sessionstart(session):
    global OBSERVER
    root = Path(os.environ["GATE_DIAG_SOURCE_ROOT"])
    output = Path(os.environ["GATE_DIAG_TRACE_DIR"])
    install_policy(Path(os.environ["GATE_DIAG_ISO"]))
    contract = json.loads(Path(os.environ["GATE_DIAG_CONTRACT"]).read_text())
    winners = source_probe(root, contract)
    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp

    OBSERVER = Observer(output, root)
    OBSERVER.emit("identity", modules=winners, observer_sha256=digest(Path(__file__)))
    for owner, names in ((OperatorApp, APP), (AttachedSession, ATTACHED)):
        for name in names:
            OBSERVER.wrap(owner, name, contract[f"{owner.__name__}.{name}"])
    for name in PROPERTIES:
        OBSERVER.wrap_property(AttachedSession, name, contract[f"AttachedSession.{name}"])


def pytest_runtest_protocol(item, nextitem):
    # A synchronous hook, deliberately not an async wrapper or lifecycle shim.
    if OBSERVER is not None:
        OBSERVER.node = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    if OBSERVER is not None:
        if any(POLICY.values()) or OBSERVER.failed:
            session.exitstatus = 3
        OBSERVER.finish(session.exitstatus)
