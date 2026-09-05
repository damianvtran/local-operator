"""A failed builder still owns every resource the runner has not received."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from local_operator.harness.subagent import _build_child_session
from local_operator.model.configure import create_stream_fn
from local_operator.providers.auth_store import AuthStore
from local_operator.session.retention import LIVE_MARKER_NAME
from local_operator.session.session import Session
from tests.unit.session.test_session import make_session


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["constructor", "async_init"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_failed_child_build_releases_real_transport_and_resources(
    tmp_path, monkeypatch, stage, error_type
):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    auth = AuthStore(tmp_path / "auth.db")
    stream = create_stream_fn(auth, {}, session_id="parent")
    parent = make_session(tmp_path, stream)
    listeners = []
    observed = {}
    failure = error_type("child initialization failed")

    def subscribe(listener):
        listeners.append(listener)
        return lambda: listeners.remove(listener)

    monkeypatch.setattr(
        "local_operator.config_watch.process_watcher",
        lambda _: SimpleNamespace(start=lambda _: None, subscribe=subscribe),
    )

    def fail_constructor(self, **kwargs):
        observed["directory"] = kwargs["transcript"].directory
        assert stream._transport.owners == 2
        raise failure

    original_init = Session.async_init

    async def fail_init(child):
        observed["child"] = child
        observed["directory"] = child._transcript.directory
        await original_init(child)
        # An allocated HTTP client proves cleanup closes the resource, not just
        # a lazy holder which had nothing to release. No network request runs.
        async with child._web_io.client(("cleanup-probe",)) as client:
            observed["web_client"] = client
        assert stream._transport.owners == 2
        assert listeners
        raise failure

    if stage == "constructor":
        monkeypatch.setattr(Session, "__init__", fail_constructor)
    else:
        monkeypatch.setattr(Session, "async_init", fail_init)
    try:
        with pytest.raises(error_type) as raised:
            await _build_child_session(
                label="review",
                prompt="inspect",
                parent_session=parent,
                model_spec=None,
                job_id="failed-child",
            )
        assert raised.value is failure
        assert stream._transport.owners == 1
        assert not stream._http.is_closed
        assert not (observed["directory"] / LIVE_MARKER_NAME).exists()
        assert not listeners
        if stage == "async_init":
            child = observed["child"]
            assert child._disposed
            assert child._tg_stack is None
            assert child._web_io._closed
            assert observed["web_client"].is_closed
        await stream.close()
        assert stream._transport.owners == 0
        assert stream._http.is_closed
    finally:
        await parent.dispose()
        await stream.close()
        await stream._http.aclose()
        auth.close()


@pytest.mark.asyncio
async def test_repeated_build_cancellation_joins_single_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    auth = AuthStore(tmp_path / "auth.db")
    stream = create_stream_fn(auth, {}, session_id="parent")
    parent = make_session(tmp_path, stream)
    init_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    children = []
    dispose_calls = []
    original_dispose = Session.dispose

    async def pending_init(child):
        children.append(child)
        init_started.set()
        await asyncio.Event().wait()

    async def pending_dispose(child):
        if child is not parent:
            dispose_calls.append(child)
            cleanup_started.set()
            await release_cleanup.wait()
        await original_dispose(child)

    monkeypatch.setattr(Session, "async_init", pending_init)
    monkeypatch.setattr(Session, "dispose", pending_dispose)
    build = asyncio.create_task(
        _build_child_session(
            label="review",
            prompt="inspect",
            parent_session=parent,
            model_spec=None,
            job_id="cancelled-child",
        )
    )
    try:
        await asyncio.wait_for(init_started.wait(), 10)
        build.cancel()
        await asyncio.wait_for(cleanup_started.wait(), 10)
        build.cancel()
        # Let the second cancellation reach the shield, without a wall-time bet.
        await asyncio.sleep(0)
        assert not build.done()
        assert dispose_calls == children
        assert stream._transport.owners == 2
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(build, 10)
        assert dispose_calls == children
        assert children[0]._disposed
        assert children[0]._web_io._closed
        assert stream._transport.owners == 1
        assert not (children[0]._transcript.directory / LIVE_MARKER_NAME).exists()
        await stream.close()
        assert stream._transport.owners == 0
        assert stream._http.is_closed
    finally:
        release_cleanup.set()
        if not build.done():
            build.cancel()
        await asyncio.gather(build, return_exceptions=True)
        await parent.dispose()
        await stream.close()
        await stream._http.aclose()
        auth.close()
