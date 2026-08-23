"""``read <url>`` sugar and its regression guard on file/spill/skill reads.

The sugar must delegate a URL to the SAME engine web_fetch uses while leaving
every existing ``read`` behaviour (file, ``spill://``, ``skill://``) untouched —
the new branch sits between the spill branch and the internal-URL branch and
must not shadow either.
"""

from __future__ import annotations

import httpx
import pytest

from local_operator.harness.types import ToolContext
from local_operator.tools import spill
from local_operator.tools.builtin import execute_read
from local_operator.web_fetch import service, tool


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv(spill.SPILL_MAX_BYTES_ENV, raising=False)
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])


@pytest.fixture
def context(tmp_path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), session_id="read-sugar")


@pytest.mark.asyncio
async def test_read_url_delegates_to_fetch_engine(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``read https://…`` returns the same shape web_fetch does.

    The engine is stubbed so the test asserts the wiring (the URL reaches
    ``run_fetch`` and its result flows back through ``read``), independent of a
    live network.
    """
    called: dict[str, object] = {}

    async def _fake_run_fetch(url: str, **kwargs):
        called["url"] = url
        called["tool_name"] = kwargs.get("tool_name")
        return ("[200] https://example.com\n\nhello world", {"url": url, "cache": "miss"}, False)

    monkeypatch.setattr(tool, "run_fetch", _fake_run_fetch)
    # execute_read imports run_fetch lazily from the tool module, so patching the
    # attribute there is what the sugar branch resolves.
    result = await execute_read("t1", {"path": "https://example.com"}, None, None, context)
    assert result.is_error is False
    assert "hello world" in result.text
    assert called["url"] == "https://example.com"
    assert called["tool_name"] == "read"


@pytest.mark.asyncio
async def test_read_url_real_engine_end_to_end(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sugar drives the real engine when a transport is available — proven
    by monkeypatching the service default to disable enrichment and using a
    transport-injected run_fetch via the tool module's own path."""

    page = "<html><body><h1>Doc</h1><p>Readable content that is long enough.</p></body></html>"
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, text=page, headers={"content-type": "text/html"})
    )

    orig = tool.run_fetch

    async def _run_with_transport(url: str, **kwargs):
        kwargs.setdefault("transport", transport)
        return await orig(url, **kwargs)

    monkeypatch.setattr(tool, "run_fetch", _run_with_transport)
    monkeypatch.setattr(
        service,
        "DEFAULT_WEB_FETCH_CONFIG",
        {**service.DEFAULT_WEB_FETCH_CONFIG, "enrich": False},
    )
    result = await execute_read("t1", {"path": "https://example.com/doc"}, None, None, context)
    assert result.is_error is False
    assert "Doc" in result.text


@pytest.mark.asyncio
async def test_read_file_still_works(context: ToolContext, tmp_path) -> None:
    """Regression guard: a plain file read is unaffected by the URL branch."""
    target = tmp_path / "note.txt"
    target.write_text("file body line one\nfile body line two\n", encoding="utf-8")
    result = await execute_read("t1", {"path": str(target)}, None, None, context)
    assert result.is_error is False
    assert "file body line one" in result.text


@pytest.mark.asyncio
async def test_read_spill_still_works(context: ToolContext) -> None:
    """Regression guard: ``read spill://…`` still resolves through its own path,
    not the URL branch."""
    meta = spill.get_store().write("x" * 20000, tool_name="bash", session_id="read-sugar")
    assert meta is not None
    result = await execute_read("t1", {"path": meta.handle, "range": "1-1"}, None, None, context)
    assert result.is_error is False


@pytest.mark.asyncio
async def test_read_skill_url_still_resolves(context: ToolContext) -> None:
    """Regression guard: a ``skill://`` URL routes to the internal resolver, not
    the fetch engine."""
    resolved: dict[str, str] = {}

    def _resolver(target: str) -> str:
        resolved["target"] = target
        return "skill body content"

    ctx = ToolContext(
        cwd=context.cwd,
        session_id="read-sugar",
        resolve_internal_url=_resolver,
    )
    result = await execute_read("t1", {"path": "skill://demo"}, None, None, ctx)
    assert result.is_error is False
    assert "skill body content" in result.text
    assert resolved["target"] == "skill://demo"
