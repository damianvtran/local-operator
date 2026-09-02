"""The cache-dir defect: upstream writes must never land in the pinned workspace.

The first paid episode proved the failure end to end: upstream's default
``cache_dir="cache"`` resolves against the worker's cwd, which is the
digest-pinned workspace, so ``_download_setup`` wrote real content into it
and the rescue worker's digest re-check refused, leaving ``rescue_required``
stuck True for an instance that was already terminated.

These tests pin the fix at three levels, cheapest first:

* ``_episode_cache_root`` mints an ABSOLUTE path OUTSIDE the workspace, under
  the episode's owned root, and never under ``/tmp``;
* after a full FakeProvider episode, ``find <workspace> -newer <selector>``
  is empty and the workspace digest recomputes to the pin;
* a rescue worker's digest re-check passes after an episode whose provider
  downloaded assets into the cache path the adapter handed it.

The last two run the REAL adapter and the REAL ``workspace_digest`` — the
same function the rescue worker calls — so they are evidence, not a mock of
the invariant.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter, _episode_cache_root
from lop_osworld_v2_adapter.providers.fake import FakeProvider

from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AdapterSelector,
    Handshake,
    PrepareParams,
    PythonRuntime,
    ScopedInfraValue,
)
from local_operator.evaluation.adapters.discovery import workspace_digest
from local_operator.evaluation.runner.episode import EpisodeRunner
from tests.unit.evaluation.adapters.osworld import fixtures
from tests.unit.evaluation.runner.conftest import (
    ScriptedModel,
    build_config,
    build_spec,
)

# The non-secret infra the adapter's inspect_requirements names.
_INFRA = tuple(
    ScopedInfraValue(name=name, purpose="benchmark_compute", value=f"test-{name}")
    for name in (
        "AWS_REGION",
        "AWS_SUBNET_ID",
        "AWS_SECURITY_GROUP_ID",
        "AWS_SCHEDULER_ROLE_ARN",
        "OSWORLD_CLIENT_PASSWORD",
        "OSWORLD_FILE_BASE_URL",
    )
)


# ---------------------------------------------------------------------------
# the path rule itself
# ---------------------------------------------------------------------------


def test_episode_cache_root_is_absolute_outside_workspace_and_episode_owned(
    tmp_path: Path,
) -> None:
    """The cache root must be absolute, NOT inside the workspace, NOT /tmp,
    and scoped to the episode so one episode's cached upload cannot be read
    by another episode that happens to share a task id."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact_root = tmp_path / "run" / "artifacts"
    artifact_root.mkdir(parents=True)

    root = _episode_cache_root(artifact_root, "ep-1")

    assert root.is_absolute(), "cache root must be absolute, never cwd-relative"
    # Outside the pinned workspace: the whole point of the fix.
    assert workspace not in root.parents and root != workspace
    # Episode-scoped: reset_cache_dir replaces contents wholesale per reset.
    assert root.name == "ep-1"
    # A sibling of the artifact root, under the parent-owned run root.
    assert root.parent.parent == artifact_root.parent
    # Never volatile: macOS purges /private/tmp with no warning.
    assert not str(root).startswith(("/tmp", "/private/tmp", os.environ.get("TMPDIR", "\0")))

    other = _episode_cache_root(artifact_root, "ep-2")
    assert other != root, "two episodes must not share a cache root"


# ---------------------------------------------------------------------------
# workspace stays byte-identical through an episode
# ---------------------------------------------------------------------------


def _write_workspace(tmp_path: Path, tasks: dict[str, str]) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "tasks").mkdir(parents=True, exist_ok=True)
    for task_id, source in tasks.items():
        (workspace / "tasks" / f"{task_id}.py").write_text(source)
    # The marker the ``find -newer`` test compares against: written once,
    # before the episode, exactly like the build script's selector.json.
    (workspace / "selector.json").write_text("{}")
    return workspace


class _Shim:
    """Drive the real in-process adapter through the runner's session shape."""

    def __init__(self, adapter: OSWorldV2Adapter, selector: AdapterSelector) -> None:
        self._adapter = adapter
        self.selector = selector

    async def handshake(self, *, timeout: float = 10.0) -> Handshake:
        del timeout
        return Handshake(
            selector=self.selector,
            metadata=self._adapter.metadata,
            python=PythonRuntime.current(),
            workspace_digest=self.selector.workspace_digest,
            selected_route="computer",
        )

    async def terminate(self) -> None:
        pass

    async def _call_raw(
        self, method: str, params: object, result_type: object, *, timeout: float
    ) -> object:
        del timeout
        return await getattr(self._adapter, method)(params)


def _selector(workspace: Path, adapter: OSWorldV2Adapter, digest: str) -> AdapterSelector:
    metadata = adapter.metadata
    return AdapterSelector(
        schema_version=ADAPTER_SCHEMA_VERSION,
        adapter_id="osworld-v2",
        distribution="lop-osworld-v2-adapter",
        version="0.1.1",
        entry_point="lop_osworld_v2_adapter:create",
        package_digest=metadata.package_digest,
        release_digest=metadata.release_digest,
        python_executable=str(Path(__import__("sys").executable).resolve()),
        workspace=str(workspace),
        workspace_digest=digest,
        route_capability="computer",
    )


@pytest.mark.asyncio
async def test_episode_leaves_workspace_byte_identical(tmp_path: Path, episode_id: str) -> None:
    """After a real episode the workspace is untouched: nothing newer than the
    selector, and the digest recomputes to the pin the episode started with."""

    workspace = _write_workspace(tmp_path, {"task_plain": fixtures.PLAIN})
    pin = workspace_digest(str(workspace))
    provider = FakeProvider(scripted_score=1.0)
    adapter = OSWorldV2Adapter(provider_factory=lambda: provider, workspace_root=workspace)
    shim = _Shim(adapter, _selector(workspace, adapter, pin))

    runner = EpisodeRunner(
        _spec(episode_id),
        build_config(tmp_path / "run"),
        selector=shim.selector,
        model=ScriptedModel(["step", "finish"]),
        launch=lambda _s: shim,
    )
    outcome = await runner.run()
    assert outcome.status == "completed", outcome.diagnostic

    # The cache root the adapter handed the provider is absolute and outside
    # the workspace — the path the fix actually routed.
    assert provider.cache_root is not None
    assert provider.cache_root.is_absolute()
    assert workspace not in provider.cache_root.parents

    # `find <workspace> -newer <selector.json>` is empty: no file was created
    # or modified after the selector was written.
    marker = workspace / "selector.json"
    newer = [p for p in workspace.rglob("*") if p.stat().st_mtime_ns > marker.stat().st_mtime_ns]
    assert newer == [], f"episode wrote into the pinned workspace: {newer}"

    # The digest the rescue worker re-checks still matches the pin.
    assert workspace_digest(str(workspace)) == pin


def _spec(episode_id: str):
    spec = build_spec(episode_id)
    object.__setattr__(spec, "task_id", "task_plain")
    object.__setattr__(spec, "infra_values", _INFRA)
    return spec


# ---------------------------------------------------------------------------
# rescue digest re-check survives an episode that downloaded assets
# ---------------------------------------------------------------------------


class _DownloadingFakeProvider(FakeProvider):
    """A fake that simulates upstream's ``_download_setup``: it writes a real
    asset into the cache path the adapter hands it, exactly as the paid
    episode's calendar/thunderbird download did."""

    async def allocate(self, plan, task, *, cache_root=None):  # type: ignore[override]
        await super().allocate(plan, task, cache_root=cache_root)
        assert cache_root is not None, "the adapter must route the cache root"
        # Mirror upstream: cache_dir/<task_id>/<asset>.
        target = cache_root / task.task_id / "downloaded-asset.ics"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"BEGIN:VCALENDAR\nEND:VCALENDAR\n")
        self.downloaded_to = target


@pytest.mark.asyncio
async def test_rescue_digest_recheck_passes_after_a_download(
    tmp_path: Path, episode_id: str
) -> None:
    """The regression: an episode that downloads assets must NOT change the
    workspace digest, so the rescue worker's re-check passes and the sweep
    retires the descriptor instead of wedging ``rescue_required``."""

    workspace = _write_workspace(tmp_path, {"task_plain": fixtures.PLAIN})
    pin = workspace_digest(str(workspace))
    provider = _DownloadingFakeProvider(scripted_score=1.0)
    adapter = OSWorldV2Adapter(provider_factory=lambda: provider, workspace_root=workspace)
    shim = _Shim(adapter, _selector(workspace, adapter, pin))

    runner = EpisodeRunner(
        _spec(episode_id),
        build_config(tmp_path / "run"),
        selector=shim.selector,
        model=ScriptedModel(["finish"]),
        launch=lambda _s: shim,
    )
    outcome = await runner.run()
    assert outcome.status == "completed", outcome.diagnostic

    # The download really happened, and it went to the episode-owned cache
    # root — not the workspace.
    assert provider.downloaded_to.exists()
    assert workspace not in provider.downloaded_to.parents

    # The rescue worker re-runs workspace_digest(selector.workspace) and
    # compares to the persisted pin. After the fix, that passes.
    assert workspace_digest(str(workspace)) == pin, (
        "workspace digest changed after an episode that downloaded assets; "
        "a rescue worker would refuse with 'content digest differs'"
    )


# ---------------------------------------------------------------------------
# prepare still allocates nothing (guard against the cache mkdir creeping in)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_does_not_create_the_cache_root(tmp_path: Path, episode_id: str) -> None:
    """prepare is declarative (episode.py:289-292). The cache root is minted at
    reset_start, the side-effect boundary — never in prepare."""

    workspace = _write_workspace(tmp_path, {"task_plain": fixtures.PLAIN})
    adapter = OSWorldV2Adapter(provider_factory=FakeProvider, workspace_root=workspace)
    await adapter.prepare(
        PrepareParams(
            operation_id=f"prepare-{episode_id}",
            episode_id=episode_id,
            secret_refs=(),
            infra_values=_INFRA,
        )
    )
    # No cache root was created anywhere under the would-be run root.
    assert not (tmp_path / "run").exists()
