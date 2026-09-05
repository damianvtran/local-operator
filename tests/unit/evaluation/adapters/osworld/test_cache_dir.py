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

from pathlib import Path

import pytest
from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter, _episode_cache_root
from lop_osworld_v2_adapter.providers.fake import FakeProvider

from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AdapterSelector,
    CloseParams,
    Handshake,
    PrepareParams,
    PythonRuntime,
    ResetStartParams,
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
    # Episode-scoped: upstream's reset_cache_dir only reassigns the attribute
    # and clears nothing, so isolation has to come from the path itself.
    assert root.name == "ep-1"
    # Durability is STRUCTURAL, not a string test: the cache root is a sibling
    # under the run root, so it inherits whatever guarantee the run root has.
    # scripts/run_episode.py puts the run root through refuse_volatile_root,
    # which is what actually keeps this off /tmp — asserting the prefix here
    # would only re-test pytest's tmp_path, which IS under $TMPDIR.
    assert root.parent.parent == artifact_root.parent
    assert root.parent.name == "osworld-cache"

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
        version="0.1.2",
        entry_point="lop_osworld_v2_adapter:create",
        package_digest=metadata.package_digest,
        release_digest=metadata.release_digest,
        python_executable=str(Path(__import__("sys").executable).resolve()),
        workspace=str(workspace),
        workspace_digest=digest,
        route_capability="computer",
    )


@pytest.mark.asyncio
async def test_episode_leaves_workspace_byte_identical(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a real episode the workspace is untouched: nothing newer than the
    selector, and the digest recomputes to the pin the episode started with."""

    workspace = _write_workspace(tmp_path, {"task_plain": fixtures.PLAIN})
    # THE FAILURE GEOMETRY. The supervisor spawns the worker with
    # cwd=selector.workspace, so a cwd-relative upstream write lands in the
    # pinned workspace. Without this chdir the test runs from the repo root
    # and a regression's stray write would land THERE, leaving the digest
    # clean for the wrong reason -- the test would pass under the bug.
    monkeypatch.chdir(workspace)
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
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression: an episode that downloads assets must NOT change the
    workspace digest, so the rescue worker's re-check passes and the sweep
    retires the descriptor instead of wedging ``rescue_required``."""

    workspace = _write_workspace(tmp_path, {"task_plain": fixtures.PLAIN})
    # Same reason as above: reproduce the worker's real cwd, or a cwd-relative
    # regression escapes into the repo root instead of the workspace.
    monkeypatch.chdir(workspace)
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


# ---------------------------------------------------------------------------
# the cwd guarantee: relative writes cannot reach the workspace at all
# ---------------------------------------------------------------------------


class _RelativeWritingFakeProvider(FakeProvider):
    """A fake that writes a HARD-CODED RELATIVE filename during allocate.

    This is the shape of the upstream helpers that ``cache_dir`` does NOT
    cover: ``evaluators/metrics/vscode.py:210`` opens ``"temp.pdf"``,
    ``slides.py:2051`` opens ``temp_extracted_<n>.jpeg``, ``others.py:64-72``
    makes ``<name>.dir``. They call the BUILTIN ``open`` at module scope, so
    no attribute installed on the env object can intercept them — the only
    thing that decides where they land is the process cwd.
    """

    async def allocate(self, plan, task, *, cache_root):  # type: ignore[override]
        await super().allocate(plan, task, cache_root=cache_root)
        # Deliberately the builtin, deliberately relative — exactly what
        # upstream does. Under the fix the cwd is the episode scratch dir,
        # so this cannot reach the pinned workspace.
        with open("temp.pdf", "wb") as handle:
            handle.write(b"%PDF-1.4 stray upstream write\n")
        self.stray_write = Path("temp.pdf").resolve()


@pytest.mark.asyncio
async def test_a_relative_upstream_write_cannot_land_in_the_workspace(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE guarantee behind the cwd move: a relative write by ANY upstream
    path — not just the cache downloads ``cache_dir`` covers — lands in the
    episode scratch dir, so the pin survives and rescue's re-check passes.

    This is what the (inert) ``env.open`` seal claimed and did not deliver:
    upstream never resolves ``open`` through the env object, so the guarantee
    has to come from the cwd, not from patching a call site nobody uses.
    """

    workspace = _write_workspace(tmp_path, {"task_plain": fixtures.PLAIN})
    pin = workspace_digest(str(workspace))
    # Reproduce the worker's real starting cwd: the pinned workspace.
    monkeypatch.chdir(workspace)

    provider = _RelativeWritingFakeProvider(scripted_score=1.0)
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

    # The stray relative write really happened...
    assert provider.stray_write.exists(), "the test's own stray write did not occur"
    # ...but it landed in the episode scratch dir, NOT the pinned workspace.
    assert workspace not in provider.stray_write.parents
    assert provider.cache_root is not None
    assert provider.stray_write.parent == provider.cache_root.resolve()
    assert not (workspace / "temp.pdf").exists()

    # Which is the property that matters: the pin is intact, so a rescue
    # worker's digest re-check passes instead of wedging rescue_required.
    assert workspace_digest(str(workspace)) == pin


@pytest.mark.asyncio
async def test_close_restores_the_cwd_the_supervisor_spawned_us_in(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reset_start moves the cwd; close must put it back, so a worker reused
    across episodes starts each one where the supervisor placed it."""

    workspace = _write_workspace(tmp_path, {"task_plain": fixtures.PLAIN})
    monkeypatch.chdir(workspace)
    entry = Path.cwd().resolve()

    provider = FakeProvider(scripted_score=1.0)
    adapter = OSWorldV2Adapter(provider_factory=lambda: provider, workspace_root=workspace)
    artifacts = tmp_path / "run" / "artifacts"
    artifacts.mkdir(parents=True)

    await adapter.prepare(
        PrepareParams(
            operation_id=f"prepare-{episode_id}",
            episode_id=episode_id,
            secret_refs=(),
            infra_values=_INFRA,
        )
    )
    await adapter.reset_start(
        ResetStartParams(
            operation_id=f"reset-{episode_id}",
            task_id="task_plain",
            episode_id=episode_id,
            artifact_root=str(artifacts),
            secrets=(),
        )
    )
    # During the episode the cwd is the scratch dir, off the workspace.
    assert Path.cwd().resolve() != entry
    assert Path.cwd().resolve() == provider.cache_root.resolve()  # type: ignore[union-attr]

    await adapter.close(CloseParams(operation_id=f"close-{episode_id}"))
    assert Path.cwd().resolve() == entry


@pytest.mark.asyncio
async def test_reset_start_observes_the_guest_exactly_once(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observation 0 costs exactly one round-trip.

    On the paid path ``observe()`` is a live HTTP call to the guest (a
    screenshot plus the a11y tree), so a duplicated call doubles reset
    latency and network work on every episode. Pinned because a stray second
    call is invisible to every other assertion — the extra result is simply
    overwritten.
    """

    workspace = _write_workspace(tmp_path, {"task_plain": fixtures.PLAIN})
    monkeypatch.chdir(workspace)
    provider = FakeProvider(scripted_score=1.0)
    adapter = OSWorldV2Adapter(provider_factory=lambda: provider, workspace_root=workspace)
    artifacts = tmp_path / "run" / "artifacts"
    artifacts.mkdir(parents=True)

    await adapter.prepare(
        PrepareParams(
            operation_id=f"prepare-{episode_id}",
            episode_id=episode_id,
            secret_refs=(),
            infra_values=_INFRA,
        )
    )
    await adapter.reset_start(
        ResetStartParams(
            operation_id=f"reset-{episode_id}",
            task_id="task_plain",
            episode_id=episode_id,
            artifact_root=str(artifacts),
            secrets=(),
        )
    )
    assert provider.observe_calls == 1
