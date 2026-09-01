"""Real-spawn coverage: launch the REAL adapter wheel through AdapterSupervisor.

Follows ``adapters/test_launch.py`` and ``runner/test_episode_subprocess.py``:
a genuine copied interpreter, the real wheel installed into its site-packages,
a real workspace with the canonical manifest, and a real handshake over the
inherited-fd RPC. This is the only test that proves the SHIPPED artifact
loads and handshakes, not a fixture.

Marked ``slow`` like the existing subprocess suite: each spawn starts a real
CPython and imports pydantic + local_operator before answering.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator.evaluation.adapters.api import AdapterSelector, Handshake
from local_operator.evaluation.adapters.supervisor import AdapterSupervisor
from tests.unit.evaluation.adapters.osworld import fixtures, spawn_helpers

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def adapter_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return spawn_helpers.build_adapter_wheel(tmp_path_factory.mktemp("wheel"))


@pytest.fixture
def spawned(tmp_path: Path, adapter_wheel: Path) -> tuple[AdapterSelector, Path]:
    """A real interpreter + installed wheel + workspace; returns the selector."""

    executable = spawn_helpers.real_interpreter(tmp_path / "venv")
    site = next(executable.parent.parent.glob("lib/python*/site-packages"))

    # The worker imports local_operator and pydantic; the copied venv has
    # neither, so drop a .pth pointing at this interpreter's real roots —
    # exactly the test_launch.py mechanism.
    repo_root = Path(__file__).resolve().parents[5]
    import pydantic

    purelib = __import__("sysconfig").get_paths().get("purelib")
    roots = [str(repo_root), str(Path(pydantic.__file__).resolve().parent.parent)]
    if purelib:
        roots.append(purelib)
    (site / "_local_operator_repo.pth").write_text("\n".join(roots) + "\n")

    package_digest = spawn_helpers.install_adapter_into_site(site, adapter_wheel)
    tasks = {"task_plain": fixtures.PLAIN}
    release_digest = spawn_helpers.release_digest_for(package_digest, tasks)
    workspace = tmp_path / "workspace"
    workspace_digest_value = spawn_helpers.write_workspace(workspace, tasks, release_digest)

    selector = AdapterSelector(
        schema_version="1.0",
        adapter_id="osworld-v2",
        distribution="lop-osworld-v2-adapter",
        version="0.1.0",
        entry_point="lop_osworld_v2_adapter:create",
        package_digest=package_digest,
        release_digest=release_digest,
        python_executable=str(executable.resolve()),
        workspace=str(workspace),
        workspace_digest=workspace_digest_value,
        route_capability="computer",
    )
    return selector, workspace


@pytest.mark.asyncio
async def test_real_wheel_handshakes_and_reaps(
    spawned: tuple[AdapterSelector, Path],
) -> None:
    selector, _workspace = spawned
    supervisor = AdapterSupervisor.launch(selector)
    try:
        handshake = await supervisor.handshake(timeout=60)
        assert isinstance(handshake, Handshake)
        assert handshake.selector == selector
        assert handshake.metadata.adapter_id == "osworld-v2"
        assert handshake.metadata.distribution == "lop-osworld-v2-adapter"
        assert handshake.workspace_digest == selector.workspace_digest
        assert handshake.metadata.capabilities.routes == ("computer",)
        assert handshake.metadata.capabilities.ask_user is True
        assert handshake.metadata.capabilities.scoring is True
    finally:
        await supervisor.terminate()
    assert supervisor.process.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(supervisor.pgid, 0)
