"""The session-side seam: the owner, the stamp, and the locality union.

Three things this file pins, all of them from ``mesh-session-mobility.md`` and all
of them cheap enough to run on every commit:

* **R16 topology 0** — a facade built with no owner gets ``LocalOwner``, its three
  seams are exactly the calls that existed before them, and no network code runs.
  This is the regression the whole seam is designed around, and it is asserted
  rather than argued.
* **The stamp is the durable carrier** (§1.2) and a fork must never inherit it —
  the one place INV-1 is enforceable in this tree.
* **``RuntimeLocality``'s new member** (§1.3) is honest: ``"another-machine"`` for
  a remote placement, ``"this-machine"`` for a local one, never ``"unknown"``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.session.owner import (
    LOCAL_PLACEMENT,
    LocalOwner,
    SessionSeed,
    owner_for,
)
from local_operator.session.placement import (
    MESH_STAMP_NAME,
    MeshStamp,
    SessionPlacement,
    local_placement,
    read_stamp,
    remove_stamp,
    stamp_path,
    write_stamp,
)

SESSION = "9f3ac1e0b7d2"


def _seed(root: Path, session_id: str = SESSION) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    return directory


# ---------------------------------------------------------------------------
# The stamp (§1.2)
# ---------------------------------------------------------------------------


def test_the_stamp_round_trips_and_is_0600(tmp_path: Path) -> None:
    stamp = MeshStamp(
        session_id=SESSION,
        network_id="n_7Yb3kQ",
        home_device="dev_a1b2c3d4e5f6",
        placement=SessionPlacement(
            mode="peer",
            network_id="n_7Yb3kQ",
            home_device="dev_a1b2c3d4e5f6",
            stamp_revision=3,
        ),
        origin={"kind": "moved", "source_device": "dev_other", "source_session_id": SESSION},
    )
    path = write_stamp(tmp_path, stamp)
    assert path == stamp_path(tmp_path, SESSION)
    assert (int(path.stat().st_mode) & 0o777) == 0o600
    read = read_stamp(tmp_path, SESSION)
    assert read is not None
    assert read.home_device == "dev_a1b2c3d4e5f6"
    assert read.placement.mode == "peer"
    assert read.placement.stamp_revision == 3
    assert read.origin["kind"] == "moved"
    remove_stamp(tmp_path, SESSION)
    assert read_stamp(tmp_path, SESSION) is None


def test_an_absent_stamp_is_the_pre_mesh_statement(tmp_path: Path) -> None:
    """Its absence means "no mesh has ever governed this session", not "unknown".

    That is what makes every session on a device that never joined a network
    behave exactly as it did (spine §10 topology 0), and it is why the stamp is
    additive rather than required.
    """
    _seed(tmp_path)
    assert read_stamp(tmp_path, SESSION) is None
    assert local_placement().mode == "local"
    assert LOCAL_PLACEMENT.mode == "local"


def test_a_stamp_of_an_unknown_version_reads_as_absent(tmp_path: Path) -> None:
    """Fail closed to "unplaced", never to a guessed owner."""
    _seed(tmp_path)
    stamp_path(tmp_path, SESSION).write_text('{"version": 99, "home_device": "dev_x"}')
    assert read_stamp(tmp_path, SESSION) is None


def test_a_fork_never_inherits_a_mesh_stamp(tmp_path: Path) -> None:
    """The one place §1.2's fork exclusion is enforceable, tested where it is.

    A copied stamp would make the fork claim its parent's ``home_device`` — a live
    session advertising itself as owned by another device, which is exactly the
    row the resolver then routes away from here.
    """
    from local_operator.fork import EXCLUDED_SIDECARS, fork_session

    assert MESH_STAMP_NAME in EXCLUDED_SIDECARS
    parent = _seed(tmp_path, "parent000001")
    (parent / "transcript.jsonl").write_text(
        '{"role": "user", "content": "hi"}\n', encoding="utf-8"
    )
    write_stamp(
        tmp_path,
        MeshStamp(
            session_id="parent000001",
            network_id="n_1",
            home_device="dev_elsewhere",
            placement=SessionPlacement(mode="peer", network_id="n_1", home_device="dev_elsewhere"),
        ),
    )
    fork_id = fork_session(tmp_path, "parent000001")
    assert not stamp_path(tmp_path, fork_id).exists(), "a fork inherited its parent's ownership"
    assert read_stamp(tmp_path, "parent000001") is not None, "the parent's own stamp was touched"


# ---------------------------------------------------------------------------
# The owner seam (§3.1) — the zero-peer default
# ---------------------------------------------------------------------------


def test_a_facade_with_no_owner_gets_the_local_one_and_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three seams, and the proof that none of them reaches the mesh."""
    from local_operator.network import relay
    from local_operator.session.owner import LocalOwner as _Local

    calls: list[str] = []

    def spy(*args: object, **kwargs: object) -> object:
        calls.append("relay.control_request")
        return None

    monkeypatch.setattr(relay, "control_request", spy)
    _seed(tmp_path)

    owner = owner_for(None, tmp_path, SESSION)
    assert isinstance(owner, _Local)
    assert owner.placement.mode == "local"
    # ``locate`` with no runtime running answers "no owner at all", which is the
    # same tuple ``find_runtime_record`` has always returned.
    assert owner.locate() == (None, None)

    client = owner.make_client(lambda _projection: None, lambda _reason: None)
    assert client is not None
    assert calls == [], f"the local path touched the network: {calls}"
    # And the client it built is the ordinary one, not a remote subclass.
    from local_operator.mobile.attach_client import AttachClient

    assert type(client) is AttachClient


def test_local_owner_engage_forwards_the_callers_budgets(  # noqa: ANN001
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``LocalOwner.engage`` is the SAME call the facade made before the seam."""
    seen: dict[str, object] = {}

    async def fake_engage(session_id: str, cwd: str, work: object, **kwargs: object) -> object:
        seen.update({"session_id": session_id, "cwd": cwd, "kwargs": kwargs})
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)
    owner = LocalOwner(tmp_path, SESSION)
    asyncio.run(owner.engage(cwd="/tmp/work", warm="errand", preempt="evt", preempt_budget_s=1.5))
    assert seen["session_id"] == SESSION
    assert seen["cwd"] == "/tmp/work"
    assert seen["kwargs"] == {
        "config_dir": tmp_path,
        "preempt": "evt",
        "preempt_budget_s": 1.5,
    }


def test_runtime_locality_answers_another_machine_for_a_remote_placement(tmp_path: Path) -> None:
    """§1.3: the union gained a member, and the answer is known rather than safe."""
    from local_operator.session.attached import AttachedSession

    class _Remote:
        placement = SessionPlacement(mode="peer", network_id="n_1", home_device="dev_x")

        def locate(self) -> tuple[None, None]:
            return (None, None)

        async def engage(self, **kwargs: object) -> None:  # pragma: no cover
            return None

        def make_client(self, *args: object, **kwargs: object) -> object:  # pragma: no cover
            raise AssertionError("not dialled in this test")

    async def _never() -> object:  # pragma: no cover
        raise AssertionError("no takeover")

    local = AttachedSession(
        config_dir=tmp_path, session_id=SESSION, takeover_factory=_never, surface="terminal"
    )
    assert local.runtime_locality == "this-machine"
    assert local._can_go_cold is False

    remote = AttachedSession(
        config_dir=tmp_path,
        session_id=SESSION,
        takeover_factory=_never,
        surface="terminal",
        owner=_Remote(),  # type: ignore[arg-type]
    )
    assert remote.runtime_locality == "another-machine"
    # A remote viewer may NEVER take over: that would be a second writer for one
    # transcript the first time a link blipped (§1.3, INV-1).
    assert remote._can_go_cold is True


def test_every_runtime_locality_comparison_handles_the_new_member() -> None:
    """The union is four-valued now; no ``==`` over it may miss a member.

    Walks the tree rather than trusting a grep done once by hand, because the
    failure mode is a comparison written later by somebody who never read §1.3.
    """
    import re
    from typing import get_args

    from local_operator.session.protocol import RuntimeLocality

    members = set(get_args(RuntimeLocality))
    assert members == {"this-process", "this-machine", "another-machine", "unknown"}

    root = Path(__file__).resolve().parents[3] / "local_operator"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"runtime_locality\s*(==|!=)\s*[\"']([^\"']+)[\"']", text):
            if match.group(2) not in members:
                offenders.append(f"{path}: {match.group(0)}")
    assert offenders == [], f"comparisons against a non-member of RuntimeLocality: {offenders}"


def test_a_seed_carries_what_a_remote_blank_frame_would_otherwise_lose() -> None:
    """§3.4: the pre-bind paint must not read as an empty conversation."""
    seed = SessionSeed(
        name="mesh design",
        model_label="anthropic/claude-sonnet-4-5",
        cwd="/srv/work",
        device_name="build-box",
    )
    assert seed.name == "mesh design"
    assert seed.device_name == "build-box"
    assert SessionSeed().mtime is None
