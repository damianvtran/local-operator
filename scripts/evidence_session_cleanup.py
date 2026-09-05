"""Evidence driver for the session-cleanup change: seed a mixed store in an
ISOLATED config dir, run a real boot + /resume + quit cycle through the
production factory and maintenance pass, and count what survives.

Run it against a scratch HOME only — it refuses to run unless
``LOCAL_OPERATOR_CONFIG_DIR`` is set and does not resolve under the real
home::

    env HOME=/tmp/lop-evidence LOCAL_OPERATOR_CONFIG_DIR=/tmp/lop-evidence/.local-operator \\
        .venv/bin/python scripts/evidence_session_cleanup.py [--enable]

Without ``--enable`` the default config is used and the count must be
30 -> 30. With ``--enable`` the config gets ``enabled: true,
remove_empty: true, max_inactive_days: 1`` and the expected removals are
printed beside the actual ones.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _refuse_real_home() -> Path:
    override = os.environ.get("LOCAL_OPERATOR_CONFIG_DIR")
    if not override:
        sys.exit("refusing: set LOCAL_OPERATOR_CONFIG_DIR to a scratch directory")
    config_dir = Path(override).resolve()
    # The REAL home from the passwd database, not $HOME — the whole point of
    # the invocation is that $HOME has been redirected to the scratch dir.
    import pwd

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    real = real_home / ".local-operator"
    if config_dir == real or real in config_dir.parents:
        sys.exit(f"refusing: {config_dir} is the real config dir")
    if "/tmp/" not in str(config_dir) + "/" and "/T/" not in str(config_dir):
        sys.exit(f"refusing: {config_dir} does not look like a scratch dir")
    return config_dir


def seed(config_dir: Path) -> dict[str, str]:
    """30 sessions of mixed shape. Returns ``{name: shape}``."""
    from local_operator.session.cleanup import mark_store
    from local_operator.session.retention import LIVE_MARKER_NAME

    sessions = config_dir / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    mark_store(sessions)
    now = time.time()
    shapes: dict[str, str] = {}
    user_row = json.dumps({"type": "message", "payload": {"role": "user", "content": "hi"}})
    machine_row = json.dumps({"type": "model_route", "payload": {"provider": "test"}})

    def stamp(directory: Path, age_days: float) -> None:
        when = now - age_days * 86400
        for entry in (directory, *directory.rglob("*")):
            os.utime(entry, (when, when))

    def make(name: str, shape: str, age_days: float, files: dict[str, str]) -> None:
        directory = sessions / name
        directory.mkdir(exist_ok=True)
        for filename, body in files.items():
            (directory / filename).write_text(body, encoding="utf-8")
        stamp(directory, age_days)
        shapes[name] = shape

    for index in range(6):
        make(f"empty{index:02d}", "empty dir", 5 + index, {})
    for index in range(5):
        make(f"sidecar{index:02d}", "attachment.json only", 5 + index, {"attachment.json": "{}"})
    for index in range(3):
        make(
            f"machine{index:02d}",
            "machine-only transcript",
            5 + index,
            {"transcript.jsonl": machine_row + "\n"},
        )
    for index in range(10):
        make(
            f"real{index:02d}", "real transcript", 5 + index, {"transcript.jsonl": user_row + "\n"}
        )
    for index in range(3):
        make(
            f"claimed{index:02d}",
            "claimed by THIS pid, empty",
            5 + index,
            {LIVE_MARKER_NAME: str(os.getpid())},
        )
    for index in range(3):
        make(f"wake{index:02d}", "empty with an armed wake", 5 + index, {})
        from local_operator.wakes.store import wakes_dir

        wakes_dir(config_dir).mkdir(parents=True, exist_ok=True)
        (wakes_dir(config_dir) / f"wake{index:02d}.json").write_text("{}", encoding="utf-8")
    assert len(shapes) == 30, len(shapes)
    return shapes


def listing(config_dir: Path) -> set[str]:
    sessions = config_dir / "sessions"
    return {p.name for p in sessions.iterdir() if p.is_dir()}


async def boot_resume_quit(config_dir: Path) -> str:
    """The real factory: create_session (boot), wait for the maintenance
    pass, create_session again with --resume (the /resume path), dispose."""
    import argparse as _argparse

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session_factory import (
        await_store_maintenance_for_tests,
        create_session,
    )

    def args(**overrides: object) -> _argparse.Namespace:
        base: dict[str, object] = dict(
            hosting="test",
            model="test-model",
            agent_name=None,
            agent_id=None,
            yolo=True,
            train=False,
        )
        base.update(overrides)
        return _argparse.Namespace(**base)

    config_manager = ConfigManager(config_dir)
    registry = AgentRegistry(config_dir)
    credentials = CredentialManager(config_dir)

    session = await create_session(
        args(), config_manager, credentials, registry, has_ui=True, defer_mcp_wiring=True
    )
    live_id = session.transcript.directory.name
    await await_store_maintenance_for_tests()
    await session.dispose()

    resumed = await create_session(
        args(resume="real00"),
        config_manager,
        credentials,
        registry,
        has_ui=True,
        defer_mcp_wiring=True,
    )
    await await_store_maintenance_for_tests()
    await resumed.dispose()
    return live_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable", action="store_true")
    parsed = parser.parse_args()
    config_dir = _refuse_real_home()

    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from local_operator.config import ConfigManager

    shapes = seed(config_dir)
    manager = ConfigManager(config_dir)
    manager.update_config({"hosting": "test", "model_name": "test-model"})
    if parsed.enable:
        manager.update_config(
            {
                "session": {
                    "cleanup": {
                        "enabled": True,
                        "remove_empty": True,
                        "max_inactive_days": 1,
                    }
                }
            }
        )
    print(f"config_dir : {config_dir}")
    print(f"cleanup    : {manager.get_nested_value(('session', 'cleanup'))}")
    before = listing(config_dir)
    print(f"before     : {len(before)} session directories")

    live_id = asyncio.run(boot_resume_quit(config_dir))

    after = listing(config_dir)
    removed = sorted(before - after)
    added = sorted(after - before)
    print(f"after      : {len(after)} session directories (+{len(added)} created by boot: {added})")
    print(f"removed    : {len(removed)}")
    for name in removed:
        print(f"  - {name:<12} {shapes.get(name, '?')}")
    seeded_after = after & set(shapes)
    print(f"seeded     : {len(shapes)} -> {len(seeded_after)}")
    log = config_dir / "sessions" / ".cleanup-log.jsonl"
    if log.exists():
        print(f"cleanup log ({log}):")
        for line in log.read_text().splitlines():
            print(f"  {line}")
    else:
        print("cleanup log: absent")
    _ = live_id
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
