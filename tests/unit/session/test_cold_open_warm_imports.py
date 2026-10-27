"""The cold open must not pay a first import while the user waits for paint.

A conversation opened from the sidebar is rendered by a COLD viewer: no runtime
is started, the transcript on disk is the only source, and every module the open
needs for the first time is paid inside the window between the click and the
first frame. That cost is invisible in review — the open still works, it is just
slower — so it is pinned here by MODULE IDENTITY rather than by a millisecond
ceiling: the modules the open imports for the first time are named, and each one
must be either warmed by ``warm_session_imports`` (so the cost is paid on a boot
thread instead of in front of the user) or listed below with the reason it is
deliberately left cold.

Measured on the module that made this necessary: ``import local_operator.teams``
is 148.3 ms from cold and 0.2 ms resident (a fresh interpreter, a loaded 10-core
box, 2026-10-27), and a team-bound open reaches it through
``AttachedSession._cold_team_manager`` -> ``team_registry``.

A fresh subprocess for the reason ``test_import_graph.py`` gives: by the time a
test body runs, pytest has imported much of the tree, so an in-process assertion
would pass on a real regression. The transcript is SYNTHESISED rather than
committed: a checkpoint naming an active team is what makes the open enter the
team path at all, and two rows are enough to drive it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

REPO = Path(__file__).resolve().parents[3]

#: The modules a cold open imports for the first time, and why each stays cold.
#:
#: The rule the warm set applies is "import eagerly only if there is no
#: module-scope side effect and no heavy dependency chain". These fail the second
#: half: ``local_operator.secrets.access`` is the seam that reaches the crypto
#: stack (cryptography, ~290 ms of bindings, and the CLI pins it OFF its own
#: startup path for exactly that reason), so warming it would move a bundle this
#: size onto every boot to save the open ~45 ms. ``wakes.store`` is 3.4 ms and is
#: only read when the session's checkpoint carries scheduled rows.
#:
#: Costs are the module's own execution time inside an instrumented open with the
#: shipped warm set applied (loader hook, one fresh process per reading).


class _ProbeResult(TypedDict):
    """What the probe prints: module NAMES, split so a third-party import can be
    asserted on without walking the whole list."""

    first_party: list[str]
    third_party_roots: list[str]
    identity: str


_LEFT_COLD: dict[str, str] = {
    "local_operator.secrets": "the encrypted store's package root; crypto chain (see below)",
    "local_operator.secrets.access": "55.3 ms cold; pulls the cryptography bindings",
    "local_operator.secrets.audit": "part of the same chain",
    "local_operator.secrets.crypto": "part of the same chain",
    "local_operator.secrets.errors": "part of the same chain",
    "local_operator.secrets.keys": "part of the same chain",
    "local_operator.secrets.store": "47.8 ms cold; SQLite store, only opened on a secret read",
    "local_operator.wakes.store": "3.4 ms; only read when the checkpoint has scheduled rows",
}

_PROBE = r"""
import asyncio, json, sys
from pathlib import Path

config_dir = Path(sys.argv[1])
session_id = sys.argv[2]
warm = sys.argv[3] == "warm"

if warm:
    from local_operator.session_factory import warm_session_imports

    warm_session_imports()

# Imported AFTER the warm so the measurement is of the OPEN, not of the module
# that hosts it: the cost of importing this file's own dependencies belongs to
# the boot, and it is warmed by name for that reason.
from local_operator.session.attached import AttachedSession

before = set(sys.modules)


async def never(*_args, **_kwargs):
    raise RuntimeError("the cold open started a runtime")


async def go():
    return await AttachedSession.cold(
        session_id, config_dir=config_dir, cwd=str(config_dir), takeover_factory=never
    )


session = asyncio.run(go())
new = set(sys.modules) - before
identity = getattr(session.frontend_state, "effective_identity", None)
print(
    json.dumps(
        {
            "first_party": sorted(n for n in new if n.startswith("local_operator.")),
            "third_party_roots": sorted(
                n for n in new if not n.startswith("local_operator.") and "." not in n
            ),
            "identity": str(identity),
        }
    )
)
"""


def _checkpoint_state(session_id: str) -> dict[str, object]:
    """The turn-end row a runtime writes, as the cold reader meets it.

    Only the fields this test depends on: the conversation's own id and an
    active team. ``_cold_team_manager`` returns early without a team, so a
    checkpoint that omits it would make every assertion below vacuous.
    """
    return {
        "state_version": 1,
        "session_id": session_id,
        "epoch": "cold-open-warm-imports",
        "sequence": 1,
        "checkpoint_id": "cp",
        "goals": "",
        "active_agent": "manager",
        "active_team": "lane-one-probe",
        "cumulative_parent_cost": 0.0,
    }


def _plant(tmp_path: Path) -> tuple[Path, str]:
    """Write the two-row transcript this open needs, under an isolated config."""
    session_id = "cold-open-warm-imports"
    directory = tmp_path / "config" / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "row-1",
            "ts": 1.0,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "user",
                "content": [{"text": "the conversation the viewer opens"}],
            },
        },
        {
            "id": "row-2",
            "ts": 2.0,
            "type": "custom",
            "payload": {
                "custom_type": "frontend_state_checkpoint_v1",
                "details": {
                    "checkpoint_id": "cp",
                    "state": _checkpoint_state(session_id),
                },
            },
        },
    ]
    (directory / "transcript.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return tmp_path / "config", session_id


def _cold_open_imports(tmp_path: Path, *, warm: bool) -> _ProbeResult:
    config_dir, session_id = _plant(tmp_path)
    env = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    env.update(
        HOME=str(tmp_path),
        LOCAL_OPERATOR_CONFIG_DIR=str(config_dir),
        PYTHONPATH=str(REPO),
    )
    process = subprocess.run(
        [sys.executable, "-c", _PROBE, str(config_dir), session_id, "warm" if warm else "cold"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=env,
        timeout=300,
    )
    assert process.returncode == 0, process.stderr[-4000:]
    return json.loads(process.stdout.strip().splitlines()[-1])


def test_the_fixture_reaches_the_team_registry_unwarmed(tmp_path: Path) -> None:
    """The premise of every assertion below, in the state that makes it visible.

    Without the warm, the open is where ``local_operator.teams`` is first
    imported — 148.3 ms of pydantic model construction and roster file reading
    paid between the click and the frame. Asserting it HERE is what keeps the
    warm set honest: if the checkpoint stopped naming a team, or
    ``_cold_team_manager`` stopped reading the registry, this test would go red
    and the coverage test below would silently stop covering anything.
    """
    result = _cold_open_imports(tmp_path, warm=False)
    assert "local_operator.teams" in result["first_party"], (
        "the cold open no longer imports the team registry — the fixture has stopped "
        f"exercising the path this file exists for; saw {result['first_party']}"
    )


def test_warming_covers_every_first_party_module_the_cold_open_imports(
    tmp_path: Path,
) -> None:
    """The guard proper: an unwarmed first-party import fails HERE, by name.

    Not a millisecond ceiling — a membership assertion, because a cold import is
    a discrete, reviewable fact while its cost is a property of the machine it
    was measured on. A new lazy import on the open path therefore fails with the
    module named in the message, which is the whole point of the test.
    """
    result = _cold_open_imports(tmp_path, warm=True)
    uncovered = sorted(set(result["first_party"]) - set(_LEFT_COLD))
    assert not uncovered, (
        "the cold open imports these modules for the first time, and neither "
        "`warm_session_imports` nor the `_LEFT_COLD` table accounts for them: "
        f"{uncovered}. Add each to `_WARM_IMPORTS` (if it has no module-scope side "
        "effect and no heavy dependency chain) or to `_LEFT_COLD` with the reason."
    )


def test_the_warm_set_names_the_open_path_modules() -> None:
    """The sweep's findings, pinned BY NAME, because the identity guard above
    cannot see all of them.

    The guard opens a two-row synthetic conversation: enough to drive the team
    registry, the tail anchor and the monitors store, but NOT the paths that
    price a model or render a peer row, which only a real transcript reaches.
    Those were found by sweeping one (10,376 rows, an active team in the
    checkpoint) and they are pinned here so a rewrite of the list cannot drop
    them silently: every one is a first-party module whose first import landed
    inside the open at 5-10 ms.

    Membership alone would pass for a name that no longer resolves, so the warm
    is RUN over the shipped list — a rename leaves an entry that the per-name
    ``except`` swallows, and the only symptom would be the stall coming back.
    """
    import sys

    from local_operator.session_factory import _WARM_IMPORTS, warm_session_imports

    expected = (
        "local_operator.model.prices",
        "local_operator.monitors.store",
        "local_operator.network.credentials.placement",
        "local_operator.network.identity",
        "local_operator.session.tail_anchor",
        "local_operator.teams",
    )
    missing = [name for name in expected if name not in _WARM_IMPORTS]
    assert not missing, f"these are imported cold inside the open: {missing}"

    warm_session_imports()
    absent = [name for name in expected if name not in sys.modules]
    assert not absent, f"warmed but absent from sys.modules: {absent}"


def test_the_cold_open_does_not_reach_the_tunnel_gateway(tmp_path: Path) -> None:
    """The heaviest thing the REVEAL path imports, pinned out of the OPEN.

    ``local_operator.tunnels.gateway`` is PyJWT plus a starlette app (~500 ms
    cumulative on a loaded box) and is imported by the sidebar wrapper that
    decides whether a conversation is a remote peer, not by the open itself.
    Keeping it off the open is the difference between "the viewer is slow" and
    "the viewer is fast and one command is slow", so it is asserted rather than
    trusted — the module's own docstring in ``test_import_graph.py`` pins the
    same stack off the server's import path, for the same reason.
    """
    result = _cold_open_imports(tmp_path, warm=True)
    offenders = sorted(
        module
        for module in [*result["first_party"], *result["third_party_roots"]]
        if module in {"jwt", "local_operator.tunnels", "local_operator.tunnels.gateway"}
    )
    assert not offenders, f"the cold open is importing the tunnel gateway stack: {offenders}"
