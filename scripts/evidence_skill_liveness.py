"""End-to-end evidence for the skill-authoring liveness fix.

NOT a test and not part of any suite. It exists because a green unit suite is
not evidence of THIS fix: the reported defect is that a subagent launched
before a skill existed could not read it for the rest of the session, and the
only way to show that gone is to build a real session, build a real child from
it, author the skill afterwards, and read it back through the child's own
``read`` tool.

Every layer below is the shipping one -- ``create_session`` composes the
session, ``_build_child_session`` composes the child exactly as ``task`` does,
and the read goes through ``ToolContext.resolve_internal_url``. The
``hosting="test"`` model is the only stand-in, because no provider call is
needed to resolve a ``skill://`` URL.

Run it against the worktree's own venv:

    .venv/bin/python scripts/evidence_skill_liveness.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Scripts self-correct their import root so they read the tree they live in
# rather than whatever an editable install points at (AGENTS.md).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "hosting": "test",
        "model": "test",
        "agent_name": None,
        "agent_id": None,
        "yolo": True,
        "train": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _say(label: str, value: object) -> None:
    print(f"\n--- {label} ---")
    print(value)


async def main() -> int:
    sandbox = Path(tempfile.mkdtemp(prefix="skill-liveness-evidence-"))
    home = sandbox / "home"
    project = sandbox / "project"
    (home / ".local-operator").mkdir(parents=True)
    project.mkdir(parents=True)

    # Isolate HOME and the config dir: this script must never write into the
    # operator's real ~/.local-operator (AGENTS.md, "Isolating a run").
    os.environ["HOME"] = str(home)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(home / ".local-operator")
    os.environ["LOCAL_OPERATOR_SKILL_EXTRA_ROOTS"] = ""
    os.chdir(project)

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.harness.subagent import _build_child_session
    from local_operator.session_factory import create_session

    config_dir = home / ".local-operator"
    global_skills = home / ".local-operator" / "skills"

    print(f"sandbox HOME       : {home}")
    print(f"global skill root  : {global_skills}")
    print(f"session cwd        : {project}")
    print(f"skill exists yet?  : {(global_skills / 'lean-formalization').exists()}")

    from local_operator.session.session import Session

    built = await create_session(
        _args(),
        ConfigManager(config_dir),
        CredentialManager(config_dir),
        AgentRegistry(config_dir),
    )
    # create_session may hand back a RemoteSession when attaching to an owned
    # runtime; this sandbox never does, and the child builder needs the real
    # Session, so assert rather than silently proving nothing.
    assert isinstance(built, Session), f"expected a local Session, got {type(built).__name__}"
    session = built

    # The child is built BEFORE the skill exists -- this is the reported
    # defect's exact shape: standing instructions that name a skill authored
    # after the child was launched.
    child = await _build_child_session(
        label="evidence-child",
        prompt="FIRST, EVERY TIME: read skill://lean-formalization",
        parent_session=session,
        model_spec=None,
        job_id="evidence-job",
    )

    # ``_build_tool_context`` is what runs at the start of every turn and is
    # what hands the ``read`` tool its resolver, so going through it is the
    # real path rather than a stashed reference.
    def _reader(target: Session):
        def read(url: str) -> str | None:
            resolve = target._build_tool_context().resolve_internal_url
            # A session without a resolver would make every read below return
            # None and the evidence vacuously "pass"; fail loudly instead.
            assert resolve is not None, "session has no internal-URL resolver"
            return resolve(url)

        return read

    parent_read = _reader(session)
    child_read = _reader(child)

    _say(
        "BEFORE — parent reads skill://lean-formalization",
        parent_read("skill://lean-formalization"),
    )
    _say(
        "BEFORE — child reads skill://lean-formalization", child_read("skill://lean-formalization")
    )

    # Author the skill exactly where the docs say to put a global one. This is
    # what an agent does with the `write` tool.
    skill_dir = global_skills / "lean-formalization"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: lean-formalization\n"
        "description: Formalize mathematical arguments in Lean 4.\n"
        "---\n"
        "\n"
        "# Lean formalization\n"
        "\n"
        "State the theorem before writing tactics.\n"
    )
    print(f"\nauthored           : {skill_dir / 'SKILL.md'}")

    # The refresh is cooldown-bounded to one filesystem probe per second, and
    # the BEFORE reads above just spent it. Showing the blocked read rather
    # than hiding it behind a sleep: this is the designed bound, not a defect,
    # and a real agent turn is seconds long so it is not reachable in practice.
    _say(
        "IMMEDIATELY AFTER (inside the 1s cooldown - probe bounded, still misses)",
        child_read("skill://lean-formalization"),
    )

    await asyncio.sleep(1.1)
    _say(
        "AFTER — parent reads skill://lean-formalization", parent_read("skill://lean-formalization")
    )
    after_child = child_read("skill://lean-formalization")
    _say("AFTER — child reads skill://lean-formalization", after_child)

    # --- the four causes that used to share one opaque message -------------
    print("\n=== diagnostics: four causes, four remedies ===")
    cases: list[tuple[str, str, str]] = [
        ("no SKILL.md", "empty-dir", ""),
        (
            "blank description",
            "blank-desc",
            "---\nname: blank-desc\n---\n\n# Body\n",
        ),
        (
            "frontmatter name != directory",
            "misnamed",
            "---\nname: actually-this\ndescription: Renamed in frontmatter.\n---\n\n# Body\n",
        ),
        (
            "malformed frontmatter",
            "broken-yaml",
            "---\ndescription: Unterminated block.\n\n# Body\n",
        ),
        (
            "disabled",
            "switched-off",
            "---\nname: switched-off\ndescription: Turned off.\nenabled: false\n---\n\n# Body\n",
        ),
    ]
    for label, dirname, content in cases:
        case_dir = global_skills / dirname
        case_dir.mkdir(parents=True, exist_ok=True)
        if content:
            (case_dir / "SKILL.md").write_text(content)
        # Past the 1.0 s cooldown, so each read genuinely re-probes.
        await asyncio.sleep(1.1)
        _say(f"diagnostic — {label}", child_read(f"skill://{dirname}"))

    # The shadow case, shown honestly: it is NOT reachable through a URL read,
    # because if the winning copy loads it also claims the name and the read
    # HITS. Both halves are shown -- the read succeeding with the project copy,
    # and the helper naming the shadowing the read cannot surface.
    project_skills = project / ".local-operator" / "skills" / "shadowed"
    project_skills.mkdir(parents=True)
    (project_skills / "SKILL.md").write_text(
        "---\nname: shadowed\ndescription: The project copy wins.\n---\n\n# Project copy\n"
    )
    global_shadow = global_skills / "shadowed"
    global_shadow.mkdir(parents=True)
    (global_shadow / "SKILL.md").write_text(
        "---\nname: shadowed\ndescription: The global copy loses the name.\n---\n\n# Global copy\n"
    )
    await asyncio.sleep(1.1)
    _say(
        "shadowing — the READ hits the winner (project root beats global)",
        child_read("skill://shadowed"),
    )

    from local_operator.skills.discovery import diagnose_missing_skill

    _say(
        "shadowing — diagnose_missing_skill names the loser and the winner",
        diagnose_missing_skill(
            "shadowed",
            [project / ".local-operator" / "skills", global_skills],
        ),
    )

    await child.dispose()
    await session.dispose()

    ok = after_child is not None and "State the theorem before writing tactics." in after_child
    print(f"\n=== RESULT: child resolved the late-authored skill: {ok} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
