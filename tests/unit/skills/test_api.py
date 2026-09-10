"""api.py tests: default_skill_roots walk-up + dedupe (RS-04/RS-11) and the
make_skill_resolver adapter (RS-09)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import local_operator.skills.api as api_module
from local_operator.skills.api import default_skill_roots, make_skill_resolver
from local_operator.skills.discovery import Skill
from local_operator.skills.protocol import MAX_READ_BYTES

_SKILLS_SUBDIR = Path(".local-operator") / "skills"


def _make_skill(root: Path, name: str) -> Skill:
    base_dir = root / name
    base_dir.mkdir(parents=True)
    skill_md = base_dir / "SKILL.md"
    skill_md.write_text(f"---\ndescription: {name} does things\n---\n# {name}")
    return Skill(
        name=name,
        description=f"{name} does things",
        file_path=skill_md,
        base_dir=base_dir,
        source=str(root),
    )


class TestDefaultSkillRoots:
    def test_walk_up_collects_ancestor_roots_deepest_first(self, tmp_path: Path) -> None:
        project = tmp_path / "repo" / "nested"
        project.mkdir(parents=True)
        roots = default_skill_roots(project)
        assert roots[0] == project / _SKILLS_SUBDIR
        assert (tmp_path / "repo" / _SKILLS_SUBDIR) in roots
        assert (tmp_path / _SKILLS_SUBDIR) in roots
        # Deepest (most project-local) root comes first.
        assert roots.index(project / _SKILLS_SUBDIR) < roots.index(tmp_path / _SKILLS_SUBDIR)

    def test_home_root_appended_last(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        roots = default_skill_roots(tmp_path / "elsewhere")
        assert roots[-1] == home / _SKILLS_SUBDIR

    def test_cwd_outside_home_still_walks_up(self, tmp_path: Path) -> None:
        # RS-04/RS-11: a repo at /opt-style paths (outside $HOME) must still
        # get its project-local roots — the walk goes to the filesystem root,
        # not just to home.
        repo = tmp_path / "srv" / "app"
        repo.mkdir(parents=True)
        roots = default_skill_roots(repo)
        assert (repo / _SKILLS_SUBDIR) in roots
        assert (tmp_path / "srv" / _SKILLS_SUBDIR) in roots

    def test_home_inside_walk_is_not_duplicated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / "repo").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        roots = default_skill_roots(home / "repo")
        # The walk reaches home; the appended home root dedupes against it.
        assert roots.count(home / _SKILLS_SUBDIR) == 1

    def test_symlinked_ancestor_deduped_by_realpath(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        monkeypatch.setenv("HOME", str(tmp_path / "nothome"))
        roots = default_skill_roots(link / "project")
        # /link/project walks through real/project (via realpath); the same
        # physical directory appears once, not twice.
        keys = {str(r.resolve()) for r in roots if r.exists()}
        assert len(keys) == len([r for r in roots if r.exists()])

    def test_default_cwd_is_process_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        roots = default_skill_roots()
        assert roots[0] == tmp_path / _SKILLS_SUBDIR


class TestMakeSkillResolver:
    def test_non_skill_url_returns_none(self, tmp_path: Path) -> None:
        resolver = make_skill_resolver({"alpha": _make_skill(tmp_path, "alpha")})
        assert resolver("file:///etc/passwd") is None
        assert resolver("http://example.com") is None
        assert resolver("memory://notes") is None

    def test_skill_url_returns_content(self, tmp_path: Path) -> None:
        resolver = make_skill_resolver({"alpha": _make_skill(tmp_path, "alpha")})
        content = resolver("skill://alpha")
        assert content is not None
        assert "# alpha" in content

    def test_empty_name_returns_error_message_with_available_skills(self, tmp_path: Path) -> None:
        skills = {
            "alpha": _make_skill(tmp_path, "alpha"),
            "beta": _make_skill(tmp_path, "beta"),
        }
        resolver = make_skill_resolver(skills)
        content = resolver("skill://")
        assert content is not None
        assert "missing a name" in content
        assert "Available skills: alpha, beta" in content

    def test_unknown_name_returns_error_message_as_content(self, tmp_path: Path) -> None:
        # RS-09: the adapter never raises; the available-names list reaches
        # the model as a clean tool result instead of an exception envelope.
        skills = {
            "alpha": _make_skill(tmp_path, "alpha"),
            "beta": _make_skill(tmp_path, "beta"),
        }
        resolver = make_skill_resolver(skills)
        content = resolver("skill://gamma")
        assert content is not None
        assert "Unknown skill: gamma" in content
        assert "Available: alpha, beta" in content

    def test_unsafe_path_returns_error_message_as_content(self, tmp_path: Path) -> None:
        resolver = make_skill_resolver({"alpha": _make_skill(tmp_path, "alpha")})
        content = resolver("skill://alpha/../../etc/passwd")
        assert content is not None
        assert "not allowed" in content

    def test_error_content_fits_in_one_tool_result(self, tmp_path: Path) -> None:
        resolver = make_skill_resolver({"alpha": _make_skill(tmp_path, "alpha")})
        content = resolver("skill://gamma")
        assert content is not None
        assert len(content.encode("utf-8")) < MAX_READ_BYTES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


class TestResolverRefreshOnMiss:
    """A skill authored mid-session must resolve without a restart.

    The defect these guard: an agent wrote a valid skill to a native global
    root and neither it nor any subagent it launched could read it for the rest
    of the session, because ``discover_skills`` ran once at session
    construction and the resolver closed over that snapshot.
    """

    @staticmethod
    def _fake_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[float], None]:
        """Advance ``time.monotonic`` explicitly instead of sleeping.

        The refresh is cooldown-bounded to one probe per second, so a test that
        writes a skill and reads it back in the same millisecond is asserting
        against the cooldown rather than against the refresh. Real turns are
        seconds apart; a controlled clock reproduces that without adding a
        second of wall time per test.
        """
        current = 1000.0

        def now() -> float:
            return current

        def advance(seconds: float) -> None:
            nonlocal current
            current += seconds

        monkeypatch.setattr(api_module.time, "monotonic", now)
        return advance

    def _write_skill_file(
        self,
        root: Path,
        dirname: str,
        *,
        name: str | None = None,
        description: str | None = "Does things.",
        enabled: object | None = None,
        body: str = "# body",
    ) -> Path:
        skill_dir = root / dirname
        skill_dir.mkdir(parents=True, exist_ok=True)
        lines = ["---"]
        if name is not None:
            lines.append(f"name: {name}")
        if description is not None:
            lines.append(f"description: {description}")
        if enabled is not None:
            lines.append(f"enabled: {enabled}")
        lines.extend(["---", "", body])
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("\n".join(lines))
        return skill_md

    def test_skill_created_after_the_resolver_resolves_on_the_next_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        advance = self._fake_clock(monkeypatch)
        root = tmp_path / "roots"
        root.mkdir()
        skills: dict[str, Skill] = {}
        resolver = make_skill_resolver(skills, [root])

        before = resolver("skill://late")
        assert before is not None and "Unknown skill: late" in before

        self._write_skill_file(root, "late", body="# late body")
        advance(2.0)
        after = resolver("skill://late")
        assert after is not None
        assert "# late body" in after

    def test_child_resolver_sees_a_skill_created_after_it_was_built(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression guard that matters: liveness through a CHILD.

        ``harness/subagent.py`` hands a child the PARENT's resolver closure, so
        propagation depends entirely on the refresh mutating the shared mapping
        IN PLACE. A future refactor that rebinds (``skills = {...}``) instead of
        calling ``.update()`` still passes every parent-side test and breaks
        every already-running subagent silently. Assert the child.
        """
        advance = self._fake_clock(monkeypatch)
        root = tmp_path / "roots"
        root.mkdir()
        skills: dict[str, Skill] = {"alpha": _make_skill(tmp_path / "existing", "alpha")}
        parent = make_skill_resolver(skills, [root])

        # Mirrors subagent.py: the child chains its own resolvers and falls
        # through to the parent's closure for skill:// URLs.
        def child(url: str) -> str | None:
            if url.startswith("mcp://"):
                return None
            return parent(url)

        before = child("skill://lean-formalization")
        assert before is not None and "Unknown skill: lean-formalization" in before

        self._write_skill_file(root, "lean-formalization", body="# lean body")
        advance(2.0)

        after = child("skill://lean-formalization")
        assert after is not None, "child resolver must see the new skill"
        assert "# lean body" in after

    def test_refresh_mutates_the_mapping_in_place(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The identity assertion behind the child test above, stated directly:
        # session_factory, the parent resolver and every child hold this one
        # object, so a rebind would orphan all of them.
        root = tmp_path / "roots"
        root.mkdir()
        skills: dict[str, Skill] = {}
        original_id = id(skills)
        advance = self._fake_clock(monkeypatch)
        resolver = make_skill_resolver(skills, [root])
        resolver("skill://nope")

        self._write_skill_file(root, "fresh")
        advance(2.0)
        resolver("skill://fresh")

        assert id(skills) == original_id
        assert "fresh" in skills

    def test_hit_path_does_no_filesystem_work(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The cost constraint: a skill that exists resolves on one dict lookup.
        # Making the fingerprint raise proves nothing above the lookup runs.
        def explode(_roots: Sequence[Path]) -> tuple[object, ...]:
            raise AssertionError("fingerprint must not run on the hit path")

        monkeypatch.setattr(api_module, "roots_fingerprint", explode)
        resolver = make_skill_resolver({"alpha": _make_skill(tmp_path, "alpha")}, [tmp_path])
        content = resolver("skill://alpha")
        assert content is not None and "# alpha" in content

    def test_unchanged_fingerprint_skips_the_full_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "roots"
        root.mkdir()
        calls: list[object] = []

        def counting_discover(roots: Sequence[Path]) -> tuple[list[Skill], list[str]]:
            calls.append(roots)
            return [], []

        monkeypatch.setattr(api_module, "discover_skills", counting_discover)
        advance = self._fake_clock(monkeypatch)
        resolver = make_skill_resolver({}, [root])

        # First miss establishes the fingerprint and scans once.
        resolver("skill://nope")
        assert len(calls) == 1

        # Past the cooldown, so the cooldown cannot be what stops the second
        # scan -- the fingerprint has to. Advancing rather than jumping the
        # clock matters: a clock that moves BACKWARDS makes the cooldown
        # comparison short-circuit and the test would pass without ever
        # reaching the fingerprint it exists to check.
        advance(2.0)
        resolver("skill://nope")
        assert len(calls) == 1

    def test_cooldown_bounds_probes_within_the_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A looping agent retrying a typo'd name pays one probe per second,
        # not one per read.
        root = tmp_path / "roots"
        root.mkdir()
        probes: list[object] = []
        real_fingerprint = api_module.roots_fingerprint

        def counting_fingerprint(roots: Sequence[Path]) -> tuple[object, ...]:
            probes.append(roots)
            return real_fingerprint(roots)

        monkeypatch.setattr(api_module, "roots_fingerprint", counting_fingerprint)
        advance = self._fake_clock(monkeypatch)
        resolver = make_skill_resolver({}, [root])

        resolver("skill://nope")
        advance(0.1)
        resolver("skill://nope")
        advance(0.1)
        resolver("skill://nope")
        assert len(probes) == 1

        # And the window does expire: the next read past it probes again.
        advance(2.0)
        resolver("skill://nope")
        assert len(probes) == 2

    def test_cooldown_suppresses_the_diagnostic_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inside the window the resolver does NO filesystem work at all.

        Diagnosis walks the roots, so letting it run inside the cooldown would
        reopen the very hole the cooldown closes: a retry loop on a bad name
        would still drive a walk per read.
        """
        root = tmp_path / "roots"
        root.mkdir()
        self._write_skill_file(root, "blank", description=None)
        advance = self._fake_clock(monkeypatch)
        resolver = make_skill_resolver({}, [root])

        first = resolver("skill://blank")
        assert first is not None and "no 'description'" in first

        advance(0.1)
        throttled = resolver("skill://blank")
        assert throttled is not None
        assert "no 'description'" not in throttled

        advance(2.0)
        recovered = resolver("skill://blank")
        assert recovered is not None and "no 'description'" in recovered

    def test_unsafe_path_error_does_not_trigger_a_refresh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Traversal and dotfile rejections are ValueErrors too. Rescanning on
        # them would let a malformed URL drive filesystem work.
        def explode(_roots: Sequence[Path]) -> tuple[object, ...]:
            raise AssertionError("an unsafe path must not cause a rescan")

        monkeypatch.setattr(api_module, "roots_fingerprint", explode)
        resolver = make_skill_resolver({"alpha": _make_skill(tmp_path, "alpha")}, [tmp_path])
        content = resolver("skill://alpha/../../etc/passwd")
        assert content is not None and "not allowed" in content

    def test_deleted_skill_stays_in_the_mapping_and_reads_as_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Growth-only: removing entries would let one agent's cleanup break a
        # sibling child mid-read. The stale entry degrades to a clear message.
        self._fake_clock(monkeypatch)
        root = tmp_path / "roots"
        root.mkdir()
        skill_md = self._write_skill_file(root, "doomed")
        skills: dict[str, Skill] = {}
        resolver = make_skill_resolver(skills, [root])
        assert resolver("skill://doomed") is not None
        assert "doomed" in skills

        skill_md.unlink()
        content = resolver("skill://doomed")
        assert content is not None
        assert "No such file or directory" in content
        assert "doomed" in skills

    def test_roots_none_preserves_todays_behaviour(self, tmp_path: Path) -> None:
        # Guards every existing caller and the parallel guide resolver: with no
        # roots there is no rescan and no diagnostic, just the bare message.
        root = tmp_path / "roots"
        root.mkdir()
        self._write_skill_file(root, "present")
        resolver = make_skill_resolver({})
        content = resolver("skill://present")
        assert content == "Unknown skill: present\nAvailable: (none)"

    def test_surviving_miss_explains_the_cause(self, tmp_path: Path) -> None:
        # A skill dropped for a blank description is invisible; without the
        # diagnostic the author sees only "Unknown skill".
        root = tmp_path / "roots"
        root.mkdir()
        self._write_skill_file(root, "blank", description=None)
        resolver = make_skill_resolver({}, [root])
        content = resolver("skill://blank")
        assert content is not None
        assert "Unknown skill: blank" in content
        assert "no 'description'" in content
