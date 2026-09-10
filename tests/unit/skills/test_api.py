"""api.py tests: default_skill_roots walk-up + dedupe (RS-04/RS-11) and the
make_skill_resolver adapter (RS-09)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import pytest

import local_operator.skills.api as api_module
import local_operator.skills.discovery as discovery_module
from local_operator.skills.api import (
    _url_name,
    default_skill_roots,
    make_skill_resolver,
)
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
        root = tmp_path / "roots"
        root.mkdir()
        skills: dict[str, Skill] = {}
        resolver = make_skill_resolver(skills, [root])

        before = resolver("skill://late")
        assert before is not None and "Unknown skill: late" in before

        self._write_skill_file(root, "late", body="# late body")
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
        resolver = make_skill_resolver(skills, [root])
        resolver("skill://nope")

        self._write_skill_file(root, "fresh")
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
        resolver = make_skill_resolver({}, [root])

        # First miss establishes the fingerprint and scans once.
        resolver("skill://nope")
        assert len(calls) == 1

        # The fingerprint is the ONLY thing that can stop the second scan --
        # there is no clock in this path any more, so a repeat miss on an
        # unchanged tree must still not rescan.
        resolver("skill://nope")
        assert len(calls) == 1

    def test_looping_typo_probes_but_never_rescans(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bound on a looping agent is the STAT, not a clock.

        R2: the time-based cooldown that used to provide this bound also made a
        skill written 50 ms after a miss unreadable for a second -- the exact
        write-then-read sequence the feature exists for. It was removed, so
        this pins what replaced it: every miss pays a sub-millisecond probe and
        an unchanged tree is NEVER rescanned, which is where the real cost is.
        """
        root = tmp_path / "roots"
        root.mkdir()
        probes: list[object] = []
        scans: list[object] = []
        real_fingerprint = api_module.roots_fingerprint

        def counting_fingerprint(roots: Sequence[Path]) -> tuple[object, ...]:
            probes.append(roots)
            return real_fingerprint(roots)

        def counting_discover(roots: Sequence[Path]) -> tuple[list[Skill], list[str]]:
            scans.append(roots)
            return [], []

        monkeypatch.setattr(api_module, "roots_fingerprint", counting_fingerprint)
        monkeypatch.setattr(api_module, "discover_skills", counting_discover)
        resolver = make_skill_resolver({}, [root])

        for _ in range(20):
            resolver("skill://nope")

        assert len(probes) == 20, "the cheap probe runs per miss; that is the design"
        assert len(scans) == 1, "an unchanged tree must never be rescanned"

    def test_skill_written_immediately_after_a_miss_is_readable(self, tmp_path: Path) -> None:
        """R2 regression: no time window may hide a just-authored skill.

        Reproduces the reported shape exactly -- a child probes for the skill,
        the parent writes it milliseconds later, the child reads again. Under
        the removed 1 s cooldown the second read returned ``Unknown skill``
        with no diagnostic; write-then-read is the normal authoring sequence,
        so this is the headline behaviour, not an edge case. No sleep and no
        fake clock: both reads happen in the same millisecond, which is the
        point.
        """
        root = tmp_path / "roots"
        root.mkdir()
        skills: dict[str, Skill] = {}
        parent = make_skill_resolver(skills, [root])

        def child(url: str) -> str | None:
            return parent(url)

        first = child("skill://lean-formalization")
        assert first is not None and "Unknown skill" in first

        self._write_skill_file(root, "lean-formalization", body="# lean body")

        second = child("skill://lean-formalization")
        assert second is not None, "a skill written after a miss must resolve at once"
        assert "# lean body" in second

    def test_diagnostic_is_not_suppressed_on_a_repeat_miss(self, tmp_path: Path) -> None:
        # The cooldown used to suppress diagnosis too, so the second read of a
        # broken skill lost its remedy. Every miss now explains itself.
        root = tmp_path / "roots"
        root.mkdir()
        self._write_skill_file(root, "blank", description=None)
        resolver = make_skill_resolver({}, [root])

        for _ in range(3):
            content = resolver("skill://blank")
            assert content is not None and "no 'description'" in content

    def test_transient_scan_failure_does_not_poison_the_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R3 regression: a one-off OSError must not freeze the fingerprint.

        The fingerprint used to be committed BEFORE ``discover_skills`` ran, so
        a single EMFILE blip left the state claiming it had ingested a tree it
        never read. Every later miss then compared equal and never rescanned --
        the skill stayed unreadable for the whole session even after the
        filesystem recovered, which is worse than the behaviour this replaced.
        """
        root = tmp_path / "roots"
        root.mkdir()
        self._write_skill_file(root, "lean", body="# lean body")

        real_discover = api_module.discover_skills
        failures: list[int] = []

        def flaky_discover(roots: Sequence[Path]) -> tuple[list[Skill], list[str]]:
            if not failures:
                failures.append(1)
                raise OSError(24, "Too many open files")
            return real_discover(roots)

        monkeypatch.setattr(api_module, "discover_skills", flaky_discover)
        resolver = make_skill_resolver({}, [root])

        during_failure = resolver("skill://lean")
        assert during_failure is not None and "Unknown skill" in during_failure

        # The tree has NOT changed since that failed read. Only an uncommitted
        # fingerprint can make this second read scan again.
        recovered = resolver("skill://lean")
        assert recovered is not None, "a healthy filesystem must be re-scanned"
        assert "# lean body" in recovered

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

    def test_concurrent_readers_both_resolve_the_same_new_skill(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R1 regression: the thread that LOSES the refresh race must still hit.

        Two real threads, because the failure only exists between them. The
        loser blocks on ``state.lock``, is released after the winner has
        already populated the shared mapping, and computes a fingerprint that
        now matches -- so a retry gated on "did MY call rescan?" skipped the
        re-lookup and answered ``Unknown skill`` for a skill sitting in the
        dict it was holding. The final assertion is the whole point: the
        mapping provably contains the name at the moment B answered.

        The scan is slowed to widen a window that is 17-42 ms in reality; a
        barrier makes both threads arrive inside it deterministically rather
        than by luck, so this cannot pass by timing.
        """
        import threading

        root = tmp_path / "roots"
        root.mkdir()
        self._write_skill_file(root, "lean", body="# lean body")

        real_discover = api_module.discover_skills
        entered = threading.Event()

        def slow_discover(roots: Sequence[Path]) -> tuple[list[Skill], list[str]]:
            entered.set()
            # Held long enough that the second thread is provably parked on the
            # lock before this one commits.
            time.sleep(0.3)
            return real_discover(roots)

        monkeypatch.setattr(api_module, "discover_skills", slow_discover)
        skills: dict[str, Skill] = {}
        resolver = make_skill_resolver(skills, [root])

        results: dict[str, str | None] = {}

        def read(tag: str) -> None:
            results[tag] = resolver("skill://lean")

        winner = threading.Thread(target=read, args=("A",))
        winner.start()
        # Only start B once A is inside the scan: that is the race, made
        # deterministic instead of hoped for.
        assert entered.wait(5.0), "the first reader never reached the scan"
        loser = threading.Thread(target=read, args=("B",))
        loser.start()
        winner.join(10.0)
        loser.join(10.0)

        assert "lean" in skills, "precondition: the refresh populated the shared mapping"
        assert results["A"] is not None and "# lean body" in results["A"]
        assert results["B"] is not None, "the losing reader returned nothing"
        assert "# lean body" in results["B"], (
            "the losing reader reported a miss for a skill already in the mapping: "
            f"{results['B']!r}"
        )

    def test_unsafe_name_in_the_netloc_drives_no_filesystem_work(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q1 at the resolver: a traversal NAME must not spend a probe either.

        ``test_unsafe_path_error_does_not_trigger_a_refresh`` covers the PATH
        portion. This covers the netloc, which the protocol's guards never
        inspect: ``skill://..`` and ``skill://%2fetc`` arrive as an ordinary
        "Unknown skill" and used to drive a full fingerprint probe on
        attacker-chosen input.
        """

        def explode(_roots: Sequence[Path]) -> tuple[object, ...]:
            raise AssertionError("an unsafe NAME must not cause a probe")

        monkeypatch.setattr(api_module, "roots_fingerprint", explode)
        root = tmp_path / "roots"
        root.mkdir()
        resolver = make_skill_resolver({}, [root])

        for url in ("skill://..", "skill://%2fetc", "skill://a%2fb", "skill://..%2f..%2fx"):
            content = resolver(url)
            assert content is not None and "Unknown skill" in content
            # And nothing from outside the roots comes back with it.
            assert "SKILL.md" not in content

    def test_posix_colon_named_skill_authored_mid_session_resolves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R9 at the resolver: the headline feature, for a legal POSIX name.

        ``:`` is an ordinary directory character on POSIX and the scanner
        registers ``a:b``, so gating the refresh on a name predicate that
        rejected it cost that skill the very liveness this resolver adds --
        silently, since the miss still read as a plain "Unknown skill".
        """
        monkeypatch.setattr(discovery_module, "_DRIVE_RESETS_JOIN", False)
        root = tmp_path / "roots"
        root.mkdir()
        skills: dict[str, Skill] = {}
        resolver = make_skill_resolver(skills, [root])

        before = resolver("skill://a:b")
        assert before is not None and "Unknown skill: a:b" in before

        self._write_skill_file(root, "a:b", body="# colon body")
        after = resolver("skill://a:b")
        assert after is not None and "# colon body" in after
        # In place, not rebound -- the mapping the session already holds.
        assert "a:b" in skills

    def test_windows_drive_name_in_the_netloc_still_drives_no_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R9's other direction: the drive door stays shut under Windows semantics.

        Percent-encoding is the delivery vehicle that matters -- ``%44%3ax``
        decodes to ``D:x`` in the NETLOC, which no path-portion guard inspects.
        """
        monkeypatch.setattr(discovery_module, "_DRIVE_RESETS_JOIN", True)

        def explode(_roots: Sequence[Path]) -> tuple[object, ...]:
            raise AssertionError("a drive-anchored NAME must not cause a probe")

        monkeypatch.setattr(api_module, "roots_fingerprint", explode)
        root = tmp_path / "roots"
        root.mkdir()
        resolver = make_skill_resolver({}, [root])

        for url in ("skill://D:x", "skill://%44%3ax", "skill://a%3ab", "skill://C%3A"):
            content = resolver(url)
            assert content is not None and "Unknown skill" in content
            assert "SKILL.md" not in content

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


class TestUrlNameDecodesPercentEncoding:
    """R11: the safety check runs on the DECODED name, so the decode is load-bearing.

    ``_url_name`` feeds ``is_plain_skill_name``, and percent-encoding is the
    delivery vehicle for every netloc attack this PR closes: ``%2fetc`` and
    ``%44%3ax`` are plain-looking single segments until they are decoded, and a
    ``_url_name`` that skipped the decode would hand both of them the refresh
    and the filesystem work the guard exists to deny. The resolver-level tests
    exercise that consequence; these pin the decode itself, so dropping the
    ``unquote`` cannot pass unnoticed on the strength of the URL's raw text
    happening to look safe.
    """

    def test_percent_encoded_name_is_decoded(self) -> None:
        # %44%3ax -> D:x: the drive shape arrives ONLY after decoding.
        assert _url_name("skill://%44%3ax") == "D:x"
        assert _url_name("skill://%2fetc") == "/etc"
        assert _url_name("skill://a%2fb") == "a/b"
        assert _url_name("skill://%2e%2e") == ".."

    def test_plain_name_survives_the_decode_unchanged(self) -> None:
        # The decode must not mangle the ordinary case it sits in front of.
        assert _url_name("skill://lean-formalization") == "lean-formalization"
        assert _url_name("skill://notes:2024") == "notes:2024"

    def test_the_decoded_name_is_what_the_safety_check_sees(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decode and the predicate, wired together as the resolver wires them."""
        monkeypatch.setattr(discovery_module, "_DRIVE_RESETS_JOIN", True)
        # Encoded, it looks like one plain segment; decoded, it is drive-anchored.
        assert discovery_module.is_plain_skill_name("%44%3ax") is True
        assert discovery_module.is_plain_skill_name(_url_name("skill://%44%3ax")) is False
