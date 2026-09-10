"""Discovery tests: frontmatter rules, scan shape, collision + ordering."""

from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path

import pytest

import local_operator.skills.discovery as discovery_module
from local_operator.skills.discovery import (
    Skill,
    diagnose_missing_skill,
    discover_skills,
    is_plain_skill_name,
    parse_frontmatter,
    roots_fingerprint,
    scan_skills_dir,
)


def _discovery_under_os_name(platform_name: str) -> types.ModuleType:
    """Re-execute ``discovery.py`` with ``os.name`` forced, and return the copy.

    Module-level constants derived from the platform are fixed at import time,
    so the only way to observe the DERIVATION (rather than its result on this
    host) is to run the module body again under each platform name.

    Loaded under a distinct module name rather than ``importlib.reload``-ing
    the live one, matching ``tests/unit/test_xdist_worker_budget.py``: the
    session running this test already holds
    ``local_operator.skills.discovery`` and many other modules hold references
    INTO it, so rebinding its constant -- even transiently -- would change the
    behaviour of concurrently running tests in the same worker. This private
    copy is discarded when the test ends.

    The spec is built BEFORE the patch and only ``exec_module`` runs under it:
    importlib resolves a source path with the running platform's rules, so a
    spec created while ``os.name == "nt"`` treats this absolute POSIX path as
    relative, joins it onto the cwd with backslashes, and raises
    ``FileNotFoundError`` before the module body ever runs.
    """
    source = Path(discovery_module.__file__)
    spec = importlib.util.spec_from_file_location("_skills_discovery_under_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "name", platform_name)
        spec.loader.exec_module(module)
    return module


def _write_skill(
    root: Path,
    dirname: str,
    *,
    name: str | None = None,
    description: str | None = "A test skill.",
    enabled: bool | None = None,
    hide: bool | None = None,
    disable_model_invocation: bool | None = None,
    body: str = "# Body",
    globs: str | list[str] | None = None,
) -> Path:
    skill_dir = root / dirname
    skill_dir.mkdir(parents=True)
    lines = ["---"]
    if name is not None:
        lines.append(f"name: {name}")
    if description is not None:
        lines.append(f"description: {description}")
    if enabled is not None:
        lines.append(f"enabled: {str(enabled).lower()}")
    if hide is not None:
        lines.append(f"hide: {str(hide).lower()}")
    if disable_model_invocation is not None:
        lines.append(f"disable-model-invocation: {str(disable_model_invocation).lower()}")
    if globs is not None:
        # YAML plain scalars cannot start with "*" (alias marker), so both
        # spellings quote their values — exactly what real SKILL.md authors
        # must do; an unquoted ``*.py`` makes the whole block malformed YAML
        # and discovery drops the skill.
        if isinstance(globs, str):
            lines.append(f'globs: "{globs}"')
        else:
            lines.append("globs:")
            lines.extend(f'  - "{pattern}"' for pattern in globs)
    lines.append("---")
    lines.append("")
    lines.append(body)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("\n".join(lines), encoding="utf-8")
    return skill_md


class TestGlobsFrontmatter:
    """The optional ``globs`` key rides on the Skill record in both ecosystem
    spellings; junk shapes degrade to no globs rather than dropping the
    skill (the description remains the primary routing signal)."""

    def test_csv_string_and_yaml_list(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        _write_skill(root, "csv", globs="*.py, src/**/*.ts ,")
        _write_skill(root, "list", globs=["*.tf", "infra/**"])
        skills = {skill.name: skill for skill in scan_skills_dir(root, source="test")}
        assert skills["csv"].globs == ("*.py", "src/**/*.ts")
        assert skills["list"].globs == ("*.tf", "infra/**")

    def test_junk_shapes_degrade_gracefully(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        # Hand-written frontmatter: the helper quotes list items, but the
        # degradation contract has to hold for unquoted non-strings too.
        skill_dir = root / "junk"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: A test skill.\n"
            "globs:\n"
            "  - 3\n"
            "  - true\n"
            '  - "*.md"\n'
            "---\n# Body",
            encoding="utf-8",
        )
        skills = {skill.name: skill for skill in scan_skills_dir(root, source="test")}
        # Only the usable pattern survives; the skill itself is kept.
        assert skills["junk"].globs == ("*.md",)

    def test_absent_key_defaults_to_empty_tuple(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        _write_skill(root, "plain")
        skills = {skill.name: skill for skill in scan_skills_dir(root, source="test")}
        assert skills["plain"].globs == ()


class TestParseFrontmatter:
    def test_parses_all_recognized_keys(self) -> None:
        text = (
            "---\nname: foo\ndescription: Does things\nenabled: true\n"
            "hide: true\ndisable-model-invocation: true\n---\nbody"
        )
        meta = parse_frontmatter(text)
        assert meta["name"] == "foo"
        assert meta["description"] == "Does things"
        assert meta["enabled"] is True
        assert meta["hide"] is True
        assert meta["disable-model-invocation"] is True

    def test_no_frontmatter_returns_empty(self) -> None:
        assert parse_frontmatter("# Just markdown") == {}

    def test_unterminated_block_returns_empty(self) -> None:
        assert parse_frontmatter("---\nname: foo\nno closing fence") == {}

    def test_malformed_yaml_returns_empty(self) -> None:
        assert parse_frontmatter("---\nname: [unclosed\n---\nbody") == {}

    def test_non_dict_yaml_returns_empty(self) -> None:
        assert parse_frontmatter("---\n- a\n- b\n---\nbody") == {}


class TestScanSkillsDir:
    def test_frontmatter_rules(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        _write_skill(root, "with-name", name="renamed")
        _write_skill(root, "fallback-name")  # no name -> dir name
        _write_skill(root, "disabled", enabled=False)
        _write_skill(root, "no-desc", description=None)
        _write_skill(root, "blank-desc", description="   ")
        _write_skill(root, "hidden", hide=True)
        _write_skill(root, "dmi", disable_model_invocation=True)

        skills = {s.name: s for s in scan_skills_dir(root, source="test")}

        assert "renamed" in skills  # frontmatter name wins
        assert skills["renamed"].base_dir == root / "with-name"
        assert "fallback-name" in skills  # name fell back to dir name
        assert "disabled" not in skills
        assert "no-desc" not in skills
        assert "blank-desc" not in skills
        assert skills["hidden"].hide is True
        assert skills["dmi"].hide is True
        assert all(s.source == "test" for s in skills.values())

    def test_skips_dotdirs_and_nondirs_and_is_non_recursive(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        _write_skill(root, "real")
        _write_skill(root, ".hidden-dir")
        # non-recursive: nested skill under a skill dir is not picked up
        nested = root / "real" / "nested"
        nested.mkdir()
        (nested / "SKILL.md").write_text("---\ndescription: nested\n---\n", encoding="utf-8")
        # a bare file named SKILL.md directly under root is ignored (no include_self)
        (root / "SKILL.md").write_text("---\ndescription: self\n---\n", encoding="utf-8")

        skills = scan_skills_dir(root, source="test")
        assert [s.name for s in skills] == ["real"]

    def test_include_self_picks_up_root_skill_md(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        (root / "SKILL.md").write_text("---\ndescription: self\n---\n", encoding="utf-8")
        skills = scan_skills_dir(root, source="test", include_self=True)
        assert [s.name for s in skills] == ["skills"]  # falls back to root dir name

    def test_realpath_dedupe_shared_seen(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        _write_skill(root, "target")
        link = tmp_path / "linked-skills" / "alias"
        link.parent.mkdir()
        (link).symlink_to(root / "target")

        seen: set[str] = set()
        first = scan_skills_dir(root, source="a", seen=seen)
        second = scan_skills_dir(link.parent, source="b", seen=seen)
        assert [s.name for s in first] == ["target"]
        assert second == []  # same physical SKILL.md, not reloaded

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert scan_skills_dir(tmp_path / "nope", source="x") == []


class TestDiscoverSkills:
    def test_deterministic_order_ci_then_exact_then_path(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        _write_skill(root, "a", name="zeta")
        _write_skill(root, "b", name="Alpha")
        _write_skill(root, "c", name="alpha")
        _write_skill(root, "d", name="beta")

        skills, warnings = discover_skills([root])
        assert warnings == []
        # "Alpha" < "alpha" on the exact-name tiebreak (uppercase first)
        assert [s.name for s in skills] == ["Alpha", "alpha", "beta", "zeta"]

    def test_earlier_root_wins_collision_with_warning(self, tmp_path: Path) -> None:
        root1 = tmp_path / "project"
        root2 = tmp_path / "home"
        _write_skill(root1, "dup", description="project version")
        _write_skill(root2, "dup", description="home version")
        _write_skill(root2, "only-home")

        skills, warnings = discover_skills([root1, root2])
        by_name = {s.name: s for s in skills}
        assert by_name["dup"].description == "project version"
        assert "only-home" in by_name
        assert len(warnings) == 1
        assert "dup" in warnings[0]
        assert "shadowed" in warnings[0]

    def test_missing_roots_skipped_silently(self, tmp_path: Path) -> None:
        root = tmp_path / "real"
        _write_skill(root, "a")
        skills, warnings = discover_skills([tmp_path / "ghost", root])
        assert [s.name for s in skills] == ["a"]
        assert warnings == []

    def test_realpath_dedupe_across_roots_no_warning(self, tmp_path: Path) -> None:
        root1 = tmp_path / "r1"
        _write_skill(root1, "shared")
        root2 = tmp_path / "r2"
        root2.mkdir()
        (root2 / "shared-link").symlink_to(root1 / "shared")

        skills, warnings = discover_skills([root1, root2])
        assert [s.name for s in skills] == ["shared"]
        assert warnings == []  # dedupe is not a collision

    def test_skill_model_fields(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        path = _write_skill(root, "mine", description="Does stuff")
        skills, _ = discover_skills([root])
        skill: Skill = skills[0]
        assert skill.file_path == path
        assert skill.base_dir == root / "mine"
        assert skill.source == str(root)
        assert skill.hide is False

    def test_enabled_falsy_spellings_drop_the_skill(self, tmp_path: Path) -> None:
        # Authors write enabled: 0 or enabled: "false" expecting a disabled
        # skill; the old identity comparison against the False singleton kept
        # them enabled. Every falsy spelling now drops.
        root = tmp_path / "skills"
        for name, value in (("zero", "0"), ("quoted", "'false'"), ("no", "no")):
            (root / name).mkdir(parents=True)
            (root / name / "SKILL.md").write_text(
                f"---\ndescription: {name}\nenabled: {value}\n---\n",
                encoding="utf-8",
            )
        (root / "live").mkdir(parents=True)
        (root / "live" / "SKILL.md").write_text(
            "---\ndescription: live\nenabled: true\n---\n", encoding="utf-8"
        )
        skills, _ = discover_skills([root])
        assert [s.name for s in skills] == ["live"]

    def test_stable_across_repeated_scans(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        for letter in ["c", "a", "b"]:
            _write_skill(root, letter)
        first, _ = discover_skills([root])
        second, _ = discover_skills([root])
        assert [s.name for s in first] == [s.name for s in second]
        assert [str(s.file_path) for s in first] == [str(s.file_path) for s in second]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


class TestRootsFingerprint:
    """The gate that decides whether a full 17-25 ms scan runs at all."""

    def test_new_skill_directory_changes_the_fingerprint(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        before = roots_fingerprint([root])
        _write_skill(root, "alpha")
        assert roots_fingerprint([root]) != before

    def test_skill_md_created_in_an_existing_directory_changes_it(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        (root / "alpha").mkdir()
        before = roots_fingerprint([root])
        (root / "alpha" / "SKILL.md").write_text("---\ndescription: A skill.\n---\n# Body")
        assert roots_fingerprint([root]) != before

    def test_skill_md_edited_in_place_changes_it(self, tmp_path: Path) -> None:
        """The row that makes per-file stats mandatory.

        Neither the root's nor the child's mtime moves for an in-place edit, so
        a fingerprint built from directory mtimes alone would report "no
        change" -- and this is a REAL case: a skill dropped at discovery for a
        blank ``description`` is absent from the mapping, so repairing its
        frontmatter is exactly a miss the gate has to notice.
        """
        root = tmp_path / "skills"
        root.mkdir()
        skill_md = _write_skill(root, "alpha", description=None)
        root_mtime = root.stat().st_mtime_ns
        child_mtime = (root / "alpha").stat().st_mtime_ns
        before = roots_fingerprint([root])

        # Rewrite with a description and a different length, preserving the
        # directory mtimes the naive fingerprint would have relied on.
        os.utime(skill_md, ns=(root_mtime + 1_000_000, root_mtime + 1_000_000))
        skill_md.write_text("---\ndescription: Now it has one.\n---\n# Body")
        os.utime(root, ns=(root_mtime, root_mtime))
        os.utime(root / "alpha", ns=(child_mtime, child_mtime))

        assert root.stat().st_mtime_ns == root_mtime
        assert (root / "alpha").stat().st_mtime_ns == child_mtime
        assert roots_fingerprint([root]) != before

    def test_unchanged_tree_is_stable(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        _write_skill(root, "alpha")
        assert roots_fingerprint([root]) == roots_fingerprint([root])

    def test_missing_root_appearing_is_a_change(self, tmp_path: Path) -> None:
        root = tmp_path / "later"
        before = roots_fingerprint([root])
        root.mkdir()
        assert roots_fingerprint([root]) != before

    def test_never_raises_on_a_hostile_tree(self, tmp_path: Path) -> None:
        # Tolerance must match the scanner's: a symlink loop or a vanishing
        # entry yields a shorter tuple, never an exception on the read path.
        root = tmp_path / "skills"
        root.mkdir()
        (root / "loop").symlink_to(root)
        (root / "dangling").symlink_to(tmp_path / "nowhere")
        (root / "plain.txt").write_text("not a directory")
        assert isinstance(roots_fingerprint([root]), tuple)


class TestDiagnoseMissingSkill:
    """One message per real cause, each naming its own remedy."""

    def test_directory_without_skill_md(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        (root / "alpha").mkdir(parents=True)
        message = diagnose_missing_skill("alpha", [root])
        assert message is not None
        assert "has no SKILL.md" in message

    def test_missing_description(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        _write_skill(root, "alpha", description=None)
        message = diagnose_missing_skill("alpha", [root])
        assert message is not None
        assert "no 'description'" in message
        assert "Add one and read the URL again" in message

    def test_malformed_frontmatter(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        (root / "alpha").mkdir(parents=True)
        # Opens a block and never closes it.
        (root / "alpha" / "SKILL.md").write_text("---\ndescription: Unterminated.\n\n# Body")
        message = diagnose_missing_skill("alpha", [root])
        assert message is not None
        assert "malformed YAML frontmatter" in message

    def test_disabled_skill(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        _write_skill(root, "alpha", enabled=False)
        message = diagnose_missing_skill("alpha", [root])
        assert message is not None
        assert "disabled by 'enabled: false'" in message

    def test_frontmatter_name_differs_from_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "skills"
        root.mkdir()
        _write_skill(root, "alpha", name="actual-name")
        message = diagnose_missing_skill("alpha", [root])
        assert message is not None
        # Names the URL that actually works rather than calling it an error:
        # the divergence is legal, documented behaviour.
        assert "skill://actual-name" in message
        assert "frontmatter name wins" in message

    def test_shadowed_by_an_earlier_root(self, tmp_path: Path) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        _write_skill(first, "alpha")
        _write_skill(second, "alpha")
        message = diagnose_missing_skill("alpha", [first, second])
        assert message is not None
        assert "is shadowed by the one at" in message
        assert "earlier roots win" in message

    def test_nothing_on_disk_says_nothing(self, tmp_path: Path) -> None:
        # A genuine typo has no remedy to name; the bare "Unknown skill" with
        # its available-names list is already the right answer.
        root = tmp_path / "skills"
        root.mkdir()
        assert diagnose_missing_skill("absent", [root]) is None

    def test_unquoted_colon_is_reported_as_invalid_yaml(self, tmp_path: Path) -> None:
        """R4 regression: the commonest authoring error must name its own cause.

        The malformed branch used to require the ``---`` delimiters to be
        ABSENT, so a file with both delimiters and invalid YAML fell through to
        the description branch and was told "has no 'description'" -- about a
        file that visibly has one, with a remedy that provably does not fix it.
        """
        root = tmp_path / "skills"
        (root / "lean").mkdir(parents=True)
        # An unquoted colon in the value: valid-looking to a human, invalid YAML.
        (root / "lean" / "SKILL.md").write_text(
            "---\nname: lean\ndescription: Lean 4: formalize proofs\n---\n\n# Body\n"
        )
        message = diagnose_missing_skill("lean", [root])
        assert message is not None
        assert "invalid YAML in its frontmatter" in message
        assert "Quote any value containing a colon" in message
        # The wrong remedy must be gone, not merely accompanied.
        assert "no 'description'" not in message
        # And the parser's own words, which are the only thing that locates it.
        assert "mapping values are not allowed" in message

    def test_empty_but_well_formed_block_still_reports_the_description(
        self, tmp_path: Path
    ) -> None:
        # The other side of R4: a block that PARSES to nothing is not malformed,
        # and telling that author to quote a colon would be the same class of
        # wrong answer in the opposite direction.
        root = tmp_path / "skills"
        (root / "alpha").mkdir(parents=True)
        (root / "alpha" / "SKILL.md").write_text("---\n---\n\n# Body\n")
        message = diagnose_missing_skill("alpha", [root])
        assert message is not None
        assert "no 'description'" in message
        assert "invalid YAML" not in message


class TestDiagnoseMissingSkillContainment:
    """Q1: a name is a directory entry, never a path. It may not leave the roots.

    ``root / name`` is an unguarded join and a skill NAME is a URL's netloc, so
    the resolver's path-portion guards never inspect it. Each shape below
    reached the filesystem outside every configured root.
    """

    @staticmethod
    def _outside(tmp_path: Path) -> Path:
        """A skill-shaped directory OUTSIDE the roots, to be reached or not."""
        secret = tmp_path / "outside" / "private-project"
        secret.mkdir(parents=True)
        (secret / "SKILL.md").write_text(
            "---\nname: internal-codename\ndescription: Secret.\n---\n# body\n"
        )
        return secret

    def test_parent_traversal_name_reads_nothing(self, tmp_path: Path) -> None:
        # skill://.. -- ".." parses as the netloc, so no path guard ever saw it.
        root = tmp_path / "roots" / "skills"
        root.mkdir(parents=True)
        assert diagnose_missing_skill("..", [root]) is None

    def test_absolute_name_is_not_an_existence_oracle(self, tmp_path: Path) -> None:
        # skill://%2fetc decodes to "/etc", and Path(root) / "/etc" IS "/etc":
        # the diagnostic distinguished "this absolute path exists" from "it does
        # not" for ANY path on the host.
        root = tmp_path / "roots" / "skills"
        root.mkdir(parents=True)
        existing = tmp_path / "outside"
        existing.mkdir()
        assert diagnose_missing_skill(str(existing), [root]) is None
        assert diagnose_missing_skill("/definitely/not/there/xyz", [root]) is None

    def test_relative_traversal_name_does_not_leak_frontmatter(self, tmp_path: Path) -> None:
        # The leak with teeth: an out-of-root SKILL.md's frontmatter `name`
        # came back in the "declares name '...'" message.
        secret = self._outside(tmp_path)
        root = tmp_path / "roots" / "skills"
        root.mkdir(parents=True)
        message = diagnose_missing_skill(f"../../outside/{secret.name}", [root])
        assert message is None

    def test_separator_bearing_names_never_touch_the_filesystem(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rejection happens BEFORE any stat, not merely before the message.

        A malformed URL that still spends a real probe defeats the design rule
        that unsafe input must not drive filesystem work, even when its output
        is safe.

        The DRIVE-RELATIVE shapes (``D:x``, ``a:b``) are deliberately absent
        from this list: they are rejected only on Windows, and are covered in
        both directions by :class:`TestPlainSkillNameDriveRuleIsPlatformGated`.
        """
        import local_operator.skills.discovery as discovery_module

        root = tmp_path / "roots" / "skills"
        root.mkdir(parents=True)

        def explode(*_args: object, **_kwargs: object) -> bool:
            raise AssertionError("an unsafe name must not reach the filesystem")

        monkeypatch.setattr(discovery_module.Path, "is_dir", explode)
        for name in ("..", "/etc", "a/b", "..\\..\\x", ".", ""):
            assert diagnose_missing_skill(name, [root]) is None

    def test_a_plain_name_still_diagnoses(self, tmp_path: Path) -> None:
        # The guard must not swallow the legitimate case it sits in front of.
        root = tmp_path / "roots" / "skills"
        (root / "alpha").mkdir(parents=True)
        message = diagnose_missing_skill("alpha", [root])
        assert message is not None and "has no SKILL.md" in message


class TestPlainSkillNameDriveRuleIsPlatformGated:
    """R9: the drive door shuts on Windows only, because ``:`` is legal on POSIX.

    Gating the drive rejection on every platform rejected a name the POSIX
    scanner genuinely registers, so a legitimately-named skill silently lost
    the mid-session authoring refresh this PR exists to deliver -- and its
    miss-path diagnostic with it. Both directions are asserted here rather
    than left to whichever CI leg happens to run, by driving
    ``_DRIVE_RESETS_JOIN`` (the seam) in each position.
    """

    # Every shape pathlib parses as a drive: it accepts ANY single printable
    # character as a drive letter, so the rule is not limited to ``[A-Za-z]``.
    _DRIVE_SHAPES = ("D:x", "a:b", "C:", "1:x", "#:x")

    def test_the_gate_is_derived_from_the_running_platform(self) -> None:
        """The seam every other test in this class DRIVES must itself be right.

        Patching ``_DRIVE_RESETS_JOIN`` proves what each branch does but says
        nothing about which branch a real host takes -- an inverted derivation
        would satisfy every other assertion here while denying POSIX skills
        and opening the drive door on Windows, which is precisely the bug.
        """
        assert discovery_module._DRIVE_RESETS_JOIN == (os.name == "nt")

    def test_the_gate_derives_both_ways_from_os_name(self) -> None:
        """R10/Q2: the derivation above is TAUTOLOGICAL on the host that runs it.

        ``_DRIVE_RESETS_JOIN == (os.name == "nt")`` compares the constant to the
        same expression that produced it, so on this POSIX host both sides are
        ``False`` for any module whose constant is ``False`` -- including a
        hardcoded ``False``, which is exactly the direction that reopens the
        Windows existence oracle the drive rule exists to close. (An INVERTED
        or hardcoded-``True`` derivation is caught, because those disagree with
        POSIX; only the dangerous direction survives.) No CI leg closes it
        either: ``filesystem-boundaries-windows`` runs two boundary files and
        never these tests, and it cannot be widened to run them -- three tests
        here ``mkdir`` a literal ``a:b`` directory, which Windows parses as
        drive-relative and refuses to create.

        So evaluate the derivation itself under BOTH platform names, which
        needs no Windows runner: re-execute the module source with ``os.name``
        patched and read what the constant comes out as. A constant that
        ignores ``os.name`` now fails on the ``nt`` side.
        """
        for platform_name, expected in (("nt", True), ("posix", False)):
            fresh = _discovery_under_os_name(platform_name)
            assert fresh._DRIVE_RESETS_JOIN is expected, platform_name
            # The derivation is only worth pinning because the predicate reads
            # it: prove the gate actually reaches behaviour in this same copy.
            assert fresh.is_plain_skill_name("D:x") is (not expected), platform_name
            assert fresh.is_plain_skill_name("plain-one") is True, platform_name

    def test_unpatched_predicate_matches_this_hosts_semantics(self) -> None:
        """End-to-end on the REAL host, with no seam patched at all.

        Belt and braces for the test above: on POSIX ``a:b`` must be admitted
        for real, not merely when the constant is forced.
        """
        expected = os.name != "nt"
        for name in self._DRIVE_SHAPES:
            assert is_plain_skill_name(name) is expected, name

    def test_posix_admits_drive_relative_shapes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On POSIX ``root / "a:b"`` is a contained child, so it must be admitted."""
        monkeypatch.setattr(discovery_module, "_DRIVE_RESETS_JOIN", False)
        for name in self._DRIVE_SHAPES:
            assert is_plain_skill_name(name) is True, name

    def test_windows_rejects_drive_relative_shapes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On Windows the drive resets the join, so it stays an existence oracle."""
        monkeypatch.setattr(discovery_module, "_DRIVE_RESETS_JOIN", True)
        for name in self._DRIVE_SHAPES:
            assert is_plain_skill_name(name) is False, name

    def test_root_and_separator_rejection_is_unconditional(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the DRIVE term is gated; every other rejection holds everywhere."""
        for gate in (False, True):
            monkeypatch.setattr(discovery_module, "_DRIVE_RESETS_JOIN", gate)
            for name in ("", ".", "..", "/etc", "\\x", "a/b", "a\\b", "a\x00b", "D:/x"):
                assert is_plain_skill_name(name) is False, (gate, name)
            for name in ("plain-one", "notes:2024", "alpha"):
                assert is_plain_skill_name(name) is True, (gate, name)

    def test_scanner_and_predicate_agree_on_posix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression with teeth: the real scanner DOES produce ``a:b``.

        A predicate that rejects a name ``scan_skills_dir`` registers makes the
        refresh gate disagree with discovery, which is exactly how the skill
        lost its liveness.
        """
        monkeypatch.setattr(discovery_module, "_DRIVE_RESETS_JOIN", False)
        root = tmp_path / "roots" / "skills"
        root.mkdir(parents=True)
        for dirname in ("a:b", "notes:2024", "plain-one"):
            _write_skill(root, dirname, name=dirname)
        registered = [s.name for s in scan_skills_dir(root, "test")]
        assert "a:b" in registered
        for name in registered:
            assert is_plain_skill_name(name) is True, name

    def test_posix_drive_shape_still_gets_a_diagnostic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Admitting the name must also restore the miss-path explanation."""
        monkeypatch.setattr(discovery_module, "_DRIVE_RESETS_JOIN", False)
        root = tmp_path / "roots" / "skills"
        (root / "a:b").mkdir(parents=True)
        message = diagnose_missing_skill("a:b", [root])
        assert message is not None and "has no SKILL.md" in message

    def test_windows_drive_shapes_spend_no_filesystem_work(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under Windows semantics the denial is still BEFORE any stat."""
        monkeypatch.setattr(discovery_module, "_DRIVE_RESETS_JOIN", True)
        root = tmp_path / "roots" / "skills"
        root.mkdir(parents=True)

        def explode(*_args: object, **_kwargs: object) -> bool:
            raise AssertionError("a drive-anchored name must not reach the filesystem")

        monkeypatch.setattr(discovery_module.Path, "is_dir", explode)
        for name in self._DRIVE_SHAPES:
            assert diagnose_missing_skill(name, [root]) is None, name
