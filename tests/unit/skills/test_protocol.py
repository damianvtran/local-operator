"""skill:// protocol tests: happy paths, unknown-name listing, traversal
rejection, symlink defense, directory listing, truncation."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.skills import protocol
from local_operator.skills.discovery import Skill
from local_operator.skills.protocol import MAX_READ_BYTES, resolve_skill_url


def _make_skill(root: Path, name: str, body: str = "# Test skill body") -> Skill:
    base_dir = root / name
    base_dir.mkdir(parents=True)
    skill_md = base_dir / "SKILL.md"
    content = f"---\ndescription: {name} does things\n---\n\n{body}\n"
    skill_md.write_text(content, encoding="utf-8")
    return Skill(
        name=name,
        description=f"{name} does things",
        file_path=skill_md,
        base_dir=base_dir,
        source=str(root),
    )


@pytest.fixture
def skills(tmp_path: Path) -> dict[str, Skill]:
    alpha = _make_skill(tmp_path, "alpha")
    beta = _make_skill(tmp_path, "beta")
    return {"alpha": alpha, "beta": beta}


class TestSchemeDispatch:
    def test_non_skill_url_returns_none(self, skills: dict[str, Skill]) -> None:
        assert resolve_skill_url("file:///etc/passwd", skills) is None
        assert resolve_skill_url("http://example.com", skills) is None
        assert resolve_skill_url("memory://notes", skills) is None

    def test_case_sensitive_scheme(self, skills: dict[str, Skill]) -> None:
        # Only lowercase skill:// is the protocol; anything else is not ours.
        assert resolve_skill_url("SKILL://alpha", skills) is None


class TestBareReads:
    def test_no_path_returns_skill_md_text(self, skills: dict[str, Skill]) -> None:
        content = resolve_skill_url("skill://alpha", skills)
        assert content is not None
        assert "# Test skill body" in content
        assert "description: alpha does things" in content

    def test_trailing_slash_also_returns_skill_md(self, skills: dict[str, Skill]) -> None:
        assert resolve_skill_url("skill://alpha/", skills) == resolve_skill_url(
            "skill://alpha", skills
        )

    def test_empty_name_raises(self, skills: dict[str, Skill]) -> None:
        with pytest.raises(ValueError, match="missing a name"):
            resolve_skill_url("skill://", skills)


class TestUnknownName:
    def test_error_lists_all_available_names(self, skills: dict[str, Skill]) -> None:
        with pytest.raises(ValueError) as excinfo:
            resolve_skill_url("skill://gamma", skills)
        message = str(excinfo.value)
        assert "Unknown skill: gamma" in message
        assert "Available: alpha, beta" in message

    def test_empty_registry_lists_none(self) -> None:
        with pytest.raises(ValueError, match=r"\(none\)"):
            resolve_skill_url("skill://anything", {})


class TestChildPaths:
    def test_file_read(self, skills: dict[str, Skill]) -> None:
        ref = skills["alpha"].base_dir / "references"
        ref.mkdir()
        (ref / "detail.md").write_text("deep detail", encoding="utf-8")
        assert resolve_skill_url("skill://alpha/references/detail.md", skills) == "deep detail"

    def test_url_encoded_segments(self, skills: dict[str, Skill]) -> None:
        (skills["alpha"].base_dir / "my file.txt").write_text("spaced", encoding="utf-8")
        assert resolve_skill_url("skill://alpha/my%20file.txt", skills) == "spaced"

    def test_directory_listing(self, skills: dict[str, Skill]) -> None:
        base = skills["alpha"].base_dir
        (base / "references").mkdir()
        (base / "scripts").mkdir()
        (base / "notes.txt").write_text("x", encoding="utf-8")

        # A dir with no visible files renders the empty marker.
        empty_listing = resolve_skill_url("skill://alpha/references", skills)
        assert empty_listing is not None
        assert "(empty directory)" in empty_listing

        # skill://alpha (no path) returns SKILL.md text, so probe '.' to list
        # the base dir: dirs get the ' (dir)' suffix, files list bare.
        root_listing = resolve_skill_url("skill://alpha/.", skills)
        assert root_listing is not None
        assert "references/ (dir)" in root_listing
        assert "scripts/ (dir)" in root_listing
        assert "notes.txt" in root_listing
        assert "SKILL.md" in root_listing

    def test_missing_path_raises_value_error(self, skills: dict[str, Skill]) -> None:
        with pytest.raises(ValueError, match="not found"):
            resolve_skill_url("skill://alpha/nope.md", skills)

    def test_binary_safe_read(self, skills: dict[str, Skill]) -> None:
        (skills["alpha"].base_dir / "blob.bin").write_bytes(b"\xff\xfe\x00data")
        content = resolve_skill_url("skill://alpha/blob.bin", skills)
        assert content is not None
        assert "data" in content  # decoded with errors='replace'


class TestTraversal:
    def test_dotdot_in_path_raises(self, skills: dict[str, Skill]) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            resolve_skill_url("skill://alpha/../../etc/passwd", skills)

    def test_deep_dotdot_raises(self, skills: dict[str, Skill]) -> None:
        with pytest.raises(ValueError):
            resolve_skill_url("skill://alpha/references/../../../secret.txt", skills)

    def test_single_dotdot_raises(self, skills: dict[str, Skill]) -> None:
        with pytest.raises(ValueError):
            resolve_skill_url("skill://alpha/..", skills)

    def test_symlink_escape_rejected(self, tmp_path: Path) -> None:
        # Skill dir contains a symlink pointing outside it; resolve() + the
        # is_relative_to re-check must catch it even though '..' never appears.
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")

        root = tmp_path / "skills"
        skill = _make_skill(root, "alpha")
        (skill.base_dir / "link.txt").symlink_to(secret)
        mapping = {"alpha": skill}

        with pytest.raises(ValueError, match="escapes"):
            resolve_skill_url("skill://alpha/link.txt", mapping)

    def test_dotdot_that_stays_inside_still_rejected(self, skills: dict[str, Skill]) -> None:
        # Even self-cancelling '..' is rejected outright — no normalization
        # games, matching the protocol contract.
        (skills["alpha"].base_dir / "sub").mkdir()
        (skills["alpha"].base_dir / "sub" / "file.txt").write_text("ok", encoding="utf-8")
        with pytest.raises(ValueError):
            resolve_skill_url("skill://alpha/sub/../sub/file.txt", skills)

    def test_percent_encoded_dotdot_rejected(self, skills: dict[str, Skill]) -> None:
        # RS-10: %2e%2e is the classic encoded bypass. It is rejected only
        # because unquote() precedes the '..' check — this test pins that.
        with pytest.raises(ValueError, match=r"\.\."):
            resolve_skill_url("skill://alpha/%2e%2e%2fsecret", skills)

    def test_percent_encoded_absolute_path_rejected(self, skills: dict[str, Skill]) -> None:
        # RS-10: %2F decodes to '/', producing an absolute path after the
        # leading URL separator is stripped.
        with pytest.raises(ValueError):
            resolve_skill_url("skill://alpha/%2Fetc%2Fpasswd", skills)

    def test_mixed_encoded_dotdot_rejected(self, skills: dict[str, Skill]) -> None:
        with pytest.raises(ValueError):
            resolve_skill_url("skill://alpha/references/%2e%2e/../etc/passwd", skills)


class TestDotfiles:
    def test_dotfile_read_rejected_with_message(self, skills: dict[str, Skill]) -> None:
        # RS-22: dotfiles are unlisted AND unreadable; the rejection message
        # says so explicitly instead of leaking "not found".
        (skills["alpha"].base_dir / ".env").write_text("SECRET=x", encoding="utf-8")
        with pytest.raises(ValueError, match="dotfiles"):
            resolve_skill_url("skill://alpha/.env", skills)

    def test_dotfile_in_subdir_rejected(self, skills: dict[str, Skill]) -> None:
        (skills["alpha"].base_dir / "references").mkdir()
        (skills["alpha"].base_dir / "references" / ".secret").write_text("x")
        with pytest.raises(ValueError, match="dotfiles"):
            resolve_skill_url("skill://alpha/references/.secret", skills)

    def test_dotfile_hidden_from_listing(self, skills: dict[str, Skill]) -> None:
        (skills["alpha"].base_dir / ".env").write_text("SECRET=x", encoding="utf-8")
        (skills["alpha"].base_dir / "notes.txt").write_text("visible", encoding="utf-8")
        listing = resolve_skill_url("skill://alpha/.", skills)
        assert listing is not None
        assert ".env" not in listing
        assert "notes.txt" in listing

    def test_encoded_dotfile_rejected(self, skills: dict[str, Skill]) -> None:
        (skills["alpha"].base_dir / ".env").write_text("SECRET=x", encoding="utf-8")
        with pytest.raises(ValueError, match="dotfiles"):
            resolve_skill_url("skill://alpha/%2Eenv", skills)


class TestListingCap:
    def test_listing_capped_at_500_with_marker(self, skills: dict[str, Skill]) -> None:
        # RS-14: an unbounded directory listing is a context-bomb vector.
        big = skills["alpha"].base_dir / "big"
        big.mkdir()
        for i in range(505):
            (big / f"file-{i:04d}.txt").write_text("x", encoding="utf-8")
        listing = resolve_skill_url("skill://alpha/big", skills)
        assert listing is not None
        lines = listing.splitlines()
        assert len(lines) == 501  # 500 entries + the overflow marker
        assert "more entries not shown" in lines[-1]
        assert "5" in lines[-1]  # the 5 overflowed entries are named

    def test_under_cap_has_no_marker(self, skills: dict[str, Skill]) -> None:
        small = skills["alpha"].base_dir / "small"
        small.mkdir()
        for i in range(3):
            (small / f"file-{i}.txt").write_text("x", encoding="utf-8")
        listing = resolve_skill_url("skill://alpha/small", skills)
        assert listing is not None
        assert "not shown" not in listing
        assert len(listing.splitlines()) == 3


class TestReadCapBoundsTheRead:
    def test_cap_bounds_read_not_just_output(
        self, skills: dict[str, Skill], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # RS-02: the 200KB cap must limit the READ. Prove the file handle
        # never reads past MAX_READ_BYTES + 1 by wrapping read().
        from local_operator.skills import protocol

        big = skills["alpha"].base_dir / "big.txt"
        big.write_bytes(b"z" * (MAX_READ_BYTES + 100_000))

        reads: list[int] = []
        real_open = Path.open

        def spy_open(self, *args, **kwargs):
            fh = real_open(self, *args, **kwargs)
            real_read = fh.read

            def read(n=-1):
                reads.append(n)
                return real_read(n)

            fh.read = read
            return fh

        monkeypatch.setattr(Path, "open", spy_open)
        content = resolve_skill_url("skill://alpha/big.txt", skills)
        assert content is not None
        assert "truncated" in content
        assert reads and max(reads) == MAX_READ_BYTES + 1
        assert protocol.MAX_READ_BYTES == 200 * 1024


class TestTruncation:
    def test_large_file_truncated_with_marker(self, skills: dict[str, Skill]) -> None:
        big = skills["alpha"].base_dir / "big.txt"
        big.write_bytes(b"x" * (MAX_READ_BYTES + 1024))
        content = resolve_skill_url("skill://alpha/big.txt", skills)
        assert content is not None
        assert len(content) <= MAX_READ_BYTES + 200  # payload + small marker
        assert "truncated" in content

    def test_at_limit_file_not_truncated(self, skills: dict[str, Skill]) -> None:
        exact = skills["alpha"].base_dir / "exact.txt"
        exact.write_bytes(b"y" * MAX_READ_BYTES)
        content = resolve_skill_url("skill://alpha/exact.txt", skills)
        assert content is not None
        assert content == "y" * MAX_READ_BYTES


class TestReferenceListing:
    """A bare read discloses the skill's reference files as PROTOCOL paths.

    The gap this closes: after reading `skill://<name>`, a model that saw
    references mentioned in the body used to guess raw filesystem paths
    (wrong on any machine with a different skills root). The bare read now
    ends with the exact `skill://<name>/<relpath>` form for every reference,
    so the protocol read is the discoverable next step.
    """

    def test_bare_read_lists_references(self, skills: dict[str, Skill]) -> None:
        base = skills["alpha"].base_dir
        refs = base / "references"
        refs.mkdir()
        (refs / "admin-api.md").write_text("# Admin API", encoding="utf-8")
        (base / "NOTES.md").write_text("notes", encoding="utf-8")
        content = resolve_skill_url("skill://alpha", skills)
        assert content is not None
        # Body still present, listing appended after it.
        assert "# Test skill body" in content
        assert "skill://alpha/<path>" in content
        assert "references/admin-api.md" in content
        assert "NOTES.md" in content
        assert content.index("# Test skill body") < content.index("references/admin-api.md")

    def test_listed_paths_resolve_via_protocol(self, skills: dict[str, Skill]) -> None:
        # The listing must never advertise a path the resolver then rejects.
        base = skills["alpha"].base_dir
        refs = base / "references"
        refs.mkdir()
        (refs / "guide.md").write_text("ref body", encoding="utf-8")
        content = resolve_skill_url("skill://alpha", skills)
        assert content is not None and "references/guide.md" in content
        assert resolve_skill_url("skill://alpha/references/guide.md", skills) == "ref body"

    def test_body_only_skill_is_byte_identical(self, skills: dict[str, Skill]) -> None:
        # No reference files -> no listing, no trailing separator: the
        # pre-listing output is preserved byte for byte.
        body = skills["alpha"].file_path.read_text(encoding="utf-8")
        assert resolve_skill_url("skill://alpha", skills) == body

    def test_skill_md_itself_not_listed(self, skills: dict[str, Skill]) -> None:
        base = skills["alpha"].base_dir
        (base / "extra.md").write_text("x", encoding="utf-8")
        content = resolve_skill_url("skill://alpha", skills)
        assert content is not None
        # Anchor on the listing header, not on "---" (frontmatter and
        # horizontal rules use the same delimiter).
        listing = content.split("Reference files (read with")[-1]
        assert "SKILL.md" not in listing
        assert "extra.md" in listing

    def test_dotfiles_not_listed(self, skills: dict[str, Skill]) -> None:
        base = skills["alpha"].base_dir
        (base / ".secret").write_text("hidden", encoding="utf-8")
        hidden_dir = base / ".git"
        hidden_dir.mkdir()
        (hidden_dir / "config").write_text("repo", encoding="utf-8")
        (base / "visible.md").write_text("v", encoding="utf-8")
        content = resolve_skill_url("skill://alpha", skills)
        assert content is not None
        assert ".secret" not in content
        assert ".git" not in content
        assert "visible.md" in content

    def test_escaping_symlinked_directory_not_followed(self, tmp_path: Path) -> None:
        # A symlink pointing outside base_dir must not leak foreign paths
        # into the listing (the resolver would reject reading them anyway).
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.md").write_text("leak", encoding="utf-8")
        skill = _make_skill(tmp_path, "linky")
        (skill.base_dir / "escape").symlink_to(outside, target_is_directory=True)
        content = resolve_skill_url("skill://linky", {"linky": skill})
        assert content is not None
        assert "leak.md" not in content

    def test_escaping_symlinked_file_not_listed(self, tmp_path: Path) -> None:
        # R1-1: a symlinked FILE whose target escapes base_dir passes
        # is_file() (which follows links) but _resolve_child rejects reading
        # it — so listing it would advertise a guaranteed failure. The
        # listing applies the resolver's own containment check instead.
        outside = tmp_path / "outside-file.md"
        outside.write_text("secret", encoding="utf-8")
        skill = _make_skill(tmp_path, "filelink")
        (skill.base_dir / "escape.md").symlink_to(outside)
        content = resolve_skill_url("skill://filelink", {"filelink": skill})
        assert content is not None
        assert "escape.md" not in content

    def test_internal_symlinks_listed_and_followed(self, tmp_path: Path) -> None:
        # R1-3: symlinks resolving INSIDE base_dir are readable by the
        # resolver, so hiding them would be the reverse parity gap. Both a
        # linked file and a linked directory must appear.
        skill = _make_skill(tmp_path, "inlink")
        base = skill.base_dir
        (base / "real.md").write_text("real", encoding="utf-8")
        (base / "alias.md").symlink_to(base / "real.md")
        subdir = base / "docs"
        subdir.mkdir()
        (subdir / "inner.md").write_text("inner", encoding="utf-8")
        (base / "docs-link").symlink_to(subdir, target_is_directory=True)
        content = resolve_skill_url("skill://inlink", {"inlink": skill})
        assert content is not None
        assert "alias.md" in content
        assert "docs/inner.md" in content
        assert resolve_skill_url("skill://inlink/alias.md", {"inlink": skill}) == "real"

    def test_symlink_cycle_terminates(self, tmp_path: Path) -> None:
        # An internal directory link cycle must not hang or duplicate the
        # walk: resolved directories are visited at most once.
        skill = _make_skill(tmp_path, "cycle")
        base = skill.base_dir
        subdir = base / "docs"
        subdir.mkdir()
        (subdir / "page.md").write_text("p", encoding="utf-8")
        (subdir / "loop").symlink_to(base, target_is_directory=True)
        content = resolve_skill_url("skill://cycle", {"cycle": skill})
        assert content is not None
        assert content.count("docs/page.md") == 1

    def test_special_characters_round_trip(self, skills: dict[str, Skill]) -> None:
        # R1-2: a filename with '#' or '%' must be listed in a form the URL
        # parser survives — urlsplit treats a raw '#' as a fragment and
        # unquote mangles a raw '%'. The listing percent-encodes, and the
        # encoded path must actually resolve.
        base = skills["alpha"].base_dir
        (base / "notes #1.md").write_text("hash", encoding="utf-8")
        (base / "100%.md").write_text("percent", encoding="utf-8")
        content = resolve_skill_url("skill://alpha", skills)
        assert content is not None
        listing = content.split("Reference files (read with")[-1]
        listed = [line for line in listing.splitlines() if line.endswith(".md")]
        assert "notes%20%231.md" in listed
        assert "100%25.md" in listed
        assert resolve_skill_url("skill://alpha/notes%20%231.md", skills) == "hash"
        assert resolve_skill_url("skill://alpha/100%25.md", skills) == "percent"

    def test_directory_listing_names_round_trip(self, skills: dict[str, Skill]) -> None:
        # R2-2: the child-path DIRECTORY listing is where the overflow marker
        # sends the model, so its names must survive the same URL round-trip
        # as the bare-read listing — a raw '#' name copied from it would be
        # cut at the fragment and fail to resolve.
        refs = skills["alpha"].base_dir / "references"
        refs.mkdir()
        (refs / "notes #1.md").write_text("hash", encoding="utf-8")
        listing = resolve_skill_url("skill://alpha/references", skills)
        assert listing is not None
        assert "notes%20%231.md" in listing
        assert resolve_skill_url("skill://alpha/references/notes%20%231.md", skills) == "hash"

    def test_symlink_alias_of_body_file_not_listed(self, tmp_path: Path) -> None:
        # R2-3: a symlink whose resolved target IS the body file re-offers
        # content the caller just returned; exclude it like the body file
        # itself (resolved-target equality, not path equality).
        skill = _make_skill(tmp_path, "bodylink")
        (skill.base_dir / "readme.md").symlink_to(skill.file_path)
        (skill.base_dir / "real-ref.md").write_text("r", encoding="utf-8")
        content = resolve_skill_url("skill://bodylink", {"bodylink": skill})
        assert content is not None
        assert "readme.md" not in content
        assert "real-ref.md" in content

    def test_listing_bounded_with_overflow_marker(self, skills: dict[str, Skill]) -> None:
        base = skills["alpha"].base_dir
        refs = base / "references"
        refs.mkdir()
        for i in range(protocol._MAX_REFERENCE_ENTRIES + 20):
            (refs / f"ref-{i:04d}.md").write_text("r", encoding="utf-8")
        content = resolve_skill_url("skill://alpha", skills)
        assert content is not None
        listed = [line for line in content.splitlines() if line.startswith("references/ref-")]
        assert len(listed) == protocol._MAX_REFERENCE_ENTRIES
        assert "more files not shown" in content

    def test_depth_capped(self, skills: dict[str, Skill]) -> None:
        base = skills["alpha"].base_dir
        deep = base / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (base / "a" / "shallow.md").write_text("s", encoding="utf-8")
        (deep / "too-deep.md").write_text("d", encoding="utf-8")
        content = resolve_skill_url("skill://alpha", skills)
        assert content is not None
        assert "a/shallow.md" in content
        assert "too-deep.md" not in content

    def test_child_path_reads_unchanged(self, skills: dict[str, Skill]) -> None:
        # The listing rides only the BARE read; a child read returns the file
        # alone, and a directory read returns the flat listing as before.
        base = skills["alpha"].base_dir
        refs = base / "references"
        refs.mkdir()
        (refs / "one.md").write_text("one", encoding="utf-8")
        assert resolve_skill_url("skill://alpha/references/one.md", skills) == "one"
        assert resolve_skill_url("skill://alpha/references", skills) == "one.md"


class TestDirectoryListingContainment:
    """R3-2: the child-path listing must agree with the resolver on symlinks.

    ``_list_directory`` is where the bare-read overflow marker sends the
    model, so a name it prints is an invitation to read. Printing one whose
    resolved target escapes ``base_dir`` is the advertise-then-reject failure
    mode R1-1 fixed on the bare-read listing, reproduced on this second
    surface.
    """

    def test_escaping_symlinked_file_not_listed(self, tmp_path: Path) -> None:
        skill = _make_skill(tmp_path, "gamma")
        refs = skill.base_dir / "references"
        refs.mkdir()
        (refs / "real.md").write_text("real", encoding="utf-8")
        outside = tmp_path / "outside.md"
        outside.write_text("SECRET", encoding="utf-8")
        (refs / "escape.md").symlink_to(outside)
        skills = {"gamma": skill}

        listing = resolve_skill_url("skill://gamma/references", skills)
        assert listing is not None
        assert listing.splitlines() == ["real.md"]
        # And the resolver still refuses the path, so nothing became readable.
        with pytest.raises(ValueError, match="escapes"):
            resolve_skill_url("skill://gamma/references/escape.md", skills)

    def test_escaping_symlinked_directory_not_listed(self, tmp_path: Path) -> None:
        skill = _make_skill(tmp_path, "delta")
        refs = skill.base_dir / "references"
        refs.mkdir()
        (refs / "real.md").write_text("real", encoding="utf-8")
        outside = tmp_path / "outside-dir"
        outside.mkdir()
        (outside / "leak.md").write_text("SECRET", encoding="utf-8")
        (refs / "escape").symlink_to(outside, target_is_directory=True)
        skills = {"delta": skill}

        listing = resolve_skill_url("skill://delta/references", skills)
        assert listing is not None
        assert listing.splitlines() == ["real.md"]
        with pytest.raises(ValueError, match="escapes"):
            resolve_skill_url("skill://delta/references/escape", skills)

    def test_internal_symlink_still_listed(self, tmp_path: Path) -> None:
        # The containment check must not cost the readable case: a symlink
        # resolving INSIDE base_dir stays listed and stays readable.
        skill = _make_skill(tmp_path, "epsilon")
        refs = skill.base_dir / "references"
        refs.mkdir()
        (refs / "real.md").write_text("real", encoding="utf-8")
        (refs / "alias.md").symlink_to(refs / "real.md")
        skills = {"epsilon": skill}

        listing = resolve_skill_url("skill://epsilon/references", skills)
        assert listing is not None
        assert listing.splitlines() == ["alias.md", "real.md"]
        assert resolve_skill_url("skill://epsilon/references/alias.md", skills) == "real"

    def test_dangling_symlink_not_listed(self, tmp_path: Path) -> None:
        # Review F1: the same advertise-then-reject class as R3-2, reached
        # through EXISTENCE rather than containment. `resolve()` is
        # non-strict, so a link to a missing sibling resolves INSIDE base_dir
        # and passes the containment check, then fails the resolver's
        # `exists()`.
        skill = _make_skill(tmp_path, "eta")
        refs = skill.base_dir / "references"
        refs.mkdir()
        (refs / "real.md").write_text("real", encoding="utf-8")
        (refs / "dangling.md").symlink_to(refs / "never-created.md")
        skills = {"eta": skill}

        listing = resolve_skill_url("skill://eta/references", skills)
        assert listing is not None
        assert listing.splitlines() == ["real.md"]
        with pytest.raises(ValueError, match="not found"):
            resolve_skill_url("skill://eta/references/dangling.md", skills)

    def test_looping_symlinks_not_listed(self, tmp_path: Path) -> None:
        # A self-loop and a mutual loop both resolve to themselves — inside
        # base_dir, so containment passes — and are ELOOP on read.
        skill = _make_skill(tmp_path, "theta")
        refs = skill.base_dir / "references"
        refs.mkdir()
        (refs / "real.md").write_text("real", encoding="utf-8")
        (refs / "selfloop.md").symlink_to(refs / "selfloop.md")
        (refs / "loop-a.md").symlink_to(refs / "loop-b.md")
        (refs / "loop-b.md").symlink_to(refs / "loop-a.md")
        skills = {"theta": skill}

        listing = resolve_skill_url("skill://theta/references", skills)
        assert listing is not None
        assert listing.splitlines() == ["real.md"]
        for name in ("selfloop.md", "loop-a.md", "loop-b.md"):
            with pytest.raises(ValueError):
                resolve_skill_url(f"skill://theta/references/{name}", skills)

    def test_live_symlinks_survive_the_existence_check(self, tmp_path: Path) -> None:
        # The existence check must not cost a working link. A file alias and
        # a directory alias both resolve, exist, and stay readable.
        skill = _make_skill(tmp_path, "iota")
        refs = skill.base_dir / "references"
        refs.mkdir()
        (refs / "real.md").write_text("real", encoding="utf-8")
        (refs / "alias.md").symlink_to(refs / "real.md")
        (refs / "sub").mkdir()
        (refs / "sub" / "deep.md").write_text("deep", encoding="utf-8")
        (refs / "dirlink").symlink_to(refs / "sub", target_is_directory=True)
        skills = {"iota": skill}

        listing = resolve_skill_url("skill://iota/references", skills)
        assert listing is not None
        assert listing.splitlines() == [
            "alias.md",
            "dirlink/ (dir)",
            "real.md",
            "sub/ (dir)",
        ]
        assert resolve_skill_url("skill://iota/references/alias.md", skills) == "real"
        assert resolve_skill_url("skill://iota/references/dirlink", skills) == "deep.md"

    def test_every_listed_name_reads(self, tmp_path: Path) -> None:
        # The invariant behind R3-2 and F1 stated directly, over every shape
        # the resolver can refuse: whatever the listing prints, a read of it
        # succeeds. Guards against a future third refusal ground being added
        # to `_resolve_child` without the listing learning about it.
        skill = _make_skill(tmp_path, "kappa")
        refs = skill.base_dir / "references"
        refs.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("SECRET", encoding="utf-8")
        outside_dir = tmp_path / "outside-dir"
        outside_dir.mkdir()
        (refs / "real.md").write_text("real", encoding="utf-8")
        (refs / "alias.md").symlink_to(refs / "real.md")
        (refs / "escape.md").symlink_to(outside)
        (refs / "escape-dir").symlink_to(outside_dir, target_is_directory=True)
        (refs / "dangling.md").symlink_to(refs / "gone.md")
        (refs / "selfloop.md").symlink_to(refs / "selfloop.md")
        skills = {"kappa": skill}

        listing = resolve_skill_url("skill://kappa/references", skills)
        assert listing is not None
        names = [line.removesuffix("/ (dir)") for line in listing.splitlines()]
        assert names == ["alias.md", "real.md"]
        for name in names:
            assert resolve_skill_url(f"skill://kappa/references/{name}", skills) is not None

    def test_escaped_entries_do_not_inflate_the_overflow_count(self, tmp_path: Path) -> None:
        # The marker promises entries the caller can still reach. Counting an
        # unlistable symlink in "N more entries not shown" would promise a
        # name that does not exist for any follow-up read.
        skill = _make_skill(tmp_path, "zeta")
        refs = skill.base_dir / "references"
        refs.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("SECRET", encoding="utf-8")
        for i in range(protocol._MAX_LISTING_ENTRIES + 5):
            (refs / f"ref-{i:04d}.md").write_text("r", encoding="utf-8")
        (refs / "zzz-escape.md").symlink_to(outside)
        skills = {"zeta": skill}

        listing = resolve_skill_url("skill://zeta/references", skills)
        assert listing is not None
        lines = listing.splitlines()
        assert "zzz-escape.md" not in lines
        # 5 overflowed real files, not 6: the escaping symlink is not one the
        # caller could reach by listing this directory again.
        assert lines[-1] == "[... 5 more entries not shown ...]"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
