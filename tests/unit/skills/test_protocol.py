"""skill:// protocol tests: happy paths, unknown-name listing, traversal
rejection, symlink defense, directory listing, truncation."""

from __future__ import annotations

from pathlib import Path

import pytest

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
        assert "(empty directory)" in resolve_skill_url(
            "skill://alpha/references", skills
        )

        # skill://alpha (no path) returns SKILL.md text, so probe '.' to list
        # the base dir: dirs get the ' (dir)' suffix, files list bare.
        root_listing = resolve_skill_url("skill://alpha/.", skills)
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

    def test_percent_encoded_absolute_path_rejected(
        self, skills: dict[str, Skill]
    ) -> None:
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
