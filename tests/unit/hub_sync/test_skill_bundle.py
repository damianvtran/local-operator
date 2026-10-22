"""The bundle carrier: deterministic build, semantic validation, safe extraction.

Every malformed archive below is BUILT HERE -- by ``zipfile`` where the defect is
an entry fact, by byte surgery where the defect is a container/header fact -- so a
refusal test names the exact rule it exercises. The valid tree always comes from
``build_bundle`` so the pinned headers and the tree-hash vector describe the real
format rather than a hand-written approximation.

The refusal tests assert the NAMED step, not just "it raised": the step is what
the CLI renders and what the shared Go/Python corpus compares (design §10.1).
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from local_operator.hub_sync.skill_bundle import (
    ENTRYPOINT,
    MAX_ARCHIVE_BYTES,
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_SKILL_MD_BYTES,
    SkillBundleError,
    SkillBundleManifest,
    SkillBundleRef,
    build_bundle,
    bundle_manifest_from_directory,
    extract_archive,
    tree_hash_of,
    validate_archive,
    validate_bundle_path,
)
from local_operator.skills.discovery import scan_skills_dir

NAME = "pdf-forms"
DESCRIPTION = "Fill and validate PDF forms."
SKILL_MD = f"---\nname: {NAME}\ndescription: {DESCRIPTION}\n---\n\n# Body\n"

REGULAR = 0o100644
EXECUTABLE = 0o100755
SYMLINK = 0o120777
DIRECTORY = 0o040755


def _tree(root: Path, *, body: str = "# Body\n", name: str = NAME) -> Path:
    """One minimal valid skill directory."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {DESCRIPTION}\n---\n\n{body}", encoding="utf-8"
    )
    (root / "references").mkdir(exist_ok=True)
    (root / "references" / "guide.md").write_text("alpha\n" * 50, encoding="utf-8")
    (root / "assets").mkdir(exist_ok=True)
    (root / "assets" / "token.bin").write_bytes(bytes(range(256)))
    return root


def _zip(
    path: Path,
    entries: list[tuple[str, bytes, int]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    comment: bytes = b"",
) -> Path:
    """A zip with explicit per-entry mode bytes (the validator reads these)."""

    with zipfile.ZipFile(path, "w", compression=compression, compresslevel=6) as archive:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, data, compress_type=compression, compresslevel=6)
        if comment:
            archive.comment = comment
    return path


def _md_bytes(name: str = NAME, description: str = DESCRIPTION) -> bytes:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n# Body\n".encode()


def _valid(tmp_path: Path) -> tuple[Path, object]:
    """A built bundle on disk plus its ``BuiltBundle``."""

    built = build_bundle(_tree(tmp_path / "source"))
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(built.data)
    return archive, built


def _patch(path: Path, offset: int, payload: bytes) -> None:
    raw = bytearray(path.read_bytes())
    raw[offset : offset + len(payload)] = payload
    path.write_bytes(bytes(raw))


def _eocd_offset(path: Path) -> int:
    return path.stat().st_size - 22


def _central_offset(path: Path) -> int:
    raw = path.read_bytes()
    start = _eocd_offset(path)
    return int.from_bytes(raw[start + 16 : start + 20], "little")


# -- building ------------------------------------------------------------------------


def test_two_builds_of_the_same_tree_are_byte_identical(tmp_path: Path) -> None:
    source = _tree(tmp_path / "source")
    first = build_bundle(source)
    second = build_bundle(source)
    assert first.data == second.data
    assert first.archive_hash == second.archive_hash
    assert first.tree_hash == second.tree_hash
    assert first.file_count == 3
    assert first.uncompressed_size == len(SKILL_MD.encode()) + 50 * len("alpha\n") + 256


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_the_archive_headers_are_the_documented_canonical_form(tmp_path: Path) -> None:
    built = build_bundle(_tree(tmp_path / "source"))
    with zipfile.ZipFile(Path(_write(tmp_path / "b.zip", built.data))) as archive:
        infos = archive.infolist()
    assert [info.filename for info in infos] == [
        ENTRYPOINT,
        "assets/token.bin",
        "references/guide.md",
    ]
    for info in infos:
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.create_system == 3
        assert info.external_attr == REGULAR << 16
        assert info.compress_type == zipfile.ZIP_DEFLATED
        assert info.extra == b"" and info.comment == b""
        assert info.flag_bits in (0, 0x0800)


def test_dot_entries_are_skipped_not_refused(tmp_path: Path) -> None:
    source = _tree(tmp_path / "source")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("hidden", encoding="utf-8")
    (source / ".DS_Store").write_bytes(b"junk")
    built = build_bundle(source)
    assert built.file_count == 3
    assert b".git" not in built.data and b".DS_Store" not in built.data


@pytest.mark.parametrize(
    "defect",
    ["symlink", "hardlink", "executable", "setuid", "fifo"],
)
def test_the_builder_refuses_an_unpublishable_source_entry(tmp_path: Path, defect: str) -> None:
    source = _tree(tmp_path / "source")
    target = source / "assets" / "payload"
    if defect == "symlink":
        target.symlink_to(source / "references" / "guide.md")
    elif defect == "hardlink":
        os.link(source / "references" / "guide.md", target)
    elif defect == "executable":
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        target.chmod(0o755)
    elif defect == "setuid":
        target.write_text("binary-ish\n", encoding="utf-8")
        target.chmod(0o4755)
    else:
        os.mkfifo(target)
    with pytest.raises(SkillBundleError) as caught:
        build_bundle(source)
    assert caught.value.code == "bundle-invalid" and caught.value.step == "source"
    assert "assets" in str(caught.value) or "payload" in str(caught.value)


def test_the_builder_requires_exactly_one_root_skill_md(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "references").mkdir()
    (source / "references" / "guide.md").write_text("x", encoding="utf-8")
    with pytest.raises(SkillBundleError) as caught:
        build_bundle(source)
    assert (caught.value.code, caught.value.step) == ("bundle-invalid", "entrypoint")


def test_the_builder_refuses_missing_frontmatter(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")
    with pytest.raises(SkillBundleError) as caught:
        build_bundle(source)
    assert (caught.value.code, caught.value.step) == ("bundle-invalid", "entrypoint")


def test_the_builder_refuses_more_files_than_the_carrier_bounds(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    for index in range(MAX_FILES):
        (source / f"f{index:04d}.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SkillBundleError) as caught:
        build_bundle(source)
    assert caught.value.code == "bundle-too-large" and caught.value.step == "entry"


def test_tree_hash_is_the_documented_vector(tmp_path: Path) -> None:
    """Re-derive the digest from the SPEC, not from the implementation."""

    source = _tree(tmp_path / "source")
    files = sorted(
        (
            (path.relative_to(source).as_posix(), path.read_bytes())
            for path in source.rglob("*")
            if path.is_file()
        ),
        key=lambda row: row[0].encode("utf-8"),
    )
    payload = "skill-tree-v1\n"
    for relative, content in files:
        digest = hashlib.sha256(content).hexdigest()
        payload += f"{relative}\0{len(content)}\0{digest}\n"
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert tree_hash_of(source) == expected
    assert build_bundle(source).tree_hash == expected


def test_tree_hash_ignores_modes_and_timestamps(tmp_path: Path) -> None:
    source = _tree(tmp_path / "source")
    before = tree_hash_of(source)
    for path in source.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
            os.utime(path, (1, 1))
    assert tree_hash_of(source) == before


def test_the_directory_manifest_carries_the_built_hashes(tmp_path: Path) -> None:
    source = _tree(tmp_path / "source")
    manifest = bundle_manifest_from_directory(source)
    built = build_bundle(source)
    assert manifest.name == NAME and manifest.description == DESCRIPTION
    assert manifest.bundle.tree_hash == built.tree_hash
    assert manifest.fingerprint == built.tree_hash
    assert manifest.bundle.hash == built.archive_hash


# -- manifest JSON ---------------------------------------------------------------------


def test_the_manifest_json_is_a_closed_strict_set(tmp_path: Path) -> None:
    built = build_bundle(_tree(tmp_path / "source"))
    manifest = built.manifest
    assert set(manifest.to_json()) == {"document_version", "name", "description", "bundle"}
    assert SkillBundleManifest.from_json(manifest.to_json()) == manifest
    with pytest.raises(SkillBundleError):
        SkillBundleManifest.from_json({**manifest.to_json(), "files": []})
    with pytest.raises(SkillBundleError):
        SkillBundleManifest.from_json({**manifest.to_json(), "unknown": 1})


def test_from_remote_ignores_the_detail_envelope(tmp_path: Path) -> None:
    built = build_bundle(_tree(tmp_path / "source"))
    payload = dict(built.manifest.to_json())
    payload.update({"id": "sk-1", "revision": '"rev"', "sync_contract": {"skill_bundles": 1}})
    assert SkillBundleManifest.from_remote(payload) == built.manifest


def test_the_manifest_is_strict_but_version_is_optional(tmp_path: Path) -> None:
    built = build_bundle(_tree(tmp_path / "source"))
    payload = dict(built.manifest.to_json())
    assert SkillBundleManifest.from_json(payload) == built.manifest
    payload["version"] = "1.2.0"
    assert SkillBundleManifest.from_json(payload).version == "1.2.0"
    # Closed key sets, at both levels: an unknown key is refused, never ignored
    # (the server's 422 equivalent), and a bundle sub-object key is not exempt.
    payload["extra"] = True
    with pytest.raises(SkillBundleError):
        SkillBundleManifest.from_json(payload)
    nested = dict(built.manifest.to_json())
    nested["bundle"] = {**nested["bundle"], "mode": "0755"}
    with pytest.raises(SkillBundleError):
        SkillBundleManifest.from_json(nested)
    mixed = dict(built.manifest.to_json())
    mixed["files"] = []
    with pytest.raises(SkillBundleError):
        SkillBundleManifest.from_json(mixed)


def test_the_ref_and_manifest_bounds_are_refused_at_parse(tmp_path: Path) -> None:
    built = build_bundle(_tree(tmp_path / "source"))
    with pytest.raises(SkillBundleError) as caught:
        replace(built.bundle, size=MAX_ARCHIVE_BYTES + 1)
    assert caught.value.code == "bundle-too-large"
    with pytest.raises(SkillBundleError):
        replace(built.bundle, entrypoint="README.md")
    with pytest.raises(SkillBundleError):
        replace(built.bundle, hash=built.bundle.hash.upper())


# -- validation ------------------------------------------------------------------------


def _bad_archive(tmp_path: Path, case: str) -> Path:
    """Build one malformed archive; each case exercises exactly one refusal."""

    md = _md_bytes()

    def archive(entries: list[tuple[str, bytes, int]], **kwargs: object) -> Path:
        return _zip(tmp_path / f"{case}.zip", entries, **kwargs)  # type: ignore[arg-type]

    if case == "traversal":
        return archive([(ENTRYPOINT, md, REGULAR), ("../x", b"boom", REGULAR)])
    if case == "absolute":
        return archive([(ENTRYPOINT, md, REGULAR), ("/x", b"boom", REGULAR)])
    if case == "drive":
        return archive([(ENTRYPOINT, md, REGULAR), ("C:\\x", b"boom", REGULAR)])
    if case == "backslash":
        return archive([(ENTRYPOINT, md, REGULAR), ("assets\\x.bin", b"boom", REGULAR)])
    if case == "hidden":
        return archive([(ENTRYPOINT, md, REGULAR), (".git/config", b"boom", REGULAR)])
    if case == "symlink":
        return archive([(ENTRYPOINT, md, REGULAR), ("assets/link", b"/etc/passwd", SYMLINK)])
    if case == "executable":
        return archive([(ENTRYPOINT, md, REGULAR), ("scripts/run.sh", b"#!/bin/sh\n", EXECUTABLE)])
    if case == "directory-entry":
        return archive([(ENTRYPOINT, md, REGULAR), ("assets/", b"", DIRECTORY)])
    if case == "device":
        return archive([(ENTRYPOINT, md, REGULAR), ("assets/dev", b"", 0o060644)])
    if case == "setuid":
        return archive([(ENTRYPOINT, md, REGULAR), ("assets/suid", b"x", 0o104755)])
    if case == "duplicate-casefold":
        return archive(
            [
                (ENTRYPOINT, md, REGULAR),
                ("references/A.md", b"one", REGULAR),
                ("references/a.md", b"two", REGULAR),
            ]
        )
    if case == "file-vs-ancestor":
        return archive(
            [(ENTRYPOINT, md, REGULAR), ("assets", b"file", REGULAR), ("assets/x", b"x", REGULAR)]
        )
    if case == "non-nfc":
        return archive([(ENTRYPOINT, md, REGULAR), ("assets/cafe\u0301.bin", b"x", REGULAR)])
    if case == "missing-skill-md":
        return archive([("references/guide.md", b"x", REGULAR)])
    if case == "wrong-frontmatter":
        return archive([(ENTRYPOINT, "---\ntitle: nope\n---\n".encode(), REGULAR)])
    if case == "non-utf8-skill-md":
        return archive([(ENTRYPOINT, b"---\nname: x\n---\n\xff\xfe", REGULAR)])
    if case == "oversize-skill-md":
        return archive(
            [
                (
                    ENTRYPOINT,
                    b"---\nname: x\ndescription: d\n---\n" + b"x" * MAX_SKILL_MD_BYTES,
                    REGULAR,
                )
            ]
        )
    if case == "too-many-entries":
        entries = [(ENTRYPOINT, md, REGULAR)]
        entries.extend((f"f{index:04d}.txt", b"x", REGULAR) for index in range(MAX_FILES))
        return archive(entries)
    if case == "archive-comment":
        return archive([(ENTRYPOINT, md, REGULAR)], comment=b"note")
    if case == "trailing-bytes":
        path = archive([(ENTRYPOINT, md, REGULAR)])
        path.write_bytes(path.read_bytes() + b"JUNK")
        return path
    if case == "prepended-bytes":
        path = archive([(ENTRYPOINT, md, REGULAR)])
        path.write_bytes(b"JUNK" + path.read_bytes())
        return path
    if case == "zip64-marker":
        path = archive([(ENTRYPOINT, md, REGULAR)])
        _patch(path, _eocd_offset(path) + 10, b"\xff\xff")
        return path
    if case == "encrypted":
        path = archive([(ENTRYPOINT, md, REGULAR)])
        _patch(path, _central_offset(path) + 8, b"\x01\x00")
        return path
    if case == "data-descriptor":
        path = archive([(ENTRYPOINT, md, REGULAR)])
        _patch(path, _central_offset(path) + 8, b"\x08\x00")
        return path
    if case == "bzip2":
        return archive([(ENTRYPOINT, md, REGULAR)], compression=zipfile.ZIP_BZIP2)
    if case == "crc":
        path = archive([(ENTRYPOINT, md, REGULAR)])
        _patch(path, 30 + len(ENTRYPOINT) + 4, b"\x00")
        return path
    if case == "oversize-file":
        path = tmp_path / "oversize-file.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive_handle:
            info = zipfile.ZipInfo(ENTRYPOINT, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system, info.external_attr = 3, REGULAR << 16
            archive_handle.writestr(info, md)
            big = zipfile.ZipInfo("assets/big.bin", date_time=(1980, 1, 1, 0, 0, 0))
            big.create_system, big.external_attr = 3, REGULAR << 16
            with archive_handle.open(big, "w") as stream:
                for _ in range(MAX_FILE_BYTES // (1024 * 1024) + 1):
                    stream.write(b"\0" * 1024 * 1024)
        return path
    if case == "oversize-archive":
        path = tmp_path / "oversize-archive.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive_handle:
            info = zipfile.ZipInfo("assets/big.bin", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system, info.external_attr = 3, REGULAR << 16
            with archive_handle.open(info, "w") as stream:
                for _ in range(MAX_ARCHIVE_BYTES // (1024 * 1024) + 1):
                    stream.write(b"\0" * 1024 * 1024)
        return path
    raise AssertionError(f"unknown case {case}")


@pytest.mark.parametrize(
    ("case", "code", "step"),
    [
        ("traversal", "bundle-invalid", "entry"),
        ("absolute", "bundle-invalid", "entry"),
        ("drive", "bundle-invalid", "entry"),
        ("backslash", "bundle-invalid", "entry"),
        ("hidden", "bundle-invalid", "entry"),
        ("symlink", "bundle-invalid", "entry"),
        ("executable", "bundle-invalid", "entry"),
        ("directory-entry", "bundle-invalid", "entry"),
        ("device", "bundle-invalid", "entry"),
        ("setuid", "bundle-invalid", "entry"),
        ("duplicate-casefold", "bundle-invalid", "entry"),
        ("file-vs-ancestor", "bundle-invalid", "entry"),
        ("non-nfc", "bundle-invalid", "entry"),
        ("missing-skill-md", "bundle-invalid", "entrypoint"),
        ("wrong-frontmatter", "bundle-invalid", "entrypoint"),
        ("non-utf8-skill-md", "bundle-invalid", "entrypoint"),
        ("oversize-skill-md", "bundle-too-large", "entrypoint"),
        ("too-many-entries", "bundle-too-large", "entry"),
        ("archive-comment", "bundle-invalid", "container"),
        ("trailing-bytes", "bundle-invalid", "container"),
        ("prepended-bytes", "bundle-invalid", "container"),
        ("zip64-marker", "bundle-invalid", "container"),
        ("encrypted", "bundle-invalid", "entry"),
        ("data-descriptor", "bundle-invalid", "entry"),
        ("bzip2", "bundle-invalid", "entry"),
        ("crc", "bundle-invalid", "content"),
        ("oversize-file", "bundle-too-large", "entry"),
        ("oversize-archive", "bundle-too-large", "size"),
    ],
)
def test_validate_refuses_each_defect_by_name(
    tmp_path: Path, case: str, code: str, step: str
) -> None:
    archive = _bad_archive(tmp_path, case)
    with pytest.raises(SkillBundleError) as caught:
        validate_archive(archive)
    assert (caught.value.code, caught.value.step) == (code, step)
    # A refusal names paths and rules, never file contents.
    assert "boom" not in str(caught.value)


def test_validate_accepts_a_built_bundle_and_its_expected_ref(tmp_path: Path) -> None:
    archive, built = _valid(tmp_path)
    inventory = validate_archive(archive, expected=built.bundle)
    assert inventory.archive_hash == built.archive_hash
    assert inventory.tree_hash == built.tree_hash
    assert (inventory.file_count, inventory.uncompressed_size, inventory.size) == (
        3,
        built.uncompressed_size,
        built.size,
    )
    assert inventory.name == NAME and inventory.description == DESCRIPTION
    assert [entry.path for entry in inventory.entries] == [
        ENTRYPOINT,
        "assets/token.bin",
        "references/guide.md",
    ]


def test_validate_refuses_a_manifest_whose_metadata_disagrees(tmp_path: Path) -> None:
    archive, built = _valid(tmp_path)
    with pytest.raises(SkillBundleError) as caught:
        validate_archive(archive, expected=replace(built.bundle, tree_hash="0" * 64))
    assert caught.value.step == "tree"
    with pytest.raises(SkillBundleError) as caught:
        validate_archive(archive, expected=replace(built.bundle, file_count=9))
    assert caught.value.step == "entry"
    with pytest.raises(SkillBundleError) as caught:
        validate_archive(archive, expected=replace(built.bundle, uncompressed_size=1))
    assert caught.value.step == "size"
    with pytest.raises(SkillBundleError) as caught:
        validate_archive(archive, expected=built.bundle, expected_name="other")
    assert caught.value.step == "entrypoint"


def test_validate_refuses_a_non_zip_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes(b"definitely not an archive")
    with pytest.raises(SkillBundleError) as caught:
        validate_archive(path)
    assert (caught.value.code, caught.value.step) == ("bundle-invalid", "container")


def test_validate_refuses_an_empty_archive(tmp_path: Path) -> None:
    path = tmp_path / "empty.zip"
    path.write_bytes(b"")
    with pytest.raises(SkillBundleError) as caught:
        validate_archive(path)
    assert (caught.value.code, caught.value.step) == ("bundle-invalid", "size")


@pytest.mark.parametrize(
    "path",
    ["../x", "/x", "C:\\x", "a\\b", ".hidden", "a//b", "a/./b", "a/../b", "a/", "a/ "],
)
def test_path_grammar_refuses_unsafe_names(path: str) -> None:
    with pytest.raises(SkillBundleError):
        validate_bundle_path(path)


# -- extraction ------------------------------------------------------------------------


def test_extract_writes_private_modes_and_verifies_the_tree(tmp_path: Path) -> None:
    archive, built = _valid(tmp_path)
    destination = tmp_path / "stage"
    assert extract_archive(archive, destination, expected=built.bundle) == built.tree_hash
    for path in destination.rglob("*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        else:
            assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert tree_hash_of(destination) == built.tree_hash


def test_extract_refuses_a_hash_mismatch_before_writing_anything(tmp_path: Path) -> None:
    archive, built = _valid(tmp_path)
    wrong = replace(built.bundle, hash=hashlib.sha256(b"other").hexdigest())
    destination = tmp_path / "stage"
    with pytest.raises(SkillBundleError) as caught:
        extract_archive(archive, destination, expected=wrong)
    assert (caught.value.code, caught.value.step) == ("bundle-hash-mismatch", "hash")
    assert not destination.exists()


def test_extract_refuses_a_tree_mismatch_and_removes_the_stage(tmp_path: Path) -> None:
    archive, built = _valid(tmp_path)
    wrong = replace(built.bundle, tree_hash=hashlib.sha256(b"other").hexdigest())
    parent = tmp_path / "parent"
    parent.mkdir()
    destination = parent / "stage"
    with pytest.raises(SkillBundleError) as caught:
        extract_archive(archive, destination, expected=wrong)
    assert (caught.value.code, caught.value.step) == ("bundle-invalid", "tree")
    assert not destination.exists()
    assert list(parent.iterdir()) == []


def test_extract_refuses_to_reuse_an_existing_destination(tmp_path: Path) -> None:
    archive, built = _valid(tmp_path)
    destination = tmp_path / "stage"
    destination.mkdir()
    with pytest.raises(SkillBundleError):
        extract_archive(archive, destination, expected=built.bundle)
    assert destination.is_dir()


def test_extract_refuses_a_tampered_entry_mode_even_with_a_matching_manifest(
    tmp_path: Path,
) -> None:
    """The hostile-manifest case: valid bytes, an executable entry, matching hash."""

    path = _zip(
        tmp_path / "hostile.zip",
        [(ENTRYPOINT, _md_bytes(), REGULAR), ("scripts/run.sh", b"#!/bin/sh\n", EXECUTABLE)],
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # The manifest is HONEST about these bytes -- hash, size and count all match
    # -- so only the entry-mode rule can refuse it. That is the property that
    # matters: a hostile publisher cannot ship an executable by describing it.
    expected = SkillBundleRef(
        "zip",
        digest,
        path.stat().st_size,
        len(_md_bytes()) + len(b"#!/bin/sh\n"),
        2,
        ENTRYPOINT,
        "0" * 64,
    )
    with pytest.raises(SkillBundleError) as caught:
        validate_archive(path, expected=expected)
    assert (caught.value.code, caught.value.step) == ("bundle-invalid", "entry")


# -- discovery -------------------------------------------------------------------------


def test_discovery_ignores_the_bundle_stage_directory(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _tree(root / "real-skill", name="real-skill")
    stage = root / f".stage-{'a' * 32}" / "content"
    _tree(stage / "staged-skill", name="staged-skill")
    skills = scan_skills_dir(root, "managed")
    # The dot-prefixed stage is INVISIBLE, not merely lower priority: discovery
    # never descends into it, so its ``content/SKILL.md`` cannot register.
    assert [skill.name for skill in skills] == ["real-skill"]


# -- performance smoke -------------------------------------------------------------------


def test_a_large_bundle_round_trips(tmp_path: Path) -> None:
    """The operator's canonical shape: 141 files, ~25 MiB, incompressible bytes.

    A SMOKE bound, not a benchmark: the point is that build + validate + extract
    finish in the time a publish/install can spend, on a host shared with many
    other sessions.
    """

    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    total = 0
    for index in range(140):
        payload = hashlib.sha256(f"file-{index}".encode()).digest() * (182_000 // 32)
        folder = source / ("assets" if index % 2 else "references")
        folder.mkdir(exist_ok=True)
        (folder / f"file-{index:03d}.bin").write_bytes(payload)
        total += len(payload)
    started = time.monotonic()
    built = build_bundle(source)
    archive = _write(tmp_path / "large.zip", built.data)
    inventory = validate_archive(archive, expected=built.bundle)
    destination = tmp_path / "stage"
    assert extract_archive(archive, destination, expected=built.bundle) == built.tree_hash
    elapsed = time.monotonic() - started
    assert inventory.file_count == 141
    assert inventory.uncompressed_size == total + len(SKILL_MD.encode())
    assert elapsed < 120, f"large-bundle round trip took {elapsed:.1f}s"
