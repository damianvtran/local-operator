"""Strict lossless manifest controls, including all attachments and wire bounds."""

import json
from pathlib import Path

import pytest

from local_operator.hub_sync.skill_document import (
    MAX_CONTENT_BYTES,
    MAX_FILE_BYTES,
    MAX_WIRE_BYTES,
    SkillDocument,
    SkillValidationError,
    parse_frontmatter,
)


def manifest(extra="", files=None):
    return {
        "document_version": 1,
        "name": "example-skill",
        "description": "Example skill",
        "files": [
            {
                "path": "SKILL.md",
                "content": (
                    "---\nname: example-skill\ndescription: Example skill\n" f"{extra}---\n# Body\n"
                ),
            },
            *(files or []),
        ],
    }


def test_every_attachment_affects_fingerprint_and_lossless_wire() -> None:
    files = [
        {"path": "scripts/check.py", "content": "print('inert')\n"},
        {"path": "references/source.md", "content": "Source prose\n"},
        {"path": "assets/table.csv", "content": "x,y\n1,2\n"},
    ]
    doc = SkillDocument.from_json(
        manifest("metadata:\n  author: Test\n  counts: [1, true, note]\nlicense: MIT\n", files)
    )
    assert SkillDocument.from_bytes(doc.wire_bytes()) == doc
    assert [f.path for f in doc.files] == sorted(f["path"] for f in manifest(files=files)["files"])
    changed = manifest(files=[{**f, "content": f["content"] + "authored"} for f in files])
    assert SkillDocument.from_json(changed).fingerprint != doc.fingerprint


@pytest.mark.parametrize(
    "path",
    [
        "../bad.md",
        "/bad.md",
        "references/../bad.md",
        "references/.env",
        "scripts\\run.py",
        "C:/bad",
        "bad.txt",
        "references//bad",
        "references/a\x00.txt",
        "references/a/b/c/d/e.md",
    ],
)
def test_unsafe_paths_refused(path) -> None:
    with pytest.raises(SkillValidationError):
        SkillDocument.from_json(manifest(files=[{"path": path, "content": "text"}]))


@pytest.mark.parametrize(
    "paths",
    [
        ["references/a.md", "references/a.md"],
        ["references/A.md", "references/a.md"],
        ["references/é.md", "references/e\u0301.md"],
    ],
)
def test_duplicate_nfc_casefold_paths_refused(paths) -> None:
    with pytest.raises(SkillValidationError) as refusal:
        SkillDocument.from_json(
            manifest(files=[{"path": path, "content": "text"} for path in paths])
        )
    assert refusal.value.code == "skill-validation"


@pytest.mark.parametrize(
    "extra",
    [
        "name: another\n",
        "unknown: field\n",
        "metadata:\n  author: null\n",
        "metadata:\n  author: {nested: value}\n",
        "metadata:\n  author: [[nested]]\n",
        "metadata:\n  1: value\n",
        "metadata:\n  author: .inf\n",
        "metadata: &x {author: value}\n",
        "metadata: {author: !!str value}\n",
        "hide: yes-string\n",
        "targets: value\n",
        "allowed-tools: [1]\n",
        "metadata:\n  a: value\n  a: duplicate\n",
    ],
)
def test_strict_frontmatter_rejects_unsafe_or_lossy_yaml(extra) -> None:
    with pytest.raises(SkillValidationError) as refusal:
        SkillDocument.from_json(manifest(extra))
    assert refusal.value.code == "skill-validation"


@pytest.mark.parametrize(
    "extra",
    [
        "license: MIT\ncompatibility: Local Operator\nversion: '1'\n",
        (
            "allowed-tools: read\ntargets: [codex, claude]\nenabled: true\nhide: false\n"
            "disable-model-invocation: true\n"
        ),
        (
            "allowed-tools: [read, write]\nmetadata:\n  authors: [Alice, Bob]\n"
            "  count: 2\n  active: true\n"
        ),
    ],
)
def test_approved_optional_frontmatter_preserved(extra) -> None:
    document = SkillDocument.from_json(manifest(extra))
    assert document.to_json() == manifest(extra)


def test_alias_only_changes_name_scalar_and_roundtrips_canonical_source() -> None:
    source = manifest(
        "# retained comment\nmetadata:\n  author: 'Example Person'\n",
        [{"path": "scripts/inert.sh", "content": "echo exact\n"}],
    )
    source["files"][0]["content"] = source["files"][0]["content"].replace(
        "name: example-skill", "name: 'example-skill'"
    )
    document = SkillDocument.from_json(source)
    rendered = document.with_name("private-alias")
    assert "name: private-alias\n" in rendered.files[0].content
    assert "# retained comment\nmetadata:\n  author: 'Example Person'" in rendered.files[0].content
    restored = rendered.with_name(document.name)
    assert parse_frontmatter(restored.files[0].content)["name"] == document.name
    assert restored.files[1] == document.files[1]


def test_multiline_name_alias_preserves_following_metadata() -> None:
    source = manifest()
    source["files"][0]["content"] = source["files"][0]["content"].replace(
        "name: example-skill\n", "name: >-\n  example-skill\n"
    )
    rendered = SkillDocument.from_json(source).with_name("alias")
    assert rendered.name == "alias"
    assert parse_frontmatter(rendered.files[0].content)["description"] == "Example skill"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra="secret"),
        lambda value: value.update(document_version=True),
        lambda value: value.update(name="different"),
        lambda value: value.update(files=[]),
        lambda value: value["files"][0].update(extra="claim"),
        lambda value: value["files"][0].update(content="\x00binary"),
    ],
)
def test_invalid_manifest_never_returns_partial_success(mutation) -> None:
    value = manifest()
    mutation(value)
    with pytest.raises(SkillValidationError):
        SkillDocument.from_json(value)


@pytest.mark.parametrize(
    "paths",
    [
        ["scripts/run", "scripts/run/file.py"],
        ["scripts/run/file.py", "scripts/run"],
        ["references/A", "references/a/file.md"],
        ["references/a/file.md", "references/A"],
        ["references/é", "references/e\u0301/file.md"],
        ["references/e\u0301/file.md", "references/é"],
    ],
)
def test_file_and_ancestor_directory_paths_collide_before_install(paths) -> None:
    with pytest.raises(SkillValidationError) as refusal:
        SkillDocument.from_json(
            manifest(files=[{"path": path, "content": "text"} for path in paths])
        )
    assert refusal.value.code == "skill-validation"


@pytest.mark.parametrize("value", ["[" * 1100 + "1" + "]" * 1100, "9" * 400])
def test_deep_yaml_and_unrepresentable_number_fail_by_name(value) -> None:
    with pytest.raises(SkillValidationError) as refusal:
        SkillDocument.from_json(manifest(f"metadata:\n  value: {value}\n"))
    assert refusal.value.code == "skill-validation"


def test_deep_json_fails_by_name() -> None:
    with pytest.raises(SkillValidationError) as refusal:
        SkillDocument.from_bytes(("[" * 1100 + "1" + "]" * 1100).encode())
    assert refusal.value.code == "skill-validation"


def test_wire_duplicate_keys_and_size_refused() -> None:
    with pytest.raises(SkillValidationError):
        SkillDocument.from_bytes(
            json.dumps(manifest())
            .replace('"document_version": 1', '"document_version": 1, "document_version": 1')
            .encode()
        )
    with pytest.raises(SkillValidationError) as refusal:
        SkillDocument.from_bytes(b" " * (MAX_WIRE_BYTES + 1))
    assert refusal.value.code == "skill-too-large"


def test_count_file_and_expanded_sum_limits() -> None:
    for files in (
        [{"path": f"references/{i}.md", "content": "text"} for i in range(64)],
        [{"path": "assets/big.txt", "content": "x" * (MAX_FILE_BYTES + 1)}],
        [
            {"path": f"assets/{i}.txt", "content": "x" * MAX_FILE_BYTES}
            for i in range(MAX_CONTENT_BYTES // MAX_FILE_BYTES + 1)
        ],
    ):
        with pytest.raises(SkillValidationError) as refusal:
            SkillDocument.from_json(manifest(files=files))
        assert refusal.value.code == "skill-too-large"


def test_source_reader_detects_symlinks_binary_and_hidden_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(manifest()["files"][0]["content"])
    script = source / "scripts" / "inert.sh"
    script.parent.mkdir()
    script.write_text("echo inert\n")
    script.chmod(0o755)
    assert len(SkillDocument.from_directory(source).files) == 2
    secret = source / ".env"
    secret.write_text("synthetic-secret")
    with pytest.raises(SkillValidationError):
        SkillDocument.from_directory(source)
    secret.unlink()
    link = source / "scripts" / "link.sh"
    link.symlink_to(script)
    with pytest.raises(SkillValidationError):
        SkillDocument.from_directory(source)
    link.unlink()
    script.write_bytes(b"\xff\x00")
    with pytest.raises(SkillValidationError):
        SkillDocument.from_directory(source)
