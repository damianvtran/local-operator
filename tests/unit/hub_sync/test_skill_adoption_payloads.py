"""Original source bytes, rather than a proposed suffix, own payload semantics."""

from dataclasses import replace

import pytest

from local_operator.hub_sync.merge import ConflictProposal
from local_operator.hub_sync.skill_adoption import (
    _merge_text,
    _norm,
    _relocation_target,
    build_proposal,
    durable_snapshot,
    read_tree,
)
from local_operator.hub_sync.skill_document import SkillFile

from .test_skill_adoption_storage import _setup


class NeverResolve:
    def resolve(self, req):
        raise AssertionError("raw payload reached prose resolver")


def test_root_csv_cannot_be_promoted_to_markdown_during_real_adoption(tmp_path):
    identity, document, baseline, storage, tree, _ = _setup(tmp_path)
    original = '"alpha\nbeta",42\n'
    (tree.root / "data.csv").write_text(original)
    hub = replace(
        document, files=document.files + (SkillFile("references/hub.md", '"alpha beta",42\n'),)
    )
    raw = read_tree(tree.root)
    snapshot = durable_snapshot(storage.config_dir, document.name, raw)
    proposal = build_proposal(document.name, raw, hub, resolver=NeverResolve())
    assert proposal.state == "ready"
    assert proposal.document is not None
    files = {file.path: file.content for file in proposal.document.files}
    assert files["references/data.csv"] == original
    assert files["references/hub.md"] == '"alpha beta",42\n'
    record = replace(baseline, fields=hub.to_json(), fingerprint=hub.fingerprint)
    storage.adopt(
        identity,
        document.name,
        proposal.document,
        record,
        expected_tree_hash=raw.tree_hash,
        snapshot=snapshot,
        carried_paths=("assets/local.bin",),
    )
    assert (tree.root / "references/data.csv").read_bytes() == original.encode()


@pytest.mark.parametrize("path", ["data.csv", "data.json", "data.yaml", "data.txt", "code.md"])
def test_original_nonprose_bytes_remain_authoritative_at_same_suffix(tmp_path, path):
    identity, document, baseline, storage, tree, _ = _setup(tmp_path)
    local = '"alpha\nbeta",42\n'
    (tree.root / path).write_text(local)
    if path == "code.md":
        (tree.root / path).chmod(0o755)
    hub = replace(
        document,
        files=document.files
        + (
            SkillFile("references/" + path, '"alpha beta",42\n'),
            SkillFile("references/hub-only.md", "Force a real document write"),
        ),
    )
    proposal = build_proposal(document.name, read_tree(tree.root), hub, resolver=NeverResolve())
    assert proposal.state == "needs-choice"
    assert proposal.document is None
    assert (tree.root / path).read_bytes() == local.encode()


def test_only_genuine_markdown_can_content_match_a_differently_named_markdown():
    remote = {"references/hub.md": "Paragraph one line.\n"}
    assert _relocation_target("requirements.md", "Paragraph one\nline.\n", remote, set()) == (
        "references/hub.md"
    )
    assert _relocation_target("data.csv", "Paragraph one\nline.\n", remote, set()) == (
        "references/data.csv"
    )
    assert _relocation_target("code.md", "Paragraph one\nline.\n", remote, set(), mode=0o755) == (
        "references/code.md"
    )


CODE_PAIRS = [
    ('```python\nVALUE = "a  b"  \n```\n', '```python\nVALUE = "a b"\n```\n'),
    ('    VALUE = "a  b"  \n', '    VALUE = "a b"\n'),
    ('`VALUE = "a  b"`', '`VALUE = "a b"`'),
    ('<pre>VALUE = "a  b"  \n</pre>', '<pre>VALUE = "a b"\n</pre>'),
    ('<code>VALUE = "a  b"</code>', '<code>VALUE = "a b"</code>'),
    ('<script>const value = "a  b";\n</script>', '<script>const value = "a b";\n</script>'),
]


@pytest.mark.parametrize("local,remote", CODE_PAIRS)
def test_meaningful_code_whitespace_never_compares_equal_or_reaches_model(local, remote):
    assert _norm(local) != _norm(remote)
    result = _merge_text(
        "references/guidance.md", local, remote, resolver=NeverResolve(), allow_llm=True
    )
    assert result.outcome == "refused"
    assert result.merged == local


@pytest.mark.parametrize("code", [pair[0] for pair in CODE_PAIRS])
def test_equal_code_can_survive_genuine_markdown_prose_merge_without_model_seeing_code(code):
    local = (
        "# Guidance\n\nVerify every municipality POC with the clerk before the pilot. "
        "Preserve unique clerk evidence.\n\n" + code + "\n"
    )
    remote = (
        "# Guidance\n\nVerify every municipality procurement deadline against the official "
        "notice. Preserve unique procurement evidence.\n\n" + code + "\n"
    )

    class FaithfulProse:
        calls = 0

        def resolve(self, req):
            self.calls += 1
            assert code not in req.local and code not in req.remote
            assert "VALUE" not in req.local + req.remote
            assert "const value" not in req.local + req.remote
            return ConflictProposal(
                req.local + "\n\n" + req.remote, covers=("l1", "r1"), model="fixture-prose-only"
            )

    model = FaithfulProse()
    result = _merge_text("references/guidance.md", local, remote, resolver=model, allow_llm=True)
    assert result.outcome == "merged", result
    assert code in str(result.merged)
    assert "municipality POC with the clerk" in str(result.merged)
    assert "municipality procurement deadline" in str(result.merged)
    assert model.calls > 0


def test_reference_relocation_never_rewrites_embedded_code_payloads():
    from local_operator.hub_sync.skill_adoption import _rewrite_relocated_refs

    code = '`read("requirements.md")`'
    source = "Read [requirements](requirements.md). " + code
    rewritten = _rewrite_relocated_refs(
        source,
        {"requirements.md": "references/requirements.md"},
        known_paths={"requirements.md", "SKILL.md"},
    )
    assert "[requirements](references/requirements.md)" in rewritten
    assert code in rewritten


@pytest.mark.parametrize("path", ["SKILL.md", "references/guidance.md"])
def test_final_document_assembly_preserves_indented_payload_trailing_blank_lines(tmp_path, path):
    identity, document, baseline, storage, tree, _ = _setup(tmp_path)
    code = '    VALUE = "a  b"  \n\n\n'
    local_body = "# Rules\n\nVerify every clerk identity before outreach.\n\n" + code
    remote_body = "# Rules\n\nCheck procurement deadlines before outreach.\n\n" + code
    main = next(file for file in document.files if file.path == "SKILL.md")
    if path == "SKILL.md":
        head = main.content.split("---", 2)[0] + "---" + main.content.split("---", 2)[1] + "---\n"
        local, remote = head + local_body, head + remote_body
    else:
        local, remote = local_body, remote_body
    (tree.root / path).parent.mkdir(parents=True, exist_ok=True)
    (tree.root / path).write_text(local)
    files = tuple(file for file in document.files if file.path != path) + (SkillFile(path, remote),)
    hub = replace(document, files=files)

    class ProseOnly:
        def resolve(self, req):
            assert "VALUE" not in req.local + req.remote
            return ConflictProposal(
                req.local + "\n\n" + req.remote, covers=("l1", "r1"), model="fixture-prose-only"
            )

    proposal = build_proposal(document.name, read_tree(tree.root), hub, resolver=ProseOnly())
    assert proposal.document is not None, proposal.to_json()
    proposed = next(file.content for file in proposal.document.files if file.path == path)
    assert code in proposed
    assert "Verify every clerk identity" in proposed
    assert "Check procurement deadlines" in proposed


@pytest.mark.parametrize("corruption", ["missing", "invented"])
def test_protected_code_final_validation_refuses_invalid_engine_assembly(monkeypatch, corruption):
    from local_operator.hub_sync import skill_adoption
    from local_operator.hub_sync.merge import MergeResult

    code = '<pre>VALUE = "a  b"\n</pre>'
    local = "Local guidance.\n\n" + code
    remote = "Hub guidance.\n\n" + code

    def invalid_engine(field, options):
        merged = field.local if corruption == "invented" else "Local and Hub guidance."
        if corruption == "invented":
            merged += "\n\n`invented code`"
        return MergeResult(field.field, "merged", merged)

    monkeypatch.setattr(skill_adoption, "merge_field", invalid_engine)
    result = _merge_text("references/guidance.md", local, remote, resolver=None, allow_llm=True)
    assert result.outcome == "refused"
    assert result.merged == local
