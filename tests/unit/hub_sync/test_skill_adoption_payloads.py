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


CONTAINER_PAYLOADS = [
    '> ~~~python\n> RAW_CODE_PAYLOAD = "a  b"  \n> ~~~\n\n',
    '- ~~~python\n  RAW_CODE_PAYLOAD = "a  b"  \n  ~~~\n\n',
    '1. Outer\n   - ~~~python\n     RAW_CODE_PAYLOAD = "a  b"  \n     ~~~\n\n',
    '>     RAW_CODE_PAYLOAD = "a  b"  \n>\n\n',
    '    RAW_CODE_PAYLOAD = "a  b"  \n\n\n',
    '<pre>\nRAW_CODE_PAYLOAD = "a  b"  \n</pre>\n\n',
    '<script>\nRAW_CODE_PAYLOAD = "a  b"  \n</script>\n\n',
    '<div data-runtime="a  b">\nRAW_CODE_PAYLOAD  \n</div>\n\n',
]


@pytest.mark.parametrize("payload", CONTAINER_PAYLOADS)
def test_commonmark_containers_default_exact_and_refuse_changed_trailing_bytes(payload):
    changed = payload.replace("  \n", "\n")
    assert changed != payload
    assert _norm(payload) != _norm(changed)
    result = _merge_text(
        "references/guidance.md", payload, changed, resolver=NeverResolve(), allow_llm=True
    )
    assert result.outcome == "refused"
    assert result.merged == payload


class OpaqueProseModel:
    def __init__(self):
        self.calls = 0

    def resolve(self, req):
        import re

        self.calls += 1
        assert "RAW_CODE_PAYLOAD" not in repr(req)
        assert "requirements.md" not in repr(req)
        assert "rubric.md" not in repr(req)
        # The real resolver owns prose only. Shared opaque identities occur once
        # in the fixture union; none of their raw payloads reaches this boundary.
        seen = set()

        def once(match):
            marker = match.group()
            if marker in seen:
                return ""
            seen.add(marker)
            return marker

        text = re.sub(r"LOP_PROTECTED_[a-f0-9]{64}", once, req.local + "\n\n" + req.remote)
        return ConflictProposal(text, covers=("l1", "r1"), model="fixture-opaque-prose")


@pytest.mark.parametrize("payload", CONTAINER_PAYLOADS)
def test_commonmark_container_bytes_survive_model_prose_merge_and_real_apply(tmp_path, payload):
    identity, document, baseline, storage, tree, _ = _setup(tmp_path)
    local = (
        "# Guidance\n\nVerify every municipality POC with the clerk before the pilot. "
        "Preserve unique clerk evidence.\n\n" + payload
    )
    remote = (
        "# Guidance\n\nVerify every municipality procurement deadline against the official "
        "notice. Preserve unique procurement evidence.\n\n" + payload
    )
    (tree.root / "references").mkdir()
    (tree.root / "references/guidance.md").write_text(local)
    hub = replace(document, files=document.files + (SkillFile("references/guidance.md", remote),))
    raw = read_tree(tree.root)
    snapshot = durable_snapshot(storage.config_dir, document.name, raw)
    model = OpaqueProseModel()
    proposal = build_proposal(document.name, raw, hub, resolver=model)
    assert proposal.document is not None, proposal.to_json()
    assert model.calls > 0
    storage.adopt(
        identity,
        document.name,
        proposal.document,
        replace(baseline, fields=hub.to_json(), fingerprint=hub.fingerprint),
        expected_tree_hash=raw.tree_hash,
        snapshot=snapshot,
        carried_paths=("assets/local.bin",),
    )
    installed = (tree.root / "references/guidance.md").read_bytes()
    assert payload.encode() in installed
    assert "municipality POC".encode() in installed
    assert "municipality procurement deadline".encode() in installed


@pytest.mark.parametrize(
    "source",
    [
        "Use `requirements.md",  # unmatched rule returns True
        "Use \\`requirements.md` safely",  # escaped/unmatched delimiter
        " Quality `requirements.md` before outreach.\n",  # parser stripped indentation
        "Quality `requirements.md`\r\nwith more guidance.\r\n",  # normalized state.src
        "> Quality `requirements.md` before outreach.\n",
        "- Quality `requirements.md` before outreach.\n",
        "<div>Quality `requirements.md`</div>\n",
        'Use `read("requirements.md")` before outreach.\n',
        "Use ` requirements.md ` before outreach.\n",
        "Use `requirements.md\nother` before outreach.\n",
        "Use `https://example.test/requirements.md` before outreach.\n",
        "Use `/requirements.md` before outreach.\n",
    ],
)
def test_unproven_or_nonliteral_inline_payload_has_no_filepath_exception(source):
    from local_operator.hub_sync.skill_adoption import _rewrite_relocated_refs

    result = _rewrite_relocated_refs(
        source,
        {"requirements.md": "references/requirements.md"},
        known_paths={"requirements.md", "SKILL.md"},
    )
    assert result == source


def test_actual_municipal_quality_bare_inline_paths_merge_and_apply_exact(tmp_path):
    identity, document, baseline, storage, tree, _ = _setup(tmp_path)
    head = f"---\nname: {document.name}\ndescription: {document.description}\n---\n"
    payload = CONTAINER_PAYLOADS[0]
    local_quality = (
        "Verify every municipality POC using `requirements.md` and `rubric.md` "
        "before the pilot. Preserve the unique POC clerk confirmation before the pilot."
    )
    hub_quality = (
        "Verify every municipality procurement deadline using `references/requirements.md` "
        "and `references/rubric.md` before the pilot. Preserve the unique procurement "
        "audit evidence before the pilot."
    )
    local = head + "# Municipal procurement\n\n## Quality\n\n" + local_quality + "\n\n" + payload
    remote = head + "# Municipal procurement\n\n## Quality\n\n" + hub_quality + "\n\n" + payload
    (tree.root / "SKILL.md").write_text(local)
    requirements = "# Requirements\n\nKeep a dated audit trail for each municipality.\n"
    rubric = "# Rubric\n\nScore procurement evidence against official records.\n"
    (tree.root / "requirements.md").write_text(requirements)
    (tree.root / "rubric.md").write_text(rubric)
    hub = replace(
        document,
        files=tuple(file for file in document.files if file.path != "SKILL.md")
        + (
            SkillFile("SKILL.md", remote),
            SkillFile("references/requirements.md", requirements),
            SkillFile("references/rubric.md", rubric),
        ),
    )
    raw = read_tree(tree.root)
    snapshot = durable_snapshot(storage.config_dir, document.name, raw)
    model = OpaqueProseModel()
    proposal = build_proposal(document.name, raw, hub, resolver=model)
    assert proposal.document is not None, proposal.to_json()
    assert model.calls > 0
    main_plan = next(plan for plan in proposal.files if plan.path == "SKILL.md")
    assert main_plan.proposed is not None
    assert "`references/requirements.md`" in main_plan.proposed
    assert "`references/rubric.md`" in main_plan.proposed
    assert "unique POC clerk confirmation" in main_plan.proposed
    assert "unique procurement audit evidence" in main_plan.proposed
    assert "references/requirements.md" in main_plan.to_json()["diff_vs_local"]
    storage.adopt(
        identity,
        document.name,
        proposal.document,
        replace(baseline, fields=hub.to_json(), fingerprint=hub.fingerprint),
        expected_tree_hash=raw.tree_hash,
        snapshot=snapshot,
        carried_paths=("assets/local.bin",),
    )
    installed = (tree.root / "SKILL.md").read_bytes()
    assert payload.encode() in installed
    assert b"`references/requirements.md`" in installed
    assert b"`references/rubric.md`" in installed
    assert (tree.root / "references/requirements.md").read_bytes() == requirements.encode()
    assert (tree.root / "references/rubric.md").read_bytes() == rubric.encode()


def test_inline_literal_relocation_diff_uses_original_captured_bytes(tmp_path):
    identity, document, baseline, storage, tree, _ = _setup(tmp_path)
    main = next(file.content for file in document.files if file.path == "SKILL.md")
    local = main + "\nRead ``requirements.md`` before outreach.\n"
    remote = main + "\nRead ``references/requirements.md`` before outreach.\n"
    (tree.root / "SKILL.md").write_text(local)
    (tree.root / "requirements.md").write_text("Requirements remain exact.\n")
    hub = replace(
        document,
        files=tuple(file for file in document.files if file.path != "SKILL.md")
        + (
            SkillFile("SKILL.md", remote),
            SkillFile("references/requirements.md", "Requirements remain exact.\n"),
        ),
    )
    proposal = build_proposal(document.name, read_tree(tree.root), hub, resolver=NeverResolve())
    assert proposal.state == "ready"  # setup also carries a local-only binary
    plan = next(plan for plan in proposal.files if plan.path == "SKILL.md")
    assert plan.local == local
    assert plan.proposed is not None
    assert "``references/requirements.md``" in plan.proposed
    diff = plan.to_json()["diff_vs_local"]
    assert "-Read ``requirements.md``" in diff
    assert "+Read ``references/requirements.md``" in diff


@pytest.mark.parametrize(
    "source,destination,literal,expected",
    [
        ("references/local.md", "references/local.md", "../requirements.md", "requirements.md"),
        ("rubric.md", "references/rubric.md", "./requirements.md", "./requirements.md"),
    ],
)
def test_inline_literal_exception_uses_frozen_source_and_destination(
    source, destination, literal, expected
):
    from local_operator.hub_sync.skill_adoption import _rewrite_relocated_refs

    text = f"Read ``{literal}`` before outreach.\n"
    result = _rewrite_relocated_refs(
        text,
        {"requirements.md": "references/requirements.md"},
        source_path=source,
        destination_path=destination,
        known_paths={"requirements.md", source},
    )
    assert result == f"Read ``{expected}`` before outreach.\n"


def test_inline_literal_ambiguous_and_traversing_paths_fail_closed():
    from local_operator.hub_sync.skill_adoption import (
        AdoptionError,
        _rewrite_relocated_refs,
    )

    # Bare ``requirements.md`` from references/ IS the relocation destination
    # and is accepted verbatim (see test below); these remain ambiguous.
    for literal in ("../../requirements.md", "sub/requirements.md"):
        with pytest.raises(AdoptionError) as failure:
            _rewrite_relocated_refs(
                f"Read `{literal}`.\n",
                {"requirements.md": "references/requirements.md"},
                source_path="references/local.md",
                known_paths={"requirements.md", "references/local.md"},
            )
        assert failure.value.code == "adoption-needs-choice"


def test_inline_literal_naming_relocation_destination_is_kept_verbatim():
    from local_operator.hub_sync.skill_adoption import _rewrite_relocated_refs

    moves = {"requirements.md": "references/requirements.md"}
    text = "Read `requirements.md` and [r](requirements.md).\n"
    assert (
        _rewrite_relocated_refs(
            text,
            moves,
            source_path="references/local.md",
            known_paths={"requirements.md", "references/local.md"},
        )
        == text
    )


def test_verbatim_mode_never_refuses_but_never_rewrites_a_guess():
    from local_operator.hub_sync.skill_adoption import _rewrite_relocated_refs

    text = "Read `sub/requirements.md` and ../../requirements.md.\n"
    assert (
        _rewrite_relocated_refs(
            text,
            {"requirements.md": "references/requirements.md"},
            source_path="references/local.md",
            known_paths={"requirements.md", "references/local.md"},
            on_ambiguous="verbatim",
        )
        == text
    )


def test_unknown_reference_definition_physical_gap_remains_exact():
    from local_operator.hub_sync.skill_adoption import _code_spans

    source = '[runtime]: /data/target  "title  "\n\n'
    assert _code_spans(source) == ((0, len(source)),)
    assert _norm(source) != _norm(source.replace("title  ", "title "))
