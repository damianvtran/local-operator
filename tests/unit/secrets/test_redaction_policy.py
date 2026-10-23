"""The redaction POLICY, in both directions, on the surfaces that read it.

WHY THIS FILE EXISTS
--------------------
The shape table is deliberately broad, and its false positives are not
cosmetic: a masked key name is copied back into the next command, and a
registered hit rewrites every later occurrence in the session.  Two operator
reports (2026-10-16 and LOH-28) were each answered with a narrow rule patch and
the class survived, because there was no prose policy and no negative-test
table to fail when a rule widened again.  ``docs/security/redaction-policy.md``
is the prose definition; this file is its executable half.

STRUCTURE
---------
``PUBLIC_COPY`` is one row per "does not count as a secret" entry in the policy.
Each row is run through every masking surface the tree has — the result pass,
the argument pass, the bare shape pass, and the registration gate — so a future
widening fails here instead of silently blocking an auditor.  ``SECRETS`` is the
positive control: the same surfaces must still mask a genuine credential.  The
last section drives the sanctioned public-copy reader, including its audit
record and its refusals.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from local_operator.harness.loop import AgentLoop
from local_operator.harness.redaction import PUBLIC_COPY_TOOLS
from local_operator.harness.types import TextContent, ToolContext
from local_operator.redaction_shapes import (
    is_registerable_component,
    scrub_shapes,
    scrub_shapes_for_arguments,
)
from local_operator.tools.public_copy import (
    TOOL_NAME,
    execute_read_public_copy,
)
from local_operator.tools.registry import TOOL_BUILDERS
from local_operator.variables import VariableStore

#: A fake GitHub token and a fake AWS secret.  Assembled from pieces so the
#: test SOURCE is not itself a credential-shaped string: this file is read
#: through the pass it tests, and a literal would be masked in the transcript
#: that a reviewer reads.  Nothing here is a real credential.
FAKE_GITHUB = "ghp_" + "A" * 36
FAKE_AWS = "wJalrXUtnFEMI" + "/K7MDENG/bPxRfiCYEXAMPLEKEY"

#: (id, text, the NAME that must never be registered).
#: Keep one row per bullet in policy §2.2.  A new non-secret class gets a row
#: here before the rule that releases it is allowed to land.
PUBLIC_COPY: tuple[tuple[str, str, str], ...] = (
    ("bare catalogue key", "wire.slash.credential", "wire.slash.credential"),
    (
        "catalogue JSON line",
        '"wire.slash.credential": "Type or paste a secret after a space; masked"',
        "wire.slash.credential",
    ),
    (
        "generated key map line",
        '"wire.slash.credential": undefined;',
        "wire.slash.credential",
    ),
    (
        "secret store name after --secret",
        "--secret wire.slash.credential",
        "wire.slash.credential",
    ),
    (
        "secret store name after --secret=",
        "--secret=wire.slash.credential",
        "wire.slash.credential",
    ),
    ("cargo crate", "cargo test -p minerva-core-engine", "minerva-core-engine"),
    ("cargo worktree crate", "cargo test -p mda-category-sources", "mda-category-sources"),
    ("pip package", "pip install local-operator", "local-operator"),
    ("environment variable NAME", "OPENAI_API_KEY", "OPENAI_API_KEY"),
    (
        "documentation prose",
        "The credential is masked in tool output; the key name is not.",
        "",
    ),
    ("placeholder reference", "${GITHUB_TOKEN}", "GITHUB_TOKEN"),
)

#: (id, text, the VALUE that must still be masked).
SECRETS: tuple[tuple[str, str, str], ...] = (
    ("github token assignment", "GITHUB_TOKEN=" + FAKE_GITHUB, FAKE_GITHUB),
    ("aws secret assignment", "AWS_SECRET_ACCESS_KEY=" + FAKE_AWS, FAKE_AWS),
    (
        "dsn password",
        "postgres://user:" + "hunter2hunter2" + "@db.example.com/app",
        "hunter2hunter2",
    ),
    (
        "bearer header",
        "Authorization: Bearer " + "abcdefghijklmnopqrstuvwxyz0123456789",
        "abcdefghijklmnopqrstuvwxyz0123456789",
    ),
    ("password flag", "--password " + "hunter2hunter2", "hunter2hunter2"),
)


def test_a_marked_identifier_field_passes_under_a_weak_name() -> None:
    """Identifiers are released by the render-site span, not by a raw label.

    LOH-28's design is the contract: a renderer that knows a field is an
    identifier wraps it in ``VariableStore.identifier_span``. The project tool
    does; a raw ``key: <long phrase>`` does not, and the base keep-masking
    control for that spelling still holds. This test pins the sanctioned route.
    """
    store = VariableStore()
    slug = "mda" + "-internal-contribution-delivery"
    marked = "key: " + store.identifier_span(slug)
    scrubbed, report = store.redact_with_report(marked)
    assert scrubbed == "key: " + slug
    assert not report.labels


#: Controls for the WIDENING BOUNDARY. A long identifier is released only under
#: a weak name (``key:``) or after ``--secret``. These rows prove the release did
#: not move to a strong name, to ``token:`` or to ``--token``.
STRONG_NAME_NEGATIVES: tuple[tuple[str, str], ...] = (
    (
        "strong name long identifier",
        "DB_PASSWORD=" + "mda" + "-internal-contribution-delivery",
    ),
    (
        "token label long identifier",
        "token: " + "mda" + "-internal-contribution-delivery",
    ),
    (
        "--token dotted name-shaped value",
        "--token " + "wire" + "." + "slash" + "." + "credential",
    ),
    ("--secret value-shaped token", "--secret " + FAKE_GITHUB),
    ("--secret single lowercase word", "--secret " + "swordfish"),
    ("--secret digit-bearing token", "--secret " + "hunter" + "2-prod-eu"),
    ("--secret= digit-bearing token", "--secret=" + "my-api-key" + "-123"),
)


@pytest.mark.parametrize(
    "case_id,text", STRONG_NAME_NEGATIVES, ids=[row[0] for row in STRONG_NAME_NEGATIVES]
)
def test_the_release_does_not_move_to_a_strong_name(case_id: str, text: str) -> None:
    """The widening is gated on the weak name / store-entry flag, not on shape."""
    redacted, report = VariableStore().redact_with_report(text)
    assert redacted != text, f"{case_id}: a strong credential position was released"
    assert report.labels, f"{case_id}: no shape label on a masked strong position"


@pytest.mark.parametrize(
    "case_id,text,name", PUBLIC_COPY, ids=[row[0] for row in PUBLIC_COPY]
)
def test_public_copy_passes_every_masking_surface(case_id: str, text: str, name: str) -> None:
    """A non-secret is unchanged on the result, argument and bare shape passes."""
    store = VariableStore()

    redacted, report = store.redact_with_report(text)
    assert redacted == text, f"{case_id}: result pass masked a public name"
    assert not report.labels, f"{case_id}: result pass reported a shape hit"

    assert store.redact_arguments(text) == text, f"{case_id}: argument pass masked a public name"
    assert scrub_shapes(text) == text, f"{case_id}: bare shape pass masked a public name"
    assert scrub_shapes_for_arguments(text) == text, (
        f"{case_id}: bare argument pass masked a public name"
    )

    # Registration is the amplifier: even a mask that is released in place must
    # never promote a NAME into the session-wide redaction set.
    if name:
        assert not is_registerable_component(name), f"{case_id}: a public name was registerable"


@pytest.mark.parametrize("case_id,text,value", SECRETS, ids=[row[0] for row in SECRETS])
def test_genuine_secret_still_masks_on_every_surface(case_id: str, text: str, value: str) -> None:
    """The positive control: a real credential value is masked in both passes."""
    store = VariableStore()

    redacted, report = store.redact_with_report(text)
    assert redacted != text, f"{case_id}: result pass published a genuine secret"
    assert report.labels, f"{case_id}: no shape label for a genuine secret"

    assert store.redact_arguments(text) != text, f"{case_id}: argument pass published a secret"
    assert scrub_shapes(text) != text, f"{case_id}: bare shape pass published a secret"


def test_the_same_evidence_shows_the_secret_masked_and_the_key_readable() -> None:
    """The operator's requested same-frame proof.

    ``wire.slash.credential`` is a public catalogue key and must pass; a genuine
    token in the next line must still be masked.  One store, one call each.
    """
    store = VariableStore()

    public = "wire.slash.credential"
    assert store.redact_with_report(public)[0] == public

    secret_line = "GITHUB_TOKEN=" + FAKE_GITHUB
    assert store.redact_with_report(secret_line)[0] != secret_line
    assert store.redact_arguments(secret_line) != secret_line


def test_secret_argument_pass_releases_a_digit_bearing_name_but_the_result_masks() -> None:
    """The pass distinction is deliberate, so it is pinned rather than implied.

    ``--secret`` binds a store ENTRY NAME; the argument pass must leave the
    command byte-identical or the lookup cannot run. A digit-bearing hyphenated
    name is therefore released there, while the result pass still masks it (a
    digit is evidence of an opaque token in output). If a future change tightens
    or widens either side, this row forces the policy text to move with it.
    """
    value = "my" + "-api" + "-key" + "-123"
    text = "--" + "secret=" + value
    store = VariableStore()
    assert store.redact_arguments(text) == text
    assert store.redact_with_report(text)[0] != text


def test_only_the_sanctioned_tool_is_exempt_from_result_redaction() -> None:
    """The exemption is a tool IDENTITY, not a caller-settable flag."""
    assert PUBLIC_COPY_TOOLS == frozenset({TOOL_NAME})
    assert TOOL_NAME in TOOL_BUILDERS
    assert "bash" not in PUBLIC_COPY_TOOLS
    assert "read" not in PUBLIC_COPY_TOOLS


@pytest.mark.asyncio
async def test_public_copy_result_skips_the_result_pass() -> None:
    """``read_public_copy`` is verbatim; every other tool still masks."""
    content = [TextContent(text="SECRET-VALUE")]

    def redact(text: str) -> str:
        return text.replace("SECRET-VALUE", "[redacted]")

    public = await AgentLoop._redact_content(content, redact, TOOL_NAME, {}, ".")
    assert public[0].text == "SECRET-VALUE"

    ordinary = await AgentLoop._redact_content(content, redact, "bash", {}, ".")
    assert ordinary[0].text == "[redacted]"


@pytest.mark.asyncio
async def test_public_copy_details_skip_the_result_pass() -> None:
    """The structured half of a public-copy result is not a second channel."""
    details = {"diff": "SECRET-VALUE"}

    def redact(text: str) -> str:
        return text.replace("SECRET-VALUE", "[redacted]")

    public = await AgentLoop._redact_details(details, redact, TOOL_NAME, {}, ".")
    assert public == details

    ordinary = await AgentLoop._redact_details(details, redact, "bash", {}, ".")
    assert ordinary == {"diff": "[redacted]"}


def _public_repo(tmp_path: Path) -> Path:
    """A miniature checkout with one public catalogue and one key map."""
    root = tmp_path / "repo"
    catalogue = root / "local_operator/i18n/catalogues/en"
    catalogue.mkdir(parents=True)
    (catalogue / "wire.slash.json").write_text(
        json.dumps(
            {"wire.slash.credential": "Type or paste a secret after a space; masked"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    keys = root / "local_operator/i18n/keys"
    keys.mkdir(parents=True)
    (keys / "wire_slash.py").write_text(
        'return Msg("wire.slash.credential", {})\n', encoding="utf-8"
    )
    # ``read_public_copy`` serves COMMITTED bytes only (the review-1 blocker):
    # the fixture must be a real HEAD, not just a working tree.
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"],
        check=True,
    )
    return root


@pytest.mark.asyncio
async def test_public_copy_reads_a_catalogue_key_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = ToolContext(cwd=str(root), session_dir=str(tmp_path / "sessions" / "abc123"))

    result = await execute_read_public_copy(
        "call-1",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        context,
    )

    assert not result.is_error, result.text
    assert "wire.slash.credential" in result.text
    assert "Type or paste a secret after a space; masked" in result.text
    assert result.details is not None
    assert result.details["public_copy"] is True
    assert result.details["policy"] == "docs/security/redaction-policy.md"

    log = tmp_path / "config" / "audit" / "public-copy-reads.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["tool"] == TOOL_NAME
    assert records[0]["session"] == "abc123"
    assert records[0]["key"] == "wire.slash.credential"
    assert records[0]["path"].endswith("wire.slash.json")
    assert len(records[0]["sha256"]) == 64


@pytest.mark.asyncio
async def test_public_copy_reads_a_key_module_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = ToolContext(cwd=str(root), session_dir=str(tmp_path / "sessions" / "abc123"))

    result = await execute_read_public_copy(
        "call-2",
        {"path": "local_operator/i18n/keys/wire_slash.py", "key": "wire.slash.credential"},
        None,
        None,
        context,
    )

    assert not result.is_error, result.text
    assert 'Msg("wire.slash.credential", {})' in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "local_operator/redaction_shapes.py",
        "local_operator/i18n/catalogues/en/../../secrets.json",
    ],
)
async def test_public_copy_refuses_non_public_paths(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = ToolContext(cwd=str(root), session_dir=str(tmp_path / "sessions" / "abc123"))

    result = await execute_read_public_copy(
        "call-3", {"path": path, "key": "x"}, None, None, context
    )
    assert result.is_error
    assert "refused" in result.text


@pytest.mark.asyncio
async def test_public_copy_refuses_a_missing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    context = ToolContext(cwd=str(root), session_dir=str(tmp_path / "sessions" / "abc123"))

    result = await execute_read_public_copy(
        "call-4",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.missing",
        },
        None,
        None,
        context,
    )
    assert result.is_error
    assert "key not found" in result.text


@pytest.mark.asyncio
async def test_public_copy_fails_closed_when_the_audit_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _public_repo(tmp_path)
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(blocked))
    context = ToolContext(cwd=str(root), session_dir=str(tmp_path / "sessions" / "abc123"))

    result = await execute_read_public_copy(
        "call-5",
        {
            "path": "local_operator/i18n/catalogues/en/wire.slash.json",
            "key": "wire.slash.credential",
        },
        None,
        None,
        context,
    )
    assert result.is_error
    assert "audit record could not be written" in result.text
    # Fail closed: the content must not leak on the error path.
    assert "Type or paste a secret" not in result.text


def test_the_policy_document_names_its_owner_and_both_lists() -> None:
    """The prose policy is part of the control, not a comment beside it."""
    text = (
        Path(__file__).resolve().parents[3] / "docs/security/redaction-policy.md"
    ).read_text(encoding="utf-8")
    assert "**Owner:** Damian Tran" in text
    assert "### 2.1 A secret" in text
    assert "### 2.2 Not a secret" in text
    assert "read_public_copy" in text
