"""A credential NAME one step away from its VALUE (follow-up to PR #3756).

WHY THIS FILE EXISTS
--------------------
Every name-driven rule reads ``NAME = VALUE``. HCP Terraform's vars API
(``attributes.{key,value}``), GitLab's CI variables API, generic key/value
arrays and Kubernetes ``env:`` entries put the name in a SIBLING field, and a
Terraform-variables map outside a plan puts it in the PARENT key, so the value
was published whole. The same follow-up strips environment qualifiers from
names (``API_KEY_PROD``), recognises a few compound names
(``GOOGLE_SERVICE_ACCOUNT_JSON``, ``…_SLACK_WEBHOOK_URL``), adds the Minerva
``mrva-…`` key shape, and closes three persistence/steer gaps.

Operator constraint (binding): mask VALUES in recognised structures only; key
names, package names and identifiers are never redacted; nothing becomes more
restrictive in general. Every positive row has a negative twin.

Every value here is synthetic and assembled from pieces.
"""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.redaction_shapes import (
    REDACTION_MARKER,
    StructureStreamGuard,
    credential_dump_notice,
    is_credential_name,
    is_strong_credential_name,
    scrub_shapes,
)

SECRET = "fAkE" + "s3cr3tK3yV4lu3" * 2 + "QQzz"
OTHER = "zzQQ" + "3ul4Vy3Kt3rc3s" * 2 + "EkAf"


def _gutter(text: str) -> str:
    return "\n".join(f"{i + 1:>6}| {line}" for i, line in enumerate(text.splitlines()))


def _forms(obj: Any) -> list[str]:
    pretty = json.dumps(obj, indent=2)
    return [json.dumps(obj), json.dumps(obj, separators=(",", ":")), pretty, _gutter(pretty)]


HCP_VARS = {
    "data": [
        {
            "id": "var-1",
            "type": "vars",
            "attributes": {"key": "AWS_SECRET_KEY", "value": SECRET, "sensitive": False},
        },
        {"id": "var-2", "type": "vars", "attributes": {"key": "region", "value": "us-east-1"}},
    ]
}
GITLAB_VARS = [
    {"variable_type": "env_var", "key": "DEPLOY_TOKEN", "value": SECRET, "protected": True},
    {"variable_type": "env_var", "key": "bucket", "value": "my-artifacts-bucket"},
]
NAME_VALUE = [{"name": "DB_PASSWORD", "value": SECRET}, {"name": "region", "value": "us-east-1"}]
VALUE_FIRST = [{"value": SECRET, "key": "API_KEY_PROD"}, {"value": "eu-west-1", "key": "region"}]
PARENT_KEY = {"vars": {"AWS_SECRET_KEY": {"value": SECRET}, "region": {"value": "us-east-1"}}}


@pytest.mark.parametrize("obj", [HCP_VARS, GITLAB_VARS, NAME_VALUE, VALUE_FIRST, PARENT_KEY])
def test_a_value_named_by_its_sibling_or_parent_is_masked_and_names_kept(obj: Any) -> None:
    for text in _forms(obj):
        out = scrub_shapes(text)
        assert SECRET not in out, text[:80]
        assert REDACTION_MARKER in out
        # Names and the neighbouring non-credential values keep their bytes.
        for kept in ("AWS_SECRET_KEY", "DEPLOY_TOKEN", "DB_PASSWORD", "API_KEY_PROD"):
            assert (kept in text) == (kept in out)
        for kept in ("us-east-1", "eu-west-1", "my-artifacts-bucket", "region", "bucket"):
            assert (kept in text) == (kept in out)


def test_the_guttered_form_keeps_every_line_and_gutter() -> None:
    text = _gutter(json.dumps(HCP_VARS, indent=2))
    out = scrub_shapes(text)
    assert len(out.splitlines()) == len(text.splitlines())
    assert [line[:8] for line in out.splitlines()] == [line[:8] for line in text.splitlines()]


@pytest.mark.parametrize(
    "text",
    [
        "variables:\n  - key: DEPLOY_TOKEN\n    value: <V>\n"
        "  - key: region\n    value: us-east-1\n",
        'env:\n- name: DB_PASSWORD\n  value: "<V>"\n- name: region\n  value: us-east-1\n',
        "vars:\n  AWS_SECRET_KEY:\n    value: <V>\n  region:\n    value: us-east-1\n",
        "env:\n- value: '<V>'\n  name: API_KEY_PROD\n- name: region\n  value: us-east-1\n",
    ],
)
def test_yaml_spellings_mask_the_value_only(text: str) -> None:
    text = text.replace("<V>", SECRET)
    for form in (text, _gutter(text)):
        out = scrub_shapes(form)
        assert SECRET not in out
        assert "us-east-1" in out and "region" in out
        assert len(out.splitlines()) == len(form.splitlines())


@pytest.mark.parametrize(
    "obj",
    [
        # Non-credential names: region, bucket, a package, a URL, a JSON config.
        [{"key": "region", "value": OTHER}, {"name": "bucket", "value": OTHER}],
        [{"name": "lodash", "value": OTHER}, {"name": "@types/node", "value": "20.11.5"}],
        [{"key": "SERVICE_URL", "value": "https://api.example.com/" + OTHER}],
        [{"key": "CONFIG_JSON", "value": OTHER}, {"key": "DATABASE_URL_PROD", "value": OTHER}],
        # Counts and cache handles are not credentials.
        [{"name": "max_tokens", "value": "4096"}, {"key": "cache_key", "value": OTHER}],
        # A reference, a template and a store NAME are not values.
        [
            {"name": "API_TOKEN", "value": "${API_TOKEN}"},
            {"name": "DB_PASSWORD", "value": "{{ .Values.db }}"},
        ],
        [{"key": "DEPLOY_TOKEN", "value": "MINERVA_DEPLOY_TOKEN"}],
        # A credential-named key whose sibling NAME is ordinary is judged by the sibling.
        {"secrets": {"name": "region", "value": OTHER}},
        # A Kubernetes ConfigMap keeps its data.
        {"kind": "ConfigMap", "data": {"LOG_LEVEL": "info", "value": OTHER}},
    ],
)
def test_ordinary_key_value_pairs_keep_every_byte(obj: Any) -> None:
    for text in _forms(obj):
        assert scrub_shapes(text) == text


def test_streamed_key_value_split_across_releases_is_masked() -> None:
    # The pipe releases at line boundaries: the name went out first, the value
    # next. Without the lookbehind the spill file kept the value.
    guard = StructureStreamGuard()
    first, _ = guard.scrub_release('[\n  {\n    "key": "DEPLOY_TOKEN",\n')
    second, _ = guard.scrub_release(f'    "value": "{SECRET}"\n  }}\n]\n')
    assert SECRET not in second and f'"value": "{REDACTION_MARKER}"' in second
    assert "DEPLOY_TOKEN" in first


@pytest.mark.parametrize(
    "name",
    [
        "MINERVA_API_KEY_PROD",
        "API_KEY_PROD",
        "API_KEY_STAGING",
        "DB_PASSWORD_2",
        "GITHUB_TOKEN_QA",
        "PGPASSWORD_PROD",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS_JSON",
        "SANCTIONS_GOVERNANCE_SLACK_WEBHOOK_URL",
        "DISCORD_WEBHOOK_URL_PROD",
    ],
)
def test_qualified_and_compound_credential_names(name: str) -> None:
    assert is_credential_name(name)
    assert is_strong_credential_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "PROD_REGION",
        "TEST_URL",
        "SERVICE_URL",
        "DATABASE_URL_PROD",
        "CONFIG_JSON",
        "PACKAGE_JSON",
        "WEBHOOK_URL",
        "GITHUB_WEBHOOK_URL",
        "SLACK_CHANNEL_ID",
        "API_KEY_COUNT",
        "PROD",
        "KEY_ID_PROD",
    ],
)
def test_qualifier_stripping_does_not_widen_ordinary_names(name: str) -> None:
    assert not is_credential_name(name)


@pytest.mark.parametrize(
    "name", ["MINERVA_API_KEY_PROD", "GOOGLE_SERVICE_ACCOUNT_JSON", "TEAM_SLACK_WEBHOOK_URL"]
)
def test_qualified_assignments_mask_the_value_and_keep_the_name(name: str) -> None:
    out = scrub_shapes(f"{name}={SECRET}\n")
    assert out == f"{name}={REDACTION_MARKER}\n"


def test_a_chat_webhook_url_is_masked_under_its_name_and_an_ordinary_url_is_not() -> None:
    hook = "https://hooks.example.com/services/T0FAKE/B0FAKE/" + OTHER
    assert hook not in scrub_shapes(f"SANCTIONS_GOVERNANCE_SLACK_WEBHOOK_URL={hook}")
    for name in ("SERVICE_URL", "GITHUB_WEBHOOK_URL", "WEBHOOK_URL"):
        assert scrub_shapes(f"{name}={hook}") == f"{name}={hook}"


def test_minerva_api_keys_are_masked_by_prefix() -> None:
    body = ("Ab3_" * 11)[:43]  # base64.RawURLEncoding of 32 bytes is 43 chars
    for prefix in ("mrva-live-", "mrva-dev-", "mrva-qa-live-", "mrva-qa-dev-"):
        assert scrub_shapes(f"key {prefix}{body} end") == f"key {REDACTION_MARKER} end"
    fallback = "mrva-live-" + str(uuid.UUID(int=0x1234))  # the RNG-failure spelling
    assert fallback not in scrub_shapes(fallback)


@pytest.mark.parametrize(
    "text",
    [
        'require.Equal(t, "mrva-dev-", imported.Application.DisplayPrefix)',
        'ResolveApplicationAuthentication(ctx, "mrva-unknown")',
        "mrva-live-short-name",
        "the mrva-live-<token> format",
    ],
)
def test_the_minerva_prefix_alone_is_not_a_key(text: str) -> None:
    assert scrub_shapes(text) == text


def test_two_collapsed_values_in_a_guttered_plan_keep_every_line_number() -> None:
    # PR #3756 QA Q11: the lines BETWEEN two masked multi-line values slid up.
    def change(address: str, tags: list[str]) -> dict[str, Any]:
        return {
            "address": address,
            "change": {
                "before": None,
                "after": {"tags": tags, "name": address},
                "after_sensitive": {"tags": True},
            },
        }

    plan = {
        "format_version": "1.2",
        "resource_changes": [change("a.one", ["x1", "x2", "x3"]), change("a.two", ["y1", "y2"])],
    }
    source = json.dumps(plan, indent=2)
    text = _gutter(source)
    out = scrub_shapes(text)
    assert "x2" not in out and "y2" not in out
    original, lines = text.splitlines(), out.splitlines()
    assert len(lines) == len(original)
    for index, line in enumerate(original):
        if '"address"' in line or '"name"' in line:
            assert lines[index] == line, index


@pytest.mark.parametrize(
    ("command", "fires"),
    [
        ("aws --region us-east-1 secretsmanager get-secret-value --secret-id x", True),
        ("aws --profile p --region=eu-west-1 secretsmanager get-secret-value --secret-id x", True),
        ("aws --no-cli-pager secretsmanager get-secret-value --secret-id x", True),
        ("aws --region us-east-1 secretsmanager get-secret-value --secret-id x > out", False),
        ("aws --region us-east-1 secretsmanager describe-secret --secret-id x", False),
        ("V=`aws secretsmanager get-secret-value --secret-id x --query SecretString`", False),
        ("V=`aws --region r secretsmanager get-secret-value --secret-id x`", False),
        ("V=`kubectl get secret x -o jsonpath='{.data.K}' | base64 -d`", False),
        ("kubectl get secret x -o jsonpath='{.data.K}' | base64 -d", True),
        ("aws s3 ls secretsmanager get-secret-value", False),
    ],
)
def test_aws_global_options_steer_and_backtick_capture_is_quiet(command: str, fires: bool) -> None:
    assert (credential_dump_notice(command) is not None) is fires


@pytest.mark.asyncio
async def test_every_details_string_is_masked_and_other_fields_kept(tmp_path: Any) -> None:
    from local_operator.harness.loop import AgentLoop
    from local_operator.variables import VariableStore

    store = VariableStore(cwd=str(tmp_path))
    details = {
        "rows": [{"line": f"DB_PASSWORD={SECRET}"}],
        "url": "https://example.com/x",
        "path": "/tmp/project/file.txt",
        "job_ids": ("job-1", "job-2"),
        "count": 3,
        "useless": False,
    }
    out = await AgentLoop._redact_details(details, store.redact, "jobs", None)
    assert out is not None
    assert SECRET not in json.dumps(out)
    assert out["rows"] == [{"line": f"DB_PASSWORD={REDACTION_MARKER}"}]
    assert out["url"] == details["url"] and out["path"] == details["path"]
    assert out["job_ids"] == ("job-1", "job-2") and out["count"] == 3 and out["useless"] is False
    assert SECRET in details["rows"][0]["line"], "the executor's own details are untouched"
    # A string with no whitespace, ``=`` or ``:`` is an IDENTIFIER its reader matches
    # exactly (FP pass: ``questions[].id``/``model`` values were masked by the
    # vendor-prefix rule); the same spelling inside prose is still scrubbed.
    ident = "tvly" + "-" + "abcdefghijklmnop"
    assert scrub_shapes(ident) != ident, "precondition: the table masks this spelling"
    ids = {"question_ids": [ident], "model": ident, "text": f"saw {ident} here"}
    kept = await AgentLoop._redact_details(ids, store.redact, "ask", None)
    assert kept is not None
    assert kept["question_ids"] == [ident] and kept["model"] == ident
    assert ident not in kept["text"]
    # Nothing long enough to carry a credential: returned as is, no thread hop.
    small = {"job_id": "j1", "pid": 4}
    assert await AgentLoop._redact_details(small, store.redact, "jobs", None) is small


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="POSIX modes only")
@pytest.mark.asyncio
async def test_transcript_is_owner_only_new_and_on_next_append(tmp_path: Path) -> None:
    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    transcript = Transcript(tmp_path / "sess")
    await transcript.append_message(Message.user("hello", input_mode="typed"))
    assert stat.S_IMODE(transcript.path.stat().st_mode) == 0o600
    os.chmod(transcript.path, 0o644)  # a transcript written by an older build
    await transcript.append_message(Message.user("again", input_mode="typed"))
    assert stat.S_IMODE(transcript.path.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_spill_search_returns_scrubbed_text(tmp_path: Path) -> None:
    # Item 5: the spill is written from the scrubbed stream, so a ``?q=`` search
    # that returns the value line WITHOUT its sibling name line cannot leak it.
    from local_operator.harness.types import TextContent, ToolContext
    from local_operator.tools.registry import create_tools
    from local_operator.variables import VariableStore

    context = ToolContext(cwd=str(tmp_path), session_id="unit-test")
    context.variables = VariableStore(cwd=str(tmp_path))
    tools = {tool.name: tool for tool in create_tools(context)}
    # Ordinary rows that no rule touches, so the search below matches only the
    # credential's two lines (a numbered ``"key": "name_123"`` is masked by the
    # pre-existing weak-name assignment rule and would flood the first page).
    rows: list[dict[str, str]] = [{"key": "region", "value": "us-east-1"}] * 4000
    rows.insert(2000, {"key": "AWS_SECRET_KEY", "value": SECRET})
    path = tmp_path / "vars.json"
    path.write_text(json.dumps({"data": rows}, indent=2))

    def text_of(result: Any) -> str:
        return "".join(block.text for block in result.content if isinstance(block, TextContent))

    result = await tools["bash"].execute("c1", {"command": f"cat {path}"}, None, None, context)
    handle = re.search(r"spill://[0-9a-f]+", text_of(result))
    assert handle is not None, "the output must have spilled for this test to mean anything"
    query = handle.group(0) + "?q=AWS_SECRET_KEY|" + SECRET[:6] + "|redacted"
    found = text_of(await tools["read"].execute("c2", {"path": query}, None, None, context))
    assert "2 of 2 match(es)" in found, found[:200]
    assert "AWS_SECRET_KEY" in found and REDACTION_MARKER in found
    assert SECRET not in found


def test_a_dsn_value_is_left_to_the_dsn_rule_which_keeps_user_and_host() -> None:
    # The corpus row "a DSN inside a JSON env array": the indirect rule runs
    # before the table and, unguarded, masked the whole URL and dropped the hit.
    text = json.dumps({"name": "MONGO_DSN", "value": f"mongodb://agent:{SECRET}@mongo.internal/db"})
    out = scrub_shapes(text)
    assert SECRET not in out
    assert "mongodb://agent:" in out and "@mongo.internal/db" in out


def test_the_yaml_walk_is_linear_on_a_log_of_value_lines() -> None:
    # Unbounded, each ``value:`` line walked every earlier line: 20,000 lines took 58 s.
    import time

    text = "".join(f"value: {i}\n" for i in range(20000))
    started = time.perf_counter()
    assert scrub_shapes(text) == text
    assert time.perf_counter() - started < 5


@pytest.mark.parametrize(
    "text",
    [
        # FP pass, 2026-11: a Secret RESOURCE path beside a sentinel phrase.
        json.dumps(
            {
                "name": "opensearch/opensearch-admin-credentials",
                "keys": ["OPENSEARCH_INITIAL_ADMIN_PASSWORD"],
                "value": "VALUE-NEVER-PRINTED-HERE",
            }
        ),
        # FP pass, 2026-11: a hyphenated ``-prod`` resource name is not stripped.
        "- minerva-nexus/secrets-prod: failedSync\\nNeeds sync/roll both.",
    ],
)
def test_resource_names_are_not_variable_names(text: str) -> None:
    assert scrub_shapes(text) == text


# --- PR #3789 remediation round 1 -------------------------------------------

POD_UUID = str(uuid.UUID(int=0x3F2B8C1D9A4E4B7F8C2D1E5A6B7C8D9E))


@pytest.mark.asyncio
async def test_a_details_only_match_masks_that_copy_and_registers_nothing(tmp_path: Any) -> None:
    """R1: details is a COPY; a match only it shows must not mask later output."""
    from local_operator.harness.loop import AgentLoop
    from local_operator.variables import VariableStore

    store = VariableStore(cwd=str(tmp_path), env={})
    details = {"message": f"wake: export DB_PASSWORD={SECRET} then retry"}
    out = await AgentLoop._redact_details(details, store.redact, "wake", None)
    assert out is not None and SECRET not in json.dumps(out), "the copy is still masked"
    # Registered, the value would be masked in every later, unrelated result.
    assert store.redact(f"later output mentions {SECRET} alone") == (
        f"later output mentions {SECRET} alone"
    )
    # The content path still registers (containment unchanged there).
    store.redact(f"DB_PASSWORD={OTHER}")
    assert OTHER not in store.redact(f"later {OTHER}")


@pytest.mark.asyncio
async def test_the_reported_kubectl_pod_uuid_stays_readable_later(tmp_path: Any) -> None:
    """R1 + R4: the review's repro, end to end through the session redactor."""
    from local_operator.harness.loop import AgentLoop
    from local_operator.variables import VariableStore

    store = VariableStore(cwd=str(tmp_path), env={})
    text = f"wake: verify psql restore then run: kubectl logs -p {POD_UUID} --tail 50"
    out = await AgentLoop._redact_details({"message": text}, store.redact, "wake", None)
    assert out == {"message": text}
    assert store.redact(f"pod {POD_UUID} is Running") == f"pod {POD_UUID} is Running"


@pytest.mark.asyncio
async def test_an_identifier_details_string_gets_the_known_values_pass(tmp_path: Any) -> None:
    """R3: no shapes on identifiers, but a value the session already holds is masked."""
    from local_operator.harness.loop import AgentLoop
    from local_operator.variables import VariableStore

    store = VariableStore(cwd=str(tmp_path), env={})
    assert store.register_redaction(SECRET)
    ident = "tvly" + "-" + "abcdefghijklmnop"
    out = await AgentLoop._redact_details(
        {"echo": SECRET, "model": ident}, store.redact, "jobs", None
    )
    assert out == {"echo": REDACTION_MARKER, "model": ident}


@pytest.mark.parametrize(
    "obj",
    [
        {"name": "Token", "value": "In Progress Review"},
        {"key": "api_key", "value": "Please set via env var"},
        {"key": "DB_PASSWORD", "value": "correct horse battery staple"},
        {"vars": {"AWS_SECRET_KEY": {"value": "set this in the vault first"}}},
    ],
)
def test_a_multi_word_value_named_one_step_away_is_prose(obj: Any) -> None:
    """R2: the indirect rule takes the direct rule's value class (no whitespace)."""
    for text in _forms(obj):
        assert scrub_shapes(text) == text


def test_a_multi_word_yaml_value_named_one_step_away_is_prose() -> None:
    text = "api_key:\n  description: shown in the form\n  value: Please set via env var\n"
    assert scrub_shapes(text) == text
    twin = f"api_key:\n  description: shown in the form\n  value: {SECRET}\n"
    assert SECRET not in scrub_shapes(twin) and "api_key:" in scrub_shapes(twin)


@pytest.mark.parametrize(
    "text",
    [
        "psql -h db.internal -p 5432 -U app appdb",
        "pg_dump -p 6432 -Fc appdb",
        "pg_restore -h h -p 5433 -d appdb dump.bin",
        "redis-cli -h cache -p 6379 ping",
        "mysql -h h -P 3306 -u app appdb",
        f"psql -h h -p {POD_UUID}",
        f"psql restore done; kubectl logs -p {POD_UUID} api",
        "verify psql restore then run: kubectl logs -p api-7f9c8d-x2k4z --tail 50",
    ],
)
def test_a_port_uuid_or_kubectl_previous_flag_is_not_a_password(text: str) -> None:
    """R4: ``-p`` owned by a port, an id or ``kubectl logs`` keeps its argument."""
    assert scrub_shapes(text) == text


@pytest.mark.parametrize(
    "text",
    [
        f"mysql -u root -p{SECRET} appdb",
        f"mysql -h h -P 3306 -u root -p{SECRET} appdb",
        f"redis-cli -h cache -p 6379 -a {SECRET} ping",
        f"psql x; kubectl exec pod -- mysql -u r -p{SECRET} db",
    ],
)
def test_a_real_inline_password_still_masks(text: str) -> None:
    out = scrub_shapes(text)
    assert SECRET not in out and REDACTION_MARKER in out


def test_a_key_or_name_fields_value_is_an_identifier() -> None:
    """QA pre-existing: ``"key": "setting_1042"`` is a NAME, never masked."""
    rows = [{"key": f"setting_{n}", "value": f"v{n}"} for n in range(1000, 1100)]
    rows.append({"key": "STRIPE_API_KEY_PROD", "value": SECRET})
    rows.append({"name": "release_20261111", "value": "on"})
    for text in _forms(rows):
        out = scrub_shapes(text)
        assert SECRET not in out, "the value under the credential name is still masked"
        assert out.count(REDACTION_MARKER) == 1
        assert "setting_1042" in out and "release_20261111" in out
    # Not a widening past the field name: a credential-named field still masks.
    assert SECRET not in scrub_shapes(json.dumps({"api_key": SECRET}))
    assert SECRET not in scrub_shapes(json.dumps({"private_key": SECRET}))


def test_a_same_line_count_plan_mask_skips_the_line_diff(monkeypatch: Any) -> None:
    """R5: equal line counts take the zip; the diff budget is not spent."""
    import local_operator.redaction_shapes as shapes

    def _fail(*_: Any) -> None:
        raise AssertionError("line diff ran on an equal-line-count mask")

    monkeypatch.setattr(shapes, "_changed_line_blocks", _fail)
    monkeypatch.setattr(
        shapes,
        "_scrub_plain_terraform_plan_json",
        lambda text: text.replace(SECRET, REDACTION_MARKER),
    )
    text = _gutter(f'{{\n  "a": "{SECRET}",\n  "b": 1\n}}')
    out = shapes._scrub_guttered_plan(text)
    assert out is not None and SECRET not in out
    assert out.splitlines()[1].startswith("     2| ")
