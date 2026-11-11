"""Credentials inside recognised STRUCTURES and ENCODINGS (2026-11 incident).

WHY THIS FILE EXISTS
--------------------
``kubectl get secret internal-jobs-sc -o yaml`` kept a Mongo Atlas DSN password
and a GCP service-account key in a transcript: the Secret's ``data:`` values are
base64, and every rule in the shape table reads plaintext. The same blind spot
covered a %xx-encoded DSN, a JSON-escaped (``\\/``) DSN and a service-account
key's ``private_key_id``. Each leak is a row below; each NEGATIVE row pins the
operator's "do not overcorrect" constraint — ConfigMaps, digests, SHAs, key
names, package and object names and base64 of ordinary text keep their bytes.

Every value here is synthetic and assembled from pieces, so the test source is
not itself credential-shaped in a transcript that reads it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.parse
from typing import Any

import pytest

from local_operator.redaction_shapes import (
    REDACTION_MARKER,
    StructureStreamGuard,
    _DecodeBudget,
    _scrub_encoded_credentials,
    _scrub_escaped_json_strings,
    credential_dump_notice,
    scrub_shapes,
    scrub_shapes_for_arguments,
    scrub_shapes_with_hits,
)

PASSWORD = "Zx9fake" + "Passw0rdQ"
DSN = f"mongodb+srv://fakeuser:{PASSWORD}@cluster0.example.mongodb.net/db?retryWrites=true"
KEY_ID = "0123456789abcdef" * 2 + "01234567"
PEM_BODY = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC" + "A" * 60
PEM = "-----BEGIN " + "PRIVATE KEY-----\n" + PEM_BODY + "\n-----END " + "PRIVATE KEY-----\n"
SA_JSON = json.dumps(
    {
        "type": "service_account",
        "project_id": "fake-proj",
        "private_key_id": KEY_ID,
        "private_key": PEM,
        "client_email": "jobs@fake-proj.iam.gserviceaccount.com",
    }
)
DOCKER_CONFIG = json.dumps(
    {"auths": {"ghcr.io": {"auth": base64.b64encode(f"robot:{PASSWORD}".encode()).decode()}}}
)


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def secret_yaml(kind: str = "Secret") -> str:
    return (
        "apiVersion: v1\n"
        "data:\n"
        f"  MONGO_URI: {b64(DSN)}\n"
        f"  sa.json: {b64(SA_JSON)}\n"
        f"  LOG_LEVEL: {b64('info')}\n"
        f"kind: {kind}\n"
        "metadata:\n"
        "  name: internal-jobs-sc\n"
        "  namespace: jobs\n"
        "type: Opaque\n"
    )


def assert_no_leak(text: str) -> None:
    for fragment in (PASSWORD, KEY_ID, PEM_BODY[:24], b64(DSN)[:24], b64(SA_JSON)[:24]):
        assert fragment not in text


# --- the incident's leaks ----------------------------------------------------


def test_kubectl_secret_yaml_masks_every_data_value_and_keeps_key_names() -> None:
    out = scrub_shapes(secret_yaml())
    assert_no_leak(out)
    assert f"  LOG_LEVEL: {REDACTION_MARKER}\n" in out  # every value, not only shaped ones
    for kept in ("MONGO_URI:", "sa.json:", "LOG_LEVEL:", "name: internal-jobs-sc", "type: Opaque"):
        assert kept in out


def test_kubectl_secret_json_masks_data_values_and_keeps_key_names() -> None:
    doc = {
        "apiVersion": "v1",
        "data": {"MONGO_URI": b64(DSN), "LOG_LEVEL": b64("info")},
        "kind": "Secret",
        "metadata": {"name": "internal-jobs-sc"},
    }
    for text in (json.dumps(doc, indent=4), json.dumps(doc)):
        out = scrub_shapes(text)
        assert_no_leak(out)
        assert b64("info") not in out
        assert json.loads(out)["data"] == {
            "MONGO_URI": REDACTION_MARKER,
            "LOG_LEVEL": REDACTION_MARKER,
        }
        assert json.loads(out)["metadata"]["name"] == "internal-jobs-sc"


def test_secret_list_masks_secrets_and_leaves_the_configmap_item() -> None:
    text = (
        "apiVersion: v1\nitems:\n"
        "- apiVersion: v1\n  data:\n"
        f"    LOG_LEVEL: {b64('info')}\n"
        "  kind: ConfigMap\n  metadata:\n    name: app-config\n"
        "- apiVersion: v1\n  data:\n"
        f"    TOKEN_A: {b64('plain-value-one')}\n"
        "  kind: Secret\n  metadata:\n    name: app-secret\n"
        "kind: List\n"
    )
    out = scrub_shapes(text)
    assert f"    LOG_LEVEL: {b64('info')}\n" in out
    assert f"    TOKEN_A: {REDACTION_MARKER}\n" in out


def test_string_data_and_block_scalars_are_masked() -> None:
    text = (
        "apiVersion: v1\nkind: Secret\nmetadata:\n  name: s\nstringData:\n"
        "  config.yaml: |\n    user: admin\n    level: debug\n"
        "  MODE: production\n"
    )
    out = scrub_shapes(text)
    assert "level: debug" not in out and "production" not in out
    assert "  config.yaml: |\n" in out and "  MODE: " in out


def test_string_data_values_are_not_registered_for_session_wide_masking() -> None:
    text = "kind: Secret\nstringData:\n  MODE: production-mode-value\n"
    _, hits = scrub_shapes_with_hits(text)
    assert all(hit.value != "production-mode-value" for hit in hits)


def test_last_applied_annotation_carries_no_data_values() -> None:
    applied = json.dumps(
        {"apiVersion": "v1", "data": {"K": b64("plain-value-one")}, "kind": "Secret"}
    )
    yaml_text = (
        "apiVersion: v1\ndata:\n  K: " + b64("plain-value-one") + "\nkind: Secret\nmetadata:\n"
        "  annotations:\n    kubectl.kubernetes.io/last-applied-configuration: |\n      "
        + applied
        + "\n  name: s\n"
    )
    assert b64("plain-value-one") not in scrub_shapes(yaml_text)
    json_text = json.dumps(
        {
            "data": {"K": b64("plain-value-one")},
            "kind": "Secret",
            "metadata": {
                "annotations": {"kubectl.kubernetes.io/last-applied-configuration": applied}
            },
        },
        indent=2,
    )
    out = scrub_shapes(json_text)
    assert b64("plain-value-one") not in out
    assert "last-applied-configuration" in out


@pytest.mark.parametrize(
    "encoded",
    [
        pytest.param(b64(DSN), id="base64 dsn"),
        pytest.param(
            base64.urlsafe_b64encode(DSN.encode()).decode().rstrip("="), id="base64url dsn"
        ),
        pytest.param(b64(SA_JSON), id="base64 service-account json"),
        pytest.param(b64(DOCKER_CONFIG), id="base64 docker config"),
        pytest.param(b64(b64(DSN)), id="double base64"),
        pytest.param(DSN.encode().hex(), id="hex dsn"),
    ],
)
def test_bare_encoded_credential_is_masked_whole(encoded: str) -> None:
    out = scrub_shapes(f"value: {encoded}\n")
    assert out == f"value: {REDACTION_MARKER}\n"
    _, hits = scrub_shapes_with_hits(encoded)
    assert [hit.label for hit in hits] == ["encoded-credential"]


def test_url_encoded_dsn_is_masked_and_the_parameter_name_kept() -> None:
    text = "https://app.example.com/connect?dsn=" + urllib.parse.quote(DSN, safe="") + "&mode=ro"
    out = scrub_shapes(text)
    assert PASSWORD not in out
    assert out == f"https://app.example.com/connect?dsn={REDACTION_MARKER}&mode=ro"


def test_json_escaped_dsn_is_masked() -> None:
    text = '{"uri": ' + json.dumps(DSN).replace("/", "\\/") + "}"
    assert PASSWORD not in scrub_shapes(text)


def test_partly_encoded_plain_dsn_keeps_the_tables_precise_mask() -> None:
    # The table already masks this spelling (keeping user and host): the decode
    # pass must not widen it to the whole run.
    text = f"postgres://app:p%40ss{PASSWORD}@db.example.com/app"
    out = scrub_shapes(text)
    assert PASSWORD not in out and "db.example.com" in out


@pytest.mark.parametrize(
    "dump", [SA_JSON, json.dumps(json.loads(SA_JSON), indent=2), json.dumps(SA_JSON)]
)
def test_service_account_private_key_id_is_masked_identifiers_kept(dump: str) -> None:
    out = scrub_shapes(dump)
    assert KEY_ID not in out and PEM_BODY[:24] not in out
    assert "fake-proj" in out and "jobs@fake-proj.iam.gserviceaccount.com" in out


def test_argument_pass_masks_encoded_credentials_but_not_secret_manifest_words() -> None:
    manifest = "kind: Secret\nstringData:\n  LOG_LEVEL: info\n  MONGO_URI_B64: " + b64(DSN) + "\n"
    out = scrub_shapes_for_arguments(manifest)
    assert "LOG_LEVEL: info" in out  # code is not rewritten on structure alone
    assert b64(DSN) not in out  # a decoded credential is spelling evidence


@pytest.mark.parametrize(
    "scheme",
    [
        "mongodb",
        "mongodb+srv",
        "postgres",
        "postgresql",
        "mysql",
        "redis",
        "rediss",
        "amqp",
        "amqps",
    ],
)
def test_every_dsn_scheme_masks_its_inline_password(scheme: str) -> None:
    assert PASSWORD not in scrub_shapes(f"{scheme}://app:{PASSWORD}@db.example.com:1234/x")
    assert PASSWORD not in scrub_shapes(b64(f"{scheme}://app:{PASSWORD}@db.example.com:1234/x"))


# --- negative controls: the "do not overcorrect" half ------------------------

NEGATIVE = [
    pytest.param(
        "apiVersion: v1\ndata:\n"
        f"  LOG_LEVEL: {b64('info')}\n  REGION: {b64('europe-west1')}\n"
        "  MONGO_HOST: cluster0.example.net\n"
        "kind: ConfigMap\nmetadata:\n  name: app-config\n",
        id="configmap base64 data",
    ),
    pytest.param(
        json.dumps(
            {"kind": "ConfigMap", "data": {"MONGO_URI": "mongo-host", "LEVEL": b64("info")}},
            indent=2,
        ),
        id="configmap json",
    ),
    pytest.param("sha256:" + hashlib.sha256(b"layer").hexdigest(), id="sha256 digest"),
    pytest.param(
        "integrity sha512-" + base64.b64encode(hashlib.sha512(b"pkg").digest()).decode(),
        id="sri hash",
    ),
    pytest.param("commit " + hashlib.sha1(b"x").hexdigest() + " fix: thing", id="git sha"),
    pytest.param(
        "data:image/png;base64," + base64.b64encode(bytes(range(256)) * 4).decode(), id="image"
    ),
    pytest.param("MONGO_URI DATABASE_PASSWORD_FILE GOOGLE_APPLICATION_CREDENTIALS", id="key names"),
    pytest.param(
        "pip install local-operator mda-category-sources minerva-core-engine", id="package names"
    ),
    pytest.param(
        "secret/internal-jobs-sc configured\nserviceaccount/jobs-runner created",
        id="k8s object names",
    ),
    pytest.param(
        "echo " + b64("hello world, this is plain text with no credential"),
        id="base64 of plain text",
    ),
    pytest.param("ref " + b64("https://example.com/path?q=1") + " ok", id="base64 of a plain URL"),
    pytest.param(
        "https://example.com/search?q=hello%20world%2C%20again&x=%2Fpath", id="url-encoded prose"
    ),
    pytest.param(
        '{"path": "C:\\/Users\\/me\\/file.txt", "msg": "caf\\u00e9"}', id="json-escaped path"
    ),
    pytest.param("kind: Secret\nmetadata:\n  name: s\ntype: Opaque\n", id="secret without data"),
    pytest.param(
        "Name:         internal-jobs-sc\nType:  Opaque\n\nData\n====\nMONGO_URI:  112 bytes\n",
        id="kubectl describe secret",
    ),
    pytest.param(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "p",
                "client_email": "a@p.iam.gserviceaccount.com",
            }
        ),
        id="service account without key",
    ),
]


NEGATIVE.append(
    pytest.param(
        "dump " + 'call = handle(data, Conf.Map{"[redacted]": value})'.encode().hex(),
        id="hex of code that already carries the marker (FP measurement)",
    )
)


@pytest.mark.parametrize("text", NEGATIVE)
def test_ordinary_text_is_byte_identical(text: str) -> None:
    assert scrub_shapes(text) == text
    assert scrub_shapes_for_arguments(text) == text


def test_configmap_keeps_ordinary_values_and_loses_only_an_encoded_credential() -> None:
    # A credential is a credential in a ConfigMap too — but only the token that
    # DECODES to one goes; the ConfigMap's other values are not Secret data.
    out = scrub_shapes(secret_yaml("ConfigMap"))
    assert f"  MONGO_URI: {REDACTION_MARKER}\n" in out
    assert f"  LOG_LEVEL: {b64('info')}\n" in out
    assert_no_leak(out)


# --- the live stream ---------------------------------------------------------


def test_stream_hold_keeps_a_data_block_until_its_kind_line_arrives() -> None:
    head = "apiVersion: v1\ndata:\n  MONGO_URI: " + b64("plain-value-one") + "\n"
    assert StructureStreamGuard().hold(head, len(head), final=False) == len("apiVersion: v1\n")
    whole = head + "kind: Secret\n"
    assert StructureStreamGuard().hold(whole, len(whole), final=False) == len(whole)
    assert StructureStreamGuard().hold("plain\nlines\n", 12, final=False) == 12


def _published_line_by_line(text: str) -> bytes:
    from local_operator.tools.builtin import _PipeRedactor

    redactor = _PipeRedactor([])
    published = b""
    for line in text.splitlines(keepends=True):
        published += redactor.feed(line.encode())
    return published + redactor.feed(b"", final=True)


def test_pipe_redactor_never_publishes_a_secret_value_split_across_reads() -> None:
    text = secret_yaml().replace(b64(DSN), b64("plain-value-one"))
    published = _published_line_by_line(text)
    assert b64("plain-value-one").encode() not in published
    assert b"MONGO_URI: " + REDACTION_MARKER.encode() in published


def test_the_stream_hold_is_load_bearing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Control: without the hold, each line is released before ``kind: Secret``
    # arrives and an unshaped Secret value paints live.
    from local_operator.tools import builtin

    monkeypatch.setattr(builtin.StructureStreamGuard, "hold", lambda self, text, cut, final: cut)
    text = secret_yaml().replace(b64(DSN), b64("plain-value-one"))
    assert b64("plain-value-one").encode() in _published_line_by_line(text)


# --- the steer, never a block ------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "kubectl get secret internal-jobs-sc -o yaml",
        "kubectl get secrets -A -o yaml",
        "kubectl get secret x -ojson",
        "kubectl get secret x --output=yaml",
        "kubectl -n jobs get secrets x --output json",
    ],
)
def test_secret_dump_commands_get_a_steering_notice(command: str) -> None:
    notice = credential_dump_notice(command)
    assert notice is not None and "jsonpath" in notice


def test_key_listing_pipeline_gets_no_notice() -> None:
    assert credential_dump_notice("kubectl get secret x -o jsonpath='{.data}' | jq 'keys'") is None


# --- bounded cost ------------------------------------------------------------


def test_decode_pass_stays_fast_on_a_large_token_heavy_output() -> None:
    # 2 MB of digests and base64 blobs: the worst ordinary shape for this pass.
    def row(i: int) -> str:
        seed = str(i).encode()
        digest = base64.b64encode(hashlib.sha512(seed).digest()).decode()
        return f"{hashlib.sha256(seed).hexdigest()} {digest}"

    lines = [row(i) for i in range(9000)]
    text = "\n".join(lines)
    # The decode pass ALONE: the pre-existing table's own cost on this input is
    # not what this bound is about.
    started = time.perf_counter()
    assert _scrub_encoded_credentials(text, [], None, 0, _DecodeBudget()) == text
    assert time.perf_counter() - started < 1.0


# --- Terraform plan JSON (second incident: TFC ``/plans/<id>/json-output``) ------

TF_DB_PASSWORD = "tf-fake-" + "Sensitive9Q"
TF_TOKEN = "tf-fake-" + "AttrSecret7Z"


def terraform_plan() -> dict[str, Any]:
    """A synthetic ``terraform show -json`` plan: one sensitive variable, one
    sensitive attribute, and non-sensitive values that must survive."""
    return {
        "format_version": "1.2",
        "terraform_version": "1.9.0",
        "variables": {
            "db_password": {"value": TF_DB_PASSWORD},
            "region": {"value": "europe-west1"},
        },
        "planned_values": {
            "root_module": {
                "resources": [
                    {
                        "address": "google_sql_user.app",
                        "type": "google_sql_user",
                        "name": "app",
                        "values": {"name": "app", "password": TF_DB_PASSWORD, "instance": "db-1"},
                        "sensitive_values": {"password": True},
                    }
                ]
            },
            "outputs": {"api_token": {"sensitive": True, "value": TF_TOKEN}},
        },
        "resource_changes": [
            {
                "address": "google_sql_user.app",
                "type": "google_sql_user",
                "name": "app",
                "change": {
                    "actions": ["create"],
                    "before": None,
                    "after": {
                        "name": "app",
                        "password": TF_DB_PASSWORD,
                        "instance": "db-1",
                        "labels": {"token": TF_TOKEN, "team": "jobs"},
                    },
                    "before_sensitive": False,
                    "after_sensitive": {"password": True, "labels": {"token": True}},
                },
            }
        ],
        "configuration": {
            "root_module": {
                "variables": {
                    "db_password": {"sensitive": True},
                    "region": {"default": "europe-west1"},
                }
            }
        },
    }


@pytest.mark.parametrize("indent", [None, 2], ids=["compact (TFC API)", "pretty"])
def test_terraform_plan_masks_exactly_the_paths_it_marks_sensitive(indent: int | None) -> None:
    text = "exit code: 0\n--- stdout ---\n" + json.dumps(terraform_plan(), indent=indent) + "\n"
    out = scrub_shapes(text)
    assert TF_DB_PASSWORD not in out and TF_TOKEN not in out
    plan = json.loads(out[out.index("{") :])
    assert plan["variables"]["db_password"]["value"] == REDACTION_MARKER
    assert plan["variables"]["region"]["value"] == "europe-west1"
    after = plan["resource_changes"][0]["change"]["after"]
    assert after == {
        "name": "app",
        "password": REDACTION_MARKER,
        "instance": "db-1",
        "labels": {"token": REDACTION_MARKER, "team": "jobs"},
    }
    assert plan["resource_changes"][0]["change"]["before"] is None
    values = plan["planned_values"]["root_module"]["resources"][0]["values"]
    assert values == {"name": "app", "password": REDACTION_MARKER, "instance": "db-1"}
    assert plan["planned_values"]["outputs"]["api_token"]["value"] == REDACTION_MARKER
    assert plan["resource_changes"][0]["address"] == "google_sql_user.app"


def test_terraform_redacted_plan_and_truncated_plan_are_left_to_the_plain_rules() -> None:
    redacted = terraform_plan()
    redacted["variables"]["db_password"]["value"] = None
    redacted["resource_changes"][0]["change"]["after"]["password"] = None
    redacted["resource_changes"][0]["change"]["after"]["labels"]["token"] = None
    redacted["planned_values"]["root_module"]["resources"][0]["values"]["password"] = None
    redacted["planned_values"]["outputs"]["api_token"]["value"] = None
    text = json.dumps(redacted)
    assert scrub_shapes(text) == text
    # Truncated JSON: not parsed, not guessed at — byte-identical when nothing
    # in it is credential-shaped.
    no_secrets = json.dumps(terraform_plan()).replace(TF_DB_PASSWORD, "x").replace(TF_TOKEN, "y")
    assert scrub_shapes(no_secrets[:-40]) == no_secrets[:-40]


@pytest.mark.parametrize(
    ("command", "fires"),
    [
        ("curl -s https://app.terraform.io/api/v2/plans/plan-AbC123/json-output", True),
        ("curl -s https://app.terraform.io/api/v2/plans/plan-AbC123/json-output-redacted", False),
        ("terraform show -json tfplan", True),
        ("terraform show tfplan", False),
        ("aws secretsmanager get-secret-value --secret-id prod/db", True),
        ("aws secretsmanager describe-secret --secret-id prod/db", False),
        ("aws secretsmanager list-secrets --query 'SecretList[].Name'", False),
    ],
)
def test_terraform_and_aws_secret_steers(command: str, fires: bool) -> None:
    notice = credential_dump_notice(command)
    assert (notice is not None) is fires
    if notice:
        assert "`" in notice and len(notice) <= 88


@pytest.mark.parametrize(
    ("command", "fires"),
    [
        # Real spellings from the false-positive measurement (names changed).
        ("kubectl --context prod -n jobs get secret app-sc -o json | python3 -c 'x'", True),
        ("kubectl --context prod -n jobs get secret app-sc -o json | jq -r '.data.BUCKET'", True),
        (
            "kubectl --context prod -n jobs get secret app-sc -o json 2>/dev/null"
            " | jq -r '.data | keys[]'",
            False,
        ),
        ("kubectl --context prod -n jobs get secrets 2>&1 | head -10", False),
        ("kubectl get secrets", True),  # the flag-free listing keeps its pre-existing steer
        ("terraform show -json tfplan 2>&1 >/dev/null | grep -E 'Warning'", False),
        ("terraform show -json tfplan > plan.json", True),
    ],
)
def test_steers_fire_on_value_dumps_and_not_on_name_listings(command: str, fires: bool) -> None:
    assert (credential_dump_notice(command) is not None) is fires


# --- round 1 remediation (PR #3756 review R1-R5/N2, QA Q1-Q6) -------------------
#
# Every value below is synthetic. ``OPAQUE`` decodes to text with no credential
# shape, so ONLY the structure rule can mask it: a leak of it is a structure-rule
# leak, not something the decode pass hides.

OPAQUE = b64("zq9-Lm2-pw-for-jobs-77")
SENSITIVE = "Zq9pw-db-" + "admin-77"


def _stream(parts: list[bytes]) -> str:
    from local_operator.tools.builtin import _PipeRedactor

    redactor = _PipeRedactor([])
    out = b"".join(redactor.feed(part) for part in parts)
    return (out + redactor.feed(b"", final=True)).decode()


def _secret_json(indent: int) -> str:
    obj = {
        "apiVersion": "v1",
        "data": {"MONGO_URI_B": OPAQUE},
        "kind": "Secret",
        "metadata": {"name": "x"},
        "type": "Opaque",
    }
    return json.dumps(obj, indent=indent) + "\n"


SPLIT_FIXTURES = {
    "kubectl yaml (data before kind)": (
        f"apiVersion: v1\ndata:\n  OTHER: {OPAQUE}\nkind: Secret\nmetadata:\n  name: x\n"
    ),
    "manifest, data last": (
        "apiVersion: v1\nkind: Secret\nmetadata:\n  name: x\ntype: Opaque\n"
        f"data:\n  OTHER: {OPAQUE}\n"
    ),
    "manifest, data then a sibling": (
        f"apiVersion: v1\nkind: Secret\ndata:\n  OTHER: {OPAQUE}\ntype: Opaque\n"
    ),
    "stringData after kind": (
        "apiVersion: v1\nkind: Secret\nmetadata:\n  name: x\nstringData:\n"
        f"  ENDPOINT_AUTH: {OPAQUE[:18]}\n"
    ),
    "kubectl -o json, indent 4": _secret_json(4),
    "kubectl -o json, indent 2": _secret_json(2),
}


@pytest.mark.parametrize("name", sorted(SPLIT_FIXTURES))
def test_live_stream_masks_a_secret_at_every_split_point(name: str) -> None:
    # R1/Q3: the first hold covered kubectl's YAML key order in one read and
    # leaked the rest (193 of 194 split points for pretty JSON).
    text = SPLIT_FIXTURES[name]
    secret = OPAQUE[:18] if "stringData" in name else OPAQUE
    data = text.encode()
    leaks = [p for p in range(1, len(data)) if secret in _stream([data[:p], data[p:]])]
    assert leaks == []
    assert secret not in _stream([line.encode() for line in text.splitlines(keepends=True)])


def _plan(count: int, indent: int | None) -> str:
    changes = [
        {
            "address": f"aws_ssm_parameter.p{i}",
            "change": {
                "actions": ["create"],
                "before": None,
                "after": {"name": f"/app/p{i}", "value": SENSITIVE if i == 0 else f"v{i}"},
                "after_sensitive": {"value": True} if i == 0 else {},
                "before_sensitive": False,
            },
        }
        for i in range(count)
    ]
    plan = {
        "format_version": "1.2",
        "terraform_version": "1.9.5",
        "resource_changes": changes,
        "planned_values": {"root_module": {}},
    }
    return json.dumps(plan, indent=indent) + "\n"


def _large_yaml_secret() -> tuple[str, list[str]]:
    values = [b64(f"opaque-{i}-" * 60) for i in range(40)] + [b64(SENSITIVE)]
    body = "".join(f"  K{i}: {value}\n" for i, value in enumerate(values))
    return f"apiVersion: v1\ndata:\n{body}kind: Secret\nmetadata:\n  name: x\n", values


@pytest.mark.parametrize("size", [4096, 65536])
def test_live_stream_masks_structures_larger_than_the_8k_cap(size: int) -> None:
    yaml_text, values = _large_yaml_secret()
    for text, secrets in (
        (yaml_text, values),
        (_plan(200, None), [SENSITIVE]),
        (_plan(200, 2), [SENSITIVE]),
    ):
        assert len(text) > 8192
        data = text.encode()
        for phase in (0, 1000, 3333):
            parts = [data[:phase]] + [data[k : k + size] for k in range(phase, len(data), size)]
            out = _stream([part for part in parts if part])
            assert not [s for s in secrets if s in out]
            assert "[live output withheld" not in out


def test_a_pretty_plan_whose_brace_arrives_alone_is_still_masked() -> None:
    text = _plan(3, 2)
    assert text.startswith("{\n")
    assert SENSITIVE not in _stream([b"{\n", text[2:].encode()])


@pytest.mark.parametrize(
    "text",
    [
        "".join(f"2026-11-10T12:00:{i % 60:02d}Z INFO worker {i} batch\n" for i in range(300)),
        json.dumps({"data": {"items": [{"id": i} for i in range(80)]}, "meta": {}}, indent=2),
        "data:\n" + "".join(f"  key{i}: value{i}\n" for i in range(100)) + "other: 1\n",
    ],
)
def test_output_without_a_structure_anchor_streams_exactly_as_before(
    text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No ``apiVersion`` and no plan opener: the guard must not move a single cut,
    # so every read publishes what it did without the guard (no new latency).
    from local_operator.tools import builtin

    def per_read() -> list[bytes]:
        redactor = builtin._PipeRedactor([])
        data = text.encode()
        reads = [redactor.feed(data[k : k + 64]) for k in range(0, len(data), 64)]
        return reads + [redactor.feed(b"", final=True)]

    guarded = per_read()
    monkeypatch.setattr(builtin.StructureStreamGuard, "hold", lambda self, text, cut, final: cut)
    assert guarded == per_read()


def test_a_data_block_over_the_hold_limit_is_withheld_not_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator import redaction_shapes

    monkeypatch.setattr(redaction_shapes, "STRUCTURE_HOLD_LIMIT", 4096)
    yaml_text, values = _large_yaml_secret()
    data = yaml_text.encode() + b"after the structure\n"
    out = _stream([data[k : k + 1024] for k in range(0, len(data), 1024)])
    assert not [s for s in values if s in out]
    # "output withheld": the sink is the settled result, not only the live view.
    assert out.count("[output withheld: a Kubernetes data block") == 1
    assert "live output withheld" not in out
    assert out.endswith("after the structure\n")


def _big_plan(target: int, *, sensitive: bool, indent: int | None = 2) -> str:
    """A plan over ``target`` chars; resource 0 carries ``SENSITIVE`` when asked.

    ``sensitive=False`` is the ``json-output-redacted`` shape: the value is
    ``null`` and the sensitivity maps still mark it.
    """
    per_change = len(_plan(100, indent)) / 100
    count = int(target / per_change * 1.05) + 1
    text = _plan(count, indent)
    if not sensitive:
        text = text.replace(json.dumps(SENSITIVE), "null", 1)
    assert len(text) > target
    return text


def _settle(data: bytes, read: int = 65536) -> tuple[str, Any]:
    """Stream ``data`` through the pipe filter into the real result sink."""
    from local_operator.tools.builtin import _BashOutput, _PipeRedactor

    redactor = _PipeRedactor([])
    sink = _BashOutput()
    held = False
    for k in range(0, len(data), read):
        sink.append(redactor.feed(data[k : k + read]))
        held = held or redactor.held_for_check > 0
    sink.append(redactor.feed(b"", final=True))
    return sink.decode(), held


def _raw_sink(data: bytes) -> str:
    """Base's settled result for ``data``: the sink fed the bytes unchanged."""
    from local_operator.tools.builtin import _BashOutput

    sink = _BashOutput()
    for k in range(0, len(data), 65536):
        sink.append(data[k : k + 65536])
    return sink.decode()


def test_a_redacted_plan_over_4_mib_settles_with_base_head_and_tail() -> None:
    # R2-1: a json-output-redacted plan over 4 MiB settled as a 164-char notice.
    data = (_big_plan(6 * 1024 * 1024, sensitive=False) + "PLAN-TRAILER-LINE\n").encode()
    settled, held = _settle(data)
    assert held  # the live view did wait for the whole plan
    assert settled == _raw_sink(data)  # nothing masked, so base's exact head + tail
    assert settled.startswith('{\n  "format_version": "1.2"')
    assert settled.endswith("PLAN-TRAILER-LINE\n")
    assert "json-output-redacted" not in settled and "withheld" not in settled


@pytest.mark.parametrize("indent", [2, None])
def test_a_plan_over_4_mib_with_a_sensitive_value_settles_masked(indent: int | None) -> None:
    text = _big_plan(5 * 1024 * 1024, sensitive=True, indent=indent)
    data = (text + "PLAN-TRAILER-LINE\n").encode()
    settled, _ = _settle(data)
    assert SENSITIVE not in settled
    assert '"address": "aws_ssm_parameter.p0"' in settled or '"address":"aws_ssm_parameter.p0"' in settled
    last = text.count('"address"') - 1
    assert f"aws_ssm_parameter.p{last}" in settled  # the tail survives too
    assert settled.endswith("PLAN-TRAILER-LINE\n")
    assert len(settled.encode()) > 3 * 1024 * 1024  # head + tail, not a notice


def _state_json(target: int) -> str:
    resources = [
        {
            "address": f"aws_lambda_function.f{i}",
            "type": "aws_lambda_function",
            "values": {
                "function_name": f"f{i}",
                # A ``"variables"`` key INSIDE the state must not make it a plan.
                "environment": [{"variables": {"LOG_LEVEL": "info", "MODE": f"m{i}"}}],
            },
            "sensitive_values": {"environment": [{}]},
        }
        for i in range(target // 300 + 1)
    ]
    state = {
        "format_version": "1.0",
        "terraform_version": "1.9.5",
        "values": {"root_module": {"resources": resources}},
    }
    return json.dumps(state, indent=2) + "\n"


def test_terraform_state_json_streams_byte_identical_to_the_unguarded_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # R2-1 (a): ``terraform show -json`` STATE (``values`` third) is never held.
    from local_operator.tools import builtin

    data = _state_json(6 * 1024 * 1024).encode()
    assert len(data) > 6 * 1024 * 1024

    def per_read() -> list[bytes]:
        redactor = builtin._PipeRedactor([])
        reads = [redactor.feed(data[k : k + 65536]) for k in range(0, len(data), 65536)]
        assert redactor.held_for_check == 0
        return reads + [redactor.feed(b"", final=True)]

    guarded = per_read()
    assert b"".join(guarded) == data
    monkeypatch.setattr(builtin.StructureStreamGuard, "hold", lambda self, text, cut, final: cut)
    assert guarded == per_read()


@pytest.mark.parametrize(
    ("fragment", "expected"),
    [
        ('{\n  "format_version": "1.2",\n  "terraform_version": "1.9.5",\n', True),
        ('{"format_version":"1.2","terraform_version":"1.9.5","vari', True),
        ('{"format_version":"1.2","terraform_version":"1.9.5","planned_values"', True),
        ('{"format_version":"1.0","terraform_version":"1.9.5","values"', False),
        ('{"format_version":"1.0","terraform_version":"1.9.5","v', True),
    ],
)
def test_a_plan_opener_needs_a_key_the_plan_rule_acts_on(fragment: str, expected: bool) -> None:
    from local_operator.redaction_shapes import _could_open_a_plan

    assert _could_open_a_plan(fragment) is expected


def test_holding_a_plan_is_linear_in_its_size() -> None:
    # R2-1: the held plan was re-joined and re-searched per read (4.92 MiB
    # 2.54 s, 12.3 MiB 11.48 s). Same-run control: a 4x larger plan must take
    # well under the 16x a per-read rescan costs.
    small = _big_plan(512 * 1024, sensitive=False).encode()
    large = _big_plan(2 * 1024 * 1024, sensitive=False).encode()

    def timed(data: bytes) -> float:
        best = float("inf")
        for _ in range(2):
            began = time.perf_counter()
            _settle(data)
            best = min(best, time.perf_counter() - began)
        return best

    assert timed(large) < timed(small) * 8


def test_a_plan_over_the_stream_bound_is_released_to_the_plain_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator import redaction_shapes

    monkeypatch.setattr(redaction_shapes, "STREAM_PLAN_HOLD_LIMIT", 4096)
    text = _plan(200, 2)
    data = text.encode() + b"after the structure\n"
    out = _stream([data[k : k + 1024] for k in range(0, len(data), 1024)])
    notice = redaction_shapes._UNCHECKED_PLAN
    assert out.count(notice) == 1 and out.startswith(notice)
    # Not dropped: every byte is there (base behaviour), and the notice never
    # sends an agent to json-output-redacted.
    assert out[len(notice) :] == text + "after the structure\n"
    assert "json-output-redacted" not in notice


def test_configmap_before_a_secret_without_a_separator_keeps_its_values() -> None:
    # Q1: the Secret's ``kind`` used to claim the ConfigMap's ``data:`` block.
    configmap = (
        "apiVersion: v1\ndata:\n  package: '@acme/jobs-runner'\n  image: jobs-svc:1.2.3\n"
        "kind: ConfigMap\nmetadata:\n  name: cfg\n"
    )
    out = scrub_shapes(configmap + secret_yaml())
    assert out.startswith(configmap)
    assert out[len(configmap) :] == scrub_shapes(secret_yaml())
    assert scrub_shapes(secret_yaml() + configmap).endswith(configmap)


def test_secret_rule_is_linear_in_concatenated_objects() -> None:
    # Q2: 2,000 ConfigMaps + a Secret took over 600 s. The control is the same
    # text with the structure rule inert (no ``kind: Secret``), timed in this run.
    configmap = "apiVersion: v1\ndata:\n  LOG_LEVEL: info\n  MODE: fast\nkind: ConfigMap\n"
    configmap += "metadata:\n  name: cfg\n" + "  labels:\n    app: jobs\n" * 3
    text = configmap * 3000 + secret_yaml()
    assert len(text) > 400_000
    control = text.replace("kind: Secret", "kind: Opaque")
    started = time.perf_counter()
    scrub_shapes(control)
    base = time.perf_counter() - started
    started = time.perf_counter()
    out = scrub_shapes(text)
    took = time.perf_counter() - started
    assert out.startswith(configmap * 3000)
    assert took < 1.5 * base + 1.0, (took, base)


def test_secret_rule_reads_through_the_read_tool_gutter() -> None:
    # Q4: `` 9| stringData:`` was no key at all, so ``read`` skipped the rule.
    from local_operator.tools.builtin import _number_lines

    text = secret_yaml().replace(b64(DSN), OPAQUE)
    numbered = _number_lines(text.splitlines(), 1)
    out = scrub_shapes(numbered)
    assert OPAQUE not in out
    assert out.count(REDACTION_MARKER) == scrub_shapes(text).count(REDACTION_MARKER)


async def _read_through_a_session(tmp_path: Any, path: Any) -> tuple[str, str]:
    """``read`` ``path`` through a Session: (last provider request repr, durable transcript)."""
    from local_operator.harness.types import (
        ModelSpec,
        StreamEndEvent,
        StreamTextDelta,
        StreamToolCallDelta,
    )
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tools.builtin import build_read_tool
    from local_operator.variables import VariableStore

    requests: list[Any] = []

    def provider(request: Any, signal: Any = None) -> Any:
        index = len(requests)
        requests.append(request)

        async def generate() -> Any:
            if index == 0:
                yield StreamToolCallDelta(
                    index=0,
                    id="r0",
                    name="read",
                    argument_delta=json.dumps({"path": str(path)}),
                )
                yield StreamEndEvent(stop_reason="toolUse")
            else:
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

        return generate()

    session = Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=100_000),
        stream_fn=provider,
        tools=[build_read_tool()],
        transcript=Transcript(tmp_path / "session"),
        system_blocks_provider=lambda *_: [],
        yolo=True,
        cwd=str(tmp_path),
        variables=VariableStore(cwd=str(tmp_path)),
    )
    await session.prompt("read the file")
    durable = (tmp_path / "session" / "transcript.jsonl").read_text()
    return repr(requests[-1]), durable


@pytest.mark.asyncio
async def test_read_tool_session_masks_a_secret_manifest(tmp_path: Any) -> None:
    # Q4 on the real path: a ``read`` through a Session, checked in the provider
    # request and the durable transcript.
    manifest = tmp_path / "secret.yaml"
    manifest.write_text(
        "apiVersion: v1\nkind: Secret\nmetadata:\n  name: x\nstringData:\n"
        f"  ENDPOINT_AUTH: {OPAQUE[:18]}\ndata:\n  BLOB: {OPAQUE}\n"
    )
    request, durable = await _read_through_a_session(tmp_path, manifest)
    for secret in (OPAQUE, OPAQUE[:18]):
        assert secret not in request
        assert secret not in durable
    assert "ENDPOINT_AUTH: " + REDACTION_MARKER in request


# --- 2026-11-09 exposure: plan variables, read gutter ----------------------------
TF_AWS_KEY_ID = "AKIA" + "QX7FAKEFAKE1234Z"
TF_AWS_SECRET = ("fAkE" + "s3cr3tK3yV4lu3" * 3)[:40]


def _variable_set_plan(*, marked: bool) -> dict[str, Any]:
    """An HCP plan whose variable set is sensitive; ``marked`` = declared in config."""
    declared: dict[str, Any] = {"sensitive": True} if marked else {}
    return {
        "format_version": "1.2",
        "terraform_version": "1.9.0",
        "variables": {
            "AWS_ACCESS_KEY": {"value": TF_AWS_KEY_ID},
            "AWS_SECRET_KEY": {"value": TF_AWS_SECRET},
            "region": {"value": "us-east-1"},
        },
        "planned_values": {"root_module": {}},
        "resource_changes": [
            {
                "address": "aws_s3_bucket.b",
                "change": {
                    "actions": ["create"],
                    "before": None,
                    "after": {"bucket": "my-bucket"},
                    "after_sensitive": {},
                },
            }
        ],
        "configuration": {
            "root_module": {
                "variables": {
                    "AWS_ACCESS_KEY": dict(declared),
                    "AWS_SECRET_KEY": dict(declared),
                    "region": {},
                }
            }
        },
    }


def _gutter(text: str) -> str:
    return "\n".join(f"{i + 1:>6}| {line}" for i, line in enumerate(text.splitlines()))


@pytest.mark.parametrize("marked", [True, False])
@pytest.mark.parametrize("indent", [2, None])
@pytest.mark.parametrize("guttered", [False, True])
def test_plan_variables_are_masked_by_name_raw_and_through_the_read_gutter(
    marked: bool, indent: int | None, guttered: bool
) -> None:
    text = json.dumps(_variable_set_plan(marked=marked), indent=indent)
    if guttered:
        text = _gutter(text)
    out = scrub_shapes(text)
    assert TF_AWS_SECRET not in out and TF_AWS_KEY_ID not in out
    # Names, the non-credential variable and resources stay readable.
    for kept in ("AWS_SECRET_KEY", "AWS_ACCESS_KEY", "us-east-1", "my-bucket", "aws_s3_bucket.b"):
        assert kept in out
    if guttered:
        lines = out.splitlines()
        assert len(lines) == len(text.splitlines())
        assert all(line.startswith(f"{i + 1:>6}| ") for i, line in enumerate(lines))


def test_a_plan_without_credential_names_is_untouched_through_the_gutter() -> None:
    plan = _variable_set_plan(marked=False)
    plan["variables"] = {"region": {"value": "us-east-1"}, "bucket": {"value": "b-1"}}
    text = _gutter(json.dumps(plan, indent=2))
    assert scrub_shapes(text) == text


@pytest.mark.asyncio
@pytest.mark.parametrize("marked", [True, False])
async def test_read_tool_session_masks_plan_variables(tmp_path: Any, marked: bool) -> None:
    saved = tmp_path / "plan.json"
    saved.write_text(json.dumps(_variable_set_plan(marked=marked), indent=2) + "\n")
    request, durable = await _read_through_a_session(tmp_path, saved)
    for secret in (TF_AWS_SECRET, TF_AWS_KEY_ID):
        assert secret not in request
        assert secret not in durable
    assert "us-east-1" in request and "AWS_SECRET_KEY" in request


# --- 2026-11 post-mortem: kubectl go-template error dump ------------------------
GOMAP_V1 = b64("plainvalueQQ9zzfakeX1")
GOMAP_V2 = b64("mrva-live-" + "a" * 32)
GOMAP_HOOK = b64("https://hooks.slack.com/services/T000/B000/XXXXfakeXXXX")


def _template_error_dump() -> str:
    """kubectl's ``-o go-template`` failure on a missing key, values fake."""
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "internal-jobs-sc"},
        "data": {"MINERVA_API_KEY_PROD": GOMAP_V2, "OTHER": GOMAP_V1, "HOOK": GOMAP_HOOK},
    }
    return (
        'error: error executing template "{{index .data \\"K\\" | base64decode}}": template: '
        'output:1:2: executing "output" at <index .data "K">: error calling index: '
        "index of untyped nil\n"
        "raw data was: " + json.dumps(secret) + "\n"
        "object given to template engine was: map[apiVersion:v1 data:map[HOOK:"
        + GOMAP_HOOK
        + " MINERVA_API_KEY_PROD:"
        + GOMAP_V2
        + " OTHER:"
        + GOMAP_V1
        + "] kind:Secret metadata:map[name:internal-jobs-sc]]\n"
    )


GOMAP_VALUES = (GOMAP_V1, GOMAP_V2, GOMAP_HOOK)


def test_go_template_error_dump_masks_the_go_map_values_and_keeps_keys() -> None:
    out = scrub_shapes(_template_error_dump())
    assert [v for v in GOMAP_VALUES if v in out] == []
    assert (
        f"data:map[HOOK:{REDACTION_MARKER} MINERVA_API_KEY_PROD:{REDACTION_MARKER} "
        f"OTHER:{REDACTION_MARKER}] kind:Secret metadata:map[name:internal-jobs-sc]]"
    ) in out


def test_go_map_rule_reads_only_the_owning_object() -> None:
    configmap = "map[apiVersion:v1 data:map[LOG_LEVEL:info MODE:fast] kind:ConfigMap]\n"
    assert scrub_shapes(configmap) == configmap
    # A ConfigMap next to a Secret keeps its values; stringData is masked too.
    both = configmap + "map[apiVersion:v1 kind:Secret stringData:map[TOKEN:tok-zq9-77]]\n"
    out = scrub_shapes(both)
    assert out.startswith(configmap)
    assert f"stringData:map[TOKEN:{REDACTION_MARKER}]" in out
    # A Secret NESTED in a ConfigMap's metadata does not claim the ConfigMap's data.
    nested = "map[data:map[A:keep-me] kind:ConfigMap metadata:map[owner:map[kind:Secret]]]\n"
    assert scrub_shapes(nested) == nested


def test_live_stream_masks_the_go_template_dump_at_every_split_point() -> None:
    data = _template_error_dump().encode()
    leaks = [
        p for p in range(1, len(data)) if any(v in _stream([data[:p], data[p:]]) for v in GOMAP_VALUES)
    ]
    assert leaks == []


@pytest.mark.parametrize("size", [4096, 65536])
def test_live_stream_masks_a_go_map_line_longer_than_the_8k_cap(size: int) -> None:
    # One line: the cap used to cut it before ``kind:Secret`` (sorted after data).
    values = [b64(f"opaque-{i}-" * 40) for i in range(60)]
    entries = " ".join(f"K{i:02d}:{value}" for i, value in enumerate(values))
    line = f"map[apiVersion:v1 data:map[{entries}] kind:Secret metadata:map[name:x]]\n"
    assert len(line) > 8192
    data = ("before\n" + line + "after\n").encode()
    out = _stream([data[k : k + size] for k in range(0, len(data), size)])
    assert [v for v in values if v in out] == []
    assert out.startswith("before\n") and out.endswith("after\n") and "K59:" in out


@pytest.mark.parametrize(
    ("command", "fires"),
    [
        ("""kubectl get secret X -o go-template='{{index .data "K" | base64decode}}' 2>&1""", True),
        ("""kubectl -n jobs get secret X -o go-template='{{index .data "K" | base64decode}}'""", True),
        ("kubectl get secret X -o jsonpath='{.data.K}' | base64 -d", True),
        ("kubectl get secret X -o jsonpath='{.data.K}' | base64 --decode; echo", True),
        ("kubectl get secret X -o jsonpath='{.data.K}' | base64 -d | psql", False),
        ("kubectl get secret X -o jsonpath='{.data.K}' | base64 -d > /tmp/k", False),
        ("""kubectl get secret X -o go-template='{{index .data "K" | base64decode}}' | wc -c""", False),
        ("export PW=$(kubectl get secret X -o jsonpath='{.data.K}' | base64 -d)", False),
        ("kubectl get secret X -o jsonpath='{.data}' | jq 'keys'", False),
        ("kubectl get secret X -o go-template='{{.metadata.name}}'", False),
    ],
)
def test_decoded_secret_to_the_terminal_is_steered(command: str, fires: bool) -> None:
    from local_operator.redaction_shapes import DUMP_SHAPES

    shape = next(s for s in DUMP_SHAPES if s.label == "kubectl-secret-decode")
    assert bool(shape.pattern.search(command)) is fires


def test_templated_secret_values_are_left_as_code() -> None:
    # R2: a Helm chart's template expressions are code, not values.
    chart = (
        'apiVersion: v1\nkind: Secret\nmetadata:\n  name: {{ include "app.fullname" . }}\n'
        "data:\n  DATABASE_URL: {{ .Values.db.url | b64enc | quote }}\n"
        '  API_KEY: {{ required "apiKey" .Values.apiKey | b64enc }}\n'
        "stringData:\n  SENTRY_DSN: ${SENTRY_DSN}\n  TOKEN: $(cat /run/token)\n"
        "  config.yaml: |\n    password: {{ .Values.pw }}\n"
    )
    assert scrub_shapes(chart) == chart
    literal = chart.replace("${SENTRY_DSN}", "literal-value-here")
    assert "SENTRY_DSN: " + REDACTION_MARKER in scrub_shapes(literal)


def test_yaml_key_scan_is_linear_on_long_whitespace() -> None:
    # R3: a 200,000-space line under ``kind: Secret`` did not finish in 118 s.
    text = "kind: Secret\ndata:\n  a" + " " * 200_000 + "b\n"
    started = time.perf_counter()
    scrub_shapes(text)
    assert time.perf_counter() - started < 2.0


def test_deep_nesting_next_to_a_plan_anchor_does_not_raise() -> None:
    # R4: RecursionError escaped ``raw_decode`` into the settled result path.
    text = '{"resource_changes": [], "after_sensitive": ' + "[" * 100_000 + "]" * 100_000 + "}"
    assert scrub_shapes(text) == text


def test_jq_program_printing_a_value_before_keys_is_steered() -> None:
    # R5: only the LAST step being ``keys`` used to exempt the whole program.
    base = "kubectl get secret x -o json | jq "
    assert credential_dump_notice(base + "'.data.PASSWORD, keys'") is not None
    assert credential_dump_notice(base + "'.data[\"PASSWORD\"] | keys'") is not None
    assert credential_dump_notice(base + "'.data | keys'") is None


@pytest.mark.parametrize("indent", [None, 2, 4, "\t"])
def test_masked_plan_keeps_its_original_layout(indent: Any) -> None:
    # N2: the plan was always re-rendered at indent 2.
    plan = json.loads(_plan(2, None))
    separators = (",", ":") if indent is None else None
    text = json.dumps(plan, indent=indent, separators=separators)
    assert scrub_shapes(text) == text.replace(SENSITIVE, REDACTION_MARKER)


def test_existing_key_rules_apply_inside_an_escaped_json_string() -> None:
    # Q5: ``aws secretsmanager get-secret-value`` prints SecretString as JSON
    # inside a JSON string; the key-name rule could not see ``\"password\"``.
    value = "AwsSm" + "Pw55zzKq"
    inner = json.dumps({"username": "jobs", "password": value})
    text = json.dumps({"Name": "prod/db", "SecretString": inner}, indent=4)
    out, hits = scrub_shapes_with_hits(text)
    assert value not in out
    assert '\\"username\\": \\"jobs\\"' in out and '"Name": "prod/db"' in out
    assert value in {hit.value for hit in hits}
    # A JSON string with no credential inside keeps every byte.
    plain = json.dumps({"payload": json.dumps({"user": "jobs", "region": "ca-central-1"})})
    assert scrub_shapes(plain) == plain


@pytest.mark.parametrize(
    "inner",
    [
        # Measured false positives of a first cut that decoded EVERY escaped
        # string: code and commands in a tool call's raw arguments.
        'curl -sk -u "$OS_USER:$OS_PASS" https://search.example/_cat',
        'items = ["Datadog MCP credential: " + name]',
    ],
)
def test_escaped_json_pass_reads_only_strings_that_are_json(inner: str) -> None:
    text = json.dumps({"command": inner})
    hits: list[Any] = []
    out = _scrub_escaped_json_strings(text, hits, None, 0, _DecodeBudget())
    assert out == text and hits == []


@pytest.mark.asyncio
async def test_mcp_server_result_is_masked_in_stored_details(tmp_path: Any) -> None:
    # Q6: the raw MCP payload rode into the transcript beside the masked text.
    from local_operator.harness.loop import AgentLoop
    from local_operator.variables import VariableStore

    store = VariableStore(cwd=str(tmp_path))
    server_result = {
        "content": [{"type": "text", "text": secret_yaml().replace(b64(DSN), OPAQUE)}],
        "isError": False,
        "meta": {"count": 1},
    }
    out = await AgentLoop._redact_details(
        {"server_result": server_result}, store.redact, "mcp__qa_dump", None
    )
    assert out is not None
    assert OPAQUE not in json.dumps(out)
    assert out["server_result"]["meta"] == {"count": 1}
    assert out["server_result"]["isError"] is False
    assert OPAQUE in json.dumps(server_result), "the executor's own payload is untouched"


def test_subagent_result_is_scrubbed_before_it_is_persisted(tmp_path: Any) -> None:
    from types import SimpleNamespace

    from local_operator.harness.subagent import _redact_child_result
    from local_operator.variables import VariableStore

    parent = SimpleNamespace(_variables=VariableStore(cwd=str(tmp_path)))
    text = "child found:\n" + secret_yaml().replace(b64(DSN), OPAQUE)
    assert OPAQUE not in _redact_child_result(parent, text)
    assert _redact_child_result(SimpleNamespace(), "plain") == "plain"

    def broken(_: str) -> str:
        raise RuntimeError("sink")

    failing = SimpleNamespace(_variables=SimpleNamespace(redact=broken))
    assert OPAQUE not in _redact_child_result(failing, text)


def test_project_rows_scrub_free_text_and_keep_identifiers(tmp_path: Any) -> None:
    from local_operator.projects import ProjectEdit, ProjectRegistry

    registry = ProjectRegistry(tmp_path)
    dsn = f"mongodb+srv://fakeuser:{PASSWORD}@cluster0.example.mongodb.net/db"
    project = registry.create_project(
        ProjectEdit(name="jobs-rotation", description=f"rotated {dsn}", progress=f"checked {dsn}")
    )
    row = (tmp_path / "projects" / f"{project.id}.json").read_text()
    assert PASSWORD not in row
    assert '"name": "jobs-rotation"' in row
    assert "cluster0.example.mongodb.net" in row
    assert registry.get_project(project.id) is not None
