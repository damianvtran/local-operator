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

import pytest

from local_operator.redaction_shapes import (
    REDACTION_MARKER,
    _DecodeBudget,
    _scrub_encoded_credentials,
    credential_dump_notice,
    scrub_shapes,
    scrub_shapes_for_arguments,
    scrub_shapes_with_hits,
    secret_data_block_hold,
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
    assert secret_data_block_hold(head, len(head)) == len("apiVersion: v1\n")
    whole = head + "kind: Secret\n"
    assert secret_data_block_hold(whole, len(whole)) == len(whole)
    assert secret_data_block_hold("plain\nlines\n", 12) == 12


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

    monkeypatch.setattr(builtin, "secret_data_block_hold", lambda text, cut: cut)
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
