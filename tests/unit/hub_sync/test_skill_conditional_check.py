"""The conditional skill check: If-None-Match on the hub read, 304 means nothing to do.

The client half of agent-server #175. Before it, every check tick downloaded the
whole skill document for every linked skill -- the catalogue is ~775 KiB across
14 skills, re-fetched hourly per client -- because the desktop never sent the
validator the server started honouring. Now the check sends the revision the
link already holds (the same baseline ``remote_revision`` the push CAS sends as
``If-Match``) and an unchanged row answers ``304`` with no body.

These tests drive a REAL loopback HTTP server through the REAL ``RadientClient``
(the redirect refusal, the header construction and the empty-body handling are
all in play -- a fake client would let a client that swallows 304 as an error
pass), then assert on what the check, the merge and the status store actually
DID:

* a matching validator is ``unchanged``: one request, zero body bytes, the same
  verdict a 200 carrying identical bytes produces, and no revision advance;
* a changed row still merges and applies exactly as before;
* a link with no recorded revision sends NO validator (a first pull has nothing
  to revalidate against);
* a 304 is never recorded as a failure in the status store.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from local_operator.clients.radient import RadientClient
from local_operator.config import ConfigManager
from local_operator.hub_sync import check as chk
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync import store as st
from local_operator.hub_sync.skill_document import SkillDocument
from local_operator.hub_sync.skill_storage import ManagedSkillStorage

from .test_skill_storage import _input

#: The revision ``_input()`` writes into its baseline, and the tag the stub serves
#: by default -- i.e. the "we already hold the current row" case.
HELD_REVISION = '"revision-1"'


class StubHub:
    """A loopback hub serving ONE skill detail row, conditional-GET aware.

    It records every request's headers and every response's (status, body bytes)
    so a test can assert BOTH halves of the saving: that the validator was sent,
    and that the hub really answered with no body at all.
    """

    def __init__(self, document: SkillDocument, *, revision: str = HELD_REVISION) -> None:
        self.document = document
        self.revision = revision
        self.requests: list[dict[str, str]] = []
        self.wire: list[tuple[int, int]] = []
        #: Negative control: answer 304 even to a request that sent no validator.
        self.force_304 = False
        hub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
                headers = dict(self.headers.items())
                hub.requests.append(headers)
                matches = headers.get("If-None-Match") == hub.revision and bool(
                    headers.get("If-None-Match")
                )
                if hub.force_304 or matches:
                    # RFC 9110 §15.4.5: a 304 carries the validators and NO body.
                    hub.wire.append((304, 0))
                    self.send_response(304)
                    self.send_header("ETag", hub.revision)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                payload = dict(hub.document.to_json())
                payload.update(
                    {
                        # The metadata envelope a real DETAIL response carries
                        # beside the manifest (agent-server, internal/responses/skills.go).
                        "revision": hub.revision,
                        "content_revision": 1,
                        "sync_contract": {"revision_cas": 1, "skills": 1},
                        "id": "sk-1",
                        "visibility": "public",
                    }
                )
                body = json.dumps({"result": payload}).encode()
                hub.wire.append((200, len(body)))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("ETag", hub.revision)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "StubHub":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join()


def _profile(
    tmp_path: Path,
    hub: StubHub,
    *,
    revision: str | None = HELD_REVISION,
) -> tuple[Any, Path, svc.HubSyncContext]:
    """A managed skill v1 on disk, plus a ctx whose client is a REAL HTTP client.

    ``revision`` is what the baseline records the link as holding; ``None`` models
    a legacy (or never-pulled) baseline that has no validator to send.
    """

    identity, document, baseline = _input()
    root = tmp_path / "profile"
    storage = ManagedSkillStorage(root)
    baseline = dataclasses.replace(baseline, remote_revision=revision)
    storage.apply(identity, document.name, document, baseline, expected_fingerprint=None)
    client = RadientClient(SecretStr("synthetic-api-key"), hub.base_url)
    ctx = svc.HubSyncContext(
        config_dir=root,
        config_manager=ConfigManager(root),
        client_for_tenant=lambda _tenant: client,
    )
    return identity, root, ctx


def test_a_matching_validator_is_unchanged_with_no_body_and_one_request(
    tmp_path: Path,
) -> None:
    """The saving, proved end to end: one conditional GET, a 304, nothing transferred."""

    doc = _input()[1]
    with StubHub(doc) as hub:
        identity, root, ctx = _profile(tmp_path, hub)

        (c,) = chk.check_hub_skills(config_dir=root, client_for_tenant=ctx.client_for_tenant)

    # The request carried the revision the link already holds, verbatim.
    assert hub.requests[0]["If-None-Match"] == HELD_REVISION
    # ... and it was the ONLY request: a 304 is not followed by an unconditional
    # re-fetch of the row.
    assert len(hub.requests) == 1
    # The hub really answered with no body: (status, bytes).
    assert hub.wire == [(304, 0)]

    # "Unchanged" is the same ItemCheck a 200 carrying identical bytes produces.
    assert c.verdict == "up-to-date"
    assert c.classification is not None and c.classification.state == "up-to-date"
    assert c.local_fp is not None and c.local_fp == c.remote_fp
    assert c.remote == doc.to_json()  # R := the baseline the 304 vouched for
    assert c.base == doc.to_json()
    # No revision ADVANCE: the tag the hub confirmed is the one we already held.
    assert c.remote_revision == HELD_REVISION
    # A body-less answer observes no sync contract, and must not invent one.
    assert c.sync_contract is None


def test_the_baseline_is_never_rewritten_by_a_304(tmp_path: Path) -> None:
    """A check writes nothing; a 304 in particular must not touch the baseline."""

    doc = _input()[1]
    with StubHub(doc) as hub:
        identity, root, ctx = _profile(tmp_path, hub)
        before = prov.read_baseline(root, "skill", identity.local_id)
        assert before is not None

        chk.check_hub_skills(config_dir=root, client_for_tenant=ctx.client_for_tenant)

        after = prov.read_baseline(root, "skill", identity.local_id)

    assert after is not None
    assert after == before
    assert after.remote_revision == HELD_REVISION


def test_a_changed_row_is_still_fetched_merged_and_applied(tmp_path: Path) -> None:
    """A 200 behaves exactly as today: classified remote-only, merged, baseline advanced."""

    v2 = _input("# New body\n")[1]
    with StubHub(v2, revision='"revision-2"') as hub:
        identity, root, ctx = _profile(tmp_path, hub)
        checks = svc.check_items(ctx, kinds=("skill",))
        (c,) = checks

        assert hub.requests[0]["If-None-Match"] == HELD_REVISION  # stale tag still sent
        assert [status for status, _bytes in hub.wire] == [200]
        assert c.verdict == "available"
        assert c.classification is not None and c.classification.state == "remote-only"
        assert c.remote_revision == '"revision-2"'

        report = svc.apply_items(ctx, kind="skill", auto=True, allow_llm=False, checks=checks)
        on_disk = ManagedSkillStorage(root).snapshot(identity.local_id).document

    assert report.reports[0].outcome == "merged" and report.reports[0].applied
    # The change LANDED: the merged bytes are on disk, not just the baseline.
    assert "New body" in on_disk.files[0].content
    # ONE request for the whole check+apply: the apply reused the check it was given.
    assert len(hub.requests) == 1
    stored = prov.read_baseline(root, "skill", identity.local_id)
    assert stored is not None
    assert stored.fields == v2.to_json()  # B := the exact integrated remote
    assert stored.remote_revision == '"revision-2"'


def test_a_link_with_no_revision_sends_no_validator(tmp_path: Path) -> None:
    """A first pull has nothing to revalidate against: the read stays unconditional."""

    doc = _input()[1]
    with StubHub(doc) as hub:
        identity, root, ctx = _profile(tmp_path, hub, revision=None)

        (c,) = chk.check_hub_skills(config_dir=root, client_for_tenant=ctx.client_for_tenant)

    assert "If-None-Match" not in hub.requests[0]
    assert [status for status, _bytes in hub.wire] == [200]
    assert c.verdict == "up-to-date"
    # A 200 still carries its revision, and that IS an observation worth keeping.
    assert c.remote_revision == HELD_REVISION


def test_a_304_is_never_recorded_as_a_failure(tmp_path: Path) -> None:
    """The status store sees a successful, up-to-date check -- not an error, not a retry."""

    doc = _input()[1]
    with StubHub(doc) as hub:
        identity, root, ctx = _profile(tmp_path, hub)

        svc.check_items(ctx, kinds=("skill",))  # records into the store

    item = st.StatusStore(root).load()["items"][f"skill:{identity.local_id}"]
    assert item["state"] == "up-to-date"
    assert item["error_class"] is None
    assert item["last_error"] is None
    assert item["attempts"] == 0
    assert item.get("next_retry_at") is None
    assert item.get("auto_retry") is True
    # The failure classes a swallowed 304 WOULD have produced never appeared.
    assert "unexpected_redirect" not in json.dumps(item)
    assert "hub-error" not in json.dumps(item)


def test_the_client_raises_not_modified_instead_of_an_api_error(tmp_path: Path) -> None:
    """Unit-level: the 304 branch runs BEFORE the redirect refusal, so it cannot mislabel.

    ``_refuse_redirect`` refuses every 3xx (correctly -- a redirect must not
    re-send a bearer). A 304 is not a redirect: it is the success case of the
    conditional read, and it must arrive as ``NotModified`` carrying the tag.
    """

    from local_operator.clients._http import NotModified

    doc = _input()[1]
    with StubHub(doc) as hub:
        client = RadientClient(SecretStr("synthetic-api-key"), hub.base_url)

        with pytest.raises(NotModified) as raised:
            client.get_skill("immutable-local", if_none_match=HELD_REVISION)

    assert raised.value.etag == HELD_REVISION

    # The plain, unconditional read of the same row is untouched.
    with StubHub(doc) as hub:
        client = RadientClient(SecretStr("synthetic-api-key"), hub.base_url)
        document = client.get_skill("immutable-local")

    assert document["name"] == doc.name


def test_a_304_without_a_validator_is_still_refused(tmp_path: Path) -> None:
    """Negative control: "unchanged" is only meaningful for a read that ASKED.

    A hub that answers 304 to an UNCONDITIONAL GET is violating RFC 9110, and
    reading that as "the row did not move" would let a broken upstream suppress a
    real update forever. It keeps the existing unexpected-redirect refusal.
    """

    from local_operator.clients._http import APIError

    doc = _input()[1]
    with StubHub(doc) as hub:
        hub.force_304 = True
        client = RadientClient(SecretStr("synthetic-api-key"), hub.base_url)

        with pytest.raises(APIError) as raised:
            client.get_skill("immutable-local")

    assert "If-None-Match" not in hub.requests[0]
    assert raised.value.status_code == 304
    assert raised.value.code == "unexpected_redirect"
