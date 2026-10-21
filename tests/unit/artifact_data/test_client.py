"""The artifact data store wire: slug decoding, paths, arg shapes, errors, headers.

The HTTP leg runs through ``httpx.MockTransport`` so the exact request the
server would see — URL, headers, body — is asserted, not inferred.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest

from local_operator.artifact_data import client as wire

ARTIFACT_URL = "https://claude.ai/artifact/GiE3GRkwXeG1q827vXX6g7"
ARTIFACT_SLUG = "7f3a9c2e-4b1d-4e8a-9c6f-2d5e8a1b3c40"


# -- slug ---------------------------------------------------------------------


def test_short_id_decodes_to_its_uuid() -> None:
    assert wire.decode_short_id("GiE3GRkwXeG1q827vXX6g7") == ARTIFACT_SLUG


@pytest.mark.parametrize(
    "url",
    [
        ARTIFACT_URL,
        ARTIFACT_URL + "/",
        ARTIFACT_URL + "?tab=1#x",
        "https://preview.claude.ai/artifact/GiE3GRkwXeG1q827vXX6g7",
        f"https://claude.ai/code/artifact/{ARTIFACT_SLUG}",
        f"https://claude.ai/code/artifact/{ARTIFACT_SLUG.upper()}",
    ],
)
def test_both_link_shapes_yield_the_slug(url: str) -> None:
    assert wire.slug_from_url(url) == ARTIFACT_SLUG


@pytest.mark.parametrize(
    "url",
    [
        "http://claude.ai/artifact/GiE3GRkwXeG1q827vXX6g7",  # not https
        "https://evil.example/artifact/GiE3GRkwXeG1q827vXX6g7",
        "https://claude.ai/artifact/GiE3GRkwXeG1q827vXX6g0",  # '0' is not base58
        "https://claude.ai/artifact/short",
        "https://claude.ai/code/artifact/not-a-uuid",
        "https://claude.ai/chat/123",
        "",
    ],
)
def test_anything_else_is_refused_as_invalid_url(url: str) -> None:
    with pytest.raises(wire.ArtifactDataError) as caught:
        wire.slug_from_url(url)
    assert caught.value.code == "invalid_url"


def test_a_short_id_at_or_over_two_to_the_128_is_refused() -> None:
    # 22 base58 digits can encode more than 128 bits; the top value must fail.
    with pytest.raises(wire.ArtifactDataError):
        wire.decode_short_id("z" * 22)


# -- paths --------------------------------------------------------------------


def test_collection_paths_have_an_odd_segment_count() -> None:
    assert wire.collection_segments("items") == ["items"]
    assert wire.collection_segments("data/users/me") == ["data", "users", "me"]
    with pytest.raises(wire.ArtifactDataError, match="names a document"):
        wire.collection_segments("items/doc-1")


@pytest.mark.parametrize("bad", ["", "a//b", "../x", "x/./y", "a b", "a/b/" + "c" * 201, "é"])
def test_bad_collection_segments_are_refused(bad: str) -> None:
    with pytest.raises(wire.ArtifactDataError) as caught:
        wire.collection_segments(bad)
    assert caught.value.code == "invalid_path"


def test_more_than_fifteen_segments_is_refused() -> None:
    with pytest.raises(wire.ArtifactDataError):
        wire.collection_segments("/".join(["a"] * 17))
    assert len(wire.collection_segments("/".join(["a"] * 15))) == 15


def test_doc_ids_are_one_segment_and_dots_alone_are_not_ids() -> None:
    assert wire.doc_path(["items"], "doc-1") == "items/doc-1"
    assert wire.doc_path(["t"], "a.b~c:d@e+f_g") == "t/a.b~c:d@e+f_g"
    for bad in ("", ".", "..", "a/b"):
        with pytest.raises(wire.ArtifactDataError):
            wire.doc_path(["items"], bad)


def test_me_is_substituted_only_in_the_users_prefix() -> None:
    assert wire.uses_me(["data", "users", "me"])
    assert wire.uses_me(["data", "users", "me", "decks", "cards"])
    assert not wire.uses_me(["items"])
    assert not wire.uses_me(["data", "users", "u_x"])
    assert wire.substitute_me(["data", "users", "me", "decks", "cards"], "u_abc") == [
        "data",
        "users",
        "u_abc",
        "decks",
        "cards",
    ]
    assert wire.substitute_me(["me"], "u_abc") == ["me"]


@pytest.mark.parametrize("bad", ["a.b", "a/b", "__x__", 'q"', "a\u200bb", ""])
def test_str_replace_field_must_be_one_plain_key(bad: str) -> None:
    with pytest.raises(wire.ArtifactDataError):
        wire.check_field_name(bad)
    assert wire.check_field_name("notes") == "notes"


# -- arg shapes ----------------------------------------------------------------


def test_write_args_match_the_servers_shapes() -> None:
    assert wire.write_args("set", "t/d", data={"a": 1}, if_version=3) == {
        "path": "t/d",
        "data": {"a": 1},
        "replace": True,
        "ifVersion": 3,
    }
    assert wire.write_args("update", "t/d", data={"a": 1}) == {"path": "t/d", "data": {"a": 1}}
    assert wire.write_args("delete", "t/d", if_version=2) == {"path": "t/d", "ifVersion": 2}
    edit = wire.write_args(
        "str_replace", "t/d", field="f", old_str="a", new_str="b", replace_all=True, if_version=4
    )
    assert edit.pop("idem")
    assert edit == {
        "path": "t/d",
        "field": "f",
        "old_str": "a",
        "new_str": "b",
        "replace_all": True,
        "ifVersion": 4,
    }
    # No pin is OMITTED, never sent as null (null would read as "pin to nothing").
    assert "ifVersion" not in wire.write_args("set", "t/d", data={})
    assert "replace_all" not in wire.write_args(
        "str_replace", "t/d", field="f", old_str="a", new_str="b"
    )


def test_str_replace_carries_a_fresh_url_safe_idem_per_call() -> None:
    first = wire.write_args("str_replace", "t/d", field="f", old_str="a", new_str="b")
    second = wire.write_args("str_replace", "t/d", field="f", old_str="a", new_str="b")
    for args in (first, second):
        assert re.fullmatch(r"[A-Za-z0-9_-]{16,}", args["idem"])
    assert first["idem"] != second["idem"]
    # Only str_replace carries one; Claude Code sends none on the other writes.
    assert "idem" not in wire.write_args("set", "t/d", data={})


# -- errors --------------------------------------------------------------------


def test_a_server_version_mismatch_keeps_code_version_and_index() -> None:
    body = json.dumps(
        {"error": {"code": "version_mismatch", "message": "stale", "version": 7, "index": 1}}
    )
    error = wire.error_from_response(409, body)
    assert (error.code, error.version, error.index) == ("version_mismatch", 7, 1)


def test_a_required_pin_is_reported_as_version_required() -> None:
    body = json.dumps({"error": {"code": "version_mismatch", "required": True}})
    assert wire.error_from_response(409, body).reason == "version_required"


def test_hyphenated_server_codes_normalize() -> None:
    body = json.dumps({"error": {"code": "resource-exhausted"}})
    assert wire.error_from_response(429, body).code == "quota_or_rate"


def test_the_token_class_403_is_credential_rejected_not_not_found() -> None:
    error = wire.error_from_response(403, "an OAuth access token is required")
    assert error.code == "credential_rejected"
    assert wire.error_from_response(403, "nope").code == "not_found"


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (404, "not_found"),
        (400, "invalid_argument"),
        (413, "too_large"),
        (401, "upstream_auth"),
        (500, "store_unavailable"),
    ],
)
def test_unknown_bodies_fall_back_to_the_status(status: int, code: str) -> None:
    assert wire.error_from_response(status, "<html>").code == code


# -- the call ------------------------------------------------------------------


def _tokens(token: str = "tok-1", fresh: str | None = None):
    async def refresh() -> str | None:
        return fresh

    async def source() -> wire.Bearer | None:
        return wire.Bearer(token=token, refresh=refresh)

    return source


@pytest.mark.asyncio
async def test_the_request_carries_the_frame_headers_and_body() -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "u_abc"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        payload = await wire.call_verb(ARTIFACT_SLUG, "whoami", {}, tokens=_tokens(), client=http)
    assert payload == {"id": "u_abc"}
    (request,) = seen
    assert str(request.url) == "https://api.anthropic.com/api/frame/db/agent"
    assert request.method == "POST"
    headers = request.headers
    assert headers["authorization"] == "Bearer tok-1"
    assert headers["anthropic-beta"] == "oauth-2025-04-20"
    assert headers["x-frame-cp"] == "go"
    assert headers["x-frame-surface"] == "code"
    assert headers["x-frame-platform"] == "cli"
    assert headers["x-frame-client-version"]
    assert "application/vnd.ant.frame-refusal+json" in headers["accept"]
    assert json.loads(request.content) == {"slug": ARTIFACT_SLUG, "verb": "whoami", "args": {}}


@pytest.mark.asyncio
async def test_a_401_retries_once_with_a_refreshed_token() -> None:
    auth: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        auth.append(request.headers["authorization"])
        if len(auth) == 1:
            return httpx.Response(401, text="expired")
        return httpx.Response(200, json={"id": "u_abc"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        await wire.call_verb(
            ARTIFACT_SLUG, "whoami", {}, tokens=_tokens(fresh="tok-2"), client=http
        )
    assert auth == ["Bearer tok-1", "Bearer tok-2"]


@pytest.mark.asyncio
async def test_a_second_401_surfaces_as_upstream_auth() -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="no")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(wire.ArtifactDataError) as caught:
            await wire.call_verb(
                ARTIFACT_SLUG, "whoami", {}, tokens=_tokens(fresh="tok-2"), client=http
            )
    assert caught.value.code == "upstream_auth"


@pytest.mark.asyncio
async def test_no_login_raises_login_required_without_a_request() -> None:
    async def none() -> wire.Bearer | None:
        return None

    def respond(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request may be sent without a bearer")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(wire.ArtifactDataError) as caught:
            await wire.call_verb(ARTIFACT_SLUG, "whoami", {}, tokens=none, client=http)
    assert caught.value.code == "login_required"
    assert "/login anthropic" in caught.value.message


def test_whoami_ids_are_validated() -> None:
    assert wire.user_id_from_whoami({"id": "u_abc"}) == "u_abc"
    for bad in ("../x", "u_abc\n"):
        with pytest.raises(wire.ArtifactDataError):
            wire.user_id_from_whoami({"id": bad})


def test_a_trailing_newline_is_never_a_valid_id_or_segment() -> None:
    # ``$`` matches before a final newline; every check must reject it.
    with pytest.raises(wire.ArtifactDataError):
        wire.check_doc_id("doc-1\n")
    with pytest.raises(wire.ArtifactDataError):
        wire.collection_segments("items\n")
    with pytest.raises(wire.ArtifactDataError):
        wire.collection_segments("data/users/me\n")
    with pytest.raises(wire.ArtifactDataError):
        wire.decode_short_id("GiE3GRkwXeG1q827vXX6g7\n")
    with pytest.raises(wire.ArtifactDataError):
        wire.check_doc_id("..\n")
