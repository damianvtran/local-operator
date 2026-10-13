"""Pin static copy independently of catalogues; policy refusals remain strings."""

import pytest
from fastapi import HTTPException, Request

from local_operator.i18n.keys import wire_errors_files
from local_operator.i18n.resolve import shipped_locales
from local_operator.server.routes import static


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route,kind", [("images", "image"), ("videos", "video"), ("audio", "audio"), ("html", "HTML")]
)
async def test_mime_envelope(test_app_client, temp_dir, route, kind):
    directory = temp_dir / "uploads"
    directory.mkdir(exist_ok=True)
    path = directory / "opaque.txt"
    path.write_bytes(b"unchanged")
    response = await test_app_client.get(
        f"/v1/static/{route}", params={"path": str(path)}, headers={"Host": "127.0.0.1:1111"}
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": {
            "code": f"wire.errors.files.{kind.lower()}_type",
            "params": {"mime_type": "text/plain"},
            "text": f"File is not an allowed {kind} type: text/plain",
        }
    }
    assert path.read_bytes() == b"unchanged"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler", [static.get_image, static.get_video, static.get_audio, static.get_html]
)
async def test_internal_error_tail(monkeypatch, handler):
    # Arbitrary exception tails are opaque, not ICU syntax or sentence fragments.
    tail = "opaque {braces}: unicode λ /private/path END"

    def fail(*args):
        raise RuntimeError(tail)

    monkeypatch.setattr(static, "_servable_path", fail)
    with pytest.raises(HTTPException) as caught:
        await handler(Request({"type": "http"}), "/synthetic")
    assert caught.value.status_code == 500
    assert caught.value.detail == {
        "code": "wire.errors.files.internal_error",
        "params": {"error": tail},
        "text": "Internal Server Error: " + tail,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,text,code,params",
    [
        (
            {"route": "no", "path": "/x"},
            "route must be one of: images, videos, audio, html",
            "sign_route",
            {"routes": "images, videos, audio, html"},
        ),
        ({"route": "images", "path": ""}, "path must not be empty.", "sign_path", {}),
        (
            {"route": "images", "path": "/x", "ttl_s": 0},
            "ttl_s must be between 1 and 3600.",
            "sign_ttl",
            {"maximum": "3600"},
        ),
        (
            {"route": "images", "path": "/x", "ttl_s": 3601},
            "ttl_s must be between 1 and 3600.",
            "sign_ttl",
            {"maximum": "3600"},
        ),
    ],
)
async def test_sign_validation_copy(body, text, code, params):
    # Bypass ONLY the dependency to isolate body validation; TCP checks the door.
    with pytest.raises(HTTPException) as caught:
        await static.sign_static_url(static.SignRequest(**body))
    assert caught.value.status_code == 400
    assert caught.value.detail == {
        "code": "wire.errors.files." + code,
        "params": params,
        "text": text,
    }


@pytest.mark.asyncio
async def test_heic_conversion_tail(monkeypatch, tmp_path):
    path = tmp_path / "photo.heic"
    monkeypatch.setattr(static, "_servable_path", lambda *args: path)

    def fail(path):
        raise RuntimeError("decode {opaque} END")

    monkeypatch.setattr(static, "convert_heic_to_png_file", fail)
    with pytest.raises(HTTPException) as caught:
        await static.get_image(Request({"type": "http"}), str(path))
    assert caught.value.status_code == 500
    assert caught.value.detail == {
        "code": "wire.errors.files.heic_conversion",
        "params": {"error": "decode {opaque} END"},
        "text": "Failed to convert HEIC/HEIF file: decode {opaque} END",
    }


@pytest.mark.asyncio
async def test_html_invalid_utf8_and_read_failure(monkeypatch, tmp_path):
    path = tmp_path / "preview.html"
    path.write_bytes(b"\xff")
    monkeypatch.setattr(static, "_servable_path", lambda *args: path)
    with pytest.raises(HTTPException) as caught:
        await static.get_html(Request({"type": "http"}), str(path))
    assert caught.value.status_code == 400
    assert caught.value.detail == {
        "code": "wire.errors.files.html_utf8",
        "params": {},
        "text": "HTML file is not valid UTF-8 text.",
    }
    assert path.read_bytes() == b"\xff"
    from pathlib import Path

    original_read = Path.read_text

    def fail(target, *args, **kwargs):
        if target == path:
            raise OSError("read {opaque} END")
        return original_read(target, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail)
    with pytest.raises(HTTPException) as caught:
        await static.get_html(Request({"type": "http"}), str(path))
    assert caught.value.status_code == 500
    assert caught.value.detail == {
        "code": "wire.errors.files.internal_error",
        "params": {"error": "read {opaque} END"},
        "text": "Internal Server Error: read {opaque} END",
    }


def test_new_namespace_english_fallback():
    # This slice must not invalidate translated locales or need translated keys.
    expected = {
        "code": "wire.errors.files.sign_ttl",
        "params": {"maximum": "3600"},
        "text": "ttl_s must be between 1 and 3600.",
    }
    for locale in shipped_locales():
        assert wire_errors_files.sign_ttl(maximum="3600").envelope(locale=locale) == expected
