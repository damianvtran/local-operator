"""Agent profile pictures: normalise, store, remove, metadata, hook, index.

The contract under test is ``local_operator.agent_avatar``'s module docstring:
one 512x512 PNG (JPEG over 1 MiB), pixels only, EXIF orientation applied,
sha256 of the stored bytes in ``agent.yml``, reason words on refusal.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
import yaml

from local_operator import agent_avatar
from local_operator.agents import AgentEditFields, AgentRegistry

Image = pytest.importorskip("PIL.Image")


def _jpeg_with_exif(size=(800, 400), orientation=6) -> bytes:
    """A landscape JPEG whose EXIF says "rotate 90° CW" and carries GPS."""
    image = Image.new("RGB", size, (200, 10, 10))
    # Mark the LEFT half green so the transposition is observable in pixels.
    image.paste((10, 200, 10), (0, 0, size[0] // 2, size[1]))
    exif = Image.Exif()
    exif[0x0112] = orientation
    exif[0x010F] = "SecretCamCo"  # Make
    exif[0x8825] = {1: "N", 2: (37.0, 46.0, 30.0), 3: "W", 4: (122.0, 25.0, 10.0)}
    out = io.BytesIO()
    image.save(out, "JPEG", exif=exif, quality=95)
    return out.getvalue()


def _png(size=(64, 64), color=(0, 0, 255, 255), mode="RGBA") -> bytes:
    out = io.BytesIO()
    Image.new(mode, size, color[: len(mode)]).save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def registry(tmp_path: Path) -> AgentRegistry:
    return AgentRegistry(config_dir=tmp_path, refresh_interval=0.0)


@pytest.fixture
def agent(registry: AgentRegistry):
    return registry.create_agent(AgentEditFields(name="picbot"))


# -- normalise ---------------------------------------------------------------


def test_normalise_strips_exif_and_gps_and_applies_rotation():
    source = _jpeg_with_exif()
    assert b"SecretCamCo" in source  # the control: the input really carries it
    result = agent_avatar.normalise(source)
    assert result.mime == "image/png"
    assert b"SecretCamCo" not in result.data
    image = Image.open(io.BytesIO(result.data))
    assert image.size == (512, 512)
    assert image.format == "PNG"
    # Nothing ancillary survives: no EXIF, no ICC, no text chunks.
    assert not image.getexif()
    assert "exif" not in image.info and "icc_profile" not in image.info
    assert set(image.info) <= {"dpi", "gamma"}  # PIL may synthesise none of these
    # Orientation 6 rotates the left (green) half to the TOP. Centre-cropped to
    # a square, the top row is green and the bottom row red.
    top = image.convert("RGB").getpixel((256, 5))
    bottom = image.convert("RGB").getpixel((256, 506))
    assert top[1] > 150 and top[0] < 80, top
    assert bottom[0] > 150 and bottom[1] < 80, bottom


def test_normalise_keeps_alpha_png_and_centre_crops():
    result = agent_avatar.normalise(_png(size=(300, 100)))
    image = Image.open(io.BytesIO(result.data))
    assert image.size == (512, 512) and image.mode == "RGBA"


def test_normalise_takes_gif_first_frame():
    frames = [Image.new("RGB", (40, 40), c) for c in ((255, 0, 0), (0, 255, 0))]
    out = io.BytesIO()
    frames[0].save(out, "GIF", save_all=True, append_images=frames[1:])
    image = Image.open(io.BytesIO(agent_avatar.normalise(out.getvalue()).data))
    assert getattr(image, "n_frames", 1) == 1
    r, g, _b = image.convert("RGB").getpixel((256, 256))
    assert r > 200 and g < 50


def test_normalise_accepts_webp():
    out = io.BytesIO()
    Image.new("RGB", (50, 50), (1, 2, 3)).save(out, "WEBP")
    assert agent_avatar.normalise(out.getvalue()).mime == "image/png"


def test_over_cap_png_falls_back_to_jpeg(monkeypatch):
    monkeypatch.setattr(agent_avatar, "STORED_MAX_BYTES", 100)
    result = agent_avatar.normalise(_png())
    assert result.mime == "image/jpeg" and result.name == agent_avatar.JPEG_NAME
    image = Image.open(io.BytesIO(result.data))
    assert image.format == "JPEG" and image.size == (512, 512)
    assert not image.getexif()


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b"#!/bin/sh\necho not an image\n", agent_avatar.UNSUPPORTED_TYPE),
        (b"BM" + b"\0" * 64, agent_avatar.UNSUPPORTED_TYPE),  # BMP is not in the gate
        # A valid PNG signature and IHDR, then garbage: sniffed, not decodable.
        (_png()[:40] + b"\0" * 40, agent_avatar.UNDECODABLE),
    ],
)
def test_refusal_words(data, reason):
    with pytest.raises(agent_avatar.AvatarError) as caught:
        agent_avatar.normalise(data)
    assert caught.value.reason == reason


def test_too_large_refused_before_decode(monkeypatch):
    monkeypatch.setattr(agent_avatar, "INPUT_MAX_BYTES", 10)
    with pytest.raises(agent_avatar.AvatarError) as caught:
        agent_avatar.normalise(_png())
    assert caught.value.reason == agent_avatar.TOO_LARGE


def test_too_many_pixels_refused_from_header():
    # 8000x6000 = 48 MP: a real header, refused before any decode.
    out = io.BytesIO()
    Image.new("1", (8000, 6000)).save(out, "PNG")
    with pytest.raises(agent_avatar.AvatarError) as caught:
        agent_avatar.normalise(out.getvalue())
    assert caught.value.reason == agent_avatar.TOO_MANY_PIXELS


def test_missing_pillow_is_a_reasoned_refusal(monkeypatch):
    from local_operator import helpers

    monkeypatch.setattr(helpers, "pillow_image_module", lambda: None)
    with pytest.raises(agent_avatar.AvatarError) as caught:
        agent_avatar.normalise(_png())
    assert caught.value.reason == agent_avatar.IMAGING_UNAVAILABLE


def test_refusal_envelope_carries_reason_as_code():
    body = agent_avatar.refusal_envelope(agent_avatar.TOO_LARGE)
    assert body["code"] == "too_large"
    assert "10" in body["text"]


# -- registry store / remove -------------------------------------------------


def test_set_writes_file_and_meta_with_matching_sha(registry, agent):
    meta = registry.set_agent_avatar(agent.id, _jpeg_with_exif())
    path = registry.agents_dir / agent.id / "avatar.png"
    data = path.read_bytes()
    assert meta.sha256 == hashlib.sha256(data).hexdigest()
    assert meta.bytes == len(data) and meta.mime == "image/png"
    on_disk = yaml.safe_load((registry.agents_dir / agent.id / "agent.yml").read_text())
    assert on_disk["avatar"]["sha256"] == meta.sha256
    # A fresh registry (another process) reads the same authority.
    reread = AgentRegistry(config_dir=registry.config_dir).get_agent(agent.id)
    assert reread.avatar is not None and reread.avatar.sha256 == meta.sha256


def test_replace_removes_stale_sibling(registry, agent, monkeypatch):
    registry.set_agent_avatar(agent.id, _png())
    agent_dir = registry.agents_dir / agent.id
    assert (agent_dir / "avatar.png").exists()
    monkeypatch.setattr(agent_avatar, "STORED_MAX_BYTES", 100)
    meta = registry.set_agent_avatar(agent.id, _png(color=(9, 9, 9, 255)))
    assert meta.mime == "image/jpeg"
    assert (agent_dir / "avatar.jpg").exists() and not (agent_dir / "avatar.png").exists()
    # And back: the JPEG goes when a PNG fits again.
    monkeypatch.setattr(agent_avatar, "STORED_MAX_BYTES", 1024 * 1024)
    registry.set_agent_avatar(agent.id, _png())
    assert (agent_dir / "avatar.png").exists() and not (agent_dir / "avatar.jpg").exists()
    # No staging temp files left behind by the atomic writes.
    assert not [p for p in agent_dir.iterdir() if p.name.endswith(".tmp")]


def test_refused_write_leaves_nothing(registry, agent):
    with pytest.raises(agent_avatar.AvatarError):
        registry.set_agent_avatar(agent.id, b"nope")
    assert registry.get_agent(agent.id).avatar is None
    assert not (registry.agents_dir / agent.id / "avatar.png").exists()


def test_unknown_agent_is_keyerror(registry):
    with pytest.raises(KeyError):
        registry.set_agent_avatar("does-not-exist", _png())
    with pytest.raises(KeyError):
        registry.clear_agent_avatar("does-not-exist")


def test_clear_removes_file_and_meta(registry, agent):
    registry.set_agent_avatar(agent.id, _png())
    assert registry.clear_agent_avatar(agent.id) is True
    assert registry.get_agent(agent.id).avatar is None
    assert not any((registry.agents_dir / agent.id / n).exists() for n in agent_avatar.STORED_NAMES)
    assert registry.clear_agent_avatar(agent.id) is False


def test_update_agent_keeps_avatar(registry, agent):
    meta = registry.set_agent_avatar(agent.id, _png())
    registry.update_agent(agent.id, AgentEditFields(description="edited"))
    assert registry.get_agent(agent.id).avatar.sha256 == meta.sha256


def test_delete_agent_removes_picture(registry, agent):
    registry.set_agent_avatar(agent.id, _png())
    registry.delete_agent(agent.id)
    assert not (registry.agents_dir / agent.id).exists()


def test_clone_carries_picture(registry, agent):
    meta = registry.set_agent_avatar(agent.id, _png())
    clone = registry.clone_agent(agent.id, "picbot-2")
    assert clone.avatar is not None and clone.avatar.sha256 == meta.sha256
    assert (registry.agents_dir / clone.id / "avatar.png").read_bytes() == (
        registry.agents_dir / agent.id / "avatar.png"
    ).read_bytes()


def test_picture_never_rides_an_export_archive(registry, agent):
    import zipfile

    registry.set_agent_avatar(agent.id, _png())
    _temp_dir, zip_path, _name = registry.export_agent_archive(agent.id)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        payload = yaml.safe_load(archive.read("agent.yml"))
    assert not names & set(agent_avatar.STORED_NAMES)
    assert "avatar" not in payload


def test_avatar_is_off_the_hub_publish_document(registry, agent):
    from local_operator.agents import instruction_set_fields

    registry.set_agent_avatar(agent.id, _png())
    fields = instruction_set_fields(registry, registry.get_agent(agent.id), {})
    assert "avatar" not in fields


def test_avatar_is_off_the_mesh_wire():
    from local_operator.network.definitions import AGENT_DEFINITION_FIELDS

    assert "avatar" not in AGENT_DEFINITION_FIELDS


def test_picture_alone_claims_an_agent_dir(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "avatar.png").write_bytes(b"x")
    assert AgentRegistry._claims_to_be_an_agent(tmp_path / "x")


# -- the hook ----------------------------------------------------------------


def test_subscribe_fires_after_commit_and_unsubscribes(registry, agent):
    seen: list[agent_avatar.AvatarChange] = []

    def callback(change):
        # Fired AFTER commit: the file and agent.yml are both already there.
        if change.sha256:
            assert change.path is not None and change.path.exists()
            assert registry.get_agent(agent.id).avatar.sha256 == change.sha256
        seen.append(change)

    unsubscribe = agent_avatar.subscribe(callback)
    try:
        meta = registry.set_agent_avatar(agent.id, _png())
        registry.clear_agent_avatar(agent.id)
        registry.clear_agent_avatar(agent.id)  # no-op: no event
    finally:
        unsubscribe()
    registry.set_agent_avatar(agent.id, _png())  # after unsubscribe: not seen
    assert [(c.agent_id, c.sha256, c.mime) for c in seen] == [
        (agent.id, meta.sha256, "image/png"),
        (agent.id, None, None),
    ]
    assert seen[1].path is None


def test_a_raising_subscriber_does_not_fail_the_write(registry, agent):
    def boom(_change):
        raise RuntimeError("consumer bug")

    unsubscribe = agent_avatar.subscribe(boom)
    try:
        assert registry.set_agent_avatar(agent.id, _png()).sha256
    finally:
        unsubscribe()


# -- session row identity ----------------------------------------------------


def _bind(root: Path, session_id: str, agent_name: str) -> None:
    from local_operator.resume import write_session_attachment

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True)
    write_session_attachment(directory, team="", agent=agent_name, goal="")


def test_session_row_fields_resolve_name_to_id_and_sha(registry, agent):
    root = registry.config_dir
    _bind(root, "s1", "picbot")
    (root / "sessions" / "s2").mkdir(parents=True)
    assert agent_avatar.session_row_fields(root, "s1") == {
        "agent_id": agent.id,
        "avatar_sha256": None,
    }
    meta = registry.set_agent_avatar(agent.id, _png())
    assert agent_avatar.session_row_fields(root, "s1")["avatar_sha256"] == meta.sha256
    assert agent_avatar.session_row_fields(root, "s2") == {
        "agent_id": None,
        "avatar_sha256": None,
    }


def test_aida_session_without_attachment_resolves_her_row(registry):
    from local_operator.aida import state as aida_state

    aida = registry.create_agent(AgentEditFields(name="aida"))
    root = registry.config_dir
    (root / "sessions" / "cos1").mkdir(parents=True)
    aida_state.write_state(root, {"session_id": "cos1"})
    assert agent_avatar.session_row_fields(root, "cos1")["agent_id"] == aida.id


def test_index_never_creates_registry_state(tmp_path):
    assert agent_avatar.agent_index(tmp_path) == {}
    assert not (tmp_path / "agents").exists()


def test_read_stored_refuses_path_segments(registry, agent):
    registry.set_agent_avatar(agent.id, _png())
    data, mime, sha = agent_avatar.read_stored(registry.config_dir, agent.id)
    assert hashlib.sha256(data).hexdigest() == sha and mime == "image/png"
    for bad in ("..", "../x", "a/b", ""):
        with pytest.raises(agent_avatar.AvatarError) as caught:
            agent_avatar.read_stored(registry.config_dir, bad)
        assert caught.value.reason == agent_avatar.NOT_FOUND


# -- CLI ---------------------------------------------------------------------


def test_cli_set_show_remove(registry, agent, tmp_path, capsys):
    import argparse

    from local_operator import cli_agent_picture

    image = tmp_path / "in.jpg"
    image.write_bytes(_jpeg_with_exif())

    def run(command, ref, file=None):
        ns = argparse.Namespace(picture_command=command, agent=ref, file=file)
        return cli_agent_picture.run(ns, registry)

    assert run("set", "picbot", str(image)) == 0
    assert "picbot" in capsys.readouterr().out
    assert run("show", agent.id) == 0  # by id as well as by name
    shown = json.loads(capsys.readouterr().out)
    assert shown["avatar"]["sha256"] == registry.get_agent(agent.id).avatar.sha256
    bad = tmp_path / "bad.txt"
    bad.write_text("hello")
    assert run("set", "picbot", str(bad)) == 1
    assert json.loads(capsys.readouterr().err)["error"] == "unsupported_type"
    assert run("remove", "picbot") == 0
    capsys.readouterr()
    assert run("show", "picbot") == 0
    assert json.loads(capsys.readouterr().out)["avatar"] is None
    assert run("show", "nobody") == 1


def test_cli_parser_registers_picture_verbs():
    from local_operator.cli import build_cli_parser

    parser = build_cli_parser()
    args = parser.parse_args(["agents", "picture", "set", "picbot", "/tmp/x.png"])
    assert (args.agents_command, args.picture_command, args.agent, args.file) == (
        "picture",
        "set",
        "picbot",
        "/tmp/x.png",
    )


# -- the writers must not drop it ---------------------------------------------


def test_an_update_agent_rewrite_keeps_the_picture_block(registry, agent):
    """A UI-style metadata edit (rename + description) must not lose the picture.

    ``update_agent`` re-dumps the whole row from ``model_dump()``, so the block
    survives only if ``AgentData.avatar`` is a real field -- a sidecar-only or
    ``extra="allow"`` carrier would be dropped here and the picture would turn
    into an orphan file the metadata no longer names (hub round: the desktop
    editor renames rows constantly).
    """
    meta = registry.set_agent_avatar(agent.id, _png())
    from local_operator.agents import AgentEditFields

    updated = registry.update_agent(
        agent.id, AgentEditFields(name="renamed", description="new purpose")
    )
    assert updated.avatar is not None and updated.avatar.sha256 == meta.sha256
    # And it is on DISK, not only in memory.
    stored = yaml.safe_load((registry.config_dir / "agents" / agent.id / "agent.yml").read_text())
    assert stored["avatar"]["sha256"] == meta.sha256
    assert stored["avatar"]["mime"] == "image/png"
    # A fresh registry (a new process) reads the same block back.
    fresh = AgentRegistry(config_dir=registry.config_dir, refresh_interval=0.0)
    assert fresh.get_agent(agent.id).avatar.sha256 == meta.sha256


def test_save_agent_round_trips_the_picture_block(registry, agent):
    meta = registry.set_agent_avatar(agent.id, _png())
    row = registry.get_agent(agent.id)
    registry.save_agent(row)
    assert registry.get_agent(agent.id).avatar.sha256 == meta.sha256


def test_clone_carries_the_picture_and_its_block(registry, agent):
    meta = registry.set_agent_avatar(agent.id, _png())
    clone = registry.clone_agent(agent.id, "picbot-copy")
    assert clone.avatar is not None and clone.avatar.sha256 == meta.sha256
    assert (registry.config_dir / "agents" / clone.id / agent_avatar.PNG_NAME).exists()


def test_import_drops_an_archives_picture_block(registry, agent, tmp_path):
    """No cloud->local picture write: an imported archive's block is discarded."""
    import zipfile

    archive = tmp_path / "in.zip"
    payload = {
        "id": "0123456789ab",
        "name": "imported",
        "label": "",
        "created_date": "2026-01-01T00:00:00+00:00",
        "version": "0.0.0",
        "avatar": {
            "sha256": "0" * 64,
            "mime": "image/png",
            "bytes": 1,
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    }
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("agent.yml", yaml.safe_dump(payload))
    imported = registry.import_agent(archive)
    assert imported.agent.avatar is None
    assert not (registry.config_dir / "agents" / imported.agent.id / "avatar.png").exists()


# -- round-1 remediation: contract filename, stale registries, hostile files -----


def _yml(registry, agent_id) -> dict:
    return yaml.safe_load((registry.agents_dir / agent_id / "agent.yml").read_text())


def _rewrite_block(registry, agent_id, **changes) -> None:
    """Edit the on-disk ``avatar`` block the way a hand edit or an agent could."""
    path = registry.agents_dir / agent_id / "agent.yml"
    payload = yaml.safe_load(path.read_text())
    for key, value in changes.items():
        if value is None:
            payload["avatar"].pop(key, None)
        else:
            payload["avatar"][key] = value
    path.write_text(yaml.safe_dump(payload))


def test_writers_always_record_the_stored_filename(registry, agent, monkeypatch):
    meta = registry.set_agent_avatar(agent.id, _png())
    assert meta.filename == "avatar.png"
    assert _yml(registry, agent.id)["avatar"]["filename"] == "avatar.png"
    assert agent_avatar.meta_json(meta)["filename"] == "avatar.png"
    monkeypatch.setattr(agent_avatar, "STORED_MAX_BYTES", 100)
    flipped = registry.set_agent_avatar(agent.id, _png())
    assert flipped.filename == "avatar.jpg" and flipped.mime == "image/jpeg"
    assert _yml(registry, agent.id)["avatar"]["filename"] == "avatar.jpg"


def test_a_legacy_block_without_filename_falls_back_by_mime(registry, agent):
    meta = registry.set_agent_avatar(agent.id, _png())
    _rewrite_block(registry, agent.id, filename=None)
    fresh = AgentRegistry(config_dir=registry.config_dir, refresh_interval=0.0)
    row = fresh.get_agent(agent.id)
    assert row.avatar is not None and row.avatar.filename is None
    data, mime, sha = agent_avatar.read(fresh.agents_dir / agent.id, row.avatar)
    assert (mime, sha) == ("image/png", meta.sha256)
    assert agent_avatar.read_stored(fresh.config_dir, agent.id)[2] == meta.sha256
    # The listing wire form always reports the EFFECTIVE name.
    assert agent_avatar.meta_json(row.avatar)["filename"] == "avatar.png"


@pytest.mark.parametrize("hostile", ["../x", "/etc/passwd", "sub/avatar.png", "avatar.jpg", "x\0y"])
def test_a_filename_that_is_not_the_plain_stored_name_reads_as_no_avatar(registry, agent, hostile):
    registry.set_agent_avatar(agent.id, _png())
    _rewrite_block(registry, agent.id, filename=hostile)
    fresh = AgentRegistry(config_dir=registry.config_dir, refresh_interval=0.0)
    with pytest.raises(agent_avatar.AvatarError) as caught:
        agent_avatar.read(fresh.agents_dir / agent.id, fresh.get_agent(agent.id).avatar)
    assert caught.value.reason == agent_avatar.NOT_FOUND
    with pytest.raises(agent_avatar.AvatarError):
        agent_avatar.read_stored(fresh.config_dir, agent.id)
    assert fresh.agent_avatar_path(agent.id) is None
    # ...and the row index advertises no picture for it either.
    agent_avatar._index_cache.clear()
    assert (
        agent_avatar.session_agent_identity(
            fresh.config_dir, "no-such-session", agent_name="picbot"
        ).avatar_sha256
        is None
    )


def test_a_symlink_swapped_in_for_the_picture_is_not_served(registry, agent, tmp_path):
    registry.set_agent_avatar(agent.id, _png())
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"TOPSECRET-FILE-CONTENT\n")
    target = registry.agents_dir / agent.id / "avatar.png"
    target.unlink()
    target.symlink_to(outside)
    meta = registry.get_agent(agent.id).avatar
    with pytest.raises(agent_avatar.AvatarError) as caught:
        agent_avatar.read(registry.agents_dir / agent.id, meta)
    assert caught.value.reason == agent_avatar.NOT_FOUND
    with pytest.raises(agent_avatar.AvatarError):
        agent_avatar.read_stored(registry.config_dir, agent.id)
    assert registry.agent_avatar_path(agent.id) is None


def test_bytes_that_no_longer_hash_to_the_recorded_sha_are_not_served(registry, agent):
    registry.set_agent_avatar(agent.id, _png())
    path = registry.agents_dir / agent.id / "avatar.png"
    path.write_bytes(path.read_bytes() + b"appended")
    with pytest.raises(agent_avatar.AvatarError) as caught:
        agent_avatar.read(registry.agents_dir / agent.id, registry.get_agent(agent.id).avatar)
    assert caught.value.reason == agent_avatar.NOT_FOUND
    assert "digest" in caught.value.detail


def test_served_sha_is_the_digest_of_the_bytes_returned(registry, agent):
    registry.set_agent_avatar(agent.id, _png())
    data, _mime, sha = agent_avatar.read(
        registry.agents_dir / agent.id, registry.get_agent(agent.id).avatar
    )
    assert hashlib.sha256(data).hexdigest() == sha


@pytest.mark.parametrize("bad", ["a\0b", "..\0", "\0", "ok\0"])
def test_a_nul_in_an_id_is_not_found_not_a_valueerror(registry, agent, bad):
    registry.set_agent_avatar(agent.id, _png())
    assert agent_avatar.is_plain_agent_id(bad) is False
    with pytest.raises(agent_avatar.AvatarError) as caught:
        agent_avatar.read_stored(registry.config_dir, bad)
    assert caught.value.reason == agent_avatar.NOT_FOUND
    assert agent_avatar.stored_sha(registry.config_dir, bad) is None


# R1: a registry holding a stale row must not erase a picture another process set.
# ``refresh_interval=3.0`` is the server's real setting; every other test uses 0.0,
# which refreshes on every call and so can never see this.


def test_a_stale_registry_update_keeps_a_picture_another_process_set(tmp_path):
    server = AgentRegistry(config_dir=tmp_path, refresh_interval=3.0)
    agent = server.create_agent(AgentEditFields(name="racer"))
    server.get_agent(agent.id)  # the server's row is now cached, with avatar=None
    cli = AgentRegistry(config_dir=tmp_path, refresh_interval=0.0)
    meta = cli.set_agent_avatar(agent.id, _png())
    assert server._agents[agent.id].avatar is None  # the stale precondition holds

    server.update_agent(agent.id, AgentEditFields(description="edited by the editor"))

    after = AgentRegistry(config_dir=tmp_path, refresh_interval=0.0).get_agent(agent.id)
    assert after.description == "edited by the editor"
    assert after.avatar is not None and after.avatar.sha256 == meta.sha256
    assert (tmp_path / "agents" / agent.id / "avatar.png").exists()


def test_a_stale_registry_set_keeps_other_processes_edits(tmp_path):
    server = AgentRegistry(config_dir=tmp_path, refresh_interval=3.0)
    agent = server.create_agent(AgentEditFields(name="racer2"))
    server.get_agent(agent.id)
    AgentRegistry(config_dir=tmp_path, refresh_interval=0.0).update_agent(
        agent.id, AgentEditFields(description="written by another process")
    )

    server.set_agent_avatar(agent.id, _png())

    after = AgentRegistry(config_dir=tmp_path, refresh_interval=0.0).get_agent(agent.id)
    assert after.description == "written by another process"
    assert after.avatar is not None


def test_a_stale_registry_clear_keeps_other_processes_edits(tmp_path):
    server = AgentRegistry(config_dir=tmp_path, refresh_interval=3.0)
    agent = server.create_agent(AgentEditFields(name="racer3"))
    server.set_agent_avatar(agent.id, _png())
    server.get_agent(agent.id)
    AgentRegistry(config_dir=tmp_path, refresh_interval=0.0).update_agent(
        agent.id, AgentEditFields(description="written by another process")
    )

    assert server.clear_agent_avatar(agent.id) is True

    after = AgentRegistry(config_dir=tmp_path, refresh_interval=0.0).get_agent(agent.id)
    assert after.description == "written by another process" and after.avatar is None


# R3: every interruption of a mime flip reads as the old picture or the new one.


def test_a_failed_yml_write_during_a_png_to_jpeg_flip_keeps_the_old_picture(
    registry, agent, monkeypatch
):
    old = registry.set_agent_avatar(agent.id, _png())
    monkeypatch.setattr(agent_avatar, "STORED_MAX_BYTES", 100)

    def _boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("local_operator.agents._write_text_atomically", _boom)
    with pytest.raises(OSError):
        registry.set_agent_avatar(agent.id, _png())
    monkeypatch.undo()

    fresh = AgentRegistry(config_dir=registry.config_dir, refresh_interval=0.0)
    row = fresh.get_agent(agent.id)
    assert row.avatar is not None and row.avatar.sha256 == old.sha256
    assert agent_avatar.read(fresh.agents_dir / agent.id, row.avatar)[2] == old.sha256
    assert not (fresh.agents_dir / agent.id / "avatar.jpg").exists()  # no orphan


def test_dying_after_the_yml_write_reads_as_the_new_picture(registry, agent, monkeypatch):
    registry.set_agent_avatar(agent.id, _png())
    monkeypatch.setattr(agent_avatar, "STORED_MAX_BYTES", 100)

    def _die(*_args, **_kwargs):
        raise KeyboardInterrupt  # the process dies before the sibling is removed

    real_remove_siblings = agent_avatar.remove_siblings
    monkeypatch.setattr(agent_avatar, "remove_siblings", _die)
    with pytest.raises(KeyboardInterrupt):
        registry.set_agent_avatar(agent.id, _png())
    # Restore only the interrupted step; the 100-byte cap stays so the next
    # write is again a JPEG and must sweep the stray PNG.
    monkeypatch.setattr(agent_avatar, "remove_siblings", real_remove_siblings)

    fresh = AgentRegistry(config_dir=registry.config_dir, refresh_interval=0.0)
    row = fresh.get_agent(agent.id)
    assert row.avatar is not None and row.avatar.filename == "avatar.jpg"
    # ``read`` bounds itself by the REAL cap; the 100-byte test cap would reject
    # any JPEG, so lift it for the read and restore it for the sweeping write.
    monkeypatch.setattr(agent_avatar, "STORED_MAX_BYTES", 1024 * 1024)
    data, mime, _sha = agent_avatar.read(fresh.agents_dir / agent.id, row.avatar)
    monkeypatch.setattr(agent_avatar, "STORED_MAX_BYTES", 100)
    assert mime == "image/jpeg" and data.startswith(b"\xff\xd8")
    # The stray PNG is harmless (yml is the authority) and the next write sweeps it.
    assert (fresh.agents_dir / agent.id / "avatar.png").exists()
    fresh.set_agent_avatar(agent.id, _png())
    assert sorted(
        n
        for n in (p.name for p in (fresh.agents_dir / agent.id).iterdir())
        if n.startswith("avatar")
    ) == ["avatar.jpg"]


# R4: staging temps and case variants never ride an archive or a clone.


def test_staging_temps_and_case_variants_stay_out_of_export_and_clone(registry, agent):
    import zipfile

    registry.set_agent_avatar(agent.id, _png())
    agent_dir = registry.agents_dir / agent.id
    (agent_dir / ".avatar.png.4242.deadbeef.tmp").write_bytes(b"photo bytes")
    (agent_dir / "Avatar.PNG").write_bytes(b"photo bytes")

    _temp_dir, zip_path, _name = registry.export_agent_archive(agent.id)
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert not [n for n in names if n.lower().startswith(("avatar", ".avatar"))], names

    clone = registry.clone_agent(agent.id, "picbot-clone")
    cloned = {p.name for p in (registry.agents_dir / clone.id).iterdir()}
    assert ".avatar.png.4242.deadbeef.tmp" not in cloned
    assert "avatar.png" in cloned  # the real picture is still carried on purpose


# R5: 16-bit greyscale must scale, not saturate.


def test_a_16_bit_greyscale_png_keeps_its_gradient():
    gradient = Image.new("I;16", (256, 256))
    gradient.putdata([x * 256 for _y in range(256) for x in range(256)])
    out = io.BytesIO()
    gradient.save(out, "PNG")
    assert Image.open(io.BytesIO(out.getvalue())).mode.startswith("I")  # the control
    image = Image.open(io.BytesIO(agent_avatar.normalise(out.getvalue()).data)).convert("L")
    dark, mid, light = (image.getpixel((x, 256)) for x in (100, 256, 410))
    # Output x maps to source x/2, whose 16-bit value scales to ~x/2 in 8 bits.
    # The old clamp made every one of these 255.
    assert dark < mid < light < 255, (dark, mid, light)
    assert abs(dark - 50) <= 3 and abs(mid - 128) <= 3 and abs(light - 205) <= 3
