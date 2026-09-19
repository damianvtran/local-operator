#!/usr/bin/env python3
"""Reproduce "an image send is refused, the same message without one is not".

WHAT THIS REPRODUCES, AND WHY IT NEEDS A DISK IMAGE.
On 2026-09-17 the boot volume hit zero bytes free. Sending a chat message WITH
an image to a running session was refused with "Read state is busy right now. It
will catch up on its own." while the identical message without the image went
through, and it started working again once space came back. Two facts have to be
demonstrated together for that to be a finding rather than a guess:

* WHICH write fails first. The image send is the largest write in the flow
  (attachment bytes plus a much bigger transcript append plus the store writes
  that follow it), so on a nearly-full volume it crosses the threshold while a
  few-KB text write still lands. Both sends are therefore issued at the SAME
  measured free space, in ONE run, and both outcomes are printed.
* WHAT THE CLIENT ACTUALLY RECEIVED. Not the exception -- the status and body a
  renderer gets, which is what the user saw.

Doing that on the operator's own volume means filling the operator's own volume,
which is how the incident happened; so everything here lives on a BOUNDED APFS
image (`hdiutil create -size 30m -fs APFS`) that is detached again on the way
out. Nothing outside the image is written, and the image is at most the size
passed to ``--image-mb``.

The stack is real throughout: the real FastAPI app over uvicorn, a real
``Session`` over a real transcript, a real ``RuntimeServer``, and the real HTTP
desktop surface. Only the provider stream is scripted (``tests/e2e/harness``),
which is the same substitution the e2e suite makes.

Run it under a worktree venv, from the repository root:

    .venv/bin/python scripts/repro-enospc-send.py --free-kb 1900

It exits non-zero when the pairing does NOT reproduce, so a silent pass is not
possible: on a machine with different APFS headroom the free-space target has to
be re-bisected (``--free-kb``), and the script says which side it landed on.

The same condition is covered in CI without a disk image, at the classification
level, by ``tests/unit/server/test_desktop_store_failures.py`` -- an ENOSPC or a
``SQLITE_CANTOPEN`` on a full volume must answer 507 ``store_out_of_space``, not
the retryable 503 the incident produced.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

#: The repository this script lives in, put on the path FIRST. ``tests.e2e``
#: supplies the one substitution this repro is allowed (the provider stream) and
#: is not an installed package, and a script's own directory -- not the working
#: directory -- is what Python puts on ``sys.path[0]``. Doing it explicitly also
#: means the run exercises the TREE THE SCRIPT LIVES IN, not whichever tree the
#: venv's editable install happens to point at.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Where the bounded image is created and mounted. Outside the repository and
#: outside any operator state: the whole point is that this run cannot fill a
#: volume anybody else needs.
IMAGE_FS = "APFS"
VOLUME_NAME = "lo-enospc-repro"

#: The image generated for the send. Contents are irrelevant to the repro, but
#: the SIZE is not: it must exceed the attachment store's externalization floor
#: (``transcript._ATTACHMENT_FLOOR_BYTES``, 1024 bytes of base64) or the image
#: never reaches the store and the run proves nothing.
IMAGE_SIDE = 300


def create_image(workdir: Path, megabytes: int) -> Path:
    image = workdir / "enospc.dmg"
    subprocess.run(
        [
            "hdiutil",
            "create",
            "-size",
            f"{megabytes}m",
            "-fs",
            IMAGE_FS,
            "-volname",
            VOLUME_NAME,
            "-quiet",
            str(image),
        ],
        check=True,
        capture_output=True,
    )
    return image


def mount(image: Path) -> Path:
    subprocess.run(
        ["hdiutil", "attach", str(image), "-nobrowse", "-quiet"],
        check=True,
        capture_output=True,
    )
    return Path("/Volumes") / VOLUME_NAME


def detach(mountpoint: Path) -> None:
    subprocess.run(
        ["hdiutil", "detach", str(mountpoint), "-force", "-quiet"],
        check=False,
        capture_output=True,
    )


def free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def fill_to(path: Path, target_free: int) -> int:
    """Fill ``path`` until free space lands on ``target_free``.

    Grown in small sequential writes, then SHORTENED by truncation. Both halves
    are necessary and both are measured, not assumed: ``mkfile`` preallocates in
    one call and is refused with 16 MB still showing free on a fragmented image,
    and growth alone stops ~1.3 MB above a sub-megabyte target because APFS
    reports free space it will not honour. Truncating the filler is the only
    lever that moves free space back UP from the low end it can reach, which is
    what makes a "text fits, image does not" window reachable at all.
    """
    filler = path / "filler.bin"
    chunk = bytes(64 * 1024)
    for _ in range(4000):
        if free_bytes(path) <= target_free:
            break
        try:
            with open(filler, "ab") as handle:
                handle.write(chunk)
                handle.flush()
        except OSError:
            break
    for _ in range(8):
        free = free_bytes(path)
        if free >= target_free:
            break
        size = filler.stat().st_size if filler.exists() else 0
        new_size = size - (target_free - free)
        if new_size <= 0 or new_size >= size:
            break
        try:
            os.truncate(filler, new_size)
        except OSError:
            break
    return free_bytes(path)


def prepare_env(root: Path, token: str) -> None:
    """Isolate the run COMPLETELY, then import the app.

    ``HOME`` as well as ``LOCAL_OPERATOR_CONFIG_DIR``: the cache root is derived
    from the home directory independently, so a run with only the config var set
    reads and writes the operator's real cache while believing it is isolated. A
    ``lop`` parent also exports two prefixes the child product itself reads --
    ``CMUX_*`` (a headless run that inherits ``CMUX_WORKSPACE_ID`` renames the
    operator's real cmux workspaces) and ``LOP_*`` (the child runtime's provider
    and model, plus a deferral flag that can make a child idle-exit) -- so both
    are stripped before anything is imported.
    """
    os.environ["HOME"] = str(root / "home")
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root / "cfg")
    os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"] = token
    os.environ.pop("LOCAL_OPERATOR_DESKTOP_ORIGINS", None)
    os.environ["LOCAL_OPERATOR_NO_SHIMMER"] = "1"
    os.environ["LOCAL_OPERATOR_NO_NOTIFICATIONS"] = "1"
    os.environ["LOCAL_OPERATOR_NO_TERMINAL_TITLE"] = "1"
    for name in [key for key in os.environ if key.startswith(("CMUX_", "LOP_"))]:
        del os.environ[name]


async def run(root: Path, target_free: int, workspace: Path) -> int:
    token = secrets.token_hex(32)
    prepare_env(root, token)
    config = root / "cfg"
    config.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )

    import httpx
    import uvicorn
    from PIL import Image as PILImage

    from local_operator.mobile.attach_client import AttachClient, find_runtime_record
    from local_operator.server.app import app
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.e2e.harness import ScriptedStream, build_session, text_turn

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    runtime = terminal = None
    try:
        for _ in range(20_000):
            if server.started:
                break
            if serving.done():
                await serving
            await asyncio.sleep(0)
        assert server.started

        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=60) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            listing = "/v1/desktop/sessions"
            created = await client.post(
                listing,
                json={
                    "request_id": "11111111-1111-4111-8111-111111111111",
                    "cwd": str(workspace),
                },
            )
            assert created.status_code == 200, created.text
            session_id = created.json()["result"]["session_id"]
            target = f"{listing}/{session_id}"

            stream = ScriptedStream(
                [text_turn("Text send answered."), text_turn("Image send answered.")]
            )
            session = build_session(config / "sessions" / session_id, stream, cwd=workspace)
            # Named, so the separate title-model errand does not consume one of
            # the two scripted turns.
            session.set_conversation_name("ENOSPC repro", user_set=True)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            (config / "sessions" / session_id / ".session.pid").write_text(str(os.getpid()))
            record, _ = find_runtime_record(config, session_id)
            assert record is not None
            terminal = AttachClient(lambda _: None, lambda _: None)
            await terminal.connect(record, session_id)

            # A live subscriber, the way the app's own window is one.
            stream_context = client.stream("GET", target + "/events")
            response = await stream_context.__aenter__()

            async def drain() -> None:
                try:
                    async for _line in response.aiter_lines():
                        pass
                except Exception:  # noqa: BLE001 - the stream closes at teardown
                    pass

            draining = asyncio.create_task(drain())

            buffer = io.BytesIO()
            PILImage.frombytes(
                "RGB", (IMAGE_SIDE, IMAGE_SIDE), os.urandom(IMAGE_SIDE**2 * 3)
            ).save(buffer, format="PNG")
            raw = buffer.getvalue()
            image_b64 = base64.b64encode(raw).decode()
            print(f"image payload: {len(raw)} bytes decoded, {len(image_b64)} chars of base64")

            reached = fill_to(root, target_free)
            print(f"free space on the image after filling: {reached} bytes")
            awaiting = await client.get(target + "/history")
            print(f"session readable before the sends: HTTP {awaiting.status_code}")

            outcomes: dict[str, tuple[int, str]] = {}
            for label, body in (
                (
                    "text-only",
                    {
                        "request_id": "22222222-2222-4222-8222-222222222222",
                        "text": "hello without an image",
                    },
                ),
                (
                    "with-image",
                    {
                        "request_id": "33333333-3333-4333-8333-333333333333",
                        "text": "",
                        "images": [{"mime_type": "image/png", "data_b64": image_b64}],
                    },
                ),
            ):
                before = free_bytes(root)
                sent = await client.post(target + "/messages", json=body)
                outcomes[label] = (sent.status_code, sent.text)
                print(
                    f"\n=== {label} send at free={before} bytes\n"
                    f"    HTTP {sent.status_code}\n    {sent.text}"
                )

            stored = (
                sorted(entry.name for entry in (config / "attachments").iterdir())
                if (config / "attachments").exists()
                else []
            )
            print(f"\nattachment store contents: {stored}")

            draining.cancel()
            await stream_context.__aexit__(None, None, None)
    finally:
        if terminal is not None:
            terminal.close()
        if runtime is not None:
            runtime.close()
        server.should_exit = True
        await serving

    text_status = outcomes["text-only"][0]
    image_status, image_body = outcomes["with-image"]
    print(f"\nverdict: text-only HTTP {text_status}, with-image HTTP {image_status}")
    if text_status == 200 and image_status != 200:
        # WHICH full-disk shape this run landed in is itself the finding, so it
        # is reported rather than summarised as "the image was refused".
        if "store_out_of_space" in image_body:
            print(
                "REPRODUCED (store): the image send was refused because the STORE\n"
                "could not be written, at a free space where the text send's\n"
                "smaller writes still landed."
            )
        else:
            print(
                "REPRODUCED (owner): the text send was admitted and its TURN died\n"
                "writing the transcript, taking the session owner with it, so the\n"
                "image send met a dead owner. Different refusal, same disk."
            )
        return 0
    print(
        "NOT REPRODUCED at this free space. Text and image agreed, which means the fill\n"
        "landed outside the window this volume produces: raise or lower --free-kb and\n"
        "re-run (the two sides are 'both succeed' above it and 'both refuse' below it)."
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--free-kb",
        type=int,
        default=2200,
        help=(
            "free space to leave on the image before sending (default: 2200 KB). The "
            "window is narrow and lands in bands -- on a 30 MB APFS image here they "
            "were ~1.2 MB (both refuse), ~1.3-1.4 MB (the pairing), ~3 MB and up "
            "(both succeed) -- so the verdict below says which side a run landed on."
        ),
    )
    parser.add_argument("--image-mb", type=int, default=30, help="size of the APFS image")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the image mounted for inspection instead of detaching it",
    )
    arguments = parser.parse_args()

    # The image itself lives in a temp dir, on a volume that has room for it.
    workdir = Path(tempfile.mkdtemp(prefix="lo-enospc-repro-"))
    image = create_image(workdir, arguments.image_mb)
    mounted = mount(image)
    try:
        return asyncio.run(run(mounted, arguments.free_kb * 1024, mounted / "ws"))
    finally:
        if arguments.keep:
            print(f"left mounted at {mounted} (image {image})")
        else:
            detach(mounted)
            shutil.rmtree(workdir, ignore_errors=True)
            print(f"detached and removed {workdir}")


if __name__ == "__main__":
    sys.exit(main())
