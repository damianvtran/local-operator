"""One quick-tunnel host per durable session family, not a site permission.

The pin precedes the approval RPC and survives cancellation, transport failure,
process death and resume. Otherwise a failed/denied prompt could be worked around
by minting another host. It lasts as long as the root session directory; ordinary
session retention removes it with that conversation. No grants are created here.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic_core import Url

from local_operator.divergence import ORIGIN_NAME
from local_operator.guardrails.provenance import read_lineage


@dataclass(frozen=True)
class PreviewPin:
    host: str
    origin: str
    allowed: bool


def quick_tunnel_origin(url: str) -> tuple[str, str] | None:
    """Parse real HTTP(S) hosts, not suffixes in paths, credentials or lookalikes."""
    try:
        # Browser HTTP URLs treat backslashes as separators and decode host
        # escapes. Account for those aliases before comparing the parsed host,
        # or a second tunnel could evade the family pin with URL spelling alone.
        # pydantic-core uses the browser-compatible Rust URL parser already
        # shipped with the harness. In particular UTS46 is non-transitional:
        # sharp-s stays distinct from ss, unlike Python's IDNA2003 codec.
        parsed = Url(url)
        host = (parsed.host or "").lower().removesuffix(".")
        port = parsed.port
        if (parsed.scheme, port) in {("https", 443), ("http", 80)}:
            port = None
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not host.endswith(".trycloudflare.com"):
        return None
    if not all(
        re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) for label in host.split(".")
    ):
        return None
    return host, f"{parsed.scheme}://{host}" + (f":{port}" if port is not None else "")


@contextmanager
def _directory(path: Path) -> Iterator[int]:
    """Anchor every component without following links, including the store.

    A path precheck followed by a name-based SQLite open is racy. Directory
    descriptors plus openat keep both marker reads and pin writes anchored to
    genuine directories even if their names are replaced by symlinks midway.
    """
    absolute = path.absolute()
    fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        if os.fstat(fd).st_uid != os.getuid():
            raise ValueError("session directory is not owned by this runtime")
        yield fd
    finally:
        os.close(fd)


def _identity(fd: int) -> tuple[int, int]:
    st = os.fstat(fd)
    return st.st_dev, st.st_ino


@contextmanager
def _family_root(session_dir: Path, ancestors: tuple[str, ...] = ()) -> Iterator[int]:
    """Reconcile durable registry, creator markers and trusted runtime parent.

    Missing markers do not sever surviving registry ancestry. Disagreement is
    not permission to pick a convenient root: fail closed. Markerless operator
    roots remain valid when no trusted source knows a parent.
    """
    store = session_dir.parent
    registry = read_lineage(store.parent)
    current = session_dir.name
    runtime_parents = dict(zip((current, *ancestors), ancestors))
    seen: set[str] = set()
    with _directory(store) as store_fd:
        while True:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", current) or current in seen:
                raise ValueError("invalid or cyclic session ancestry")
            seen.add(current)
            fd = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=store_fd)
            try:
                if os.fstat(fd).st_uid != os.getuid():
                    raise ValueError("unowned session ancestry")
                try:
                    marker_fd = os.open(
                        ORIGIN_NAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                    )
                except FileNotFoundError:
                    marker = {}
                else:
                    with os.fdopen(marker_fd, encoding="utf-8") as fh:
                        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                            raise ValueError("invalid ancestry marker")
                        marker = json.load(fh)
                if not isinstance(marker, dict):
                    raise ValueError("invalid ancestry marker")
                row = registry.get(current, {})
                parents = [marker.get("parent", ""), row.get("parent", "")]
                parents.append(runtime_parents.get(current, ""))
                if any(not isinstance(parent, str) for parent in parents):
                    raise ValueError("invalid session parent")
                known = {parent for parent in parents if parent}
                if len(known) > 1:
                    raise ValueError("inconsistent session ancestry")
                if not known:
                    if marker.get("origin") == "subagent" or row:
                        raise ValueError("missing delegated session ancestry")
                    # Validate the held inode still occupies the same store slot.
                    # A name swap cannot redirect the following descriptor I/O.
                    check = os.stat(current, dir_fd=store_fd, follow_symlinks=False)
                    if (check.st_dev, check.st_ino) != _identity(fd):
                        raise ValueError("session ancestry changed")
                    yield fd
                    with _directory(store) as verify_store:
                        check = os.stat(current, dir_fd=verify_store, follow_symlinks=False)
                        if (
                            _identity(verify_store) != _identity(store_fd)
                            or (check.st_dev, check.st_ino) != _identity(fd)
                        ):
                            raise ValueError("session ancestry changed")
                    return
                current = known.pop()
            finally:
                os.close(fd)


def pin_quick_tunnel(
    session_dir: Path, host: str, origin: str, ancestors: tuple[str, ...] = ()
) -> PreviewPin:
    """First durable writer wins; no path-based reopen and no SQLite sidecars.

    The unreleased pin is a single private regular file locked with flock.
    O_NOFOLLOW/openat prevent symlink swaps from redirecting a write. A partial
    write after a crash is invalid JSON and fails closed, never a new identity.
    fsync precedes any RPC; failed approvals cannot reset the committed pin.
    """
    with _family_root(session_dir, ancestors) as root_fd:
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
        # Darwin openat(O_CREAT) can spuriously return ENOENT to the loser
        # of a concurrent create. Exclusive creation has an atomic EEXIST
        # contract; the losing caller opens the existing inode without create.
        try:
            fd = os.open(
                "browser-preview-origin.json",
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=root_fd,
            )
        except FileExistsError:
            fd = os.open("browser-preview-origin.json", flags, dir_fd=root_fd)
        with os.fdopen(fd, "r+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            st = os.fstat(fh.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_uid != os.getuid():
                raise ValueError("invalid preview pin file")
            contents = fh.read()
            if contents:
                row = json.loads(contents)
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("host"), str)
                    or not isinstance(row.get("origin"), str)
                    or quick_tunnel_origin(row["origin"]) != (row["host"], row["origin"])
                ):
                    raise ValueError("invalid preview pin")
            else:
                row = {"host": host, "origin": origin}
                fh.write(json.dumps(row))
                fh.flush()
                os.fsync(fh.fileno())
                os.fsync(root_fd)
            check = os.stat("browser-preview-origin.json", dir_fd=root_fd, follow_symlinks=False)
            if (check.st_dev, check.st_ino) != _identity(fh.fileno()):
                raise ValueError("preview pin changed during access")
            return PreviewPin(host=row["host"], origin=row["origin"], allowed=row["host"] == host)


def fixed_origin_instruction(origin: str, *, denied: bool = False) -> str:
    # i18n: ignore model-facing browser tool policy, not operator UI copy
    next_step = (
        "Access to the pinned origin was denied; do not re-request it. Use a "
        "fixed localhost/stable preview origin instead. "
        if denied
        else "Coordinate reuse of the original origin only if access is permitted, "
        "or use a fixed localhost/stable preview origin. "
    )
    return (
        "Fixed-origin rule: serve test builds on ONE fixed origin per task "
        "(localhost on one fixed port, or a stable preview URL), and reuse it "
        "across rounds and manager/child/sibling agents. This session family "
        f"is pinned to quick-tunnel host {origin}. Do not create or request "
        f"another trycloudflare.com host. {next_step}"
        "Existing site approvals still apply; this rule grants no access. "
        "Cancellation, approval failure and runtime restart do not reset the pin."
    )
