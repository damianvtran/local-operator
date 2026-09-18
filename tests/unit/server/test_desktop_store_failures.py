"""Three store conditions, three answers: the ladder that used to say "busy".

Before this, every ``sqlite3.Error`` reaching the desktop control plane's shared
failure ladder -- lock contention, a full disk, an unopenable store, a corrupt
one -- was answered with the CONTENTION sentence ("Read state is busy right now.
It will catch up on its own."), raised ``from None`` with no log record. On
2026-09-17 the boot volume hit zero bytes free and an operator was told a read
state was momentarily busy about a condition no retry clears, over the one
action the client suggested ("Send it again") that could not help. Nothing was
logged, which is why attributing it took an hour.

These tests hold the two halves that failure needs: the CLASSIFICATION (which
condition is this) and the RECORD (what the server logged about it). The
classification is a pure function over ``sqlite_errorname`` -- not over SQLite's
prose, which is ambiguous exactly where it matters: a full volume and a
read-only directory both report ``SQLITE_CANTOPEN`` over the identical text
"unable to open database file", so the volume's own free space is the second
signal and each direction of that rule is asserted below.
"""

from __future__ import annotations

import errno
import logging
import shutil
import sqlite3
from types import SimpleNamespace
from typing import Any, cast

import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_profiles, desktop_sessions
from local_operator.server.routes.desktop_sessions import errors

# The classifier itself lives under ``session/`` so the TUI can reach it without
# importing ``local_operator.server`` (agent review round 1, R1). Patched through
# THAT module rather than the server shim: ``shutil`` is looked up in the module
# the implementation runs in, so a patch on the re-exporting shim would be a
# silent no-op.
from local_operator.session import store_failures
from local_operator.session.store_failures import (
    BUSY_MESSAGE,
    FULL_VOLUME_FLOOR_BYTES,
    OUT_OF_SPACE_MESSAGE,
    STORE_BUSY,
    STORE_OUT_OF_SPACE,
    STORE_UNAVAILABLE,
    StoreFailure,
    display_root,
    out_of_space_message,
    sqlite_store_failure,
    store_failure,
    unavailable_message,
    volume_is_full,
)

pytestmark = pytest.mark.asyncio


def ladder_request(root: Any, *, session_id: str | None = "0123456789ab") -> Any:
    """The slice of ``Request`` the ladder reads, without a socket under it."""
    return cast(
        Any,
        SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(config_manager=SimpleNamespace(config_dir=root))
            ),
            method="POST",
            url=SimpleNamespace(path="/v1/desktop/sessions/0123456789ab/messages"),
            path_params={"session_id": session_id} if session_id else {},
        ),
    )


def locked_database_error(tmp_path) -> sqlite3.Error:
    """A GENUINE ``SQLITE_BUSY``: a second writer against a held write lock."""
    path = tmp_path / "locked.db"
    holder = sqlite3.connect(path)
    holder.execute("CREATE TABLE t (a)")
    holder.commit()
    holder.execute("BEGIN IMMEDIATE")
    try:
        contender = sqlite3.connect(path, timeout=0)
        try:
            contender.execute("BEGIN IMMEDIATE")
        finally:
            contender.close()
    except sqlite3.Error as error:
        return error
    finally:
        holder.close()
    raise AssertionError("an immediate write lock did not produce contention")


def corrupt_database_error(tmp_path) -> sqlite3.Error:
    """A GENUINE ``SQLITE_NOTADB``, raised by a real file that is not a database."""
    path = tmp_path / "corrupt.db"
    path.write_text("this is not a database")
    try:
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE t (a)")
    except sqlite3.Error as error:
        return error
    raise AssertionError("a file that is not a database was opened as one")


def simulated(errorname: str, message: str = "simulated") -> sqlite3.Error:
    """An ``OperationalError`` carrying ``errorname``.

    Simulated rather than induced, for these two: ``SQLITE_FULL`` needs a full
    volume and ``SQLITE_CANTOPEN`` a store that cannot be created, and neither is
    reachable inside a test suite that must run on CI. The genuine article for
    both is the bounded-disk-image repro (``scripts/repro-enospc-send.py``), whose
    capture is on the PR; what is simulated here is only the exception's identity,
    which is exactly what the classifier reads.
    """
    error = sqlite3.OperationalError(message)
    error.sqlite_errorname = errorname
    return error


def failing_volume(monkeypatch, free: int) -> None:
    monkeypatch.setattr(
        store_failures.shutil, "disk_usage", lambda _root: SimpleNamespace(free=free)
    )


# --------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("errorname", "free", "code", "status"),
    [
        # Contention is the only retryable one, and the only one that keeps the
        # sentence the app already relays.
        ("SQLITE_BUSY", 10**12, STORE_BUSY, 503),
        ("SQLITE_LOCKED", 10**12, STORE_BUSY, 503),
        # SQLite says "out of space" outright.
        ("SQLITE_FULL", 10**12, STORE_OUT_OF_SPACE, 507),
        # The family the incident actually hit: the volume decides.
        ("SQLITE_CANTOPEN", 1024, STORE_OUT_OF_SPACE, 507),
        ("SQLITE_IOERR_WRITE", 1024, STORE_OUT_OF_SPACE, 507),
        ("SQLITE_CANTOPEN", 10**12, STORE_UNAVAILABLE, 500),
        ("SQLITE_IOERR_FSYNC", 10**12, STORE_UNAVAILABLE, 500),
        ("SQLITE_READONLY_DIRECTORY", 10**12, STORE_UNAVAILABLE, 500),
        ("SQLITE_PERM", 10**12, STORE_UNAVAILABLE, 500),
        # The store is there and unreadable for its own reasons.
        ("SQLITE_NOTADB", 10**12, STORE_UNAVAILABLE, 500),
        ("SQLITE_CORRUPT", 10**12, STORE_UNAVAILABLE, 500),
        # No errorname at all: never answered as the transient one.
        ("", 10**12, STORE_UNAVAILABLE, 500),
    ],
)
async def test_classifier_maps_each_errorname_to_its_class(
    errorname, free, code, status, tmp_path, monkeypatch
):
    failing_volume(monkeypatch, free)
    failure = sqlite_store_failure(simulated(errorname), tmp_path)
    assert (failure.code, failure.status) == (code, status)


async def test_a_full_volume_is_decided_by_the_volumes_own_free_space(tmp_path, monkeypatch):
    """``CANTOPEN`` is ambiguous, so the volume answers for it -- both ways.

    One exception object, two classifications, differing only in the free space
    the probe reports: that is what proves the second signal is load-bearing
    rather than decorative. A rule that ignored the volume would call a
    permissions problem out-of-space, and one that ignored ``CANTOPEN`` would
    have kept calling the operator's incident "check the machine".
    """
    error = simulated("SQLITE_CANTOPEN", "unable to open database file")

    failing_volume(monkeypatch, FULL_VOLUME_FLOOR_BYTES - 1)
    assert sqlite_store_failure(error, tmp_path).code == STORE_OUT_OF_SPACE

    failing_volume(monkeypatch, FULL_VOLUME_FLOOR_BYTES)
    assert sqlite_store_failure(error, tmp_path).code == STORE_UNAVAILABLE


async def test_a_real_busy_database_and_a_real_corrupt_one_classify_apart(tmp_path, monkeypatch):
    """The two classes a test CAN produce for real, through the real errornames.

    ``locked_database_error`` and ``corrupt_database_error`` are raised by SQLite
    itself, so this pins the classifier against the codes CPython actually
    attaches rather than against this suite's idea of them.
    """
    monkeypatch.setattr(
        store_failures.shutil, "disk_usage", lambda _root: SimpleNamespace(free=10**12)
    )
    busy = locked_database_error(tmp_path)
    assert busy.sqlite_errorname == "SQLITE_BUSY"
    failure = sqlite_store_failure(busy, tmp_path)
    assert (failure.code, failure.status, failure.message) == (
        STORE_BUSY,
        503,
        BUSY_MESSAGE,
    )

    corrupt = corrupt_database_error(tmp_path)
    assert corrupt.sqlite_errorname == "SQLITE_NOTADB"
    unavailable = sqlite_store_failure(corrupt, tmp_path)
    assert (unavailable.code, unavailable.status) == (STORE_UNAVAILABLE, 500)


async def test_a_bare_sqlite_error_is_never_reported_as_transient():
    """The acceptance case: no errorname must not mean "busy, it will heal".

    An unclassifiable store error defaulting to the RETRYABLE sentence is the
    original bug wearing a smaller hat -- the client would keep its retry hint
    and the user would keep pressing send.
    """
    bare = sqlite3.Error("something odd happened")
    failure = sqlite_store_failure(bare, None)
    assert (failure.code, failure.status) == (STORE_UNAVAILABLE, 500)
    assert failure.message != BUSY_MESSAGE
    assert "Retrying will not help" in failure.message


async def test_only_space_oserrors_are_ours():
    """A non-space ``OSError`` must be left to the route that owns it.

    The ladder sits under every desktop control-plane route, so claiming
    arbitrary ``OSError``s would swallow the failures whose own routes have
    better words for them -- ``move_session`` answers an unusable target path
    with a 409 naming it.
    """
    space = store_failure(OSError(errno.ENOSPC, "No space left on device"), None)
    assert space is not None and space.code == STORE_OUT_OF_SPACE
    quota = store_failure(OSError(errno.EDQUOT, "Disc quota exceeded"), None)
    assert quota is not None and quota.code == STORE_OUT_OF_SPACE
    assert store_failure(FileNotFoundError(2, "No such file or directory"), None) is None
    assert store_failure(PermissionError(13, "Permission denied"), None) is None
    assert store_failure(ValueError("not a store error"), None) is None


async def test_an_unmeasurable_volume_does_not_claim_the_disk_is_full(monkeypatch, tmp_path):
    """ "The disk is full" is a claim that has to be earned."""

    def broken(_root):
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr(store_failures.shutil, "disk_usage", broken)
    assert volume_is_full(tmp_path) is False


# --------------------------------------------------------------------------
# The ladder: status, body and the log record
# --------------------------------------------------------------------------


async def test_the_ladder_logs_the_real_exception_with_route_and_session(
    tmp_path, monkeypatch, caplog
):
    """The record this ladder owes: which store failed, where, and why.

    Its absence is the reason the incident took an hour of log archaeology: the
    same condition had been recorded elsewhere in the runtime log three times
    while the request that refused the user left nothing behind.

    Driven with a GAINED store error (a real ``SQLITE_NOTADB`` from a real file
    that is not a database), the class whose traceback is the finding. The
    routine contention class is the opposite case and is pinned separately
    below (review round 1, R5).
    """
    monkeypatch.setattr(
        store_failures.shutil, "disk_usage", lambda _root: SimpleNamespace(free=10**12)
    )
    with caplog.at_level(logging.ERROR):
        with pytest.raises(HTTPException) as raised:
            async with errors(ladder_request(tmp_path)):
                raise corrupt_database_error(tmp_path)
    failure = raised.value
    assert failure.status_code == 500
    assert cast("dict[str, Any]", failure.detail)["code"] == STORE_UNAVAILABLE

    (record,) = [entry for entry in caplog.records if "desktop store failure" in entry.message]
    assert record.levelno == logging.ERROR
    assert "store_unavailable" in record.getMessage()
    assert "/v1/desktop/sessions/0123456789ab/messages" in record.getMessage()
    assert "0123456789ab" in record.getMessage()

    # The REAL exception, still live: the client's copy may never carry it,
    # because a store error names file paths. Asserted on the RENDERED line --
    # what an operator greps in the log -- rather than on the record object's
    # ``exc_info``, because that is the artefact the claim is about and it does
    # not depend on how the capture plumbing happens to carry the record: one
    # run of the full suite handed back a record whose ``exc_info`` was None
    # while the very same record still rendered its traceback into the report.
    rendered = caplog.text
    assert "Traceback (most recent call last)" in rendered
    assert "sqlite3.DatabaseError" in rendered
    assert "file is not a database" in rendered
    # And it names the route, not just the error: the archaeology this exists to
    # prevent was "which request logged this?" as much as "what failed?".
    assert "desktop_sessions.py" in rendered


async def test_routine_contention_is_logged_without_a_traceback(tmp_path, caplog):
    """Contention is routine, so its record carries no stack (review round 1, R5).

    A lock that clears on its own is not a finding, and a full traceback per
    retry is log noise that buries the records worth reading. The line is still
    emitted -- code, route and session -- so a contention that does NOT clear
    stays attributable; what it drops is only the stack.
    """
    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException) as raised:
            async with errors(ladder_request(tmp_path)):
                raise locked_database_error(tmp_path)
    assert raised.value.status_code == 503
    (record,) = [entry for entry in caplog.records if "desktop store failure" in entry.message]
    assert record.levelno == logging.WARNING
    assert "store_busy" in record.getMessage()
    assert "/v1/desktop/sessions/0123456789ab/messages" in record.getMessage()
    assert "Traceback (most recent call last)" not in caplog.text


def test_every_ladder_call_site_passes_the_request() -> None:
    """The ladder takes a REQUIRED request: prove no call site omits it.

    This is the guard for the exact failure a rebase can smuggle in (QA round 1,
    Q1): upstream added call sites while this branch made the parameter
    required, the textual conflict resolved to code that still compiles, and a
    call site that omits the argument raises ``TypeError`` at REQUEST time --
    invisible to a green unit run and to a reviewer reading the diff, visible
    only when somebody opens that route.

    A walk rather than a grep, so a call broken across lines or written with
    keyword arguments is inspected the same way, and it asserts it FOUND call
    sites at all: an empty walk would otherwise pass vacuously if the helper were
    renamed.
    """
    import ast
    from pathlib import Path

    routes = Path(__file__).resolve().parents[3] / "local_operator" / "server" / "routes"
    calls: list[tuple[str, int, int, bool]] = []
    for path in sorted(routes.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "errors"
            ):
                first = node.args[0] if node.args else None
                calls.append(
                    (
                        path.name,
                        node.lineno,
                        len(node.args),
                        isinstance(first, ast.Name) and first.id == "request",
                    )
                )
    # 53 at the time of writing, across six modules: a floor rather than an exact
    # count so an unrelated route does not fail this test, but high enough that a
    # walk which silently stopped finding them cannot pass.
    assert len(calls) >= 50, calls
    # The REQUEST is what must be named first, so that is what the walk pins. The
    # ladder takes an optional second argument now — a route's own sentence
    # composer, which ``POST /v1/desktop/attention/seen`` passes because a receipt
    # clear is not a message send (QA round 2, Q1) — and an exact arity would turn
    # every future composer into a failure of this test rather than of its own.
    assert [call for call in calls if not call[3]] == [], calls
    assert [call for call in calls if call[2] not in (1, 2)] == [], calls


async def test_a_full_volume_answers_out_of_space_and_says_what_to_do(
    tmp_path, monkeypatch, caplog
):
    failing_volume(monkeypatch, 1024)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(HTTPException) as raised:
            async with errors(ladder_request(tmp_path)):
                raise simulated("SQLITE_CANTOPEN", "unable to open database file")
    detail = cast("dict[str, Any]", raised.value.detail)
    assert raised.value.status_code == 507
    assert detail["code"] == STORE_OUT_OF_SPACE
    assert detail["message"] == out_of_space_message(tmp_path)
    assert "Free some space" in detail["message"]
    # WHERE to free it (design round 1, D1): "this computer is out of disk space"
    # is true and unreachable on a machine with more than one volume, and the
    # renderer paints this sentence verbatim rather than keeping its own copy.
    assert "volume holding" in detail["message"]
    (record,) = [entry for entry in caplog.records if "desktop store failure" in entry.message]
    assert record.levelno == logging.ERROR


async def test_the_sentences_name_the_config_root_and_shorten_it_to_home(tmp_path, monkeypatch):
    """The copy is actionable: it names the directory the operator can go to.

    ``~``-relative when the root is under the home directory (the shipped case),
    absolute when it is not (a relocated root, an isolated run), and the same
    for both sentences that ask the operator to go and look.
    """
    assert out_of_space_message(tmp_path) == OUT_OF_SPACE_MESSAGE.format(root=str(tmp_path))
    assert str(tmp_path) in unavailable_message(tmp_path)

    monkeypatch.setenv("HOME", str(tmp_path))
    relocated = tmp_path / ".local-operator"
    assert display_root(relocated) == "~/.local-operator"
    assert "the volume holding ~/.local-operator" in out_of_space_message(relocated)
    assert display_root(tmp_path) == "~"


async def test_the_ladder_re_raises_what_it_cannot_classify(tmp_path):
    """An unmounted-volume ``OSError`` must keep reaching its own route."""
    with pytest.raises(FileNotFoundError):
        async with errors(ladder_request(tmp_path)):
            raise FileNotFoundError(2, "No such file or directory")


async def test_a_space_oserror_from_a_non_sqlite_write_is_the_same_condition(tmp_path, caplog):
    """The transcript append and the attachment store raise ENOSPC, not sqlite.

    A message that could not be persisted is the same condition to the user as a
    store that could not be written, so it must not arrive as a bare 500 either.
    """
    with caplog.at_level(logging.ERROR):
        with pytest.raises(HTTPException) as raised:
            async with errors(ladder_request(tmp_path)):
                raise OSError(errno.ENOSPC, "No space left on device")
    assert raised.value.status_code == 507
    assert cast("dict[str, Any]", raised.value.detail)["code"] == STORE_OUT_OF_SPACE


# --------------------------------------------------------------------------
# The real route boundary
# --------------------------------------------------------------------------


@pytest_asyncio.fixture
async def api(tmp_path, monkeypatch):
    """The real routers over a real config root, with the real receipt store."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "store-failure-test")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.include_router(desktop_sessions.router)
    app.include_router(desktop_profiles.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer store-failure-test"},
    ) as client:
        yield client, tmp_path
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


async def test_a_corrupt_receipt_store_refuses_over_http_on_another_modules_route(api, caplog):
    """``store_unavailable`` at a REAL store boundary, on a route that is not
    ``desktop_sessions``' own.

    The ladder is imported by ``desktop_profiles`` (and by the catalogue,
    lifecycle, radient and wake modules), so its promise is about every desktop
    control-plane route. This drives one of THEIRS: the receipt store is a real
    file, made genuinely unreadable by writing prose into it, and no collaborator
    is mocked -- the DatabaseError is SQLite's.
    """
    client, root = api
    (root / "desktop-receipts.db").write_text("this is not a database")

    with caplog.at_level(logging.ERROR):
        response = await client.post(
            "/v1/desktop/profiles/install",
            json={"request_id": "11111111-1111-4111-8111-111111111111", "name": "coder"},
        )
    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert detail["code"] == STORE_UNAVAILABLE
    assert detail["message"] != BUSY_MESSAGE
    # SQLite's own text never reaches the client -- the rule this ladder keeps
    # for store errors. The CONFIG ROOT does now, deliberately: that is a path
    # this process chose as the place to look, not one SQLite's message carried
    # (see the constants' comments in ``session/store_failures.py``).
    assert "not a database" not in detail["message"]
    assert "unable to open database file" not in detail["message"]
    assert any(
        "desktop store failure store_unavailable" in record.getMessage()
        for record in caplog.records
    )


async def test_an_unopenable_receipt_store_on_a_full_volume_answers_507(api, monkeypatch, caplog):
    """The incident's own shape, at the real boundary: store cannot be created,
    volume full.

    ``mkdir`` at the receipt store's path is what a create cannot do here -- a
    directory is a path SQLite refuses to open, reported as the same
    ``SQLITE_CANTOPEN`` a full volume produces. The volume is what the probe
    reports; on the day of the incident that was true, and the operator was told
    the store was busy.
    """
    client, root = api
    (root / "desktop-receipts.db").mkdir()
    failing_volume(monkeypatch, 0)

    with caplog.at_level(logging.ERROR):
        response = await client.post(
            "/v1/desktop/sessions",
            json={
                "request_id": "22222222-2222-4222-8222-222222222222",
                "cwd": str(root),
            },
        )
    assert response.status_code == 507, response.text
    detail = response.json()["detail"]
    assert detail["code"] == STORE_OUT_OF_SPACE
    assert "Free some space on the volume holding" in detail["message"]
    assert any(
        "desktop store failure store_out_of_space" in record.getMessage()
        for record in caplog.records
    )


async def test_the_same_unopenable_store_with_space_answers_unavailable(api, monkeypatch):
    """Same database, same errorname, a volume with room: NOT a disk problem."""
    client, root = api
    (root / "desktop-receipts.db").mkdir()
    failing_volume(monkeypatch, 10**12)

    response = await client.post(
        "/v1/desktop/sessions",
        json={"request_id": "33333333-3333-4333-8333-333333333333", "cwd": str(root)},
    )
    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert detail["code"] == STORE_UNAVAILABLE
    assert detail["message"] != BUSY_MESSAGE


async def test_the_volume_probe_reads_and_never_writes(tmp_path, monkeypatch):
    """Detection must not perform the operation that is failing.

    ``browser_bridge/state.py`` records the incident behind this rule: a full
    disk turned a read-only path computation into a second ``OSError`` raised
    from inside the handler for the first one, and the daemon failed to boot in
    precisely the disk-full scenario it exists to survive. The stand-in below is
    installed on the CLASSIFIER's own ``shutil`` binding -- so it constrains what
    this module can reach for, without becoming a global that would break
    ``tmp_path`` (an ``os.open`` denial does exactly that: ``tempfile`` uses one
    to create a directory).

    ``disk_usage`` is forwarded to the real one, so the assertion is that the
    probe ran AND that it reached for nothing write-shaped while it did.
    """
    consulted: list[str] = []

    class _ReadOnlyShutil:
        def __getattr__(self, name: str) -> Any:
            if name in {
                "mkstemp",
                "mkdtemp",
                "mkdir",
                "chmod",
                "copy",
                "copyfile",
                "move",
                "rmtree",
            }:
                raise AssertionError(f"the full-volume probe called shutil.{name}")
            return getattr(shutil, name)

    class _RecordingShutil(_ReadOnlyShutil):
        def disk_usage(self, path: Any) -> Any:
            consulted.append(str(path))
            return shutil.disk_usage(path)

    monkeypatch.setattr(store_failures, "shutil", _RecordingShutil())
    assert volume_is_full(tmp_path) in (True, False)
    assert consulted == [str(tmp_path)]


async def test_the_default_refusal_keeps_the_classifiers_send_path_sentence(tmp_path):
    """The per-route composer is OPT-IN: every other route still says "message".

    QA round 2 (Q1) found the receipts route painting the classifier's send-path
    copy about a receipt clear, and the fix gives that one route its own composer.
    This pins the other half of the boundary — a ladder arm with no composer still
    answers with the classifier's sentence — so a later tidy-up cannot move the
    send path's copy without a test saying it did.
    """
    with pytest.raises(HTTPException) as raised:
        async with errors(ladder_request(tmp_path)):
            raise simulated("SQLITE_FULL")
    detail = cast("dict[str, Any]", raised.value.detail)
    assert detail["code"] == STORE_OUT_OF_SPACE
    assert detail["message"] == out_of_space_message(tmp_path)
    assert "the message could not be written" in detail["message"]


async def test_a_composer_passed_to_the_ladder_replaces_only_the_sentence(tmp_path):
    """What a route supplies is the SENTENCE, never the status, code or level.

    The receipts route composes its own copy; the contract it must not touch is the
    rest of the refusal, because a client keys on the code (the renderer withholds
    its retry hint by matching ``store_unavailable``) and the log record is the
    operator's evidence.
    """
    seen: list[tuple[str, Any]] = []

    def composed(failure: StoreFailure, root: Any) -> str:
        seen.append((failure.code, root))
        return "SENTINEL-COMPOSED"

    with pytest.raises(HTTPException) as raised:
        async with errors(ladder_request(tmp_path), composed):
            raise simulated("SQLITE_FULL")
    detail = cast("dict[str, Any]", raised.value.detail)
    assert raised.value.status_code == 507, "the status is the classifier's"
    assert detail["code"] == STORE_OUT_OF_SPACE, "the code is the classifier's"
    assert detail["message"] == "SENTINEL-COMPOSED"
    assert seen == [(STORE_OUT_OF_SPACE, tmp_path)], "the composer gets the store root"
