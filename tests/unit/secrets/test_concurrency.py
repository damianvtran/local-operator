"""Ten readers and a writer against one store, for real.

The operator runs ~10 lop sessions at once, so this is the store's actual
steady state rather than a stress scenario. The design measured 10 concurrent
readers doing 60 decrypts each against a live writer at 21 ms with zero errors;
these tests re-run that shape so a change to the pragmas — dropping WAL, or
removing ``busy_timeout`` — shows up as ``database is locked`` here rather than
as an intermittent failure in the operator's session a week later.

Both a THREAD version and a PROCESS version, because they catch different
things. Threads share one SQLite library instance and exercise the Python-level
connection handling; separate processes are the real deployment shape and are
the only way to exercise the cross-process file locking that WAL actually
depends on.
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from local_operator.secrets.crypto import generate_master_key
from local_operator.secrets.store import SecretStore

READERS = 10
READS_EACH = 60


def test_ten_threads_read_while_one_writes(store: SecretStore) -> None:
    """No errors, no corruption, every read returns a valid value."""
    store.set("SHARED", b"shared-value")
    store.set("CHURN", b"initial")

    errors: list[BaseException] = []

    def reader() -> int:
        count = 0
        try:
            for _ in range(READS_EACH):
                assert store.get("SHARED") == b"shared-value"
                count += 1
        except BaseException as exc:  # noqa: BLE001 - recorded and re-raised below
            errors.append(exc)
        return count

    def writer() -> int:
        count = 0
        try:
            for index in range(READS_EACH):
                store.update("CHURN", f"value-{index}".encode())
                count += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        return count

    with concurrent.futures.ThreadPoolExecutor(max_workers=READERS + 1) as pool:
        futures = [pool.submit(reader) for _ in range(READERS)]
        futures.append(pool.submit(writer))
        counts = [future.result() for future in futures]

    assert not errors, f"concurrent access raised: {errors[:3]}"
    assert counts == [READS_EACH] * (READERS + 1)
    # The store is still coherent afterwards, which is the corruption check
    # that matters: every record still authenticates.
    assert store.get("SHARED") == b"shared-value"
    assert store.get("CHURN") == f"value-{READS_EACH - 1}".encode()
    assert len(store.list()) == 2


_WORKER = textwrap.dedent("""
    import sys
    from pathlib import Path
    from local_operator.secrets.store import SecretStore

    base = Path(sys.argv[1])
    key = bytes.fromhex(sys.argv[2])
    role = sys.argv[3]
    reads = int(sys.argv[4])

    store = SecretStore(key, base=base)
    if role == "reader":
        for _ in range(reads):
            assert store.get("SHARED") == b"shared-value", "wrong value"
    else:
        for index in range(reads):
            store.update("CHURN", f"value-{index}".encode())
    print("OK")
    """)


def test_ten_processes_read_while_one_writes(config_root: Path, tmp_path: Path) -> None:
    """The real deployment shape: separate processes, real file locking.

    A thread test cannot fail the way ten `lop` sessions fail, because threads
    in one process share SQLite's connection cache and its internal mutexes.
    This spawns actual interpreters against one store file — the case WAL's
    cross-process locking exists for.
    """
    key = generate_master_key()
    store = SecretStore(key, base=config_root)
    store.set("SHARED", b"shared-value")
    store.set("CHURN", b"initial")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    arguments = [str(config_root), key.hex()]
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, *arguments, "reader", str(READS_EACH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for _ in range(READERS)
    ]
    processes.append(
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, *arguments, "writer", str(READS_EACH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
    )

    failures = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=180)
        if process.returncode != 0 or "OK" not in stdout:
            failures.append((process.returncode, stderr[-800:]))

    assert not failures, f"{len(failures)} of {len(processes)} workers failed: {failures[:2]}"
    assert store.get("SHARED") == b"shared-value"
    assert len(store.list()) == 2


def test_concurrent_writers_do_not_lose_records(config_root: Path) -> None:
    """Ten processes each storing a distinct secret; all ten must survive.

    Last-writer-wins on the same row would be one bug; losing whole rows to an
    unserialised transaction is the one this catches. ``BEGIN IMMEDIATE`` plus
    ``busy_timeout`` is what makes the insert-after-existence-check atomic.
    """
    key = generate_master_key()
    SecretStore(key, base=config_root).initialize()

    writer = textwrap.dedent("""
        import sys
        from pathlib import Path
        from local_operator.secrets.store import SecretStore
        store = SecretStore(bytes.fromhex(sys.argv[2]), base=Path(sys.argv[1]))
        store.set(f"SECRET_{sys.argv[3]}", f"value-{sys.argv[3]}".encode())
        print("OK")
        """)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", writer, str(config_root), key.hex(), str(index)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for index in range(READERS)
    ]
    failures = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=180)
        if process.returncode != 0:
            failures.append(stderr[-800:])

    assert not failures, f"writers failed: {failures[:2]}"
    store = SecretStore(key, base=config_root)
    names = {record.name for record in store.list()}
    assert names == {f"SECRET_{index}" for index in range(READERS)}
    for index in range(READERS):
        assert store.get(f"SECRET_{index}") == f"value-{index}".encode()
