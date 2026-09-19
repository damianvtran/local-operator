"""The tunnel installer's platform arms, and the two refusals that were wrong.

Three defects, each of which turned into the WRONG MESSAGE for the user rather
than a crash they could act on:

* **A22** — a Linux without systemd (Devuan, Alpine, most containers, WSL2
  without systemd) raised ``FileNotFoundError`` from ``lop tunnel install``,
  which ``tunnels/cli.py`` catches as ``OSError`` and reports as *"check network
  access and your Radient login"*. The missing user service manager was named as
  a network problem.
* **C6** — ``_read_origin_auth`` refused a file whose POSIX mode bits said
  "group/other readable", and on Windows EVERY ordinary file says that
  (``0o666``), so the OpenCode harness on a Radient tunnel could not be
  configured at all — with a remedy (``chmod 600``) that does not exist there.
* **A1** — there was no Windows arm at all; ``service_path()`` raised
  ``ValueError`` on that platform.

The Windows arm is exercised through fakes. What is decidable here — the task
XML's content, the argv schtasks is handed, the refusal messages — is asserted;
that a real Task Scheduler accepts the registration is not, and the PR says so.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from local_operator import supervisors
from local_operator.tunnels import cli as tunnel_cli
from local_operator.tunnels import install as tunnel_install


def _completed(args: list[str], code: int = 0, stdout: str = "", stderr: str = "") -> object:
    return subprocess.CompletedProcess(args, code, stdout, stderr)


# ---------------------------------------------------------------------------
# A22 — a missing systemd is not a network problem
# ---------------------------------------------------------------------------


def test_install_without_systemd_names_the_supervisor_not_a_network_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE A22 REGRESSION TEST.

    ``subprocess.run(check=False)`` does not suppress ``FileNotFoundError``, so
    the missing binary escaped as an ``OSError`` — the type ``tunnels/cli.py``
    translates into "check network access and your Radient login".
    """
    monkeypatch.setattr(supervisors, "supervisor", lambda: "systemctl")
    monkeypatch.setattr(supervisors, "systemd_unit_path", lambda unit: tmp_path / unit)
    monkeypatch.setattr(supervisors.shutil, "which", lambda _name: None)

    with pytest.raises(ValueError) as caught:
        tunnel_install.install()

    message = str(caught.value)
    assert "systemctl" in message
    assert "network" not in message.lower()
    assert isinstance(caught.value, ValueError), "the CLI's OSError arm must not see this"


def test_a_supervisor_that_fails_reports_its_own_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(supervisors, "supervisor", lambda: "systemctl")
    monkeypatch.setattr(supervisors, "systemd_unit_path", lambda unit: tmp_path / unit)
    monkeypatch.setattr(supervisors.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        tunnel_install.subprocess,
        "run",
        lambda *a, **k: _completed(list(a[0]), 1, "", "Unit lop-tunnel.service not found."),
    )

    with pytest.raises(ValueError) as caught:
        tunnel_install.install()

    assert "Unit lop-tunnel.service not found." in str(caught.value)


def test_service_path_refuses_in_one_sentence_with_no_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisors, "supervisor", lambda: None)

    with pytest.raises(ValueError, match="no supported user service supervisor"):
        tunnel_install.service_path()


# ---------------------------------------------------------------------------
# The Linux unit
# ---------------------------------------------------------------------------


def test_the_unit_records_the_store_and_keeps_the_token_private(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``UMask=0077`` is load-bearing: it is what keeps cloudflared.token 0600."""
    store = tmp_path / "config-root"
    unit = tunnel_install.render_systemd(store)

    assert "[Unit]" in unit and "[Install]" in unit
    assert "ExecStart=" in unit and "local_operator.tunnels.service" in unit
    assert f"LOCAL_OPERATOR_CONFIG_DIR={store}" in unit
    assert "UMask=0077" in unit
    assert "Restart=on-failure" in unit
    assert "After=network-online.target" in unit


def test_install_writes_the_unit_and_enables_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(supervisors, "supervisor", lambda: "systemctl")
    monkeypatch.setattr(supervisors, "systemd_unit_path", lambda unit: tmp_path / unit)
    monkeypatch.setattr(supervisors.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        tunnel_install.subprocess,
        "run",
        lambda *a, **k: calls.append(list(a[0])) or _completed(list(a[0])),
    )

    tunnel_install.install()

    assert (tmp_path / tunnel_install.SYSTEMD_UNIT).exists()
    assert calls[0] == ["systemctl", "--user", "daemon-reload"]
    assert calls[1][:3] == ["systemctl", "--user", "enable"]


# ---------------------------------------------------------------------------
# C6 — the POSIX mode gate
# ---------------------------------------------------------------------------


def _auth_file(tmp_path: Path, mode: int) -> Path:
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"username": "u", "password": "p"}))
    path.chmod(mode)
    return path


def test_the_mode_gate_still_refuses_a_public_file_on_posix(tmp_path: Path) -> None:
    """Unchanged where it is meaningful: this is a real privacy control."""
    with pytest.raises(ValueError, match="chmod 600"):
        tunnel_cli._read_origin_auth(_auth_file(tmp_path, 0o666))


def test_a_private_file_is_accepted_on_posix(tmp_path: Path) -> None:
    assert tunnel_cli._read_origin_auth(_auth_file(tmp_path, 0o600)) == {
        "username": "u",
        "password": "p",
    }


def test_the_mode_gate_does_not_reject_every_file_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """THE C6 REGRESSION TEST.

    Windows reports ``st_mode == 0o100666`` for an ordinary file, so the old
    unconditional gate refused EVERY file — including one that is private by
    ACL — and told the user to run a command that does not exist there.
    """
    monkeypatch.setattr(tunnel_cli.os, "name", "nt")
    path = _auth_file(tmp_path, 0o666)

    assert tunnel_cli._read_origin_auth(path) == {"username": "u", "password": "p"}


def test_windows_still_refuses_a_malformed_auth_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Skipping the mode gate must not skip the guarantee that holds everywhere."""
    monkeypatch.setattr(tunnel_cli.os, "name", "nt")
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"username": "u", "password": "p", "extra": "x"}))

    with pytest.raises(ValueError, match="username and password only"):
        tunnel_cli._read_origin_auth(path)


# ---------------------------------------------------------------------------
# The Windows task arm
# ---------------------------------------------------------------------------


def test_install_registers_a_task_and_keeps_its_definition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created: list[tuple[str, str]] = []
    runs: list[tuple[str, ...]] = []
    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(tunnel_install.config, "directory", lambda base=None: tmp_path)
    monkeypatch.setattr(
        supervisors,
        "create_task",
        lambda name, xml: created.append((name, xml)) or (True, "registered"),
    )
    monkeypatch.setattr(
        supervisors,
        "schtasks",
        lambda *args, **kw: runs.append(args) or _completed(list(args), 0, "", ""),
    )

    tunnel_install.install()

    name, xml = created[0]
    assert name == tunnel_install.TASK_NAME
    assert "local_operator.tunnels.service" in xml
    assert "LOCAL_OPERATOR_CONFIG_DIR" in xml, "the store is part of the contract"
    assert ("/Run", "/TN", tunnel_install.TASK_NAME) in runs
    record = tmp_path / "service-task.xml"
    assert record.exists()
    assert record.read_text(encoding="utf-8") == xml


def test_a_refused_registration_is_reported_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(tunnel_install.config, "directory", lambda base=None: tmp_path)
    monkeypatch.setattr(
        supervisors, "create_task", lambda _n, _x: (False, "ERROR: Access is denied.")
    )

    with pytest.raises(ValueError) as caught:
        tunnel_install.install()

    assert "Access is denied." in str(caught.value)


def test_the_linux_unit_quotes_a_store_with_a_space(tmp_path: Path) -> None:
    """A5: the tunnel arm's quoting is now the shared rule, and still applied.

    systemd 255 reads an unquoted ``Environment=`` assignment up to the first
    space and ignores the rest ("Invalid environment assignment, ignoring:
    b/store"), so the connector would watch a different store than the one it
    was installed for.
    """
    base = tmp_path / "a b" / "store"

    text = tunnel_install.render_systemd(base)

    assert f'Environment="LOCAL_OPERATOR_CONFIG_DIR={base}"' in text
    assert 'ExecStart="' in text


def test_uninstall_deregisters_the_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``/End`` BEFORE ``/Delete``: the delete deregisters without stopping.

    ``schtasks /Delete`` does not interrupt the program the task runs, so an
    uninstall that only deletes left the connector serving after reporting it
    removed. The order is the assertion — a delete that happened first would
    leave nothing for the end to stop.
    """
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(tunnel_install.config, "directory", lambda base=None: tmp_path)
    monkeypatch.setattr(
        supervisors,
        "schtasks",
        lambda *args, **kw: calls.append(args) or _completed(list(args), 0, "", ""),
    )
    monkeypatch.setattr(
        supervisors,
        "delete_task",
        lambda name: calls.append(("/Delete", "/TN", name)) or (True, "deleted"),
    )
    (tmp_path / "service-task.xml").write_text("<Task/>", encoding="utf-8")

    tunnel_install.uninstall()

    assert [call[0] for call in calls] == ["/End", "/Delete"], calls
    assert calls[0] == tuple(supervisors.task_end_args(tunnel_install.TASK_NAME))
    assert not (tmp_path / "service-task.xml").exists()


def test_the_task_verbs_map_onto_schtasks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs: list[tuple[str, ...]] = []
    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(tunnel_install.config, "directory", lambda base=None: tmp_path)
    (tmp_path / "service-task.xml").write_text("<Task/>", encoding="utf-8")
    monkeypatch.setattr(
        supervisors,
        "schtasks",
        lambda *args, **kw: runs.append(args) or _completed(list(args), 0, "", ""),
    )

    tunnel_install.action("restart")

    assert runs[0][0] == "/End"
    assert runs[1][0] == "/Run"
