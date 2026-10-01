"""``lop operator`` anchor transfer + setup: the verbs the onboarding runner drives.

These cells pin the PUBLIC half of operator authority — the statement a peer
receives — and the local bootstrap's receipt vocabulary (design §3.3 step 7,
§3.7). No keychain is touched: the anchor is a synthetic statement and every
store read is monkeypatched, so the cells run on any platform including a CI
container. What they assert is the BYTES discipline (canonical or refused) and
the receipts, not the local ladder (``test_operator_authority`` owns that).
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.operator import OperatorAnchor, handlers
from local_operator.operator.trust import anchor_bytes
from local_operator.operator.verify import key_id_for, spki_fp


def _anchor() -> OperatorAnchor:
    spki = b"\x04" + bytes(range(64))
    return OperatorAnchor(
        key_id=key_id_for(spki),
        spki=spki,
        backend="file-only",
        presence=False,
        label="",
        created_at=0,
    )


def test_anchor_export_writes_the_canonical_statement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--file`` writes exactly ``anchor_bytes`` and prints the frozen trio."""
    anchor = _anchor()
    monkeypatch.setattr(
        "local_operator.operator.handlers.load_staged_anchor", lambda *a, **k: anchor
    )
    monkeypatch.setattr("local_operator.operator.handlers.config_dir", lambda: tmp_path)
    target = tmp_path / "operator-anchor.json"

    code = handlers._anchor(Namespace(anchor_command="export", file=str(target), json=True))

    assert code == 0
    assert target.read_bytes() == anchor_bytes(
        anchor
    ), "the exported bytes must be the ones install lands"
    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == str(target)
    assert payload["key_id"] == anchor.key_id
    assert payload["spki_fp"] == spki_fp(anchor.spki)
    assert "statement" not in payload


def test_anchor_export_without_a_file_puts_the_statement_on_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No file means stdout IS the transfer form; ``--json`` carries it too."""
    anchor = _anchor()
    monkeypatch.setattr(
        "local_operator.operator.handlers.load_staged_anchor", lambda *a, **k: anchor
    )
    monkeypatch.setattr("local_operator.operator.handlers.config_dir", lambda: tmp_path)

    assert handlers._anchor(Namespace(anchor_command="export", file="", json=False)) == 0
    assert capsys.readouterr().out.encode("utf-8") == anchor_bytes(anchor)

    assert handlers._anchor(Namespace(anchor_command="export", file="", json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["statement"].encode("utf-8") == anchor_bytes(anchor)
    assert payload["path"] == ""


def test_anchor_export_says_the_product_action_when_nothing_is_set_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No key: a refusal naming setup, never a terminal command (§2.9)."""
    monkeypatch.setattr("local_operator.operator.handlers.load_staged_anchor", lambda *a, **k: None)
    monkeypatch.setattr(
        "local_operator.operator.handlers.load_anchor",
        lambda *a, **k: SimpleNamespace(anchor=None, usable=False, path="/etc/x"),
    )
    monkeypatch.setattr("local_operator.operator.handlers.config_dir", lambda: tmp_path)

    assert handlers._anchor(Namespace(anchor_command="export", file="", json=False)) == 1
    message = capsys.readouterr().err
    # D2 (design round 1): #1877's settled wording — the interim agent action,
    # because no surface raises the setup card at this head.
    assert "ask Local Operator to set it up for you" in message
    assert "`lop " not in message, "no refusal may name a terminal command"


def test_install_from_refuses_non_canonical_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE BYTES RULE: a re-serialised statement lands nothing (F4b)."""
    anchor = _anchor()
    pretty = anchor_bytes(anchor)
    reformatted = json.dumps(json.loads(pretty.decode("utf-8")), separators=(",", ":")).encode(
        "utf-8"
    )
    source = tmp_path / "statement.json"
    source.write_bytes(reformatted)

    code = handlers.install_anchor_from(source, tmp_path / "config", print_only=True)

    assert code == 1
    assert "fresh export" in capsys.readouterr().err


def test_install_from_accepts_canonical_bytes_and_prints_the_privileged_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    anchor = _anchor()
    source = tmp_path / "statement.json"
    source.write_bytes(anchor_bytes(anchor))
    root = tmp_path / "config"

    code = handlers.install_anchor_from(source, root, print_only=True)

    assert code == 0
    printed = capsys.readouterr().out
    assert "sudo install" in printed
    assert str(root) in printed, "the staged path under THIS root must be what the command lands"


def test_install_from_refuses_garbage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "statement.json"
    source.write_bytes(b"not json at all")

    assert handlers.install_anchor_from(source, tmp_path / "config", print_only=True) == 1
    assert "not one this build recognises" in capsys.readouterr().err


def _setup_fakes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, install_rc: int = 0
) -> OperatorAnchor:
    """Everything ``_setup`` touches, stubbed: no keychain, no sudo, no disk."""
    anchor = _anchor()

    def _first_load(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            anchor=None, usable=False, path="/etc/x", reason="no anchor installed"
        )

    def _final_load(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(anchor=anchor, usable=True, path="/etc/x", reason="")

    state = {"reads": 0}

    def _load(*args: Any, **kwargs: Any) -> Any:
        state["reads"] += 1
        return _final_load() if state["reads"] > 1 else _first_load()

    monkeypatch.setattr(handlers, "load_anchor", _load)
    monkeypatch.setattr(handlers, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(handlers, "_existing_key", lambda root, preference: None)
    monkeypatch.setattr(
        handlers,
        "create_key",
        lambda *, config_root, preference="auto": SimpleNamespace(
            key_id=anchor.key_id,
            spki=anchor.spki,
            backend="file-only",
            presence=False,
            rung=None,
            reused=False,
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(handlers, "anchor_for_handle", lambda handle, *, label="": anchor)
    monkeypatch.setattr(handlers, "stage_anchor", lambda root, anchor: tmp_path / "staged.json")
    monkeypatch.setattr(handlers, "install_anchor", lambda *a, **k: install_rc)
    monkeypatch.setattr(handlers, "anchor_path", lambda: tmp_path / "anchor-target.json")
    monkeypatch.setattr(
        handlers,
        "operator_authority_report",
        lambda: {"level": "operator-file-only"},
    )
    return anchor


def test_setup_emits_the_frozen_receipt_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _setup_fakes(monkeypatch, tmp_path)

    code = handlers._setup(Namespace(json=True, sudo_secret=""))

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["step"] for row in payload["receipts"]] == [
        "proposed",
        "consent",
        "generated",
        "installed",
        "verified",
    ]
    assert all(row["ok"] for row in payload["receipts"])
    assert payload["state"] == "installed"
    assert payload["level"] == "operator-file-only"


def test_setup_records_a_failed_admin_gesture_as_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A declined or failed gesture is a receipt, not a dead end (§3.7)."""
    _setup_fakes(monkeypatch, tmp_path, install_rc=1)

    code = handlers._setup(Namespace(json=True, sudo_secret=""))

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    installed = [row for row in payload["receipts"] if row["step"] == "installed"]
    assert installed and installed[0]["ok"] is False
    assert payload["state"] == "not_installed"
    assert "not installed yet" in installed[0]["detail"]


def test_setup_without_an_admin_tool_never_prints_the_by_hand_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D1 (design round 1): no admin tool is a receipt, not a sudo dump.

    The reproduction: `_setup` on a machine where `sudo` does not exist printed
    "Run this by hand:" plus two commands that cannot be followed there, above the
    receipts, and marked the consent row `[ok]` while its own sentence said nothing
    would be installed. Set up must instead report consent as blocked and the
    install as not done, in the reader's words, and never reach the direct verb's
    by-hand block.
    """
    _setup_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(handlers.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        handlers,
        "install_anchor",
        lambda *a, **k: pytest.fail("must not run when no admin tool exists"),
    )

    code = handlers._setup(Namespace(json=False, sudo_secret=""))

    assert code == 1
    captured = capsys.readouterr()
    assert "by hand" not in captured.out and "by hand" not in captured.err
    assert "sudo" not in captured.out
    assert "[blocked] consent:" in captured.out
    assert "no administrator tool" in captured.out
    assert "[blocked] installed:" in captured.out
    assert "not trusted yet" in captured.out


def test_setup_is_idempotent_on_an_installed_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Already installed: nothing is replaced, and the receipts say so."""
    anchor = _anchor()
    monkeypatch.setattr(handlers, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(
        handlers,
        "load_anchor",
        lambda *a, **k: SimpleNamespace(anchor=anchor, usable=True, path="/etc/x", reason=""),
    )
    monkeypatch.setattr(
        handlers,
        "_existing_key",
        lambda root, preference: SimpleNamespace(
            key_id=anchor.key_id, presence=False, backend="file-only", rung=None, close=lambda: None
        ),
    )
    monkeypatch.setattr(
        handlers, "install_anchor", lambda *a, **k: pytest.fail("must not reinstall")
    )

    code = handlers._setup(Namespace(json=True, sudo_secret=""))

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "installed"
    assert "nothing replaced" in payload["receipts"][2]["detail"]
