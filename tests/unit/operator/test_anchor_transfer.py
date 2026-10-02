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

from local_operator.operator import OperatorAnchor, handlers, keychain
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


#: A synthetic but well-formed P-256 uncompressed point, the same shape ``_anchor`` builds.
_A_POINT = b"\x04" + bytes(range(64))


class _StubPresenceStore:
    """A presence store that answers the REAL ``load() -> Signer | None`` contract.

    The setup fakes used to hand ``_setup`` a ``SimpleNamespace`` carrying BOTH ``key_id``
    and ``close`` — a chimera that satisfied each half of the two disagreeing halves of
    ``_setup`` and so let the resume crash ship green. This answers what the real backends
    answer: ``create`` counts and returns the handle, ``load`` returns a ``Signer`` (with
    the handle at ``signer.handle``) once a key exists, and ``None`` before that.

    ``present=True`` models the resume state — the key is on the machine and no anchor is
    needed to have put it there — so ``_existing_key`` is exercised for real rather than
    stubbed over.
    """

    def __init__(self, *, present: bool = False) -> None:
        self.handle = keychain.KeyHandle(
            backend=keychain.SECURE_ENCLAVE,
            key_id=key_id_for(_A_POINT),
            spki=_A_POINT,
            presence=True,
        )
        self.created = 1 if present else 0

    def create(self) -> Any:
        self.created += 1
        return self.handle

    def load(self) -> Any:
        return keychain.Signer(self.handle) if self.created else None

    def supported(self) -> bool:
        return True


def _stub_store(monkeypatch: pytest.MonkeyPatch, stub: _StubPresenceStore) -> None:
    """Wire the stub into BOTH callers of ``choose_backend``, as the authority tests do.

    ``handlers._existing_key`` imports it per call; ``sign.create_key`` bound it at import
    time. Patching one and not the other would silently exercise the real ladder on a
    machine that has a Secure Enclave.
    """
    from local_operator.operator import sign

    monkeypatch.setattr(keychain, "choose_backend", lambda preference, *, config_root: stub)
    monkeypatch.setattr(sign, "choose_backend", lambda preference, *, config_root: stub)


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
    # NO ``_existing_key`` STUB (agent review follow-up / P1 resume fix): the whole
    # point of these setup cells is the boundary between the probe and the verb, and a
    # stub there is how the resume path's type disagreement stayed hidden. The stub
    # store answers ``load() -> None`` while nothing has been created, so the real
    # ``_existing_key`` runs and reports the fresh-key state.
    _stub_store(monkeypatch, _StubPresenceStore())
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
    """Already installed, with a REAL signer behind the probe: nothing is replaced.

    The cell this replaces stubbed ``_existing_key`` with a ``SimpleNamespace`` carrying
    both ``key_id`` and ``close``, so it bypassed the boundary entirely and could never
    see the signature/consent short-circuit dereference a signer. It now drives the real
    ``_existing_key`` through the stub store, where the key is present and the anchor is
    installed.
    """
    anchor = _anchor()
    store = _StubPresenceStore(present=True)
    _stub_store(monkeypatch, store)
    monkeypatch.setattr(handlers, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(
        handlers,
        "load_anchor",
        lambda *a, **k: SimpleNamespace(anchor=anchor, usable=True, path="/etc/x", reason=""),
    )
    monkeypatch.setattr(
        handlers, "install_anchor", lambda *a, **k: pytest.fail("must not reinstall")
    )
    monkeypatch.setattr(handlers, "create_key", lambda **k: pytest.fail("must not replace the key"))

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
    assert "nothing replaced" in payload["receipts"][2]["detail"]
    assert store.handle.key_id in payload["receipts"][2]["detail"]
    assert store.created == 1, "the short-circuit must not create another key"


def test_setup_resumes_when_the_key_exists_but_the_anchor_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE OPERATOR-BLOCKING RESUME CELL: the key is here, the anchor is not.

    This is the state an interrupted setup leaves (and this machine's state): the probe at
    the top of ``_setup`` finds a key, ``load_anchor`` reports nothing usable, so the verb
    must stage the anchor for the EXISTING key and carry on through the privileged step.
    Before the fix it died here with ``AttributeError: '_KeyagentSigner' object has no
    attribute 'key_id'`` — the probe returned a signer and the verb dereferenced it as a
    handle.

    ``_existing_key`` is NOT stubbed: the bug lived in the real probe/verb interaction, so
    the stub store provides the signer shape and the real boundary unwraps it.
    """
    store = _StubPresenceStore(present=True)
    _stub_store(monkeypatch, store)
    monkeypatch.setattr(handlers, "config_dir", lambda: tmp_path)

    anchor = _anchor()
    reads = {"n": 0}

    def _load(*args: Any, **kwargs: Any) -> Any:
        # Only the load AFTER the install reports a usable anchor; the probe at the top
        # sees the uninstalled state, which is what makes this the resume path.
        reads["n"] += 1
        if reads["n"] == 1:
            return SimpleNamespace(
                anchor=None, usable=False, path="/etc/x", reason="no anchor installed"
            )
        return SimpleNamespace(anchor=anchor, usable=True, path="/etc/x", reason="")

    monkeypatch.setattr(handlers, "load_anchor", _load)
    monkeypatch.setattr(
        handlers, "create_key", lambda **k: pytest.fail("must reuse the existing key")
    )
    monkeypatch.setattr(handlers, "install_anchor", lambda *a, **k: 0)

    real_stage = handlers.stage_anchor
    staged: list[Any] = []

    def _stage(root: Path, statement: Any) -> Path:
        staged.append(statement)
        return real_stage(root, statement)

    monkeypatch.setattr(handlers, "stage_anchor", _stage)

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
    # Nothing was replaced: the anchor staged is the one for the key already on disk.
    assert staged, "the resume path staged no anchor"
    assert staged[0].key_id == store.handle.key_id
    assert staged[0].spki == store.handle.spki
    assert store.created == 1, "the resume path created a second key"
    assert store.handle.key_id in payload["receipts"][2]["detail"]


def test_the_existing_key_boundary_returns_a_handle_never_a_signer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The BOUNDARY PIN: ``_existing_key`` answers a ``KeyHandle`` or ``None``.

    ``backend.load()`` returns ``Signer | None``; every consumer of this probe wants the
    public handle. A revert to returning the signer would otherwise surface only as the
    ``existing.key_id`` crash the resume cell above reproduces, so this pins the type at
    the boundary: the result carries ``spki`` and is not a ``Signer``.
    """
    _stub_store(monkeypatch, _StubPresenceStore())
    assert handlers._existing_key(tmp_path, "auto") is None

    store = _StubPresenceStore(present=True)
    _stub_store(monkeypatch, store)
    handle = handlers._existing_key(tmp_path, "auto")

    assert isinstance(handle, keychain.KeyHandle)
    assert not isinstance(handle, keychain.Signer)
    assert handle.key_id == store.handle.key_id
    assert handle.spki == store.handle.spki
