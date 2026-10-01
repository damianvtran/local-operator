"""The onboarding runner against a scripted transport — no SSH anywhere in CI.

WHY A FAKE TRANSPORT AND NOT A MINIATURE SSH SERVER. The step machine is what
this slice adds; sshd is not. A stub that answers the runner's own interface
(``probe/connect/run/copy/close``) exercises every branch the drill's real SSH
goes through — the receipts, the gates, halt-on-contradiction, credential
cleanup, the retry — while a real pair of machines stays what the E2E drill is
for. What the fake CANNOT prove is stated where it matters (the pin itself is
also pinned by ``SshTransport``'s own cells below, which drive the real class
against a monkeypatched probe).

THE FAKE APPROVALS MODULE drives slice (a)'s frozen surface
(``onboard_approvals``): the cells assert what the runner ASKS of the store —
transitions, receipts, the refile on a contradiction — not how slice (a) writes
it. When (a) merges, both fakes keep working because both speak the documented
shape.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from local_operator.network import onboard, onboard_approvals
from local_operator.network import store as network_store
from local_operator.network.types import MeshRefusal
from local_operator.operator import OperatorAnchor
from local_operator.operator.trust import statement_digest
from local_operator.operator.verify import key_id_for, spki_fp

# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeApprovals:
    """Slice (a)'s store, minimally: one record, receipts, transitions."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.appended: list[tuple[str, str, dict[str, Any]]] = []
        self.finished: list[tuple[str, str, str]] = []
        self.refiled: list[dict[str, Any]] = []
        self.verify_refusal: MeshRefusal | None = None

    def load(self, approval_id: str) -> Any:
        return self.record if approval_id == self.record["approval_id"] else None

    def verify_signature(self, record: Any) -> None:
        if self.verify_refusal is not None:
            raise self.verify_refusal
        return None

    def begin_run(self, approval_id: str, run_id: str) -> Any:
        self.record["state"] = "connecting"
        return self.record

    def append_receipt(self, approval_id: str, run_id: str, receipt: dict[str, Any]) -> None:
        self.appended.append((approval_id, run_id, receipt))
        self.record.setdefault("receipts", []).append(receipt)

    def finish(self, approval_id: str, state: str, run_id: str) -> None:
        self.finished.append((approval_id, state, run_id))
        self.record["state"] = state

    def refile(self, approval_id: str, finding: dict[str, Any]) -> str:
        self.refiled.append(finding)
        return "ap_fresh000"


def _result(argv: tuple[str, ...], **fields: Any) -> onboard.CommandResult:
    fields.setdefault("rc", 0)
    return onboard.CommandResult(
        argv, int(fields["rc"]), fields.get("stdout", ""), fields.get("stderr", ""), at=0.0
    )


class FakeTransport:
    """A scripted transport. ``outputs`` maps a substring to a canned result."""

    def __init__(
        self,
        *,
        fingerprint: str = "SHA256:abc",
        outputs: list[tuple[str, dict[str, Any]]] | None = None,
        copy_rc: int = 0,
        connect_refusal: MeshRefusal | None = None,
    ) -> None:
        self.fingerprint = fingerprint
        self.outputs = outputs or []
        self.copy_rc = copy_rc
        self.connect_refusal = connect_refusal
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.connected: onboard.ResolvedCredential | None = None

    def probe(self) -> onboard.Probe:
        return onboard.Probe(
            ok=True,
            host="node",
            port=22,
            user="ec2-user",
            banner="SSH-2.0-OpenSSH_8.7",
            host_key_fp=self.fingerprint,
            host_keys=("node ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA",),
            at=0.0,
        )

    def connect(
        self, credential: onboard.ResolvedCredential | None, *, expected_fingerprint: str
    ) -> onboard.Probe:
        self.calls.append(("connect", (expected_fingerprint,)))
        if self.connect_refusal is not None:
            raise self.connect_refusal
        self.connected = credential
        return self.probe()

    def run(
        self, argv: Sequence[str], *, timeout: float, stdin: bytes | None = None
    ) -> onboard.CommandResult:
        command = " ".join(str(part) for part in argv)
        self.calls.append(("run", tuple(str(part) for part in argv)))
        for token, fields in self.outputs:
            if token in command:
                return _result(tuple(str(p) for p in argv), **fields)
        return _result(tuple(str(p) for p in argv))

    def copy(self, local_path: Path, remote_path: str) -> onboard.CommandResult:
        self.calls.append(("copy", (str(local_path), remote_path)))
        return _result((), rc=self.copy_rc)

    def close(self) -> None:
        self.calls.append(("close", ()))


PRE_READ_OK = (
    "os=Linux\n"
    "arch=x86_64\n"
    "user=ec2-user\n"
    "lop=yes\n"
    "lop_path=/usr/local/bin/lop\n"
    "lop_version=v0.63.2\n"
    "lop_update=yes\n"
    "uv=yes\n"
    "systemctl=yes\n"
    "linger=yes\n"
    "sudo=yes\n"
    "sudo_nopass=yes\n"
    "anchor=\n"
    "pre_read=done\n"
)


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


def _record(**what_over: Any) -> dict[str, Any]:
    anchor = _anchor()
    what: dict[str, Any] = {
        "install": True,
        "connect": True,
        "build": "0.64.12",
        "network_id": "n_1",
        "role": "drive",
        "anchor": {
            "key_id": anchor.key_id,
            "spki_fp": spki_fp(anchor.spki),
            "statement_digest": statement_digest(anchor),
        },
        "unattended": True,
        "grant": ["approve"],
    }
    what.update(what_over)
    return {
        "schema": 1,
        "approval_id": "ap_aaaa1111",
        "kind": "device_onboard",
        "state": "approved",
        "request_id": "req_1",
        "request_digest": "sha256:0",
        "device": {
            "device_id": "d_node",
            "name": "cloud-node-1",
            "fingerprint": "ABCD",
            "host": "99.79.190.164",
            "user": "ec2-user",
            "transport": "ssh",
            "host_key_fp": "SHA256:abc",
        },
        "what": what,
        "credential_ref": {"kind": "ssh", "ref": "path:/home/op/.ssh/lop.pem"},
        "expires_at": 4_000_000_000.0,
        "signature": {"by": "operator:lop-op-1", "alg": "ES256", "sig": "x"},
        "receipts": [],
    }


@pytest.fixture()
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The default config root the runner (and the store it writes) resolves to."""
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(
        "local_operator.network.identity.load",
        lambda *a, **k: SimpleNamespace(device_id="d_mac", name="this-mac"),
    )
    monkeypatch.setattr(
        "local_operator.network.store.list_networks",
        lambda *a, **k: [SimpleNamespace(network_id="n_1", name="damian-mesh")],
    )
    return root


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> OperatorAnchor:
    """The LOCAL operator store: a synthetic statement, no keychain anywhere.

    TWO targets on purpose: the runner imports ``load_anchor`` from the package
    and ``load_staged_anchor`` from ``trust`` at call time, so a patch on one
    home would leave the other import resolving the real filesystem.
    """
    anchor = _anchor()
    monkeypatch.setattr("local_operator.operator.trust.load_staged_anchor", lambda *a, **k: anchor)
    monkeypatch.setattr("local_operator.operator.load_staged_anchor", lambda *a, **k: anchor)
    monkeypatch.setattr(
        "local_operator.operator.load_anchor",
        lambda *a, **k: SimpleNamespace(anchor=anchor, usable=True, path="/etc/x"),
    )
    return anchor


def _invite_payload(token: Path) -> dict[str, Any]:
    return {
        "ok": True,
        "invite_id": "inv_1",
        "path": str(token),
        "expires_in_s": 600,
        "role": "drive",
        "hosts": [],
        "network_id": "n_1",
        "network_name": "damian-mesh",
    }


def _happy_run_local(token: Path):
    def run_local(argv: list[str], *, timeout: float) -> onboard.CommandResult:
        joined = " ".join(argv)
        if "invite" in joined:
            return onboard.CommandResult(
                tuple(argv), 0, json.dumps(_invite_payload(token)), "", at=0.0
            )
        if "ready" in joined:
            return onboard.CommandResult(
                tuple(argv),
                0,
                json.dumps({"ok": True, "checks": [{"check": "build", "ok": True}]}),
                "",
                at=0.0,
            )
        return onboard.CommandResult(tuple(argv), 0, "", "", at=0.0)

    return run_local


HAPPY_OUTPUTS = [
    ("uname", {"stdout": PRE_READ_OK}),
    ("lop-update", {"stdout": "rebuilt\n"}),
    ("lop --version", {"stdout": "v0.64.12\n"}),
    (
        "identity show",
        {"stdout": json.dumps({"ok": True, "device_id": "d_node", "fingerprint": "FP"})},
    ),
    (
        "network join",
        {
            "stdout": json.dumps(
                {
                    "ok": True,
                    "network_id": "n_1",
                    "name": "damian-mesh",
                    "members": 2,
                    "device_id": "d_node",
                    "fingerprint": "FP",
                    "sas": "123456",
                }
            )
        },
    ),
    (
        "operator install",
        {"stdout": "anchor installed at /etc/local-operator/operators/501.json\n"},
    ),
    ("operator trust", {"stdout": "trusted   : True\n"}),
    ("member grant", {"stdout": json.dumps({"ok": True, "added": ["approve", "unattended"]})}),
    ("network restart", {"stdout": json.dumps({"ok": True, "action": "restart"})}),
    ("loginctl show-user", {"stdout": "yes\n"}),
    ("network doctor", {"stdout": json.dumps({"ok": True, "checks": []})}),
    ("network peers", {"stdout": json.dumps({"ok": True, "peers": []})}),
]


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_the_step_machine_runs_the_frozen_order_and_folds_to_connected(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(outputs=HAPPY_OUTPUTS)

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "connected", payload
    assert payload["ok"] is True
    assert [row["step"] for row in payload["steps"]] == list(onboard.STEP_NAMES)
    assert all(row["ok"] for row in payload["steps"])
    assert [state for _, state, _ in fake.finished] == ["connected"]
    # The pre-approval contract in one assertion: NOTHING ran before the invite
    # step, and the invite step is local — no transport call at all.
    first_transport_call = transport.calls[0]
    assert first_transport_call[0] == "connect"
    # The invite's pre-answer is ON DISK, one-shot, keyed by the invite id, and
    # written as the approval (never as "human").
    from local_operator.network import store as network_store

    decision = network_store.pair_decision("inv_1")
    assert decision is not None
    assert decision.decision == "admit"
    assert decision.matched is True
    assert decision.answered_by == "approval:ap_aaaa1111"
    # The automated join is the ONLY spelling used on the node.
    join_calls = [c for c in transport.calls if c[0] == "run" and "network join" in " ".join(c[1])]
    assert len(join_calls) == 1
    assert "--automated" in " ".join(join_calls[0][1])


def test_the_credential_temp_is_unlinked_and_its_value_never_lands_in_receipts(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    secret_value = b"super-secret-key-material"
    temp_seen: list[Path] = []

    def secret_runner(argv: list[str], **kwargs: Any) -> Any:
        class Done:
            returncode = 0
            stdout = secret_value
            stderr = b""

        # The resolver asks `lop secret get NAME`; nothing else is expected.
        assert argv[-2:] == ["get", "node-key"] or "secret" in argv, argv
        return Done()

    def resolve(ref: dict[str, Any]) -> onboard.ResolvedCredential:
        credential = onboard.resolve_credential(
            {"kind": "ssh", "ref": "node-key"}, runner=secret_runner
        )
        assert credential.cleanup_dir is not None
        temp_seen.append(credential.cleanup_dir)
        assert stat.S_IMODE((credential.cleanup_dir / "key").stat().st_mode) == 0o600
        return credential

    record = _record()
    record["credential_ref"] = {"kind": "ssh", "ref": "node-key"}
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(outputs=HAPPY_OUTPUTS)

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=resolve,
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "connected"
    assert temp_seen, "the resolver never ran"
    assert not temp_seen[0].exists(), "the credential temp outlived the run"
    # Never in receipts, never in the payload: the VALUE does not appear in the
    # trail, and no receipt carries key material of any spelling.
    blob = json.dumps(payload, default=str)
    assert "super-secret-key-material" not in blob
    for _, _, receipt in fake.appended:
        assert "super-secret-key-material" not in json.dumps(receipt, default=str)


# ---------------------------------------------------------------------------
# Halt-on-contradiction
# ---------------------------------------------------------------------------


def test_a_contradicted_pre_read_halts_files_a_fresh_request_and_runs_nothing_more(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    # The card approved macOS; the machine says Linux. That is step zero's halt.
    record = _record(os="Darwin")
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(outputs=[("uname", {"stdout": PRE_READ_OK})])

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "failed"
    assert payload["error"] is not None and payload["error"]["code"] == "pre_read_contradiction"
    failing = payload["steps"][-1]
    assert failing["step"] == "pre_read" and failing["ok"] is False
    assert "OS" in failing["detail"] or "os" in failing["detail"]
    # The fresh-request refile went through the adapter with the finding.
    assert fake.refiled and fake.refiled[0]["check"] == "os"
    assert "A new request ap_fresh000" in failing["detail"]
    # NOTHING state-changing ran: the only transport calls are the connect and
    # the pre-read itself.
    run_calls = [c for c in transport.calls if c[0] == "run"]
    assert len(run_calls) == 1
    assert not [c for c in transport.calls if c[0] == "copy"]


def test_a_changed_host_key_halts_as_a_contradiction(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(
        connect_refusal=MeshRefusal(
            "host_key_changed",
            "the host key at 99.79.190.164:22 is no longer the one this request was "
            "approved against",
        )
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "failed"
    assert payload["error"]["code"] == "pre_read_contradiction"
    assert fake.refiled and fake.refiled[0]["check"] == "host_key_fp"
    assert not [c for c in transport.calls if c[0] == "run"]


# ---------------------------------------------------------------------------
# Failure, refusal and retry
# ---------------------------------------------------------------------------


def test_a_join_mismatch_is_a_failed_receipt_naming_the_step(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    outputs = [
        ("uname", {"stdout": PRE_READ_OK}),
        ("lop-update", {"stdout": "rebuilt\n"}),
        ("lop --version", {"stdout": "v0.64.12\n"}),
        (
            "identity show",
            {"stdout": json.dumps({"ok": True, "device_id": "d_node", "fingerprint": "FP"})},
        ),
        (
            "network join",
            {
                "rc": 1,
                "stderr": "the code did not match; the invite was not admitted\n",
            },
        ),
    ]
    transport = FakeTransport(outputs=outputs)

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "failed"
    failing = payload["steps"][-1]
    assert failing["step"] == "join" and failing["ok"] is False
    assert "did not complete" in failing["detail"]
    assert [state for _, state, _ in fake.finished] == ["failed"]
    # The runner stopped: nothing after the join ran.
    commands = " | ".join(" ".join(c[1]) for c in transport.calls if c[0] == "run")
    assert "member grant" not in commands
    assert "operator install" not in commands
    # A DEAD RUN'S PRE-ANSWER DOES NOT OUTLIVE IT (round-1 Finding 1's secondary
    # note): the run wrote the admit decision at the invite and died at the
    # join; the runner that wrote it — and only it — clears it.
    assert network_store.pair_decision("inv_1", isolated) is None


def test_a_retry_reuses_the_record_with_a_new_run_id(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)

    failing = FakeTransport(
        outputs=[
            ("uname", {"stdout": PRE_READ_OK}),
            ("lop-update", {"rc": 1, "stderr": "network down\n"}),
        ]
    )
    first = onboard.execute_approval(
        "ap_aaaa1111",
        transport=failing,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )
    assert first["state"] == "failed"
    assert first["steps"][-1]["step"] == "install"

    second_transport = FakeTransport(outputs=HAPPY_OUTPUTS)
    second = onboard.execute_approval(
        "ap_aaaa1111",
        transport=second_transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert second["state"] == "connected"
    assert second["ok"] is True
    assert second["run_id"] != first["run_id"], "a retry must mint a NEW run id"
    run_ids = {run_id for _, run_id, _ in fake.appended}
    assert len(run_ids) == 2, "receipts from both runs must be on the ONE record"
    assert "ap_aaaa1111" == record["approval_id"]


def test_a_tampered_record_cannot_mint_an_invite_or_pre_answer_a_join(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F4 AT THE FIRST EDGE (agent review round 1, Finding 1).

    The invite step runs before any gate and writes real things — a live invite
    token and the admit pre-answer. The reviewer's repro showed a tampered record
    doing both and only meeting the refusal at the pre_read gate; with the
    verification inside ``begin_run`` the refusal is the ONLY event: no token is
    minted, no decision file appears, and no receipt is written.
    """
    record = _record()
    fake = FakeApprovals(record)
    fake.verify_refusal = MeshRefusal(
        "approval_record_tampered", "the record's signature does not verify"
    )
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    minted: list[list[str]] = []

    def run_local(argv: list[str], *, timeout: float) -> onboard.CommandResult:
        minted.append([str(part) for part in argv])
        return _result(tuple(str(part) for part in argv))

    with pytest.raises(MeshRefusal) as excinfo:
        onboard.execute_approval("ap_aaaa1111", transport=FakeTransport(), run_local=run_local)

    assert excinfo.value.code == "approval_record_tampered"
    assert minted == [], "an invite must not be minted from an unverified record"
    assert network_store.pair_decision("inv_1", isolated) is None
    assert fake.appended == [], "nothing may run, so no receipt may be written"
    assert fake.finished == []


def test_a_terminal_record_is_never_re_run(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = _record()
    record["state"] = "connected"
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)

    with pytest.raises(MeshRefusal) as excinfo:
        onboard.execute_approval("ap_aaaa1111", transport=FakeTransport())
    assert excinfo.value.code == "approval_already_connected"
    assert fake.appended == []


def test_an_expired_record_cannot_start_a_run(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = _record()
    record["expires_at"] = 1.0  # long past
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)

    with pytest.raises(MeshRefusal) as excinfo:
        onboard.execute_approval("ap_aaaa1111", transport=FakeTransport())
    assert excinfo.value.code == "approval_expired"


# ---------------------------------------------------------------------------
# SshTransport's own pin
# ---------------------------------------------------------------------------


def test_connect_refuses_a_host_key_that_moved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = onboard.SshTransport(host="node", user="ec2-user")
    monkeypatch.setattr(
        onboard,
        "probe",
        lambda *a, **k: onboard.Probe(
            ok=True,
            host="node",
            port=22,
            user="ec2-user",
            banner="SSH-2.0-OpenSSH",
            host_key_fp="SHA256:different",
            host_keys=("node ssh-ed25519 AAAA",),
            at=0.0,
        ),
    )
    with pytest.raises(MeshRefusal) as excinfo:
        transport.connect(None, expected_fingerprint="SHA256:approved")
    assert excinfo.value.code == "host_key_changed"
    assert transport._known_hosts is None, "no seed may be written when the pin refuses"


def test_the_known_hosts_seed_is_private_and_removed_on_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = onboard.SshTransport(host="node", user="ec2-user")
    monkeypatch.setattr(
        onboard,
        "probe",
        lambda *a, **k: onboard.Probe(
            ok=True,
            host="node",
            port=22,
            user="ec2-user",
            banner="SSH-2.0-OpenSSH",
            host_key_fp="SHA256:approved",
            host_keys=("node ssh-ed25519 AAAA",),
            at=0.0,
        ),
    )
    found = transport.connect(None, expected_fingerprint="SHA256:approved")
    assert found.host_key_fp == "SHA256:approved"
    seed = transport._known_hosts
    assert seed is not None and seed.exists()
    assert stat.S_IMODE(seed.stat().st_mode) == 0o600
    assert "ssh-ed25519" in seed.read_text(encoding="utf-8")
    transport.close()
    assert not seed.exists(), "the seed must not outlive the run"


# ---------------------------------------------------------------------------
# The credential-free contract, stated mechanically
# ---------------------------------------------------------------------------


def test_probe_speaks_no_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """``probe`` runs one credential-free keyscan and reads the banner only."""

    recorded: list[list[str]] = []

    class FakeSock:
        def __init__(self) -> None:
            self._sent = False

        def settimeout(self, value: float) -> None:
            return None

        def recv(self, size: int) -> bytes:
            if self._sent:
                return b""
            self._sent = True
            return b"SSH-2.0-OpenSSH_9.0\r\n"

        def close(self) -> None:
            return None

        def __enter__(self) -> FakeSock:
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    monkeypatch.setattr("socket.create_connection", lambda *a, **k: FakeSock())

    class Done:
        returncode = 0
        stdout = "node ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIQQQ\n"
        stderr = ""

    def runner(argv: list[str], **kwargs: Any) -> Any:
        recorded.append([str(part) for part in argv])
        return Done()

    found = onboard.probe("node", runner=runner)
    assert found.ok is True
    assert found.banner.startswith("SSH-2.0")
    assert found.host_key_fp.startswith("SHA256:")
    assert len(recorded) == 1
    keyscan = recorded[0]
    # THE CONTRACT: no `-i`, no identity, no secret-bearing option anywhere.
    for forbidden in ("-i", "IdentityFile", "IdentityAgent", "password"):
        assert forbidden not in keyscan, keyscan
    assert "ssh-keyscan" in " ".join(keyscan) or keyscan[0] == "ssh-keyscan"
