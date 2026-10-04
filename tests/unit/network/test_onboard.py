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
    """Slice (a)'s store, minimally: one record, receipts, transitions.

    Method names and keyword shapes mirror ``local_operator.network.approvals``
    as MERGED (the rebase bound the adapter to this surface): ``load_record``,
    ``verify_for_run``, ``begin_run``, ``append_receipt``, ``mark_connected``,
    ``mark_failed``, and the refile mint (``create_request`` +
    ``new_request_id``).
    """

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.appended: list[tuple[str, str, dict[str, Any]]] = []
        self.finished: list[tuple[str, str, str]] = []
        self.refiled: list[dict[str, Any]] = []
        self.verify_refusal: MeshRefusal | None = None

    def load_record(self, approval_id: str, **_kwargs: Any) -> Any:
        return self.record if approval_id == self.record["approval_id"] else None

    def verify_for_run(self, approval_id: str, **_kwargs: Any) -> Any:
        if self.verify_refusal is not None:
            raise self.verify_refusal
        return self.record

    def begin_run(self, approval_id: str, *, run_id: str = "", **_kwargs: Any) -> Any:
        self.record["state"] = "connecting"
        return self.record

    def append_receipt(
        self,
        approval_id: str,
        *,
        run_id: str,
        step: str,
        ok: bool,
        detail: str = "",
        digest: str = "",
        at: Any = None,
        **_kwargs: Any,
    ) -> None:
        row = {
            "run_id": run_id,
            "step": step,
            "ok": ok,
            "detail": detail,
            "digest": digest,
            "at": at if at is not None else 0.0,
        }
        self.appended.append((approval_id, run_id, row))
        self.record.setdefault("receipts", []).append(row)

    def mark_failed(
        self,
        approval_id: str,
        *,
        run_id: str,
        step: str = "",
        detail: str = "",
        **_kwargs: Any,
    ) -> None:
        self.finished.append((approval_id, "failed", run_id))
        self.record["state"] = "failed"
        self.record.setdefault("receipts", []).append(
            {
                "run_id": run_id,
                "step": step,
                "ok": False,
                "detail": detail,
                "digest": "",
                "at": 0.0,
            }
        )

    def mark_connected(
        self,
        approval_id: str,
        *,
        run_id: str,
        step: str = "verify",
        detail: str = "",
        **_kwargs: Any,
    ) -> None:
        self.finished.append((approval_id, "connected", run_id))
        self.record["state"] = "connected"
        self.record.setdefault("receipts", []).append(
            {
                "run_id": run_id,
                "step": step,
                "ok": True,
                "detail": detail,
                "digest": "",
                "at": 0.0,
            }
        )

    def create_request(self, **fields: Any) -> dict[str, Any]:
        self.refiled.append(dict(fields))
        return {"approval_id": "ap_fresh000"}

    def new_request_id(self) -> str:
        return "req_fresh000"


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
        # Mirrors ``SshTransport``'s early record: the halt's refile reads the
        # probe the refusal came from (the real transport sets it before the
        # fingerprint check), so the fake carries the same surface.
        self.last_probe: onboard.Probe | None = None

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


def _invite_payload(token: Path, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "invite_id": "inv_1",
        "path": str(token),
        "expires_in_s": 600,
        "role": "drive",
        "hosts": [],
        "network_id": "n_1",
        "network_name": "damian-mesh",
    }
    payload.update(overrides)
    return payload


def _happy_run_local(token: Path, **invite_overrides: Any):
    def run_local(argv: list[str], *, timeout: float) -> onboard.CommandResult:
        joined = " ".join(argv)
        if "invite" in joined:
            return onboard.CommandResult(
                tuple(argv), 0, json.dumps(_invite_payload(token, **invite_overrides)), "", at=0.0
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
    # The fresh-request refile went through the adapter against the store's
    # mint: the created request CARRIES THE CORRECTED FACT (os: Linux, observed)
    # while the old record's failed state stays where it was.
    assert fake.refiled and fake.refiled[0]["what"]["os"] == "Linux"
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
    # The probe the refusal came from: the real transport records it before the
    # fingerprint check, so the refile can name the key that actually answered.
    transport.last_probe = onboard.Probe(
        ok=True,
        host="node",
        port=22,
        user="ec2-user",
        banner="SSH-2.0-OpenSSH",
        host_key_fp="SHA256:v2-shifted",
        host_keys=("node ssh-ed25519 AAAA",),
        at=0.0,
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
    # The refile carries the CORRECTED host key (the observed value, from the
    # probe the halt came from) on the fresh request, against the store's
    # merged mint surface.
    assert fake.refiled and fake.refiled[0]["device"]["host_key_fp"] == "SHA256:v2-shifted"
    assert not [c for c in transport.calls if c[0] == "run"]


def test_a_missing_host_key_halts_with_the_observed_key_so_the_refile_carries_it(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Drill finding, 2026-10-03: a card filed without a host-key fingerprint
    halted at pre_read, and the auto-refiled replacement was minted with the SAME
    hole — so the remedy reproduced the failure.

    The halt observes the key (a credential-free read) before halting, and the
    fresh request carries it: the field lands on the refile.
    """
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record()
    record["device"].pop("host_key_fp", None)  # the filed-without-key card
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(fingerprint="SHA256:observed-now")

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "failed"
    assert payload["error"]["code"] == "pre_read_contradiction"
    assert "host-key fingerprint" in payload["error"]["message"]
    failing = payload["steps"][-1]
    assert failing["step"] == "pre_read" and failing["ok"] is False
    # The fresh request CARRIES the observed key — the field lands.
    assert fake.refiled and fake.refiled[0]["device"]["host_key_fp"] == "SHA256:observed-now"
    assert "A new request ap_fresh000" in failing["detail"]
    # Nothing state-changing ran: no run/copy before the halt.
    assert not [c for c in transport.calls if c[0] in ("run", "copy")]


def test_a_missing_host_key_with_no_observation_files_no_unrunnable_replacement(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the key cannot even be observed (the machine never answered the
    read), NOTHING is refiled — a replacement without the key would reproduce
    the very halt this is, and the sentence says a new request is needed instead.
    """
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record()
    record["device"].pop("host_key_fp", None)
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(fingerprint="")  # the probe observes nothing

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "failed"
    assert payload["error"]["code"] == "pre_read_contradiction"
    failing = payload["steps"][-1]
    # The sentence scopes to what is missing and names it (design round 1, D2):
    # nothing claims corrected facts, because nothing was corrected.
    assert "Nothing can run until this changes" in failing["detail"]
    assert "the request does not say which host key to expect" in failing["detail"]
    assert "A new request is needed then" in failing["detail"]
    assert "corrected facts" not in failing["detail"]
    assert fake.refiled == []


def test_an_unmapped_contradiction_mints_no_identical_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Design round 1, D2: ``install_scope``/``sudo`` findings carry no fact the
    refile can apply, and the earlier shape minted an IDENTICAL card under a
    "corrected facts" sentence — the same lie-class as the keyless refile, one
    family over."""
    created: list[dict[str, Any]] = []

    class FakeModule:
        """Only the surface ``refile_after_contradiction`` reads."""

        def load_record(self, approval_id: str, **_kwargs: Any) -> Any:
            return {
                "approval_id": approval_id,
                "kind": "device_onboard",
                "device": {"device_id": "d_node", "name": "cloud-node-1"},
                "what": {"build": "0.64.12"},
                "requested_by": {},
                "credential_ref": {},
            }

        def create_request(self, **fields: Any) -> dict[str, Any]:
            created.append(fields)
            return {"approval_id": "ap_never"}

        def new_request_id(self) -> str:
            return "ap_never"

    monkeypatch.setattr(onboard_approvals, "_module", lambda: FakeModule())
    finding = {
        "check": "sudo",
        "approved": "the anchor install was approved",
        "observed": "no sudo on the machine",
        "why": "the admin approval cannot be raised on this machine",
    }
    assert onboard_approvals.refile_after_contradiction("ap_aaaa1111", finding) is None
    assert created == []


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


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        # The live shape: the node's own taxonomy, plus the receipt's refusal
        # word beside it under its own name (N3) — never a bare second ``code``.
        (
            {
                "stage": "offer_read",
                "class": "link_crypto",
                "kind": "auth",
                "host": "127.0.0.1:4098",
                "records_sent": 0,
                "records_received": 0,
                "local_only": True,
            },
            {
                "stage": "offer_read",
                "class": "link_crypto",
                "kind": "auth",
                "refusal_code": "join_failed",
            },
        ),
        # A node block carrying its own ``code`` keeps it: its word wins over
        # the receipt's reconstruction.
        (
            {
                "stage": "preflight",
                "class": "invite",
                "kind": "invite_expired",
                "code": "node_expired",
            },
            {
                "stage": "preflight",
                "class": "invite",
                "kind": "invite_expired",
                "refusal_code": "node_expired",
            },
        ),
        # An older node has no block key at all: the receipt stays silent about
        # a class it was not told rather than guessing.
        (None, None),
        # Empty values are filtered, never shipped as blanks; the refusal word
        # still lands under ``refusal_code``.
        (
            {"stage": "", "class": "refused", "kind": None, "code": ""},
            {"class": "refused", "refusal_code": "join_failed"},
        ),
    ],
)
def test_the_receipt_copies_the_nodes_join_block(
    isolated: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    block: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> None:
    """F4: the inviter-side agent reads the joiner's class without an SSH session.

    The node's ``join --json`` refusal body carries a ``join`` block; the receipt
    copies the node's own keys verbatim and files the receipt's refusal word
    beside them as ``refusal_code`` (design round 1, N3 — one object must not mix
    two taxonomies under one bare name). QA round 1 probed this over the frozen
    fakes; this cell is the regression guard it asked for.
    """
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    refusal: dict[str, Any] = {
        "ok": False,
        "code": "join_failed",
        "message": ("could not join: the handshake at 127.0.0.1:4098 stopped (LinkCryptoError)"),
    }
    if block is not None:
        refusal["join"] = block
    outputs = [
        ("uname", {"stdout": PRE_READ_OK}),
        ("lop-update", {"stdout": "rebuilt\n"}),
        ("lop --version", {"stdout": "v0.64.12\n"}),
        (
            "identity show",
            {"stdout": json.dumps({"ok": True, "device_id": "d_node", "fingerprint": "FP"})},
        ),
        ("network join", {"rc": 1, "stdout": json.dumps(refusal)}),
        # The failure path probes the node's own relay for the sentence; a
        # stopped relay keeps this cell about the copy, not the relay branches.
        (
            "network status",
            {
                "stdout": json.dumps(
                    {"ok": True, "relay_running": False, "relay_answering": False, "networks": []}
                )
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
    copied = failing["data"].get("join")
    if expected is None:
        assert copied is None, copied
    else:
        assert copied == expected, copied


def test_a_join_failure_names_the_already_serving_relay_and_keeps_the_nodes_message(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Drill finding, 2026-10-03: four re-onboards of an already-joined node
    failed at ``join`` as a bare ``join_failed``, while the node's own systemd
    relay held :4097 (its log: ``OSError: [Errno 98] Address already in use``).

    Two things change on the failure path only: the node's own sentence (the
    payload's ``message``) rides beside the code instead of being swallowed, and
    the status probe's verdict — a relay already serving there — is named as the
    cause beside the remedy that actually worked.
    """
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
                "stdout": json.dumps(
                    {
                        "ok": False,
                        "code": "join_failed",
                        "message": "could not join: nothing was listening at 192.168.0.155:4097",
                    }
                ),
            },
        ),
        (
            "network status",
            {
                "stdout": json.dumps(
                    {
                        "ok": True,
                        "relay_running": True,
                        "relay_answering": True,
                        "relay_state": "live",
                        "port": 4097,
                        "listening": {"address": "0.0.0.0", "port": 4097},
                    }
                )
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
    detail = failing["detail"]
    assert "cannot be cleared as the cause" in detail
    # The node's own SENTENCE rides (design round 1, D6): the code is a machine
    # field now, never part of the sentence.
    assert "could not join: nothing was listening at 192.168.0.155:4097" in detail
    assert "join_failed" not in detail
    assert failing["data"]["code"] == "join_failed"
    # What the probe VERIFIED: the serving port, the named machine — and, since
    # round 1 (D4/Q-1), the same form its siblings carry: the restart is not a
    # remedy here either, and no target comparison it could not make.
    assert "a relay is already serving :4097 on cloud-node-1" in detail
    assert "could not match it to the network this join is for" in detail
    assert "Restarting the relay would not change what it serves." in detail
    assert "lop network restart" not in detail
    assert "Retry the join once the cause in the node's message is cleared" in detail
    assert "relay step" not in detail
    assert "can hold the connection" not in detail
    assert failing["data"]["relay_serving"] is True
    assert failing["data"]["relay_port"] == 4097
    assert "relay_serves_target" not in failing["data"]
    assert failing["data"]["node_refused"] is True
    # The runner stopped at the join, exactly as before.
    commands = " | ".join(" ".join(c[1]) for c in transport.calls if c[0] == "run")
    assert "member grant" not in commands


def test_a_wedged_relay_failure_names_no_unverified_port_or_mechanism(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review round 1, R-MINOR: the earlier shape fell back to the port it was
    ASKED about and asserted the relay serving — a claim the probe did not
    verify. A registered relay that did not answer this probe says exactly
    that, with the same one action."""
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
                "stdout": json.dumps(
                    {
                        "ok": False,
                        "code": "join_failed",
                        "message": "could not join: nothing was listening at 192.168.0.155:4097",
                    }
                ),
            },
        ),
        (
            "network status",
            {
                "stdout": json.dumps(
                    {
                        "ok": True,
                        "relay_running": True,
                        "relay_answering": False,
                        "relay_state": "wedged",
                        "port": 4097,
                    }
                )
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

    failing = payload["steps"][-1]
    detail = failing["detail"]
    assert "could not confirm it serving" in detail
    assert "state: wedged" in detail
    assert "a relay is already serving" not in detail
    assert "serving :4097" not in detail
    assert "lop network restart" in detail
    assert "retry the join" in detail
    assert "relay_serving" not in failing["data"]
    assert failing["data"]["relay_state"] == "wedged"


def test_a_join_failure_where_the_relay_already_serves_the_target_network_adopts_it(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Drill decision, 2026-10-04: a relay already serving the SAME network is
    not a collision — the join failure names it as correct and left in place,
    and never prescribes a restart for it."""
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
                "stdout": json.dumps(
                    {
                        "ok": False,
                        "code": "join_failed",
                        "message": (
                            "could not join: the handshake at 127.0.0.1:4098 stopped "
                            "(LinkCryptoError)"
                        ),
                    }
                ),
            },
        ),
        (
            "network status",
            {
                "stdout": json.dumps(
                    {
                        "ok": True,
                        "relay_running": True,
                        "relay_answering": True,
                        "relay_state": "live",
                        "port": 4097,
                        "listening": {"address": "0.0.0.0", "port": 4097},
                        # The invite payload names n_1/damian-mesh; the relay
                        # already serves exactly that.
                        "networks": [{"network_id": "n_1", "name": "damian-mesh", "epoch": 1}],
                    }
                )
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

    failing = payload["steps"][-1]
    detail = failing["detail"]
    assert "already serving :4097 on cloud-node-1" in detail
    assert "it serves damian-mesh" in detail
    assert "left as it is" in detail
    assert "not the cause of this failure" in detail
    assert "LinkCryptoError" in detail
    # Round-1 D3: a next move is named, the doubled "already" is gone, and no
    # restart is prescribed for a relay that was never the problem.
    assert detail.count("already") == 1, detail
    assert "Retry the join once the cause in the node's message is cleared" in detail
    assert "nothing about the relay needs restarting" in detail
    assert "lop network restart" not in detail
    assert failing["data"]["relay_serves_target"] is True
    assert failing["data"]["relay_serving"] is True


def test_a_join_failure_against_a_hand_started_relay_says_only_what_restart_does(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round-1 D1, hand-started case; round-2 D6 makes the fixture the drill
    node's EXACT state: a unit file exists (the arm's failed install left it —
    ``installed`` reads true) while a hand-started ``lop network serve`` is what
    serves :4097. The kind therefore rides the probe's verified
    ``relay_served_by``, never the file — and the sentence must still be the
    hand-started one, with the supervised register absent. The restart cannot
    be the fix here either (the target can appear only after a completed join);
    the note states the one mechanism that IS true of a hand-started relay
    (restart replaces it with the service's own) and keeps the next move."""
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
                "stdout": json.dumps(
                    {
                        "ok": False,
                        "code": "join_failed",
                        "message": "could not join: nothing was listening at 192.168.0.155:4097",
                    }
                ),
            },
        ),
        (
            "network status",
            {
                "stdout": json.dumps(
                    {
                        "ok": True,
                        # D6, THE NODE'S STATE: the file exists AND is not the
                        # kind signal — the serving process is (relay_served_by).
                        "installed": True,
                        "relay_served_by": "manual",
                        "relay_running": True,
                        "relay_answering": True,
                        "relay_state": "live",
                        "port": 4097,
                        "listening": {"address": "0.0.0.0", "port": 4097},
                        "networks": [{"network_id": "n_other", "name": "lab-mesh", "epoch": 3}],
                    }
                )
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

    failing = payload["steps"][-1]
    detail = failing["detail"]
    assert "a relay is already serving :4097 on cloud-node-1" in detail
    assert "but it does not serve damian-mesh" in detail
    assert "it serves lab-mesh instead" in detail
    assert "the network this join is for can appear there only after a completed join" in detail
    assert "the relay is not the cause of this failure" in detail
    assert "That relay was started by hand, not by the service" in detail
    assert "`lop network restart` replaces it with the service's own relay" in detail
    assert "neither changes what the relay serves" in detail
    assert "Retry the join once the cause in the node's message is cleared" in detail
    # The supervised register must NOT appear in this kind's sentence.
    assert "the service's own, and restarting it would not change" not in detail
    assert failing["data"]["relay_serves_target"] is False
    assert failing["data"]["relay_serving"] is True


def test_a_join_failure_against_the_supervised_relay_names_the_same_served_set(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round-1 D1, supervised case: the unit restarts back onto the SAME served
    set (it serves ``store.list_networks()``; a failed handshake never writes
    the target), so prescribing a restart as the fix would be the same
    remedy-that-cannot-act one branch over. The note says what is true: it is
    the service's own, and a restart would not change what it serves.

    Round-2 D6: "the service's own" now rides the probe's VERIFIED
    ``relay_served_by`` (the supervisor's own pid answers on the record's pid),
    never unit-file presence."""
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
                "stdout": json.dumps(
                    {
                        "ok": False,
                        "code": "join_failed",
                        "message": "could not join: nothing was listening at 192.168.0.155:4097",
                    }
                ),
            },
        ),
        (
            "network status",
            {
                "stdout": json.dumps(
                    {
                        "ok": True,
                        # The supervised unit serves the same set after any
                        # restart — the fact this branch must not bury. The kind
                        # is the verified one (D6), not the file.
                        "installed": True,
                        "relay_served_by": "service",
                        "relay_running": True,
                        "relay_answering": True,
                        "relay_state": "live",
                        "port": 4097,
                        "listening": {"address": "0.0.0.0", "port": 4097},
                        "networks": [{"network_id": "n_other", "name": "lab-mesh", "epoch": 3}],
                    }
                )
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

    failing = payload["steps"][-1]
    detail = failing["detail"]
    assert "a relay is already serving :4097 on cloud-node-1" in detail
    assert "but it does not serve damian-mesh" in detail
    assert "it serves lab-mesh instead" in detail
    assert "the relay is not the cause of this failure" in detail
    assert (
        "That relay is the service's own, and restarting it would not change what it serves."
        in detail
    )
    assert "Retry the join once the cause in the node's message is cleared" in detail
    # No hand-start mechanism, and no bare restart command, in this kind's
    # sentence.
    assert "started by hand" not in detail
    assert "lop network restart" not in detail
    assert failing["data"]["relay_serves_target"] is False
    assert failing["data"]["relay_serving"] is True


def test_empty_served_rows_do_not_leak_a_broken_network_list(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round-1 R-NIT-1: two all-empty served rows used to join into ", " and
    render "a different network (, )"."""
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
                "stdout": json.dumps(
                    {
                        "ok": False,
                        "code": "join_failed",
                        "message": "could not join: nothing was listening at 192.168.0.155:4097",
                    }
                ),
            },
        ),
        (
            "network status",
            {
                "stdout": json.dumps(
                    {
                        "ok": True,
                        "relay_running": True,
                        "relay_answering": True,
                        "relay_state": "live",
                        "port": 4097,
                        "listening": {"address": "0.0.0.0", "port": 4097},
                        "networks": [
                            {"network_id": "", "name": "", "epoch": 1},
                            {"network_id": "", "name": "", "epoch": 2},
                        ],
                    }
                )
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

    detail = payload["steps"][-1]["detail"]
    assert "(, )" not in detail
    assert "it serves a different network instead" in detail
    assert "Restarting the relay would not change what it serves." in detail
    assert payload["steps"][-1]["data"]["relay_serves_target"] is False


def test_an_answering_relay_with_no_network_is_its_own_branch(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round-2 D7: the answering relay listed NO network — the ordinary
    fresh-device shape — and the probe DID tell that. The copy says so (it
    serves no network yet; the join target can appear only after a completed
    join), not "could not tell", and still names no restart as the fix."""
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
                "stdout": json.dumps(
                    {
                        "ok": False,
                        "code": "join_failed",
                        "message": "could not join: nothing was listening at 192.168.0.155:4097",
                    }
                ),
            },
        ),
        (
            "network status",
            {
                "stdout": json.dumps(
                    {
                        "ok": True,
                        "relay_running": True,
                        "relay_answering": True,
                        "relay_state": "live",
                        "port": 4097,
                        "listening": {"address": "0.0.0.0", "port": 4097},
                        "networks": [],
                    }
                )
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

    failing = payload["steps"][-1]
    detail = failing["detail"]
    assert "it serves no network yet" in detail
    assert "the network this join is for can appear there only after a completed join" in detail
    assert "the relay is not the cause of this failure" in detail
    assert "could not match" not in detail
    assert "Retry the join once the cause in the node's message is cleared" in detail
    assert failing["data"]["relay_serves_target"] is False
    assert failing["data"]["relay_serving"] is True


def test_an_undecodable_target_with_a_served_list_keeps_the_generic_sentence(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round-1 Q-1: with NO usable target (the mint payload carries no ids and
    the token bytes do not decode) and a served list present, the old shape
    rendered the different-network sentence and claimed
    ``relay_serves_target: false`` — a comparison the flow could not make. The
    generic sentence stays, and the field is absent."""
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
                "stdout": json.dumps(
                    {
                        "ok": False,
                        "code": "join_failed",
                        "message": "could not join: nothing was listening at 192.168.0.155:4097",
                    }
                ),
            },
        ),
        (
            "network status",
            {
                "stdout": json.dumps(
                    {
                        "ok": True,
                        "relay_running": True,
                        "relay_answering": True,
                        "relay_state": "live",
                        "port": 4097,
                        "listening": {"address": "0.0.0.0", "port": 4097},
                        "networks": [{"network_id": "n_other", "name": "lab-mesh", "epoch": 3}],
                    }
                )
            },
        ),
    ]
    transport = FakeTransport(outputs=outputs)

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        # The mint payload carries no network ids; the token bytes do not decode.
        run_local=_happy_run_local(token, network_id="", network_name=""),
    )

    failing = payload["steps"][-1]
    detail = failing["detail"]
    assert "could not match it to the network this join is for" in detail
    assert "could not tell" not in detail
    assert "does not serve" not in detail
    assert "it serves lab-mesh instead" not in detail
    assert "Retry the join once the cause in the node's message is cleared" in detail
    assert failing["data"]["relay_serving"] is True
    assert failing["data"]["relay_port"] == 4097
    assert "relay_serves_target" not in failing["data"]


# ---------------------------------------------------------------------------
# Join entry: satisfied when the node is already an active member (slice A)
# ---------------------------------------------------------------------------


def _mint_token(token: Path, *, network_id: str = "n_1", epoch: int = 1) -> None:
    """Write a REAL decodable invite token — the entry check reads its epoch."""
    from local_operator.network import invite as invite_mod
    from local_operator.network import wire as wire_mod
    from local_operator.network.types import NetworkRecord

    record = NetworkRecord(
        network_id=network_id, name="damian-mesh", epoch=epoch, self_device_id="d_mac"
    )
    minted = invite_mod.mint(record, wire_mod.b64u(bytes(range(32))), role="drive", ttl_s=600.0)
    token.write_text(minted.token, encoding="utf-8")


def _inviter_record(*, member: bool = True, burned: bool = False):
    """THIS device's record for the invite's network, as the store holds it."""
    from local_operator.network.types import MemberRecord, NetworkRecord

    rows = (
        [
            MemberRecord(
                device_id="d_node",
                public_key="pk-node",
                name="cloud-node-1",
                role="drive",
                lifecycle="active",
            )
        ]
        if member
        else []
    )
    record = NetworkRecord(
        network_id="n_1",
        name="damian-mesh",
        epoch=1,
        self_device_id="d_mac",
        members=rows,
    )
    if burned:
        record.removed_ids = ["d_node"]
    return record


def _status_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "network_id": "n_1",
        "name": "damian-mesh",
        "epoch": 1,
        "role": "drive",
        "trust": "active",
        "members": 2,
        "links": 1,
        "stale": "",
        "self_device_id": "d_node",
        "membership_state": "active",
        "membership": {"state": "active", "sentence": "this device is an active member"},
    }
    row.update(overrides)
    return row


def _status_output(*rows: dict[str, Any]) -> dict[str, Any]:
    """The node's ``lop network status --json`` as the entry check reads it."""
    return {
        "stdout": json.dumps(
            {
                "ok": True,
                "relay_running": True,
                "relay_answering": True,
                "relay_state": "live",
                "port": 4097,
                "networks": list(rows),
            }
        )
    }


def test_an_already_active_member_skips_the_join_without_a_push_or_dial(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run 10: re-joining a clean active member must satisfy the step, not re-run it.

    The node's own row said ``membership_state: active`` at the invite's epoch,
    the member table carried the device, and the attempt still went to the wire —
    where it exercised the join MECHANISM rather than the requirement ("is this
    node admitted?"). Both local reads agree here, so the step is satisfied:
    no push, no dial, and the receipt carries the reads.
    """
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    _mint_token(token)
    network_store.save(_inviter_record(), None)
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(
        outputs=HAPPY_OUTPUTS + [("network status", _status_output(_status_row()))]
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "connected", payload
    join = {row["step"]: row for row in payload["steps"]}["join"]
    assert join["ok"] is True
    assert join["detail"] == (
        "cloud-node-1 is already an active member of damian-mesh (epoch 1); admission "
        "is satisfied and no re-join was attempted — the invite goes unused and expires."
    )
    assert join["data"]["membership"] == {
        "state": "active",
        "epoch": 1,
        "device_id": "d_node",
        "source": ["node status", "inviter member table"],
    }
    assert join["data"]["device_id"] == "d_node"
    # NO PUSH, NO DIAL — the invite never leaves this machine and no join runs.
    assert not [c for c in transport.calls if c[0] == "copy" and "lop-invite-" in c[1][1]]
    assert not [c for c in transport.calls if c[0] == "run" and "network join" in " ".join(c[1])]


def test_a_same_named_other_network_row_cannot_satisfy(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Names are not unique by design — the id is the identity (reviewer r1, MINOR-1).

    Reproduced at head ``daf9a4819``: with a same-named row for a DIFFERENT network
    id listed BEFORE the invite's own row, the ``network_id OR name`` matcher read
    the other network's ``active`` row and reported satisfied while the invite's
    own row says ``removed`` — the outcome was row-order dependent. The id-only
    match must fall through to the join here.
    """
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    _mint_token(token)
    network_store.save(_inviter_record(), None)
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    other = _status_row(network_id="n_9", membership_state="active")
    target_row = _status_row(membership_state="removed")
    transport = FakeTransport(
        outputs=HAPPY_OUTPUTS + [("network status", _status_output(other, target_row))]
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "connected", payload
    join = {row["step"]: row for row in payload["steps"]}["join"]
    assert "already an active member" not in join["detail"]
    assert "joined damian-mesh as d_node" in join["detail"]
    assert [c for c in transport.calls if c[0] == "copy"]
    join_calls = [c for c in transport.calls if c[0] == "run" and "network join" in " ".join(c[1])]
    assert len(join_calls) == 1


def test_a_node_missing_from_the_inviters_table_still_joins(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One side cannot satisfy the gate: the node says active, the table lacks the
    row — and the join upserting that row is exactly what the table needs."""
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    _mint_token(token)
    network_store.save(_inviter_record(member=False), None)
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(
        outputs=HAPPY_OUTPUTS + [("network status", _status_output(_status_row()))]
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "connected", payload
    join = {row["step"]: row for row in payload["steps"]}["join"]
    assert join["ok"] is True
    assert "already an active member" not in join["detail"]
    assert "joined damian-mesh as d_node" in join["detail"]
    assert [c for c in transport.calls if c[0] == "copy"]
    join_calls = [c for c in transport.calls if c[0] == "run" and "network join" in " ".join(c[1])]
    assert len(join_calls) == 1


def test_a_burned_device_is_refused_before_any_push_or_dial(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``removed_ids`` is forever: the refusal is pre-empted from the member table
    with the relay's own sentence, so a removed device spends no ceremony."""
    from local_operator.network import relay as relay_mod

    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    _mint_token(token)
    inviter = _inviter_record(burned=True)
    network_store.save(inviter, None)
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(
        outputs=[
            ("uname", {"stdout": PRE_READ_OK}),
            ("lop-update", {"stdout": "rebuilt\n"}),
            ("lop --version", {"stdout": "v0.64.12\n"}),
            (
                "identity show",
                {"stdout": json.dumps({"ok": True, "device_id": "d_node", "fingerprint": "FP"})},
            ),
        ]
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "failed", payload
    failing = payload["steps"][-1]
    assert failing["step"] == "join" and failing["ok"] is False
    assert failing["detail"] == relay_mod.membership_conflict(inviter, "d_node")
    assert "burned id is never admitted again" in failing["detail"]
    assert failing["data"]["code"] == "device_id_conflict"
    assert not [c for c in transport.calls if c[0] == "copy"]
    assert not [c for c in transport.calls if c[0] == "run" and "network join" in " ".join(c[1])]


def test_an_epoch_mismatch_still_joins(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The join's job is to write the CURRENT secret: a node whose row sits at an
    epoch other than the invite's gets the join, not the skip."""
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    _mint_token(token, epoch=1)
    network_store.save(_inviter_record(), None)
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(
        outputs=HAPPY_OUTPUTS + [("network status", _status_output(_status_row(epoch=2)))]
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "connected", payload
    join = {row["step"]: row for row in payload["steps"]}["join"]
    assert "already an active member" not in join["detail"]
    assert [c for c in transport.calls if c[0] == "copy"]
    join_calls = [c for c in transport.calls if c[0] == "run" and "network join" in " ".join(c[1])]
    assert len(join_calls) == 1


def test_a_relay_down_node_row_without_membership_state_still_joins(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A read that cannot tell falls through: the node's relay-down fallback rows
    carry no ``membership_state``, so the join runs exactly as before."""
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    _mint_token(token)
    network_store.save(_inviter_record(), None)
    fallback = _status_row()
    fallback.pop("membership_state")
    fallback.pop("membership", None)
    record = _record()
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(
        outputs=HAPPY_OUTPUTS + [("network status", _status_output(fallback))]
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "connected", payload
    join = {row["step"]: row for row in payload["steps"]}["join"]
    assert "already an active member" not in join["detail"]
    assert [c for c in transport.calls if c[0] == "copy"]
    join_calls = [c for c in transport.calls if c[0] == "run" and "network join" in " ".join(c[1])]
    assert len(join_calls) == 1


def test_a_fresh_admission_is_unchanged_by_the_entry_check(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No inviter record at all: nothing to confirm with — the join runs exactly
    as before and its success receipt keeps its shape."""
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    _mint_token(token)
    # Deliberately NO ``network_store.save(...)``: the fresh device's shape.
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
    join = {row["step"]: row for row in payload["steps"]}["join"]
    assert join["detail"] == (
        "joined damian-mesh as d_node; the code compare passed on the inviting side"
    )
    assert join["data"]["device_id"] == "d_node"
    assert join["data"]["sas"] == "123456"
    assert join["data"]["network_id"] == "n_1"
    assert join["data"]["identity_minted"] is False
    assert [c for c in transport.calls if c[0] == "copy"]
    join_calls = [c for c in transport.calls if c[0] == "run" and "network join" in " ".join(c[1])]
    assert len(join_calls) == 1


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


# ---------------------------------------------------------------------------
# The install step's refreshed resolve (drill finding F3, 2026-10-04)
# ---------------------------------------------------------------------------

#: The drill's node: a build present (v0.67.3), no updater script — the shape
#: of the two failed receipts (s6/s7), which takes the uv-tool-reinstall arm.
STALE_NODE_PRE_READ = PRE_READ_OK.replace("lop_version=v0.63.2", "lop_version=v0.67.3").replace(
    "lop_update=yes", "lop_update=no"
)

#: uv's resolver-class text, wrapped the way a narrow terminal renders it (the
#: drill's node receipt carried it as one line; the flattening must survive both).
UV_INDEX_HIDDEN = (
    "  × No solution found when resolving dependencies:\n"
    "  ╰─▶ Because there is no version of local-operator==0.67.4 and you require\n"
    "      local-operator==0.67.4, we can conclude that your requirements are\n"
    "      unsatisfiable.\n"
)


def _install_step_outputs(
    pre_read: str, refresh: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """HAPPY_OUTPUTS, with this section's pre-read/install rows swapped in.

    The first-match-wins fake gets the refresh row first: a resolve WITH the
    refreshed index succeeds, while a resolve without it hits the stale-cache
    failure the drill measured. Drop `--refresh` from the command and the cells
    that use this helper fail.
    """
    overrides = [
        ("uname", {"stdout": pre_read}),
        ("--refresh", refresh),
        ("tool install", {"rc": 1, "stderr": UV_INDEX_HIDDEN}),
        ("lop --version", {"stdout": "v0.67.4\n"}),
    ]
    skip = {"uname", "lop-update", "lop --version"}
    return [*overrides, *(row for row in HAPPY_OUTPUTS if row[0] not in skip)]


def test_a_cached_index_cannot_hide_the_build_from_the_refreshed_resolve(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Drill finding F3: two runs failed at `install` — raw "no version of
    local-operator==0.67.4 … unsatisfiable", 6 minutes and ~1 h after the
    release was published — because the node's uv served a CACHED simple-index
    response; the same node's curl showed the version present, and `--refresh`
    cured it by hand.

    The fake models exactly that boundary (see _install_step_outputs): the
    refreshed resolve succeeds, the cached spelling fails — so this cell fails
    if the install command loses `--refresh`.
    """
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record(build="0.67.4")
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    transport = FakeTransport(
        outputs=_install_step_outputs(STALE_NODE_PRE_READ, {"stdout": "installed\n"})
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "connected", payload
    install = next(row for row in payload["steps"] if row["step"] == "install")
    assert install["ok"] is True
    assert install["data"]["method"] == "uv-tool-reinstall"
    commands = [" ".join(call[1]) for call in transport.calls if call[0] == "run"]
    assert any(
        "uv tool install --force --refresh local-operator==0.67.4" in command
        for command in commands
    ), commands


def test_a_fresh_machine_installs_with_the_index_refreshed_too(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The no-build arm carries the refresh as well — a machine that never ran
    our tool can still hold a cached index response from other work — and there
    is nothing to replace there, so no `--force`."""
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record(build="0.67.4")
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    fresh_pre_read = (
        PRE_READ_OK.replace("lop=yes", "lop=no")
        .replace("lop_path=/usr/local/bin/lop\n", "")
        .replace("lop_version=v0.63.2", "lop_version=")
        .replace("lop_update=yes", "lop_update=no")
    )
    transport = FakeTransport(
        outputs=_install_step_outputs(fresh_pre_read, {"stdout": "installed\n"})
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "connected", payload
    commands = [" ".join(call[1]) for call in transport.calls if call[0] == "run"]
    install_commands = [command for command in commands if "uv tool install" in command]
    assert any(
        "uv tool install --refresh local-operator==0.67.4" in command
        for command in install_commands
    ), install_commands
    assert all("--force" not in command for command in install_commands), install_commands


def test_the_install_refusal_names_the_cached_index_as_the_likely_cause(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Drill finding F3, the copy half: the drill read raw uv text as the WHOLE
    failure — "no version … unsatisfiable" — with no cause and no remedy. The
    resolver-class sentence now leads with the action, names the cached index as
    the likely cause, hedges for genuine absence (so the reader can tell "retry
    may cure" from "the version may not exist yet"), keeps one name for the
    artefact, and leaves uv's words flattened and glyph-stripped at the end —
    the retry runs exactly the command the drill cured by hand, without naming
    a terminal command (§2.9).

    The branch split is part of the pin: a non-resolver failure keeps its
    original surface.
    """
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record(build="0.67.4")
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)

    resolver = FakeTransport(
        outputs=[
            ("uname", {"stdout": STALE_NODE_PRE_READ}),
            ("tool install", {"rc": 1, "stderr": UV_INDEX_HIDDEN}),
        ]
    )
    first = onboard.execute_approval(
        "ap_aaaa1111",
        transport=resolver,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert first["state"] == "failed"
    failing = first["steps"][-1]
    assert failing["step"] == "install" and failing["ok"] is False
    detail = failing["detail"]
    # D2: the action leads; a clipping surface keeps cause+remedy, not uv's words.
    assert detail.startswith("the approved build could not be installed. Retry the install")
    # D1: the cache is named as the likely cause and the escape keeps a genuine
    # absence from turning the retry into a loop.
    assert "a cached index is the likely cause" in detail
    assert "it is not on the index" in detail
    assert "ask Local Operator to file a fresh request with the corrected tag" in detail
    # D4: the window matches the drill's own clock (6 minutes, then ~1 h).
    assert "shortly before the run" in detail
    assert "minutes earlier" not in detail
    # D3: one name for the artefact ("the approved build <tag>") in the prose.
    assert "The machine's uv could not see the approved build 0.67.4" in detail
    # N1/N2: uv's words ride flattened and glyph-stripped, never with a doubled stop.
    assert "Because there is no version of local-operator==0.67.4" in detail
    assert "unsatisfiable" in detail
    assert "\n" not in detail and "×" not in detail and "╰─▶" not in detail
    assert ".)." not in detail
    assert detail.endswith(".)")
    assert failing["data"]["method"] == "uv-tool-reinstall"

    generic = FakeTransport(
        outputs=[
            ("uname", {"stdout": STALE_NODE_PRE_READ}),
            ("tool install", {"rc": 1, "stderr": "network down\n"}),
        ]
    )
    second = onboard.execute_approval(
        "ap_aaaa1111",
        transport=generic,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )
    assert second["steps"][-1]["detail"] == (
        "the approved build could not be installed: network down"
    )


def test_a_long_resolver_message_keeps_the_head_and_drops_terminal_glyphs(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Design round 1 (N2/N3), pinned at the excerpt's own boundary: uv's
    box-drawing furniture ("×", "╰─▶") is terminal-only and can render as tofu
    in a UI sheet, and when the flattened words run long the excerpt keeps the
    HEAD — "No solution found when resolving dependencies" is the part that
    names the resolver — not the last 300 characters."""
    _install_fakes(monkeypatch)
    token = tmp_path / "invite.token"
    token.write_text("token-bytes", encoding="utf-8")
    record = _record(build="0.67.4")
    fake = FakeApprovals(record)
    monkeypatch.setattr(onboard_approvals, "_module", lambda: fake)
    long_uv = (
        "  × No solution found when resolving dependencies:\n"
        "  ╰─▶ Because there is no version of local-operator==0.67.4 and you require\n"
        "      local-operator==0.67.4, we can conclude that your requirements are\n"
        + ("      adding solver context " * 20)
        + "      unsatisfiable.\n"
    )
    transport = FakeTransport(
        outputs=[
            ("uname", {"stdout": STALE_NODE_PRE_READ}),
            ("tool install", {"rc": 1, "stderr": long_uv}),
        ]
    )

    payload = onboard.execute_approval(
        "ap_aaaa1111",
        transport=transport,
        resolve=lambda ref: onboard.ResolvedCredential(kind="file", label=ref["ref"]),
        local_cli=["lop"],
        run_local=_happy_run_local(token),
    )

    assert payload["state"] == "failed"
    detail = payload["steps"][-1]["detail"]
    assert "No solution found when resolving dependencies" in detail  # head kept
    assert "…" in detail  # the cut is marked
    assert "unsatisfiable" not in detail  # the tail really was dropped
    assert "×" not in detail and "╰─▶" not in detail
    assert detail.endswith(".)")
    assert ".)." not in detail


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
