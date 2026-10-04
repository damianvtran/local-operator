"""``lop network approvals``: the frozen ``--json`` shapes and the refusal copy.

The store's own lifecycle is pinned in ``test_approvals_store``; this file pins
the CLI CONTRACT the desktop and the agent both read — the exact keys of each
verb's payload (§3.5), the exit/refusal shape, the sentence a host with no
signing surface gets (which must name a product action and no terminal command,
§2.9), and that ``run`` refuses truthfully while this build ships no runner
without touching the record.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from tests.unit.network.test_approvals_store import _make_key

REQUEST_ARGS: dict[str, Any] = {
    "host": "99.79.190.164",
    "user": "ec2-user",
    "name": "cloud-node-1",
    "device": "",
    "fingerprint": "",
    "host_key_fp": "SHA256:abc",
    "network": "",
    "role": "drive",
    "credential_ref": "",
    "expires": 3600.0,
    "no_unattended": False,
    "grant": [],
    "session": "s1",
    "request_id": "",
    "json": True,
}


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated config root, with "no installed anchor" pinned.

    The pin matters as much here as in the store tests: without it a developer's
    real anchor selects a different branch and the refusal cells would report the
    machine rather than the code.
    """
    from local_operator.operator import trust

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    def absent(uid: int | str | None = None) -> Any:
        return trust.AnchorLoad(
            anchor=None,
            path=trust.anchor_path(uid),
            root_owned=False,
            reason="pinned absent by the test",
            exists=False,
        )

    monkeypatch.setattr(trust, "load_anchor", absent)
    return config_dir


def _request(root: Path, capsys: pytest.CaptureFixture[str], **overrides: Any) -> dict[str, Any]:
    fields = {**REQUEST_ARGS, **overrides}
    rc = net_cli.main(Namespace(network_command="approvals", approvals_command="request", **fields))
    out = capsys.readouterr().out
    assert rc == 0, out
    return json.loads(out)


# ---------------------------------------------------------------------------
# The frozen shapes
# ---------------------------------------------------------------------------


def test_request_answers_the_frozen_card_shape(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_key(root)
    payload = _request(root, capsys)
    assert set(payload) >= {"approval_id", "state", "device", "what", "expires_at"}
    assert payload["state"] == "requested"
    assert payload["approval_id"].startswith("ap_")
    # The card's material: the where-block and the scope list.
    assert payload["device"]["host"] == "99.79.190.164"
    assert payload["device"]["name"] == "cloud-node-1"
    assert payload["what"]["install"] is True
    assert payload["what"]["unattended"] is True
    assert payload["what"]["grant"] == ["approve"]
    # The anchor trio is derived, never caller-supplied (F4b).
    anchor = payload["what"]["anchor"]
    assert set(anchor) == {"key_id", "spki_fp", "statement_digest"}
    assert anchor["statement_digest"].startswith("sha256:")
    # And the record the badge lists carries the same keys.
    rc = net_cli.main(Namespace(network_command="approvals", approvals_command="list", json=True))
    assert rc == 0
    listed = json.loads(capsys.readouterr().out)
    row = listed["approvals"][0]
    assert set(row) >= {"approval_id", "state", "device", "what", "requested_by", "expires_at"}
    assert row["approval_id"] == payload["approval_id"]


def test_a_request_without_a_host_key_fp_observes_and_records_one(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drill finding, 2026-10-03: a card filed without the host key cannot run —
    step zero halts on the missing fingerprint, and its auto-refiled replacement
    was minted with the same hole. The request OBSERVES the key through the
    credential-free handshake (the whole of pre-approval contact) and records it
    on the card."""
    from local_operator.network import onboard

    _make_key(root)
    calls: list[tuple[str, int, str]] = []

    def recording_probe(host: str, *, port: int = 22, user: str = "", **_: Any) -> Any:
        calls.append((host, port, user))
        return onboard.Probe(
            ok=True,
            host=host,
            port=port,
            user=user,
            banner="SSH-2.0-OpenSSH_9.2",
            host_key_fp="SHA256:observed-at-filing",
            at=0.0,
        )

    monkeypatch.setattr(onboard, "probe", recording_probe)
    payload = _request(root, capsys, host_key_fp="")
    assert calls == [("99.79.190.164", 22, "ec2-user")]
    assert payload["device"]["host_key_fp"] == "SHA256:observed-at-filing"


def test_a_request_whose_host_key_cannot_be_observed_is_refused(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No observation and no flag ⇒ REFUSE, not a card that cannot run: the
    acceptance is that a card filed by the flow can always execute."""
    from local_operator.network import onboard

    _make_key(root)

    def dead_probe(host: str, *, port: int = 22, user: str = "", **_: Any) -> Any:
        return onboard.Probe(
            ok=False,
            host=host,
            port=port,
            user=user,
            detail=f"nothing accepted a connection at {host}:{port} (ConnectionRefusedError)",
            at=0.0,
        )

    monkeypatch.setattr(onboard, "probe", dead_probe)
    fields = {**REQUEST_ARGS, "host_key_fp": ""}
    rc = net_cli.main(Namespace(network_command="approvals", approvals_command="request", **fields))
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["code"] == "host_key_unobserved"
    assert "could not be observed" in payload["message"]
    # D7 (design round 1): the copy says NOTHING WAS FILED — the refusal is not
    # a card that later halts — and trades the un-introduced "a key read"
    # jargon for the observable condition.
    assert "nothing was filed" in payload["message"]
    assert "answers a key read" not in payload["message"]


def test_a_given_host_key_fp_never_probes(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag is the fallback path and stays hermetic: a supplied fingerprint
    is used verbatim, with no network read."""
    from local_operator.network import onboard

    _make_key(root)

    def forbidden_probe(*_: Any, **__: Any) -> Any:
        raise AssertionError("a supplied --host-key-fp must not be re-probed")

    monkeypatch.setattr(onboard, "probe", forbidden_probe)
    payload = _request(root, capsys)
    assert payload["device"]["host_key_fp"] == "SHA256:abc"


def test_the_human_list_cues_states_and_separates_records(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The prose half of `list` (design review round 1, D3/D4/D7).

    Two records must not read as one block: a blank row sits between them. The
    state word carries its family glyph, and the scope list spells the two
    installs apart ("install build" / "install anchor") with the grants as one
    `grant: approve` item instead of a repeated bare verb.
    """
    _make_key(root)
    _request(root, capsys)
    _request(root, capsys, host="192.0.2.9", user="devon", name="devon-laptop")
    rc = net_cli.main(Namespace(network_command="approvals", approvals_command="list", json=False))
    out = capsys.readouterr().out
    assert rc == 0, out
    # D4 — the state cue, from the notice family's own glyph set.
    assert " — · requested" in out, out
    # D3 — a blank row between the two blocks.
    lines = out.splitlines()
    headers = [index for index, line in enumerate(lines) if line.startswith("approval ")]
    assert len(headers) == 2, lines
    assert lines[headers[1] - 1] == "", lines
    # D7 — the scope wording.
    assert "install build, install anchor," in out, out
    assert "grant: approve" in out, out
    assert "install, install" not in out, out


def test_state_word_carries_its_glyph_and_an_unknown_state_stays_plain() -> None:
    from local_operator.network.cli import _approval_lines

    base: dict[str, Any] = {"approval_id": "ap_x", "state": "approved", "what": {}}
    assert _approval_lines(base)[0] == "approval ap_x — ✓ approved"
    # An unknown (future) state must not gain a wrong cue.
    plain = _approval_lines({**base, "state": "queued_future"})[0]
    assert plain == "approval ap_x — queued_future", plain


def test_every_store_state_has_a_glyph_cue() -> None:
    from local_operator.network import approvals as store
    from local_operator.network.cli import _STATE_GLYPHS

    assert set(_STATE_GLYPHS) == set(store.STATES)


def test_approve_signs_with_the_local_key_and_its_key_id_round_trips(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    anchor = _make_key(root)
    filed = _request(root, capsys)
    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="approve",
            approval=filed["approval_id"],
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    payload = json.loads(out)
    assert payload["state"] == "approved"
    assert payload["signature"]["key_id"] == anchor.key_id


def test_deny_answers_the_frozen_shape_and_a_second_decision_refuses(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_key(root)
    filed = _request(root, capsys)
    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="deny",
            approval=filed["approval_id"],
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    payload = json.loads(out)
    assert set(payload) >= {"approval_id", "state", "signature"}
    assert payload["state"] == "denied"
    assert payload["signature"] == {"key_id": ""}

    # The write-once rule reaches the CLI as its own code, not a traceback.
    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="deny",
            approval=filed["approval_id"],
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    refused = json.loads(out)
    assert refused["ok"] is False
    assert refused["code"] == "approval_decision_conflict"


def test_withdraw_settles_the_filers_own_request_without_the_operator(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The honest primitive through the CLI: the filer withdraws, the frozen
    shape answers ``withdrawn``, and the human half reads self-settled — never
    as a decline (the operator was not asked) and never as a lapse."""
    _make_key(root)
    filed = _request(root, capsys)
    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="withdraw",
            approval=filed["approval_id"],
            session="s1",
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    payload = json.loads(out)
    assert payload["state"] == "withdrawn"
    assert set(payload) >= {"ok", "approval_id", "state"}

    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="show",
            approval=filed["approval_id"],
            json=False,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert " — ✗ withdrawn" in out, out
    assert "withdrawn by: cli s1" in out, out
    assert "self-settled by the filer; the operator was not asked" in out, out
    assert "denied" not in out, out
    assert "declined" not in out, out


def test_a_foreign_surface_cannot_withdraw_via_the_cli(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_key(root)
    filed = _request(root, capsys)
    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="withdraw",
            approval=filed["approval_id"],
            session="s9",
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    refused = json.loads(out)
    assert refused["ok"] is False
    assert refused["code"] == "approval_requester_mismatch"
    assert "nothing was written" in refused["message"]


def test_withdraw_refuses_once_the_operator_has_answered(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_key(root)
    filed = _request(root, capsys)
    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="approve",
            approval=filed["approval_id"],
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="withdraw",
            approval=filed["approval_id"],
            session="s1",
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    refused = json.loads(out)
    assert refused["code"] == "approval_withdraw_conflict"


# ---------------------------------------------------------------------------
# Refusals: the copy rule (§2.9) and the truth of `run`
# ---------------------------------------------------------------------------


def _assert_sets_up_without_a_command(sentence: str) -> None:
    assert "ask Local Operator to set up operator authority" in sentence, sentence
    assert "`" not in sentence, sentence
    assert "lop operator" not in sentence, sentence


def test_request_without_a_local_anchor_names_the_setup(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = net_cli.main(
        Namespace(network_command="approvals", approvals_command="request", **REQUEST_ARGS)
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    refused = json.loads(out)
    assert refused["code"] == "approval_anchor_unavailable"
    _assert_sets_up_without_a_command(refused["message"])


def test_approve_headless_refuses_with_the_setup_sentence(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No loadable operator key: refuse BEFORE anything is written.

    The key is *made* first (the record needs an anchor trio to exist at all),
    then the signer lookup is pinned to None — the state a headless host is in,
    expressed at the seam rather than by deleting files whose names are the
    backend's business.
    """
    _make_key(root)
    filed = _request(root, capsys)
    import local_operator.operator.sign as sign_mod

    def no_signer(**kwargs: Any) -> Any:
        return None

    monkeypatch.setattr(sign_mod, "load_signer", no_signer)

    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="approve",
            approval=filed["approval_id"],
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    refused = json.loads(out)
    assert refused["code"] == "approval_signing_unavailable"
    _assert_sets_up_without_a_command(refused["message"])
    # And nothing was decided.
    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="show",
            approval=filed["approval_id"],
            json=True,
        )
    )
    record = json.loads(capsys.readouterr().out)["approval"]
    assert record["state"] == "requested"
    assert record["signature"] is None


def test_run_refuses_truthfully_and_leaves_the_record_approved(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pinned sentence, and the property that makes it honest: no state change.

    THE SEAM IS STUBBED ABSENT (slice (b), the rebase fold): ``onboard.
    step_runner`` now exists in this tree, and this cell's home is a build
    WITHOUT the runner — the constant's own note asks for a cell that keeps
    the sentence from rotting. The filled seam has its own cell next door.
    """
    monkeypatch.setattr(net_cli, "_approval_step_runner", lambda record: None)
    _make_key(root)
    filed = _request(root, capsys)
    net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="approve",
            approval=filed["approval_id"],
            json=True,
        )
    )
    capsys.readouterr()

    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="run",
            approval=filed["approval_id"],
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    refused = json.loads(out)
    assert refused["code"] == "approval_runner_missing"
    assert refused["message"] == net_cli.APPROVAL_RUNNER_MISSING_SENTENCE
    assert "`" not in refused["message"]

    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="show",
            approval=filed["approval_id"],
            json=True,
        )
    )
    record = json.loads(capsys.readouterr().out)["approval"]
    assert record["state"] == "approved", "a refused run must not leave the record mid-flight"
    assert record["receipts"] == []


def test_the_run_verb_finds_the_step_runner_and_hands_it_the_record(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slice (b) filled the seam; the verb hands the runner the record's id.

    The runner itself is stubbed here — the machine it drives has its own
    fake-transport cells in ``test_onboard.py`` — so this cell pins the WIRING
    (the lookup, the call shape, the emitted payload), which is exactly what a
    rebase can silently drop.
    """
    _make_key(root)
    filed = _request(root, capsys)
    net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="approve",
            approval=filed["approval_id"],
            json=True,
        )
    )
    capsys.readouterr()

    seen: list[dict[str, Any]] = []

    def fake_execute(approval_id: str, **kwargs: Any) -> dict[str, Any]:
        seen.append({"approval_id": approval_id, **kwargs})
        return {
            "ok": True,
            "approval_id": approval_id,
            "state": "connected",
            "steps": [],
            "next": "done",
        }

    monkeypatch.setattr("local_operator.network.onboard.execute_approval", fake_execute)

    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="run",
            approval=filed["approval_id"],
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert seen and seen[0]["approval_id"] == filed["approval_id"]
    assert json.loads(out)["state"] == "connected"


def test_run_refuses_an_unapproved_record_with_its_own_sentence(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_key(root)
    filed = _request(root, capsys)
    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="run",
            approval=filed["approval_id"],
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    refused = json.loads(out)
    assert refused["code"] == "approval_not_runnable"
    assert "requested" in refused["message"]


def test_run_supersedes_a_connecting_card_whose_lease_is_stale(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drill finding, 2026-10-04 (F1): a runner killed mid-flight wedged the card
    in ``connecting`` and ``run`` refused every retry. A lease naming a process
    that is gone re-enters (the runner supersedes it below); a lease still alive
    keeps refusing — pinned in the sibling cell — so a live runner can never be
    double-entered."""
    from local_operator.network import approvals as A

    _make_key(root)
    filed = _request(root, capsys)
    net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="approve",
            approval=filed["approval_id"],
            json=True,
        )
    )
    capsys.readouterr()
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()  # the runner that died mid-flight, as a real reaped pid
    A.begin_run(filed["approval_id"], run_id="run_dead", runner_pid=child.pid, root=root)

    seen: list[dict[str, Any]] = []

    def fake_execute(approval_id: str, **kwargs: Any) -> dict[str, Any]:
        seen.append({"approval_id": approval_id, **kwargs})
        return {"ok": True, "approval_id": approval_id, "state": "connected", "steps": []}

    monkeypatch.setattr("local_operator.network.onboard.execute_approval", fake_execute)
    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="run",
            approval=filed["approval_id"],
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert seen and seen[0]["approval_id"] == filed["approval_id"]


def test_run_refuses_a_connecting_card_with_a_live_lease(
    root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard, through the verb: a run that is still reporting is never
    overtaken — no runner call, one sentence naming the in-flight run."""
    from local_operator.network import approvals as A

    _make_key(root)
    filed = _request(root, capsys)
    net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="approve",
            approval=filed["approval_id"],
            json=True,
        )
    )
    capsys.readouterr()
    A.begin_run(filed["approval_id"], run_id="run_live", runner_pid=os.getpid(), root=root)

    called: list[str] = []
    monkeypatch.setattr(
        "local_operator.network.onboard.execute_approval",
        lambda approval_id, **kwargs: called.append(approval_id) or {"ok": True},
    )
    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="run",
            approval=filed["approval_id"],
            json=True,
        )
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    refused = json.loads(out)
    assert refused["code"] == "approval_run_in_flight"
    assert "in flight" in refused["message"]
    assert called == [], "a live run must never be double-entered"


def test_unknown_id_is_a_not_found_class_refusal_with_its_own_code(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = net_cli.main(
        Namespace(
            network_command="approvals", approvals_command="show", approval="ap_nope", json=True
        )
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    assert json.loads(out)["code"] == "unknown_approval"


def test_a_request_id_retry_is_idempotent_from_the_cli(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA round 1, Q1: the flag's own promise, made reachable through the CLI.

    ``--request-id`` says "resend the SAME id with the same payload to retry
    without duplicating", and this verb re-derives ``created_at``/``expires_at``
    on every invocation — so before the window fallback the second call came back
    ``approval_request_conflict`` and the only recovery (drop the id) could file a
    second card. The second call here is the SAME command run again: it must
    return the FIRST record, by id. The intent half still bites, so the cell
    carries it too — a retry that changed a field is a new request.
    """
    _make_key(root)
    request_id = "req_9f3ac1e0b7d2ab"
    first = _request(root, capsys, request_id=request_id)
    second = _request(root, capsys, request_id=request_id)
    assert second["approval_id"] == first["approval_id"], (first, second)
    assert second["state"] == first["state"], second

    rc = net_cli.main(
        Namespace(
            network_command="approvals",
            approvals_command="request",
            **{**REQUEST_ARGS, "request_id": request_id, "host": "other.example"},
        )
    )
    out = capsys.readouterr().out
    assert rc == 1, out
    assert json.loads(out)["code"] == "approval_request_conflict"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_the_verb_family_is_registered_and_a_bare_verb_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "approvals" in net_cli._ACTIONS
    assert "approvals" in net_cli._HANDLERS
    rc = net_cli.main(Namespace(network_command="approvals", approvals_command=None))
    err = capsys.readouterr().err
    assert rc == 2
    for verb in ("list", "show", "request", "approve", "deny", "withdraw", "run"):
        assert verb in err, err
