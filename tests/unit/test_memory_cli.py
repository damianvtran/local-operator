"""Human commands exercise the shipped CLI and the shared slash runner, synthetic config only.

Round-1 design/UX findings pinned here: D1/U2 (a printed ``scope/id`` is accepted back),
D2 (conflicts are listed with a resolution path), U1 (default view spans every authorised
scope and labels captured vs stated), U3/D3 (field-flag edits; quoted-JSON error).
"""

import json
import os
import subprocess
import sys
import time

import pytest

from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore
from local_operator.memory_cli import _date, render_human, run


def human(text, root, identity, session_dir=None, prefix="/memory"):
    out, failed = run(text, config=root, identity=identity, session_dir=session_dir, prefix=prefix)
    return render_human(out, failed, prefix), failed


@pytest.fixture
def rig(tmp_path):
    root = tmp_path / "home" / ".local-operator"
    identity = MemoryIdentity("cli-session", "", "")
    session_dir = root / "sessions" / "cli-session"
    store = MemoryStore(root, identity, session_dir, origin="cli-device")
    return root, identity, session_dir, store


def test_shipped_cli_inspection_correction_and_explicit_forget(tmp_path, monkeypatch):
    root = tmp_path / "home" / ".local-operator"
    home = root.parent
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    store = MemoryStore(root, MemoryIdentity("cli-session", "", "user"), origin="cli-device")
    item = store.put("global", {"title": "CLI proof", "body": "CLI evidence docs/proof.md"})
    ref = "global/" + item["id"]

    def cli(*args, expected=0):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
        response = subprocess.run(
            [sys.executable, "-m", "local_operator.cli", "memory", *args],
            cwd=home,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert response.returncode == expected, (response.returncode, response.stderr)
        return response.stdout

    # Default is the human view; --json is the machine one (U6/D13).
    assert "CLI proof" in cli("list") and "{" not in cli("list")
    assert json.loads(cli("list", "--json"))["items"][0]["id"] == item["id"]
    # Copy-pasted scope/id works for show (D1/U2) and the bare id keeps working.
    assert "CLI evidence" in cli("show", ref) and "CLI evidence" in cli("show", item["id"])
    assert "concurrently" in cli("edit", ref, "--revision", "stale", "--body", "x", expected=1)
    # Unquoted multi-word search needs no quotes (U4).
    assert "CLI proof" in cli("search", "CLI", "proof")
    out = cli("edit", ref, "--revision", item["version"], "--body", "New proof")
    assert out.startswith("Corrected global/") and "CLI proof" in out
    current = store.show("global", item["id"])
    assert "Nothing forgotten" in cli("forget", ref, "--revision", current["version"], expected=1)
    assert store.show("global", item["id"])["status"] == "active"
    forgot = cli("forget", ref, "--revision", current["version"], "--yes")
    assert forgot.startswith("Forgotten global/") and "cannot be undone" in forgot
    assert "No matching" in cli("search", "proof")
    assert "Trusted-peer sync" in cli("status")


def test_bad_human_input_and_disabled_store_inspection(tmp_path):
    identity = MemoryIdentity()
    text, failed = human("forget global/x --revision v --yes", tmp_path, identity)
    assert failed and not (tmp_path / "memory").exists()
    empty, failed = human("list", tmp_path, identity)
    assert not failed and "Nothing retained yet" in empty and "Searched: global" in empty
    invalid, failed = human("--scope other list", tmp_path, identity)
    assert failed and "Invalid memory command" in invalid


def test_d1_printed_reference_round_trips_and_prefix_is_unique_and_scoped(rig, monkeypatch):
    import local_operator.memory as memory

    root, identity, session_dir, store = rig
    # Fixed ids sharing a long prefix: UUIDv7 ids minted within one millisecond do, and a
    # clock-dependent assertion would be flaky.
    ids = iter(
        [
            "0aaaaaaa-1111-7000-8000-000000000001",  # entity a
            "0aaaaaaa-1111-7000-8000-0000000000a1",  # its op
            "0aaaaaaa-1111-7000-8000-000000000002",  # entity b
            "0aaaaaaa-1111-7000-8000-0000000000b2",  # its op
        ]
    )
    monkeypatch.setattr(memory, "_id", lambda: next(ids))
    a = store.put("global", {"title": "Alpha", "body": "alpha docs/a.md"})
    b = store.put("session", {"title": "Beta", "body": "beta docs/b.md"})
    listed, _ = human("list", root, identity, session_dir)
    # Every printed row is a paste-ready reference that show accepts as-is.
    for ref in (f"global/{a['id']}", f"session/{b['id']}"):
        assert ref in listed
        shown, failed = human(f"show {ref}", root, identity, session_dir)
        assert not failed and ref in shown
    ambiguous, failed = human("show 0aaaaaaa-1111", root, identity, session_dir)
    assert failed and "matches 2 memories" in ambiguous
    prefixed, failed = human(
        "show 0aaaaaaa-1111-7000-8000-00000000000", root, identity, session_dir
    )
    assert failed, "still ambiguous: both end in 1/2 after this prefix"
    unique, failed = human("show 0aaaaaaa-1111-7000-8000-000000000001", root, identity, session_dir)
    assert not failed and "Alpha" in unique
    scoped, failed = human(
        "show 0aaaaaaa-1111-7000-8000-00000000000 --scope global", root, identity
    )
    assert not failed and "Alpha" in scoped, "a scope narrows an otherwise ambiguous prefix"
    short, failed = human("show 01a1", root, identity, session_dir)
    assert failed and "not found" in short
    # A prefix never reaches an owner the identity is not authorised for.
    other = MemoryIdentity("other-session")
    hidden, failed = human(f"show session/{b['id']}", root, other, session_dir)
    assert failed and "not found" in hidden


def test_d2_conflicts_are_listed_with_age_origin_and_a_resolution_path(rig):
    root, identity, session_dir, store = rig
    item = store.put("global", {"title": "Port", "body": "Port is 80 docs/p.md"})
    base = store.show("global", item["id"])
    # A concurrent edit from another device, delivered by sync.
    peer = MemoryStore(root, MemoryIdentity(), origin="peer-device")
    other = peer.prepare_fact({"title": "Port (peer)", "body": "Port is 8080 docs/p.md"})
    import hashlib

    from local_operator.memory import _dump

    op = {
        "v": 1,
        "op_id": "peer-op-1",
        "entity_id": item["id"],
        "origin_device": "peer-device",
        "seq": 1,
        "hlc": [int(time.time() * 1000) + 5, 0, "peer-device"],
        "parents": [],
        "kind": "put",
        "scope": "global",
        "owner_key": "global",
        "fact": other,
        "body_sha": hashlib.sha256(_dump(other).encode()).hexdigest(),
    }
    store.apply_ops([op], origins={"peer-device"}, contiguous=False)
    shown, failed = human(f"show global/{item['id']}", root, identity, session_dir)
    assert not failed
    assert "Conflicts: 1 other concurrent edit(s) kept, not merged" in shown
    assert base["version"] in shown
    assert "from cli-device" in shown and "--resolve" in shown
    # Resolving parents every head, so the conflict is gone afterwards.
    head = store.show("global", item["id"])
    fixed, failed = human(
        f"edit global/{item['id']} --revision {head['version']} --resolve --body 'Port is 8080'",
        root,
        identity,
        session_dir,
    )
    assert not failed and fixed.startswith("Corrected")
    assert store.show("global", item["id"])["conflicts"] == []


def test_u1_default_list_spans_authorised_scopes_and_labels_captured_vs_stated(rig):
    root, identity, session_dir, store = rig
    store.put("global", {"title": "Stated fact", "body": "stated docs/s.md"})
    store.put("session", {"title": "Captured locator", "body": "docs/c.md", "status": "candidate"})
    listed, failed = human("list", root, identity, session_dir)
    assert not failed
    assert "Searched: global, session." in listed
    assert "[captured]" in listed and "[stated]" in listed
    searched, _ = human("search locator", root, identity, session_dir)
    assert "Captured locator" in searched and "[captured]" in searched
    # No session bound: the empty state names what was searched and how to add a session.
    cold, _ = human("list", root, MemoryIdentity(), None)
    assert "Searched: global." in cold and "--session ID" in cold
    unbound, failed = human("list --scope session", root, MemoryIdentity(), None)
    assert failed and "--session ID" in unbound and "No authorized session" not in unbound


def test_u3_field_flag_edit_keeps_other_fields_and_json_errors_are_clear(rig):
    root, identity, session_dir, store = rig
    item = store.put(
        "global",
        {"title": "Host", "body": "Host is a docs/h.md", "tags": ["infra"], "aliases": ["hst"]},
    )
    ref = f"global/{item['id']}"
    shown, _ = human(f"show {ref}", root, identity, session_dir)
    # show prints a copy-pasteable edit command including the revision just read.
    assert f"--revision {item['version']}" in shown and f"edit {ref}" in shown
    done, failed = human(
        f"edit {ref} --revision {item['version']} --title 'Host v2' --verified",
        root,
        identity,
        session_dir,
    )
    assert not failed and done.startswith("Corrected")
    fact = store.show("global", item["id"])["fact"]
    assert fact["title"] == "Host v2" and fact["body"] == "Host is a docs/h.md"
    assert fact["tags"] == ["infra"] and fact["aliases"] == ["hst"]
    assert fact["verified_at"] is not None and fact["source"]["author"] == "user"
    version = store.show("global", item["id"])["version"]
    for bad in (
        f"edit {ref} --revision {version} --replace {{a:1}}",
        f"edit {ref} --revision {version} --replace '{{a:1}}'",
    ):
        message, failed = human(bad, root, identity, session_dir)
        assert failed and "invalid JSON" in message and "store unavailable" not in message
    # Nothing to change is a clear instruction, not a silent no-op.
    nothing, failed = human(f"edit {ref} --revision {version}", root, identity, session_dir)
    assert failed and "Nothing to change" in nothing


def test_d4_forget_confirmation_names_the_item_and_prints_the_paste_ready_command(rig):
    root, identity, session_dir, store = rig
    item = store.put("global", {"title": "Delete me", "body": "docs/d.md"})
    ref = f"global/{item['id']}"
    prompt, failed = human(f"forget {ref} --revision {item['version']}", root, identity)
    assert failed and "Delete me" in prompt
    assert f"/memory forget {ref} --revision {item['version']} --yes" in prompt
    assert store.show("global", item["id"])["status"] == "active"
    lop_prompt, _ = human(
        f"forget {ref} --revision {item['version']}", root, identity, prefix="lop memory"
    )
    assert f"lop memory forget {ref}" in lop_prompt


def test_hints_for_non_global_scopes_carry_the_binding_flags_on_the_cli(rig):
    root, identity, session_dir, store = rig
    item = store.put("session", {"title": "S", "body": "docs/s.md"})
    cli_view, _ = human(
        "list --session cli-session", root, identity, session_dir, prefix="lop memory"
    )
    # execute() derives the flag text from parsed args, so a pasted command stays valid.
    assert f"lop memory show session/{item['id']} --session cli-session" in cli_view


def test_status_reports_effective_state_and_names_the_config_key(rig, tmp_path):
    root, identity, session_dir, store = rig
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yml").write_text(json.dumps({"values": {"memory": {"enabled": False}}}))
    status, failed = human("status", root, identity, session_dir)
    assert not failed
    assert "Agent memory: off (memory.enabled)" in status
    assert "Trusted-peer sync: off" in status and "{" not in status
    listed, _ = human("list", root, identity, session_dir)
    assert "Agent memory is OFF (memory.enabled=false)" in listed


def test_human_dates_and_next_step_shown_once_on_list_only():
    now = 1_800_000_000.0
    assert _date(now - 2 * 86400, now=now).endswith("(2d ago)")
    assert _date(now + 3 * 3600, now=now).endswith("(in 3h)")
    assert _date(now + 5, now=now).endswith("(in <1m)")  # D9: never "in just"
    assert _date(None, "never") == "never"


def test_empty_list_has_no_destructive_next_hint(rig):
    root, identity, session_dir, store = rig
    text, _ = human("list", root, identity, session_dir)
    assert "forget" not in text and "edit" not in text


def test_forgotten_item_shows_a_body_removed_line(rig):
    root, identity, session_dir, store = rig
    item = store.put("global", {"title": "Gone", "body": "docs/g.md"})
    store.put("global", {}, entity=item["id"], expected_version=item["version"], forget=True)
    shown, failed = human(f"show global/{item['id']}", root, identity, session_dir)
    assert not failed and "Body removed (forgotten)" in shown
    with pytest.raises(MemoryError):
        store.put("global", {"title": "x", "body": "y"}, entity=item["id"], expected_version="v")
