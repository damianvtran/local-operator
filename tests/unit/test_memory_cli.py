"""Human commands exercise the shipped CLI, with synthetic config/home only."""

import json
import os
import subprocess
import sys

from local_operator.memory import MemoryIdentity, MemoryStore
from local_operator.memory_cli import render_human, run


def test_shipped_cli_inspection_correction_and_explicit_forget(tmp_path, monkeypatch):
    root = tmp_path / "home" / ".local-operator"
    home = root.parent
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    store = MemoryStore(root, MemoryIdentity("cli-session", "", "user"), origin="cli-device")
    item = store.put("global", {"title": "CLI proof", "body": "CLI evidence docs/proof.md"})

    def cli(*args, expected=0):
        env = {
            key: value for key, value in os.environ.items() if not key.startswith(("CMUX_", "LOP_"))
        }
        response = subprocess.run(
            [sys.executable, "-m", "local_operator.cli", "memory", *args],
            cwd=home,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert response.returncode == expected, (response.returncode, response.stderr)
        return response.stdout

    listed = json.loads(cli("list"))
    assert listed["items"][0]["id"] == item["id"]
    assert json.loads(cli("show", item["id"]))["fact"]["body"].endswith("proof.md")
    assert "concurrently" in cli(
        "edit",
        item["id"],
        "--version",
        "stale",
        "--json",
        json.dumps({"title": "CLI proof", "body": "New proof"}),
        expected=1,
    )
    changed = json.loads(
        cli(
            "edit",
            item["id"],
            "--version",
            item["version"],
            "--json",
            json.dumps({"title": "CLI proof", "body": "New proof"}),
        )
    )
    assert "Nothing forgotten" in cli(
        "forget", item["id"], "--version", changed["version"], expected=1
    )
    assert store.show("global", item["id"])["status"] == "active"
    assert (
        json.loads(cli("forget", item["id"], "--version", changed["version"], "--yes"))["status"]
        == "forgotten"
    )
    assert json.loads(cli("search", "proof")) == []
    assert "sync" in json.loads(cli("status"))


def test_bad_human_input_and_disabled_store_inspection(tmp_path):
    text, failed = run("forget x --version v", config=tmp_path)
    assert failed and "Nothing forgotten" in text
    assert not (tmp_path / "memory").exists()
    empty, failed = run("list", config=tmp_path)
    assert not failed and "No retained memories" in render_human(empty)
    invalid, failed = run("--scope other list", config=tmp_path)
    assert failed and "Invalid memory command" in invalid


def test_human_rendering_is_readable_and_teaches_next_steps_once():
    now = 1_800_000_000.0
    fact = {
        "title": "T",
        "body": "B",
        "source": {"author": "main", "kind": "reported", "session": "s1"},
        "created_at": now - 2 * 86400,
        "verified_at": None,
        "source_time": None,
        "expires_at": now + 3 * 3600,
    }
    shown = render_human(
        json.dumps(
            {
                "scope": "global",
                "id": "i",
                "status": "active",
                "version": "v",
                "fact": fact,
                "conflicts": [],
                "revisions": 1,
            }
        )
    )
    assert "by main (reported) · session s1" in shown
    assert "Verified: never verified" in shown and "{" not in shown
    assert "--json" not in shown, "next-step hint belongs to list only"
    listed = render_human(json.dumps({"items": [], "active_tokens": 0, "ops": 0}))
    assert listed.count("Next:") == 1
    from local_operator.memory_cli import _date

    assert _date(now - 2 * 86400, now=now).endswith("(2d ago)")
    assert _date(now + 3 * 3600, now=now).endswith("(in 3h)")
    assert _date(None, "never") == "never"
