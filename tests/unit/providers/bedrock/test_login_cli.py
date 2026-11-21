"""``lop login`` CLI outcomes specific to Amazon Bedrock (#4270 round 1).

* D4/Q7 — "I'll run it myself" is a choice: exit 0, no "Login failed:" lead-in,
  the command on its own indented line, nothing stored.
* Q6 — ``lop login status`` must not list the sealed ``BEDROCK_SECRET_<hex>``
  refs as if they were keys the user set.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from local_operator.providers import auth_cli
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.bedrock import login as blogin
from local_operator.providers.oauth.callback_server import LoginDeferredError

COMMAND = "aws sso login --profile corp-sso"


def test_run_it_yourself_exits_zero_with_the_command_set_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def deferred(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise LoginDeferredError(
            "Nothing was stored. Run this in a terminal, then connect Amazon Bedrock again:",
            command=COMMAND,
        )

    monkeypatch.setattr(blogin, "login_bedrock", deferred)
    with closing(AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)) as store:
        assert auth_cli.run_login("bedrock", tmp_path, store) == 0
        assert store.list_credentials(provider=None) == []
    out = capsys.readouterr().out
    assert "Login failed" not in out
    assert "Nothing was stored." in out
    assert f"\n    {COMMAND}\n" in out


def test_status_hides_sealed_bedrock_refs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref = "BEDROCK_SECRET_" + "a" * 32
    monkeypatch.setattr(
        auth_cli, "stored_login_key_names", lambda _root: [ref, "OPENROUTER_API_KEY"]
    )
    with closing(AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)) as store:
        lines = auth_cli.format_logins(store, tmp_path)
    text = "\n".join(lines)
    assert "secret store  OPENROUTER_API_KEY=<set>" in text
    assert "BEDROCK_SECRET_" not in text
