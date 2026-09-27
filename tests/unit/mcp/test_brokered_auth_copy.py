"""A borrowing device must not be told to run a login command it cannot use.

REVIEW ROUND 1, R1 (audit round 2's F5). ``_auth_required_text`` answered every
"needs a grant, none stored" failure with ``/mcp login <name> to authorize``, which is
right only for a device that OWNS the sign-in. On a borrower it is wrong twice over:

* a headless borrower cannot open a browser at all, so the command cannot work;
* a borrower that CAN open one creates a LOCAL grant, which then wins over the borrow
  (``manager._brokered_mcp_auth``'s local-wins rule) — a silent account switch.

Auth round 1 fixed the owner's side (the ``interactive_required`` refusal); these cells
pin the borrower's side, which is where the operator actually reads it. The composed
line must name the owning device and put the command there, and it must do so on both
auth failure shapes, without disturbing a single non-borrowing render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_operator.mcp.auth import McpAuthChallengeError, McpAuthRequiredError
from local_operator.mcp.manager import McpManager

URL = "https://srv.example/mcp"
OWNER = "damians-MacBook-Pro"


class _Client:
    """The two facts the manager is allowed to use — and it is allowed no others."""

    def __init__(self, owner: str = OWNER, *, borrows: bool = True, explode: bool = False) -> None:
        self.owner = owner
        self._borrows = borrows
        self._explode = explode

    def should_borrow(self, key: str) -> bool:
        if self._explode:
            raise RuntimeError("placement unreadable")
        return self._borrows

    def owner_label(self, key: str) -> str:
        if self._explode:
            raise RuntimeError("placement unreadable")
        return self.owner


class _Store:
    """A store that brokers: ``mesh_client`` is what ``build_auth_store`` attaches."""

    def __init__(self, client: Any) -> None:
        self.mesh_client = client


def _challenge(*, oauth_available: bool = True) -> McpAuthChallengeError:
    return McpAuthChallengeError(
        URL, status_code=401, oauth_available=oauth_available, has_stored_grant=False
    )


def _borrowing(tmp_path: Path, **client_kwargs: Any) -> tuple[McpManager, str]:
    """A real manager on an injected, borrowing store — the production call shape."""
    # ``Any`` deliberately: the store here is a duck-typed stand-in for the real
    # ``ManagedAuthStore``, exactly as the mcp suite's other fake-store fixtures are
    # typed (``test_manager._oauth_manager``), so the manager's annotation does not
    # turn a fixture into a type error.
    store: Any = _Store(_Client(**client_kwargs))
    manager = McpManager(str(tmp_path), auth_store=store)
    return manager, manager._auth_failure_text("linear", _challenge(), store=store)


def test_a_borrowed_server_names_the_owning_device(tmp_path: Path) -> None:
    """The R1 arm: the line says WHERE the sign-in is, not "log in here"."""
    _, text = _borrowing(tmp_path)

    assert OWNER in text, text
    assert "/mcp login linear there" in text, text
    assert "to authorize" not in text, text


def test_a_borrowed_server_never_recommends_a_local_grant(tmp_path: Path) -> None:
    """Both auth shapes, because both used to name a local command.

    ``reauth`` is the OTHER local verb and is wrong here for the same reason: it would
    replace a credential that is not on this device.
    """
    store: Any = _Store(_Client())
    manager = McpManager(str(tmp_path), auth_store=store)

    challenge = manager._auth_failure_text("linear", _challenge(), store=store)
    required = manager._auth_failure_text("linear", McpAuthRequiredError(URL), store=store)

    for text in (challenge, required):
        assert OWNER in text, text
        assert "/mcp login linear there" in text, text
        assert "/mcp reauth" not in text, text


def test_the_borrowed_line_survives_the_toast_clamp() -> None:
    """Design constraint D1/D4: the toast clamps the TAIL, and one ``—`` only."""
    from local_operator.network.credentials.messages import render_borrowed_signin

    text = render_borrowed_signin(OWNER, "linear")
    assert len(text) <= 58, (len(text), text)
    assert "—" not in text, text


def test_a_non_borrowing_server_keeps_the_local_command(tmp_path: Path) -> None:
    """The overwhelmingly common case: a device that does NOT borrow is untouched."""
    store: Any = _Store(_Client(borrows=False))
    manager = McpManager(str(tmp_path), auth_store=store)

    assert manager._auth_failure_text("linear", _challenge(), store=store) == (
        "/mcp login linear to authorize"
    )


def test_no_store_and_no_mesh_client_keep_every_existing_line(tmp_path: Path) -> None:
    """``store=None`` (the display side, and the twenty-odd old call sites) is a no-op."""
    assert McpManager._auth_failure_text("linear", _challenge()) == (
        "/mcp login linear to authorize"
    )
    broken: Any = _Store(_Client(explode=True))
    manager = McpManager(str(tmp_path), auth_store=broken)
    assert (
        manager._auth_failure_text("linear", _challenge(), store=manager._effective_auth_store())
        == "/mcp login linear to authorize"
    )


def test_an_unreadable_placement_degrades_instead_of_raising(tmp_path: Path) -> None:
    """A failure is being DESCRIBED here: a broken store must not replace it with one."""
    store: Any = _Store(_Client(explode=True))
    manager = McpManager(str(tmp_path), auth_store=store)

    text = manager._auth_failure_text("linear", _challenge(), store=store)

    assert text == "/mcp login linear to authorize"
