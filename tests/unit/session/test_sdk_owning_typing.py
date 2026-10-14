"""Static and runtime pins for the SDK owning-surface typing (review R1).

WHY THIS FILE EXISTS. Review round 1 found that widening the shared
``SessionProtocol`` broke the whole-tree gates; the agreed fix is a separate
``OwningSessionProtocol`` plus type-only ``open_session`` overloads that pick
the view by ``mode``. That fix is only half-visible at runtime -- the
``assert_type`` calls below live inside ``if TYPE_CHECKING`` and are checked
by the repo's whole-tree pyright gate, which is where "a default-own caller
can call the owning methods without Any/cast" is actually PROVEN. The runtime
tests pin the surface the overloads are built on (three overloads exist with
the right mode literals and returns; the split protocols carry the right
members), so a future edit that quietly reverts to a widened base protocol
turns both halves red together.

Nothing here imports engine modules at runtime beyond the cheap protocol and
session facades; the static block imports nothing at runtime at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pyright-only: mode-precise SDK typing, asserted statically
    import typing as static_typing

    from local_operator.session.protocol import (
        OwningSessionProtocol,
        SessionProtocol,
        ViewerSessionProtocol,
    )
    from local_operator.session.spec import SessionRoots, SessionSpec
    from local_operator.session.tool_readiness import ToolReadiness

    async def _static_owning_surface(spec: SessionSpec, roots: SessionRoots) -> None:
        """The three typed views the overloads promise, checked statically."""
        from local_operator import sdk

        async with sdk.open_session(spec, roots=roots) as session:  # default IS own
            static_typing.assert_type(session, OwningSessionProtocol)
            unsubscribe = session.subscribe_requests(lambda _row: None)
            unsubscribe()
            receipt = await session.shutdown_owned_work("static", timeout_s=1.0)
            static_typing.assert_type(receipt.complete, bool)
            readiness = await session.wait_for_tools(["bash"], timeout_s=1.0)
            static_typing.assert_type(readiness, ToolReadiness)
            static_typing.assert_type(readiness.missing, tuple[str, ...])

        async with sdk.open_session(spec, roots=roots, mode="own") as session:
            static_typing.assert_type(session, OwningSessionProtocol)
            session.subscribe_requests(lambda _row: None)

        # Attach does NOT expose the owning surface: the viewer protocol has
        # no such members, so a call here would be a pyright error -- the
        # negative cannot be spelled as an assert, but the positive type
        # narrows to the viewer exactly.
        async with sdk.open_session(spec, roots=roots, mode="attach") as session:
            static_typing.assert_type(session, ViewerSessionProtocol)

        dynamic = static_typing.cast(static_typing.Literal["own", "attach"], "own")
        async with sdk.open_session(spec, roots=roots, mode=dynamic) as session:
            # A dynamic-mode caller keeps the BASE type and narrows explicitly.
            static_typing.assert_type(session, SessionProtocol)


def test_open_session_overloads_are_the_typed_surface() -> None:
    """Exactly three overloads: owning for default/own, viewer for attach, base
    for dynamic -- and the implementation under them stays the untouched
    ``asynccontextmanager`` async generator."""
    import typing

    from local_operator import sdk

    overloads = typing.get_overloads(sdk.open_session)
    assert len(overloads) == 3
    assert [fn.__annotations__.get("mode") for fn in overloads] == [
        "Literal['own']",
        "Literal['attach']",
        "Literal['own', 'attach']",
    ]
    assert [fn.__annotations__.get("return") for fn in overloads] == [
        "AbstractAsyncContextManager[OwningSessionProtocol]",
        "AbstractAsyncContextManager[ViewerSessionProtocol]",
        "AbstractAsyncContextManager[SessionProtocol]",
    ]
    assert overloads[0].__kwdefaults__ == {"mode": "own", "allow_multi_root": False}
    assert overloads[1].__kwdefaults__ == {"allow_multi_root": False}
    # functools.wraps chain left by asynccontextmanager
    implementation = getattr(sdk.open_session, "__wrapped__", None)
    assert implementation is not None
    assert "SessionProtocol" in implementation.__annotations__.get("return", "")


def test_protocol_split_is_where_the_owning_surface_lives() -> None:
    """OwningSessionProtocol extends the shared contract; the shared protocol
    carries none of the owning members; a real Session conforms to the owning
    extension. This is the shape the SDK overloads name (R1)."""
    from typing import _get_protocol_attrs  # type: ignore[attr-defined]

    from local_operator.session.protocol import (
        OwningSessionProtocol,
        SessionProtocol,
        ViewerSessionProtocol,
    )
    from local_operator.session.session import Session

    assert SessionProtocol in getattr(OwningSessionProtocol, "__mro__")
    owning = {"subscribe_requests", "shutdown_owned_work", "wait_for_tools"}
    assert owning <= set(_get_protocol_attrs(OwningSessionProtocol))
    assert not owning & set(_get_protocol_attrs(SessionProtocol))
    assert not owning & set(_get_protocol_attrs(ViewerSessionProtocol))
    assert callable(Session.subscribe_requests)
    assert callable(Session.shutdown_owned_work)
    assert callable(Session.wait_for_tools)
