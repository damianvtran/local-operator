"""Software-only normal public finite recovery for existing lifecycle scenarios.

Keep the bare second-share refusal as a negative, then restore explicitly through
request -> software-signed normal approval -> public apply. This is not a product
signature bypass: the real verifier/store/run/receipt and public CLI all execute.
"""

import contextlib
import io
import json
from pathlib import Path

import pytest

from local_operator.network import approvals as A
from local_operator.operator import trust
from tests.unit.network.test_approvals_store import _make_key, _sign
from tests.unit.network.test_credentials_real_link import _lop_network


def restore_finite(root: Path, key: str, peer: str, *, scope: str = "session") -> None:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            trust,
            "load_anchor",
            lambda uid=None: trust.AnchorLoad(
                anchor=None,
                path=trust.anchor_path(uid),
                root_owned=False,
                reason="software recovery fixture",
                exists=False,
            ),
        )
        _make_key(root)

        def software(**kw):
            return _sign(
                root,
                A.signed_payload(
                    kind=kw["kind"],
                    request_id=kw["request_id"],
                    request_digest=kw["request_digest"],
                    decision="approve",
                    decided_at=kw["decided_at"],
                ),
            )

        patch.setattr(A, "sign_decision", software)
        base = ["credential", "share", key, "--with", peer, "--scope", scope]
        assert _lop_network(*base) == 1, "bare share must not erase the explicit withdrawal"
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            assert _lop_network(*base, "--request-approval", "--json") == 0
        aid = json.loads(capture.getvalue())["approval_id"]
        assert _lop_network("approvals", "approve", aid) == 0
        assert _lop_network(*base, "--approval", aid) == 0
